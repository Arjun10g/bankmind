"""Credit chunkers — three strategies as specified in CLAUDE.md § 3.2.

  1. financial_statement: detect financial-statement headings; tables stay atomic;
                          narrative sections 400-600 tok; tables kept whole regardless
                          of length, tagged contains_table=True.
  2. semantic:            same algorithm as compliance, threshold 0.45; never breaks
                          inside a table range.
  3. narrative_section:   split at SEC Item boundaries (Item 1, 1A, 2, ..., 7, 7A, 8, ...).
                          Sub-split long items at paragraph (max 512 tok). Each chunk
                          carries item_number, item_title, section_type metadata.
"""
from __future__ import annotations

import re

from pipelines.shared.chunking_base import (
    Chunk,
    count_tokens,
    make_chunk_id,
    metadata_for_chunk,
    pack_units_to_chunks,
    split_into_paragraphs,
)
from pipelines.shared.semantic_chunker import semantic_chunk


# === SEC Item → section_type mapping =========================================
# 10-K item structure (also used in 10-Q with different items).
ITEM_TO_SECTION_TYPE = {
    "Item 1":  "business",
    "Item 1A": "risk_factors",
    "Item 1B": "unresolved_comments",
    "Item 1C": "cybersecurity",
    "Item 2":  "properties",
    "Item 3":  "legal_proceedings",
    "Item 4":  "mine_safety",
    "Item 5":  "market_for_stock",
    "Item 6":  "selected_financial_data",
    "Item 7":  "mda",
    "Item 7A": "market_risk",
    "Item 8":  "financial_statements",
    "Item 9":  "accountant_changes",
    "Item 9A": "controls",
    "Item 9B": "other_information",
    "Item 10": "directors_officers",
    "Item 11": "executive_compensation",
    "Item 12": "security_ownership",
    "Item 13": "related_transactions",
    "Item 14": "accountant_fees",
    "Item 15": "exhibits",
}

# Heuristic keywords for financial-statement-section detection (when SEC Items
# aren't useful — e.g. for 8-K, 6-K, 40-F)
FIN_STMT_KEYWORDS = [
    ("balance_sheet",       ["consolidated balance sheet", "statement of financial position"]),
    ("income_statement",    ["consolidated statement of income", "consolidated statements of income",
                             "consolidated income statement", "statement of operations"]),
    ("cash_flow",           ["consolidated statement of cash flows", "statements of cash flows"]),
    ("equity_changes",      ["statement of changes in equity", "consolidated statements of changes in equity",
                             "statement of stockholders’ equity", "statement of shareholders' equity"]),
    ("comprehensive_income",["statement of comprehensive income", "statements of comprehensive income"]),
    ("notes",               ["notes to consolidated financial statements", "notes to the financial statements"]),
]


def _section_type_from_heading(heading: str, section_number: str) -> str:
    snum = (section_number or "").strip()
    if snum in ITEM_TO_SECTION_TYPE:
        return ITEM_TO_SECTION_TYPE[snum]
    h = (heading or "").lower()
    for stype, keywords in FIN_STMT_KEYWORDS:
        for kw in keywords:
            if kw in h:
                return stype
    return "narrative"


def _table_ranges(parsed_doc: dict) -> list[tuple[int, int]]:
    return [(t["char_start"], t["char_end"]) for t in parsed_doc.get("tables", [])]


def _chunk_overlaps_any_table(chunk_start: int, chunk_end: int,
                              table_ranges: list[tuple[int, int]]) -> bool:
    return any(not (chunk_end <= ts or chunk_start >= te) for ts, te in table_ranges)


# === Strategy 1: Financial Statement Boundary ================================

FS_NARRATIVE_MIN = 200
FS_NARRATIVE_MAX = 600


def chunk_financial_statement(parsed_doc: dict) -> list[Chunk]:
    full_text: str = parsed_doc["full_text"]
    sections: list[dict] = parsed_doc["sections"]
    tables: list[dict] = parsed_doc.get("tables", [])
    md = metadata_for_chunk(parsed_doc)
    doc_id = parsed_doc["doc_id"]
    doc_title = parsed_doc["doc_title"]
    doc_type = parsed_doc["doc_type"]

    # Build a sorted list of "spans" — each is either a table (kept atomic) or a
    # narrative range between tables (sub-chunkable). Section metadata is
    # determined by which section the span starts in.
    table_ranges = sorted([(t["char_start"], t["char_end"], t) for t in tables])

    spans: list[tuple[str, int, int, dict | None]] = []
    cursor = 0
    for ts, te, tbl in table_ranges:
        if ts > cursor:
            spans.append(("narrative", cursor, ts, None))
        spans.append(("table", ts, te, tbl))
        cursor = te
    if cursor < len(full_text):
        spans.append(("narrative", cursor, len(full_text), None))

    def section_for(pos: int) -> dict:
        for sec in sections:
            if sec["char_start"] <= pos < sec["char_end"]:
                return sec
        return {"heading": "", "section_number": "", "char_start": 0, "char_end": 0}

    out: list[Chunk] = []
    idx = 0

    for kind, start, end, tbl in spans:
        if kind == "table":
            sec = section_for(start)
            text = tbl["markdown"]
            tok = count_tokens(text)
            section_type = _section_type_from_heading(sec["heading"], sec["section_number"])
            if section_type == "narrative":
                section_type = "table"
            out.append(Chunk(
                chunk_id=make_chunk_id(doc_id, "financial_statement", idx),
                doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="credit",
                chunk_strategy="financial_statement", chunk_index=idx,
                content=text, content_tokens=tok,
                char_start=start, char_end=end,
                section_title=sec["heading"],
                section_number=sec["section_number"],
                hierarchy_path=sec["heading"],
                contains_table=True,
                section_type=section_type,
                **md,
            ))
            idx += 1
        else:
            narrative_text = full_text[start:end]
            paras = split_into_paragraphs(narrative_text, base_offset=start)
            packed = pack_units_to_chunks(
                paras, min_tokens=FS_NARRATIVE_MIN, max_tokens=FS_NARRATIVE_MAX
            )
            for text, cs, ce, tok in packed:
                sec = section_for(cs)
                section_type = _section_type_from_heading(sec["heading"], sec["section_number"])
                out.append(Chunk(
                    chunk_id=make_chunk_id(doc_id, "financial_statement", idx),
                    doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="credit",
                    chunk_strategy="financial_statement", chunk_index=idx,
                    content=text, content_tokens=tok,
                    char_start=cs, char_end=ce,
                    section_title=sec["heading"],
                    section_number=sec["section_number"],
                    hierarchy_path=sec["heading"],
                    contains_table=False,
                    section_type=section_type,
                    **md,
                ))
                idx += 1
    return out


# === Strategy 2: Semantic ====================================================

SEMANTIC_THRESHOLD_CREDIT = 0.45
SEMANTIC_WINDOW = 3
SEMANTIC_MIN = 200
SEMANTIC_MAX = 600


def chunk_semantic(parsed_doc: dict) -> list[Chunk]:
    full_text: str = parsed_doc["full_text"]
    sections: list[dict] = parsed_doc["sections"]
    tables: list[dict] = parsed_doc.get("tables", [])
    md = metadata_for_chunk(parsed_doc)
    doc_id = parsed_doc["doc_id"]
    doc_title = parsed_doc["doc_title"]
    doc_type = parsed_doc["doc_type"]

    forbidden = [(t["char_start"], t["char_end"]) for t in tables]

    pieces = semantic_chunk(
        full_text,
        threshold=SEMANTIC_THRESHOLD_CREDIT,
        window=SEMANTIC_WINDOW,
        min_tokens=SEMANTIC_MIN,
        max_tokens=SEMANTIC_MAX,
        forbidden_break_ranges=forbidden,
    )

    table_ranges = [(t["char_start"], t["char_end"]) for t in tables]

    def section_for(pos: int) -> dict:
        for sec in sections:
            if sec["char_start"] <= pos < sec["char_end"]:
                return sec
        return {"heading": "", "section_number": ""}

    out: list[Chunk] = []
    for i, (text, cs, ce, tok) in enumerate(pieces):
        sec = section_for(cs)
        contains_table = _chunk_overlaps_any_table(cs, ce, table_ranges)
        section_type = _section_type_from_heading(sec["heading"], sec["section_number"])
        out.append(Chunk(
            chunk_id=make_chunk_id(doc_id, "semantic", i),
            doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="credit",
            chunk_strategy="semantic", chunk_index=i,
            content=text, content_tokens=tok,
            char_start=cs, char_end=ce,
            section_title=sec["heading"],
            section_number=sec["section_number"],
            hierarchy_path=sec["heading"],
            contains_table=contains_table,
            section_type=section_type,
            **md,
        ))
    return out


# === Strategy 3: Narrative Section (SEC Item boundaries) =====================

NARRATIVE_MIN = 200
NARRATIVE_MAX = 512


def chunk_narrative_section(parsed_doc: dict) -> list[Chunk]:
    full_text: str = parsed_doc["full_text"]
    sections: list[dict] = parsed_doc["sections"]
    tables: list[dict] = parsed_doc.get("tables", [])
    md = metadata_for_chunk(parsed_doc)
    doc_id = parsed_doc["doc_id"]
    doc_title = parsed_doc["doc_title"]
    doc_type = parsed_doc["doc_type"]

    # Filter to Item-style sections (or all top-level sections if no Items found).
    item_sections = [s for s in sections if s["section_number"].startswith("Item ")]
    if not item_sections:
        # 8-K, 6-K, 40-F often lack Items — fall back to all sections of the smallest level.
        if not sections:
            # Whole doc as a single chunk (or paragraph-packed)
            paras = split_into_paragraphs(full_text)
            packed = pack_units_to_chunks(paras, min_tokens=NARRATIVE_MIN, max_tokens=NARRATIVE_MAX)
            return [
                Chunk(
                    chunk_id=make_chunk_id(doc_id, "narrative_section", i),
                    doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="credit",
                    chunk_strategy="narrative_section", chunk_index=i,
                    content=text, content_tokens=tok,
                    char_start=cs, char_end=ce,
                    section_type="narrative",
                    **md,
                )
                for i, (text, cs, ce, tok) in enumerate(packed)
            ]
        min_level = min(s["level"] for s in sections)
        item_sections = [s for s in sections if s["level"] == min_level]

    table_ranges = [(t["char_start"], t["char_end"]) for t in tables]

    out: list[Chunk] = []
    idx = 0
    for s_idx, sec in enumerate(item_sections):
        sec_end = item_sections[s_idx + 1]["char_start"] if s_idx + 1 < len(item_sections) else len(full_text)
        sec_text = full_text[sec["char_start"]:sec_end]
        sec_tokens = count_tokens(sec_text)
        section_type = _section_type_from_heading(sec["heading"], sec["section_number"])

        if sec_tokens <= NARRATIVE_MAX:
            out.append(Chunk(
                chunk_id=make_chunk_id(doc_id, "narrative_section", idx),
                doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="credit",
                chunk_strategy="narrative_section", chunk_index=idx,
                content=sec_text.strip(), content_tokens=sec_tokens,
                char_start=sec["char_start"], char_end=sec_end,
                section_title=sec["heading"],
                section_number=sec["section_number"],
                hierarchy_path=f"{sec['section_number']} {sec['heading']}".strip(),
                contains_table=_chunk_overlaps_any_table(sec["char_start"], sec_end, table_ranges),
                section_type=section_type,
                **md,
            ))
            idx += 1
        else:
            paras = split_into_paragraphs(sec_text, base_offset=sec["char_start"])
            packed = pack_units_to_chunks(paras, min_tokens=NARRATIVE_MIN, max_tokens=NARRATIVE_MAX)
            for text, cs, ce, tok in packed:
                out.append(Chunk(
                    chunk_id=make_chunk_id(doc_id, "narrative_section", idx),
                    doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="credit",
                    chunk_strategy="narrative_section", chunk_index=idx,
                    content=text, content_tokens=tok,
                    char_start=cs, char_end=ce,
                    section_title=sec["heading"],
                    section_number=sec["section_number"],
                    hierarchy_path=f"{sec['section_number']} {sec['heading']}".strip(),
                    contains_table=_chunk_overlaps_any_table(cs, ce, table_ranges),
                    section_type=section_type,
                    **md,
                ))
                idx += 1
    return out


CHUNKERS = {
    "financial_statement": chunk_financial_statement,
    "semantic": chunk_semantic,
    "narrative_section": chunk_narrative_section,
}
