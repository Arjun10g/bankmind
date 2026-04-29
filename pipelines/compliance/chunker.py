"""Compliance chunkers — three strategies as specified in CLAUDE.md § 3.1.

  1. regulatory_boundary: split at section boundaries; if too long, split at paragraph.
                          Min 100, max 600 tokens. Overlap = 0.
  2. semantic:            sentence-transformer-based topic boundary detection.
                          Threshold 0.5, window 3, min 150 / max 512.
  3. hierarchical:        level-1 sections become parent chunks (~800-1200 tok),
                          sub-sections become child chunks (~150-300 tok) with
                          parent_chunk_id linkage. Both indexed.
"""
from __future__ import annotations

from pipelines.shared.chunking_base import (
    Chunk,
    count_tokens,
    make_chunk_id,
    metadata_for_chunk,
    pack_units_to_chunks,
    split_into_paragraphs,
)
from pipelines.shared.semantic_chunker import semantic_chunk


# === Strategy 1: Regulatory Boundary =========================================

REG_BOUNDARY_MIN = 100
REG_BOUNDARY_MAX = 600


def chunk_regulatory_boundary(parsed_doc: dict) -> list[Chunk]:
    full_text: str = parsed_doc["full_text"]
    sections: list[dict] = parsed_doc["sections"]
    md = metadata_for_chunk(parsed_doc)
    doc_id = parsed_doc["doc_id"]
    doc_title = parsed_doc["doc_title"]
    doc_type = parsed_doc["doc_type"]

    # If no sections detected (rare for regulatory docs), fall back to paragraph chunks
    if not sections:
        paras = split_into_paragraphs(full_text)
        units = pack_units_to_chunks(paras, min_tokens=REG_BOUNDARY_MIN, max_tokens=REG_BOUNDARY_MAX)
        return [
            Chunk(
                chunk_id=make_chunk_id(doc_id, "regulatory_boundary", i),
                doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="compliance",
                chunk_strategy="regulatory_boundary", chunk_index=i,
                content=text, content_tokens=tok,
                char_start=cs, char_end=ce,
                **md,
            )
            for i, (text, cs, ce, tok) in enumerate(units)
        ]

    out: list[Chunk] = []
    idx = 0
    for sec in sections:
        sec_text = full_text[sec["char_start"]:sec["char_end"]]
        sec_text_stripped = sec_text.strip()
        if not sec_text_stripped:
            continue
        sec_tokens = count_tokens(sec_text_stripped)

        # Within budget? Emit as a single chunk.
        if REG_BOUNDARY_MIN <= sec_tokens <= REG_BOUNDARY_MAX:
            chunk_pieces = [(sec_text_stripped, sec["char_start"], sec["char_end"], sec_tokens)]
        elif sec_tokens < REG_BOUNDARY_MIN:
            # Too small — still emit; merging across regulatory clauses risks bleed.
            chunk_pieces = [(sec_text_stripped, sec["char_start"], sec["char_end"], sec_tokens)]
        else:
            # Too long — split at paragraph
            paras = split_into_paragraphs(sec_text, base_offset=sec["char_start"])
            chunk_pieces = pack_units_to_chunks(
                paras, min_tokens=REG_BOUNDARY_MIN, max_tokens=REG_BOUNDARY_MAX
            )

        for text, cs, ce, tok in chunk_pieces:
            out.append(Chunk(
                chunk_id=make_chunk_id(doc_id, "regulatory_boundary", idx),
                doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="compliance",
                chunk_strategy="regulatory_boundary", chunk_index=idx,
                content=text, content_tokens=tok,
                char_start=cs, char_end=ce,
                section_title=sec["heading"],
                section_number=sec["section_number"],
                hierarchy_path=sec["heading"],
                **md,
            ))
            idx += 1
    return out


# === Strategy 2: Semantic Similarity =========================================

SEMANTIC_THRESHOLD_COMPLIANCE = 0.5
SEMANTIC_WINDOW = 3
SEMANTIC_MIN = 150
SEMANTIC_MAX = 512


def chunk_semantic(parsed_doc: dict) -> list[Chunk]:
    full_text: str = parsed_doc["full_text"]
    md = metadata_for_chunk(parsed_doc)
    doc_id = parsed_doc["doc_id"]
    doc_title = parsed_doc["doc_title"]
    doc_type = parsed_doc["doc_type"]

    pieces = semantic_chunk(
        full_text,
        threshold=SEMANTIC_THRESHOLD_COMPLIANCE,
        window=SEMANTIC_WINDOW,
        min_tokens=SEMANTIC_MIN,
        max_tokens=SEMANTIC_MAX,
    )
    sections = parsed_doc["sections"]

    def section_for_offset(pos: int) -> tuple[str, str]:
        for sec in sections:
            if sec["char_start"] <= pos < sec["char_end"]:
                return sec["heading"], sec["section_number"]
        return "", ""

    return [
        Chunk(
            chunk_id=make_chunk_id(doc_id, "semantic", i),
            doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="compliance",
            chunk_strategy="semantic", chunk_index=i,
            content=text, content_tokens=tok,
            char_start=cs, char_end=ce,
            section_title=section_for_offset(cs)[0],
            section_number=section_for_offset(cs)[1],
            **md,
        )
        for i, (text, cs, ce, tok) in enumerate(pieces)
    ]


# === Strategy 3: Hierarchical =================================================

HIER_PARENT_MIN = 400
HIER_PARENT_MAX = 1200
HIER_CHILD_MIN = 100
HIER_CHILD_MAX = 350


def chunk_hierarchical(parsed_doc: dict) -> list[Chunk]:
    """Build parent (level-1 or top-level) and child (sub-section) chunks.

    NOTE: per CLAUDE.md, parents should also carry an LLM-generated 1-sentence
    summary of their content. We defer that summary generation to tomorrow when
    ANTHROPIC_API_KEY is available — tonight we just build the structural
    parent/child relationships. The summary field can be backfilled in place.
    """
    full_text: str = parsed_doc["full_text"]
    sections: list[dict] = parsed_doc["sections"]
    md = metadata_for_chunk(parsed_doc)
    doc_id = parsed_doc["doc_id"]
    doc_title = parsed_doc["doc_title"]
    doc_type = parsed_doc["doc_type"]

    if not sections:
        # No structural info — fall back to regulatory_boundary semantics
        return chunk_regulatory_boundary(parsed_doc)

    # Identify the minimum level present — that's our "parent" tier.
    min_level = min(s["level"] for s in sections)

    out: list[Chunk] = []
    idx = 0

    # Group by parent: each parent owns the contiguous range of sub-sections that
    # follow it until the next same-level (or shallower) section.
    parent_indices = [i for i, s in enumerate(sections) if s["level"] == min_level]

    for p_idx, parent_section_i in enumerate(parent_indices):
        parent_sec = sections[parent_section_i]
        # Determine end of parent group
        if p_idx + 1 < len(parent_indices):
            group_end_char = sections[parent_indices[p_idx + 1]]["char_start"]
        else:
            group_end_char = len(full_text)

        parent_text = full_text[parent_sec["char_start"]:group_end_char].strip()
        if not parent_text:
            continue
        parent_tokens = count_tokens(parent_text)
        # Truncate parent text representation if it's enormous — parent chunks are
        # for context, not full content.
        if parent_tokens > HIER_PARENT_MAX:
            # Take just the first ~HIER_PARENT_MAX tokens worth (approx by chars)
            approx_chars = int(len(parent_text) * (HIER_PARENT_MAX / parent_tokens))
            parent_text_stored = parent_text[:approx_chars] + "\n[...truncated for parent context]"
            parent_tokens_stored = count_tokens(parent_text_stored)
        else:
            parent_text_stored = parent_text
            parent_tokens_stored = parent_tokens

        parent_chunk = Chunk(
            chunk_id=make_chunk_id(doc_id, "hierarchical", idx),
            doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="compliance",
            chunk_strategy="hierarchical", chunk_index=idx,
            content=parent_text_stored, content_tokens=parent_tokens_stored,
            char_start=parent_sec["char_start"], char_end=group_end_char,
            section_title=parent_sec["heading"],
            section_number=parent_sec["section_number"],
            hierarchy_path=parent_sec["heading"],
            chunk_level="parent",
            **md,
        )
        out.append(parent_chunk)
        idx += 1

        # Sub-sections of this parent (sections strictly after it, before group end)
        for s in sections[parent_section_i + 1:]:
            if s["char_start"] >= group_end_char:
                break
            child_text = full_text[s["char_start"]:s["char_end"]].strip()
            if not child_text:
                continue
            # Pack child text into [HIER_CHILD_MIN, HIER_CHILD_MAX]
            child_tokens = count_tokens(child_text)
            if child_tokens <= HIER_CHILD_MAX:
                pieces = [(child_text, s["char_start"], s["char_end"], child_tokens)]
            else:
                paras = split_into_paragraphs(child_text, base_offset=s["char_start"])
                pieces = pack_units_to_chunks(
                    paras, min_tokens=HIER_CHILD_MIN, max_tokens=HIER_CHILD_MAX
                )

            for text, cs, ce, tok in pieces:
                out.append(Chunk(
                    chunk_id=make_chunk_id(doc_id, "hierarchical", idx),
                    doc_id=doc_id, doc_title=doc_title, doc_type=doc_type, module="compliance",
                    chunk_strategy="hierarchical", chunk_index=idx,
                    content=text, content_tokens=tok,
                    char_start=cs, char_end=ce,
                    section_title=s["heading"],
                    section_number=s["section_number"],
                    hierarchy_path=f"{parent_sec['heading']} > {s['heading']}",
                    parent_chunk_id=parent_chunk.chunk_id,
                    chunk_level="child",
                    **md,
                ))
                idx += 1

    return out


CHUNKERS = {
    "regulatory_boundary": chunk_regulatory_boundary,
    "semantic": chunk_semantic,
    "hierarchical": chunk_hierarchical,
}
