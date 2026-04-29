"""Shared primitives for all chunkers.

Provides:
  - `Chunk` dataclass — schema mirrors compliance_chunks / credit_chunks Supabase tables
  - `count_tokens()` — tiktoken cl100k (closest open analogue to text-embedding-3-large
                       tokenizer; fine for size-bound decisions)
  - `split_into_sentences()` — char-offset-preserving sentence segmentation
  - `pack_paragraphs()` — pack adjacent paragraphs into chunks within a token budget

The single design invariant: **every Chunk's char_start/char_end is an absolute
offset into the source ParsedDoc.full_text**. This is what Track A overlap
scoring depends on — get this wrong and the eval is broken.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    module: str                              # 'compliance' | 'credit'
    chunk_strategy: str
    chunk_index: int
    content: str
    content_tokens: int
    char_start: int                          # absolute offset in source doc full_text
    char_end: int

    # Section context
    section_title: str = ""
    section_number: str = ""
    hierarchy_path: str = ""                 # e.g. "Item 1 > Risk Factors"

    # Hierarchical chunker only
    parent_chunk_id: Optional[str] = None
    chunk_level: str = "leaf"                # 'parent' | 'child' | 'leaf'

    # Credit-specific
    contains_table: bool = False
    section_type: str = ""                   # 'income_statement' | 'balance_sheet' | 'mda' | 'risk_factors' | 'notes' | 'narrative'

    # Domain metadata (passed through from ParsedDoc.metadata)
    # Compliance:
    regulatory_body: str = ""
    jurisdiction: str = ""
    doc_version: str = ""
    effective_date: str = ""
    # Credit:
    company_ticker: str = ""
    company_name: str = ""
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    industry_sector: str = ""
    naics_code: str = ""
    filing_date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))


# Regex for sentence splitting. Conservative — splits at .!? followed by space + capital,
# but won't split on abbreviations like "U.S." or "e.g." (too brittle to enumerate all).
# Returns positions in the original text so char offsets are preserved.
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(\[])")


def split_into_sentences(text: str, *, base_offset: int = 0) -> list[tuple[str, int, int]]:
    """Return [(sentence_text, char_start, char_end)] with offsets in original `text`.

    `base_offset` is added to every offset — pass the section's char_start when
    splitting a section so the returned offsets are absolute in the doc.
    """
    if not text.strip():
        return []
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for m in _SENT_END.finditer(text):
        end = m.start()
        sent = text[cursor:end].strip()
        if sent:
            sent_start = base_offset + cursor + (len(text[cursor:]) - len(text[cursor:].lstrip()))
            out.append((sent, sent_start, base_offset + end))
        cursor = m.end()
    # Final sentence
    sent = text[cursor:].strip()
    if sent:
        sent_start = base_offset + cursor + (len(text[cursor:]) - len(text[cursor:].lstrip()))
        out.append((sent, sent_start, base_offset + len(text)))
    return out


# Paragraph splitter: blank-line-separated blocks. Returns offsets in original text.
_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_into_paragraphs(text: str, *, base_offset: int = 0) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for m in _PARA_SPLIT.finditer(text):
        block = text[cursor:m.start()].strip()
        if block:
            block_start = base_offset + cursor + (len(text[cursor:m.start()]) -
                                                  len(text[cursor:m.start()].lstrip()))
            out.append((block, block_start, base_offset + m.start()))
        cursor = m.end()
    block = text[cursor:].strip()
    if block:
        block_start = base_offset + cursor + (len(text[cursor:]) -
                                              len(text[cursor:].lstrip()))
        out.append((block, block_start, base_offset + len(text)))
    return out


def pack_units_to_chunks(
    units: list[tuple[str, int, int]],   # (text, char_start, char_end)
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[tuple[str, int, int, int]]:    # returns (text, char_start, char_end, n_tokens)
    """Greedy pack: combine adjacent units until adding the next would exceed max_tokens.

    Yield a chunk when current size >= min_tokens (or no more units to add). Single
    units larger than max_tokens get emitted as-is (caller's responsibility to split
    further if needed — usually fine for our use cases since regulatory paragraphs
    rarely exceed 600 tokens).
    """
    out: list[tuple[str, int, int, int]] = []
    if not units:
        return out

    cur_texts: list[str] = []
    cur_start = units[0][1]
    cur_end = units[0][1]
    cur_tokens = 0

    for text, start, end in units:
        unit_tokens = count_tokens(text)
        # If single unit > max, flush current and emit oversized unit alone
        if unit_tokens >= max_tokens:
            if cur_texts:
                out.append(("\n\n".join(cur_texts), cur_start, cur_end, cur_tokens))
                cur_texts, cur_tokens = [], 0
            out.append((text, start, end, unit_tokens))
            cur_start = end
            cur_end = end
            continue

        # Would adding this unit blow the budget?
        if cur_tokens + unit_tokens > max_tokens and cur_tokens >= min_tokens:
            out.append(("\n\n".join(cur_texts), cur_start, cur_end, cur_tokens))
            cur_texts = [text]
            cur_start = start
            cur_end = end
            cur_tokens = unit_tokens
        else:
            if not cur_texts:
                cur_start = start
            cur_texts.append(text)
            cur_end = end
            cur_tokens += unit_tokens

    if cur_texts:
        out.append(("\n\n".join(cur_texts), cur_start, cur_end, cur_tokens))

    return out


def metadata_for_chunk(parsed_doc: dict) -> dict:
    """Extract the metadata fields a Chunk needs from a ParsedDoc dict."""
    md = parsed_doc.get("metadata", {})
    return {
        # Compliance-side
        "regulatory_body": md.get("regulatory_body", ""),
        "jurisdiction": md.get("jurisdiction", ""),
        "doc_version": md.get("doc_version", "") or "",
        "effective_date": md.get("effective_date", "") or "",
        # Credit-side
        "company_ticker": md.get("company_ticker", ""),
        "company_name": md.get("company_name", ""),
        "fiscal_year": md.get("fiscal_year"),
        "fiscal_quarter": md.get("fiscal_quarter"),
        "industry_sector": md.get("industry_sector", ""),
        "naics_code": md.get("naics_code", ""),
        "filing_date": md.get("filing_date", ""),
    }


def make_chunk_id(doc_id: str, strategy: str, idx: int) -> str:
    # Deterministic UUIDv5 — re-running chunking with the same inputs yields stable IDs.
    name = f"{doc_id}|{strategy}|{idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))
