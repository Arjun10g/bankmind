"""Parse heterogeneous source documents into a uniform `ParsedDoc` shape.

Inputs:
  - Regulatory PDFs (Basel, Bank Act, Fed Reg W) → pdfplumber
  - Regulatory HTML (OSFI, FINTRAC, GDPR)        → BeautifulSoup
  - EDGAR filings (10-K/10-Q/8-K/40-F/6-K)       → BeautifulSoup (XBRL-aware via tag stripping)

Output (per source file):
  ParsedDoc with:
    - full_text:    one big string, the canonical text
    - pages:        list of (page_number, char_start, char_end) — populated for PDFs only
    - sections:     list of (heading, level, section_number, char_start, char_end)
    - tables:       list of (markdown_repr, char_start, char_end) — credit module only

CRITICAL: char_start/char_end indices into full_text are the foundation for the
dual-track evaluation (Track A overlap-based relevance). Sections, pages, and
tables MUST have accurate offsets — every chunker reads from these.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from bs4 import BeautifulSoup, NavigableString, Tag


# --- Data shapes ---------------------------------------------------------------

@dataclass
class ParsedSection:
    heading: str
    level: int                # 1..6
    section_number: str       # "1.2.3" or "Article 5" or "Item 7A" etc., "" if unknown
    char_start: int
    char_end: int


@dataclass
class ParsedPage:
    page_number: int
    char_start: int
    char_end: int


@dataclass
class ParsedTable:
    markdown: str
    char_start: int
    char_end: int
    n_rows: int
    n_cols: int


@dataclass
class ParsedDoc:
    doc_id: str
    doc_title: str
    doc_type: str
    module: str                            # 'compliance' | 'credit'
    metadata: dict
    full_text: str
    pages: list[ParsedPage] = field(default_factory=list)
    sections: list[ParsedSection] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "module": self.module,
            "metadata": self.metadata,
            "full_text": self.full_text,
            "n_chars": len(self.full_text),
            "pages": [vars(p) for p in self.pages],
            "sections": [vars(s) for s in self.sections],
            "tables": [vars(t) for t in self.tables],
        }


# --- Section detection (regex-based, used for PDFs and as fallback) ------------

# Note: ordering matters — more specific patterns first.
# Each pattern captures (section_number, heading_text).
SECTION_PATTERNS = [
    # SEC 10-K Items: "Item 1.", "Item 1A.", "Item 7.", etc.
    (re.compile(r"^\s*(Item\s+\d+[A-Z]?)\.?\s+(.{3,200})$", re.MULTILINE), "item"),
    # GDPR-style articles: "Article 5", "Article 17 — Right to erasure"
    (re.compile(r"^\s*(Article\s+\d+[a-z]?)\s*[—:.\-]?\s*(.{3,200})$", re.MULTILINE), "article"),
    # Chapters: "Chapter I", "Chapter 1 — Title"
    (re.compile(r"^\s*(Chapter\s+(?:\d+|[IVXLCDM]+))\s*[—:.\-]?\s*(.{3,200})$", re.MULTILINE), "chapter"),
    # Numbered sections: "1. Title", "1.2 Title", "1.2.3 Title"
    (re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{3,200})$", re.MULTILINE), "numbered"),
]


def detect_sections_regex(text: str) -> list[ParsedSection]:
    """Run all section regexes; merge by char_start; assign levels by depth."""
    candidates: dict[int, ParsedSection] = {}

    for pat, kind in SECTION_PATTERNS:
        for m in pat.finditer(text):
            number = m.group(1).strip()
            heading = m.group(2).strip()
            char_start = m.start()
            # Determine level
            if kind == "item":
                level = 2  # SEC Items are sub-document
            elif kind == "chapter":
                level = 1
            elif kind == "article":
                level = 2
            elif kind == "numbered":
                # depth = number of dots + 1 (1 → level 1, 1.2 → level 2, 1.2.3 → level 3)
                level = min(number.count(".") + 1, 6)
            else:
                level = 3
            # Earliest match at a given char_start wins (most specific pattern, since list-ordered)
            if char_start not in candidates:
                candidates[char_start] = ParsedSection(
                    heading=heading, level=level, section_number=number,
                    char_start=char_start, char_end=char_start,  # filled in below
                )

    sections = sorted(candidates.values(), key=lambda s: s.char_start)
    # Fill char_end as start of the next section (or end of text)
    for i, sec in enumerate(sections):
        sec.char_end = sections[i + 1].char_start if i + 1 < len(sections) else len(text)
    return sections


# --- PDF parsing ---------------------------------------------------------------

def parse_pdf(path: Path) -> tuple[str, list[ParsedPage], list[ParsedSection], list[ParsedTable]]:
    full_text_parts: list[str] = []
    pages: list[ParsedPage] = []
    cursor = 0

    with pdfplumber.open(str(path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            block = text + "\n\n"
            char_start = cursor
            full_text_parts.append(block)
            cursor += len(block)
            pages.append(ParsedPage(page_number=page_idx, char_start=char_start,
                                    char_end=cursor))
    full_text = "".join(full_text_parts)
    sections = detect_sections_regex(full_text)
    # Tables in PDFs are noisy; defer extraction (the credit pipeline mostly cares
    # about EDGAR HTML tables, which we handle via BeautifulSoup below).
    tables: list[ParsedTable] = []
    return full_text, pages, sections, tables


# --- HTML parsing --------------------------------------------------------------

# Tags whose text we drop entirely (script/style/etc.)
_HTML_DROP_TAGS = {"script", "style", "noscript", "head", "meta", "link"}
# XBRL tags that EDGAR filings embed inline; we keep their text content.
# (BeautifulSoup .get_text() handles this naturally — we don't strip them.)


def _table_to_markdown(table: Tag) -> tuple[str, int, int]:
    """Convert a <table> Tag to a simple markdown representation."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    if not rows:
        return "", 0, 0

    n_cols = max(len(r) for r in rows)
    # Pad ragged rows
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    md_lines = []
    header = rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines), len(rows), n_cols


def parse_html(path: Path, *, extract_tables: bool) -> tuple[
    str, list[ParsedSection], list[ParsedTable]
]:
    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(_HTML_DROP_TAGS):
        tag.decompose()

    # Walk the tree once. For each visible element, append its text and record
    # heading/table positions with accurate char offsets.
    parts: list[str] = []
    sections: list[ParsedSection] = []
    tables: list[ParsedTable] = []
    cursor = 0

    body = soup.body or soup

    for element in body.descendants:
        if isinstance(element, NavigableString):
            # Skip if any ancestor is a heading/table — those are handled at the tag level
            if any(isinstance(p, Tag) and p.name in (
                "h1", "h2", "h3", "h4", "h5", "h6", "table", "script", "style"
            ) for p in element.parents):
                continue
            text = str(element).strip()
            if text:
                block = text + " "
                parts.append(block)
                cursor += len(block)
            continue

        if not isinstance(element, Tag):
            continue

        # Block boundaries — push a newline so paragraphs separate
        if element.name in ("p", "div", "li", "br", "section", "article"):
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")
                cursor += 1
            continue

        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading_text = element.get_text(" ", strip=True)
            if not heading_text:
                continue
            # Try to extract a section number from the heading text
            m = re.match(r"^\s*(Article\s+\d+[a-z]?|Chapter\s+(?:\d+|[IVXLCDM]+)|"
                         r"Item\s+\d+[A-Z]?|\d+(?:\.\d+){0,3})\.?\s*[—:.\-]?\s*(.*)$",
                         heading_text)
            if m:
                section_number = m.group(1).strip()
                heading_clean = (m.group(2) or "").strip() or heading_text
            else:
                section_number = ""
                heading_clean = heading_text

            level = int(element.name[1])
            char_start = cursor
            block = f"\n\n{heading_text}\n\n"
            parts.append(block)
            cursor += len(block)
            sections.append(ParsedSection(
                heading=heading_clean, level=level, section_number=section_number,
                char_start=char_start, char_end=char_start,  # filled later
            ))
            continue

        if element.name == "table" and extract_tables:
            md, n_rows, n_cols = _table_to_markdown(element)
            if not md:
                continue
            char_start = cursor
            block = f"\n\n{md}\n\n"
            parts.append(block)
            cursor += len(block)
            tables.append(ParsedTable(
                markdown=md, char_start=char_start, char_end=cursor,
                n_rows=n_rows, n_cols=n_cols,
            ))
            continue

    full_text = "".join(parts)

    # Backfill section char_end and collapse runs of whitespace
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    # Reconstruct offsets after whitespace collapse: section/table char_starts
    # were tracked in the unnormalized stream. Re-derive their positions by
    # finding their heading/markdown text in the normalized full_text.
    for sec in sections:
        idx = full_text.find(sec.heading)
        if idx >= 0:
            sec.char_start = idx
    sections.sort(key=lambda s: s.char_start)
    for i, sec in enumerate(sections):
        sec.char_end = sections[i + 1].char_start if i + 1 < len(sections) else len(full_text)

    for tbl in tables:
        idx = full_text.find(tbl.markdown)
        if idx >= 0:
            tbl.char_start = idx
            tbl.char_end = idx + len(tbl.markdown)

    # If HTML had no semantic headings, fall back to regex-based section detection
    if not sections:
        sections = detect_sections_regex(full_text)

    return full_text, sections, tables


# --- Top-level dispatcher ------------------------------------------------------

def parse_document(
    path: Path,
    metadata: dict,
    module: str,
) -> ParsedDoc:
    """Parse a single document. Dispatches by file extension."""
    ext = path.suffix.lower().lstrip(".")
    extract_tables = (module == "credit")

    if ext == "pdf":
        full_text, pages, sections, tables = parse_pdf(path)
    elif ext in ("html", "htm", "xhtml"):
        full_text, sections, tables = parse_html(path, extract_tables=extract_tables)
        pages = []
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")

    return ParsedDoc(
        doc_id=metadata.get("doc_id", path.stem),
        doc_title=metadata.get("doc_title", path.stem),
        doc_type=metadata.get("doc_type", "unknown"),
        module=module,
        metadata=metadata,
        full_text=full_text,
        pages=pages,
        sections=sections,
        tables=tables,
    )
