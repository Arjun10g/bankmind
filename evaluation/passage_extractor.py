"""Extract chunking-agnostic source passages from parsed documents.

Source passages are the **atomic unit of truth** for evaluation:
  - Selected from raw parsed text BEFORE any chunking.
  - Each carries an absolute (char_start, char_end) into the source doc's full_text.
  - Track A scores retrieved chunks by character-level overlap with these ranges.
  - Track B uses these as the ground truth Claude reads to write reference answers.

Selection criteria (per CLAUDE.md § 7.1):
  - 150-400 tokens, self-contained (start at sentence boundary, no "see above" refs)
  - Diversity: stratified across documents, doc types, section types
  - Avoid headers/boilerplate (very short content, no narrative)
  - At least N sentences apart from other selected passages in the same doc
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class SourcePassage:
    passage_id: str
    module: str
    source_doc_id: str
    source_doc_title: str
    source_doc_type: str
    passage_text: str
    char_start: int
    char_end: int
    n_tokens: int
    page_number: Optional[int]
    section_title: str
    section_number: str

    def to_dict(self) -> dict:
        return asdict(self)


# Heuristic boilerplate / non-narrative signals
_BOILERPLATE = re.compile(
    r"|".join([
        r"^\s*Footnote",
        r"^\s*Skip to",
        r"^\s*Search",
        r"^\s*Language selection",
        r"^\s*Return to footnote",
        r"^\s*Page \d+",
        r"^\s*Table of Contents",
        r"^\s*Item \d+[A-Z]?\s*$",     # bare item header alone
    ]),
    re.IGNORECASE,
)
_REFERENCE_TO_PRIOR = re.compile(
    r"\b(as discussed above|see above|the above|the foregoing|the table below|"
    r"the figure below|the chart below|the previous|the preceding|see Note \d+|"
    r"refer to page \d+)\b",
    re.IGNORECASE,
)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """[(sentence_text, char_start, char_end)] over `text`."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.start()
        s = text[cursor:end]
        st = s.lstrip()
        if st:
            offset = len(s) - len(st)
            out.append((st.rstrip(), cursor + offset, end))
        cursor = m.end()
    s = text[cursor:]
    st = s.lstrip()
    if st:
        offset = len(s) - len(st)
        out.append((st.rstrip(), cursor + offset, len(text)))
    return out


def _section_for(parsed_doc: dict, char_pos: int) -> tuple[str, str]:
    for sec in parsed_doc.get("sections", []):
        if sec["char_start"] <= char_pos < sec["char_end"]:
            return sec.get("heading", ""), sec.get("section_number", "")
    return "", ""


def _page_for(parsed_doc: dict, char_pos: int) -> Optional[int]:
    for p in parsed_doc.get("pages", []):
        if p["char_start"] <= char_pos < p["char_end"]:
            return p.get("page_number")
    return None


def _is_self_contained(text: str) -> bool:
    """Reject obvious mid-context fragments and boilerplate."""
    text = text.strip()
    if len(text) < 50:
        return False
    if not text[0].isupper() and text[0] not in '"(["':
        return False
    if _BOILERPLATE.search(text):
        return False
    if _REFERENCE_TO_PRIOR.search(text):
        return False
    # Reject if mostly digits/punctuation (table content)
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.5:
        return False
    return True


def _candidate_passages_for_doc(
    parsed_doc: dict,
    *,
    min_tokens: int,
    max_tokens: int,
    target_tokens: int,
    min_sentences_apart: int,
) -> list[SourcePassage]:
    """Slide a sentence-window over the doc, emit candidate passages."""
    full_text: str = parsed_doc["full_text"]
    sentences = _split_sentences(full_text)
    if len(sentences) < 4:
        return []

    out: list[SourcePassage] = []
    last_emitted_idx = -10**9
    i = 0
    while i < len(sentences):
        # Greedily extend until we hit `target_tokens`
        j = i
        cur_tokens = 0
        while j < len(sentences):
            cur_tokens += len(_ENC.encode(sentences[j][0]))
            if cur_tokens >= target_tokens:
                j += 1
                break
            j += 1
        if cur_tokens < min_tokens or cur_tokens > max_tokens:
            i += 1
            continue

        cs = sentences[i][1]
        ce = sentences[j - 1][2] if j - 1 < len(sentences) else len(full_text)
        passage_text = full_text[cs:ce].strip()

        if not _is_self_contained(passage_text):
            i += 1
            continue

        # Enforce minimum spacing from previously emitted passage in this doc
        if i - last_emitted_idx < min_sentences_apart:
            i += 1
            continue
        last_emitted_idx = i

        section_title, section_number = _section_for(parsed_doc, cs)
        out.append(SourcePassage(
            passage_id="",                                     # filled in by caller
            module=parsed_doc["module"],
            source_doc_id=parsed_doc["doc_id"],
            source_doc_title=parsed_doc["doc_title"],
            source_doc_type=parsed_doc["doc_type"],
            passage_text=passage_text,
            char_start=cs,
            char_end=ce,
            n_tokens=cur_tokens,
            page_number=_page_for(parsed_doc, cs),
            section_title=section_title,
            section_number=section_number,
        ))
        i = j  # jump to end of this passage
    return out


def extract_passages(
    parsed_docs: list[dict],
    *,
    n_passages: int,
    min_tokens: int = 150,
    max_tokens: int = 400,
    target_tokens: int = 250,
    min_sentences_apart: int = 8,
    seed: int = 42,
) -> list[SourcePassage]:
    """Top-level: enumerate candidates per doc, then diversity-sample N total.

    Diversity sampling:
      - Round-robin through (doc_type) buckets so single-doc giants don't dominate
      - Within bucket, weighted by doc_id so all docs are represented
    """
    rng = random.Random(seed)
    by_doc_type: dict[str, list[SourcePassage]] = {}
    for doc in parsed_docs:
        cands = _candidate_passages_for_doc(
            doc, min_tokens=min_tokens, max_tokens=max_tokens,
            target_tokens=target_tokens, min_sentences_apart=min_sentences_apart,
        )
        if not cands:
            continue
        rng.shuffle(cands)
        by_doc_type.setdefault(doc["doc_type"], []).extend(cands)

    # Round-robin allocation: pick from each doc_type bucket in turn
    selected: list[SourcePassage] = []
    type_keys = list(by_doc_type.keys())
    rng.shuffle(type_keys)
    cursors = {k: 0 for k in type_keys}
    seen_doc_ids: dict[str, int] = {}                            # cap per doc

    while len(selected) < n_passages and any(cursors[k] < len(by_doc_type[k]) for k in type_keys):
        for k in type_keys:
            if len(selected) >= n_passages:
                break
            if cursors[k] >= len(by_doc_type[k]):
                continue
            # Find next candidate from this bucket whose doc isn't already maxed
            while cursors[k] < len(by_doc_type[k]):
                p = by_doc_type[k][cursors[k]]
                cursors[k] += 1
                # Cap: max 3 passages per individual doc
                if seen_doc_ids.get(p.source_doc_id, 0) >= 3:
                    continue
                seen_doc_ids[p.source_doc_id] = seen_doc_ids.get(p.source_doc_id, 0) + 1
                p.passage_id = f"{p.module}_p{len(selected):03d}"
                selected.append(p)
                break

    return selected
