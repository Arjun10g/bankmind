"""Semantic chunking — used by both compliance and credit modules.

Algorithm (per CLAUDE.md):
  1. Split text into sentences (preserving char offsets).
  2. Embed each sentence with a small sentence-transformer.
  3. Walk a sliding window of size W; for each gap between sentence i and i+1,
     compute cosine similarity of the W-sentence windows on either side.
  4. Boundary point = where similarity drops below `threshold`.
  5. Pack sentence segments between boundaries into chunks within [min_tokens, max_tokens].

For credit module: also pass `forbidden_break_ranges` (table char ranges) — boundaries
that fall inside a table are skipped so tables stay atomic in their chunks.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking_base import count_tokens, split_into_sentences


_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def semantic_chunk(
    full_text: str,
    *,
    threshold: float = 0.5,
    window: int = 3,
    min_tokens: int = 150,
    max_tokens: int = 512,
    forbidden_break_ranges: list[tuple[int, int]] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[tuple[str, int, int, int]]:
    """Return [(chunk_text, char_start, char_end, n_tokens)].

    `forbidden_break_ranges`: list of (char_start, char_end) within which boundaries
    may not be placed (e.g. table ranges in credit docs).
    """
    sentences = split_into_sentences(full_text)
    if not sentences:
        return []
    if len(sentences) < 2 * window:
        # Too few sentences — emit as a single chunk
        text = full_text.strip()
        if not text:
            return []
        return [(text, 0, len(full_text), count_tokens(text))]

    forbidden = forbidden_break_ranges or []

    def in_forbidden(pos: int) -> bool:
        return any(s <= pos < e for s, e in forbidden)

    sent_texts = [s[0] for s in sentences]
    model = _get_model(model_name)
    embeddings = model.encode(sent_texts, show_progress_bar=False, convert_to_numpy=True)

    # Compute boundary scores: similarity between window before gap i and after.
    # Gap i is between sentence i and sentence i+1, valid for i in [window-1, n-window-1].
    boundaries: list[int] = [0]   # sentence indices where a new chunk starts
    n = len(sentences)
    for i in range(window - 1, n - window):
        before = embeddings[i - window + 1 : i + 1].mean(axis=0)
        after = embeddings[i + 1 : i + 1 + window].mean(axis=0)
        sim = _cosine(before, after)
        gap_char_pos = sentences[i][2]    # end of sentence i
        if sim < threshold and not in_forbidden(gap_char_pos):
            boundaries.append(i + 1)
    boundaries.append(n)

    # Boundaries partitions sentences into segments. Some segments will already
    # exceed max_tokens (dense regulatory/financial text often has few topic
    # shifts at threshold=0.5). Subdivide those at sentence boundaries first,
    # then pack the resulting pieces.
    raw_segments: list[tuple[int, int]] = []     # sentence-index ranges
    for i in range(len(boundaries) - 1):
        s_start = boundaries[i]
        s_end = boundaries[i + 1]
        if s_start >= s_end:
            continue
        # Subdivide oversized segments at sentence boundaries — pack greedily
        cur_start = s_start
        cur_tokens = 0
        for j in range(s_start, s_end):
            stoks = count_tokens(sent_texts[j])
            if cur_tokens + stoks > max_tokens and (j - cur_start) > 0:
                raw_segments.append((cur_start, j))
                cur_start = j
                cur_tokens = stoks
            else:
                cur_tokens += stoks
        if cur_start < s_end:
            raw_segments.append((cur_start, s_end))

    # Materialize segments
    materialized: list[tuple[str, int, int, int]] = []
    for s_start, s_end in raw_segments:
        text = " ".join(sent_texts[s_start:s_end])
        char_start = sentences[s_start][1]
        char_end = sentences[s_end - 1][2]
        materialized.append((text, char_start, char_end, count_tokens(text)))

    # Merge adjacent segments that are too small (< min_tokens) into the next one,
    # provided combined size stays within max_tokens.
    out: list[tuple[str, int, int, int]] = []
    i = 0
    while i < len(materialized):
        text, cs, ce, tok = materialized[i]
        # Try to absorb next segment if current is below min and combined fits
        while (tok < min_tokens and i + 1 < len(materialized)
               and tok + materialized[i + 1][3] <= max_tokens):
            ntext, _ncs, nce, ntok = materialized[i + 1]
            text = text + " " + ntext
            ce = nce
            tok += ntok
            i += 1
        out.append((text, cs, ce, tok))
        i += 1
    return out
