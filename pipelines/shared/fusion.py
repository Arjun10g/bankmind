"""Score fusion across retrieval channels.

Per CLAUDE.md § 6.3, three fusion methods to compare:

  1. RRF (reciprocal rank fusion) — score-magnitude-agnostic. Best when channel
     scores aren't directly comparable (BM25 raw scores vs cosine sims).
     score_i = Σ 1 / (k + rank_i)

  2. Convex combination — α·dense + (1-α)·sparse, after min-max normalization
     so the channels are on the same [0,1] scale. Use when relative score gaps
     matter, not just rankings.

  3. Hierarchical — query-aware routing. If the query has exact-term signals
     (regulatory codes, fiscal years, quoted phrases) → weight sparse higher.
     If purely semantic → weight dense higher. Short queries → sparse-only.

NOTE: Qdrant has native RRF for inside-Qdrant fusion (used by HybridRetriever
in 'hybrid' mode). The functions here are for client-side fusion when results
come from multiple Qdrant queries (e.g., Multi-Query expansion fans out to
several queries and we fuse the unioned results).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .retriever import ScoredChunk


# --- RRF -------------------------------------------------------------------

def rrf(
    result_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Reciprocal rank fusion.

    `k` smooths the rank weighting; k=60 is the standard from the original
    paper. Lower k amplifies rank differences (rank 1 dominates more);
    higher k flattens.
    """
    scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] += 1.0 / (k + rank)
            if chunk.chunk_id not in payloads:
                payloads[chunk.chunk_id] = chunk.payload
    fused = [
        ScoredChunk(chunk_id=cid, score=s, payload=payloads[cid])
        for cid, s in scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_k] if top_k else fused


# --- Convex combination ----------------------------------------------------

def _min_max_normalize(results: list[ScoredChunk]) -> dict[str, float]:
    """Returns {chunk_id: normalized_score in [0,1]}. Empty input → {}."""
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return {r.chunk_id: 1.0 for r in results}
    span = hi - lo
    return {r.chunk_id: (r.score - lo) / span for r in results}


def convex_combination(
    dense: list[ScoredChunk],
    sparse: list[ScoredChunk],
    *,
    alpha: float = 0.7,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """final_score = α · dense_norm + (1−α) · sparse_norm.

    Min-max normalizes each channel first — RAW BM25 scores (10-20+) and
    cosine sims (0-1) are NOT comparable; failing to normalize will let BM25
    dominate the fused order entirely. Tune α via the eval; CLAUDE.md
    suggests sweeping {0.5, 0.6, 0.7, 0.8}.
    """
    dn = _min_max_normalize(dense)
    sn = _min_max_normalize(sparse)
    payloads: dict[str, dict] = {}
    for r in dense + sparse:
        payloads.setdefault(r.chunk_id, r.payload)

    combined: dict[str, float] = defaultdict(float)
    for cid, s in dn.items():
        combined[cid] += alpha * s
    for cid, s in sn.items():
        combined[cid] += (1.0 - alpha) * s

    fused = [ScoredChunk(chunk_id=cid, score=score, payload=payloads[cid])
             for cid, score in combined.items()]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_k] if top_k else fused


# --- Hierarchical (query-aware routing) -----------------------------------

# Regulatory codes (B-20, E-23), fiscal years (2024), quoted phrases ("..."),
# 4+ digit numbers — strong signals that exact-term matching matters.
_EXACT_TERM_PATTERNS = re.compile(
    r'\b[A-Z]-\d+|"[^"]+"|\b\d{4,}\b|\bItem \d+[A-Z]?\b',
)


def hierarchical(
    query: str,
    dense: list[ScoredChunk],
    sparse: list[ScoredChunk],
    *,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Intelligent fusion router based on query characteristics.

      - Query < 5 tokens                → sparse-only (too few words for embeddings)
      - Has exact-term signals          → α=0.4 (sparse-heavy)
      - Long query (>15 tokens), no exact → α=0.85 (dense-heavy)
      - Default                         → RRF (rank-based, ignores score scale)
    """
    tokens = query.split()
    has_exact = bool(_EXACT_TERM_PATTERNS.search(query))

    if len(tokens) < 5:
        return sparse[:top_k] if top_k else sparse
    if has_exact:
        return convex_combination(dense, sparse, alpha=0.4, top_k=top_k)
    if len(tokens) > 15:
        return convex_combination(dense, sparse, alpha=0.85, top_k=top_k)
    return rrf([dense, sparse], top_k=top_k)
