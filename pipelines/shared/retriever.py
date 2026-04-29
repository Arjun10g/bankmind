"""Hybrid retrieval over Qdrant.

One class, three modes, fully composable with the query transformers and
reranker cascade. Returns `ScoredChunk` objects so downstream consumers
(rerankers, fusion, generation) don't need to know they came from Qdrant.

Modes:
  - "dense"  : single named-vector search (any of dense_128…dense_1024)
  - "sparse" : single sparse-vector search (splade or bm25)
  - "hybrid" : Qdrant prefetch + RRF fusion of dense + splade + bm25 in one call

Per CLAUDE.md spec, each (module, chunk_strategy) pair gets its own collection
(`bankmind_{module}_{strategy}`). The retriever picks the right collection at
query time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qdrant_client.http import models as rest

from .embedder import DIMENSIONS, MatryoshkaEmbedder
from .qdrant_client import _dense_name, collection_name, get_client
from .sparse_encoder import BM25Encoder, SparseVec, SpladeEncoder


@dataclass
class ScoredChunk:
    chunk_id: str
    score: float
    payload: dict
    # Convenience accessors for the most-used payload fields
    @property
    def content(self) -> str:
        return self.payload.get("content", "")

    @property
    def doc_id(self) -> str:
        return self.payload.get("doc_id", "")

    @property
    def char_start(self) -> int:
        return int(self.payload.get("char_start", 0))

    @property
    def char_end(self) -> int:
        return int(self.payload.get("char_end", 0))


def _to_filter(filters: Optional[dict]) -> Optional[rest.Filter]:
    """Convert a {field: value} dict into a Qdrant Filter (all conjunctive)."""
    if not filters:
        return None
    must: list[rest.FieldCondition] = []
    for field_name, value in filters.items():
        if isinstance(value, list):
            must.append(rest.FieldCondition(
                key=field_name,
                match=rest.MatchAny(any=value),
            ))
        else:
            must.append(rest.FieldCondition(
                key=field_name,
                match=rest.MatchValue(value=value),
            ))
    return rest.Filter(must=must)


class HybridRetriever:
    """One Qdrant client, three retrieval modes, a clean Pythonic interface."""

    def __init__(
        self,
        *,
        embedder: Optional[MatryoshkaEmbedder] = None,
        splade: Optional[SpladeEncoder] = None,
        bm25: Optional[BM25Encoder] = None,
    ):
        self.client = get_client()
        # Lazy-loadable encoders — sparse-only retrieval shouldn't pay for mxbai
        self._embedder = embedder
        self._splade = splade
        self._bm25 = bm25

    @property
    def embedder(self) -> MatryoshkaEmbedder:
        if self._embedder is None:
            self._embedder = MatryoshkaEmbedder()
        return self._embedder

    @property
    def splade(self) -> SpladeEncoder:
        if self._splade is None:
            self._splade = SpladeEncoder()
        return self._splade

    @property
    def bm25(self) -> BM25Encoder:
        if self._bm25 is None:
            self._bm25 = BM25Encoder()
        return self._bm25

    # --- query encoding helpers ----------------------------------------------

    def encode_query_dense(self, query: str, dim: int) -> np.ndarray:
        if dim not in DIMENSIONS:
            raise ValueError(f"dim must be one of {DIMENSIONS}, got {dim}")
        full = self.embedder.embed_queries([query])           # (1, 1024)
        return self.embedder.truncate(full, dim)[0]            # (dim,)

    def encode_query_splade(self, query: str) -> SparseVec:
        return self.splade.encode_query(query)

    def encode_query_bm25(self, query: str) -> SparseVec:
        return self.bm25.encode_query(query)

    # --- core search ----------------------------------------------------------

    def search(
        self,
        *,
        query: str,
        module: str,
        chunk_strategy: str,
        mode: str = "hybrid",
        embedding_dim: int = 512,
        sparse_name: str = "splade",       # used only when mode='sparse'
        top_k: int = 10,
        prefetch_limit: int = 50,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        """Run one retrieval. Returns top_k ScoredChunks ordered by score desc."""
        coll = collection_name(module, chunk_strategy)
        qfilter = _to_filter(filters)

        if mode == "dense":
            qvec = self.encode_query_dense(query, embedding_dim)
            res = self.client.query_points(
                collection_name=coll,
                query=qvec.tolist(),
                using=_dense_name(embedding_dim),
                limit=top_k,
                with_payload=True,
                query_filter=qfilter,
            )
            return [ScoredChunk(chunk_id=str(p.id), score=float(p.score), payload=p.payload or {})
                    for p in res.points]

        if mode == "sparse":
            if sparse_name == "splade":
                sv = self.encode_query_splade(query)
            elif sparse_name == "bm25":
                sv = self.encode_query_bm25(query)
            else:
                raise ValueError(f"unknown sparse_name: {sparse_name}")
            res = self.client.query_points(
                collection_name=coll,
                query=rest.SparseVector(indices=sv.indices, values=sv.values),
                using=sparse_name,
                limit=top_k,
                with_payload=True,
                query_filter=qfilter,
            )
            return [ScoredChunk(chunk_id=str(p.id), score=float(p.score), payload=p.payload or {})
                    for p in res.points]

        if mode == "hybrid":
            qvec = self.encode_query_dense(query, embedding_dim)
            q_splade = self.encode_query_splade(query)
            q_bm25 = self.encode_query_bm25(query)
            res = self.client.query_points(
                collection_name=coll,
                prefetch=[
                    rest.Prefetch(query=qvec.tolist(),
                                  using=_dense_name(embedding_dim),
                                  limit=prefetch_limit, filter=qfilter),
                    rest.Prefetch(query=rest.SparseVector(indices=q_splade.indices,
                                                          values=q_splade.values),
                                  using="splade",
                                  limit=prefetch_limit, filter=qfilter),
                    rest.Prefetch(query=rest.SparseVector(indices=q_bm25.indices,
                                                          values=q_bm25.values),
                                  using="bm25",
                                  limit=prefetch_limit, filter=qfilter),
                ],
                query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return [ScoredChunk(chunk_id=str(p.id), score=float(p.score), payload=p.payload or {})
                    for p in res.points]

        raise ValueError(f"unknown mode: {mode}")

    # --- multi-mode search returning per-channel results separately ----------
    # Useful when the caller wants to apply its own fusion (convex, hierarchical).

    def search_separate_channels(
        self,
        *,
        query: str,
        module: str,
        chunk_strategy: str,
        embedding_dim: int = 512,
        top_k: int = 50,
        filters: Optional[dict] = None,
    ) -> tuple[list[ScoredChunk], list[ScoredChunk], list[ScoredChunk]]:
        """Returns (dense_results, splade_results, bm25_results). Each is top_k.

        Use this when you want to feed dense + sparse into a non-RRF fusion
        (convex combination, hierarchical, etc.).
        """
        return (
            self.search(query=query, module=module, chunk_strategy=chunk_strategy,
                        mode="dense", embedding_dim=embedding_dim,
                        top_k=top_k, filters=filters),
            self.search(query=query, module=module, chunk_strategy=chunk_strategy,
                        mode="sparse", sparse_name="splade",
                        top_k=top_k, filters=filters),
            self.search(query=query, module=module, chunk_strategy=chunk_strategy,
                        mode="sparse", sparse_name="bm25",
                        top_k=top_k, filters=filters),
        )
