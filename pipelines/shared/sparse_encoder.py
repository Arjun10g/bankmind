"""Sparse encoders for hybrid retrieval.

Two channels — both produce sparse vectors in the format Qdrant expects
(parallel arrays of token-index ints and float weights).

  - SPLADE: learned sparse representation. Captures term importance + soft
            term expansion. Strong on domain-specific corpora.
            Model: prithivida/Splade_PP_en_v1 (SPLADE++, distilled).
            (Replaces CLAUDE.md's naver/splade-cocondenser-ensembledistil —
            same SPLADE family, fastembed-native, comparable quality.)

  - BM25:   classic lexical bag-of-words with IDF + length normalization.
            Stateless tokenization (Snowball stemmer) — no corpus fit needed
            because fastembed's BM25 normalizes via fixed defaults; corpus-
            relative IDF is applied at query time. Excellent baseline for
            exact-term queries (regulatory codes, ticker symbols, fiscal
            years, etc.).

Output shape (per text): SparseVec = {"indices": [int, ...], "values": [float, ...]}.
Both are passed straight to qdrant_client.models.SparseVector at upsert time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fastembed import SparseEmbedding, SparseTextEmbedding


SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
BM25_MODEL = "Qdrant/bm25"


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


_MODEL_CACHE: dict[str, SparseTextEmbedding] = {}


def _get_model(model_name: str) -> SparseTextEmbedding:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SparseTextEmbedding(model_name=model_name)
    return _MODEL_CACHE[model_name]


def _to_sparse_vec(emb: SparseEmbedding) -> SparseVec:
    """fastembed returns numpy arrays; Qdrant wants Python lists."""
    return SparseVec(
        indices=[int(i) for i in emb.indices.tolist()],
        values=[float(v) for v in emb.values.tolist()],
    )


class SpladeEncoder:
    """SPLADE++ encoder — same call shape for documents and queries."""

    def __init__(self, model_name: str = SPLADE_MODEL):
        self.model_name = model_name
        self.model = _get_model(model_name)

    def encode(self, texts: list[str], *, batch_size: int = 16) -> list[SparseVec]:
        return [_to_sparse_vec(e) for e in self.model.embed(texts, batch_size=batch_size)]

    def encode_query(self, query: str) -> SparseVec:
        return self.encode([query])[0]


class BM25Encoder:
    """fastembed's BM25 — corpus-aware via Qdrant's native handling.

    Documents and queries are encoded with different methods because BM25 has
    asymmetric weighting: doc weights include TF + length norm, query weights
    are pure IDF.
    """

    def __init__(self, model_name: str = BM25_MODEL):
        self.model_name = model_name
        self.model = _get_model(model_name)

    def encode_documents(self, texts: list[str], *, batch_size: int = 32) -> list[SparseVec]:
        return [_to_sparse_vec(e) for e in self.model.embed(texts, batch_size=batch_size)]

    def encode_query(self, query: str) -> SparseVec:
        # fastembed's BM25 has a query_embed method that emits IDF-only weights
        return _to_sparse_vec(next(self.model.query_embed(query)))
