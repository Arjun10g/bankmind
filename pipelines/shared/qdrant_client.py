"""Centralized Qdrant client setup + naming conventions.

One collection per (module, chunk_strategy) pair — 6 total. This makes
strategy-vs-strategy retrieval benchmarking a clean per-collection comparison
with no payload-filter overhead.

Each collection schema:
  - 5 named dense vectors: dense_128, dense_256, dense_512, dense_768, dense_1024
                           (cosine distance; mxbai-embed-large Matryoshka)
  - 2 sparse vectors:      splade, bm25
  - payload:               full Chunk dataclass minus the redundant chunk_strategy
                           (which is implicit from the collection name)
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient

DENSE_DIMENSIONS: tuple[int, ...] = (128, 256, 512, 768, 1024)
SPARSE_VECTOR_NAMES: tuple[str, ...] = ("splade", "bm25")

COMPLIANCE_STRATEGIES = ("regulatory_boundary", "semantic", "hierarchical")
CREDIT_STRATEGIES = ("financial_statement", "semantic", "narrative_section")


def _dense_name(dim: int) -> str:
    return f"dense_{dim}"


DENSE_VECTOR_NAMES: tuple[str, ...] = tuple(_dense_name(d) for d in DENSE_DIMENSIONS)


def collection_name(module: str, strategy: str, *, prefix: str | None = None) -> str:
    """{prefix}_{module}_{strategy} — the canonical naming convention."""
    if prefix is None:
        prefix = os.environ.get("QDRANT_COLLECTION_PREFIX", "bankmind")
    return f"{prefix}_{module}_{strategy}"


def all_collection_specs(prefix: str | None = None) -> list[tuple[str, str, str]]:
    """Return [(module, strategy, collection_name), ...] for all 6 collections."""
    out: list[tuple[str, str, str]] = []
    for s in COMPLIANCE_STRATEGIES:
        out.append(("compliance", s, collection_name("compliance", s, prefix=prefix)))
    for s in CREDIT_STRATEGIES:
        out.append(("credit", s, collection_name("credit", s, prefix=prefix)))
    return out


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Cached client. Loads .env on first call."""
    load_dotenv()
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in environment (or .env)."
        )
    return QdrantClient(url=url, api_key=api_key, timeout=60)
