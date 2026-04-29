"""Create the 6 Qdrant collections (one per module × strategy).

Each collection:
  - 5 named dense vectors (mxbai Matryoshka dims)
  - 2 named sparse vectors (SPLADE, BM25)
  - HNSW indexing on dense vectors (m=16, ef_construct=128)
  - Payload indexes for common filters

Idempotent: if a collection already exists, skip it (use --recreate to drop+create).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client.http import models as rest

from pipelines.shared.qdrant_client import (
    DENSE_DIMENSIONS,
    SPARSE_VECTOR_NAMES,
    _dense_name,
    all_collection_specs,
    get_client,
)


# Payload fields we want indexed for fast filtered search.
# Most are keyword (string equality); a couple are integer for range filters.
PAYLOAD_INDEXES = [
    ("doc_id", "keyword"),
    ("doc_type", "keyword"),
    ("module", "keyword"),
    ("regulatory_body", "keyword"),
    ("jurisdiction", "keyword"),
    ("company_ticker", "keyword"),
    ("section_type", "keyword"),
    ("chunk_level", "keyword"),
    ("contains_table", "bool"),
    ("fiscal_year", "integer"),
    ("fiscal_quarter", "integer"),
]


def vector_config() -> dict:
    """Named dense vector config: 5 entries, all cosine, all HNSW."""
    return {
        _dense_name(dim): rest.VectorParams(
            size=dim,
            distance=rest.Distance.COSINE,
            hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=128),
        )
        for dim in DENSE_DIMENSIONS
    }


def sparse_config() -> dict:
    """Named sparse vector config: SPLADE + BM25, both with default IDF modifier off
    (we compute weights ourselves; let Qdrant just dot-product)."""
    return {
        name: rest.SparseVectorParams(
            index=rest.SparseIndexParams(on_disk=False),
        )
        for name in SPARSE_VECTOR_NAMES
    }


def ensure_collection(client, name: str, *, recreate: bool) -> str:
    """Create the collection if missing. Returns 'created' | 'skipped' | 'recreated'."""
    exists = client.collection_exists(name)
    if exists and not recreate:
        return "skipped"
    if exists and recreate:
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=vector_config(),
        sparse_vectors_config=sparse_config(),
    )
    # Payload indexes
    for field_name, field_type in PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=field_type,
        )
    return "recreated" if exists else "created"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="Drop and recreate collections that already exist (DESTRUCTIVE).")
    args = ap.parse_args()

    client = get_client()
    specs = all_collection_specs()

    print(f"Setting up {len(specs)} Qdrant collections "
          f"({'recreate=True' if args.recreate else 'idempotent mode'})\n")

    statuses: list[tuple[str, str, str]] = []
    for module, strategy, name in specs:
        status = ensure_collection(client, name, recreate=args.recreate)
        statuses.append((name, module, strategy))
        print(f"  {status:>10s}  {name}  (module={module}, strategy={strategy})")

    # Verify
    print("\n=== Verification ===")
    server_collections = {c.name for c in client.get_collections().collections}
    for name, module, strategy in statuses:
        if name not in server_collections:
            print(f"  ! MISSING: {name}")
            return 1
        info = client.get_collection(name)
        print(f"  {name}: vectors={list(info.config.params.vectors.keys())}, "
              f"sparse={list(info.config.params.sparse_vectors.keys())}, "
              f"points={info.points_count or 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
