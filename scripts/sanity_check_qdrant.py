"""Hybrid-search sanity check across all 6 collections.

For each test query, run three searches and print top-3 results from each:
  1. Dense-only (dense_512 — middle of the Matryoshka range)
  2. Sparse-only (SPLADE)
  3. Hybrid via Qdrant's prefetch + RRF fusion (dense_512 + SPLADE + BM25)

Goal: prove the embedding + sparse + storage pipeline is end-to-end correct,
that the loaded payloads are queryable, and that hybrid search returns
sensible results.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client.http import models as rest

from pipelines.shared.embedder import MatryoshkaEmbedder
from pipelines.shared.qdrant_client import all_collection_specs, get_client
from pipelines.shared.sparse_encoder import BM25Encoder, SpladeEncoder


# (module, query) — picked to hit specific known content
TEST_QUERIES = [
    ("compliance", "What is the Tier 1 capital ratio requirement under Basel III?"),
    ("compliance", "How does FINTRAC define a politically exposed person?"),
    ("compliance", "What are the residential mortgage underwriting standards in OSFI B-20?"),
    ("credit",     "What are the key credit risk factors disclosed in the 10-K?"),
    ("credit",     "How did JPMorgan's net interest income change year over year?"),
    ("credit",     "What is Goldman Sachs' Tier 1 capital ratio?"),
]

DENSE_VEC = "dense_512"
TOP_K = 3


def truncate(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def search_dense(client, coll, qvec, top_k):
    res = client.query_points(
        collection_name=coll,
        query=qvec.tolist(),
        using=DENSE_VEC,
        limit=top_k,
    )
    return res.points


def search_sparse(client, coll, q_sparse, vector_name, top_k):
    res = client.query_points(
        collection_name=coll,
        query=rest.SparseVector(indices=q_sparse.indices, values=q_sparse.values),
        using=vector_name,
        limit=top_k,
    )
    return res.points


def search_hybrid(client, coll, qvec, q_splade, q_bm25, top_k):
    """RRF fusion of dense + SPLADE + BM25. Each prefetch retrieves top-K each,
    then Qdrant fuses with reciprocal rank fusion."""
    res = client.query_points(
        collection_name=coll,
        prefetch=[
            rest.Prefetch(query=qvec.tolist(),
                          using=DENSE_VEC, limit=top_k * 4),
            rest.Prefetch(query=rest.SparseVector(indices=q_splade.indices,
                                                  values=q_splade.values),
                          using="splade", limit=top_k * 4),
            rest.Prefetch(query=rest.SparseVector(indices=q_bm25.indices,
                                                  values=q_bm25.values),
                          using="bm25", limit=top_k * 4),
        ],
        query=rest.FusionQuery(fusion=rest.Fusion.RRF),
        limit=top_k,
    )
    return res.points


def main() -> int:
    client = get_client()
    embedder = MatryoshkaEmbedder()
    splade = SpladeEncoder()
    bm25 = BM25Encoder()

    # Group collections by module
    coll_by_module: dict[str, list[tuple[str, str]]] = {"compliance": [], "credit": []}
    for module, strategy, name in all_collection_specs():
        coll_by_module[module].append((strategy, name))

    overall_ok = True

    for module, query in TEST_QUERIES:
        print(f"\n{'=' * 100}")
        print(f"Q: {query}")
        print(f"   module: {module}")

        # Encode once per query
        q_dense_full = embedder.embed_queries([query])[0]            # (1024,)
        q_dense_512 = embedder.truncate(q_dense_full[None, :], 512)[0]
        q_splade = splade.encode_query(query)
        q_bm25 = bm25.encode_query(query)

        for strategy, coll in coll_by_module[module]:
            print(f"\n  [{strategy}]")
            try:
                dense = search_dense(client, coll, q_dense_512, TOP_K)
                splade_res = search_sparse(client, coll, q_splade, "splade", TOP_K)
                hybrid = search_hybrid(client, coll, q_dense_512, q_splade, q_bm25, TOP_K)
            except Exception as e:
                print(f"    ✗ search failed: {type(e).__name__}: {e}")
                overall_ok = False
                continue

            for label, points in [("dense", dense), ("splade", splade_res), ("hybrid", hybrid)]:
                if not points:
                    print(f"    {label:>7s}: (no results)")
                    continue
                for i, p in enumerate(points):
                    content = truncate(p.payload.get("content", ""), 110)
                    src = p.payload.get("section_title") or p.payload.get("doc_id", "")
                    print(f"    {label if i == 0 else '':>7s}  #{i+1}  score={p.score:.3f}  "
                          f"[{truncate(src, 40)}]  {content}")

    print(f"\n{'=' * 100}\nSANITY_CHECK_OK={overall_ok}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
