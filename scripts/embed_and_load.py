"""Embed all chunks at every Matryoshka dim + SPLADE + BM25, upsert to Qdrant.

For each (module, strategy) pair:
  1. Load chunks from data/processed/{module}/chunks_{strategy}.jsonl
  2. Compute mxbai full-1024-dim dense embeddings (one forward pass), truncate to all 5 dims
  3. Compute SPLADE sparse vectors
  4. Compute BM25 sparse vectors
  5. Build PointStructs (chunk_id as point ID, payload = chunk dict, vectors as named map)
  6. Upsert in batches of 64 to the matching Qdrant collection

Idempotency: skips a collection if `points_count` already equals the chunk count.
Use `--force` to re-embed everything (slow). `--module` / `--strategy` to filter.

Total compute on CPU: ~30-90 minutes depending on machine, dominated by mxbai dense
embedding pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qdrant_client.http import models as rest
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.shared.embedder import DIMENSIONS, MatryoshkaEmbedder
from pipelines.shared.qdrant_client import (
    SPARSE_VECTOR_NAMES,
    _dense_name,
    all_collection_specs,
    get_client,
)
from pipelines.shared.sparse_encoder import BM25Encoder, SpladeEncoder

PROCESSED_DIR = ROOT / "data" / "processed"

UPSERT_BATCH = 64       # points per Qdrant upsert call
EMBED_BATCH = 64        # docs per dense embedding forward pass (MPS handles 64 well)


# Fields excluded from payload — large or redundant.
PAYLOAD_DROP = {"chunk_id"}   # chunk_id is the Qdrant point ID; no need to store twice


def load_chunks(module: str, strategy: str) -> list[dict]:
    path = PROCESSED_DIR / module / f"chunks_{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing chunk file: {path}")
    return [json.loads(line) for line in path.open()]


def chunk_to_payload(chunk: dict) -> dict:
    return {k: v for k, v in chunk.items() if k not in PAYLOAD_DROP}


def build_points(
    chunks: list[dict],
    dense_by_dim: dict[int, "np.ndarray"],
    splade_vecs: list,
    bm25_vecs: list,
) -> list[rest.PointStruct]:
    points: list[rest.PointStruct] = []
    for i, chunk in enumerate(chunks):
        named_dense = {_dense_name(d): dense_by_dim[d][i].tolist() for d in DIMENSIONS}
        named_sparse = {
            "splade": rest.SparseVector(
                indices=splade_vecs[i].indices, values=splade_vecs[i].values),
            "bm25": rest.SparseVector(
                indices=bm25_vecs[i].indices, values=bm25_vecs[i].values),
        }
        points.append(rest.PointStruct(
            id=chunk["chunk_id"],          # UUIDv5 from chunking_base.make_chunk_id
            vector={**named_dense, **named_sparse},
            payload=chunk_to_payload(chunk),
        ))
    return points


def process_collection(
    client,
    module: str,
    strategy: str,
    coll_name: str,
    embedder: MatryoshkaEmbedder,
    splade: SpladeEncoder,
    bm25: BM25Encoder,
    *,
    force: bool,
) -> dict:
    chunks = load_chunks(module, strategy)
    n_target = len(chunks)
    info = client.get_collection(coll_name)
    n_existing = info.points_count or 0

    if n_existing >= n_target and not force:
        print(f"  ⏭  {coll_name}: {n_existing} points already ≥ {n_target} chunks. Skipping (use --force to re-embed).")
        return {"collection": coll_name, "skipped": True, "n_chunks": n_target,
                "n_existing": n_existing}

    print(f"  ▶  {coll_name}: embedding {n_target} chunks "
          f"(currently {n_existing} points; force={force})")

    texts = [c["content"] for c in chunks]

    # 1. Dense — single mxbai forward pass for all chunks, then truncate to 5 dims.
    t0 = time.perf_counter()
    dense_by_dim = embedder.embed_documents_all_dims(texts, show_progress=True)
    dense_t = time.perf_counter() - t0
    print(f"     dense embeddings ({n_target} × {DIMENSIONS}): {dense_t:.1f}s "
          f"({n_target / max(dense_t, 1e-6):.1f} chunks/s)")

    # 2. SPLADE sparse
    t0 = time.perf_counter()
    splade_vecs = splade.encode(texts, batch_size=16)
    splade_t = time.perf_counter() - t0
    print(f"     SPLADE sparse: {splade_t:.1f}s ({n_target / max(splade_t, 1e-6):.1f} chunks/s)")

    # 3. BM25 sparse
    t0 = time.perf_counter()
    bm25_vecs = bm25.encode_documents(texts, batch_size=64)
    bm25_t = time.perf_counter() - t0
    print(f"     BM25 sparse:   {bm25_t:.1f}s ({n_target / max(bm25_t, 1e-6):.1f} chunks/s)")

    # 4. Build + upsert in batches
    points = build_points(chunks, dense_by_dim, splade_vecs, bm25_vecs)
    t0 = time.perf_counter()
    n_upserted = 0
    for i in tqdm(range(0, len(points), UPSERT_BATCH), desc=f"     upsert({coll_name.split('_')[-1]})", leave=False):
        batch = points[i:i + UPSERT_BATCH]
        client.upsert(collection_name=coll_name, points=batch, wait=False)
        n_upserted += len(batch)
    upsert_t = time.perf_counter() - t0
    print(f"     upserted {n_upserted} points: {upsert_t:.1f}s")

    # Verify after a brief pause (wait=False above)
    info = client.get_collection(coll_name)
    n_final = info.points_count or 0
    return {
        "collection": coll_name,
        "skipped": False,
        "n_chunks": n_target,
        "n_existing_before": n_existing,
        "n_existing_after": n_final,
        "n_upserted": n_upserted,
        "dense_seconds": round(dense_t, 1),
        "splade_seconds": round(splade_t, 1),
        "bm25_seconds": round(bm25_t, 1),
        "upsert_seconds": round(upsert_t, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=["compliance", "credit"], default=None,
                    help="Filter to one module")
    ap.add_argument("--strategy", default=None,
                    help="Filter to one strategy name")
    ap.add_argument("--force", action="store_true",
                    help="Re-embed even if collection already has points")
    args = ap.parse_args()

    client = get_client()
    embedder = MatryoshkaEmbedder(batch_size=EMBED_BATCH)
    splade = SpladeEncoder()
    bm25 = BM25Encoder()

    specs = all_collection_specs()
    if args.module:
        specs = [s for s in specs if s[0] == args.module]
    if args.strategy:
        specs = [s for s in specs if s[1] == args.strategy]

    print(f"\nProcessing {len(specs)} collection(s)\n")
    results = []
    grand_t0 = time.perf_counter()
    for module, strategy, coll_name in specs:
        try:
            r = process_collection(client, module, strategy, coll_name,
                                    embedder, splade, bm25, force=args.force)
            results.append(r)
        except Exception as e:
            print(f"  ✗  {coll_name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results.append({"collection": coll_name, "error": str(e)})
    print(f"\nTotal elapsed: {time.perf_counter() - grand_t0:.1f}s")

    # Summary
    summary_path = PROCESSED_DIR / "_qdrant_load_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nSummary → {summary_path.relative_to(ROOT)}")

    # Final state
    print("\n=== Final collection state ===")
    for module, strategy, coll_name in all_collection_specs():
        info = client.get_collection(coll_name)
        print(f"  {coll_name}: {info.points_count or 0} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
