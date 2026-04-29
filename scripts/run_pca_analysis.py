"""Phase 5 — PCA eigenstructure analysis.

For each module (compliance, credit):
  1. Pull all dense_1024 vectors from the 3 strategy collections (aggregated)
  2. Fit full-rank PCA (sklearn)
  3. Detect elbow via three methods (Kneedle, second-derivative, 95%-variance)
  4. Persist:
       evaluation/results/{module}/pca_eigenstructure.json   (eigenvalues, cumvar, elbows)
       evaluation/results/{module}/pca_model.joblib          (fitted PCA — for query-time projection)

Why aggregate across strategies? PCA is invariant to redundant samples — the
eigenstructure reflects the corpus-level embedding geometry. Aggregating gives
a denser sample without distorting the principal directions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.shared.pca_analyzer import fit_pca, save
from pipelines.shared.qdrant_client import (
    DENSE_DIMENSIONS,
    _dense_name,
    all_collection_specs,
    get_client,
)


SCROLL_BATCH = 256


def fetch_dense_1024(client, collection_name: str) -> np.ndarray:
    """Scroll through a collection and return its dense_1024 vectors as (n, 1024)."""
    info = client.get_collection(collection_name)
    expected = info.points_count or 0
    if expected == 0:
        return np.zeros((0, 1024), dtype=np.float32)

    vectors: list[list[float]] = []
    offset = None
    pbar = tqdm(total=expected, desc=f"  scroll {collection_name.split('_', 1)[1]}", leave=False)
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_BATCH,
            with_payload=False,
            with_vectors=[_dense_name(1024)],
            offset=offset,
        )
        if not points:
            break
        for p in points:
            v = p.vector.get(_dense_name(1024)) if isinstance(p.vector, dict) else p.vector
            if v is not None:
                vectors.append(v)
        pbar.update(len(points))
        if offset is None:
            break
    pbar.close()
    if not vectors:
        return np.zeros((0, 1024), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def pull_module_embeddings(client, module: str) -> np.ndarray:
    """Aggregate dense_1024 vectors from all strategies belonging to `module`."""
    specs = [(s, n) for m, s, n in all_collection_specs() if m == module]
    print(f"\n[{module}] aggregating {len(specs)} collections")
    parts: list[np.ndarray] = []
    for strategy, name in specs:
        v = fetch_dense_1024(client, name)
        print(f"  {strategy:25s}  {len(v):>6,d} vectors")
        parts.append(v)
    if not parts:
        return np.zeros((0, 1024), dtype=np.float32)
    out = np.concatenate(parts, axis=0)
    print(f"  total                      {len(out):>6,d} vectors")
    return out


def main() -> int:
    client = get_client()
    out_root = ROOT / "evaluation" / "results"

    summaries: dict[str, dict] = {}

    for module in ("compliance", "credit"):
        embeddings = pull_module_embeddings(client, module)
        if len(embeddings) < 100:
            print(f"  ! not enough embeddings for {module}; skipping")
            continue

        print(f"  fitting PCA on ({embeddings.shape[0]} × {embeddings.shape[1]})...")
        t0 = time.perf_counter()
        # Use 'aggregated' as the source_strategy label since we mixed all 3
        pca, result = fit_pca(embeddings, module=module, source_strategy="aggregated")
        elapsed = time.perf_counter() - t0
        print(f"  fit done in {elapsed:.1f}s")

        out_dir = out_root / module
        save(
            pca, result,
            model_path=out_dir / "pca_model.joblib",
            json_path=out_dir / "pca_eigenstructure.json",
        )

        summaries[module] = {
            "n_embeddings": result.n_embeddings,
            "elbow_kneedle": result.elbow_kneedle,
            "elbow_kneedle_snapped_to_matryoshka": result.elbow_kneedle_snapped,
            "elbow_second_derivative": result.elbow_second_deriv,
            "elbow_95pct_variance": result.elbow_95pct,
            "cumvar_at_dims": {
                "128": round(result.cumulative_variance_at_128, 4),
                "256": round(result.cumulative_variance_at_256, 4),
                "512": round(result.cumulative_variance_at_512, 4),
                "768": round(result.cumulative_variance_at_768, 4),
                "1024": round(result.cumulative_variance_at_1024, 4),
            },
            "fit_seconds": round(elapsed, 1),
        }

        print(f"\n  [{module}] PCA findings:")
        print(f"    Kneedle elbow:           dim {result.elbow_kneedle}  (snapped to Matryoshka: {result.elbow_kneedle_snapped})")
        print(f"    Second-derivative elbow: dim {result.elbow_second_deriv}")
        print(f"    95%-variance elbow:      dim {result.elbow_95pct}")
        print(f"    Cumulative variance at Matryoshka dims:")
        for d in DENSE_DIMENSIONS:
            cv = getattr(result, f"cumulative_variance_at_{d}")
            print(f"      dim={d:>4d}: {cv * 100:>5.1f}%")

    # Cross-module summary
    summary_path = out_root / "_pca_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote summary → {summary_path.relative_to(ROOT)}")

    if "compliance" in summaries and "credit" in summaries:
        c = summaries["compliance"]
        cr = summaries["credit"]
        print(f"\n=== Cross-module comparison ===")
        print(f"  Kneedle elbow:        compliance={c['elbow_kneedle']}  vs  credit={cr['elbow_kneedle']}  "
              f"(Δ = {cr['elbow_kneedle'] - c['elbow_kneedle']:+d})")
        print(f"  95%-variance dim:     compliance={c['elbow_95pct_variance']}  vs  credit={cr['elbow_95pct_variance']}  "
              f"(Δ = {cr['elbow_95pct_variance'] - c['elbow_95pct_variance']:+d})")
        print(f"  Cumvar @ dim 256:     compliance={c['cumvar_at_dims']['256']:.3f}  vs  credit={cr['cumvar_at_dims']['256']:.3f}")
        print(f"  Cumvar @ dim 512:     compliance={c['cumvar_at_dims']['512']:.3f}  vs  credit={cr['cumvar_at_dims']['512']:.3f}")
        if c["elbow_kneedle"] < cr["elbow_kneedle"]:
            print(f"\n  → Hypothesis CONFIRMED: regulatory text has lower intrinsic dimensionality.")
        elif c["elbow_kneedle"] > cr["elbow_kneedle"]:
            print(f"\n  → Hypothesis REJECTED: credit-narrative text has lower intrinsic dimensionality.")
        else:
            print(f"\n  → Hypothesis INCONCLUSIVE: both modules have the same Kneedle elbow.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
