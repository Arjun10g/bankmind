"""Sweep hybrid_convex's α parameter to see if it can beat raw BM25.

The retrieval benchmark used α=0.7 (CLAUDE.md default — dense-heavy) and
hybrid_convex underperformed in both modules. Hypothesis: α was wrong because
sparse (BM25) is the strong channel, not dense. Sweeping α tests this directly.

Setup mirrors the retrieval benchmark Stage 1:
  - chunking=semantic, dim=512, no rerank, no transform
  - hybrid_convex blends DENSE + SPLADE (the two channels its function takes)

α=1.0 → all dense; α=0.0 → all SPLADE. Track A only — pure retrieval, no LLM cost.

Output: evaluation/results/{module}/hybrid_convex_alpha_sweep.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import evaluate_track_a
from pipelines.shared.fusion import convex_combination
from pipelines.shared.retriever import HybridRetriever

EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = ROOT / "evaluation" / "results"

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_STRATEGY = "semantic"
FIXED_DIM = 512
TOP_K = 10
PREFETCH_LIMIT = 50


def main() -> int:
    retriever = HybridRetriever()
    summary: dict = {}

    for module in ("compliance", "credit"):
        qa_pairs = json.loads((EVAL_DIR / f"{module}_qa.json").read_text())

        print(f"\n{'=' * 100}")
        print(f"[{module}] hybrid_convex α sweep — chunking={FIXED_STRATEGY}, dim={FIXED_DIM}")
        print(f"  α=1.0 → all dense  ·  α=0.0 → all SPLADE")
        print(f"{'=' * 100}")

        # Reuse channels — encode-once, fuse-many. Massive speedup over
        # re-running the full retrieval per α.
        print(f"  pre-encoding queries + retrieving {PREFETCH_LIMIT} candidates per channel...")
        t0 = time.perf_counter()
        per_query_channels: list[tuple[dict, list, list]] = []   # (qa, dense, splade)
        for qa in qa_pairs:
            if qa["track"] != "A":
                continue
            d, s, _ = retriever.search_separate_channels(
                query=qa["question"], module=module, chunk_strategy=FIXED_STRATEGY,
                embedding_dim=FIXED_DIM, top_k=PREFETCH_LIMIT,
            )
            per_query_channels.append((qa, d, s))
        print(f"  → encoded {len(per_query_channels)} queries in {time.perf_counter() - t0:.1f}s")

        results: dict = {}
        for alpha in ALPHAS:
            t0 = time.perf_counter()

            def make_retrieve_fn(_alpha=alpha):
                # Build a lookup so the evaluator's retrieve_fn can fuse the cached channels
                cache = {qa["question"]: (d, s) for qa, d, s in per_query_channels}
                def fn(query: str):
                    d, s = cache.get(query, ([], []))
                    return convex_combination(d, s, alpha=_alpha, top_k=TOP_K)
                return fn

            track_a_agg, _ = evaluate_track_a(
                qa_pairs, make_retrieve_fn(), top_k=TOP_K,
            )
            elapsed = time.perf_counter() - t0
            results[f"{alpha:.1f}"] = {**track_a_agg, "alpha": alpha,
                                       "elapsed_seconds": round(elapsed, 1)}
            print(f"  α={alpha:.1f}: NDCG={track_a_agg['ndcg']:.3f}  "
                  f"R@5={track_a_agg['recall_at_5']:.3f}  "
                  f"MRR={track_a_agg['mrr']:.3f}")

        # Find best
        best_alpha = max(results.keys(), key=lambda k: results[k]["ndcg"])
        best_ndcg = results[best_alpha]["ndcg"]
        print(f"\n  → best α = {best_alpha}  (NDCG={best_ndcg:.3f})")

        # Persist
        out_dir = OUT_DIR / module
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hybrid_convex_alpha_sweep.json").write_text(
            json.dumps({"results": results, "best_alpha": best_alpha,
                        "best_ndcg": best_ndcg}, indent=2)
        )
        summary[module] = {
            "best_alpha": best_alpha,
            "best_ndcg": best_ndcg,
            "by_alpha": {k: round(v["ndcg"], 3) for k, v in results.items()},
        }

    (OUT_DIR / "_alpha_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 100}\nALPHA SWEEP SUMMARY\n{'=' * 100}")
    print(f"{'α':>4}  | " + " | ".join(f"{m:>15s}" for m in ("compliance", "credit")))
    print("-" * 50)
    for alpha in ALPHAS:
        row = f"{alpha:>4.1f}  | "
        cells = []
        for m in ("compliance", "credit"):
            ndcg = summary[m]["by_alpha"][f"{alpha:.1f}"]
            cells.append(f"{ndcg:>15.3f}")
        row += " | ".join(cells)
        print(row)
    print(f"\n  best α — compliance: {summary['compliance']['best_alpha']} (NDCG={summary['compliance']['best_ndcg']:.3f})")
    print(f"  best α — credit:     {summary['credit']['best_alpha']} (NDCG={summary['credit']['best_ndcg']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
