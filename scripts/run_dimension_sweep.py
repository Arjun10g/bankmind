"""Phase 7 dimension sweep — does credit survive at lower dims per the PCA finding?

Per CLAUDE.md § 7.3 + the Phase 5 PCA result:
  PCA suggests credit reaches 92.6% variance at dim 256, 81.9% at dim 128.
  Compliance reaches 91.3% at 256, 78.1% at 128. So credit *should* tolerate
  more aggressive Matryoshka truncation than compliance.

Setup:
  - Fixed: chunking=semantic (the chunking benchmark winner)
  - Fixed: hybrid retrieval (dense_{dim} + SPLADE + BM25, RRF-fused)
  - Vary: embedding_dim ∈ {128, 256, 512, 768, 1024}
  - Both modules

Output:
  evaluation/results/{module}/dimension_sweep.json
  evaluation/results/_dimension_sweep_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import evaluate_track_a, evaluate_track_b
from pipelines.shared.embedder import DIMENSIONS
from pipelines.shared.llm import claude_text
from pipelines.shared.retriever import HybridRetriever, ScoredChunk

EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = ROOT / "evaluation" / "results"

FIXED_STRATEGY = "semantic"     # chunking benchmark winner
TOP_K = 10
TOP_K_FOR_GEN = 5


_GENERATE_PROMPT = """You are a senior {role}. Answer the user's question using ONLY the passages below. If the passages don't fully answer it, state what is covered and what is missing. Be specific and cite passage numbers when stating a fact.

Question: {query}

Passages:
{passages}

Answer (3-5 sentences, no preamble):"""


def make_retrieve_fn(retriever, module, dim, top_k):
    def fn(query: str):
        return retriever.search(
            query=query, module=module, chunk_strategy=FIXED_STRATEGY,
            mode="hybrid", embedding_dim=dim, top_k=top_k,
        )
    return fn


def make_generate_fn(module: str):
    role = "compliance officer" if module == "compliance" else "credit analyst"
    def fn(query: str, top: list[ScoredChunk]) -> str:
        passages = "\n\n".join(
            f"[{i+1}] (doc: {c.payload.get('doc_id','?')}, section: {c.payload.get('section_title','')})\n{c.content[:1500]}"
            for i, c in enumerate(top)
        )
        return claude_text(
            _GENERATE_PROMPT.format(role=role, query=query, passages=passages),
            max_tokens=400,
        )
    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", choices=["compliance", "credit"],
                    default=["compliance", "credit"])
    ap.add_argument("--skip-track-b", action="store_true",
                    help="Skip Track B (much faster — Track A only)")
    args = ap.parse_args()

    retriever = HybridRetriever()
    summary: dict = {}

    for module in args.modules:
        qa_pairs = json.loads((EVAL_DIR / f"{module}_qa.json").read_text())

        print(f"\n{'=' * 100}")
        print(f"[{module}] dimension sweep — chunking={FIXED_STRATEGY}  "
              f"dims={list(DIMENSIONS)}  top_k={TOP_K}  skip_track_b={args.skip_track_b}")
        print(f"{'=' * 100}")

        module_results: dict = {}
        for dim in DIMENSIONS:
            print(f"\n  ▶ dim={dim}")
            t0 = time.perf_counter()

            retrieve_fn = make_retrieve_fn(retriever, module, dim, TOP_K)
            track_a_agg, _ = evaluate_track_a(qa_pairs, retrieve_fn, top_k=TOP_K)

            track_b_agg = {}
            if not args.skip_track_b:
                gen_fn = make_generate_fn(module)
                track_b_agg, _ = evaluate_track_b(qa_pairs, retrieve_fn, gen_fn,
                                                   top_k_for_gen=TOP_K_FOR_GEN)

            elapsed = time.perf_counter() - t0
            module_results[str(dim)] = {
                "dim": dim,
                **track_a_agg,
                **track_b_agg,
                "elapsed_seconds": round(elapsed, 1),
            }

            print(f"    Track A: NDCG@10={track_a_agg.get('ndcg', 0):.3f}  "
                  f"MRR={track_a_agg.get('mrr', 0):.3f}  "
                  f"R@5={track_a_agg.get('recall_at_5', 0):.3f}  "
                  f"avg_lat={track_a_agg.get('avg_latency_ms', 0):.0f}ms  "
                  f"p95={track_a_agg.get('p95_latency_ms', 0):.0f}ms")
            if track_b_agg:
                print(f"    Track B: Composite={track_b_agg.get('track_b_composite', 0):.3f}  "
                      f"BERT_F1={track_b_agg.get('track_b_bertscore_f1', 0):.3f}  "
                      f"Concept={track_b_agg.get('track_b_concept_coverage') or 0:.3f}")
            print(f"    elapsed: {elapsed:.1f}s")

        out_dir = OUT_DIR / module
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dimension_sweep.json").write_text(json.dumps(module_results, indent=2))
        summary[module] = module_results

    (OUT_DIR / "_dimension_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # Cross-module summary table
    print(f"\n{'=' * 100}")
    print("DIMENSION SWEEP SUMMARY")
    print(f"{'=' * 100}")
    print(f"{'dim':>6}  | " + " | ".join(f"{m:>22s}" for m in args.modules))
    print(f"        | " + " | ".join("ndcg / r@5 / track_b" for _ in args.modules))
    print("-" * (8 + 25 * len(args.modules)))
    for dim in DIMENSIONS:
        row = f"{dim:>6}  | "
        cells = []
        for m in args.modules:
            r = summary[m][str(dim)]
            cells.append(f"{r.get('ndcg',0):.3f} / {r.get('recall_at_5',0):.3f} / {r.get('track_b_composite',0) if r.get('track_b_composite') else 0:.3f}")
        row += " | ".join(f"{c:>22s}" for c in cells)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
