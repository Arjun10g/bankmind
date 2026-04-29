"""Phase 7 chunking benchmark — the most important controlled experiment.

Per CLAUDE.md § 7.4:
  - Vary ONLY the chunking strategy (3 per module).
  - Fixed: dim=512, hybrid retrieval (dense + SPLADE + BM25 with RRF), no rerank,
    no query transform.
  - Track A scored via overlap relevance — fair across strategies (no
    chunk-ID-based bias).
  - Track B scored via answer quality (semantic sim + BERTScore F1 + concept
    coverage) — does chunking affect generation, not just retrieval?

Args:
  --modules: subset to run (default both)
  --top-k: retrieval top_k (default 10)
  --skip-track-b: skip Track B (faster; useful when iterating)

Outputs:
  evaluation/results/{module}/chunking_benchmark.json
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
from pipelines.shared.llm import claude_text
from pipelines.shared.retriever import HybridRetriever, ScoredChunk

EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = ROOT / "evaluation" / "results"

STRATEGIES = {
    "compliance": ["regulatory_boundary", "semantic", "hierarchical"],
    "credit":     ["financial_statement", "semantic", "narrative_section"],
}

FIXED_DIM = 512
TOP_K_FOR_GEN = 5


_GENERATE_PROMPT = """You are a senior {role}. Answer the user's question using ONLY the passages below. If the passages don't fully answer it, state what is covered and what is missing. Be specific and cite passage numbers when stating a fact.

Question: {query}

Passages:
{passages}

Answer (3-5 sentences, no preamble):"""


def make_retrieve_fn(retriever: HybridRetriever, module: str, strategy: str, top_k: int):
    def fn(query: str):
        return retriever.search(
            query=query, module=module, chunk_strategy=strategy,
            mode="hybrid", embedding_dim=FIXED_DIM, top_k=top_k,
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
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-track-b", action="store_true")
    args = ap.parse_args()

    retriever = HybridRetriever()
    summary: dict = {}

    for module in args.modules:
        qa_path = EVAL_DIR / f"{module}_qa.json"
        if not qa_path.exists():
            print(f"  ! missing {qa_path}; run scripts/generate_qa_pairs.py first")
            continue
        qa_pairs = json.loads(qa_path.read_text())

        print(f"\n{'=' * 100}")
        print(f"[{module}] chunking benchmark — {len(qa_pairs)//2} queries × {len(STRATEGIES[module])} strategies")
        print(f"  fixed: dim={FIXED_DIM}, retrieval=hybrid+RRF, reranker=none, transform=none")
        print(f"{'=' * 100}")

        module_results: dict = {}
        for strategy in STRATEGIES[module]:
            print(f"\n  ▶ {strategy}")
            t0 = time.perf_counter()

            retrieve_fn = make_retrieve_fn(retriever, module, strategy, args.top_k)
            track_a_agg, _ = evaluate_track_a(qa_pairs, retrieve_fn, top_k=args.top_k)

            track_b_agg = {}
            if not args.skip_track_b:
                gen_fn = make_generate_fn(module)
                track_b_agg, _ = evaluate_track_b(qa_pairs, retrieve_fn, gen_fn,
                                                   top_k_for_gen=TOP_K_FOR_GEN)

            elapsed = time.perf_counter() - t0
            module_results[strategy] = {
                **track_a_agg,
                **track_b_agg,
                "elapsed_seconds": round(elapsed, 1),
            }

            # Pretty-print key metrics
            print(f"    Track A: NDCG@10={track_a_agg.get('ndcg', 0):.3f}  "
                  f"MRR={track_a_agg.get('mrr', 0):.3f}  "
                  f"Recall@5={track_a_agg.get('recall_at_5', 0):.3f}  "
                  f"P95={track_a_agg.get('p95_latency_ms', 0):.0f}ms")
            if track_b_agg:
                print(f"    Track B: Composite={track_b_agg.get('track_b_composite', 0):.3f}  "
                      f"Sem={track_b_agg.get('track_b_semantic_sim', 0):.3f}  "
                      f"BERT_F1={track_b_agg.get('track_b_bertscore_f1', 0):.3f}  "
                      f"Concept={track_b_agg.get('track_b_concept_coverage') or 0:.3f}")
            print(f"    elapsed: {elapsed:.1f}s")

        out_dir = OUT_DIR / module
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunking_benchmark.json"
        out_path.write_text(json.dumps(module_results, indent=2))
        print(f"\n  → {out_path.relative_to(ROOT)}")
        summary[module] = module_results

    (OUT_DIR / "_chunking_benchmark_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_chunking_benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    # Cross-module winner table
    print(f"\n{'=' * 100}")
    print("WINNERS")
    for module in args.modules:
        if module not in summary:
            continue
        ranked = sorted(summary[module].items(), key=lambda kv: kv[1].get("ndcg", 0), reverse=True)
        print(f"  [{module}] by NDCG@10:")
        for s, m in ranked:
            print(f"    {s:25s}  ndcg={m.get('ndcg', 0):.3f}  composite_b={m.get('track_b_composite', 0):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
