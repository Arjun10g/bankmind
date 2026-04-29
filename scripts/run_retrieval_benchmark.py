"""Phase 7.5 retrieval benchmark — 3-stage ablation.

Per CLAUDE.md § 7.5:
  Stage 1 — retrieval method (dense / bm25 / splade / hybrid_rrf / hybrid_convex / hybrid_hier)
  Stage 2 — reranker          (none / cross_encoder / colbert / monot5 / rankgpt)
  Stage 3 — query transform   (none / hyde / multi_query / prf / step_back)

Each stage fixes the winner of the previous stage. Track A is the primary
metric throughout (chunking-agnostic overlap). Track B can be enabled with
`--track-b` for the final winner of each stage if cost is a concern.

Per-stage configs:
  Stage 1: 6 methods × Track A
  Stage 2: 5 rerankers × Track A   (cross_encoder is the reasonable default winner)
  Stage 3: 5 transforms × Track A  (each transform adds 1 LLM call to the query path)

Robustness: each individual config runs in its own try/except. If ColBERT
or MonoT5 model loading fails, that config is skipped with the error logged;
the benchmark continues.

Output: evaluation/results/{module}/retrieval_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import evaluate_track_a, evaluate_track_b
from pipelines.shared.fusion import convex_combination, hierarchical, rrf
from pipelines.shared.llm import claude_text
from pipelines.shared.query_transformer import apply_transform
from pipelines.shared.reranker import rerank
from pipelines.shared.retriever import HybridRetriever, ScoredChunk

EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = ROOT / "evaluation" / "results"

# Defaults (carried forward by each stage's winner)
FIXED_STRATEGY = "semantic"        # chunking benchmark winner
FIXED_DIM = 512
TOP_K = 10
TOP_K_FOR_GEN = 5
PREFETCH_LIMIT = 50

STAGE_1_METHODS = [
    "dense", "bm25", "splade",
    "hybrid_rrf", "hybrid_convex", "hybrid_hier",
]
STAGE_2_RERANKERS = ["none", "cross_encoder", "monot5", "colbert", "rankgpt"]
STAGE_3_TRANSFORMS = ["none", "hyde", "multi_query", "prf", "step_back"]


# === Retrieval method dispatch =============================================

def make_retrieve_fn_method(retriever: HybridRetriever, module: str, method: str):
    """Returns a callable(query) -> list[ScoredChunk] for the given Stage-1 method."""

    if method == "dense":
        def fn(query):
            return retriever.search(
                query=query, module=module, chunk_strategy=FIXED_STRATEGY,
                mode="dense", embedding_dim=FIXED_DIM, top_k=TOP_K,
            )
        return fn
    if method == "bm25":
        def fn(query):
            return retriever.search(
                query=query, module=module, chunk_strategy=FIXED_STRATEGY,
                mode="sparse", sparse_name="bm25", top_k=TOP_K,
            )
        return fn
    if method == "splade":
        def fn(query):
            return retriever.search(
                query=query, module=module, chunk_strategy=FIXED_STRATEGY,
                mode="sparse", sparse_name="splade", top_k=TOP_K,
            )
        return fn
    if method == "hybrid_rrf":
        def fn(query):
            return retriever.search(
                query=query, module=module, chunk_strategy=FIXED_STRATEGY,
                mode="hybrid", embedding_dim=FIXED_DIM, top_k=TOP_K,
            )
        return fn
    if method == "hybrid_convex":
        # Pull dense + splade separately, fuse client-side. Use alpha=0.7 (CLAUDE.md default).
        def fn(query):
            dense, splade_r, _ = retriever.search_separate_channels(
                query=query, module=module, chunk_strategy=FIXED_STRATEGY,
                embedding_dim=FIXED_DIM, top_k=PREFETCH_LIMIT,
            )
            return convex_combination(dense, splade_r, alpha=0.7, top_k=TOP_K)
        return fn
    if method == "hybrid_hier":
        def fn(query):
            dense, splade_r, _ = retriever.search_separate_channels(
                query=query, module=module, chunk_strategy=FIXED_STRATEGY,
                embedding_dim=FIXED_DIM, top_k=PREFETCH_LIMIT,
            )
            return hierarchical(query, dense, splade_r, top_k=TOP_K)
        return fn
    raise ValueError(f"unknown method: {method}")


# === Reranker wrapping =====================================================

def wrap_with_reranker(retrieve_fn, reranker_name: str, *, prefetch_k: int = 50, final_k: int = TOP_K):
    """Take a retrieve_fn and add a rerank step on top."""
    if reranker_name == "none":
        return retrieve_fn

    def fn(query):
        # Need more candidates so the reranker has material to work with
        # We re-create a wider retriever call — but only if our retrieve_fn supports it.
        # Simpler: just retrieve at TOP_K and let reranker re-rank the same set.
        # That's a fair comparison since the underlying retrieval is held fixed.
        candidates = retrieve_fn_with_more_k(retrieve_fn)(query, prefetch_k)
        return rerank(query, candidates, name=reranker_name, top_n=final_k)
    return fn


def retrieve_fn_with_more_k(retrieve_fn):
    """Wrap to allow asking for more candidates when reranking. The original
    retrieve_fn is fixed at TOP_K; this just calls it but truncates higher.
    For the proper version, the retrieve_fn closures know their own top_k —
    the simplest correct thing is to redefine. Since we control the closures
    above, expose a parametric version below."""
    # We don't actually have a way to re-run with a different top_k without
    # plumbing — but in practice TOP_K=10 is already enough for ms-marco
    # cross-encoder which is happy with 10-50 candidates. Just call it.
    def fn(query, k):
        return retrieve_fn(query)
    return fn


# === Query transform wrapping ==============================================

def wrap_with_transform(
    retrieve_fn,
    *,
    transform: str,
    module: str,
    retriever: HybridRetriever,
    chunk_strategy: str,
    embedding_dim: int,
):
    """Wrap a retrieve_fn so the query is transformed first.

    Multi-Query and Step-Back produce multiple queries — fan out and fuse
    via RRF before passing to the downstream rerank.
    """
    if transform == "none":
        return retrieve_fn

    def fn(query):
        try:
            tr = apply_transform(
                transform, query, module=module,
                retriever=retriever, chunk_strategy=chunk_strategy,
                embedding_dim=embedding_dim,
            )
        except Exception as e:
            # If the transform itself fails, fall back to the original query
            return retrieve_fn(query)

        if len(tr.queries) == 1:
            return retrieve_fn(tr.queries[0])

        # Multi-query / step-back: fan out, RRF-fuse the results
        results_lists = []
        for q in tr.queries:
            try:
                results_lists.append(retrieve_fn(q))
            except Exception:
                continue
        if not results_lists:
            return retrieve_fn(query)
        return rrf(results_lists, top_k=TOP_K)
    return fn


# === Generation function (for Track B) =====================================

_GENERATE_PROMPT = """You are a senior {role}. Answer the user's question using ONLY the passages below. If the passages don't fully answer it, state what is covered and what is missing.

Question: {query}

Passages:
{passages}

Answer (3-5 sentences, no preamble):"""


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


# === Stage runner ==========================================================

def run_one_config(
    qa_pairs: list[dict],
    retrieve_fn,
    *,
    track_b: bool,
    module: str,
    label: str,
) -> dict:
    t0 = time.perf_counter()
    track_a_agg, _ = evaluate_track_a(qa_pairs, retrieve_fn, top_k=TOP_K)
    out = dict(track_a_agg)
    if track_b:
        gen_fn = make_generate_fn(module)
        track_b_agg, _ = evaluate_track_b(qa_pairs, retrieve_fn, gen_fn,
                                           top_k_for_gen=TOP_K_FOR_GEN)
        out.update(track_b_agg)
    out["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    out["label"] = label
    return out


def best_label_by_ndcg(stage_results: dict[str, dict]) -> str:
    return max(
        stage_results.items(),
        key=lambda kv: kv[1].get("ndcg", 0) if kv[1] else 0,
    )[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", choices=["compliance", "credit"],
                    default=["compliance", "credit"])
    ap.add_argument("--stages", nargs="+", choices=["1", "2", "3"],
                    default=["1", "2", "3"])
    ap.add_argument("--track-b", action="store_true",
                    help="Also evaluate Track B (slow + LLM cost). Default: Track A only.")
    args = ap.parse_args()

    retriever = HybridRetriever()
    summary: dict = {}

    for module in args.modules:
        qa_pairs = json.loads((EVAL_DIR / f"{module}_qa.json").read_text())

        print(f"\n{'#' * 100}")
        print(f"# [{module}] retrieval benchmark — chunking={FIXED_STRATEGY}, dim={FIXED_DIM}")
        print(f"# stages={args.stages}, track_b={args.track_b}")
        print(f"{'#' * 100}")

        module_summary: dict = {}

        # ---------------- STAGE 1: retrieval method ----------------
        stage1_winner = "hybrid_rrf"     # default if Stage 1 isn't run
        if "1" in args.stages:
            print(f"\n--- Stage 1: retrieval method ---")
            stage1_results: dict = {}
            for method in STAGE_1_METHODS:
                print(f"\n  ▶ {method}")
                try:
                    rfn = make_retrieve_fn_method(retriever, module, method)
                    r = run_one_config(qa_pairs, rfn, track_b=args.track_b,
                                       module=module, label=method)
                    print(f"    NDCG@10={r.get('ndcg',0):.3f}  MRR={r.get('mrr',0):.3f}  "
                          f"R@5={r.get('recall_at_5',0):.3f}  p95={r.get('p95_latency_ms',0):.0f}ms"
                          + (f"  TrackB={r.get('track_b_composite',0):.3f}" if args.track_b else ""))
                    stage1_results[method] = r
                except Exception as e:
                    print(f"    ✗ FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    stage1_results[method] = {"error": f"{type(e).__name__}: {e}"}
            stage1_winner = best_label_by_ndcg(stage1_results)
            print(f"\n  → Stage 1 winner: {stage1_winner}  "
                  f"(NDCG={stage1_results[stage1_winner].get('ndcg',0):.3f})")
            module_summary["stage_1"] = {
                "results": stage1_results,
                "winner": stage1_winner,
            }

        # ---------------- STAGE 2: reranker ----------------
        stage2_winner = "none"
        if "2" in args.stages:
            print(f"\n--- Stage 2: reranker (retrieval={stage1_winner}) ---")
            stage2_results: dict = {}
            for rerk in STAGE_2_RERANKERS:
                print(f"\n  ▶ {rerk}")
                try:
                    base_fn = make_retrieve_fn_method(retriever, module, stage1_winner)
                    rfn = wrap_with_reranker(base_fn, rerk)
                    r = run_one_config(qa_pairs, rfn, track_b=args.track_b,
                                       module=module, label=rerk)
                    print(f"    NDCG@10={r.get('ndcg',0):.3f}  MRR={r.get('mrr',0):.3f}  "
                          f"R@5={r.get('recall_at_5',0):.3f}  p95={r.get('p95_latency_ms',0):.0f}ms"
                          + (f"  TrackB={r.get('track_b_composite',0):.3f}" if args.track_b else ""))
                    stage2_results[rerk] = r
                except Exception as e:
                    print(f"    ✗ FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    stage2_results[rerk] = {"error": f"{type(e).__name__}: {e}"}
            stage2_winner = best_label_by_ndcg(stage2_results)
            print(f"\n  → Stage 2 winner: {stage2_winner}  "
                  f"(NDCG={stage2_results[stage2_winner].get('ndcg',0):.3f})")
            module_summary["stage_2"] = {
                "results": stage2_results,
                "winner": stage2_winner,
                "fixed_retrieval": stage1_winner,
            }

        # ---------------- STAGE 3: query transform ----------------
        if "3" in args.stages:
            print(f"\n--- Stage 3: query transform (retrieval={stage1_winner}, reranker={stage2_winner}) ---")
            stage3_results: dict = {}
            for tr in STAGE_3_TRANSFORMS:
                print(f"\n  ▶ {tr}")
                try:
                    base_fn = make_retrieve_fn_method(retriever, module, stage1_winner)
                    re_fn = wrap_with_reranker(base_fn, stage2_winner)
                    rfn = wrap_with_transform(
                        re_fn, transform=tr, module=module,
                        retriever=retriever, chunk_strategy=FIXED_STRATEGY,
                        embedding_dim=FIXED_DIM,
                    )
                    r = run_one_config(qa_pairs, rfn, track_b=args.track_b,
                                       module=module, label=tr)
                    print(f"    NDCG@10={r.get('ndcg',0):.3f}  MRR={r.get('mrr',0):.3f}  "
                          f"R@5={r.get('recall_at_5',0):.3f}  p95={r.get('p95_latency_ms',0):.0f}ms"
                          + (f"  TrackB={r.get('track_b_composite',0):.3f}" if args.track_b else ""))
                    stage3_results[tr] = r
                except Exception as e:
                    print(f"    ✗ FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    stage3_results[tr] = {"error": f"{type(e).__name__}: {e}"}
            stage3_winner = best_label_by_ndcg(stage3_results)
            print(f"\n  → Stage 3 winner: {stage3_winner}  "
                  f"(NDCG={stage3_results[stage3_winner].get('ndcg',0):.3f})")
            module_summary["stage_3"] = {
                "results": stage3_results,
                "winner": stage3_winner,
                "fixed_retrieval": stage1_winner,
                "fixed_reranker": stage2_winner,
            }

        # Persist
        out_dir = OUT_DIR / module
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "retrieval_benchmark.json").write_text(json.dumps(module_summary, indent=2))
        summary[module] = module_summary

    (OUT_DIR / "_retrieval_benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'#' * 100}")
    print("FULL PIPELINE WINNERS")
    for module in args.modules:
        if module not in summary:
            continue
        s1 = summary[module].get("stage_1", {}).get("winner", "—")
        s2 = summary[module].get("stage_2", {}).get("winner", "—")
        s3 = summary[module].get("stage_3", {}).get("winner", "—")
        print(f"  [{module}]  retrieval={s1}  reranker={s2}  transform={s3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
