"""Query execution for the Gradio app.

Wires HybridRetriever + (optional) reranker + (optional) generator into one
function that returns enough material for the UI to display:
  - top-k chunks with source attribution
  - generated answer (if requested)
  - per-stage latencies for the performance breakdown
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from pipelines.shared.fusion import convex_combination, hierarchical, rrf
from pipelines.shared.guardrails import GuardrailReport, check as run_guardrails
from pipelines.shared.llm import claude_text
from pipelines.shared.query_logger import chunk_for_log, log_query
from pipelines.shared.query_transformer import apply_transform
from pipelines.shared.reranker import rerank
from pipelines.shared.retriever import HybridRetriever, ScoredChunk


_GENERATE_PROMPT = """You are a senior {role}. Answer the user's question using ONLY the passages below. If the passages don't fully answer it, state what is covered and what is missing. Cite passages by their bracketed number when stating a fact.

Question: {query}

Passages:
{passages}

Answer (3-5 sentences, no preamble):"""


@dataclass
class QueryResult:
    answer: Optional[str]
    chunks: list[ScoredChunk]
    timings: dict[str, float]
    transformed_queries: list[str]
    config_summary: str
    guardrail_report: Optional[GuardrailReport] = None
    query_id: Optional[str] = None


_RETRIEVER: Optional[HybridRetriever] = None


def _retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


def run_query(
    *,
    query: str,
    module: str,
    chunk_strategy: str,
    embedding_dim: int,
    retrieval_method: str,        # "dense" | "bm25" | "splade" | "hybrid_rrf" | "hybrid_convex" | "hybrid_hier"
    reranker: str,                # "none" | "cross_encoder" | "monot5" | "colbert" | "rankgpt"
    query_transform: str,         # "none" | "hyde" | "multi_query" | "prf" | "step_back"
    top_k: int = 10,
    final_k: int = 5,
    generate_answer: bool = False,
) -> QueryResult:
    if not query.strip():
        return QueryResult(answer=None, chunks=[], timings={}, transformed_queries=[],
                           config_summary="(empty query)")

    timings: dict[str, float] = {}
    retr = _retriever()

    # 1) Query transform
    t0 = time.perf_counter()
    try:
        tr = apply_transform(
            query_transform, query, module=module,
            retriever=retr, chunk_strategy=chunk_strategy,
            embedding_dim=embedding_dim,
        )
    except Exception as e:
        # If the transform fails (e.g. no API key), fall back to passthrough
        from pipelines.shared.query_transformer import TransformResult
        tr = TransformResult(queries=[query], transform_name="none-fallback",
                             extras={"error": str(e)})
    timings["transform_ms"] = (time.perf_counter() - t0) * 1000

    # 2) Retrieval (per query then union if multiple)
    def _retrieve_one(q: str) -> list[ScoredChunk]:
        if retrieval_method == "dense":
            return retr.search(query=q, module=module, chunk_strategy=chunk_strategy,
                               mode="dense", embedding_dim=embedding_dim, top_k=top_k)
        if retrieval_method in ("bm25", "splade"):
            return retr.search(query=q, module=module, chunk_strategy=chunk_strategy,
                               mode="sparse", sparse_name=retrieval_method, top_k=top_k)
        if retrieval_method == "hybrid_rrf":
            return retr.search(query=q, module=module, chunk_strategy=chunk_strategy,
                               mode="hybrid", embedding_dim=embedding_dim, top_k=top_k)
        if retrieval_method == "hybrid_convex":
            d, s, _ = retr.search_separate_channels(
                query=q, module=module, chunk_strategy=chunk_strategy,
                embedding_dim=embedding_dim, top_k=50,
            )
            return convex_combination(d, s, alpha=0.7, top_k=top_k)
        if retrieval_method == "hybrid_hier":
            d, s, _ = retr.search_separate_channels(
                query=q, module=module, chunk_strategy=chunk_strategy,
                embedding_dim=embedding_dim, top_k=50,
            )
            return hierarchical(q, d, s, top_k=top_k)
        raise ValueError(f"unknown retrieval_method: {retrieval_method}")

    t0 = time.perf_counter()
    if len(tr.queries) == 1:
        retrieved = _retrieve_one(tr.queries[0])
    else:
        retrieved = rrf([_retrieve_one(q) for q in tr.queries], top_k=top_k)
    timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

    # 3) Optional rerank
    t0 = time.perf_counter()
    if reranker != "none":
        try:
            top = rerank(query, retrieved, name=reranker, top_n=final_k)
        except Exception as e:
            # Reranker model load can fail; fall back to retrieval order
            top = retrieved[:final_k]
            timings["reranker_error"] = str(e)
    else:
        top = retrieved[:final_k]
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # 4) Optional generate
    answer = None
    if generate_answer:
        t0 = time.perf_counter()
        role = "compliance officer" if module == "compliance" else "credit analyst"
        passages = "\n\n".join(
            f"[{i+1}] (doc: {c.payload.get('doc_id','?')}, section: {c.payload.get('section_title','')})\n{c.content[:1500]}"
            for i, c in enumerate(top)
        )
        try:
            answer = claude_text(
                _GENERATE_PROMPT.format(role=role, query=query, passages=passages),
                max_tokens=400,
            )
        except Exception as e:
            answer = f"_(generation failed: {type(e).__name__}: {e})_"
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

    # 5) Guardrails (rule-based, no LLM cost)
    t0 = time.perf_counter()
    guardrail_report = run_guardrails(module, answer, top, query)
    timings["guardrails_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = sum(v for k, v in timings.items() if k.endswith("_ms"))

    config_summary = (
        f"module={module}  strategy={chunk_strategy}  dim={embedding_dim}  "
        f"retrieval={retrieval_method}  reranker={reranker}  transform={query_transform}  "
        f"top_k={top_k}  final_k={final_k}  generate={generate_answer}"
    )

    config_dict = {
        "module": module,
        "chunk_strategy": chunk_strategy,
        "embedding_dim": embedding_dim,
        "retrieval_method": retrieval_method,
        "reranker": reranker,
        "query_transform": query_transform,
        "top_k": top_k,
        "final_k": final_k,
        "generate_answer": generate_answer,
    }

    # 6) Persist to query log (Phase 10 observability)
    qid = log_query(
        query=query,
        config=config_dict,
        timings=timings,
        transformed_queries=tr.queries if query_transform != "none" else [],
        top_chunks=[chunk_for_log(c) for c in top],
        answer=answer,
        guardrail_report=guardrail_report.to_dict(),
    )

    return QueryResult(
        answer=answer,
        chunks=top,
        timings=timings,
        transformed_queries=tr.queries if query_transform != "none" else [],
        config_summary=config_summary,
        guardrail_report=guardrail_report,
        query_id=qid,
    )
