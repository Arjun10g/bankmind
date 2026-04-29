"""Query execution for the Gradio app.

Wires HybridRetriever + (optional) reranker + (optional) generator into one
function that returns enough material for the UI to display:
  - top-k chunks with source attribution
  - generated answer (if requested)
  - per-stage latencies for the performance breakdown

Also supports multi-turn chat. When `chat_history` is non-empty, the user's
latest message is rewritten to a standalone retrieval query (an LLM call,
about 100 tokens) so follow-ups like "tell me more about that" actually find
relevant passages. Generation then receives the full conversation history.
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


# === Generation prompt: long-form, structured, citation-driven ==============

_SYSTEM_PROMPT = """You are a senior {role} answering questions for a colleague.

Voice and style:
- Speak in the professional but direct register a senior banker uses with peers.
- Do NOT begin with throat-clearing phrases ("Great question", "Based on the passages", "I'd be happy to help"). Start with the substantive answer.
- Multi-paragraph answers are encouraged for any non-trivial question. Use bullet lists for enumerable facts.
- Cite supporting passages inline using their bracketed number, e.g. "Tier 1 capital must be at least 6% of risk-weighted assets [2]".
- Quote specific clause text, dollar figures, dates, or section numbers verbatim from the passages whenever possible.
- If the passages partially answer the question, give what they support, then state explicitly what is NOT covered. Do not bluff.

Conversation behaviour:
- Treat each turn as part of an ongoing conversation. Refer back to prior turns when natural.
- If the user's follow-up is ambiguous, ask one short clarifying question, then answer the most likely interpretation.
- Stay in character as a {role}. Don't break the fourth wall about being an AI."""


_USER_TURN_TEMPLATE = """Conversation so far:
{history}

User's latest question: {query}

Retrieved passages:
{passages}

Write your answer. Be substantive, cite passages by [n], and structure with bullets/headers if appropriate."""


_NO_HISTORY_TEMPLATE = """Question: {query}

Retrieved passages:
{passages}

Write your answer. Be substantive, cite passages by [n], and structure with bullets/headers if appropriate."""


# === Follow-up rewriter prompt ===============================================

_REWRITE_SYSTEM = """You rewrite ambiguous follow-up questions in a chat conversation into standalone, fully-self-contained queries that an information retrieval system can answer without seeing prior turns.

Rules:
- Resolve every pronoun, reference, and ellipsis ("that", "those", "it", "the second one", "what about credit?") using the conversation history.
- Inline the topic, entities, and constraints from prior turns.
- Stay close to the user's original wording where possible. Do not invent details.
- Output a single declarative search query. No quotes, no preamble, no explanation."""


_REWRITE_USER = """Conversation so far:
{history}

The user's latest message: {query}

Rewrite that latest message into a standalone retrieval query that captures the full intent."""


# === Data shapes =============================================================

@dataclass
class QueryResult:
    answer: Optional[str]
    chunks: list[ScoredChunk]
    timings: dict[str, float]
    transformed_queries: list[str]
    config_summary: str
    guardrail_report: Optional[GuardrailReport] = None
    query_id: Optional[str] = None
    rewritten_query: Optional[str] = None  # populated when chat history is non-empty


_RETRIEVER: Optional[HybridRetriever] = None


def _retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


def _format_history(history: list[tuple[str, str]], *, max_turns: int = 6) -> str:
    """Format the chat history as a transcript for the prompt.
    `history` is a list of (user_message, assistant_message) tuples."""
    if not history:
        return "(none)"
    recent = history[-max_turns:]
    lines = []
    for u, a in recent:
        if u:
            lines.append(f"USER: {u}")
        if a:
            lines.append(f"ASSISTANT: {a}")
    return "\n".join(lines)


def _rewrite_followup(query: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a possibly-ambiguous follow-up into a standalone query.

    No-op when history is empty. On any LLM error, returns the original query.
    """
    if not history:
        return query
    try:
        rewritten = claude_text(
            _REWRITE_USER.format(history=_format_history(history), query=query),
            system=_REWRITE_SYSTEM,
            max_tokens=200,
            temperature=0.0,
        )
        # Defensive: strip surrounding quotes / leading/trailing whitespace
        rewritten = rewritten.strip().strip('"').strip("'").strip()
        return rewritten or query
    except Exception:
        return query


def run_query(
    *,
    query: str,
    module: str,
    chunk_strategy: str,
    embedding_dim: int,
    retrieval_method: str,
    reranker: str,
    query_transform: str,
    top_k: int = 10,
    final_k: int = 5,
    generate_answer: bool = True,
    chat_history: Optional[list[tuple[str, str]]] = None,
    max_answer_tokens: int = 2000,
) -> QueryResult:
    if not query.strip():
        return QueryResult(answer=None, chunks=[], timings={}, transformed_queries=[],
                           config_summary="(empty query)")

    timings: dict[str, float] = {}
    retr = _retriever()
    history = chat_history or []

    # 0) Follow-up rewrite (only when there's prior history). One LLM call.
    rewritten_query = None
    retrieval_query = query
    if history:
        t0 = time.perf_counter()
        rewritten_query = _rewrite_followup(query, history)
        retrieval_query = rewritten_query
        timings["rewrite_ms"] = (time.perf_counter() - t0) * 1000

    # 1) Optional pre-retrieval transform (HyDE / multi-query / PRF / step-back)
    t0 = time.perf_counter()
    try:
        tr = apply_transform(
            query_transform, retrieval_query, module=module,
            retriever=retr, chunk_strategy=chunk_strategy,
            embedding_dim=embedding_dim,
        )
    except Exception as e:
        from pipelines.shared.query_transformer import TransformResult
        tr = TransformResult(queries=[retrieval_query], transform_name="none-fallback",
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
            top = rerank(retrieval_query, retrieved, name=reranker, top_n=final_k)
        except Exception as e:
            top = retrieved[:final_k]
            timings["reranker_error"] = str(e)
    else:
        top = retrieved[:final_k]
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # 4) Generation (now richer + history-aware)
    answer = None
    if generate_answer:
        t0 = time.perf_counter()
        role = "compliance officer" if module == "compliance" else "credit analyst"
        passages = "\n\n".join(
            f"[{i+1}] (doc: {c.payload.get('doc_id','?')}, section: {c.payload.get('section_title','')})\n{c.content[:1800]}"
            for i, c in enumerate(top)
        )
        if history:
            user_prompt = _USER_TURN_TEMPLATE.format(
                history=_format_history(history), query=query, passages=passages,
            )
        else:
            user_prompt = _NO_HISTORY_TEMPLATE.format(query=query, passages=passages)
        try:
            answer = claude_text(
                user_prompt,
                system=_SYSTEM_PROMPT.format(role=role),
                max_tokens=max_answer_tokens,
                temperature=0.0,
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
        f"top_k={top_k}  final_k={final_k}  generate={generate_answer}  "
        f"chat_turns={len(history)}"
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
        "chat_turns": len(history),
        "rewritten_query": rewritten_query,
    }

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
        rewritten_query=rewritten_query,
    )
