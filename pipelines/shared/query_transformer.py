"""Query transformations — pre-retrieval rewrites that improve recall.

All four CLAUDE.md transformations:

  1. HyDE     — generate a hypothetical answer, retrieve against the answer's
                embedding (closes the query/passage style gap).
  2. Multi-Query — generate N reformulations stressing different aspects;
                retrieve for each and union.
  3. PRF      — Pseudo-Relevance Feedback. Run an initial retrieval; use the
                top-k results' content to extract expansion terms; rerun with
                the expanded query.
  4. Step-Back — abstract the question to a higher-level principle, retrieve
                that, use as context for the original specific question.

Each transformer exposes a `transform(query, ...)` method whose return type
is `TransformResult` — either a single rewritten query (HyDE, PRF, Step-Back)
or a list of queries (Multi-Query). The caller decides what to do with them
(embed-and-search the rewrite, fan out, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm import claude_json, claude_text
from .retriever import HybridRetriever, ScoredChunk


@dataclass
class TransformResult:
    """Output of a query transformer.

    Most transformers produce one or more rewritten query strings.
    Step-Back also produces a "background" query — store that separately.
    """
    queries: list[str]                                       # >= 1
    transform_name: str
    extras: dict = field(default_factory=dict)               # method-specific


# --- HyDE ------------------------------------------------------------------

_HYDE_SYSTEM = {
    "compliance": "You are a senior compliance officer at a major bank.",
    "credit": "You are a senior credit analyst at a major investment bank.",
}

_HYDE_PROMPT = """Write a detailed, factual passage that would directly answer this question.
The passage should read like an excerpt from a {doc_kind}, not a summary or generic answer.
Use specific terminology, regulatory clause references, or financial figures where appropriate.

Question: {query}

Write only the passage. No preamble, no formatting."""


def hyde(query: str, *, module: str) -> TransformResult:
    """Hypothetical Document Embeddings.

    Generate what an answering passage would look like, then embed and search
    against THAT instead of the original short query. Closes the stylistic
    gap between informal queries and formal regulatory/financial prose.
    """
    doc_kind = "regulatory guideline" if module == "compliance" else "10-K filing or analyst report"
    answer = claude_text(
        _HYDE_PROMPT.format(query=query, doc_kind=doc_kind),
        system=_HYDE_SYSTEM[module],
        max_tokens=400,
    )
    return TransformResult(
        queries=[answer],
        transform_name="hyde",
        extras={"original_query": query, "doc_kind": doc_kind},
    )


# --- Multi-Query -----------------------------------------------------------

_MULTI_QUERY_PROMPT = """Generate {n} alternative phrasings of this question, each emphasizing a different aspect or using different vocabulary that might appear in {doc_kind}. Keep each variant under 25 words.

Question: {query}

Return ONLY a JSON array of strings: ["variant 1", "variant 2", ...]"""


def multi_query(query: str, *, module: str, n: int = 4) -> TransformResult:
    """Generate N semantically diverse reformulations; caller fans out."""
    doc_kind = "regulatory documents" if module == "compliance" else "financial filings"
    raw = claude_json(
        _MULTI_QUERY_PROMPT.format(query=query, n=n, doc_kind=doc_kind),
        system=_HYDE_SYSTEM[module],
        max_tokens=400,
    )
    if isinstance(raw, dict) and "queries" in raw:
        variants = raw["queries"]
    elif isinstance(raw, list):
        variants = raw
    else:
        variants = [str(raw)]
    variants = [str(v).strip() for v in variants if str(v).strip()][:n]
    if query not in variants:
        variants = [query] + variants                           # always include the original
    return TransformResult(
        queries=variants,
        transform_name="multi_query",
        extras={"original_query": query, "n_requested": n},
    )


# --- Pseudo-Relevance Feedback (PRF) --------------------------------------

_PRF_PROMPT = """Given the user's question and the top retrieved passages below, identify 5-8 KEY DOMAIN TERMS from the passages that should be added to the query to improve retrieval recall. Prefer terms that appear in the passages but NOT in the original query.

Question: {query}

Top passages:
{passages}

Return a single expanded query string that combines the original question with the additional terms. Keep it under 50 words. No preamble — return only the expanded query."""


def prf(
    query: str,
    *,
    retriever: HybridRetriever,
    module: str,
    chunk_strategy: str,
    embedding_dim: int = 512,
    n_initial: int = 5,
) -> TransformResult:
    """Pseudo-Relevance Feedback: 1st-pass retrieve → extract expansion terms → 2nd-pass query.

    Classic IR technique. Effective on domain-specific corpora because the
    top passages contain the exact terminology the indexer used.
    """
    initial = retriever.search(
        query=query, module=module, chunk_strategy=chunk_strategy,
        mode="hybrid", embedding_dim=embedding_dim, top_k=n_initial,
    )
    passages_text = "\n\n".join(
        f"[{i+1}] {(c.content or '').strip()[:600]}"
        for i, c in enumerate(initial)
    )
    expanded = claude_text(
        _PRF_PROMPT.format(query=query, passages=passages_text),
        system=_HYDE_SYSTEM[module],
        max_tokens=200,
    )
    return TransformResult(
        queries=[expanded.strip()],
        transform_name="prf",
        extras={"original_query": query, "n_initial": len(initial)},
    )


# --- Step-Back -------------------------------------------------------------

_STEP_BACK_PROMPT = """The following is a specific question. Generate a more abstract, principle-level question whose answer would provide useful BACKGROUND context for the specific one. The abstract question should ask about the underlying purpose, framework, or concept — not the specific details.

Specific question: {query}

Return ONLY the abstract question, no preamble or explanation."""


def step_back(query: str, *, module: str) -> TransformResult:
    """Step-Back Prompting: derive a more abstract version of the question.

    The CALLER is expected to retrieve for BOTH the specific and the abstract
    queries, then provide both sets of context to the generator. This works
    because specific regulatory/financial questions often need the governing
    principle to be answered correctly.
    """
    abstract = claude_text(
        _STEP_BACK_PROMPT.format(query=query),
        system=_HYDE_SYSTEM[module],
        max_tokens=120,
    )
    return TransformResult(
        queries=[query, abstract.strip()],                       # both, in original order
        transform_name="step_back",
        extras={"original_query": query, "abstract_query": abstract.strip()},
    )


# --- Top-level dispatcher --------------------------------------------------

def apply_transform(
    transform: str,
    query: str,
    *,
    module: str,
    retriever: Optional[HybridRetriever] = None,
    chunk_strategy: Optional[str] = None,
    embedding_dim: int = 512,
    n_multi_query: int = 4,
) -> TransformResult:
    """Run any of the four transforms by name. Returns TransformResult.

    `transform="none"` returns a passthrough (the original query). PRF requires
    `retriever` and `chunk_strategy` to be passed in for its first-pass retrieval.
    """
    t = transform.lower()
    if t in ("none", ""):
        return TransformResult(queries=[query], transform_name="none")
    if t == "hyde":
        return hyde(query, module=module)
    if t == "multi_query":
        return multi_query(query, module=module, n=n_multi_query)
    if t == "prf":
        if retriever is None or chunk_strategy is None:
            raise ValueError("PRF requires retriever + chunk_strategy")
        return prf(query, retriever=retriever, module=module,
                   chunk_strategy=chunk_strategy, embedding_dim=embedding_dim)
    if t == "step_back":
        return step_back(query, module=module)
    raise ValueError(f"unknown transform: {transform}")
