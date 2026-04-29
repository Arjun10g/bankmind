"""End-to-end smoke test for the Phase 6 retrieval architecture.

For each test query, run the full pipeline:
   1. Query transform (one of: none / hyde / multi_query / prf / step_back)
   2. Hybrid retrieval (Qdrant: dense + SPLADE + BM25 with RRF fusion)
   3. Rerank cascade (cross-encoder → top 5)
   4. Generate an answer with Claude using the top-5 as context

Skips heavyweight rerankers (ColBERT / MonoT5 / RankGPT) — those are
exercised individually in Phase 7 eval. This smoke test is just to prove
the plumbing works end-to-end with one cheap cascade.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.shared.llm import claude_text
from pipelines.shared.query_transformer import apply_transform
from pipelines.shared.reranker import rerank
from pipelines.shared.retriever import HybridRetriever, ScoredChunk


# (module, chunk_strategy, query, transform)
TEST_CASES = [
    ("compliance", "regulatory_boundary",
     "What does OSFI Guideline B-20 require for residential mortgage underwriting?",
     "none"),
    ("compliance", "semantic",
     "How is a politically exposed person defined under Canadian AML rules?",
     "hyde"),
    ("credit", "narrative_section",
     "What are Goldman Sachs' key risk factors disclosed in the 10-K?",
     "step_back"),
]


_GENERATE_PROMPT = """You are a senior {role}. Answer the user's question using ONLY the passages below. Cite passages by their bracketed number when stating a fact. If the passages don't fully answer the question, say what is covered and what is missing.

Question: {query}

Passages:
{passages}

Answer:"""


def generate_answer(query: str, top: list[ScoredChunk], module: str) -> str:
    role = "compliance officer" if module == "compliance" else "credit analyst"
    passages = "\n\n".join(
        f"[{i+1}] (doc: {c.payload.get('doc_id','?')}, section: {c.payload.get('section_title','')})\n{c.content[:1500]}"
        for i, c in enumerate(top)
    )
    return claude_text(
        _GENERATE_PROMPT.format(role=role, query=query, passages=passages),
        max_tokens=600,
    )


def main() -> int:
    retriever = HybridRetriever()

    print(f"\n{'=' * 100}")
    print(f"Phase 6 retrieval smoke test")
    print(f"{'=' * 100}")

    for module, strategy, query, transform in TEST_CASES:
        print(f"\n{'─' * 100}")
        print(f"Q: {query}")
        print(f"   module={module}  strategy={strategy}  transform={transform}")

        timings: dict[str, float] = {}

        # 1. Query transform
        t0 = time.perf_counter()
        tr = apply_transform(transform, query, module=module,
                             retriever=retriever, chunk_strategy=strategy)
        timings["transform"] = time.perf_counter() - t0
        if transform != "none":
            print(f"\n   [{tr.transform_name}] produced {len(tr.queries)} queries:")
            for i, q in enumerate(tr.queries):
                preview = q.replace("\n", " ")[:200]
                print(f"     [{i+1}] {preview}{'…' if len(q) > 200 else ''}")

        # 2. Retrieve for each transformed query, union by chunk_id
        t0 = time.perf_counter()
        seen: dict[str, ScoredChunk] = {}
        for q in tr.queries:
            results = retriever.search(
                query=q, module=module, chunk_strategy=strategy,
                mode="hybrid", embedding_dim=512, top_k=20,
            )
            for c in results:
                if c.chunk_id not in seen or c.score > seen[c.chunk_id].score:
                    seen[c.chunk_id] = c
        candidates = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:30]
        timings["retrieve"] = time.perf_counter() - t0
        print(f"\n   retrieved {len(candidates)} candidates ({timings['retrieve']*1000:.0f} ms)")

        # 3. Rerank with cross-encoder → top 5
        t0 = time.perf_counter()
        top = rerank(query, candidates, name="cross_encoder", top_n=5)
        timings["rerank"] = time.perf_counter() - t0
        print(f"   reranked to top 5 ({timings['rerank']*1000:.0f} ms)")

        for i, c in enumerate(top):
            snippet = (c.content or "").replace("\n", " ")[:120]
            print(f"     #{i+1}  score={c.score:.3f}  [{c.payload.get('section_title','')[:50]}]  {snippet}{'…' if len(c.content) > 120 else ''}")

        # 4. Generate
        t0 = time.perf_counter()
        answer = generate_answer(query, top, module)
        timings["generate"] = time.perf_counter() - t0
        print(f"\n   ANSWER ({timings['generate']*1000:.0f} ms):")
        for line in answer.split("\n"):
            print(f"     {line}")

        total_ms = sum(timings.values()) * 1000
        print(f"\n   timings: " + "  ".join(f"{k}={v*1000:.0f}ms" for k, v in timings.items()) +
              f"  TOTAL={total_ms:.0f}ms")

    print(f"\n{'=' * 100}")
    print(f"SMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
