"""Reranker cascade.

Per CLAUDE.md § 6.4, four rerankers (Cohere dropped — paid):

  1. Cross-Encoder  — joint (query, passage) BERT scoring. Fast, strong baseline.
                       Model: cross-encoder/ms-marco-MiniLM-L-6-v2.
  2. ColBERT        — late-interaction MaxSim. More expressive on long passages.
                       Model: colbert-ir/colbertv2.0 via RAGatouille.
  3. MonoT5         — T5 fine-tuned to score "true"/"false" relevance.
                       Strong on domain-specific text. Model: castorini/monot5-base-msmarco.
  4. RankGPT        — Claude prompted to rank N passages by relevance via JSON.
                       Most flexible, most expensive.

Each reranker exposes the same interface:
    rerank(query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]

Cascade orchestrator (`rerank_cascade`) chains: cheap reranker on K=100 →
expensive on K=20 → final K=5. Configurable per experiment. Defaults to a
single-stage cross-encoder (the strongest accuracy-per-millisecond option).

NOTE: ColBERT and MonoT5 model loads are heavy (100s of MB). Lazy-loaded
and cached — first call pays the cost; subsequent calls reuse the model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .llm import claude_json
from .retriever import ScoredChunk


# === Cross-Encoder =========================================================

CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class CrossEncoderReranker:
    """ms-marco-MiniLM-L-6-v2 — fast, ~80MB, strong baseline."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [[query, c.content] for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [replace(c, score=float(s)) for c, s in ranked[:top_n]]


# === MonoT5 ================================================================

MONOT5_MODEL = os.environ.get("MONOT5_MODEL", "castorini/monot5-base-msmarco")


class MonoT5Reranker:
    """T5 trained to emit "true"/"false" tokens for (query, passage) relevance.

    Score = softmax(logit_true) at the first generated position. Strong on
    domain-specific text per the literature (better than cross-encoder for
    passages with rare terminology).

    KNOWN ISSUE (this venv): transformers v5.6 tries to convert castorini's
    SentencePiece tokenizer to tiktoken-fast format and fails. Even with
    use_fast=False + legacy=True + sentencepiece installed, the conversion
    path is hit. Workaround would be downgrading transformers to v4.x; not
    pursued because of risk to sentence-transformers / chunking. Use
    cross_encoder or rankgpt instead in this environment.
    """

    def __init__(self, model_name: str = MONOT5_MODEL):
        # Newer transformers (≥5.x) tries to auto-convert SentencePiece → tiktoken-fast
        # and chokes on castorini/monot5 (2020) which only ships spiece.model.
        # Force the slow tokenizer path via AutoTokenizer(use_fast=False).
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        # Cache token IDs for "true" / "false"
        self.true_id = self.tokenizer("true").input_ids[0]
        self.false_id = self.tokenizer("false").input_ids[0]

    def _score_pair(self, query: str, passage: str) -> float:
        import torch
        prompt = f"Query: {query} Document: {passage} Relevant:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # First generated token's logits
        logits = out.scores[0][0]
        true_logit = float(logits[self.true_id])
        false_logit = float(logits[self.false_id])
        # Softmax over the two-class problem
        m = max(true_logit, false_logit)
        e_true = np.exp(true_logit - m)
        e_false = np.exp(false_logit - m)
        return float(e_true / (e_true + e_false))

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        scored = [(c, self._score_pair(query, c.content)) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [replace(c, score=s) for c, s in scored[:top_n]]


# === ColBERT (RAGatouille) =================================================

class ColBERTReranker:
    """ColBERTv2 late-interaction reranker via RAGatouille.

    Late interaction: each query token gets its own embedding, MaxSim over
    passage tokens, sum across query tokens. Slower than cross-encoder but
    more expressive for long passages.

    KNOWN ISSUE (this venv): RAGatouille's HF_ColBERT class accesses
    `_tied_weights_keys`, an attribute renamed to `all_tied_weights_keys`
    in transformers v5. Same root cause as MonoT5 — would require
    transformers v4 to use here. Use cross_encoder or rankgpt instead.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        # Lazy import — RAGatouille pulls in heavy ColBERT deps
        from ragatouille import RAGPretrainedModel
        self.model_name = model_name
        self.model = RAGPretrainedModel.from_pretrained(model_name)

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        passages = [c.content for c in chunks]
        results = self.model.rerank(query=query, documents=passages, k=min(top_n, len(passages)))
        # RAGatouille returns [{result_index, score, content}, ...]
        out: list[ScoredChunk] = []
        for r in results:
            idx = r["result_index"]
            score = float(r["score"])
            out.append(replace(chunks[idx], score=score))
        return out


# === RankGPT (Claude) ======================================================

_RANKGPT_PROMPT = """You are an expert at ranking passages by relevance to a query.

Query: {query}

Below are {n} numbered passages. Rank them from MOST to LEAST relevant for answering the query. Use document content, specificity, and direct relevance — not surface-level keyword overlap alone.

{passages}

Return ONLY a JSON array of passage numbers in ranked order (most relevant first). Example: [3, 1, 7, 2, ...]. Include all {n} numbers exactly once."""


class RankGPTReranker:
    """LLM-based reranking. Most expressive (handles complex multi-part queries),
    most expensive."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        n = len(chunks)
        passages_text = "\n\n".join(
            f"[{i+1}] {(c.content or '').strip()[:1000]}"
            for i, c in enumerate(chunks)
        )
        try:
            order = claude_json(
                _RANKGPT_PROMPT.format(query=query, n=n, passages=passages_text),
                max_tokens=512,
            )
        except Exception:
            return chunks[:top_n]                                # fallback: original order
        if isinstance(order, dict):
            order = order.get("ranking", []) or list(order.values())[0]
        # Validate + dedupe
        seen: set[int] = set()
        ranked: list[ScoredChunk] = []
        for item in order:
            try:
                idx = int(item) - 1
            except (ValueError, TypeError):
                continue
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                # Score = inverse rank in the LLM's ordering
                ranked.append(replace(chunks[idx], score=1.0 / (len(ranked) + 1)))
            if len(ranked) >= top_n:
                break
        # Backfill any missing chunks (LLM returned fewer than n)
        if len(ranked) < top_n:
            for i, c in enumerate(chunks):
                if i not in seen:
                    ranked.append(replace(c, score=1.0 / (len(ranked) + 1)))
                    if len(ranked) >= top_n:
                        break
        return ranked


# === Cascade orchestrator =================================================

# Cached singletons — first use triggers the model load
_MODEL_CACHE: dict[str, object] = {}


def get_reranker(name: str):
    """Lazy-instantiate and cache rerankers by name."""
    name = name.lower()
    if name not in _MODEL_CACHE:
        if name == "cross_encoder":
            _MODEL_CACHE[name] = CrossEncoderReranker()
        elif name == "monot5":
            _MODEL_CACHE[name] = MonoT5Reranker()
        elif name == "colbert":
            _MODEL_CACHE[name] = ColBERTReranker()
        elif name == "rankgpt":
            _MODEL_CACHE[name] = RankGPTReranker()
        else:
            raise ValueError(f"unknown reranker: {name}")
    return _MODEL_CACHE[name]


def rerank(
    query: str,
    chunks: list[ScoredChunk],
    *,
    name: str,
    top_n: int,
) -> list[ScoredChunk]:
    """Single-pass rerank. `name` ∈ {none, cross_encoder, monot5, colbert, rankgpt}."""
    if name.lower() in ("none", ""):
        return chunks[:top_n]
    return get_reranker(name).rerank(query, chunks, top_n)


def rerank_cascade(
    query: str,
    chunks: list[ScoredChunk],
    *,
    stages: list[tuple[str, int]],
) -> list[ScoredChunk]:
    """Apply rerankers in sequence, each narrowing the candidate pool.

    Example: stages=[("cross_encoder", 20), ("rankgpt", 5)] → first stage takes
    ~100 candidates down to 20 cheaply; second stage uses Claude on those 20
    to pick the best 5. Final output has 5 chunks.
    """
    current = chunks
    for name, top_n in stages:
        current = rerank(query, current, name=name, top_n=top_n)
    return current
