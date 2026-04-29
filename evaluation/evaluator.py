"""Dual-track RAG evaluator (per CLAUDE.md § 7.2).

  Track A — retrieval quality
    Per query: binary relevance for each retrieved chunk = char-level overlap
    with the source passage's (char_start, char_end) range > 0.3.
    Aggregated metrics: NDCG@10, MRR, MAP, Recall@{1,3,5,10}, latency p50/p95/p99.
    THIS IS WHAT MAKES THE EVAL CHUNK-AGNOSTIC — fair across all 6 strategies.

  Track B — answer quality
    Per query: RAG-generated answer vs Claude's reference answer (read directly
    from the raw passage). Three complementary measures:
      - Semantic similarity (cosine on all-MiniLM-L6-v2)
      - BERTScore F1 (token-level semantic overlap; distilbert-base-uncased)
      - Key concept coverage (% of `key_concepts` present in the RAG answer)
    Composite = mean of the three.

The evaluator is RETRIEVAL-AGNOSTIC: it takes a callable `retrieve_fn(query)
→ list[ScoredChunk]` and a `generate_fn(query, top) → str`. Benchmark drivers
(chunking, dimension sweep, retrieval) construct those callables for their
specific configurations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from pipelines.shared.retriever import ScoredChunk


# === Track A — retrieval metrics ===========================================

def overlap_relevance(qa: dict, chunks: list[ScoredChunk], *, threshold: float = 0.3) -> list[int]:
    """Binary relevance per retrieved chunk via char-overlap with source passage.

    A chunk is relevant iff:
      same source_doc_id  AND  |overlap| / |source_passage_range| > threshold
    """
    src_doc = qa["source_doc_id"]
    src_start = qa["char_start"]
    src_end = qa["char_end"]
    src_len = max(src_end - src_start, 1)
    out: list[int] = []
    for c in chunks:
        if c.payload.get("doc_id") != src_doc:
            out.append(0)
            continue
        cs = int(c.payload.get("char_start", 0))
        ce = int(c.payload.get("char_end", 0))
        ov_start = max(src_start, cs)
        ov_end = min(src_end, ce)
        ov = max(0, ov_end - ov_start)
        out.append(1 if (ov / src_len) > threshold else 0)
    return out


def ndcg_at_k(relevance: list[int], k: int) -> float:
    if not relevance:
        return 0.0
    rel_k = relevance[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel_k))
    # IDCG: ideal sorted (relevant first), bounded by sum(rel_k)
    n_rel = sum(rel_k)
    idcg = sum(1 / np.log2(i + 2) for i in range(n_rel))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr(relevance: list[int]) -> float:
    for rank, r in enumerate(relevance, start=1):
        if r:
            return float(1.0 / rank)
    return 0.0


def recall_at_k(relevance: list[int], k: int, *, n_relevant_total: int = 1) -> float:
    """For our setup we only know relevant chunks via overlap with the single
    source passage — assume `n_relevant_total = 1` (passage exists ⇒ at least
    one chunk overlaps it). Returns 1 if any of top-k is relevant."""
    return float(any(relevance[:k])) if n_relevant_total else 0.0


def average_precision(relevance: list[int]) -> float:
    """AP at the position of each relevant doc, averaged. With binary relevance
    and ≥1 relevant target this collapses to MRR-like behavior."""
    if not relevance:
        return 0.0
    hits = 0
    s = 0.0
    for rank, r in enumerate(relevance, start=1):
        if r:
            hits += 1
            s += hits / rank
    return float(s / hits) if hits > 0 else 0.0


@dataclass
class TrackARecord:
    qa_id: str
    ndcg: float
    mrr: float
    map_: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    latency_ms: float


def evaluate_track_a(
    qa_pairs: list[dict],
    retrieve_fn: Callable[[str], list[ScoredChunk]],
    *,
    top_k: int = 10,
) -> tuple[dict, list[TrackARecord]]:
    """For each Track A pair, retrieve, score, aggregate."""
    per: list[TrackARecord] = []
    for qa in qa_pairs:
        if qa["track"] != "A":
            continue
        t0 = time.perf_counter()
        chunks = retrieve_fn(qa["question"])
        latency = (time.perf_counter() - t0) * 1000
        chunks = chunks[:top_k]
        relevance = overlap_relevance(qa, chunks)
        per.append(TrackARecord(
            qa_id=qa["qa_id"],
            ndcg=ndcg_at_k(relevance, k=10),
            mrr=mrr(relevance),
            map_=average_precision(relevance),
            recall_at_1=recall_at_k(relevance, 1),
            recall_at_3=recall_at_k(relevance, 3),
            recall_at_5=recall_at_k(relevance, 5),
            recall_at_10=recall_at_k(relevance, 10),
            latency_ms=latency,
        ))

    if not per:
        return {}, []

    lat = np.array([r.latency_ms for r in per])
    agg = {
        "n_queries": len(per),
        "ndcg":         float(np.mean([r.ndcg for r in per])),
        "mrr":          float(np.mean([r.mrr for r in per])),
        "map":          float(np.mean([r.map_ for r in per])),
        "recall_at_1":  float(np.mean([r.recall_at_1 for r in per])),
        "recall_at_3":  float(np.mean([r.recall_at_3 for r in per])),
        "recall_at_5":  float(np.mean([r.recall_at_5 for r in per])),
        "recall_at_10": float(np.mean([r.recall_at_10 for r in per])),
        "avg_latency_ms": float(lat.mean()),
        "p50_latency_ms": float(np.percentile(lat, 50)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
    }
    return agg, per


# === Track B — answer quality metrics ======================================

# Lazy-loaded models (heavy)
_SIM_MODEL = None


def _get_sim_model():
    global _SIM_MODEL
    if _SIM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SIM_MODEL


@dataclass
class TrackBRecord:
    qa_id: str
    rag_answer: str
    semantic_similarity: float
    bertscore_f1: float
    key_concept_coverage: Optional[float]
    composite: float
    latency_ms: float


def _semantic_sim(answer: str, reference: str) -> float:
    if not answer.strip() or not reference.strip():
        return 0.0
    model = _get_sim_model()
    emb = model.encode([answer, reference], convert_to_numpy=True, normalize_embeddings=True)
    return float(emb[0] @ emb[1])


def _bertscore_f1(answer: str, reference: str) -> float:
    if not answer.strip() or not reference.strip():
        return 0.0
    from bert_score import score as bs
    _, _, F1 = bs([answer], [reference], lang="en",
                  model_type="distilbert-base-uncased", verbose=False)
    return float(F1[0])


def _concept_coverage(answer: str, key_concepts: list[str]) -> Optional[float]:
    if not key_concepts:
        return None
    a = answer.lower()
    hit = sum(1 for kc in key_concepts if kc.lower() in a)
    return hit / len(key_concepts)


def evaluate_track_b(
    qa_pairs: list[dict],
    retrieve_fn: Callable[[str], list[ScoredChunk]],
    generate_fn: Callable[[str, list[ScoredChunk]], str],
    *,
    top_k_for_gen: int = 5,
) -> tuple[dict, list[TrackBRecord]]:
    """Run RAG on each Track B question, score generated answer vs reference."""
    # Pre-batch BERTScore for efficiency
    rag_answers: list[str] = []
    references: list[str] = []
    qa_meta: list[dict] = []
    latencies: list[float] = []
    for qa in qa_pairs:
        if qa["track"] != "B":
            continue
        t0 = time.perf_counter()
        chunks = retrieve_fn(qa["question"])[:top_k_for_gen]
        rag = generate_fn(qa["question"], chunks)
        latencies.append((time.perf_counter() - t0) * 1000)
        rag_answers.append(rag)
        references.append(qa.get("reference_answer", ""))
        qa_meta.append(qa)

    if not rag_answers:
        return {}, []

    # Batch BERTScore (much faster than per-pair)
    from bert_score import score as bs
    _, _, F1 = bs(rag_answers, references, lang="en",
                  model_type="distilbert-base-uncased", verbose=False)
    bert_f1 = [float(x) for x in F1]

    # Batch semantic sim
    sim_model = _get_sim_model()
    a_emb = sim_model.encode(rag_answers, convert_to_numpy=True, normalize_embeddings=True)
    r_emb = sim_model.encode(references, convert_to_numpy=True, normalize_embeddings=True)
    sims = (a_emb * r_emb).sum(axis=1).tolist()

    per: list[TrackBRecord] = []
    for i, qa in enumerate(qa_meta):
        sim = float(sims[i])
        f1 = bert_f1[i]
        cc = _concept_coverage(rag_answers[i], qa.get("key_concepts", []))
        composite = float(np.mean([sim, f1, cc if cc is not None else sim]))
        per.append(TrackBRecord(
            qa_id=qa["qa_id"],
            rag_answer=rag_answers[i],
            semantic_similarity=sim,
            bertscore_f1=f1,
            key_concept_coverage=cc,
            composite=composite,
            latency_ms=latencies[i],
        ))

    cc_vals = [r.key_concept_coverage for r in per if r.key_concept_coverage is not None]
    agg = {
        "n_queries": len(per),
        "track_b_semantic_sim":      float(np.mean([r.semantic_similarity for r in per])),
        "track_b_bertscore_f1":      float(np.mean([r.bertscore_f1 for r in per])),
        "track_b_concept_coverage":  float(np.mean(cc_vals)) if cc_vals else None,
        "track_b_composite":         float(np.mean([r.composite for r in per])),
        "track_b_avg_latency_ms":    float(np.mean(latencies)),
        "track_b_p95_latency_ms":    float(np.percentile(latencies, 95)),
    }
    return agg, per
