# BankMind

A production-grade multi-domain RAG platform for financial intelligence, built and benchmarked end-to-end. Two specialised pipelines sit on shared infrastructure: a Compliance Assistant covering OSFI, FINTRAC, Basel III/IV, the Bank Act, GDPR, and Federal Reserve Regulation W; and a Credit Analyst Copilot covering five major banks across SEC EDGAR 10-K, 10-Q, 8-K, 40-F, and 6-K filings.

The project ships with a complete dual-track evaluation harness, a PCA-based intrinsic dimensionality study, six chunking strategies tested in a controlled benchmark, a three-stage retrieval ablation (retrieval method, reranker, query transform), and a Gradio dashboard that renders all of it live.

Live demo: see the Hugging Face Spaces link in the repository description (or run locally per "Quick start" below).

## Highlights

* 32,963 chunks indexed in Qdrant with 5 Matryoshka dense embedding dimensions plus SPLADE plus BM25, queryable via native hybrid search.
* Dual-track evaluation pipeline: Track A scores retrieval via character-overlap with raw source passages (chunking-agnostic), Track B scores generation against reference answers written from the raw passages (retrieval-agnostic).
* Six chunking strategies benchmarked head-to-head with controlled variables. Best strategy per module identified empirically.
* Hypothesis-rejecting PCA finding: credit-narrative text has *lower* intrinsic dimensionality than regulatory text, opposite to the original prediction. Validated by a downstream dimension sweep.
* End-to-end retrieval ablation: 6 retrieval methods, 4 rerankers, 4 query transforms. Production winner pipeline identified.
* Rule-based guardrails: citation enforcement, number grounding (credit), staleness warnings, temporal mismatch detection, all wired into the UI.
* Cost-controlled UI: pure retrieval is free; every LLM-using feature is gated by an explicit toggle.

## Headline empirical results

### Chunking benchmark (dim=512, hybrid+RRF, no rerank, no transform, 50 evaluation questions per module)

| Module | Best strategy | NDCG@10 | Recall@5 | Track-B Composite |
|---|---|---:|---:|---:|
| Compliance | semantic | 0.759 | 0.880 | 0.799 |
| Credit | semantic | 0.592 | 0.800 | 0.804 |

Semantic chunking beats domain-aware chunking strategies by 17 to 94 percent relative NDCG. Domain-structured chunkers (OSFI section boundaries, SEC Item boundaries, financial-statement boundaries) all lose to semantic boundary detection.

### Dimension sweep (chunking=semantic, hybrid+RRF)

| Dim | Compliance NDCG | Credit NDCG |
|---:|---:|---:|
| 128 | 0.767 | 0.618 |
| 256 | 0.768 | 0.608 |
| 512 | 0.762 | 0.602 |
| 768 | 0.805 | 0.623 |
| 1024 | 0.813 | 0.616 |

Compliance shows a 6 percent NDCG lift above dim 512. Credit is essentially flat across all dimensions (0.021 spread). The credit corpus tolerates aggressive Matryoshka truncation, validating the prior PCA finding empirically. Production implication: credit can ship at dim 128 (8x storage savings) at no measurable retrieval cost.

### Retrieval ablation (3-stage)

Compliance pipeline winner: chunking=semantic, dim=1024, retrieval=BM25, reranker=RankGPT, transform=step_back. NDCG@10 = 0.834, Recall@5 = 0.920.

Stage-by-stage findings:
1. **BM25 alone wins both modules** as the retrieval method. Dense, SPLADE, and all hybrid variants (RRF, convex, hierarchical) underperform raw lexical BM25. Regulatory and financial corpora are exact-term-rich (regulatory codes, fiscal years, dollar figures), and exact-match retrieval nails them.
2. **RankGPT reranking** lifts NDCG by about 0.034 over BM25 alone, but adds 11 to 16 seconds of p95 latency. Cross-encoder gives a smaller lift (0.012) at much lower cost (about 500 ms p95). MonoT5 and ColBERT both blocked on a transformers v5 compatibility issue (documented in source).
3. **PRF and step_back query transforms** add about 0.023 NDCG over the BM25+RankGPT baseline. **HyDE catastrophically broke compliance** (-0.295 NDCG). Documented mechanism: HyDE generates a hypothetical answer in a different stylistic register, but BM25 is exact-term-based, so the rewritten query's stems no longer match the corpus. Lesson: HyDE only works on top of dense or hybrid retrieval, never on a pure-sparse pipeline.

### PCA eigenstructure analysis (the project's novel contribution)

Both modules' full-rank PCA was fit on aggregated dense_1024 embeddings (14,318 vectors compliance, 18,645 vectors credit).

| Metric | Compliance | Credit |
|---|---:|---:|
| Kneedle elbow | dim 206 | dim 176 |
| 95 percent variance | dim 336 | dim 316 |
| Cumulative variance at dim 256 | 91.3% | 92.6% |

The original hypothesis ("regulatory language is formulaic, so its PCA elbow appears at lower dimension") was rejected. Credit-narrative text has *lower* intrinsic dimensionality than regulatory text. Revised mental model: corpus topical breadth dominates over language formulaicness as the driver of intrinsic dimensionality. Compliance is a union of 6+ unrelated regulatory frameworks across 4 jurisdictions, so it spans more semantic territory; credit is the same SEC 10-K template repeated across 5 banks, so it compresses well.

The downstream dimension sweep confirmed this empirically: compliance NDCG monotonically improves from 0.762 to 0.813 above dim 512; credit's NDCG stays in [0.602, 0.623] across all five dimensions.

## What was built

### Phases (per the project plan)

1. Infrastructure: Qdrant Cloud cluster with 6 collections, each holding 5 named dense vectors, 2 named sparse vectors (SPLADE and BM25), and 11 payload indexes.
2. Data ingestion: 13 regulatory documents and 25 SEC EDGAR filings downloaded with idempotent retries, polite rate-limiting, and full metadata sidecars.
3. Chunking: 6 strategies (3 per module): regulatory_boundary, semantic, hierarchical (compliance); financial_statement, semantic, narrative_section (credit). All produce stable UUID5 chunk IDs and absolute character offsets for Track A overlap scoring.
4. Embedding: open-source `mixedbread-ai/mxbai-embed-large-v1` (Matryoshka heads at 128, 256, 512, 768, 1024) plus SPLADE++ plus BM25 via `fastembed`. MPS auto-detection, subprocess isolation per collection to prevent macOS Metal compiler thrashing.
5. PCA eigenstructure analysis: full-rank sklearn PCA, three elbow detection methods (Kneedle, second derivative, 95 percent variance), per-module persistence.
6. Retrieval architecture: HybridRetriever class supporting dense, sparse, and 3 hybrid fusion modes; 4 query transformers; 4 rerankers (cross-encoder, MonoT5, ColBERT, RankGPT); cascade orchestrator.
7. Evaluation: chunking-agnostic source passage extractor, dual-track QA generator (50 questions per module, dual-tracked into 200 total QA pairs), Track A overlap-based evaluator, Track B answer-quality evaluator (semantic similarity + BERTScore F1 + key concept coverage).
8. Frontend: 5-tab Gradio dashboard with live querying, sample question chips, source corpus listing, full pipeline configuration, guardrail panel, and per-module performance dashboard rendering every benchmark JSON.
9. Guardrails: rule-based (no LLM), confidence scoring, citation enforcement, number grounding for credit, staleness warnings, temporal-query mismatch detection.
10. Logging: per-query JSONL log capturing config, timings, top chunks, generated answer, and guardrail report.

### Open-source model substitutions vs the original spec

| Spec | Substituted with | Reason |
|---|---|---|
| OpenAI `text-embedding-3-large` (1536-dim Matryoshka) | `mixedbread-ai/mxbai-embed-large-v1` (1024-dim Matryoshka, Apache 2.0) | Free, runs locally, true trained Matryoshka heads at every reported dimension |
| Cohere Rerank | Dropped (4 remaining open or LLM-based rerankers cover the comparison) | Paid baseline; comparison surface intact without it |
| Supabase Postgres + pgvector | Qdrant Cloud (Apache 2.0, free 1 GB tier) | Native named vectors store all 5 Matryoshka dims per point; native sparse and hybrid in one query call |
| `naver/splade-cocondenser-ensembledistil` | `prithivida/Splade_PP_en_v1` via `fastembed` | Same SPLADE family, fastembed-native (Qdrant integration), comparable quality |

## Quick start

### Local development

```bash
# 1. Clone and create venv (Python 3.11)
git clone <this repo>
cd <repo>
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Copy env template and fill in secrets
cp .env.example .env
# At minimum set: ANTHROPIC_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 3. Launch the dashboard
python app.py
# Then open http://localhost:7860
```

The dashboard works out of the box for performance tabs (charts read from JSON files in the repo) and for queries that hit Qdrant. The Anthropic key is only needed for query transforms, RankGPT reranking, and answer generation, all of which default to off.

### Docker

```bash
docker build -t bankmind .
docker run -p 7860:7860 \
  -e ANTHROPIC_API_KEY=... \
  -e QDRANT_URL=... \
  -e QDRANT_API_KEY=... \
  bankmind
```

### Reproduce the data and evaluations

The repo ships with all evaluation result JSONs, but the data pipeline is fully reproducible. Run scripts in this order:

```bash
python scripts/download_compliance_docs.py
python scripts/download_edgar_filings.py
python scripts/parse_documents.py
python scripts/run_chunking.py
python scripts/setup_qdrant_collections.py
bash scripts/embed_and_load_all.sh
python scripts/run_pca_analysis.py
python scripts/extract_source_passages.py
python scripts/generate_qa_pairs.py
python scripts/run_chunking_benchmark.py
python scripts/run_dimension_sweep.py
python scripts/run_retrieval_benchmark.py
python scripts/sweep_hybrid_convex_alpha.py
```

Each script is idempotent and skips work that's already been done.

## Repository structure

```
app/                   Gradio frontend
  main.py              5-tab UI with sample queries and source listings
  query_pipeline.py    Single function the UI calls; wires retrieve, rerank, generate, guardrails, log
  charts.py            Plotly figure builders for the performance tabs

pipelines/shared/      Reusable retrieval and evaluation infrastructure
  embedder.py          Matryoshka wrapper over mxbai-embed-large-v1
  sparse_encoder.py    SPLADE and BM25 via fastembed
  qdrant_client.py     Centralised client and naming convention
  chunking_base.py     Chunk dataclass, token counting, sentence and paragraph splitters
  semantic_chunker.py  Sentence-transformer boundary detection
  document_parser.py   PDF (pdfplumber) and HTML (BeautifulSoup) parsing with absolute char offsets
  retriever.py         HybridRetriever class (dense, sparse, hybrid modes)
  fusion.py            RRF, convex combination, hierarchical fusion
  query_transformer.py HyDE, Multi-Query, PRF, Step-Back
  reranker.py          Cross-encoder, MonoT5, ColBERT, RankGPT, cascade orchestrator
  pca_analyzer.py      Full-rank PCA + 3 elbow detection methods
  guardrails.py        Confidence, citation enforcement, number grounding, staleness, temporal warnings
  query_logger.py      Append-only JSONL per query
  llm.py               Anthropic client wrapper with caching and retry

pipelines/compliance/  Compliance-specific chunkers
pipelines/credit/      Credit-specific chunkers

evaluation/
  passage_extractor.py Chunking-agnostic source passage extraction
  qa_generator.py      Dual-track QA generation
  evaluator.py         Track A and Track B scoring
  results/             Persisted benchmark outputs (per module + summaries)

scripts/               CLI entry points for every pipeline stage

docs/
  WORK_LOG.md          Detailed session-by-session work log
```

## Findings worth highlighting

* **Semantic chunking dominates**: across both modules, embedding-driven topic boundary detection beat structure-driven chunking strategies by 17 to 94 percent NDCG. Domain-specific chunking is not always optimal; what matters is whether the boundaries align with how queries naturally probe the content.
* **BM25 is the right starting point** for retrieval over regulatory and financial corpora. Dense and SPLADE underperform; hybrid variants are competitive but don't beat raw BM25 on NDCG. Both corpora are exact-term-rich.
* **Cross-module ranking is not consistent below the winner**: the second-best chunking strategy differs per module (regulatory_boundary for compliance vs narrative_section for credit). Domain-specific chunking matters; assuming one strategy generalises is the wrong assumption.
* **Track A and Track B disagree by direction not magnitude**: a 22-point Track A NDCG gap between strategies collapses to a 7.6-point Track B composite gap. The LLM as post-hoc compensator narrows the answer-quality gap. Implication: retrieval quality matters more for citations than for end-user answer accuracy.
* **The PCA hypothesis was rejected, and that's the more interesting finding**: corpus topical breadth, not language formulaicness, is what drives intrinsic dimensionality. The dimension sweep validates this with empirical numbers.
* **Production is simpler than the spec suggested**: the winning pipeline is BM25 plus an LLM reranker plus a small query transform. None of the dense embedding cost is on the critical path of the production winner. The dense pipeline is a distraction in retrieval but useful for the dimension sweep narrative.

For the full session-by-session work log including incidents (overnight MPS thrashing, Metal compiler crash, transformers v5 reranker incompatibility), failure modes, decisions, and next steps, see [`docs/WORK_LOG.md`](docs/WORK_LOG.md).

## License

MIT.
