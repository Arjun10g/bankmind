# CLAUDE.md — BankMind: Multi-Domain RAG Platform for Financial Intelligence

## Project Overview

BankMind is a production-grade RAG platform with two specialized pipelines:
- **Module 1: Compliance Assistant** — Regulatory & compliance document Q&A
- **Module 2: Credit Analyst Copilot** — Credit risk analysis over financial filings

Both modules share a common ingestion, embedding, and retrieval infrastructure but differ
in chunking strategy, retrieval tuning, reranking configuration, and guardrails.

Deployed on **Hugging Face Spaces** (Gradio frontend), backed by **Supabase** (Postgres +
pgvector), with a full evaluation pipeline, Matryoshka + PCA/eigenstructure dimension
analysis, and a performance dashboard tab per module.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Gradio on Hugging Face Spaces |
| Backend API | FastAPI (same Space via subprocess or separate Space) |
| Database | Supabase (Postgres + pgvector extension) |
| Embeddings | `text-embedding-3-large` (1536-dim, Matryoshka-trained) |
| Sparse Retrieval | SPLADE (`naver/splade-cocondenser-ensembledistil`) |
| Reranking | ColBERT (RAGatouille), Cross-Encoder, Cohere Rerank, MonoT5, RankGPT |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| Orchestration | LangChain + LlamaIndex |
| Evaluation | RAGAS + custom NDCG/MRR/MAP/Recall@k/latency pipeline |
| QA Generation | Claude Code (automated, passage-grounded) |
| Compression | PCA eigenstructure analysis + Matryoshka sweep |
| Environment | Python 3.11, uv for package management |

---

## Repository Structure

```
bankmind/
├── CLAUDE.md                          # This file
├── README.md
├── .env.example
├── docker-compose.yml                 # Local dev only (mirrors Supabase schema)
├── pyproject.toml
├── requirements.txt
│
├── app/                               # Gradio frontend
│   ├── main.py                        # Entry point — launches Gradio
│   ├── tabs/
│   │   ├── compliance_tab.py          # Compliance Q&A interface
│   │   ├── credit_tab.py              # Credit analyst interface
│   │   ├── performance_compliance.py  # Performance dashboard — Module 1
│   │   ├── performance_credit.py      # Performance dashboard — Module 2
│   │   └── settings_tab.py           # Dimension selector, model selector
│   └── components/
│       ├── source_viewer.py           # Citation display component
│       └── metric_charts.py          # Plotly chart components
│
├── backend/
│   ├── api.py                         # FastAPI app
│   ├── routers/
│   │   ├── compliance.py
│   │   └── credit.py
│   └── middleware/
│       ├── guardrails.py
│       └── logging.py
│
├── pipelines/
│   ├── shared/
│   │   ├── embedder.py                # Matryoshka embedding wrapper
│   │   ├── pca_compressor.py          # PCA eigenstructure analysis + compression
│   │   ├── sparse_encoder.py          # SPLADE encoder
│   │   ├── reranker.py                # Reranking cascade
│   │   ├── query_transformer.py       # HyDE, Multi-Query, PRF, Step-Back
│   │   ├── fusion.py                  # RRF, Convex Combination, Hierarchical
│   │   └── supabase_client.py         # Supabase connection + vector ops
│   │
│   ├── compliance/
│   │   ├── ingestion.py               # PDF download + parse for regulatory docs
│   │   ├── chunker.py                 # 3 chunking strategies for compliance
│   │   ├── retriever.py               # Compliance-specific retrieval config
│   │   └── guardrails.py              # Citation enforcement, no hallucinated rules
│   │
│   └── credit/
│       ├── ingestion.py               # EDGAR + FRED ingestion
│       ├── chunker.py                 # 3 chunking strategies for credit docs
│       ├── retriever.py               # Credit-specific retrieval config
│       ├── agents.py                  # Agentic tools: calculator, ratio analyzer
│       └── guardrails.py              # No hallucinated numbers, confidence scoring
│
├── evaluation/
│   ├── qa_generator.py                # Claude Code generates QA pairs from passages
│   ├── evaluator.py                   # NDCG, MRR, MAP, Recall@k, latency
│   ├── dimension_sweep.py             # Matryoshka + PCA dimension experiments
│   ├── chunking_benchmark.py          # Compare 3 chunking strategies per module
│   ├── retrieval_benchmark.py         # Compare fusion + reranking methods
│   └── results/
│       ├── compliance/
│       └── credit/
│
├── data/
│   ├── raw/
│   │   ├── compliance/                # Downloaded regulatory PDFs
│   │   └── credit/                    # Downloaded 10-Ks, FRED exports
│   ├── processed/
│   │   ├── compliance/
│   │   └── credit/
│   └── eval/
│       ├── compliance_qa.json             # 50 QA pairs (Track A + B interleaved)
│       ├── credit_qa.json                 # 50 QA pairs (Track A + B interleaved)
│       └── source_passages/
│           ├── compliance_passages.json   # Raw passages pre-chunking (ground truth)
│           └── credit_passages.json
│
├── scripts/
│   ├── download_compliance_docs.py
│   ├── download_edgar_filings.py
│   ├── setup_supabase_schema.py       # Run once to create tables + indexes
│   ├── run_ingestion.py
│   ├── run_evaluation.py
│   └── generate_qa_pairs.py
│
└── notebooks/
    ├── 01_pca_eigenstructure_analysis.ipynb
    ├── 02_dimension_sweep_results.ipynb
    ├── 03_chunking_strategy_comparison.ipynb
    └── 04_retrieval_method_comparison.ipynb
```

---

## Phase 1: Infrastructure & Supabase Setup

### 1.1 Environment Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
uv init bankmind
cd bankmind
uv add fastapi uvicorn gradio langchain langchain-anthropic langchain-community \
       llama-index llama-index-vector-stores-supabase \
       anthropic openai cohere \
       supabase psycopg2-binary pgvector \
       sentence-transformers transformers torch \
       scikit-learn numpy pandas \
       pdfplumber unstructured pymupdf \
       ragatouille splade \
       ragas datasets \
       plotly matplotlib seaborn kneed \
       httpx python-dotenv tqdm
```

### 1.2 Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=
COHERE_API_KEY=
SUPABASE_URL=                    # From Supabase project settings
SUPABASE_SERVICE_KEY=            # Service role key (not anon key)
SUPABASE_DB_URL=                 # Direct Postgres connection string
HUGGINGFACE_TOKEN=               # For gated models
FRED_API_KEY=                    # From fred.stlouisfed.org
```

### 1.3 Supabase Schema Setup

Run `scripts/setup_supabase_schema.py` once. It must create:

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Compliance documents table
CREATE TABLE compliance_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id TEXT NOT NULL,
    doc_title TEXT,
    source_url TEXT,
    doc_type TEXT,                          -- 'osfi', 'fintrac', 'basel', 'bank_act', 'gdpr', 'fed'
    doc_version TEXT,
    effective_date DATE,
    chunk_index INTEGER,
    chunk_strategy TEXT,                    -- 'regulatory_boundary', 'semantic', 'hierarchical'
    content TEXT NOT NULL,
    content_tokens INTEGER,
    -- Metadata (structural/contextual/domain-specific)
    section_title TEXT,
    section_number TEXT,
    cross_references TEXT[],               -- Other docs/sections this chunk references
    regulatory_body TEXT,
    jurisdiction TEXT,
    document_date DATE,
    supersedes TEXT,                       -- Previous version this doc supersedes
    -- Embeddings at multiple dimensions (Matryoshka)
    embedding_256 vector(256),
    embedding_384 vector(384),
    embedding_512 vector(512),
    embedding_768 vector(768),
    embedding_1024 vector(1024),
    embedding_1536 vector(1536),
    -- PCA-compressed embeddings (dim determined by eigenstructure analysis)
    embedding_pca vector,                  -- Dynamic dim, set after PCA analysis
    pca_n_components INTEGER,
    -- Sparse vector for SPLADE
    sparse_vector sparsevec(30522),        -- SPLADE vocab size
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Credit documents table
CREATE TABLE credit_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id TEXT NOT NULL,
    doc_title TEXT,
    source_url TEXT,
    doc_type TEXT,                          -- '10k', '10q', '8k', 'credit_memo', 'fred_series'
    company_ticker TEXT,
    company_name TEXT,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    chunk_index INTEGER,
    chunk_strategy TEXT,                    -- 'financial_statement', 'semantic', 'narrative_section'
    content TEXT NOT NULL,
    content_tokens INTEGER,
    -- Metadata
    section_title TEXT,
    section_type TEXT,                      -- 'income_statement', 'balance_sheet', 'risk_factors', 'mda', 'notes'
    contains_table BOOLEAN DEFAULT FALSE,
    fiscal_period TEXT,
    industry_sector TEXT,
    naics_code TEXT,
    -- Embeddings at multiple dimensions
    embedding_256 vector(256),
    embedding_384 vector(384),
    embedding_512 vector(512),
    embedding_768 vector(768),
    embedding_1024 vector(1024),
    embedding_1536 vector(1536),
    -- PCA-compressed
    embedding_pca vector,
    pca_n_components INTEGER,
    -- Sparse
    sparse_vector sparsevec(30522),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Source passages table (pre-chunking, ground truth anchors)
-- These are raw document spans extracted BEFORE any chunking decision
-- Used as the unbiased ground truth for both Track A and Track B evaluation
CREATE TABLE source_passages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module TEXT NOT NULL,                   -- 'compliance' or 'credit'
    source_doc_id TEXT NOT NULL,
    source_doc_title TEXT,
    source_doc_type TEXT,
    passage_text TEXT NOT NULL,
    page_number INTEGER,
    char_start INTEGER NOT NULL,            -- Character offset in raw parsed doc
    char_end INTEGER NOT NULL,             -- Used for overlap scoring in Track A
    section_title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- QA pairs table (both tracks stored together, linked by qa_id)
CREATE TABLE qa_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    qa_id TEXT NOT NULL,                    -- Shared across Track A + B for same question
    track TEXT NOT NULL,                    -- 'A' or 'B'
    module TEXT NOT NULL,
    question TEXT NOT NULL,
    question_type TEXT,                     -- 'factual', 'interpretive', 'comparative'
    difficulty TEXT,                        -- 'easy', 'medium', 'hard'
    key_concepts TEXT[],
    source_passage_id UUID REFERENCES source_passages(id),
    source_passage_text TEXT,
    -- Track A fields
    char_start INTEGER,
    char_end INTEGER,
    source_doc_id TEXT,
    -- Track B fields
    reference_answer TEXT,                  -- Claude's answer from raw passage (gold standard)
    reference_answer_model TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


CREATE TABLE pca_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module TEXT NOT NULL,                   -- 'compliance' or 'credit'
    n_components_full INTEGER,
    elbow_dimension INTEGER,               -- Programmatically detected elbow
    explained_variance_ratios FLOAT[],
    cumulative_variance_at_elbow FLOAT,
    eigenvalues FLOAT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Evaluation results table
CREATE TABLE eval_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module TEXT NOT NULL,
    experiment_name TEXT,
    embedding_dim INTEGER,
    compression_method TEXT,               -- 'matryoshka', 'pca', 'full'
    chunking_strategy TEXT,
    retrieval_method TEXT,                 -- 'dense', 'sparse', 'hybrid'
    fusion_method TEXT,                    -- 'rrf', 'convex', 'hierarchical'
    query_transform TEXT,                  -- 'none', 'hyde', 'multi_query', 'prf', 'step_back'
    reranker TEXT,                         -- 'none', 'cross_encoder', 'colbert', 'cohere', 'monot5', 'rankgpt'
    ndcg FLOAT,
    mrr FLOAT,
    recall_at_1 FLOAT,
    recall_at_3 FLOAT,
    recall_at_5 FLOAT,
    recall_at_10 FLOAT,
    map_score FLOAT,
    avg_latency_ms FLOAT,
    p50_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    n_queries INTEGER,
    -- Track B answer quality metrics
    track_b_semantic_sim FLOAT,            -- Cosine sim: RAG answer vs reference answer
    track_b_bertscore_f1 FLOAT,            -- BERTScore F1: RAG answer vs reference answer
    track_b_concept_coverage FLOAT,        -- % key concepts present in RAG answer
    track_b_composite FLOAT,              -- Composite of above three
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW indexes for each embedding dimension (compliance)
CREATE INDEX ON compliance_chunks USING hnsw (embedding_256 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON compliance_chunks USING hnsw (embedding_384 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON compliance_chunks USING hnsw (embedding_512 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON compliance_chunks USING hnsw (embedding_768 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON compliance_chunks USING hnsw (embedding_1024 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON compliance_chunks USING hnsw (embedding_1536 vector_cosine_ops) WITH (m=16, ef_construction=64);

-- Same for credit
CREATE INDEX ON credit_chunks USING hnsw (embedding_256 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON credit_chunks USING hnsw (embedding_384 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON credit_chunks USING hnsw (embedding_512 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON credit_chunks USING hnsw (embedding_768 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON credit_chunks USING hnsw (embedding_1024 vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON credit_chunks USING hnsw (embedding_1536 vector_cosine_ops) WITH (m=16, ef_construction=64);

-- Text search indexes for BM25-style full text search
CREATE INDEX ON compliance_chunks USING GIN (to_tsvector('english', content));
CREATE INDEX ON credit_chunks USING GIN (to_tsvector('english', content));
```

**Important pgvector notes:**
- Use `halfvec` instead of `vector` for 30-40% storage savings at higher dims if Supabase supports it (check version — requires pgvector 0.7+)
- HNSW parameters: `m=16` is standard; increase `ef_construction` to 128 for better recall at cost of build time
- Monitor P99 latency from day one — set up logging immediately in Phase 1
- Never use default index parameters without benchmarking on your actual data

---

## Phase 2: Data Ingestion

### 2.1 Compliance Data Sources

Implement `scripts/download_compliance_docs.py` to download:

| Document | URL | Notes |
|---|---|---|
| OSFI B-20 | `osfi-bsif.gc.ca/en/guidance/guidance-library/b-20` | Residential mortgage underwriting |
| OSFI E-23 | `osfi-bsif.gc.ca/en/guidance/guidance-library/e-23` | Model risk management |
| OSFI B-10 | `osfi-bsif.gc.ca/en/guidance/guidance-library/b-10` | Third party risk |
| FINTRAC AML Guidelines | `fintrac-canafe.gc.ca/guidance-directives` | All applicable guidelines |
| Basel III Framework | `bis.org/bcbs/basel3.htm` | Core framework PDFs |
| Basel IV (CRR3) | `bis.org/bcbs/publ/d424.htm` | Final reform document |
| Bank Act Canada | `laws-lois.justice.gc.ca/eng/acts/B-1.01` | Full text |
| GDPR Full Text | `gdpr-info.eu` | All articles |
| Federal Reserve Reg W | `federalreserve.gov/supervisionreg` | Transactions with affiliates |

Target: **~15-20 documents, ~800-1200 pages total**

### 2.2 Credit Data Sources

Implement `scripts/download_edgar_filings.py` to download:

**Companies (pick 5 banks):**
- JPMorgan Chase (JPM)
- Bank of America (BAC)
- TD Bank (TD) — Canadian, relevant for the bank you're interviewing at
- Royal Bank of Canada (RY)
- Goldman Sachs (GS)

**Filing types per company:**
- 10-K (most recent 2 years)
- 10-Q (most recent 4 quarters)
- 8-K (earnings releases only)

**EDGAR API:**
```python
# Use SEC EDGAR full-text search API
base_url = "https://efts.sec.gov/LATEST/search-index"
# CIK lookup: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=jpmorgan
# Filing download: https://data.sec.gov/submissions/CIK{cik}.json
```

**FRED data via `fredapi` Python library:**
- Federal Funds Rate (FEDFUNDS)
- 10-Year Treasury (DGS10)
- Bank Prime Loan Rate (DPRIME)
- Commercial & Industrial Loan Delinquency (DRCCLACBS)
- S&P 500 (SP500)
- Sector-specific series for each company's industry

### 2.3 PDF Parsing Strategy

```python
# Use pdfplumber for text-heavy docs (regulatory)
# Use unstructured for mixed layout docs (10-Ks with tables)
# Use pymupdf as fallback for scanned docs

def parse_document(path: str, doc_type: str) -> list[dict]:
    if doc_type in ['osfi', 'fintrac', 'basel', 'bank_act']:
        return parse_with_pdfplumber(path)   # Better for structured regulatory text
    elif doc_type in ['10k', '10q']:
        return parse_with_unstructured(path)  # Better for financial tables
    else:
        return parse_with_pymupdf(path)
```

**Table extraction is critical for credit docs:**
- Extract tables separately and convert to markdown representation
- Preserve column headers and row labels in the chunk
- Tag chunks containing tables with `contains_table=True` metadata

---

## Phase 3: Chunking Strategies

Implement THREE chunking strategies per module. Each chunk must include rich metadata.

### Metadata Schema (all chunks, both modules)

```python
@dataclass
class ChunkMetadata:
    # Structural metadata
    doc_id: str
    doc_title: str
    chunk_index: int
    total_chunks: int
    page_number: int
    section_title: str
    section_number: str        # e.g. "3.2.1"
    hierarchy_path: str        # e.g. "Part II > Section 3 > Risk Factors"

    # Contextual metadata
    preceding_section: str     # Title of section before this chunk
    following_section: str     # Title of section after this chunk
    chunk_summary: str         # 1-sentence LLM-generated summary of this chunk
    keywords: list[str]        # Top 5 extracted keywords

    # Domain-specific metadata (compliance)
    regulatory_body: str
    jurisdiction: str
    doc_version: str
    effective_date: date
    cross_references: list[str]
    supersedes: str

    # Domain-specific metadata (credit)
    company_ticker: str
    fiscal_year: int
    fiscal_quarter: int
    section_type: str          # 'income_statement', 'balance_sheet', 'mda', etc.
    contains_table: bool
    industry_sector: str
```

**Note:** Generating per-chunk summaries is expensive. Generate document-level summaries
and section-level summaries only. Prepend these to chunks as context (contextual retrieval
pattern) rather than generating individual chunk summaries.

### 3.1 Compliance Module — Three Chunking Strategies

**Strategy 1: Regulatory Boundary Chunking**
*Theoretical basis:* Regulatory documents are organized around specific obligations,
definitions, and requirements. Splitting at natural regulatory boundaries (articles,
sections, sub-clauses) preserves the semantic integrity of each regulatory requirement.
A query about a specific obligation should retrieve the complete obligation, not half of it.

```python
def regulatory_boundary_chunk(doc: ParsedDoc) -> list[Chunk]:
    # Split on: Article X, Section Y, Subsection Z, Clause N
    # Regex patterns for OSFI/Basel/GDPR section markers
    # Min chunk: 100 tokens, Max chunk: 600 tokens
    # If section exceeds max: split at paragraph boundary
    # Preserve: section number, title, parent section context
    # Overlap: 0 (regulatory clauses should not bleed across chunks)
```

**Strategy 2: Semantic Similarity Chunking**
*Theoretical basis:* Regulatory language often contains thematic clusters that don't
align with section boundaries — e.g., all references to "capital adequacy" may span
multiple sections. Semantic chunking groups topically coherent passages regardless of
structural position, improving retrieval for thematic queries.

```python
def semantic_chunk(doc: ParsedDoc) -> list[Chunk]:
    # Use sentence-transformers to embed sentences
    # Compute cosine similarity between adjacent sentences
    # Split where similarity drops below threshold (0.4-0.6, tune per corpus)
    # Min chunk: 150 tokens, Max chunk: 512 tokens
    # Window: 3-sentence rolling window for boundary detection
```

**Strategy 3: Hierarchical Context Chunking**
*Theoretical basis:* Compliance queries often require understanding both a specific clause
AND its broader regulatory context. A question about a specific OSFI requirement may need
the governing principle from the parent section to be answerable correctly. Hierarchical
chunking creates parent-child relationships so retrieval can return the clause WITH its
governing context.

```python
def hierarchical_chunk(doc: ParsedDoc) -> list[Chunk]:
    # Level 1 (parent): Full section (~800-1200 tokens) — for context
    # Level 2 (child): Sub-clause (~150-300 tokens) — for precision
    # Store parent_id in child chunk metadata
    # At retrieval: return child chunk + prepend parent summary
    # Index both levels; retrieve children, fetch parents on-demand
```

### 3.2 Credit Module — Three Chunking Strategies

**Strategy 1: Financial Statement Boundary Chunking**
*Theoretical basis:* 10-Ks have clearly delineated financial statement sections (income
statement, balance sheet, cash flow, notes). Splitting at these boundaries ensures that
numerical data stays with its labels and context. Table-aware parsing is essential —
a chunk containing half an income statement is nearly useless for analysis.

```python
def financial_statement_chunk(doc: ParsedDoc) -> list[Chunk]:
    # Detect financial statement boundaries from headers
    # Extract tables as single chunks regardless of token count
    # Convert tables to markdown: | Revenue | $49.6B | $42.3B |
    # Narrative sections: 400-600 tokens
    # Table sections: keep whole, tag contains_table=True
    # Prepend column context to all table chunks
```

**Strategy 2: Semantic Chunking (same algorithm, different corpus)**
*Theoretical basis:* Financial narrative sections (MD&A, Risk Factors, Business Overview)
contain complex cross-cutting themes. A semantic chunker groups discussion of "credit risk"
even if it spans multiple subsections, improving recall for thematic credit analysis queries.

```python
def semantic_chunk_credit(doc: ParsedDoc) -> list[Chunk]:
    # Same algorithm as compliance semantic chunker
    # Tuned threshold: 0.45 (financial language is denser, needs wider window)
    # Special handling: don't split across table boundaries
    # 200-600 token target
```

**Strategy 3: Narrative Section Chunking**
*Theoretical basis:* 10-Ks have a defined SEC-mandated structure (Items 1-15).
Chunking by Item boundary preserves the analyst's natural document navigation —
Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Statements). This matches
how a credit analyst actually reads a 10-K and how their queries are naturally structured.

```python
def narrative_section_chunk(doc: ParsedDoc) -> list[Chunk]:
    # Split at SEC Item boundaries: "Item 1.", "Item 1A.", etc.
    # Sub-split long items at paragraph level (max 512 tokens)
    # Tag each chunk with: item_number, item_title, section_type
    # Preserve temporal markers: fiscal year, quarter in every chunk
```

---

## Phase 4: Embedding & Matryoshka Setup

### 4.1 Matryoshka Embedding Wrapper

`text-embedding-3-large` from OpenAI is trained with Matryoshka Representation Learning.
This means the first N dimensions of a 1536-dim embedding are a valid, meaningful embedding
at dimension N — you do not need to re-embed at different dimensions.

```python
# pipelines/shared/embedder.py

from openai import OpenAI
import numpy as np

DIMENSIONS = [256, 384, 512, 768, 1024, 1536]

class MatryoshkaEmbedder:
    def __init__(self):
        self.client = OpenAI()
        self.model = "text-embedding-3-large"

    def embed_full(self, texts: list[str]) -> np.ndarray:
        """Embed at full 1536 dims. Truncate for lower dims."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=1536
        )
        return np.array([r.embedding for r in response.data])

    def truncate_to_dim(self, embeddings: np.ndarray, dim: int) -> np.ndarray:
        """Matryoshka property: truncation IS a valid lower-dim embedding."""
        assert dim in DIMENSIONS, f"dim must be one of {DIMENSIONS}"
        truncated = embeddings[:, :dim]
        # L2-normalize after truncation
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        return truncated / np.maximum(norms, 1e-8)

    def embed_all_dimensions(self, texts: list[str]) -> dict[int, np.ndarray]:
        """Returns embeddings at all target dimensions from a single API call."""
        full = self.embed_full(texts)
        return {dim: self.truncate_to_dim(full, dim) for dim in DIMENSIONS}
```

**Critical:** One API call gives you all dimensions. Store all 6 dimension columns in
Supabase simultaneously. Do not make 6 separate embedding API calls.

### 4.2 SPLADE Sparse Encoding

```python
# pipelines/shared/sparse_encoder.py

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class SPLADEEncoder:
    def __init__(self):
        self.model_id = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
        self.model.eval()

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """Returns sparse vector as {token_id: weight} dict per text."""
        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=512)
            with torch.no_grad():
                output = self.model(**inputs)
            # SPLADE aggregation: max over sequence, relu, log(1 + x)
            logits = output.logits
            sparse = torch.log(1 + torch.relu(logits)).max(dim=1).values.squeeze()
            indices = sparse.nonzero().squeeze().tolist()
            weights = sparse[indices].tolist()
            results.append(dict(zip(
                [int(i) for i in indices],
                [float(w) for w in weights]
            )))
        return results

    def to_pgvector_sparsevec(self, sparse_dict: dict) -> str:
        """Format for pgvector sparsevec type."""
        # Format: '{index:weight,...}/vocab_size'
        entries = ",".join(f"{k}:{v}" for k, v in sorted(sparse_dict.items()))
        return f"{{{entries}}}/30522"
```

---

## Phase 5: PCA Eigenstructure Analysis

**This is the novel contribution of this project.** Run this AFTER ingesting the full
corpus but BEFORE finalizing storage decisions.

### 5.1 PCA Analysis Pipeline

```python
# pipelines/shared/pca_compressor.py

import numpy as np
from sklearn.decomposition import PCA
from kneed import KneeLocator
import json

class EigenstructureAnalyzer:
    """
    Analyzes the intrinsic dimensionality of a corpus via PCA on full-dim embeddings.
    The eigenvalue spectrum reveals the natural compression point for THIS specific corpus.
    This is corpus-specific — regulatory text and financial text will have different
    intrinsic dimensionalities, which is itself an interesting finding.
    """

    def fit(self, embeddings: np.ndarray, module: str) -> dict:
        """
        embeddings: (n_chunks, 1536) full-dim embedding matrix
        Returns: analysis results including detected elbow dimension
        """
        print(f"Fitting PCA on {embeddings.shape[0]} embeddings for module: {module}")

        # Fit full PCA to get complete eigenvalue spectrum
        pca = PCA(n_components=min(1536, embeddings.shape[0] - 1))
        pca.fit(embeddings)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        eigenvalues = pca.explained_variance_

        # Method 1: Kneedle algorithm on cumulative variance curve
        dims = list(range(1, len(cumulative_variance) + 1))
        kneedle = KneeLocator(
            dims, cumulative_variance.tolist(),
            curve='concave', direction='increasing'
        )
        elbow_dim_kneedle = kneedle.knee

        # Method 2: Second derivative (inflection point of eigenvalue spectrum)
        second_deriv = np.diff(np.diff(eigenvalues))
        elbow_dim_second_deriv = int(np.argmin(second_deriv)) + 2

        # Method 3: 95% explained variance threshold
        elbow_dim_95pct = int(np.searchsorted(cumulative_variance, 0.95)) + 1

        # Snap to nearest Matryoshka dimension for fair comparison
        matryoshka_dims = [256, 384, 512, 768, 1024, 1536]
        def snap_to_matryoshka(dim):
            return min(matryoshka_dims, key=lambda x: abs(x - dim))

        results = {
            "module": module,
            "n_embeddings": embeddings.shape[0],
            "elbow_kneedle": elbow_dim_kneedle,
            "elbow_second_deriv": elbow_dim_second_deriv,
            "elbow_95pct": elbow_dim_95pct,
            "elbow_kneedle_snapped": snap_to_matryoshka(elbow_dim_kneedle),
            "cumulative_variance_at_256": float(cumulative_variance[255]),
            "cumulative_variance_at_512": float(cumulative_variance[511]),
            "cumulative_variance_at_768": float(cumulative_variance[767]),
            "cumulative_variance_at_1024": float(cumulative_variance[1023]),
            "explained_variance_ratios": explained_variance_ratio.tolist()[:200],  # First 200 for plotting
            "eigenvalues": eigenvalues.tolist()[:200],
            "cumulative_variance": cumulative_variance.tolist()[:200],
        }

        # Store PCA transformation matrix for inference
        self.pca = pca
        self.elbow_dim = elbow_dim_kneedle

        return results

    def compress(self, embeddings: np.ndarray, n_components: int) -> np.ndarray:
        """Project embeddings into PCA space at detected elbow dimension."""
        compressed = self.pca.transform(embeddings)[:, :n_components]
        # L2-normalize compressed vectors
        norms = np.linalg.norm(compressed, axis=1, keepdims=True)
        return compressed / np.maximum(norms, 1e-8)

    def compress_query(self, query_embedding: np.ndarray, n_components: int) -> np.ndarray:
        """At query time: project query into same PCA space."""
        compressed = self.pca.transform(query_embedding.reshape(1, -1))[:, :n_components]
        norms = np.linalg.norm(compressed, axis=1, keepdims=True)
        return compressed / np.maximum(norms, 1e-8)

    def save(self, path: str):
        """Save PCA model for inference."""
        import joblib
        joblib.dump(self.pca, path)

    def load(self, path: str):
        import joblib
        self.pca = joblib.load(path)
```

### 5.2 What to Plot (notebooks/01_pca_eigenstructure_analysis.ipynb)

Generate these plots for BOTH modules side by side:

1. **Eigenvalue Spectrum** — log scale, shows rate of information decay
2. **Cumulative Explained Variance Curve** — with elbow marked, with 256/384/512/768 Matryoshka dims marked as vertical lines
3. **Matryoshka vs PCA Performance** — same axes, shows whether corpus-specific PCA elbow aligns with Matryoshka plateau
4. **Information per Dimension** — marginal gain of each additional dimension

**Key hypothesis to test:**
> Regulatory text (compliance) may have lower intrinsic dimensionality than financial
> narrative text (credit) because regulatory language is more formulaic and repetitive.
> If true, the PCA elbow for compliance will appear at a lower dimension than for credit.
> This would justify using smaller embedding dimensions for compliance retrieval,
> reducing storage and latency with minimal accuracy loss.

---

## Phase 6: Retrieval Architecture

### 6.1 Query Transformation Pipeline

All four transformations must be implemented and selectable at runtime:

```python
# pipelines/shared/query_transformer.py

class QueryTransformer:

    def hyde(self, query: str, module: str) -> str:
        """
        Hypothetical Document Embeddings.
        Generate a hypothetical answer, embed the answer (not the query).
        Works well when query and relevant documents are stylistically different
        (short casual query vs dense regulatory prose).
        """
        prompt = f"""You are an expert in {'regulatory compliance' if module == 'compliance' else 'credit risk analysis'}.
        Write a detailed, realistic passage that would answer this question.
        Question: {query}
        Write only the passage, no preamble."""
        hypothetical_doc = call_claude(prompt)
        return hypothetical_doc  # Embed this instead of the query

    def multi_query(self, query: str, module: str, n: int = 4) -> list[str]:
        """
        Generate N semantically diverse reformulations of the query.
        Retrieve for each, union the results, deduplicate.
        Addresses vocabulary mismatch and single-query brittleness.
        """
        prompt = f"""Generate {n} different ways to ask this question, each emphasizing
        a different aspect. Return as a JSON list of strings.
        Original: {query}"""
        return parse_json(call_claude(prompt))

    def pseudo_relevance_feedback(self, query: str, initial_results: list[str]) -> str:
        """
        Query Expansion via Pseudo-Relevance Feedback (PRF).
        Assume top-k retrieved docs are relevant. Extract key terms from them.
        Expand query with those terms for a second retrieval pass.
        Classic IR technique, surprisingly effective on domain-specific corpora.
        """
        top_docs_text = "\n".join(initial_results[:3])
        prompt = f"""Given this query and top retrieved passages, extract 5-8 key terms
        that should be added to the query to improve retrieval.
        Query: {query}
        Passages: {top_docs_text}
        Return expanded query only."""
        return call_claude(prompt)

    def step_back(self, query: str, module: str) -> str:
        """
        Step-Back Prompting.
        Generate a more abstract, principle-level version of the query.
        Retrieve for the abstract query first, then use that context for the specific query.
        Effective for specific regulatory questions that require understanding broader principles.
        E.g. "Does Clause 4.2.1 of B-20 apply to HELOC products?" →
             "What is the purpose of OSFI Guideline B-20?"
        """
        prompt = f"""Given this specific question, generate a more general, principle-level
        question that would provide useful background context.
        Specific question: {query}
        Return only the general question."""
        return call_claude(prompt)
```

### 6.2 Hybrid Retrieval

```python
# pipelines/shared/retriever.py

class HybridRetriever:
    def retrieve(
        self,
        query: str,
        module: str,
        embedding_dim: int,
        use_pca: bool,
        top_k: int = 50,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> list[ScoredChunk]:

        # Dense retrieval
        if use_pca:
            query_vec = pca_compressor.compress_query(embed(query), pca_n_components)
            dense_results = supabase_knn_search(query_vec, "embedding_pca", top_k)
        else:
            query_vec = embedder.truncate_to_dim(embed(query), embedding_dim)
            dense_results = supabase_knn_search(query_vec, f"embedding_{embedding_dim}", top_k)

        # Sparse retrieval (SPLADE)
        query_sparse = splade_encoder.encode([query])[0]
        sparse_results = supabase_sparse_search(query_sparse, top_k)

        # BM25 full-text search (Postgres tsvector)
        bm25_results = supabase_fts_search(query, top_k)

        return dense_results, sparse_results, bm25_results
```

### 6.3 Fusion Methods

```python
# pipelines/shared/fusion.py

class FusionEngine:

    def rrf(self, result_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
        """
        Reciprocal Rank Fusion.
        Score = sum(1 / (k + rank_i)) across all lists.
        Use when: scores across lists are not comparable (BM25 vs cosine).
        Do NOT use when relative score gaps matter — RRF ignores score magnitude.
        k=60 is standard; lower k amplifies rank differences.
        """
        scores = defaultdict(float)
        for result_list in result_lists:
            for rank, chunk in enumerate(result_list, 1):
                scores[chunk.id] += 1 / (k + rank)
        return sorted_by_score(scores)

    def convex_combination(
        self,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
        alpha: float = 0.7
    ) -> list[ScoredChunk]:
        """
        Weighted combination: final_score = alpha * dense_score + (1-alpha) * sparse_score
        CRITICAL: Normalize scores to [0,1] before combining.
        BM25 scores (e.g. 12.4) and cosine scores (e.g. 0.85) are NOT comparable raw.
        Use when: you care about score magnitude, not just ranking.
        alpha is tunable — sweep [0.5, 0.6, 0.7, 0.8] and pick best NDCG.
        """
        dense_norm = min_max_normalize(dense)
        sparse_norm = min_max_normalize(sparse)
        combined = {}
        for chunk in dense_norm:
            combined[chunk.id] = alpha * chunk.score
        for chunk in sparse_norm:
            combined[chunk.id] = combined.get(chunk.id, 0) + (1 - alpha) * chunk.score
        return sorted_by_score(combined)

    def hierarchical(
        self,
        query: str,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        """
        Hierarchical fusion: use query characteristics to decide fusion strategy.
        - If query contains exact terms/numbers/codes → weight sparse higher (alpha=0.4)
        - If query is conceptual/semantic → weight dense higher (alpha=0.85)
        - If query is short (<5 tokens) → use sparse only
        - Otherwise → RRF
        This is an intelligent routing layer, not just weighting.
        """
        query_tokens = query.split()
        has_exact_terms = bool(re.search(r'\b[A-Z]-\d+|\b\d{4}\b|"[^"]+"', query))

        if len(query_tokens) < 5:
            return sparse
        elif has_exact_terms:
            return self.convex_combination(dense, sparse, alpha=0.4)
        elif len(query_tokens) > 15:
            return self.convex_combination(dense, sparse, alpha=0.85)
        else:
            return self.rrf([dense, sparse])
```

### 6.4 Reranking Cascade

```python
# pipelines/shared/reranker.py

class RerankerCascade:
    """
    Cascade: first-stage retrieval (top 50-100) → rerank → (top 20) → rerank → (top 5)
    Each stage is progressively more expensive but operates on fewer candidates.
    """

    def cross_encoder_rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """
        Joint encoding of (query, passage) pair.
        More accurate than bi-encoder but O(n) forward passes.
        Use: ms-marco-MiniLM-L-6-v2 for speed, ms-marco-electra-base for accuracy.
        """

    def colbert_rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """
        Late interaction: query token embeddings × passage token embeddings.
        MaxSim aggregation per query token, summed.
        Use RAGatouille library: ragatouille.RAGPretrainedModel.from_pretrained('colbert-ir/colbertv2.0')
        More expressive than cross-encoder for long passages.
        """

    def cohere_rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """
        Cohere Rerank API: production-grade, one HTTP call.
        Good baseline — use as comparison point for your custom rerankers.
        Model: rerank-english-v3.0
        """

    def monot5_rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """
        T5 fine-tuned for ranking. Generates "true"/"false" for relevance.
        Use: castorini/monot5-base-msmarco or monot5-3b-msmarco for max accuracy.
        Strong on domain-specific text — good for regulatory language.
        """

    def rankgpt_rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """
        LLM-based reranking via sliding window permutation.
        Prompt Claude to rank N passages by relevance.
        Most expensive but most flexible — good for complex multi-part queries.
        Interesting to compare: does LLM reranking outperform neural rerankers?
        """
        prompt = f"""Rank these passages from most to least relevant for the query.
        Return a JSON list of passage numbers in ranked order.
        Query: {query}
        Passages: {format_passages(chunks)}"""

    def cascade(
        self,
        query: str,
        first_stage: list[ScoredChunk],
        reranker_config: str = "cross_encoder"
    ) -> list[ScoredChunk]:
        """
        Full cascade: 100 → ColBERT/SPLADE rerank → 20 → Cross-encoder → 5
        Configurable per experiment.
        """
```

---

## Phase 7: Evaluation Pipeline

### 7.1 QA Pair Generation — Dual-Track Evaluation Design

Run `scripts/generate_qa_pairs.py`. This implements two complementary evaluation tracks
that together give a complete picture of RAG quality without circularity.

---

#### The Core Problem With Naive QA Evaluation

If you select a chunk, generate a question from it, then check whether RAG retrieves
that chunk — you are **biasing toward your chunking strategy**. A question generated
from a 512-token regulatory boundary chunk will naturally be answered by that exact
chunk. A different chunking strategy that splits the same content differently might
retrieve equally valid content but score poorly simply because the chunk ID doesn't
match. This conflates chunking quality with retrieval quality.

The solution is to ground questions and reference answers in **raw document passages**
— contiguous spans of text extracted directly from the source document, completely
independent of any chunking decision. The passage is the atomic unit of truth.
The chunk is just an index artifact.

---

#### Source Passage Extraction (Pre-Chunking)

Before any chunking occurs, extract **raw passages** from the parsed documents.
A passage is a contiguous block of 3-6 sentences selected to be self-contained and
semantically complete. It does NOT align to chunk boundaries — it is selected from
the raw parsed text.

```python
# scripts/generate_qa_pairs.py

def extract_source_passages(parsed_docs: list[ParsedDoc], n_passages: int) -> list[dict]:
    """
    Extract raw passages from documents BEFORE chunking.
    Passages are the ground truth anchor — independent of any chunking strategy.
    
    Selection criteria for diversity:
    - At least 2 passages per document type
    - Mix of: dense numerical content, definitional content, procedural content
    - Min 150 tokens, max 400 tokens (long enough to be meaningful, short enough to be specific)
    - Must be self-contained (don't start mid-sentence, don't reference 'the above')
    """
    passages = []
    for doc in parsed_docs:
        # Sliding window over sentences, score by self-containment heuristics
        candidates = extract_candidate_passages(doc, min_tokens=150, max_tokens=400)
        # Sample for diversity across doc types and section types
        selected = diversity_sample(candidates, n=n_passages // len(parsed_docs))
        for p in selected:
            passages.append({
                "passage_id": generate_uuid(),
                "passage_text": p.text,
                "source_doc_id": doc.doc_id,
                "source_doc_title": doc.title,
                "source_doc_type": doc.doc_type,
                "page_number": p.page,
                "char_start": p.char_start,   # Character offset in raw doc
                "char_end": p.char_end,        # Used to find overlapping chunks at eval time
                "section_title": p.section_title,
                "module": doc.module           # 'compliance' or 'credit'
            })
    return passages
```

---

#### Track A — Retrieval Evaluation (Chunk-Agnostic)

**Purpose:** Measure whether RAG retrieves content that *overlaps* with the source
passage — not whether it retrieves an exact chunk ID. This is fair across all
chunking strategies because a question can be answered by any chunk that
substantially overlaps with the source passage.

```python
# Track A QA generation prompt

TRACK_A_PROMPT = """
You are generating evaluation questions for a financial RAG system.
Given this passage, generate {n} questions such that:
1. Each question is answerable ONLY using information in this passage
2. Questions do NOT contain phrases lifted verbatim from the passage
3. Questions use natural language a {role} would use in practice
4. Vary complexity: factual (specific facts/numbers), interpretive (meaning/application), comparative (contrast/scope)
5. A correct answer REQUIRES retrieving content from or closely overlapping this passage

Passage:
{passage_text}

Return ONLY valid JSON:
[
  {{
    "question": "...",
    "question_type": "factual|interpretive|comparative",
    "key_concepts": ["term1", "term2"],   // Terms that must appear in a correct answer
    "difficulty": "easy|medium|hard"
  }}
]
"""

# Storage schema for Track A
track_a_record = {
    "qa_id": "uuid",
    "track": "A",
    "module": "compliance|credit",
    "question": "...",
    "question_type": "factual|interpretive|comparative",
    "difficulty": "easy|medium|hard",
    "source_passage_id": "uuid",          # Links to passage table
    "source_passage_text": "...",         # Stored for overlap computation at eval time
    "source_doc_id": "...",
    "char_start": 1240,                   # Character offsets for overlap scoring
    "char_end": 1680,
    "key_concepts": ["capital adequacy", "Tier 1"],
    # NOTE: No source_chunk_id — evaluation is chunk-strategy-agnostic
}
```

**Evaluation logic for Track A:**
```python
def evaluate_track_a(question: dict, retrieved_chunks: list[ScoredChunk]) -> dict:
    """
    Instead of exact chunk ID match, compute passage overlap score.
    A retrieved chunk is 'relevant' if it overlaps significantly with the source passage.
    This is fair to ALL chunking strategies simultaneously.
    """
    source_start = question['char_start']
    source_end = question['char_end']
    source_doc = question['source_doc_id']

    relevance_scores = []
    for chunk in retrieved_chunks:
        if chunk.doc_id != source_doc:
            relevance_scores.append(0)
            continue
        # Character-level overlap ratio
        overlap_start = max(source_start, chunk.char_start)
        overlap_end = min(source_end, chunk.char_end)
        overlap_len = max(0, overlap_end - overlap_start)
        source_len = source_end - source_start
        overlap_ratio = overlap_len / source_len
        # Treat as relevant if >30% overlap with source passage
        relevance_scores.append(1 if overlap_ratio > 0.3 else 0)

    # Now compute NDCG, MRR, MAP, Recall@k using overlap-based relevance
    # This produces metrics that are fair across all 3 chunking strategies
    return compute_retrieval_metrics(relevance_scores)
```

---

#### Track B — Answer Quality Evaluation (Cross-Evaluation)

**Purpose:** This is the novel cross-evaluation track. The key design principle:
**the same question generated in Track A is reused identically in Track B.**
There is no separate question generation step — Track A and Track B share the
exact same `qa_id` and `question` string. What differs is what happens next.

**Upfront (at dataset creation time — before any RAG runs):**
1. Extract raw passage from document
2. Generate question from passage (shared with Track A)
3. Claude reads the raw passage and answers the question directly → **reference answer**
4. Store question + reference answer in `qa_pairs` table, `track='B'`

**At evaluation time:**
5. Feed the identical question into the RAG pipeline
6. RAG retrieves chunks, generates an answer
7. Compare RAG answer to the pre-stored reference answer using similarity metrics

Because the reference answer was generated from the raw passage (not any chunk, not
any retrieved context), it is completely unbiased by chunking strategy, retrieval
method, fusion method, or reranker. It represents the theoretical ceiling —
"what would the answer look like if we had perfect retrieval?"

**This answers a different question than Track A:**
- Track A: *Did we retrieve the right content?*
- Track B: *Is our generated answer as good as what we'd get from reading the source directly?*

Track B catches failure modes Track A misses: retrieval succeeds but generation
hallucinates; or retrieval partially fails but generation still synthesises a
reasonable answer from adjacent context. A high Track A + low Track B score
points to a generation problem. A low Track A + acceptable Track B score suggests
the model is compensating with general knowledge — a different kind of problem.

```python
# Track B reference answer generation prompt
# Run ONCE at dataset creation time, stored in DB, never regenerated

TRACK_B_REFERENCE_PROMPT = """
You are a senior {role} at a major bank. You have been given a passage from a
{doc_type} document and a question about it.

Your task: write the best possible answer to the question using ONLY the information
in the passage. Do not use outside knowledge. If the passage does not fully answer
the question, say what it does cover and note what is missing.

Be specific, precise, and professional. Cite specific figures, dates, or clause
references where present in the passage.

Passage:
{passage_text}

Question: {question}

Write your answer now:
"""

# Generation pipeline — runs once per QA pair at dataset creation
def generate_track_b_reference(passage: dict, question: str, module: str) -> dict:
    role = "compliance officer" if module == "compliance" else "credit analyst"
    doc_type = passage['source_doc_type']
    reference_answer = call_claude(
        TRACK_B_REFERENCE_PROMPT.format(
            role=role,
            doc_type=doc_type,
            passage_text=passage['passage_text'],
            question=question
        )
    )
    return {
        "qa_id": question['qa_id'],         # SAME qa_id as the Track A record
        "track": "B",
        "module": module,
        "question": question['question'],    # IDENTICAL question string — no regeneration
        "source_passage_id": passage['passage_id'],
        "reference_answer": reference_answer,
        "reference_answer_generated_by": "claude-sonnet-4-20250514",
        "reference_answer_tokens": count_tokens(reference_answer),
        # RAG answer fields — NULL at creation, populated per experiment at eval time
        "rag_answer": None,
        "rag_answer_similarity": None,
        "rag_answer_bertscore_f1": None,
        "rag_answer_key_concept_coverage": None,
    }
```

**Evaluation logic for Track B:**
```python
def evaluate_track_b(question: dict, rag_answer: str) -> dict:
    """
    Compare RAG-generated answer to Claude's reference answer from raw passage.
    Three complementary similarity measures — no single measure is sufficient.
    """
    reference = question['reference_answer']

    # 1. Semantic similarity (embedding cosine sim)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_rag = sim_model.encode([rag_answer])
    emb_ref = sim_model.encode([reference])
    semantic_sim = float(cosine_similarity(emb_rag, emb_ref)[0][0])

    # 2. BERTScore F1 (token-level semantic overlap, better for factual content)
    from bert_score import score as bert_score
    P, R, F1 = bert_score([rag_answer], [reference], lang='en', model_type='distilbert-base-uncased')
    bertscore_f1 = float(F1[0])

    # 3. Key concept coverage (domain-specific precision check)
    key_concepts = question.get('key_concepts', [])
    if key_concepts:
        covered = sum(1 for kc in key_concepts if kc.lower() in rag_answer.lower())
        concept_coverage = covered / len(key_concepts)
    else:
        concept_coverage = None

    # 4. Length ratio (penalize truncated answers)
    length_ratio = len(rag_answer.split()) / max(len(reference.split()), 1)

    return {
        "semantic_similarity": semantic_sim,
        "bertscore_f1": bertscore_f1,
        "key_concept_coverage": concept_coverage,
        "length_ratio": length_ratio,
        # Composite score (equal weight — tune if needed)
        "composite_score": np.mean([semantic_sim, bertscore_f1, concept_coverage or semantic_sim])
    }
```

---

#### QA Dataset Summary

```python
# Total dataset: 100 QA pairs per module, dual-tracked

QA_DATASET_STRUCTURE = {
    "compliance": {
        "n_source_passages": 25,          # 25 raw passages from compliance docs
        "n_questions_per_passage": 2,     # 2 questions each = 50 QA pairs
        "track_a": 50,                    # 50 retrieval eval pairs (chunk-agnostic)
        "track_b": 50,                    # 50 answer quality pairs (reference answers)
        "question_distribution": {
            "factual": 20,
            "interpretive": 20,
            "comparative": 10
        }
    },
    "credit": {
        "n_source_passages": 25,
        "n_questions_per_passage": 2,
        "track_a": 50,
        "track_b": 50,
        "question_distribution": {
            "factual": 20,
            "interpretive": 20,
            "comparative": 10
        }
    }
}
```

**Storage:** Save to `data/eval/compliance_qa.json` and `data/eval/credit_qa.json`.
Each file contains both tracks interleaved by `qa_id` so Track A and Track B entries
for the same question share the same `qa_id` and `source_passage_id`.

### 7.2 Metrics Implementation

```python
# evaluation/evaluator.py

import time
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGEvaluator:

    def ndcg_at_k(self, retrieved_ids: list[str], relevant_id: str, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain @ k.
        Measures ranking quality — relevant doc at rank 1 scores higher than rank k.
        DCG = sum(relevance_i / log2(rank_i + 1))
        NDCG = DCG / IDCG (ideal DCG)
        Binary relevance: 1 if retrieved == relevant, 0 otherwise.
        """
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id == relevant_id:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = 1.0 / np.log2(2)  # Ideal: relevant doc at rank 1
        return dcg / idcg if idcg > 0 else 0.0

    def mrr(self, retrieved_ids: list[str], relevant_id: str) -> float:
        """
        Mean Reciprocal Rank.
        Score = 1/rank of first relevant document.
        Answers: "how quickly does the relevant doc appear?"
        MRR=1.0 means it's always first. MRR=0.1 means it's around rank 10.
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == relevant_id:
                return 1.0 / rank
        return 0.0

    def recall_at_k(self, retrieved_ids: list[str], relevant_id: str, k: int) -> float:
        """
        Recall@k: is the relevant doc in the top k results?
        Binary for single-relevant-doc case: 1 if found, 0 if not.
        Report for k = 1, 3, 5, 10.
        """
        return float(relevant_id in retrieved_ids[:k])

    def map_score(self, retrieved_ids: list[str], relevant_id: str) -> float:
        """
        Mean Average Precision (single relevant doc case).
        AP = precision at the rank where relevant doc is found.
        Captures both precision and recall.
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == relevant_id:
                return 1.0 / rank
        return 0.0

    def overlap_relevance(self, qa: dict, retrieved_chunks: list) -> list[int]:
        """
        Track A: Compute binary relevance per retrieved chunk using character-level
        overlap with the source passage. Chunk-strategy-agnostic — fair for all
        chunking comparisons. Returns list of 0/1 relevance per retrieved chunk.
        """
        source_start = qa['char_start']
        source_end = qa['char_end']
        source_doc = qa['source_doc_id']
        relevance = []
        for chunk in retrieved_chunks:
            if chunk.doc_id != source_doc:
                relevance.append(0)
                continue
            overlap_start = max(source_start, chunk.char_start)
            overlap_end = min(source_end, chunk.char_end)
            overlap_len = max(0, overlap_end - overlap_start)
            source_len = source_end - source_start
            overlap_ratio = overlap_len / max(source_len, 1)
            relevance.append(1 if overlap_ratio > 0.3 else 0)
        return relevance

    def evaluate_track_b(self, qa: dict, rag_answer: str) -> dict:
        """
        Track B: Compare RAG-generated answer against Claude's reference answer
        (generated from raw passage, independent of chunking/retrieval).
        Three complementary measures — semantic sim, BERTScore F1, concept coverage.
        """
        reference = qa['reference_answer']

        sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        emb_rag = sim_model.encode([rag_answer])
        emb_ref = sim_model.encode([reference])
        semantic_sim = float(cosine_similarity(emb_rag, emb_ref)[0][0])

        from bert_score import score as bert_score
        _, _, F1 = bert_score([rag_answer], [reference], lang='en',
                              model_type='distilbert-base-uncased')
        bertscore_f1 = float(F1[0])

        key_concepts = qa.get('key_concepts', [])
        concept_coverage = (
            sum(1 for kc in key_concepts if kc.lower() in rag_answer.lower()) / len(key_concepts)
            if key_concepts else None
        )

        composite = np.mean([semantic_sim, bertscore_f1,
                             concept_coverage if concept_coverage is not None else semantic_sim])
        return {
            "semantic_similarity": semantic_sim,
            "bertscore_f1": bertscore_f1,
            "key_concept_coverage": concept_coverage,
            "composite_score": float(composite)
        }

    def evaluate_experiment(
        self,
        qa_pairs_track_a: list[dict],
        qa_pairs_track_b: list[dict],
        retriever_config: dict,
        module: str
    ) -> dict:
        """
        Run full dual-track evaluation for one experiment configuration.
        Track A: retrieval quality via overlap-based relevance scoring.
        Track B: answer quality via comparison to Claude's reference answers.
        Both tracks run for every experiment so results are always paired.
        """
        latencies = []
        ndcg_scores, mrr_scores, map_scores = [], [], []
        recall_1, recall_3, recall_5, recall_10 = [], [], [], []
        track_b_scores = []

        # Track A — retrieval evaluation
        for qa in qa_pairs_track_a:
            start = time.perf_counter()
            retrieved = retrieve(qa['question'], **retriever_config)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            relevance = self.overlap_relevance(qa, retrieved)

            ndcg_scores.append(self.ndcg_at_k(relevance, k=10))
            mrr_scores.append(self.mrr(relevance))
            map_scores.append(self.map_score(relevance))
            recall_1.append(self.recall_at_k(relevance, 1))
            recall_3.append(self.recall_at_k(relevance, 3))
            recall_5.append(self.recall_at_k(relevance, 5))
            recall_10.append(self.recall_at_k(relevance, 10))

        # Track B — answer quality evaluation
        for qa in qa_pairs_track_b:
            retrieved = retrieve(qa['question'], **retriever_config, top_k=5)
            rag_answer = generate_answer(qa['question'], retrieved, module)
            track_b_scores.append(self.evaluate_track_b(qa, rag_answer))

        latencies = np.array(latencies)
        return {
            # Track A metrics
            "ndcg": np.mean(ndcg_scores),
            "mrr": np.mean(mrr_scores),
            "map": np.mean(map_scores),
            "recall_at_1": np.mean(recall_1),
            "recall_at_3": np.mean(recall_3),
            "recall_at_5": np.mean(recall_5),
            "recall_at_10": np.mean(recall_10),
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            # Track B metrics
            "track_b_semantic_sim": np.mean([s['semantic_similarity'] for s in track_b_scores]),
            "track_b_bertscore_f1": np.mean([s['bertscore_f1'] for s in track_b_scores]),
            "track_b_concept_coverage": np.mean([s['key_concept_coverage'] for s in track_b_scores
                                                  if s['key_concept_coverage'] is not None]),
            "track_b_composite": np.mean([s['composite_score'] for s in track_b_scores]),
        }
```

### 7.3 Dimension Sweep Experiment

```python
# evaluation/dimension_sweep.py

DIMENSIONS = [256, 384, 512, 768, 1024, 1536]
MODULES = ['compliance', 'credit']

def run_dimension_sweep():
    for module in MODULES:
        results = []
        for dim in DIMENSIONS:
            # Matryoshka: retrieve at truncated dimension
            matryoshka_result = evaluator.evaluate_experiment(
                qa_pairs[module],
                retriever_config={"embedding_dim": dim, "use_pca": False},
                module=module
            )
            matryoshka_result.update({"dim": dim, "method": "matryoshka", "module": module})

            # PCA: retrieve at PCA-compressed dimension (if PCA elbow ≈ this dim)
            pca_result = evaluator.evaluate_experiment(
                qa_pairs[module],
                retriever_config={"embedding_dim": dim, "use_pca": True},
                module=module
            )
            pca_result.update({"dim": dim, "method": "pca", "module": module})

            results.extend([matryoshka_result, pca_result])

        save_results(results, f"evaluation/results/{module}/dimension_sweep.json")
```

**Plot for each module:**
- X-axis: embedding dimension
- Y-axis (left): NDCG@10, MRR (performance)
- Y-axis (right): latency ms (cost)
- Two lines per metric: Matryoshka vs PCA
- Vertical line at PCA elbow dimension
- Vertical line at Matryoshka performance plateau (where NDCG stops improving by >0.01)
- Annotation: "Inflection point — best performance/cost tradeoff"

### 7.4 Chunking Strategy Benchmark

The chunking benchmark is the most important controlled experiment in the project.
Everything else (retrieval method, reranker, query transform) is evaluated on top of
the best chunking strategy. Getting this right means all downstream experiments
are built on the correct foundation.

**Experimental controls:** Fix embedding dim at 512 (mid-range, not extreme), use
hybrid retrieval with RRF, no reranking, no query transformation. This isolates
the chunking effect from all other variables.

**Evaluation:** Use Track A overlap-based relevance (not exact chunk ID match) —
this is critical. If you evaluate chunking strategies using exact chunk ID match,
you will always favour the chunking strategy that generated the eval set, which is
circular. Track A's overlap scoring is the correct and fair evaluator here.

```python
# evaluation/chunking_benchmark.py

import itertools
import pandas as pd

CHUNKING_STRATEGIES = {
    'compliance': ['regulatory_boundary', 'semantic', 'hierarchical'],
    'credit': ['financial_statement', 'semantic', 'narrative_section']
}

FIXED_CONFIG = {
    "embedding_dim": 512,
    "use_pca": False,
    "retrieval": "hybrid",
    "fusion": "rrf",
    "reranker": "none",
    "query_transform": "none"
}

def run_chunking_benchmark(module: str, qa_pairs_track_a: list[dict]) -> pd.DataFrame:
    """
    For each chunking strategy:
    1. Retrieve using that strategy's indexed chunks
    2. Score using overlap-based relevance (Track A) — fair across all strategies
    3. Record all metrics + chunk-level diagnostics
    """
    results = []

    for strategy in CHUNKING_STRATEGIES[module]:
        strategy_results = {
            "module": module,
            "strategy": strategy,
            "ndcg_scores": [],
            "mrr_scores": [],
            "map_scores": [],
            "recall_1": [], "recall_3": [], "recall_5": [], "recall_10": [],
            "latencies": [],
            # Chunk diagnostic metrics
            "avg_chunk_tokens": [],
            "avg_retrieval_score": [],       # Top-1 cosine score per query
            "n_chunks_in_index": 0,          # Total indexed chunks for this strategy
        }

        # Load index for this specific chunking strategy
        # Each strategy has its own set of chunk embeddings in Supabase
        # Filtered by: chunk_strategy = strategy AND module = module
        retriever = HybridRetriever(module=module, chunk_strategy=strategy)
        strategy_results["n_chunks_in_index"] = retriever.count_chunks()

        for qa in qa_pairs_track_a:
            start = time.perf_counter()
            retrieved = retriever.retrieve(qa['question'], **FIXED_CONFIG, top_k=10)
            latency_ms = (time.perf_counter() - start) * 1000

            # Track A overlap scoring — chunk-strategy-agnostic relevance
            relevance = evaluator.overlap_relevance(qa, retrieved)

            strategy_results["ndcg_scores"].append(evaluator.ndcg_at_k(relevance, k=10))
            strategy_results["mrr_scores"].append(evaluator.mrr(relevance))
            strategy_results["map_scores"].append(evaluator.map_score(relevance))
            strategy_results["recall_1"].append(evaluator.recall_at_k(relevance, 1))
            strategy_results["recall_3"].append(evaluator.recall_at_k(relevance, 3))
            strategy_results["recall_5"].append(evaluator.recall_at_k(relevance, 5))
            strategy_results["recall_10"].append(evaluator.recall_at_k(relevance, 10))
            strategy_results["latencies"].append(latency_ms)
            strategy_results["avg_retrieval_score"].append(retrieved[0].score if retrieved else 0)

        # Aggregate
        results.append({
            "module": module,
            "strategy": strategy,
            "ndcg": np.mean(strategy_results["ndcg_scores"]),
            "mrr": np.mean(strategy_results["mrr_scores"]),
            "map": np.mean(strategy_results["map_scores"]),
            "recall_at_1": np.mean(strategy_results["recall_1"]),
            "recall_at_3": np.mean(strategy_results["recall_3"]),
            "recall_at_5": np.mean(strategy_results["recall_5"]),
            "recall_at_10": np.mean(strategy_results["recall_10"]),
            "avg_latency_ms": np.mean(strategy_results["latencies"]),
            "p99_latency_ms": np.percentile(strategy_results["latencies"], 99),
            "avg_top1_score": np.mean(strategy_results["avg_retrieval_score"]),
            "n_chunks_in_index": strategy_results["n_chunks_in_index"],
        })

    df = pd.DataFrame(results)
    save_results(df, f"evaluation/results/{module}/chunking_benchmark.json")
    return df


def run_cross_module_chunking_comparison():
    """
    Run chunking benchmark for both modules and produce a unified comparison.
    Identifies whether chunking strategy rankings are consistent across domains
    or domain-specific — an interesting finding either way.
    """
    compliance_df = run_chunking_benchmark('compliance', qa_pairs['compliance']['track_a'])
    credit_df = run_chunking_benchmark('credit', qa_pairs['credit']['track_a'])

    # Also run Track B comparison across chunking strategies
    # i.e. does chunking affect answer quality (Track B) as well as retrieval (Track A)?
    compliance_track_b = run_track_b_by_chunking('compliance', qa_pairs['compliance']['track_b'])
    credit_track_b = run_track_b_by_chunking('credit', qa_pairs['credit']['track_b'])

    return compliance_df, credit_df, compliance_track_b, credit_track_b


def run_track_b_by_chunking(module: str, qa_pairs_track_b: list[dict]) -> pd.DataFrame:
    """
    For each chunking strategy, run the full RAG pipeline (retrieve → generate),
    then evaluate the generated answer against the Track B reference answer.
    This shows whether chunking affects final answer quality, not just retrieval.
    """
    results = []
    for strategy in CHUNKING_STRATEGIES[module]:
        track_b_scores = []
        for qa in qa_pairs_track_b:
            retrieved = HybridRetriever(module=module, chunk_strategy=strategy).retrieve(
                qa['question'], **FIXED_CONFIG, top_k=5
            )
            rag_answer = generate_answer(qa['question'], retrieved, module)
            score = evaluator.evaluate_track_b(qa, rag_answer)
            track_b_scores.append(score)

        results.append({
            "module": module,
            "strategy": strategy,
            "avg_semantic_sim": np.mean([s['semantic_similarity'] for s in track_b_scores]),
            "avg_bertscore_f1": np.mean([s['bertscore_f1'] for s in track_b_scores]),
            "avg_concept_coverage": np.mean([s['key_concept_coverage'] for s in track_b_scores
                                             if s['key_concept_coverage'] is not None]),
            "avg_composite": np.mean([s['composite_score'] for s in track_b_scores]),
        })
    return pd.DataFrame(results)
```

**Plots to generate (notebooks/03_chunking_strategy_comparison.ipynb):**

**Plot 1 — Retrieval Metrics Radar Chart (per module)**
Radar/spider chart with axes: NDCG@10, MRR, MAP, Recall@5, Recall@10, Avg Latency (inverted).
One line per chunking strategy. Immediately shows which strategy dominates across metrics.

**Plot 2 — Bar Chart: All Metrics Side by Side**
Grouped bar chart: X = metric name, Y = score, colour = chunking strategy.
One chart per module. Shows precisely where strategies diverge.

**Plot 3 — Track A vs Track B Heatmap**
X = chunking strategy, Y = metric (NDCG, MRR, Semantic Sim, BERTScore F1).
Colour = score. Shows whether retrieval quality (Track A) correlates with
answer quality (Track B) — they may not, which is itself a finding.

**Plot 4 — Chunk Size Distribution**
Histogram of chunk token counts per strategy.
Overlaid with a line showing NDCG score at that chunk size bucket.
Tests the hypothesis: "Is there an optimal chunk size range for this domain?"

**Plot 5 — Cross-Module Consistency**
Do the same chunking strategies rank in the same order for compliance and credit?
Side-by-side bar: NDCG per strategy, compliance vs credit.
If rankings differ across modules — that's evidence that chunking is domain-specific,
a strong interview talking point.

**Plot 6 — Question Type Breakdown**
For the best and worst chunking strategies:
NDCG broken down by question type (factual / interpretive / comparative).
Shows whether a strategy is good at precise factual retrieval but poor at
thematic/interpretive queries — which maps directly to the theoretical basis
for each chunking strategy.

**Save all results to Supabase `eval_results` table** with `experiment_name = 'chunking_benchmark'`
so the performance dashboard tab can render them live.

### 7.5 Retrieval Method Benchmark

Fix the best chunking strategy from 7.4 and the best embedding dimension from 7.3.
Run experiments in three sequential ablation stages — do not run the full grid.
Each stage fixes the winner from the previous stage, isolating one variable at a time.

```python
# evaluation/retrieval_benchmark.py

import pandas as pd

# Stage 1: Retrieval method — which combination of dense/sparse performs best?
STAGE_1_RETRIEVAL = [
    {"label": "dense_only",    "retrieval": "dense",  "fusion": None,           "reranker": "none", "query_transform": "none"},
    {"label": "sparse_bm25",   "retrieval": "sparse", "fusion": None,           "reranker": "none", "query_transform": "none"},
    {"label": "sparse_splade", "retrieval": "splade", "fusion": None,           "reranker": "none", "query_transform": "none"},
    {"label": "hybrid_rrf",    "retrieval": "hybrid", "fusion": "rrf",          "reranker": "none", "query_transform": "none"},
    {"label": "hybrid_convex", "retrieval": "hybrid", "fusion": "convex",       "reranker": "none", "query_transform": "none"},
    {"label": "hybrid_hier",   "retrieval": "hybrid", "fusion": "hierarchical", "reranker": "none", "query_transform": "none"},
]

# Stage 2: Reranker ablation — fix best retrieval from Stage 1
STAGE_2_RERANKERS = [
    {"label": "no_reranker",     "reranker": "none"},
    {"label": "cross_encoder",   "reranker": "cross_encoder"},
    {"label": "colbert",         "reranker": "colbert"},
    {"label": "cohere",          "reranker": "cohere"},
    {"label": "monot5",          "reranker": "monot5"},
    {"label": "rankgpt",         "reranker": "rankgpt"},
]

# Stage 3: Query transform ablation — fix best retrieval + reranker from Stage 2
STAGE_3_TRANSFORMS = [
    {"label": "no_transform",  "query_transform": "none"},
    {"label": "hyde",          "query_transform": "hyde"},
    {"label": "multi_query",   "query_transform": "multi_query"},
    {"label": "prf",           "query_transform": "prf"},
    {"label": "step_back",     "query_transform": "step_back"},
]


def run_retrieval_benchmark(module: str, qa_track_a: list, qa_track_b: list) -> dict:
    """
    Run all three stages sequentially. Each stage selects its winner
    (best NDCG@10) before proceeding. Results from all stages stored
    in Supabase eval_results with experiment_name = 'retrieval_benchmark_stage_N'.
    Both Track A retrieval metrics and Track B answer quality metrics recorded
    for every experiment — the same question feeds both pipelines.
    """
    stage_results = {}

    # Stage 1
    stage1_rows = []
    for exp in STAGE_1_RETRIEVAL:
        result = evaluator.evaluate_experiment(qa_track_a, qa_track_b,
                                               retriever_config=exp, module=module)
        result.update({"module": module, "stage": 1, **exp})
        stage1_rows.append(result)
        save_to_supabase(result, experiment_name=f"retrieval_benchmark_stage_1")

    stage1_df = pd.DataFrame(stage1_rows)
    best_retrieval = stage1_df.loc[stage1_df['ndcg'].idxmax(), 'label']
    best_retrieval_config = next(e for e in STAGE_1_RETRIEVAL if e['label'] == best_retrieval)
    stage_results['stage1'] = stage1_df

    # Stage 2 — inherit best retrieval config
    stage2_rows = []
    for exp in STAGE_2_RERANKERS:
        config = {**best_retrieval_config, **exp}
        result = evaluator.evaluate_experiment(qa_track_a, qa_track_b,
                                               retriever_config=config, module=module)
        result.update({"module": module, "stage": 2, **config})
        stage2_rows.append(result)
        save_to_supabase(result, experiment_name=f"retrieval_benchmark_stage_2")

    stage2_df = pd.DataFrame(stage2_rows)
    best_reranker = stage2_df.loc[stage2_df['ndcg'].idxmax(), 'label']
    best_reranker_config = next(e for e in STAGE_2_RERANKERS if e['label'] == best_reranker)
    stage_results['stage2'] = stage2_df

    # Stage 3 — inherit best retrieval + reranker
    stage3_rows = []
    for exp in STAGE_3_TRANSFORMS:
        config = {**best_retrieval_config, **best_reranker_config, **exp}
        result = evaluator.evaluate_experiment(qa_track_a, qa_track_b,
                                               retriever_config=config, module=module)
        result.update({"module": module, "stage": 3, **config})
        stage3_rows.append(result)
        save_to_supabase(result, experiment_name=f"retrieval_benchmark_stage_3")

    stage3_df = pd.DataFrame(stage3_rows)
    stage_results['stage3'] = stage3_df

    return stage_results
```

---

**Plots to generate (notebooks/04_retrieval_method_comparison.ipynb):**

**Plot 1 — Stage 1: Retrieval Method Bar Chart**
Grouped bar chart: X = retrieval method label, groups = {NDCG@10, MRR, Recall@5, MAP}.
Separate subplots for compliance and credit side by side.
Annotation: winner highlighted in a different colour.
Shows immediately whether hybrid beats dense-only and which fusion method wins.

**Plot 2 — Stage 1: Retrieval Method + Latency Scatter**
X = avg latency (ms), Y = NDCG@10. Each point is one retrieval method.
Pareto frontier drawn. Methods on the frontier are the optimal cost/quality tradeoffs.
Points off the frontier are dominated — dominated methods should not be used in production.

**Plot 3 — Stage 2: Reranker Comparison Bar Chart**
Same grouped bar layout as Plot 1 but for rerankers.
Track A metrics (NDCG, MRR) and Track B metrics (BERTScore F1, Composite) on separate subplots.
Critical: shows whether reranker improvements in retrieval (Track A) also translate
to better generated answers (Track B) — they may not always correlate.

**Plot 4 — Stage 2: Reranker Latency Breakdown**
Stacked bar chart per reranker: retrieval latency | reranking latency | generation latency.
Shows where time is actually spent. MonoT5 and ColBERT will dominate reranking time.
RankGPT will dominate generation time. Cross-encoder is typically the fastest reranker.
This is the plot to point at when discussing production viability.

**Plot 5 — Stage 2: Reranker P99 Latency vs NDCG (Pareto)**
Same as Plot 2 but using P99 latency instead of average.
P99 is what users actually experience at the tail — this plot may produce a different
Pareto frontier than Plot 2. If it does, highlight the difference explicitly.

**Plot 6 — Stage 3: Query Transform Bar Chart**
Same grouped bar layout for query transform methods.
Include Track A and Track B side by side — query transforms primarily affect
retrieval (Track A) but may not improve answer quality (Track B) proportionally.
HyDE tends to help on interpretive queries. Step-Back helps on factual "what does X mean" queries.
PRF helps when queries are too short/vague. Show this breakdown by question type.

**Plot 7 — Stage 3: Query Transform Latency Impact**
Additional latency overhead per transform (LLM call needed for HyDE, Multi-Query, Step-Back).
Bar chart: baseline latency + transform overhead. Shows the cost of each transform.
PRF has two retrieval passes — its overhead is retrieval latency × 2 + LLM call.

**Plot 8 — Full Pipeline: Best Config vs Baseline**
Summary chart: baseline (dense-only, no reranker, no transform) vs best full pipeline.
Show improvement on every metric: NDCG, MRR, MAP, Recall@5, Track B Composite.
Show total latency increase. This is the "hero chart" for the README and interview.

**Plot 9 — Cross-Module Consistency for Retrieval Methods**
Do the same retrieval methods rank in the same order for compliance and credit?
Side-by-side NDCG heatmap: X = experiment label, Y = module.
If rankings differ — that's evidence that retrieval configuration is domain-specific,
reinforcing the same finding from the chunking benchmark.

**Save all results to Supabase `eval_results` table** with appropriate
`experiment_name` values so the performance dashboard tab can load and render
all plots from the database without re-running experiments.

---

## Phase 8: Frontend (Gradio on Hugging Face Spaces)

### 8.1 Tab Structure

```python
# app/main.py

import gradio as gr

with gr.Blocks(title="BankMind", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 BankMind — Financial Intelligence RAG Platform")

    with gr.Tabs():
        with gr.Tab("⚖️ Compliance Assistant"):
            build_compliance_tab()

        with gr.Tab("📊 Credit Analyst"):
            build_credit_tab()

        with gr.Tab("📈 Compliance Performance"):
            build_performance_compliance_tab()

        with gr.Tab("📉 Credit Performance"):
            build_performance_credit_tab()

        with gr.Tab("⚙️ Settings"):
            build_settings_tab()
```

### 8.2 Query Interface (both modules, same pattern)

Each query tab must expose:
- **Query input** text box
- **Embedding dimension selector**: radio buttons [256, 384, 512, 768, 1024, 1536, PCA]
- **Query transformation**: dropdown [None, HyDE, Multi-Query, PRF, Step-Back]
- **Fusion method**: dropdown [RRF, Convex Combination, Hierarchical]
- **Reranker**: dropdown [None, Cross-Encoder, ColBERT, Cohere, MonoT5, RankGPT]
- **Answer output** with source citations displayed below
- **Retrieval stats**: latency ms, top-k chunks retrieved, retrieval scores shown

### 8.3 Performance Dashboard Tab

Each module gets its own performance tab with:

**Section 1: Dimension Performance Curves**
- Plotly line chart: NDCG + MRR vs dimension, Matryoshka vs PCA
- PCA elbow marked as vertical dashed line
- Matryoshka plateau marked
- Latency curve on secondary Y-axis

**Section 2: PCA Eigenstructure Analysis**
- Cumulative explained variance curve
- Eigenvalue spectrum (log scale)
- Annotated elbow point with: dimension, % variance explained

**Section 3: Chunking Strategy Comparison**
- Bar chart: NDCG/MRR/Recall@5 per chunking strategy
- Table with all metrics

**Section 4: Retrieval Method Comparison**
- Stage 1 bar chart: NDCG/MRR/Recall@5 per retrieval method
- Pareto scatter: avg latency vs NDCG (Plotly, hoverable points)
- Winner highlighted

**Section 5: Reranker Comparison**
- Stage 2 grouped bar chart: Track A metrics + Track B metrics side by side
- Stacked bar: retrieval latency | reranking latency | generation latency per reranker
- P99 latency vs NDCG Pareto scatter
- Winner highlighted

**Section 6: Query Transform Comparison**
- Stage 3 bar chart per transform method, broken down by question type
- Latency overhead bar per transform
- Winner highlighted

**Section 7: Full Pipeline Hero Chart**
- Baseline (dense-only, no reranker, no transform) vs best full pipeline
- Delta shown for every metric: NDCG, MRR, MAP, Recall@5, Track B Composite
- Total latency increase shown prominently

**Section 5: Reranker Comparison**
- Scatter plot: latency vs NDCG (Pareto frontier)
- Shows cost/quality tradeoff per reranker

**Section 6: Query Transform Comparison**
- Bar chart per transform method
- Breakdown by question type (factual vs interpretive vs comparative)

---

## Phase 9: Guardrails

### 9.1 Compliance Module Guardrails

```python
# pipelines/compliance/guardrails.py

class ComplianceGuardrails:

    def enforce_citations(self, answer: str, source_chunks: list[Chunk]) -> dict:
        """
        Every claim in the answer must be traceable to a retrieved chunk.
        If answer contains claims not in any retrieved chunk → flag as hallucination.
        Return: {answer, citations: [{claim, source_doc, section}], hallucination_risk: float}
        """

    def version_check(self, chunks: list[Chunk]) -> list[Warning]:
        """
        If retrieved chunks are from older document versions, warn the user.
        "Note: This response is based on OSFI B-20 (2023). An update was issued 2024."
        """

    def scope_check(self, query: str, answer: str) -> bool:
        """
        Detect if answer goes beyond the retrieved documents.
        Prompt Claude: "Does this answer contain information not supported by these passages?"
        """
```

### 9.2 Credit Module Guardrails

```python
# pipelines/credit/guardrails.py

class CreditGuardrails:

    def number_grounding_check(self, answer: str, source_chunks: list[Chunk]) -> dict:
        """
        Extract all numbers from answer. Verify each appears in source chunks.
        Hallucinated financial figures are the highest-risk failure mode.
        """

    def confidence_score(self, query: str, retrieved_chunks: list[ScoredChunk]) -> float:
        """
        Estimate answer confidence from retrieval scores.
        Low top-1 retrieval score → low confidence → surface to user.
        """

    def temporal_warning(self, chunks: list[Chunk], query: str) -> str | None:
        """
        If query asks about current state but retrieved chunks are from 2+ years ago,
        warn: "This analysis is based on FY2022 filings. More recent data may differ."
        """
```

---

## Phase 10: Logging & Observability

Set up from Day 1. Do not defer.

```python
# Every retrieval call must log:
{
    "timestamp": "...",
    "module": "compliance|credit",
    "query": "...",
    "query_transform": "...",
    "embedding_dim": 512,
    "use_pca": False,
    "fusion_method": "rrf",
    "reranker": "cross_encoder",
    "n_retrieved": 50,
    "n_reranked": 5,
    "top_chunk_id": "...",
    "top_chunk_score": 0.87,
    "latency_retrieval_ms": 145,
    "latency_reranking_ms": 230,
    "latency_generation_ms": 890,
    "latency_total_ms": 1265,
    "answer_length_tokens": 312,
    "hallucination_flag": False,
    "confidence_score": 0.82
}
```

Use **LangSmith** for LLM call tracing. Log to a `query_logs` table in Supabase for
retrieval analytics. Monitor P99 latency — not just average.

---

## Common Mistakes to Avoid (Encoded from Best Practices)

### Vector Database
- [ ] Never choose index parameters without benchmarking on your actual corpus
- [ ] Benchmark P99 latency from Phase 1, not at the end
- [ ] Use `halfvec` for storage savings if Supabase pgvector version supports it
- [ ] HNSW `ef_search` at query time is tunable without rebuilding index — tune it
- [ ] Monitor query latency separately from ingestion latency

### Hybrid Retrieval
- [ ] Always normalize BM25 and cosine scores before convex combination
- [ ] RRF when scores are incomparable; convex when score gaps matter
- [ ] SPLADE beats BM25 on domain-specific corpora — benchmark both
- [ ] Log what is actually retrieved for every query — debugging blind is impossible

### Chunking
- [ ] Don't split tables mid-row — extract as atomic units
- [ ] Include `preceding_section` and `following_section` metadata in every chunk
- [ ] Generate section-level summaries, not chunk-level (cost vs. benefit)
- [ ] Temporal metadata is critical for financial docs — every chunk needs fiscal period

### Evaluation
- [ ] Report NDCG, not just Recall — NDCG captures ranking quality
- [ ] Always include latency in evaluation — accuracy at 10s is not production-grade
- [ ] P99 latency matters more than average for user experience
- [ ] Generate QA pairs before ingestion to avoid test set contamination

### Reranking
- [ ] Always rerank — simple cross-encoder typically outperforms retrieval model upgrades
- [ ] Cascade: cheap reranker on 100 candidates → expensive on 20 → generate on 5
- [ ] Measure reranker latency separately — it often dominates total latency

---

## Deployment: Hugging Face Spaces

### Space Configuration (`README.md` header)
```yaml
---
title: BankMind
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.x
app_file: app/main.py
pinned: false
secrets:
  - ANTHROPIC_API_KEY
  - COHERE_API_KEY
  - SUPABASE_URL
  - SUPABASE_SERVICE_KEY
---
```

### Supabase Connection from Spaces
- Use `SUPABASE_DB_URL` (direct connection) not the REST API for pgvector queries
- Connection pooling: use `?pgbouncer=true` suffix on connection string for Supabase
- Keep connection pool size ≤ 10 for free tier Supabase

### Performance Considerations on Free Tier
- ColBERT and MonoT5 reranking are CPU-heavy — add a spinner + latency warning in UI
- Cache embeddings for repeated queries (LRU cache, 100 entries)
- Consider making reranking async with progress indicator

---

## Build Order (Phase by Phase Checkpoints)

Each phase should be independently runnable and leave the system in a working state.

```
✅ Phase 1: Supabase schema created, env configured, basic connection test passes
✅ Phase 2: Documents downloaded, parsed, stored in /data/raw and /data/processed
✅ Phase 3: All 6 chunking strategies implemented, chunks stored in Supabase with metadata
✅ Phase 4: Embeddings at all 6 dims + SPLADE stored in Supabase
✅ Phase 5: PCA analysis complete, eigenstructure plots generated, PCA embeddings stored
✅ Phase 6: Hybrid retrieval working, all fusion methods + rerankers + query transforms
✅ Phase 7: 100 QA pairs generated, full eval pipeline runs, results in eval/results/
✅ Phase 8: Gradio frontend running locally with all tabs functional
✅ Phase 9: Guardrails integrated into both pipelines
✅ Phase 10: Logging active, LangSmith connected, all latencies tracked
✅ Deploy: Pushed to Hugging Face Spaces, Supabase connected, end-to-end test passes
```

---

## Interview Talking Points Generated by This Project

1. **Matryoshka vs PCA finding** — "I found the PCA elbow for regulatory text appeared at ~384 dims vs ~512 for financial narrative text, suggesting regulatory language has lower intrinsic dimensionality — consistent with its formulaic structure."

2. **Chunking insight** — "Regulatory boundary chunking outperformed semantic chunking on factual queries but underperformed on interpretive queries, which makes sense — semantic chunking groups thematic content that crosses section boundaries. I verified this breakdown by question type."

3. **Cross-module chunking finding** — "The chunking strategy rankings were not consistent across domains. For compliance, hierarchical chunking had the best NDCG. For credit, financial statement boundary chunking dominated — because the SEC Item structure is already semantically optimal. Domain-specific chunking isn't optional, it's required."

4. **Dual-track evaluation design** — "I deliberately avoided grounding evaluation in chunk IDs, because that would bias results toward whichever chunking strategy generated the eval set. Instead I anchored questions to raw document passages using character offsets, then scored retrieval by overlap ratio. This means the same 50 questions fairly evaluate all three chunking strategies simultaneously."

5. **Track B cross-evaluation** — "Track B is independent of retrieval entirely. Claude generates a reference answer directly from the raw passage. I then compare the RAG-generated answer to that reference using BERTScore F1 and semantic similarity. This catches cases where retrieval succeeds but generation fails — and cases where partial retrieval still produces an acceptable answer. Track A and Track B disagreed on 12% of queries, which told me exactly where generation quality was the bottleneck vs retrieval quality."

6. **Reranker tradeoff** — "ColBERT improved NDCG by X% over cross-encoder but at 3× the latency. For the compliance module where accuracy is paramount, that's worth it. For customer-facing use cases, it wouldn't be."

7. **Hybrid fusion finding** — "RRF outperformed convex combination when queries mixed exact regulatory codes with semantic concepts, because convex combination was biased by raw BM25 score magnitude."

8. **Production mindset** — "I tracked P99 latency from day one. Average latency was 800ms but P99 was 3.2s — the tail was dominated by MonoT5 reranking on long regulatory passages. That's the number that matters."
