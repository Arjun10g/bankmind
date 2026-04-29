"""BankMind Gradio frontend.

Five tabs:
  1. Compliance Q&A: live retrieval over compliance corpus
  2. Credit Q&A: live retrieval over credit corpus
  3. Compliance Performance: PCA, dim sweep, chunking, retrieval ablation charts
  4. Credit Performance: same charts for credit
  5. About: pipeline overview + cost-control notes

LLM-using features (query transforms, LLM-based reranker, answer generation) are
OFF by default. Each is gated by a checkbox so the user controls when an LLM API
is called. Pure retrieval (any chunking + dense/sparse/hybrid + cross-encoder
reranker) is free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.charts import (
    chunking_figure,
    dim_sweep_figure,
    pca_figure,
    retrieval_stage_1_figure,
    retrieval_stage_2_figure,
    retrieval_stage_3_figure,
)
from app.query_pipeline import run_query


COMPLIANCE_STRATEGIES = ["regulatory_boundary", "semantic", "hierarchical"]
CREDIT_STRATEGIES = ["financial_statement", "semantic", "narrative_section"]
DIMS = [128, 256, 512, 768, 1024]
RETRIEVAL_METHODS = ["bm25", "splade", "dense", "hybrid_rrf", "hybrid_convex", "hybrid_hier"]
RERANKERS = ["none", "cross_encoder", "monot5", "colbert", "rankgpt"]
TRANSFORMS = ["none", "hyde", "multi_query", "prf", "step_back"]


# Sample queries shown as clickable examples in each Q&A tab.
COMPLIANCE_EXAMPLES = [
    "What does OSFI Guideline B-20 require for residential mortgage underwriting?",
    "How does FINTRAC define a politically exposed person?",
    "What are the Tier 1 capital ratio requirements under Basel III?",
    "What does OSFI E-23 require for enterprise model risk management?",
    "What are the data subject rights granted under GDPR?",
    "Summarise Federal Reserve Regulation W transaction limits with affiliates.",
    "What governance expectations does OSFI B-10 set for third party risk management?",
    "How does Basel III define risk-weighted assets for credit exposures?",
]
CREDIT_EXAMPLES = [
    "What credit risk factors did JPMorgan disclose in its most recent 10-K?",
    "What is Goldman Sachs' Tier 1 capital ratio for the latest reporting period?",
    "How did Bank of America's net interest income change year over year?",
    "Summarise the cybersecurity disclosures in JPMorgan's 10-K.",
    "What does Goldman Sachs say about market risk in its 10-Q?",
    "What is TD Bank's allowance for credit losses methodology?",
    "Compare the regulatory capital position across the latest 40-F filings.",
    "What are the principal liquidity risk drivers disclosed in the bank 10-Ks?",
]


# Source corpus listings for the "What can I ask about?" panel.
COMPLIANCE_SOURCES_MD = """
**13 regulatory documents** spanning four jurisdictions:

| Doc ID | Title | Body | Jurisdiction |
|---|---|---|---|
| `osfi_b20` | Residential Mortgage Underwriting Practices and Procedures (B-20) | OSFI | Canada |
| `osfi_e23` | Enterprise-Wide Model Risk Management (E-23) | OSFI | Canada |
| `osfi_b10` | Third Party Risk Management Guideline (B-10) | OSFI | Canada |
| `osfi_integrity_security` | Integrity and Security Guideline | OSFI | Canada |
| `fintrac_guide11_client_id` | Client Identification Methods (Guide 11) | FINTRAC | Canada |
| `basel_iii_framework_2011` | Basel III: A Global Regulatory Framework (BCBS 189) | BCBS | International |
| `basel_iii_finalising_2017` | Basel III: Finalising Post-Crisis Reforms (BCBS d424) | BCBS | International |
| `basel_d440`, `basel_d457`, `basel_d544` | Additional BCBS publications | BCBS | International |
| `bank_act_canada` | Bank Act (S.C. 1991, c. 46) | Justice Canada | Canada |
| `gdpr_consolidated` | General Data Protection Regulation (EU 2016/679) | EU Parliament | European Union |
| `fed_reg_w` | Regulation W: 12 CFR Part 223 | Federal Reserve Board | United States |

Total: about 5.5 million characters, indexed as 14,318 chunks across three chunking strategies.
"""

CREDIT_SOURCES_MD = """
**25 SEC EDGAR filings** from five major banks:

| Ticker | Company | Filings retrieved |
|---|---|---|
| `JPM` | JPMorgan Chase & Co | 2 x 10-K, 4 x 10-Q, 1 x 8-K (earnings) |
| `BAC` | Bank of America Corporation | 2 x 10-K, 4 x 10-Q, 1 x 8-K (earnings) |
| `GS` | The Goldman Sachs Group, Inc. | 2 x 10-K, 4 x 10-Q, 1 x 8-K (earnings) |
| `TD` | Toronto-Dominion Bank | 1 x 40-F (annual), 4 x 6-K (interim) |
| `RY` | Royal Bank of Canada | 1 x 40-F (annual), 4 x 6-K (interim) |

TD and RY file 40-F + 6-K (foreign private issuer forms) instead of 10-K + 10-Q.

Total: about 15.4 million characters, indexed as 18,645 chunks across three chunking strategies.
"""


def _format_chunks(chunks) -> str:
    """Render top chunks as Markdown with citations."""
    if not chunks:
        return "_(no results)_"
    lines = []
    for i, c in enumerate(chunks, 1):
        sec = c.payload.get("section_title", "")
        doc = c.payload.get("doc_id", "")
        snippet = (c.content or "").strip().replace("\n", " ")
        if len(snippet) > 350:
            snippet = snippet[:350] + "…"
        lines.append(
            f"**[{i}]** _doc: `{doc}` · §_ `{sec[:80]}` · _score:_ {c.score:.3f}\n\n{snippet}\n\n---\n"
        )
    return "\n".join(lines)


def _format_timings(timings: dict) -> str:
    if not timings:
        return ""
    parts = [f"**{k.replace('_ms','').replace('_',' ')}**: {v:.0f} ms"
             for k, v in timings.items() if k.endswith("_ms")]
    return "  ·  ".join(parts)


_SEVERITY_BADGE = {
    "high": "🔴 **HIGH**",
    "warning": "🟡 **warning**",
    "info": "🔵 _info_",
}


def _format_guardrails(report) -> str:
    if report is None:
        return ""
    lines = []
    label_color = {"low": "🔴", "medium": "🟡", "high": "🟢"}.get(report.confidence_label, "")
    lines.append(
        f"**Confidence**: {label_color} `{report.confidence_label}`  ·  "
        f"score = {report.confidence:.2f}"
    )
    if report.citation_coverage is not None:
        lines.append(
            f"**Citation coverage**: {report.citation_coverage * 100:.0f}%  "
            f"({len(report.unsupported_sentences)} sentences without supporting chunks)"
        )
    if report.grounded_numbers or report.ungrounded_numbers:
        lines.append(
            f"**Number grounding**: {report.grounded_numbers} ✅  ·  "
            f"{len(report.ungrounded_numbers)} ❌"
            + (f"  →  ungrounded: `{', '.join(report.ungrounded_numbers[:5])}`"
               if report.ungrounded_numbers else "")
        )
    if report.warnings:
        lines.append("\n**Warnings:**")
        for w in report.warnings:
            badge = _SEVERITY_BADGE.get(w.severity, w.severity)
            lines.append(f"- {badge} `{w.code}` — {w.message}")
    if report.unsupported_sentences:
        lines.append(
            "\n_Unsupported sentences in the answer (potential hallucination):_"
        )
        for s in report.unsupported_sentences[:3]:
            lines.append(f"  - _{s[:200]}{'…' if len(s) > 200 else ''}_")
    return "\n\n".join(lines)


# =============================================================================
# Q&A tab builder
# =============================================================================

def _build_qa_tab(module: str, strategies: list[str], default_strategy: str):
    examples = COMPLIANCE_EXAMPLES if module == "compliance" else CREDIT_EXAMPLES
    sources_md = COMPLIANCE_SOURCES_MD if module == "compliance" else CREDIT_SOURCES_MD

    with gr.Column():
        gr.Markdown(f"### {module.title()} Q&A: live retrieval over the {module} corpus")
        gr.Markdown(
            "Pure retrieval queries (no LLM checkboxes ticked) are **free**. Each "
            "ticked LLM option below adds 1 LLM API call to the query path."
        )

        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="Query",
                    lines=2,
                    placeholder=(
                        "e.g. What are the residential mortgage underwriting standards in OSFI B-20?"
                        if module == "compliance"
                        else "e.g. What are Goldman Sachs' key credit risk factors disclosed in the 10-K?"
                    ),
                )
            with gr.Column(scale=1):
                run_btn = gr.Button("Run query", variant="primary")

        with gr.Accordion("💡 Sample questions (click to use)", open=True):
            gr.Examples(
                examples=[[ex] for ex in examples],
                inputs=[query],
                label="",
                examples_per_page=10,
            )

        with gr.Accordion("📚 What's in the corpus?", open=False):
            gr.Markdown(sources_md)

        with gr.Accordion("Pipeline configuration", open=False):
            with gr.Row():
                strategy = gr.Radio(strategies, value=default_strategy, label="Chunking strategy")
                dim = gr.Radio([str(d) for d in DIMS], value="512", label="Embedding dim")
            with gr.Row():
                method = gr.Dropdown(RETRIEVAL_METHODS, value="hybrid_rrf",
                                     label="Retrieval method")
                reranker = gr.Dropdown(RERANKERS, value="none", label="Reranker")
                transform = gr.Dropdown(TRANSFORMS, value="none", label="Query transform")
            with gr.Row():
                top_k = gr.Slider(5, 30, value=10, step=1, label="Retrieval top_k")
                final_k = gr.Slider(3, 15, value=5, step=1, label="Final top_n")
                generate = gr.Checkbox(value=False, label="Generate answer (LLM call, paid)")

        with gr.Row():
            timings_md = gr.Markdown(label="Timings", value="")
            config_md = gr.Markdown(label="Config", value="")

        answer_md = gr.Markdown(label="Answer", value="_(generation off; toggle 'Generate answer' to enable)_")

        with gr.Accordion("🛡️ Guardrails", open=True):
            guardrails_md = gr.Markdown(value="_(run a query to see guardrails)_")

        gr.Markdown("### Top retrieved chunks")
        chunks_md = gr.Markdown()

    def _run(q, strat, d, m, r, t, k, fk, gen):
        result = run_query(
            query=q, module=module, chunk_strategy=strat,
            embedding_dim=int(d), retrieval_method=m, reranker=r,
            query_transform=t, top_k=int(k), final_k=int(fk),
            generate_answer=bool(gen),
        )
        ans = result.answer if result.answer is not None else "_(generation off)_"
        return (
            _format_timings(result.timings),
            f"`{result.config_summary}`  ·  query_id=`{result.query_id}`",
            ans,
            _format_guardrails(result.guardrail_report),
            _format_chunks(result.chunks),
        )

    run_btn.click(
        _run,
        inputs=[query, strategy, dim, method, reranker, transform, top_k, final_k, generate],
        outputs=[timings_md, config_md, answer_md, guardrails_md, chunks_md],
    )


# =============================================================================
# Performance tab builder
# =============================================================================

def _build_perf_tab(module: str):
    with gr.Column():
        gr.Markdown(f"### {module.title()} performance: pre-computed evaluation results")

        with gr.Row():
            gr.Plot(value=pca_figure(module))
        with gr.Row():
            gr.Plot(value=dim_sweep_figure(module))
        with gr.Row():
            gr.Plot(value=chunking_figure(module))

        gr.Markdown("### Retrieval method ablation (3-stage)")
        with gr.Row():
            gr.Plot(value=retrieval_stage_1_figure(module))
        with gr.Row():
            gr.Plot(value=retrieval_stage_2_figure(module))
        with gr.Row():
            gr.Plot(value=retrieval_stage_3_figure(module))


# =============================================================================
# Build the app
# =============================================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="BankMind") as demo:
        gr.Markdown(
            "# 🏦 BankMind\n"
            "_Multi-domain RAG for financial intelligence: compliance and credit._"
        )
        with gr.Tabs():
            with gr.Tab("⚖️ Compliance Q&A"):
                _build_qa_tab("compliance", COMPLIANCE_STRATEGIES, "semantic")
            with gr.Tab("📊 Credit Q&A"):
                _build_qa_tab("credit", CREDIT_STRATEGIES, "semantic")
            with gr.Tab("📈 Compliance Performance"):
                _build_perf_tab("compliance")
            with gr.Tab("📉 Credit Performance"):
                _build_perf_tab("credit")
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
                    **Architecture:** see the project [`README.md`](README.md) for the full work log and design rationale.

                    **Cost notes:**
                    - Pure retrieval (`bm25` / `splade` / `dense` / `hybrid_*` with `none` reranker / `none` transform / generate=off) makes **0 LLM calls** and is free.
                    - Each LLM-using option adds 1 API call:
                      - `query_transform`: hyde / multi_query / prf / step_back. 1 call to rewrite/expand
                      - `reranker=rankgpt`: 1 call per query at rerank time
                      - `generate=on`: 1 call to produce the final answer
                    - Worst case (HyDE/PRF + RankGPT + generate): 3 calls per query, roughly $0.05 each.

                    **Production pipelines (per the evaluation):**
                    - **Compliance**: chunking=semantic, dim=1024, BM25, RankGPT, step_back  (NDCG@10 = 0.834, R@5 = 0.92)
                    - **Credit**: chunking=semantic, dim=128 (or any), BM25, RankGPT  (NDCG@10 = 0.691, R@5 = 0.82)

                    **Current limitations:**
                    - Credit Stage 3 (query transform comparison) was halted to conserve credits. Easy to resume.
                    - MonoT5 + ColBERT rerankers blocked on a transformers v5 compatibility issue. cross_encoder and rankgpt are the verified rerankers.
                    """
                )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True,
               theme=gr.themes.Soft())
