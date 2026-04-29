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
        gr.Markdown(f"### {module.title()} chat: ask follow-ups, the assistant remembers context")
        gr.Markdown(
            "Each turn runs retrieval over the {0} corpus and an LLM generates the answer. "
            "Follow-ups are auto-rewritten into standalone queries before retrieval. "
            "Two LLM calls per follow-up turn (rewrite + generate); ~$0.01 to $0.02 each. "
            "Toggle 'Generate answer' off in pipeline configuration to make queries free."
            .format(module)
        )

        chatbot = gr.Chatbot(
            label=f"{module.title()} conversation",
            height=420,
        )

        # Two state objects: the chat history pairs (for the rewriter) and the
        # last QueryResult (for the chunk + guardrail panels).
        history_state = gr.State([])               # list[(user, assistant)]
        last_result_state = gr.State(None)         # QueryResult | None

        with gr.Row():
            with gr.Column(scale=4):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder=(
                        "Ask anything about the compliance corpus, then ask follow-ups..."
                        if module == "compliance"
                        else "Ask anything about the credit filings corpus, then ask follow-ups..."
                    ),
                    lines=2,
                )
            with gr.Column(scale=1, min_width=120):
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("New conversation", variant="secondary")

        with gr.Accordion("💡 Sample questions (click to load into the input)", open=True):
            gr.Examples(
                examples=[[ex] for ex in examples],
                inputs=[user_input],
                label="",
                examples_per_page=10,
            )

        with gr.Accordion("📚 What's in the corpus?", open=False):
            gr.Markdown(sources_md)

        with gr.Accordion("⚙️ Pipeline configuration", open=False):
            gr.Markdown(
                "Defaults are the production winners from the evaluation (see Performance tabs). "
                "All settings here apply to the **next** chat turn."
            )
            with gr.Row():
                strategy = gr.Radio(strategies, value=default_strategy, label="Chunking strategy")
                dim = gr.Radio([str(d) for d in DIMS], value="1024", label="Embedding dim")
            with gr.Row():
                method = gr.Dropdown(RETRIEVAL_METHODS, value="bm25",
                                     label="Retrieval method")
                reranker = gr.Dropdown(RERANKERS, value="cross_encoder", label="Reranker")
                transform = gr.Dropdown(TRANSFORMS, value="none", label="Query transform")
            with gr.Row():
                top_k = gr.Slider(5, 30, value=20, step=1, label="Retrieval top_k")
                final_k = gr.Slider(3, 15, value=8, step=1, label="Passages sent to LLM")
                generate = gr.Checkbox(value=True, label="Generate answer (LLM call, paid)")
            with gr.Row():
                max_tokens = gr.Slider(500, 4000, value=2000, step=100,
                                       label="Max answer tokens")

        with gr.Row():
            timings_md = gr.Markdown(value="")
            config_md = gr.Markdown(value="")

        with gr.Accordion("🛡️ Guardrails (last turn)", open=True):
            guardrails_md = gr.Markdown(value="_(send a message to see guardrails)_")

        with gr.Accordion("📄 Retrieved passages (last turn)", open=False):
            chunks_md = gr.Markdown(value="_(send a message to see retrieved passages)_")

    # ---- Chat handlers ----------------------------------------------------

    def _on_send(user_msg, hist_pairs, strat, d, m, r, t, k, fk, gen, mx, chat_msgs):
        """Append user msg, run pipeline, append assistant msg.

        gr.Chatbot defaults to the messages format on modern Gradio: a list of
        {"role": "user"/"assistant", "content": "..."} dicts. hist_pairs is our
        parallel (user, assistant) tuple list used by the LLM rewriter.
        """
        user_msg = (user_msg or "").strip()
        if not user_msg:
            return (chat_msgs or []), hist_pairs, gr.update(), "", "", "_(empty input)_", "_(no chunks)_", None

        result = run_query(
            query=user_msg,
            module=module,
            chunk_strategy=strat,
            embedding_dim=int(d),
            retrieval_method=m,
            reranker=r,
            query_transform=t,
            top_k=int(k),
            final_k=int(fk),
            generate_answer=bool(gen),
            chat_history=hist_pairs or [],
            max_answer_tokens=int(mx),
        )

        assistant_msg = result.answer if result.answer else "_(generation is off in pipeline configuration)_"

        new_chat_msgs = (chat_msgs or []) + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        new_pairs = (hist_pairs or []) + [(user_msg, assistant_msg)]

        config_line = (
            f"`{result.config_summary}`  ·  query_id=`{result.query_id or '—'}`"
        )
        if result.rewritten_query and result.rewritten_query != user_msg:
            config_line += f"\n\n_Follow-up rewritten as:_ `{result.rewritten_query}`"

        return (
            new_chat_msgs,
            new_pairs,
            gr.update(value=""),                  # clear input
            _format_timings(result.timings),
            config_line,
            _format_guardrails(result.guardrail_report),
            _format_chunks(result.chunks),
            result,
        )

    def _on_clear():
        return [], [], None, "", "", "_(send a message to see guardrails)_", "_(send a message to see retrieved passages)_"

    send_inputs = [user_input, history_state,
                   strategy, dim, method, reranker, transform,
                   top_k, final_k, generate, max_tokens, chatbot]
    send_outputs = [chatbot, history_state, user_input,
                    timings_md, config_md, guardrails_md, chunks_md, last_result_state]

    send_btn.click(_on_send, inputs=send_inputs, outputs=send_outputs)
    user_input.submit(_on_send, inputs=send_inputs, outputs=send_outputs)
    clear_btn.click(
        _on_clear, inputs=[],
        outputs=[chatbot, history_state, last_result_state,
                 timings_md, config_md, guardrails_md, chunks_md],
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

                    **How chat works:**
                    - Each turn the latest user message is rewritten into a standalone retrieval query (1 LLM call) using the prior conversation as context. This makes follow-ups like "tell me more about that" actually retrieve the right passages.
                    - Retrieval runs on the rewritten query (free).
                    - Optional reranker (cross-encoder is local/free; rankgpt costs an extra LLM call).
                    - Generation produces the answer using the full chat history plus the retrieved passages (1 LLM call).
                    - First turn: 1 LLM call (~$0.005). Follow-up turns: 2 LLM calls (~$0.01).

                    **Production pipelines (per the evaluation):**
                    - **Compliance**: chunking=semantic, dim=1024, BM25, cross-encoder reranker  (NDCG@10 ≈ 0.789 with cross-encoder, 0.811 with RankGPT)
                    - **Credit**: chunking=semantic, dim=128, BM25  (NDCG@10 ≈ 0.688)

                    **Free path**: untick "Generate answer" in Pipeline Configuration. Retrieval, reranking with cross-encoder, and the guardrail panel all run with zero LLM cost.

                    **Current limitations:**
                    - Credit Stage 3 of the retrieval benchmark (query transform comparison) was halted during evaluation to conserve credits. Easy to resume.
                    - MonoT5 + ColBERT rerankers are blocked on a transformers v5 compatibility issue. cross_encoder and rankgpt are the verified rerankers.
                    """
                )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True,
               theme=gr.themes.Soft())
