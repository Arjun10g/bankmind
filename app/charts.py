"""Plotly chart builders for the performance dashboard tabs.

Reads pre-computed JSON outputs from `evaluation/results/{module}/...` and
returns Plotly figures Gradio can render via `gr.Plot`.
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# === Dimension sweep =======================================================

def dim_sweep_figure(module: str) -> go.Figure:
    data = _safe_load(RESULTS / module / "dimension_sweep.json")
    fig = go.Figure()
    if not data:
        fig.add_annotation(text=f"No dimension_sweep.json for {module}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    dims = sorted(int(k) for k in data.keys())
    ndcg = [data[str(d)]["ndcg"] for d in dims]
    recall_5 = [data[str(d)]["recall_at_5"] for d in dims]
    track_b = [data[str(d)].get("track_b_composite", 0) or 0 for d in dims]
    p95 = [data[str(d)].get("p95_latency_ms", 0) for d in dims]

    fig.add_trace(go.Scatter(x=dims, y=ndcg, name="NDCG@10", mode="lines+markers",
                             line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=dims, y=recall_5, name="Recall@5", mode="lines+markers",
                             line=dict(color="#2ca02c", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=dims, y=track_b, name="Track-B Composite",
                             mode="lines+markers", line=dict(color="#ff7f0e", width=2)))
    fig.add_trace(go.Scatter(x=dims, y=p95, name="p95 latency (ms)", mode="lines+markers",
                             yaxis="y2", line=dict(color="#d62728", width=1, dash="dash")))
    fig.update_layout(
        title=f"Dimension Sweep — {module.title()} (chunking=semantic, hybrid+RRF)",
        xaxis=dict(title="Embedding dimension", type="category",
                   categoryorder="array", categoryarray=[str(d) for d in dims]),
        yaxis=dict(title="Score (0–1)", range=[0, 1]),
        yaxis2=dict(title="Latency (ms)", overlaying="y", side="right"),
        legend=dict(x=0.02, y=0.98), height=420, margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# === Chunking benchmark ====================================================

def chunking_figure(module: str) -> go.Figure:
    data = _safe_load(RESULTS / module / "chunking_benchmark.json")
    fig = go.Figure()
    if not data:
        fig.add_annotation(text=f"No chunking_benchmark.json for {module}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    strategies = list(data.keys())
    metrics = ["ndcg", "mrr", "recall_at_5", "track_b_composite"]
    titles = ["NDCG@10", "MRR", "Recall@5", "Track-B Composite"]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for metric, title, color in zip(metrics, titles, palette):
        ys = [data[s].get(metric, 0) or 0 for s in strategies]
        fig.add_trace(go.Bar(name=title, x=strategies, y=ys, marker_color=color))
    fig.update_layout(
        title=f"Chunking Benchmark — {module.title()} (dim=512, hybrid+RRF, no rerank)",
        barmode="group", xaxis_title="Chunking strategy", yaxis_title="Score (0–1)",
        yaxis=dict(range=[0, 1]), height=420, margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# === Retrieval benchmark per stage =========================================

def _stage_figure(module: str, stage_key: str, title: str) -> go.Figure:
    data = _safe_load(RESULTS / module / "retrieval_benchmark.json")
    fig = go.Figure()
    if not data or stage_key not in data:
        fig.add_annotation(text=f"No {stage_key} for {module}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    stage = data[stage_key]
    results = stage["results"]
    labels = list(results.keys())
    ndcg = [results[k].get("ndcg", 0) or 0 if "error" not in results[k] else 0 for k in labels]
    recall = [results[k].get("recall_at_5", 0) or 0 if "error" not in results[k] else 0 for k in labels]
    p95 = [results[k].get("p95_latency_ms", 0) if "error" not in results[k] else 0 for k in labels]

    fig.add_trace(go.Bar(name="NDCG@10", x=labels, y=ndcg, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Recall@5", x=labels, y=recall, marker_color="#2ca02c"))
    fig.add_trace(go.Scatter(name="p95 latency (ms)", x=labels, y=p95, mode="lines+markers",
                             yaxis="y2", line=dict(color="#d62728", dash="dash")))
    winner = stage.get("winner", "")
    if winner:
        title = f"{title}  →  winner: {winner}"
    fig.update_layout(
        title=title, barmode="group",
        xaxis_title="", yaxis=dict(title="Score (0–1)", range=[0, 1]),
        yaxis2=dict(title="Latency (ms)", overlaying="y", side="right"),
        height=380, margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def retrieval_stage_1_figure(module: str) -> go.Figure:
    return _stage_figure(module, "stage_1", f"Stage 1: Retrieval Method — {module.title()}")


def retrieval_stage_2_figure(module: str) -> go.Figure:
    return _stage_figure(module, "stage_2", f"Stage 2: Reranker — {module.title()}")


def retrieval_stage_3_figure(module: str) -> go.Figure:
    return _stage_figure(module, "stage_3", f"Stage 3: Query Transform — {module.title()}")


# === PCA eigenstructure ===================================================

def pca_figure(module: str) -> go.Figure:
    data = _safe_load(RESULTS / module / "pca_eigenstructure.json")
    fig = go.Figure()
    if not data:
        fig.add_annotation(text=f"No pca_eigenstructure.json for {module}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    cum = data.get("cumulative_variance", [])
    eig = data.get("eigenvalues", [])
    dims = list(range(1, len(cum) + 1))

    fig.add_trace(go.Scatter(x=dims, y=cum, name="Cumulative variance",
                             mode="lines", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=dims, y=eig[:len(dims)], name="Eigenvalue (λ)",
                             mode="lines", yaxis="y2",
                             line=dict(color="#ff7f0e", width=1)))

    elbow = data.get("elbow_kneedle")
    if elbow:
        fig.add_vline(x=elbow, line_dash="dash", line_color="green",
                      annotation_text=f"Kneedle elbow @ dim {elbow}",
                      annotation_position="top")
    elbow_95 = data.get("elbow_95pct")
    if elbow_95:
        fig.add_vline(x=elbow_95, line_dash="dot", line_color="purple",
                      annotation_text=f"95% var @ dim {elbow_95}",
                      annotation_position="bottom")

    for d in (128, 256, 512, 768, 1024):
        if d < len(cum):
            fig.add_vline(x=d, line_dash="dot", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=f"PCA Eigenstructure — {module.title()} ({data.get('n_embeddings','?')} embeddings)",
        xaxis_title="Component", yaxis=dict(title="Cumulative variance", range=[0, 1.05]),
        yaxis2=dict(title="Eigenvalue (log scale)", overlaying="y", side="right",
                    type="log"),
        height=420, margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
