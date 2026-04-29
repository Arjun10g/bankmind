"""PCA eigenstructure analysis — the project's novel-contribution piece.

Hypothesis (per CLAUDE.md): regulatory text and financial narrative text have
different *intrinsic dimensionalities*. Regulatory language is more formulaic
and repetitive, so its PCA elbow should appear at a lower dimension than
credit-narrative text. If true, this justifies smaller embedding dims for the
compliance module — meaningful storage + latency savings with negligible
accuracy loss.

Pipeline:
  1. Fit full-rank PCA on the (n_chunks × 1024) embedding matrix
  2. Detect the elbow via THREE complementary methods:
       a. Kneedle algorithm on cumulative-variance curve
       b. Second-derivative inflection of eigenvalue spectrum
       c. 95%-explained-variance threshold
  3. Snap each elbow to the nearest Matryoshka dim for fair side-by-side
     comparison (Matryoshka vs PCA at the same dim)
  4. Save eigenvalues + cumulative variance + elbows to JSON
  5. Persist the fitted PCA transform so query-time embeddings can be
     projected into the same space later
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from kneed import KneeLocator
from sklearn.decomposition import PCA

MATRYOSHKA_DIMS: tuple[int, ...] = (128, 256, 512, 768, 1024)


def _snap_to_matryoshka(dim: int) -> int:
    return min(MATRYOSHKA_DIMS, key=lambda x: abs(x - dim))


@dataclass
class PCAResult:
    module: str
    source_strategy: str               # which collection's chunks we fit PCA on
    n_embeddings: int
    full_dim: int
    elbow_kneedle: int
    elbow_kneedle_snapped: int
    elbow_second_deriv: int
    elbow_95pct: int
    cumulative_variance_at_128: float
    cumulative_variance_at_256: float
    cumulative_variance_at_512: float
    cumulative_variance_at_768: float
    cumulative_variance_at_1024: float
    # First N entries of each curve — keep small enough to inline in JSON
    explained_variance_ratios: list[float] = field(default_factory=list)
    eigenvalues: list[float] = field(default_factory=list)
    cumulative_variance: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_pca(embeddings: np.ndarray, *, module: str, source_strategy: str,
            curve_max_points: int = 1024) -> tuple[PCA, PCAResult]:
    """Fit PCA + summarize. Returns (fitted_pca, result_struct)."""
    n, d = embeddings.shape
    if n < 2:
        raise ValueError(f"need ≥2 embeddings to fit PCA, got {n}")

    n_components = min(d, n - 1)
    pca = PCA(n_components=n_components, svd_solver="auto")
    pca.fit(embeddings)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    eig = pca.explained_variance_

    # 1) Kneedle on cumulative variance curve
    dims = np.arange(1, len(cum) + 1)
    try:
        kneedle = KneeLocator(
            dims.tolist(), cum.tolist(),
            curve="concave", direction="increasing", S=1.0,
        )
        elbow_kneedle = int(kneedle.knee) if kneedle.knee is not None else int(np.argmax(cum >= 0.9)) + 1
    except Exception:
        elbow_kneedle = int(np.searchsorted(cum, 0.9)) + 1

    # 2) Second-derivative inflection of eigenvalue spectrum
    if len(eig) >= 3:
        sd = np.diff(np.diff(eig))
        elbow_second_deriv = int(np.argmin(sd)) + 2  # +2 because two diffs lose 2 indices
    else:
        elbow_second_deriv = 1

    # 3) 95% variance threshold
    elbow_95 = int(np.searchsorted(cum, 0.95)) + 1

    def _at(target_dim: int) -> float:
        idx = min(target_dim, len(cum)) - 1
        return float(cum[idx])

    # Truncate the long lists for JSON-inlining
    keep = min(curve_max_points, len(cum))
    result = PCAResult(
        module=module,
        source_strategy=source_strategy,
        n_embeddings=int(n),
        full_dim=int(d),
        elbow_kneedle=int(elbow_kneedle),
        elbow_kneedle_snapped=int(_snap_to_matryoshka(elbow_kneedle)),
        elbow_second_deriv=int(elbow_second_deriv),
        elbow_95pct=int(elbow_95),
        cumulative_variance_at_128=_at(128),
        cumulative_variance_at_256=_at(256),
        cumulative_variance_at_512=_at(512),
        cumulative_variance_at_768=_at(768),
        cumulative_variance_at_1024=_at(1024),
        explained_variance_ratios=evr[:keep].tolist(),
        eigenvalues=eig[:keep].tolist(),
        cumulative_variance=cum[:keep].tolist(),
    )
    return pca, result


def project(pca: PCA, embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Project embeddings into the first n_components PCA directions, then L2-normalize."""
    out = pca.transform(embeddings)[:, :n_components]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def save(pca: PCA, result: PCAResult, *, model_path: Path, json_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, str(model_path))
    json_path.write_text(json.dumps(result.to_dict(), indent=2))


def load(model_path: Path) -> PCA:
    return joblib.load(str(model_path))
