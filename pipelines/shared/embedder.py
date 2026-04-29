"""Matryoshka embedding wrapper for `mixedbread-ai/mxbai-embed-large-v1`.

Native Matryoshka heads (per the model's training): [128, 256, 512, 768, 1024].
A single forward pass produces a 1024-dim embedding; lower dims are obtained
by truncating the leading-N components and L2-renormalizing.

Conventions (per mxbai documentation):
  - Documents: embed plain text.
  - Queries: prepend the prompt
        "Represent this sentence for searching relevant passages: "
    Significantly improves retrieval recall — this is what the model was
    contrastively trained against.

Usage:
    embedder = MatryoshkaEmbedder()
    full = embedder.embed_documents(["chunk text 1", "chunk text 2"])  # (n, 1024)
    dim_512 = embedder.truncate(full, 512)                              # (n, 512)
    all_dims = embedder.embed_documents_all_dims(["..."])               # dict[int, (n, dim)]
    q = embedder.embed_queries(["what is the capital ratio?"])          # (1, 1024) with query prompt
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

DIMENSIONS: tuple[int, ...] = (128, 256, 512, 768, 1024)
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

_DEFAULT_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1"
)

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _best_device() -> str:
    """Pick the fastest available backend: CUDA > MPS > CPU.

    Override with EMBEDDING_DEVICE=cpu (or =cuda, =mps) — useful when MPS gets
    into a bad state (Metal compiler service crash after sleep, etc.) and you
    want to force the slow-but-reliable CPU path without code changes.
    """
    forced = os.environ.get("EMBEDDING_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda", "mps"}:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_model(model_name: str = _DEFAULT_MODEL_NAME) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        device = _best_device()
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[model_name]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with divide-by-zero guard."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class MatryoshkaEmbedder:
    """One model load → embeddings at any of the trained Matryoshka dims."""

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = _get_model(model_name)
        # Sanity check: confirm the model produces 1024-dim output we expect.
        # Newer sentence-transformers versions renamed the accessor.
        dim = (self.model.get_embedding_dimension()
               if hasattr(self.model, "get_embedding_dimension")
               else self.model.get_sentence_embedding_dimension())
        if dim != 1024:
            raise RuntimeError(
                f"{model_name} produced {dim}-dim embeddings; expected 1024 "
                f"(Matryoshka dims hard-coded to {DIMENSIONS})"
            )

    # --- core API --------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed documents at full dim (1024). Returned vectors are L2-normalized."""
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll normalize after truncation
        ).astype(np.float32)
        return _l2_normalize(emb)

    def embed_queries(
        self,
        queries: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed queries with the mxbai retrieval prompt prefix."""
        prefixed = [QUERY_PROMPT + q for q in queries]
        return self.embed_documents(prefixed, show_progress=show_progress)

    @staticmethod
    def truncate(embeddings: np.ndarray, dim: int) -> np.ndarray:
        """Truncate to `dim` columns and L2-renormalize.

        Matryoshka property: the first `dim` components are themselves a valid
        embedding at that dimension, after renormalization.
        """
        if dim not in DIMENSIONS:
            raise ValueError(f"dim must be one of {DIMENSIONS}, got {dim}")
        if dim > embeddings.shape[1]:
            raise ValueError(
                f"requested dim {dim} > embedding dim {embeddings.shape[1]}"
            )
        return _l2_normalize(embeddings[:, :dim])

    # --- batch API: all dims in one shot --------------------------------------

    def embed_documents_all_dims(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> dict[int, np.ndarray]:
        """One forward pass → embeddings at every Matryoshka dim."""
        full = self.embed_documents(texts, show_progress=show_progress)
        return {dim: self.truncate(full, dim) for dim in DIMENSIONS}

    def embed_queries_all_dims(
        self,
        queries: list[str],
        *,
        show_progress: bool = False,
    ) -> dict[int, np.ndarray]:
        full = self.embed_queries(queries, show_progress=show_progress)
        return {dim: self.truncate(full, dim) for dim in DIMENSIONS}
