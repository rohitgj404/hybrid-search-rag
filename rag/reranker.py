"""Rerankers that refine the fused candidate list.

Two backends:

* :class:`CrossEncoderReranker` — a cross-encoder model scores each
  (query, chunk) pair jointly, which is much more accurate than the
  bi-encoder dot products used in retrieval. Requires
  ``pip install sentence-transformers``.
* :class:`ScoreReranker` — dependency-free fallback that interpolates the
  normalized vector and keyword scores (min-max scaled). It cannot beat a
  cross-encoder but still repairs ranking noise from the fusion step.

:func:`build_reranker` mirrors :func:`rag.embeddings.build_embedder`: ``"auto"`
prefers the cross-encoder and falls back to the score interpolator.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import RagConfig


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        texts: list[str],
        *,
        vector_scores: list[float] | None = None,
        keyword_scores: list[float] | None = None,
    ) -> list[float]:
        """Return one score per text in ``texts`` (higher is better)."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        texts: list[str],
        *,
        vector_scores: list[float] | None = None,
        keyword_scores: list[float] | None = None,
    ) -> list[float]:
        scores = self._model.predict([(query, t) for t in texts])
        return [float(s) for s in np.asarray(scores)]


class ScoreReranker:
    """Interpolate min-max normalized vector and keyword scores."""

    def __init__(self, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def rerank(
        self,
        query: str,
        texts: list[str],
        *,
        vector_scores: list[float] | None = None,
        keyword_scores: list[float] | None = None,
    ) -> list[float]:
        v = self._normalize(np.asarray(vector_scores or [], dtype=np.float64))
        k = self._normalize(np.asarray(keyword_scores or [], dtype=np.float64))
        return list(self.alpha * v + (1.0 - self.alpha) * k)


def build_reranker(config: RagConfig) -> Reranker:
    if config.reranker == "score":
        return ScoreReranker(config.alpha)
    if config.reranker == "cross-encoder":
        return CrossEncoderReranker(config.cross_encoder_model)
    # "auto"
    try:
        return CrossEncoderReranker(config.cross_encoder_model)
    except Exception:
        return ScoreReranker(config.alpha)
