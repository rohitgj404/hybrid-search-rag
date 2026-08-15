"""Embedding backends for semantic vector search.

Two backends are provided:

* :class:`SentenceTransformerEmbedder` — real neural embeddings from the
  `sentence-transformers` library. Requires ``pip install sentence-transformers``.
* :class:`TfidfEmbedder` — a zero-dependency TF-IDF vectorizer built on numpy.
  It gives genuine cosine-similarity retrieval with no model download, which
  keeps the pipeline runnable in a bare VSCode environment.

:func:`build_embedder` selects a backend. ``"auto"`` prefers
sentence-transformers and silently falls back to TF-IDF when the library or
the model is unavailable (e.g. offline).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from .config import RagConfig

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Interface every embedding backend implements."""

    dim: int

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized vectors."""
        ...


class TfidfEmbedder:
    """Bag-of-words TF-IDF vectors with L2 normalization.

    Vector search against these vectors is cosine similarity over weighted
    term overlap — a solid lexical-semantic baseline that works offline.
    """

    def __init__(self, min_df: int = 1, sublinear_tf: bool = True):
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.dim = 0
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        doc_freqs: Counter = Counter()
        for text in texts:
            doc_freqs.update(set(_tokenize(text)))
        n = len(texts)
        vocab = {
            term: idx
            for idx, (term, df) in enumerate(doc_freqs.items())
            if df >= self.min_df
        }
        self._vocab = vocab
        self._idf = {
            term: math.log((1 + n) / (1 + df)) + 1.0
            for term, df in doc_freqs.items()
            if term in vocab
        }
        self.dim = len(vocab)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.dim == 0:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        rows: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for term, tf in Counter(_tokenize(text)).items():
                idx = self._vocab.get(term)
                if idx is None:
                    continue
                weight = (1 + math.log(tf)) if self.sublinear_tf else float(tf)
                vec[idx] = weight * self._idf[term]
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            rows.append(vec)
        if rows:
            return np.stack(rows)
        return np.zeros((0, self.dim), dtype=np.float32)


class SentenceTransformerEmbedder:
    """Neural embeddings from a sentence-transformers model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, texts: list[str]) -> None:
        # Pre-trained models need no corpus-level fitting.
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )


def build_embedder(config: RagConfig) -> Embedder:
    """Build the configured embedder, with automatic fallback."""
    if config.embedder == "tfidf":
        return TfidfEmbedder()
    if config.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(config.embedding_model)
    # "auto": prefer real embeddings, fall back to TF-IDF on any failure
    # (library missing, offline model download, OOM, ...).
    try:
        return SentenceTransformerEmbedder(config.embedding_model)
    except Exception:
        return TfidfEmbedder()
