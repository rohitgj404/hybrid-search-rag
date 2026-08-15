"""A tiny numpy vector store with cosine-similarity search.

Vectors are assumed to be L2-normalized (both embedding backends guarantee
this), so search is a single matrix multiplication. Good enough for
prototype-to-medium corpora; swap in FAISS / pgvector / Chroma for scale.
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._blocks: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected (n, {self.dim}) vectors, got {vectors.shape}"
            )
        self._blocks.append(vectors)
        self._matrix = None  # invalidate cache

    @property
    def size(self) -> int:
        return sum(b.shape[0] for b in self._blocks)

    def _matrix_view(self) -> np.ndarray:
        if self._matrix is None:
            if self._blocks:
                self._matrix = np.vstack(self._blocks)
            else:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        return self._matrix

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``(row_index, cosine_similarity)`` pairs, best first."""
        matrix = self._matrix_view()
        n = matrix.shape[0]
        if n == 0:
            return []
        top_k = min(top_k, n)
        sims = matrix @ np.asarray(query, dtype=np.float32).reshape(-1)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]
