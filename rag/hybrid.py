"""Fusion of multiple ranked lists — the heart of hybrid search.

Reciprocal Rank Fusion (RRF) combines the *rankings* produced by the vector
store and the BM25 index instead of their raw scores, which sidesteps the
problem that the two score distributions are not comparable. See the paper
"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
Methods" (Cormack, Clarke, Buettcher 2009).
"""

from __future__ import annotations


def rrf_fuse(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse several ranked lists of ``(doc_id, score)`` into one ranking.

    The RRF score of a document is the sum over all lists of
    ``1 / (k + rank)`` where ``rank`` is 1-based. Only documents present in
    at least one list appear in the result.

    Returns a list of ``(doc_id, rrf_score)`` sorted descending by score.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
