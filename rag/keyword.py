"""Okapi BM25 keyword index with a pure-Python implementation.

BM25 provides the lexical leg of the hybrid search: it excels at exact
term matching (product names, error codes, identifiers) where pure vector
search can be fuzzy. It needs no external dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, the same vocabulary used by keyword search."""
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed corpus of documents.

    Documents are added with :meth:`add_document` and the index is finalized
    with :meth:`build` before searching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_term_counts: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._doc_freqs: Counter = Counter()
        self._n = 0
        self._avg_len = 0.0
        self._idf: dict[str, float] = {}

    def add_document(self, text: str) -> None:
        counts = Counter(tokenize(text))
        self._doc_term_counts.append(counts)
        self._doc_lengths.append(sum(counts.values()))
        self._n += 1
        for term in counts:
            self._doc_freqs[term] += 1

    def build(self) -> None:
        """Compute corpus-level statistics (average length, IDF)."""
        n = self._n
        self._avg_len = sum(self._doc_lengths) / max(n, 1)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self._doc_freqs.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs ranked by BM25, best first.

        Documents with no overlapping terms receive a zero score and are
        excluded from the results.
        """
        if not self._doc_term_counts:
            return []
        query_terms = tokenize(query)
        k1, b = self.k1, self.b
        avg_len = self._avg_len or 1.0

        scores: list[float] = []
        for i, counts in enumerate(self._doc_term_counts):
            doc_len = self._doc_lengths[i]
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + k1 * (1 - b + b * doc_len / avg_len)
                score += idf * (tf * (k1 + 1)) / denom
            scores.append(score)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked if scores[i] > 0.0][:top_k]
