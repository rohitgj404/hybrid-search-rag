"""Hybrid RAG: semantic vector search + BM25 keyword retrieval + reranking.

The package is dependency-light on purpose: with only ``numpy`` and ``requests``
installed it runs a fully working pipeline (TF-IDF vectors + BM25 + score
reranking). Installing ``sentence-transformers`` unlocks real semantic
embeddings and cross-encoder reranking, which are picked up automatically.
"""

__version__ = "0.1.0"
