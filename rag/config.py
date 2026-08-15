"""Central configuration for the RAG pipeline.

Every field has a sensible default and can be overridden through an
environment variable (``RAG_*``) or by constructing :class:`RagConfig`
directly with keyword arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass
class RagConfig:
    """Configuration used across ingestion, retrieval, reranking, and generation."""

    # --- Indexing -------------------------------------------------------
    index_dir: str = field(default_factory=lambda: os.environ.get("RAG_INDEX_DIR", ".rag_index"))
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("RAG_CHUNK_OVERLAP", 150)

    # --- Embeddings -----------------------------------------------------
    # "auto" | "sentence-transformers" | "tfidf"
    embedder: str = os.environ.get("RAG_EMBEDDER", "auto")
    embedding_model: str = os.environ.get(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Reranking ------------------------------------------------------
    # "auto" | "cross-encoder" | "score"
    reranker: str = os.environ.get("RAG_RERANKER", "auto")
    cross_encoder_model: str = os.environ.get(
        "RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    # Weight given to the vector score vs. the keyword score when the
    # "score" reranker is used (alpha = 1.0 is pure semantic, 0.0 pure keyword).
    alpha: float = _env_float("RAG_ALPHA", 0.5)

    # --- Retrieval ------------------------------------------------------
    # How many candidates each retriever returns before fusion/reranking.
    top_k_hybrid: int = _env_int("RAG_TOP_K_HYBRID", 20)
    # How many chunks are returned after reranking.
    top_k_final: int = _env_int("RAG_TOP_K_FINAL", 5)
    # Smoothing constant for Reciprocal Rank Fusion.
    rrf_k: int = _env_int("RAG_RRF_K", 60)

    # --- Generation -----------------------------------------------------
    llm_model: str = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")
    # Any OpenAI-compatible endpoint. Examples:
    #   OpenAI:          https://api.openai.com/v1
    #   Ollama:          http://localhost:11434/v1
    #   LM Studio:       http://localhost:1234/v1
    #   vLLM:            http://localhost:8000/v1
    llm_base_url: str | None = os.environ.get("RAG_LLM_BASE_URL")
    llm_api_key: str | None = os.environ.get("RAG_LLM_API_KEY")
    temperature: float = _env_float("RAG_TEMPERATURE", 0.2)
    max_tokens: int = _env_int("RAG_MAX_TOKENS", 512)
