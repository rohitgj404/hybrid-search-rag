"""Tests for the hybrid RAG system.

Run with:  python -m pytest
"""

import numpy as np

from rag.chunker import chunk_document, chunk_text
from rag.config import RagConfig
from rag.embeddings import TfidfEmbedder
from rag.hybrid import rrf_fuse
from rag.keyword import BM25Index
from rag.pipeline import RAGPipeline
from rag.reranker import ScoreReranker
from rag.vectorstore import VectorStore


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_chunk_text_respects_size():
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 40  # ~1780 chars
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) >= 3
    assert all(c for c in chunks)
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_overlap_continuity():
    sentence = "The quick brown fox jumps over the lazy dog. "
    chunks = chunk_text(sentence * 40, chunk_size=400, overlap=80)
    # The second chunk should begin with the tail of the first chunk.
    tail = chunks[0][-30:].strip()
    assert tail in chunks[1]


def test_chunk_text_hard_cuts_long_sentence():
    text = "word " * 500  # one giant "sentence", no punctuation
    chunks = chunk_text(text, chunk_size=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_document_annotates_source():
    chunks = chunk_document("Hello world. Second sentence.", source="a.md")
    assert len(chunks) == 1
    assert chunks[0].source == "a.md"
    assert chunks[0].index == 0


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def test_bm25_prefers_relevant_document():
    index = BM25Index()
    index.add_document("Python is a general-purpose programming language.")
    index.add_document("Cats are small furry animals that purr and nap.")
    index.add_document("Python has excellent libraries like numpy and pandas.")
    index.build()
    hits = index.search("python libraries", top_k=2)
    assert hits[0][0] == 2  # the numpy/pandas doc
    assert hits[0][1] > 0.0


def test_bm25_returns_empty_for_no_overlap():
    index = BM25Index()
    index.add_document("Cats purr.")
    index.build()
    assert index.search("python") == []


def test_bm25_tokenization_is_lowercase():
    index = BM25Index()
    index.add_document("OpenAI released GPT-4 in 2023.")
    index.build()
    hits = index.search("openai gpt 4")
    assert hits and hits[0][0] == 0


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------
def test_rrf_fusion_ranks_present_in_both_lists_first():
    a = [(0, 0.9), (1, 0.8)]
    b = [(1, 0.7), (2, 0.6)]
    fused = rrf_fuse([a, b], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == 1  # rank 2 in both lists
    assert set(ids) == {0, 1, 2}


def test_rrf_fusion_empty_input():
    assert rrf_fuse([[], []]) == []


# ---------------------------------------------------------------------------
# TF-IDF embeddings + vector store
# ---------------------------------------------------------------------------
def test_tfidf_cosine_search_finds_similar():
    embedder = TfidfEmbedder()
    corpus = [
        "The cat sat on the mat.",
        "Dogs love to run in the park.",
        "Machine learning models are trained on data.",
    ]
    embedder.fit(corpus)
    store = VectorStore(embedder.dim)
    store.add(embedder.embed(corpus))
    query = embedder.embed(["a cat on the mat"])[0]
    hits = store.search(query, top_k=2)
    assert hits[0][0] == 0
    assert hits[0][1] > 0.5


def test_tfidf_embedding_is_normalized():
    embedder = TfidfEmbedder()
    embedder.fit(["one two three four five"])
    vec = embedder.embed(["one two"])[0]
    assert np.isclose(np.linalg.norm(vec), 1.0)


# ---------------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------------
def test_score_reranker_normalizes_and_interpolates():
    reranker = ScoreReranker(alpha=0.5)
    scores = reranker.rerank(
        "q",
        ["a", "b"],
        vector_scores=[10.0, 20.0],
        keyword_scores=[0.0, 5.0],
    )
    assert abs(scores[0]) < 1e-9
    assert abs(scores[1] - 1.0) < 1e-9
    assert scores[1] > scores[0]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def test_pipeline_search_finds_keyword_match():
    config = RagConfig(embedder="tfidf", reranker="score", top_k_final=3)
    pipeline = RAGPipeline(config)
    pipeline.add_text(
        "Connection timeout errors happen when the firewall blocks port 443.",
        source="troubleshooting.md",
    )
    pipeline.add_text(
        "The server returns a 429 rate limit error above 100 requests per minute.",
        source="troubleshooting.md",
    )
    pipeline.build_index()
    results = pipeline.search("how do I fix connection timeout errors")
    assert results
    assert "timeout" in results[0].chunk.text.lower()


def test_pipeline_query_generates_answer(monkeypatch):
    config = RagConfig(embedder="tfidf", reranker="score")
    pipeline = RAGPipeline(config)
    pipeline.add_text(
        "Hybrid retrieval combines vector embeddings with BM25 keyword search.",
        source="readme.md",
    )
    pipeline.build_index()

    import rag.pipeline as pipeline_module

    captured = {}

    class FakeLLM:
        def __init__(self, config):
            captured["config"] = config

        def complete(self, messages, temperature=None, max_tokens=None):
            captured["messages"] = messages
            return "Hybrid search fuses vector and keyword results. [Source 1]"

    monkeypatch.setattr(pipeline_module, "LLMClient", FakeLLM)

    result = pipeline.query("how does hybrid retrieval work")
    assert result.answer.startswith("Hybrid search")
    assert len(result.chunks) == 1
    assert captured["messages"][0]["role"] == "system"
    assert "Context" in captured["messages"][1]["content"]


def test_save_load_roundtrip(tmp_path):
    index_dir = str(tmp_path / "idx")
    config = RagConfig(embedder="tfidf", reranker="score", index_dir=index_dir)
    pipeline = RAGPipeline(config)
    pipeline.add_text("Alpha bravo charlie delta.", source="a.md")
    pipeline.build_index()
    pipeline.save()

    restored = RAGPipeline(config)
    restored.load()
    assert restored.is_ready
    results = restored.search("bravo charlie")
    assert results[0].chunk.source == "a.md"


def test_search_before_index_raises():
    pipeline = RAGPipeline(RagConfig(embedder="tfidf", reranker="score"))
    try:
        pipeline.search("anything")
    except RuntimeError as exc:
        assert "Index is empty" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
