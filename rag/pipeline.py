"""Orchestrates ingestion, hybrid retrieval, reranking, and generation.

The retrieval pipeline is a standard two-stage design:

1. **Candidate generation** — the vector store (semantic) and the BM25 index
   (keyword) each return their top ``top_k_hybrid`` chunks independently.
2. **Fusion** — Reciprocal Rank Fusion merges the two rankings.
3. **Reranking** — the fused candidates are scored jointly against the query
   (cross-encoder, or a score interpolation fallback) and the top
   ``top_k_final`` are kept.
4. **Generation** — the winning chunks are assembled into a context and sent
   to an OpenAI-compatible LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk, chunk_document
from .config import RagConfig
from .embeddings import build_embedder
from .hybrid import rrf_fuse
from .keyword import BM25Index
from .llm import LLMClient
from .reranker import build_reranker
from .vectorstore import VectorStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html", ".json", ".csv", ".py"}


@dataclass
class RetrievedChunk:
    """A chunk that survived retrieval, with its scores."""

    chunk: Chunk
    score: float
    vector_score: float
    keyword_score: float


@dataclass
class RAGResult:
    answer: str
    chunks: list[RetrievedChunk]


class RAGPipeline:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig()
        self.embedder = build_embedder(self.config)
        self.reranker = build_reranker(self.config)
        self._chunks: list[Chunk] = []
        self._vector_store: VectorStore | None = None
        self._bm25 = BM25Index()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return len(self._chunks) > 0

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def add_text(self, text: str, source: str = "memory") -> None:
        for chunk in chunk_document(
            text, source, self.config.chunk_size, self.config.chunk_overlap
        ):
            self._chunks.append(chunk)

    def add_file(self, path: str | Path) -> None:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        self.add_text(text, source=str(p))

    def ingest_directory(self, directory: str | Path) -> int:
        """Ingest every supported file under ``directory`` (recursive).

        Returns the number of files added.
        """
        root = Path(directory)
        count = 0
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                self.add_file(p)
                count += 1
        return count

    def build_index(self) -> None:
        """Embed all chunks and build the BM25 index. Call after ingesting."""
        if not self._chunks:
            raise ValueError("No chunks to index. Add documents first.")
        texts = [c.text for c in self._chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)
        store = VectorStore(self.embedder.dim)
        store.add(vectors)
        self._vector_store = store
        for chunk in self._chunks:
            self._bm25.add_document(chunk.text)
        self._bm25.build()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path | None = None) -> Path:
        """Persist chunk metadata; vectors/BM25 are rebuilt on load."""
        index_dir = Path(directory or self.config.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunks": [
                {"text": c.text, "source": c.source, "index": c.index}
                for c in self._chunks
            ],
        }
        (index_dir / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return index_dir

    def load(self, directory: str | Path | None = None) -> None:
        """Restore a previously saved index."""
        index_dir = Path(directory or self.config.index_dir)
        payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self._chunks = [
            Chunk(text=c["text"], source=c["source"], index=c.get("index", 0))
            for c in payload["chunks"]
        ]
        self.build_index()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Hybrid retrieval: vector + BM25 -> RRF fusion -> reranking."""
        if not self.is_ready or self._vector_store is None:
            raise RuntimeError(
                "Index is empty. Ingest documents and build the index first."
            )
        candidates = top_k or self.config.top_k_hybrid

        # Stage 1: independent candidate generation.
        query_vector = self.embedder.embed([query])[0]
        vector_hits = self._vector_store.search(query_vector, top_k=candidates)
        keyword_hits = self._bm25.search(query, top_k=candidates)

        # Stage 2: Reciprocal Rank Fusion.
        fused = rrf_fuse([vector_hits, keyword_hits], k=self.config.rrf_k)
        candidate_ids = [doc_id for doc_id, _ in fused][:candidates]
        if not candidate_ids:
            return []

        # Stage 3: rerank the fused candidates.
        texts = [self._chunks[i].text for i in candidate_ids]
        vec_scores = {doc_id: s for doc_id, s in vector_hits}
        kw_scores = {doc_id: s for doc_id, s in keyword_hits}
        rerank_scores = self.reranker.rerank(
            query,
            texts,
            vector_scores=[vec_scores.get(i, 0.0) for i in candidate_ids],
            keyword_scores=[kw_scores.get(i, 0.0) for i in candidate_ids],
        )

        order = sorted(
            range(len(candidate_ids)), key=lambda i: rerank_scores[i], reverse=True
        )
        results = [
            RetrievedChunk(
                chunk=self._chunks[candidate_ids[pos]],
                score=float(rerank_scores[pos]),
                vector_score=float(vec_scores.get(candidate_ids[pos], 0.0)),
                keyword_score=float(kw_scores.get(candidate_ids[pos], 0.0)),
            )
            for pos in order
        ]
        return results[: self.config.top_k_final]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def query(self, query: str) -> RAGResult:
        """Retrieve context for ``query`` and generate an answer with the LLM."""
        results = self.search(query)
        if results:
            context = "\n\n".join(
                f"[Source {i}] ({r.chunk.source})\n{r.chunk.text}"
                for i, r in enumerate(results, start=1)
            )
        else:
            context = "No relevant context was retrieved for this question."

        system_prompt = (
            "You are a precise research assistant. Answer the user's question "
            "using ONLY the provided context passages, citing them as "
            "[Source 1], [Source 2], etc. If the context does not contain the "
            "answer, say so explicitly. Do not invent facts."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        client = LLMClient(self.config)
        answer = client.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return RAGResult(answer=answer, chunks=results)
