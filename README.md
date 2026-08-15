# Hybrid RAG

A self-contained retrieval-augmented generation (RAG) system in Python that
combines **semantic vector search** and **keyword retrieval** with
**reranking**, ready to run and debug from VSCode.

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────────────┐
│  Documents  │──▶│  Chunk + Embed   │──▶│  Semantic vectors (cosine)   │
└─────────────┘   └──────────────────┘   └──────────────┬──────────────┘
                                                        │  top-k each
                                          ┌─────────────▼──────────────┐
                                          │  Reciprocal Rank Fusion    │
                                          │  (RRF)                     │
                                          └─────────────┬──────────────┘
                                                        │  fused candidates
                                          ┌─────────────▼──────────────┐
                                          │  Rerank (cross-encoder /   │
                                          │  score interpolation)      │
                                          └─────────────┬──────────────┘
                                                        │  top-k_final
                                          ┌─────────────▼──────────────┐
                                          │  LLM answer (OpenAI-       │
                                          │  compatible endpoint)      │
                                          └────────────────────────────┘
```

## Features

- **Hybrid retrieval** — a numpy cosine vector store (semantic) and a
  pure-Python Okapi BM25 index (keyword) each return candidates
  independently; **Reciprocal Rank Fusion** merges the two rankings without
  score-calibration headaches.
- **Reranking** — fused candidates are re-scored against the query by a
  cross-encoder model, with a dependency-free score-interpolation fallback.
- **Works in two modes:**
  - *Lightweight* — only `numpy` + `requests`. Vector search uses TF-IDF
    vectors (real cosine similarity, offline, no model download) and
    reranking uses normalized score interpolation.
  - *Full* — `pip install sentence-transformers` unlocks neural embeddings
    (`all-MiniLM-L6-v2`) and cross-encoder reranking
    (`ms-marco-MiniLM-L6-v2`). Everything switches automatically.
- **Any LLM** — generation talks to any OpenAI-compatible endpoint: OpenAI,
  Azure, **Ollama**, **LM Studio**, vLLM, llama.cpp, Jan.
- **VSCode-ready** — debugger launch configs, pytest discovery, and a
  recommended-extensions file are included.

## Project layout

```
.
├── rag/                    # the package
│   ├── config.py           # RagConfig (env-var overridable)
│   ├── chunker.py          # sentence-aware chunking with overlap
│   ├── embeddings.py       # sentence-transformers + TF-IDF fallback
│   ├── vectorstore.py      # numpy cosine vector store
│   ├── keyword.py          # pure-Python Okapi BM25
│   ├── hybrid.py           # Reciprocal Rank Fusion
│   ├── reranker.py         # cross-encoder + score-interpolation fallback
│   ├── llm.py              # OpenAI-compatible chat client
│   ├── pipeline.py         # RAGPipeline: ingest → search → query
│   └── cli.py              # ingest / search / query commands
├── tests/test_rag.py       # pytest suite
├── data/sample_docs/       # sample markdown corpus
├── .vscode/                # settings, launch.json, extensions
└── main.py                 # root entry point (F5-friendly)
```

## Quickstart in VSCode

> Prerequisite: Python 3.10+. If VSCode's Python extension can't find an
> interpreter, install one from [python.org](https://python.org) and tick
> *"Add Python to PATH"* during setup.

**1. Create a virtual environment and install dependencies** — in the
VSCode terminal (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

For real semantic embeddings and cross-encoder reranking, install the full
set instead:

```powershell
pip install -r requirements-full.txt
```

(The first run downloads the embedding/reranker models from Hugging Face.)

**2. Configure the Python interpreter** — VSCode will usually detect
`.venv` automatically. If not, run the *Python: Select Interpreter* command
and choose `.venv\Scripts\python.exe` (matches `.vscode/settings.json`).

**3. Index the sample documents:**

```powershell
python main.py ingest data/sample_docs
```

You should see `Indexed 5 files -> N chunks` and the index saved to
`.rag_index`.

**4. Search without an LLM** (no API key needed):

```powershell
python main.py search "how do I fix connection timeout errors"
```

**5. Ask a question** — configure an LLM and run:

```powershell
python main.py query "how do I fix connection timeout errors"
```

Or hit **F5** with the *"RAG: Query (answer with LLM)"* config — the
debugger pauses on breakpoints in the retrieval and generation code.

### Running the tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

The suite covers chunking, BM25, RRF fusion, TF-IDF vectors, rerankers, and
an end-to-end pipeline run with a mocked LLM.

## Configuring the LLM

`query` needs an OpenAI-compatible endpoint. Set environment variables
(matching the names in `rag/config.py`), or pass `--llm-*` flags:

| Provider   | Command |
|------------|---------|
| OpenAI     | `set RAG_LLM_API_KEY=sk-...` (or `OPENAI_API_KEY`) — defaults to `gpt-4o-mini` |
| Ollama     | `set RAG_LLM_BASE_URL=http://localhost:11434/v1` and `set RAG_LLM_MODEL=llama3.1` |
| LM Studio  | `set RAG_LLM_BASE_URL=http://localhost:1234/v1` |
| vLLM       | `set RAG_LLM_BASE_URL=http://localhost:8000/v1` |

For example, with Ollama running locally:

```powershell
$env:RAG_LLM_BASE_URL = "http://localhost:11434/v1"
$env:RAG_LLM_MODEL = "llama3.1"
python main.py query "what does the hybrid search feature do?"
```

## Configuration reference

Everything is configurable through environment variables (`RAG_*`) or
`rag.config.RagConfig`:

| Variable                | Default                                              | Purpose |
|-------------------------|------------------------------------------------------|---------|
| `RAG_INDEX_DIR`         | `.rag_index`                                         | Where `save()`/`load()` keep the index |
| `RAG_CHUNK_SIZE`        | `800`                                                | Target chunk size in characters |
| `RAG_CHUNK_OVERLAP`     | `150`                                                | Chars shared between consecutive chunks |
| `RAG_EMBEDDER`          | `auto`                                               | `auto` \| `sentence-transformers` \| `tfidf` |
| `RAG_EMBEDDING_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2`             | Embedding model |
| `RAG_RERANKER`          | `auto`                                               | `auto` \| `cross-encoder` \| `score` |
| `RAG_CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2`              | Reranker model |
| `RAG_ALPHA`             | `0.5`                                                | Vector-vs-keyword weight for the score reranker |
| `RAG_TOP_K_HYBRID`      | `20`                                                 | Candidates per retriever before fusion |
| `RAG_TOP_K_FINAL`       | `5`                                                  | Chunks kept after reranking |
| `RAG_RRF_K`             | `60`                                                 | RRF smoothing constant |
| `RAG_LLM_MODEL`         | `gpt-4o-mini`                                        | Chat model name |
| `RAG_LLM_BASE_URL`      | *(OpenAI)*                                           | OpenAI-compatible base URL |
| `RAG_LLM_API_KEY`       | *(none)*                                             | API key |
| `RAG_TEMPERATURE`       | `0.2`                                                | Sampling temperature |
| `RAG_MAX_TOKENS`        | `512`                                                | Max generated tokens |

## How the retrieval pipeline works

1. **Candidate generation.** The query is embedded and cosine-searched
   against the vector store; independently, the same query runs through
   BM25. Each retriever returns its top `top_k_hybrid` chunks.
2. **Fusion.** The two rankings are merged with Reciprocal Rank Fusion:
   each document scores `Σ 1/(k + rank)` across the lists, where `rank` is
   1-based. This compares *positions* rather than raw scores, so the
   incomparable vector/BERT and BM25 scales don't matter.
3. **Reranking.** The fused candidates are scored jointly with the query.
   With the full install this is a cross-encoder (reads query+chunk
   together — much stronger than the retrieval dot products). Without it,
   the fallback interpolates min-max normalized vector and keyword scores
   (`alpha` controls the blend).
4. **Generation.** The top `top_k_final` chunks become a cited context
   block, and an OpenAI-compatible LLM produces the answer.

## Using the library from code

```python
from rag.config import RagConfig
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(RagConfig())          # or RagConfig(embedder="tfidf")
pipeline.ingest_directory("data/sample_docs")
pipeline.build_index()
pipeline.save()                               # reload later with pipeline.load()

for hit in pipeline.search("connection timeout errors"):
    print(hit.score, hit.chunk.source, hit.chunk.text[:80])

result = pipeline.query("How do I fix connection timeouts?")
print(result.answer)
for i, chunk in enumerate(result.chunks, 1):
    print(f"[{i}] {chunk.chunk.source}")
```

## Notes and limitations

- BM25 uses exact lowercased tokens — word forms like *timeout/timeouts*
  don't match each other. Real semantic embeddings (full install) cover
  that gap; without them, phrase queries to match the source wording.
- The vector store keeps everything in memory (numpy). It's great up to
  tens of thousands of chunks; for production scale, swap in FAISS,
  pgvector, or Chroma behind the same `VectorStore` interface.
- The index persists chunk metadata; vectors and BM25 statistics are
  rebuilt on `load()` from the chunk texts.
