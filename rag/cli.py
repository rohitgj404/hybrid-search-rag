"""Command-line interface for the hybrid RAG system.

Examples:
    python main.py ingest data/sample_docs
    python main.py search "how do I fix connection timeout errors"
    python main.py query "how do I fix connection timeout errors"
"""

from __future__ import annotations

import argparse
import sys

from .config import RagConfig
from .pipeline import RAGPipeline

_INDEX_DIR_DEFAULT = ".rag_index"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand. None means 'use the default'."""
    parser.add_argument(
        "--index-dir",
        default=None,
        help=f"Index directory (default: {_INDEX_DIR_DEFAULT} or $RAG_INDEX_DIR)",
    )
    parser.add_argument(
        "--embedder",
        default=None,
        choices=["auto", "sentence-transformers", "tfidf"],
        help="Embedding backend (default: auto)",
    )
    parser.add_argument(
        "--reranker",
        default=None,
        choices=["auto", "cross-encoder", "score"],
        help="Reranker backend (default: auto)",
    )
    parser.add_argument(
        "--top-k-hybrid",
        type=int,
        default=None,
        help="Candidates per retriever before fusion (default: 20)",
    )
    parser.add_argument(
        "--top-k-final",
        type=int,
        default=None,
        help="Chunks returned after reranking (default: 5)",
    )


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1",
    )
    parser.add_argument("--llm-api-key", default=None, help="API key")


def _config_from_args(args: argparse.Namespace) -> RagConfig:
    """Build a RagConfig, honoring defaults and env vars for unset flags."""
    overrides = {
        name: getattr(args, name)
        for name in (
            "index_dir",
            "embedder",
            "reranker",
            "top_k_hybrid",
            "top_k_final",
            "llm_model",
            "llm_base_url",
            "llm_api_key",
        )
        if getattr(args, name, None) is not None
    }
    return RagConfig(**overrides)


def _cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(_config_from_args(args))
    files = pipeline.ingest_directory(args.directory)
    if files == 0:
        print(f"No supported documents found in {args.directory}", file=sys.stderr)
        return 1
    pipeline.build_index()
    index_dir = pipeline.save()
    print(f"Indexed {files} files -> {pipeline.num_chunks} chunks")
    print(f"Index saved to {index_dir}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(_config_from_args(args))
    pipeline.load()
    results = pipeline.search(args.query)
    if not results:
        print("No results found.")
        return 1
    for r in results:
        snippet = " ".join(r.chunk.text.split())[:180]
        print(
            f"{r.score:+.4f}  vec={r.vector_score:+.3f}  "
            f"kw={r.keyword_score:+.3f}  [{r.chunk.source}]"
        )
        print(f"    {snippet}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(_config_from_args(args))
    pipeline.load()
    result = pipeline.query(args.query)
    print(result.answer)
    print()
    if result.chunks:
        print("Sources:")
        for i, r in enumerate(result.chunks, start=1):
            print(f"  [{i}] {r.chunk.source} (score {r.score:+.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Hybrid RAG: semantic vector search + BM25 + reranking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index a directory of documents")
    ingest.add_argument("directory", help="Directory to scan (recursively)")
    _add_common_args(ingest)
    ingest.set_defaults(func=_cmd_ingest)

    search = sub.add_parser("search", help="Retrieve chunks without generating")
    search.add_argument("query", help="Search query")
    _add_common_args(search)
    search.set_defaults(func=_cmd_search)

    query = sub.add_parser("query", help="Ask a question and get an answer")
    query.add_argument("query", help="Question")
    _add_common_args(query)
    _add_llm_args(query)
    query.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy codepage; UTF-8 keeps the output
    # (e.g. em-dashes in document snippets) readable in VSCode's terminal.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
