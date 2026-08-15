"""Entry point for the hybrid RAG system.

Run inside VSCode with F5 (see .vscode/launch.json), or from a terminal:

    python main.py ingest data/sample_docs
    python main.py query "How do I fix connection timeout errors?"
"""

from rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
