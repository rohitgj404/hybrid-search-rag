"""Split documents into overlapping chunks that respect sentence boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single indexed piece of text."""

    text: str
    source: str
    index: int


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving paragraph boundaries.

    Returns a flat list of sentences with normalized whitespace.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        collapsed = " ".join(para.split())
        for part in re.split(r"(?<=[.!?])\s+", collapsed):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def _overlap_prefix(text: str, overlap: int) -> str:
    """Return the last ~``overlap`` characters of ``text``, trimmed to start
    at a word boundary so the overlap reads naturally."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:].lstrip()
    first_space = tail.find(" ")
    if 0 < first_space <= max(1, overlap // 2):
        tail = tail[first_space + 1 :]
    return tail.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Chunk ``text`` into pieces of roughly ``chunk_size`` characters.

    Chunks never cut through sentences, consecutive chunks share up to
    ``overlap`` characters, and a single sentence longer than ``chunk_size``
    is hard-cut rather than dropped.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        # A sentence longer than chunk_size: hard-cut it.
        while len(sent) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sent[:chunk_size].strip())
            sent = sent[chunk_size:].lstrip()

        if not current:
            current = sent
            continue

        if len(current) + len(sent) + 1 > chunk_size:
            chunks.append(current.strip())
            prefix = _overlap_prefix(current, overlap)
            current = (prefix + " " + sent).strip() if prefix else sent
        else:
            current = current + " " + sent

    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Chunk a document and annotate each chunk with its source."""
    return [
        Chunk(text=t, source=source, index=i)
        for i, t in enumerate(chunk_text(text, chunk_size, overlap))
    ]
