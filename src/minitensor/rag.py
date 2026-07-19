"""Small local RAG building blocks for educational examples.

This module deliberately avoids network services and external model downloads.
The hashing encoder is deterministic and is useful for demonstrating the
retrieval and reranking contracts of a RAG pipeline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .tensor import Tensor, tensor

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class DocumentChunk:
    """A source document fragment returned by the chunker."""

    chunk_id: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    """A chunk and its query similarity."""

    chunk: DocumentChunk
    score: float


def split_text(
    text: str,
    *,
    source: str = "document",
    max_chars: int = 240,
) -> list[DocumentChunk]:
    """Split Markdown/plain text by paragraphs, then by a character limit."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[DocumentChunk] = []
    for paragraph in paragraphs:
        pieces = [
            paragraph[start : start + max_chars]
            for start in range(0, len(paragraph), max_chars)
        ]
        for piece in pieces:
            chunk_id = f"{Path(source).stem}:{len(chunks)}"
            chunks.append(DocumentChunk(chunk_id, source, piece))
    return chunks


def load_markdown(paths: Iterable[str | Path], *, max_chars: int = 240) -> list[DocumentChunk]:
    """Read local Markdown files and return deterministic chunks."""
    chunks: list[DocumentChunk] = []
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        chunks.extend(
            split_text(
                path.read_text(encoding="utf-8"),
                source=str(path),
                max_chars=max_chars,
            )
        )
    return chunks


class HashingEncoder:
    """A deterministic bag-of-words hashing encoder backed by MiniTensor."""

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.dimension = dimension

    def _token_index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dimension, 1.0 if value & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> Tensor:
        """Encode texts as L2-normalized float vectors."""
        values = np.zeros((len(texts), self.dimension), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in _TOKEN_PATTERN.findall(text.lower()):
                index, sign = self._token_index(token)
                values[row, index] += sign
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.where(norms == 0, 1.0, norms)
        return tensor(values)

    def encode_one(self, text: str) -> Tensor:
        return self.encode([text])


def cosine_similarity(query: Tensor, documents: Tensor) -> Tensor:
    """Compute cosine similarity between one query and document rows."""
    if query.ndim != 1 or documents.ndim != 2 or query.shape[0] != documents.shape[1]:
        raise ValueError("Expected query shape (dim,) and documents shape (count, dim).")
    query_norm = (query * query).sum() ** 0.5
    document_norms = (documents * documents).sum(axis=1) ** 0.5
    denominator = document_norms * query_norm
    return (documents @ query.reshape(-1, 1)).reshape(-1) / (denominator + 1e-12)


def retrieve(
    query: str,
    chunks: Sequence[DocumentChunk],
    encoder: HashingEncoder,
    *,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """Retrieve the top-k chunks by in-memory cosine similarity."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not chunks:
        return []
    query_vector = encoder.encode_one(query)[0]
    document_vectors = encoder.encode([chunk.text for chunk in chunks])
    scores = cosine_similarity(query_vector, document_vectors).data
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [RetrievalResult(chunks[index], float(scores[index])) for index in indices]


__all__ = [
    "DocumentChunk",
    "HashingEncoder",
    "RetrievalResult",
    "cosine_similarity",
    "load_markdown",
    "retrieve",
    "split_text",
]
