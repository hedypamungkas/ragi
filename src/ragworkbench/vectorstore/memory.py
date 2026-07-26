"""ragworkbench/vectorstore/memory -- in-memory cosine VectorStore (pure stdlib).

The always-available backend (no extra). Holds an :class:`EmbeddingClient`; embeds chunks
lazily on ``add``. The cosine math mirrors :mod:`ragworkbench.retrieval.retriever` (dot/norm
with a 1e-10 epsilon). For corpora beyond ~100k chunks use a real ANN backend
(faiss/chroma/pgvector) -- this one is O(n) per query, fine for eval-scale corpora.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ragworkbench.ingest.filters import matches_filter
from ragworkbench.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from ragworkbench.protocols import EmbeddingClient

_METHOD = "vectorstore:memory"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Pure-stdlib cosine VectorStore. Holds an embedder; embeds on ``add``."""

    def __init__(self, embedder: EmbeddingClient | None = None):
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float] | None] = []

    async def add(self, chunks: list[Chunk]) -> None:
        to_embed = [c for c in chunks if c.embedding is None]
        if to_embed and self._embedder is not None:
            vectors = await self._embedder.embed_batch([c.content for c in to_embed])
            for c, v in zip(to_embed, vectors, strict=True):
                c.embedding = v
        for c in chunks:
            self._chunks.append(c)
            self._vectors.append(c.embedding)

    async def search(
        self,
        query,
        *,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        if isinstance(query, str):
            if self._embedder is None:
                return []
            query = await self._embedder.embed(query)
        if query is None or not self._vectors:
            return []
        scored: list[tuple[float, Chunk]] = []
        for chunk, vec in zip(self._chunks, self._vectors, strict=True):
            if vec is None:
                continue
            if filter and not matches_filter(chunk.metadata, filter):
                continue
            scored.append((_cosine(query, vec), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievalResult(chunk=c, score=s, retrieval_method=_METHOD) for s, c in scored[:top_k]]

    @property
    def count(self) -> int:
        return len(self._chunks)
