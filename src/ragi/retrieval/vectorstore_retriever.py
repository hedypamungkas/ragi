"""ragi/retrieval/vectorstore_retriever -- Retriever over a pluggable VectorStore.

The scale path. Unlike :class:`SemanticRetriever` (in-memory cosine via the process embedding
cache), this delegates indexing + ANN search to a :class:`VectorStore` backend
(memory/faiss/chroma/pgvector). Built specially by ``build_pipeline`` (it needs a store
constructed from the ``vectorstore:`` config, so it is NOT registered via the standard
registry path) -- see ``registry.build_pipeline``'s ``vectorstore`` branch.

Indexes lazily on the first ``retrieve()`` (keeps ``build_pipeline`` sync); the store embeds
the corpus once via ``embed_batch``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ragi.retrieval.retriever import BaseRetriever
from ragi.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from ragi.protocols import EmbeddingClient

_logger = logging.getLogger(__name__)


class VectorStoreRetriever(BaseRetriever):
    """Lazily index chunks into a VectorStore; embed the query per call."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingClient | None = None,
        store: Any = None,
    ):
        self._chunks = chunks
        self._embedder = embedder
        self._store = store
        self._indexed = False

    async def _ensure_indexed(self) -> None:
        if not self._indexed:
            if self._store is None:
                raise ValueError(
                    "VectorStoreRetriever needs a store; build one via ragi.vectorstore.build_store(config, embedder)"
                )
            await self._store.add(self._chunks)
            self._indexed = True

    async def retrieve(
        self, query: str, *, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[RetrievalResult]:
        await self._ensure_indexed()
        if self._embedder is None:
            return []
        qv = await self._embedder.embed(query)
        if qv is None:
            return []
        return await self._store.search(qv, top_k=top_k, filter=metadata_filter)
