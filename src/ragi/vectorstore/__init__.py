"""ragi/vectorstore -- pluggable VectorStore backends + ``build_store`` factory.

``InMemoryVectorStore`` is always available (pure stdlib). External backends (faiss/chroma/
pgvector) are import-gated behind their extras and constructed lazily by :func:`build_store`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragi.vectorstore.memory import InMemoryVectorStore

if TYPE_CHECKING:
    from ragi.protocols import EmbeddingClient

__all__ = ["InMemoryVectorStore", "build_store"]


def build_store(conf: dict[str, Any], embedder: EmbeddingClient | None = None):
    """Build a VectorStore from config. ``backend`` defaults to ``memory``.

    Config shape::

        vectorstore: {backend: memory|faiss|chroma|pgvector, ...}
    """
    backend = (conf.get("backend") or "memory").lower()
    if backend == "memory":
        return InMemoryVectorStore(embedder=embedder)
    if backend == "faiss":
        from ragi.vectorstore.faiss import FaissVectorStore

        return FaissVectorStore(embedder=embedder, dim=conf.get("dim"))
    if backend == "chroma":
        from ragi.vectorstore.chroma import ChromaVectorStore

        return ChromaVectorStore(embedder=embedder, collection=conf.get("collection", "ragi"), path=conf.get("path"))
    if backend == "pgvector":
        from ragi.vectorstore.pgvector import PgvectorStore

        return PgvectorStore(embedder=embedder, dsn=conf.get("dsn"), table=conf.get("table", "ragi_chunks"))
    raise ValueError(f"unknown vectorstore backend {backend!r}; use memory|faiss|chroma|pgvector")
