"""ragi/embeddings -- embedding clients + shared embedding index cache.

Lifted from ``koboi/rag/retriever.py`` and decoupled: zero ``koboi.*`` imports.
Ships:

- :class:`OpenAIEmbeddingClient` -- OpenAI-compatible ``/embeddings`` adapter over
  :mod:`ragi._internal.http` (fail-soft: returns None on any error so the
  calling retriever degrades to lexical).
- :class:`MockEmbeddingClient` -- deterministic bag-of-words vector for tests
  (no API key, no network).
- The process-wide :data:`_EMBEDDING_CACHE` (+ :func:`clear_embedding_cache` /
  :func:`set_embedding_cache_path`) shared by Semantic/Hybrid retrievers.

Both clients satisfy :class:`ragi.protocols.EmbeddingClient`.
"""

from __future__ import annotations

from ragi.embeddings.cache import (
    _EMBEDDING_CACHE,
    _EmbeddingIndexCache,
    clear_embedding_cache,
    set_embedding_cache_path,
)
from ragi.embeddings.mock import MockEmbeddingClient
from ragi.embeddings.openai import OpenAIEmbeddingClient

__all__ = [
    "OpenAIEmbeddingClient",
    "MockEmbeddingClient",
    "_EmbeddingIndexCache",
    "_EMBEDDING_CACHE",
    "clear_embedding_cache",
    "set_embedding_cache_path",
]
