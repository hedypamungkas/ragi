"""Unit tests for ragi.protocols (runtime_checkable contracts) + ragi.log."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from ragi.log import get_logger
from ragi.protocols import (
    ChatClient,
    Chunker,
    EmbeddingClient,
    Parser,
    Reranker,
    Retriever,
    Scorer,
    VectorStore,
)


class TestProtocols:
    """The Protocols are @runtime_checkable: isinstance verifies structural conformance."""

    def test_chunker_conforms(self):
        assert isinstance(SimpleNamespace(chunk=lambda d: []), Chunker)
        assert not isinstance(SimpleNamespace(no_chunk=1), Chunker)

    def test_parser_conforms(self):
        assert isinstance(SimpleNamespace(extract=lambda n, d: ("", {})), Parser)
        assert not isinstance(object(), Parser)

    def test_retriever_conforms(self):
        async def _retrieve(self, query, *, top_k=5, metadata_filter=None):
            return []

        assert isinstance(SimpleNamespace(retrieve=_retrieve), Retriever)
        assert not isinstance(SimpleNamespace(), Retriever)

    def test_embedding_client_conforms(self):
        async def _embed(self, text):
            return []

        async def _embed_batch(self, texts):
            return []

        assert isinstance(SimpleNamespace(embed=_embed, embed_batch=_embed_batch), EmbeddingClient)

    def test_chat_client_conforms(self):
        async def _complete(self, messages):
            return ""

        assert isinstance(SimpleNamespace(complete=_complete), ChatClient)

    def test_vector_store_conforms(self):
        async def _add(self, chunks):
            return None

        async def _search(self, query, *, top_k, filter=None):
            return []

        assert isinstance(SimpleNamespace(add=_add, search=_search, count=0), VectorStore)

    def test_reranker_conforms(self):
        async def _rerank(self, query, documents, *, top_n):
            return []

        assert isinstance(SimpleNamespace(provider="x", rerank=_rerank), Reranker)

    def test_scorer_conforms(self):
        async def _score(self, case, output, context):
            return None

        assert isinstance(SimpleNamespace(name="x", score=_score), Scorer)


class TestLog:
    def test_default_logger_namespace(self):
        assert get_logger().name == "ragi"

    def test_named_logger(self):
        assert get_logger("custom").name == "custom"

    def test_returns_logging_logger(self):
        assert isinstance(get_logger(), logging.Logger)
