"""Unit tests for the cross-encoder rerank stage."""

from __future__ import annotations

import asyncio

import pytest

from ragworkbench.errors import LLMInvalidRequestError
from ragworkbench.retrieval.rerank import CrossEncoderReranker, MockReranker, build_rerank_client
from ragworkbench.retrieval.retriever import KeywordRetriever
from ragworkbench.types import Chunk

CHUNKS = [
    Chunk("c1", "d1", "cats are great pets"),
    Chunk("c2", "d2", "dogs are loyal pets"),
    Chunk("c3", "d3", "the sun is a star"),
]


def test_rerank_reorders_and_stamps():
    wrapped = CrossEncoderReranker(KeywordRetriever(chunks=CHUNKS), MockReranker(), fetch_multiplier=2)
    res = asyncio.run(wrapped.retrieve("dogs loyal", top_k=2))
    assert res[0].chunk.doc_id == "d2"
    assert res[0].retrieval_method.startswith("rerank:mock(")


def test_rerank_failsoft_on_dead_backend():
    class DeadReranker(MockReranker):
        async def rerank(self, query, documents, top_n):  # noqa: ARG002
            return None

    wrapped = CrossEncoderReranker(KeywordRetriever(chunks=CHUNKS), DeadReranker())
    res = asyncio.run(wrapped.retrieve("dogs", top_k=2))
    assert res and "rerank:failed" in res[0].retrieval_method


def test_build_rerank_client_validation():
    assert build_rerank_client(None) is None
    assert build_rerank_client("not a dict") is None
    # jina with no api_key and no env -> None (warned, base used unwrapped)
    assert build_rerank_client({"provider": "jina"}) is None
    # unknown provider -> fail-fast
    with pytest.raises(LLMInvalidRequestError):
        build_rerank_client({"provider": "bogus"})


def test_build_rerank_client_env_fallback(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "envkey")
    backend = build_rerank_client({"provider": "jina"})
    assert backend is not None and backend.provider == "jina"
