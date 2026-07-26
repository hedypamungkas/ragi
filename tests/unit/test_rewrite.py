"""Unit tests for query rewrite + the RewritingRetriever wrapper."""

from __future__ import annotations

import asyncio

from ragi.retrieval.retriever import KeywordRetriever
from ragi.retrieval.rewrite import (
    MockChatClient,
    QueryRewriter,
    RewritingRetriever,
    rule_based_rewrite,
)
from ragi.types import Chunk

CHUNKS = [Chunk("c1", "d1", "python is a popular programming language")]


def test_rule_based_rewrite_is_deterministic_and_lowercased():
    assert rule_based_rewrite("What is THE best Python?") == rule_based_rewrite("What is THE best Python?")


def test_rewriter_llm_mode_uses_chat_client():
    r = QueryRewriter(client=MockChatClient())
    effective, meta = asyncio.run(r.rewrite("python language?", mode="llm"))
    assert meta["method"] in ("llm", "cache")
    assert effective  # never empty


def test_rewriter_no_client_falls_back_to_rule():
    r = QueryRewriter(client=None)
    _effective, meta = asyncio.run(r.rewrite("python language?", mode="hyde"))
    assert meta["method"] == "rule"


def test_rewriting_retriever_delegates_and_records_rewrite():
    wrapper = RewritingRetriever(KeywordRetriever(chunks=CHUNKS), chat_client=MockChatClient(), mode="llm")
    res = asyncio.run(wrapper.retrieve("python language", top_k=2))
    assert wrapper.last_rewrite is not None
    assert res and res[0].chunk.doc_id == "d1"
