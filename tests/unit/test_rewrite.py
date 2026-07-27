"""Unit tests for query rewrite + the RewritingRetriever wrapper."""

from __future__ import annotations

import asyncio

import pytest

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
    out = rule_based_rewrite("What is THE best Python?")
    # Deterministic across calls...
    assert out == rule_based_rewrite("What is THE best Python?")
    # ...and the output is always fully lowercased.
    assert out == out.lower()


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


class TestRuleBasedRewrite:
    def test_drops_filler_and_lowercases(self):
        # "what" is a content term here (not in the rewrite stoplist); "is"/"the" drop
        assert rule_based_rewrite("What is THE best Python?") == "what best python"

    def test_all_stopwords_keeps_original(self):
        # every token is a stopword -> fall back to the stripped original (never empty)
        assert rule_based_rewrite("the a an") == "the a an"

    def test_collapses_whitespace(self):
        assert rule_based_rewrite("python   language") == "python language"


class _RaisingChat:
    async def complete(self, messages):  # noqa: ARG002
        raise RuntimeError("llm down")


class _EmptyChat:
    async def complete(self, messages):  # noqa: ARG002
        return "   "


class TestQueryRewriterModes:
    async def test_rule_mode(self):
        _eff, meta = await QueryRewriter().rewrite("what is python?", mode="rule")
        assert meta["method"] == "rule"

    async def test_cache_hit_on_second_call(self):
        r = QueryRewriter(client=MockChatClient(responses=["rewritten"]))
        await r.rewrite("q", mode="llm")
        eff, meta = await r.rewrite("q", mode="llm")
        assert meta["method"] == "cache" and eff == "rewritten"

    async def test_llm_call_failure_falls_back(self):
        r = QueryRewriter(client=_RaisingChat())
        _eff, meta = await r.rewrite("what is python", mode="hyde")
        assert meta["method"] == "rule-fallback"

    async def test_empty_llm_response_falls_back_to_rule(self):
        r = QueryRewriter(client=_EmptyChat())
        eff, meta = await r.rewrite("what is python", mode="llm")
        assert meta["method"] == "rule-fallback"
        assert eff == "what python"  # rule-normalized ("is" dropped, "what" kept)

    async def test_empty_llm_no_fallback_to_raw_uses_raw_query(self):
        r = QueryRewriter(client=_EmptyChat(), config={"fallback_to_raw": False})
        eff, meta = await r.rewrite("what is python", mode="llm")
        assert meta["method"] == "rule-fallback"
        assert eff == "what is python"  # raw query, NOT rule-normalized

    async def test_fifo_eviction_when_cache_full(self):
        r = QueryRewriter(client=MockChatClient(), config={"query_cache_size": 1})
        await r.rewrite("q1", mode="llm")  # fills the 1-slot cache
        await r.rewrite("q2", mode="llm")  # evicts q1
        _eff, meta = await r.rewrite("q1", mode="llm")  # q1 now a miss -> re-rewritten
        assert meta["method"] == "llm"  # not cache -> eviction happened


class TestMockChatClient:
    async def test_responses_mode_pops_in_order(self):
        c = MockChatClient(responses=["a", "b"])
        assert await c.complete([]) == "a"
        assert await c.complete([]) == "b"

    async def test_responses_exhausted_raises(self):
        c = MockChatClient(responses=["one"])
        await c.complete([])
        with pytest.raises(IndexError):
            await c.complete([])

    async def test_default_transform_strips_question_mark(self):
        c = MockChatClient()
        out = await c.complete([{"role": "user", "content": "hello?"}])
        assert out.endswith("simplified") and "?" not in out

    async def test_transform_on_empty_messages(self):
        c = MockChatClient()
        out = await c.complete([])
        assert out.endswith("simplified")

    async def test_custom_transform(self):
        c = MockChatClient(transform=lambda s: s.upper())
        assert await c.complete([{"role": "user", "content": "hi"}]) == "HI"
