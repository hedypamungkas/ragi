"""Unit tests for the cross-encoder rerank stage."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from ragi.errors import LLMInvalidRequestError
from ragi.retrieval.rerank import (
    CohereRerankBackend,
    CrossEncoderReranker,
    JinaRerankBackend,
    LocalBGERerankBackend,
    MockReranker,
    _clamp01,
    _int,
    _parse_rerank_results,
    _sigmoid,
    build_rerank_client,
)
from ragi.retrieval.retriever import KeywordRetriever
from ragi.types import Chunk

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


# --------------------------------------------------------------------------- #
# score helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_clamp01(self):
        assert _clamp01(-0.5) == 0.0
        assert _clamp01(1.5) == 1.0
        assert _clamp01(0.25) == 0.25

    def test_int_coerces(self):
        assert _int(1.9) == 1
        assert _int("3") == 3

    def test_sigmoid_zero_is_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_positive_saturates(self):
        assert _sigmoid(8.0) > 0.99

    def test_sigmoid_negative_saturates(self):
        assert _sigmoid(-8.0) < 0.01


class TestParseRerankResults:
    def test_normal(self):
        out = _parse_rerank_results(
            {"results": [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]}
        )
        assert out == [(1, 0.8), (0, 0.2)]

    def test_clamps_oversize_score(self):
        assert _parse_rerank_results({"results": [{"index": 0, "relevance_score": 5.0}]}) == [(0, 1.0)]

    def test_default_score_when_missing(self):
        assert _parse_rerank_results({"results": [{"index": 0}]}) == [(0, 0.0)]

    def test_skips_malformed_row(self):
        out = _parse_rerank_results(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"relevance_score": 0.9},
                    {"index": "x", "relevance_score": 0.1},
                ]
            }
        )
        assert out == [(0, 0.5)]  # two malformed rows skipped, one good kept

    def test_empty(self):
        assert _parse_rerank_results({}) == []


# --------------------------------------------------------------------------- #
# HTTP backends (fake transport)
# --------------------------------------------------------------------------- #
class _FakeTransport:
    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc
        self.closed = False

    async def post(self, path, body):
        if self._exc:
            raise self._exc
        return self._payload

    async def close(self):
        self.closed = True


class TestJinaBackend:
    async def test_rerank_happy(self):
        b = JinaRerankBackend(api_key="k")
        b._transport = _FakeTransport(
            {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
        )
        assert await b.rerank("q", ["a", "b"], top_n=2) == [(1, 0.9), (0, 0.1)]

    async def test_rerank_failsoft_on_error(self):
        b = JinaRerankBackend(api_key="k")
        b._transport = _FakeTransport(exc=RuntimeError("net down"))
        assert await b.rerank("q", ["a"], top_n=1) is None

    async def test_close(self):
        b = JinaRerankBackend(api_key="k")
        t = _FakeTransport()
        b._transport = t
        await b.close()
        assert t.closed is True


class TestCohereBackend:
    async def test_rerank_happy(self):
        b = CohereRerankBackend(api_key="k")
        b._transport = _FakeTransport({"results": [{"index": 0, "relevance_score": 0.7}]})
        assert await b.rerank("q", ["a"], top_n=1) == [(0, 0.7)]

    async def test_rerank_failsoft_on_error(self):
        b = CohereRerankBackend(api_key="k")
        b._transport = _FakeTransport(exc=ValueError("bad"))
        assert await b.rerank("q", ["a"], top_n=1) is None

    async def test_close(self):
        b = CohereRerankBackend(api_key="k")
        t = _FakeTransport()
        b._transport = t
        await b.close()
        assert t.closed is True


# --------------------------------------------------------------------------- #
# local BGE backend (fake sentence_transformers)
# --------------------------------------------------------------------------- #
def _install_fake_st(monkeypatch, scores=None, exc=None):
    class _Inst:
        def predict(self, pairs):
            if exc:
                raise exc
            return scores if scores is not None else [0.9 - 0.1 * i for i in range(len(pairs))]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=lambda model: _Inst()))


class TestLocalBGEBackend:
    def test_missing_dep_raises(self):
        with pytest.raises(LLMInvalidRequestError, match="rerank-local"):
            LocalBGERerankBackend()

    async def test_rerank_happy(self, monkeypatch):
        _install_fake_st(monkeypatch, scores=[0.9, 0.1])
        b = LocalBGERerankBackend()
        out = await b.rerank("q", ["a", "b"], top_n=2)
        assert out[0][0] == 0  # higher score first
        assert all(0.0 <= s <= 1.0 for _, s in out)

    async def test_rerank_failsoft_on_predict_error(self, monkeypatch):
        _install_fake_st(monkeypatch, exc=RuntimeError("model boom"))
        b = LocalBGERerankBackend()
        assert await b.rerank("q", ["a"], top_n=1) is None


# --------------------------------------------------------------------------- #
# build_rerank_client extra branches
# --------------------------------------------------------------------------- #
class TestBuildRerankClient:
    def test_local_with_fake_dep(self, monkeypatch):
        _install_fake_st(monkeypatch)
        b = build_rerank_client({"provider": "local", "model": "custom-model"})
        assert b.provider == "local"

    def test_cohere_with_key(self):
        b = build_rerank_client({"provider": "cohere", "api_key": "k"})
        assert b.provider == "cohere"

    def test_jina_with_overrides(self):
        b = build_rerank_client(
            {"provider": "jina", "api_key": "k", "model": "x", "base_url": "http://h", "timeout": 5}
        )
        assert b.provider == "jina"


# --------------------------------------------------------------------------- #
# CrossEncoderReranker branches
# --------------------------------------------------------------------------- #
class TestCrossEncoderRerankerBranches:
    async def test_threshold_filters_some(self):
        wrapped = CrossEncoderReranker(KeywordRetriever(chunks=CHUNKS), MockReranker(), score_threshold=0.5)
        res = await wrapped.retrieve("dogs loyal pets", top_k=3)
        # only the all-keywords doc clears 0.5
        assert any(r.chunk.doc_id == "d2" for r in res)
        assert all(r.chunk.doc_id == "d2" for r in res)

    async def test_threshold_filters_all_falls_back(self):
        wrapped = CrossEncoderReranker(KeywordRetriever(chunks=CHUNKS), MockReranker(), score_threshold=1.5)
        res = await wrapped.retrieve("dogs loyal", top_k=2)
        assert len(res) >= 1  # falls back to base order

    async def test_out_of_range_index_skipped(self):
        captured: list[str] = []

        class WeirdBackend(MockReranker):
            async def rerank(self, q, docs, top_n):  # noqa: ARG002
                captured.extend(docs)
                return [(99, 0.9), (0, 0.5)]

        wrapped = CrossEncoderReranker(KeywordRetriever(chunks=CHUNKS), WeirdBackend())
        res = await wrapped.retrieve("dogs loyal", top_k=2)
        # index 99 is out of range -> dropped; only the valid index-0 base result survives.
        assert len(res) == 1
        assert res[0].chunk.content == captured[0]  # the surviving in-range result
        assert res[0].retrieval_method.startswith("rerank:mock(")

    async def test_empty_base_returns_empty(self):
        wrapped = CrossEncoderReranker(KeywordRetriever(chunks=CHUNKS), MockReranker())
        res = await wrapped.retrieve("zzzznomatchxyz", top_k=2)
        assert res == []

    async def test_close_delegates_to_backend(self):
        wrapped = CrossEncoderReranker(KeywordRetriever(chunks=CHUNKS), MockReranker())
        await wrapped.close()  # MockReranker.close is a no-op; must not raise
