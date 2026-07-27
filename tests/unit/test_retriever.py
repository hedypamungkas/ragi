"""Unit tests for ragi.retrieval.retriever -- lexical + semantic retriever internals.

Covers the stemmer import-gate, resolve_retriever fallback, keyword/bm25 stopword +
synonym + empty-query + metadata-filter branches, and SemanticRetriever's embedding-cache
+ fail-soft fallback paths. Network/optional deps (Sastrawi) are faked; the shared
process-level embedding cache is cleared per semantic test for isolation.
"""

from __future__ import annotations

import sys
import types

import pytest

from ragi.embeddings.cache import clear_embedding_cache
from ragi.retrieval import retriever as rmod
from ragi.retrieval.retriever import (
    BM25Retriever,
    HybridRetriever,
    KeywordRetriever,
    SemanticRetriever,
    _normalize_stemmer,
    resolve_retriever,
)
from ragi.types import Chunk

CHUNKS = [
    Chunk("c1", "d1", "the cat sat on the mat"),
    Chunk("c2", "d2", "python is a programming language"),
    Chunk("c3", "d3", "dogs are loyal animals"),
]


# --------------------------------------------------------------------------- #
# _normalize_stemmer
# --------------------------------------------------------------------------- #
class _FakeStemmer:
    def stem(self, w):
        return w + "X"


class _FakeFactory:
    def create_stemmer(self):
        return _FakeStemmer()


def _install_fake_sastrawi(monkeypatch):
    pkg = types.ModuleType("Sastrawi")
    sub = types.ModuleType("Sastrawi.Stemmer")
    leaf = types.ModuleType("Sastrawi.Stemmer.StemmerFactory")
    leaf.StemmerFactory = _FakeFactory
    pkg.Stemmer = sub
    monkeypatch.setitem(sys.modules, "Sastrawi", pkg)
    monkeypatch.setitem(sys.modules, "Sastrawi.Stemmer", sub)
    monkeypatch.setitem(sys.modules, "Sastrawi.Stemmer.StemmerFactory", leaf)


class TestNormalizeStemmer:
    def test_none_and_false_disable(self):
        assert _normalize_stemmer(None) is None
        assert _normalize_stemmer(False) is None

    def test_id_without_extra_warns_and_disables(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "Sastrawi", raising=False)
        assert _normalize_stemmer("id") is None

    def test_unknown_value_disables(self):
        assert _normalize_stemmer("fr") is None
        assert _normalize_stemmer(True) is None  # True is NOT a valid alias

    def test_id_with_extra_returns_stem_fn(self, monkeypatch):
        _install_fake_sastrawi(monkeypatch)
        fn = _normalize_stemmer("id")
        assert fn is not None
        assert fn("running") == "runningX"


# --------------------------------------------------------------------------- #
# resolve_retriever
# --------------------------------------------------------------------------- #
class TestResolveRetriever:
    def test_known_name(self):
        assert isinstance(resolve_retriever("bm25", CHUNKS), BM25Retriever)

    def test_unknown_falls_back_to_keyword(self):
        assert isinstance(resolve_retriever("nope", CHUNKS), KeywordRetriever)

    def test_no_retrievers_registered_raises(self, monkeypatch):
        monkeypatch.setattr(rmod.retriever_registry, "get", lambda name: None)
        with pytest.raises(ValueError, match="No retrievers registered"):
            resolve_retriever("keyword", CHUNKS)


# --------------------------------------------------------------------------- #
# KeywordRetriever branches
# --------------------------------------------------------------------------- #
class TestKeywordRetriever:
    async def test_empty_query_after_stopword_removal(self):
        r = KeywordRetriever(CHUNKS, stopwords=True)
        assert await r.retrieve("the a an") == []

    async def test_synonym_expansion_matches(self):
        r = KeywordRetriever(CHUNKS, synonyms={"cat": ["feline"]})
        res = await r.retrieve("cat")
        assert res and res[0].chunk.id == "c1"

    async def test_empty_content_chunk_scores_zero_and_is_excluded(self):
        chunks = [Chunk("empty", "de", ""), Chunk("c2", "d2", "python programming")]
        r = KeywordRetriever(chunks)
        res = await r.retrieve("python")
        assert all(c.chunk.id != "empty" for c in res)

    async def test_metadata_filter_scopes(self):
        tagged = [
            Chunk("c1", "d1", "refund policy", metadata={"src": "a"}),
            Chunk("c2", "d2", "refund terms", metadata={"src": "b"}),
        ]
        r = KeywordRetriever(tagged)
        res = await r.retrieve("refund", metadata_filter={"src": "a"})
        assert res and all(c.chunk.id == "c1" for c in res)


# --------------------------------------------------------------------------- #
# BM25Retriever branches
# --------------------------------------------------------------------------- #
class TestBM25Retriever:
    async def test_stopwords_and_synonyms_applied(self):
        r = BM25Retriever(CHUNKS, stopwords=True, synonyms={"cat": ["feline"]})
        res = await r.retrieve("cat the")  # 'the' dropped, 'cat' expanded
        assert res and res[0].chunk.id == "c1"

    async def test_empty_query_returns_empty(self):
        r = BM25Retriever(CHUNKS, stopwords=True)
        assert await r.retrieve("the a an") == []

    async def test_metadata_filter_excludes(self):
        tagged = [
            Chunk("c1", "d1", "refund policy", metadata={"src": "a"}),
            Chunk("c2", "d2", "refund terms", metadata={"src": "b"}),
        ]
        r = BM25Retriever(tagged)
        res = await r.retrieve("refund", metadata_filter={"src": "a"})
        assert res and all(c.chunk.id == "c1" for c in res)


# --------------------------------------------------------------------------- #
# SemanticRetriever branches
# --------------------------------------------------------------------------- #
class _VecEmbedder:
    async def embed(self, text):
        return [float(len(text)), 1.0, 0.5]


class _NoneEmbedder:
    async def embed(self, text):  # noqa: ARG002
        return None


class _QueryNoneEmbedder:
    def __init__(self, none_for):
        self._none_for = none_for

    async def embed(self, text):
        return None if text == self._none_for else [float(len(text)), 1.0]


@pytest.fixture(autouse=True)
def _clean_embedding_cache():
    clear_embedding_cache()
    yield
    clear_embedding_cache()


class TestSemanticRetriever:
    async def test_no_embedder_falls_back_to_keyword(self):
        r = SemanticRetriever(CHUNKS)  # no embedder
        res = await r.retrieve("python")
        assert res and all(x.retrieval_method == "semantic (fallback to keyword)" for x in res)

    async def test_corpus_build_failure_falls_back(self):
        r = SemanticRetriever(CHUNKS, embedder=_NoneEmbedder())
        res = await r.retrieve("python")
        assert res and all("fallback" in x.retrieval_method for x in res)

    async def test_query_embedding_none_falls_back(self):
        # corpus builds OK, but the query embed returns None -> per-query fallback
        r = SemanticRetriever(CHUNKS, embedder=_QueryNoneEmbedder(none_for="python"))
        res = await r.retrieve("python")
        assert res and all("fallback" in x.retrieval_method for x in res)

    async def test_happy_semantic_pass(self):
        r = SemanticRetriever(CHUNKS, embedder=_VecEmbedder())
        res = await r.retrieve("python")
        assert res and all(x.retrieval_method == "semantic" for x in res)

    async def test_query_embedding_cached_on_repeat(self):
        calls = {"n": 0}

        class _Counting:
            async def embed(self, text):
                calls["n"] += 1
                return [float(len(text)), 1.0]

        r = SemanticRetriever(CHUNKS, embedder=_Counting())
        await r.retrieve("python")
        after_first = calls["n"]
        await r.retrieve("python")  # query embed served from cache
        assert calls["n"] == after_first

    async def test_query_cache_fifo_eviction(self):
        r = SemanticRetriever(CHUNKS, embedder=_VecEmbedder())
        r._query_cache_size = 1
        await r.retrieve("python")  # fills the 1-slot query cache
        await r.retrieve("language")  # evicts "python"
        # third distinct query exercises a cache-miss path post-eviction (no assert needed)


# --------------------------------------------------------------------------- #
# HybridRetriever (smoke: both legs + fusion produce ranked results)
# --------------------------------------------------------------------------- #
class TestHybridRetriever:
    async def test_fuses_keyword_and_semantic(self):
        r = HybridRetriever(CHUNKS, embedder=_VecEmbedder())
        res = await r.retrieve("python programming", top_k=2)
        assert res and all(x.retrieval_method == "hybrid" for x in res)
        assert res[0].chunk.id == "c2"

    async def test_hybrid_without_embedder_still_returns(self):
        r = HybridRetriever(CHUNKS)  # semantic leg degrades to keyword
        res = await r.retrieve("python")
        assert res and all(x.retrieval_method == "hybrid" for x in res)
