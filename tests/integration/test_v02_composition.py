"""Integration tests for v0.2 build_pipeline composition (rerank + rewrite wrappers)."""

from __future__ import annotations

import os
import tempfile

import ragi


def _corpus() -> str:
    d = tempfile.mkdtemp()
    (open(os.path.join(d, "a.txt"), "w").write("the mitochondrion is the powerhouse of the cell"))
    (open(os.path.join(d, "b.txt"), "w").write("photosynthesis converts sunlight into chemical energy"))
    return d


def test_base_retriever_when_no_wrappers():
    ragi.register_builtins()
    r = ragi.build_pipeline(
        {"enabled": True, "chunker": "paragraph", "retriever": "bm25", "documents": [{"path": _corpus()}]}
    )
    assert type(r).__name__ == "BM25Retriever"


def test_rerank_wraps_base():
    ragi.register_builtins()
    r = ragi.build_pipeline(
        {
            "enabled": True,
            "chunker": "paragraph",
            "retriever": "bm25",
            "rerank": {"provider": "jina", "api_key": "fake"},  # no network at construction
            "documents": [{"path": _corpus()}],
        }
    )
    assert type(r).__name__ == "CrossEncoderReranker"


def test_rewrite_then_rerank_nested_correctly():
    # rewrite wrapper sits OUTSIDE rerank: query is rewritten, then the rerank-wrapped
    # retriever over-fetches + rescores. Outer type = CrossEncoderReranker, inner = RewritingRetriever.
    ragi.register_builtins()
    r = ragi.build_pipeline(
        {
            "enabled": True,
            "chunker": "paragraph",
            "retriever": "bm25",
            "query_rewrite": True,
            "rerank": {"provider": "jina", "api_key": "fake"},
            "documents": [{"path": _corpus()}],
        }
    )
    assert type(r).__name__ == "CrossEncoderReranker"
    assert type(r._base).__name__ == "RewritingRetriever"


def test_rerank_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "envkey")
    ragi.register_builtins()
    r = ragi.build_pipeline(
        {
            "enabled": True,
            "chunker": "paragraph",
            "retriever": "bm25",
            "rerank": {"provider": "jina"},
            "documents": [{"path": _corpus()}],
        }
    )
    assert type(r).__name__ == "CrossEncoderReranker"
