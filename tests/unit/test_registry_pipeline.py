"""Unit tests for ragi.registry builders + _load_documents source dispatch + build_pipeline.

Local-file / glob / dir loading uses tmp_path; http/s3/firecrawl sources are exercised by
monkeypatching the fetch entry-points in ``ragi.ingest.sources`` (which ``_load_documents``
imports at call time).
"""

from __future__ import annotations

import pytest

import ragi
from ragi.registry import (
    ComponentEntry,
    _build_chunker,
    _build_retriever,
    _load_documents,
    _resolve_kwargs,
    build_pipeline,
    chunker_registry,
    retriever_registry,
)
from ragi.retrieval.retriever import KeywordRetriever
from ragi.types import Chunk

ragi.register_builtins()


# --------------------------------------------------------------------------- #
# _resolve_kwargs
# --------------------------------------------------------------------------- #
class TestResolveKwargs:
    def test_alias_maps_yaml_key_to_param(self):
        entry = ComponentEntry(
            cls=type("C", (), {}), parameters={"size": {"default": 100}}, config_aliases={"chunk_size": "size"}
        )
        assert _resolve_kwargs(entry, {"chunk_size": 50}) == {"size": 50}

    def test_default_used_when_yaml_key_absent(self):
        entry = ComponentEntry(
            cls=type("C", (), {}), parameters={"size": {"default": 100}}, config_aliases={"chunk_size": "size"}
        )
        assert _resolve_kwargs(entry, {}) == {"size": 100}

    def test_no_default_no_value_omits_kwarg(self):
        entry = ComponentEntry(cls=type("C", (), {}), parameters={"required": {}}, config_aliases={})
        assert _resolve_kwargs(entry, {}) == {}


# --------------------------------------------------------------------------- #
# _build_chunker / _build_retriever fallbacks
# --------------------------------------------------------------------------- #
class TestBuilders:
    def test_build_chunker_unknown_falls_back_to_paragraph(self):
        c = _build_chunker({"chunker": "does-not-exist"})
        assert c.__class__.__name__ == "ParagraphChunker"

    def test_build_chunker_no_chunkers_registered_raises(self, monkeypatch):
        monkeypatch.setattr(chunker_registry, "get", lambda name: None)
        with pytest.raises(ValueError, match="No chunkers registered"):
            _build_chunker({})

    def test_build_retriever_unknown_falls_back_to_keyword(self):
        r = _build_retriever(_chunks(), {"retriever": "does-not-exist"})
        assert isinstance(r, KeywordRetriever)

    def test_build_retriever_no_retrievers_registered_raises(self, monkeypatch):
        monkeypatch.setattr(retriever_registry, "get", lambda name: None)
        with pytest.raises(ValueError, match="No retrievers registered"):
            _build_retriever(_chunks(), {})

    def test_build_retriever_injects_embedder(self):
        class _Inject:
            def __init__(self, chunks, embedder=None):
                self.chunks = chunks
                self.embedder = embedder

            async def retrieve(self, *a, **kw):
                return []

        retriever_registry.register("__inject_probe__", _Inject, inject=["embedder"])
        r = _build_retriever(_chunks(), {"retriever": "__inject_probe__"}, embedder="FAKE_EMBEDDER")
        assert r.embedder == "FAKE_EMBEDDER"


def _chunks():
    return [Chunk("c1", "d1", "the quick brown fox")]


# --------------------------------------------------------------------------- #
# _load_documents -- source dispatch + edge cases
# --------------------------------------------------------------------------- #
class TestLoadDocuments:
    def test_local_file_loads_and_carries_source_format(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("the mitochondrion is the powerhouse", encoding="utf-8")
        _chunker, chunks = _load_documents({"chunker": "paragraph", "documents": [{"path": str(f)}]})
        assert chunks
        assert all(c.metadata.get("source_format") == "text" for c in chunks)
        assert all(c.metadata.get("source") == "doc" for c in chunks)

    def test_bare_string_path(self, tmp_path):
        f = tmp_path / "bare.txt"
        f.write_text("some content here", encoding="utf-8")
        _chunker, chunks = _load_documents({"documents": [str(f)]})
        assert chunks and "some content" in chunks[0].content

    def test_directory_recursed(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta content", encoding="utf-8")
        _chunker, chunks = _load_documents({"documents": [{"path": str(tmp_path)}]})
        joined = " ".join(c.content for c in chunks)
        assert "alpha" in joined and "beta" in joined

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "x.txt").write_text("globbed content", encoding="utf-8")
        _chunker, chunks = _load_documents({"documents": [{"path": str(tmp_path / "*.txt")}]})
        assert chunks and any("globbed" in c.content for c in chunks)

    def test_glob_no_match_warns(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            _chunker, chunks = _load_documents({"documents": [{"path": str(tmp_path / "*.nomatch")}]})
        assert chunks == []
        assert any("matched no files" in r.message for r in caplog.records)

    def test_missing_file_path_warns(self, caplog, tmp_path):
        with caplog.at_level("WARNING"):
            _chunker, chunks = _load_documents({"documents": [{"path": str(tmp_path / "absent.txt")}]})
        assert chunks == []
        assert any("does not exist" in r.message for r in caplog.records)

    def test_oversize_document_skipped(self, tmp_path, caplog):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB
        with caplog.at_level("WARNING"):
            _chunker, chunks = _load_documents({"max_document_size_mb": 1, "documents": [{"path": str(f)}]})
        assert chunks == []
        assert any("exceeds max_document_size_mb" in r.message for r in caplog.records)

    def test_empty_text_document_skipped(self, tmp_path, caplog):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with caplog.at_level("INFO"):
            _chunker, chunks = _load_documents({"documents": [{"path": str(f)}]})
        assert chunks == []

    def test_http_source(self, monkeypatch):
        import ragi.ingest.sources as src

        monkeypatch.setattr(
            src, "fetch_http_entry", lambda entry, cache, max_bytes: iter([("web.txt", b"web content here")])
        )
        _chunker, chunks = _load_documents({"documents": [{"source": "http", "url": "http://x"}]})
        assert chunks and "web content" in chunks[0].content

    def test_https_source_routes_to_http_fetcher(self, monkeypatch):
        import ragi.ingest.sources as src

        monkeypatch.setattr(
            src, "fetch_http_entry", lambda entry, cache, max_bytes: iter([("web.txt", b"https content")])
        )
        _chunker, chunks = _load_documents({"documents": [{"source": "https", "url": "https://x"}]})
        assert chunks and "https content" in chunks[0].content

    def test_s3_source(self, monkeypatch):
        import ragi.ingest.sources as src

        monkeypatch.setattr(
            src, "fetch_s3_entry", lambda entry, cache, max_bytes: iter([("o.txt", b"s3 object content")])
        )
        _chunker, chunks = _load_documents({"documents": [{"source": "s3", "bucket": "b"}]})
        assert chunks and "s3 object" in chunks[0].content

    def test_firecrawl_source(self, monkeypatch):
        import ragi.ingest.sources as src

        # url-less firecrawl entry: reaches the firecrawl branch directly (the real
        # ``fetch_firecrawl_entry`` early-returns without a seed url, so the mock stands
        # in here just to prove the branch wiring).
        monkeypatch.setattr(
            src, "fetch_firecrawl_entry", lambda entry, cache: iter([("page", b"crawled page content")])
        )
        _chunker, chunks = _load_documents({"documents": [{"source": "firecrawl"}]})
        assert chunks and "crawled page" in chunks[0].content

    def test_firecrawl_url_entry_routes_to_firecrawl_not_http(self, monkeypatch):
        # Regression: a realistic url-bearing firecrawl entry must reach the firecrawl
        # fetcher, not the generic HTTP fetcher. Previously the dispatch predicate
        # ``source == "http" or "url" in entry`` misrouted it to HTTP before the firecrawl
        # branch was ever reached. This test fails on that bug and passes once the
        # explicit-source routing is checked before the ``url``-shorthand fallback.
        import ragi.ingest.sources as src

        calls: dict[str, list[dict]] = {"http": [], "firecrawl": []}

        def _spy_http(entry, cache, *, max_bytes):
            calls["http"].append(entry)
            return iter([])

        def _spy_firecrawl(entry, cache):
            calls["firecrawl"].append(entry)
            return iter([("page", b"crawled page content")])

        monkeypatch.setattr(src, "fetch_http_entry", _spy_http)
        monkeypatch.setattr(src, "fetch_firecrawl_entry", _spy_firecrawl)

        _chunker, chunks = _load_documents(
            {"documents": [{"source": "firecrawl", "url": "https://site.example", "api_key": "k"}]}
        )
        assert calls["firecrawl"] and calls["http"] == []
        assert chunks and "crawled page" in chunks[0].content

    def test_unsupported_source_warns(self, caplog):
        with caplog.at_level("WARNING"):
            _chunker, chunks = _load_documents({"documents": [{"source": "ftp", "path": "x"}]})
        assert chunks == []
        assert any("Unknown/unsupported document source" in r.message for r in caplog.records)

    def test_non_dict_non_str_entry_ignored(self):
        _chunker, chunks = _load_documents({"documents": [42]})
        assert chunks == []


# --------------------------------------------------------------------------- #
# build_pipeline disabled / empty
# --------------------------------------------------------------------------- #
class TestBuildPipelineGuard:
    def test_disabled_returns_none(self):
        assert build_pipeline({"enabled": False, "documents": [{"path": "x"}]}) is None

    def test_empty_conf_returns_none(self):
        assert build_pipeline({}) is None

    def test_enabled_but_no_documents_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert build_pipeline({"enabled": True, "documents": []}) is None
        assert any("no documents loaded" in r.message for r in caplog.records)
