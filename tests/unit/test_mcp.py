"""Unit tests for ragi.export.mcp -- read-only MCP server (fake FastMCP; no [mcp] extra).

The ``mcp`` SDK import is call-time gated inside ``RagWorkbenchMCPServer.__init__``, so the
module imports without the extra. We inject a fake ``FastMCP`` into ``sys.modules`` to
exercise the tool registration + run delegation + ``serve_from_config`` branches.
"""

from __future__ import annotations

import sys
import types

import pytest

import ragi
from ragi.errors import LLMInvalidRequestError
from ragi.export.mcp import RagWorkbenchMCPServer, _format_results, serve_from_config


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _FakeFastMCP:
    def __init__(self, name, host=None, port=None):
        self.name = name
        self.host = host
        self.port = port
        self.tools = {}
        self.run_calls = []

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco

    def run(self, transport):
        self.run_calls.append(transport)


def _install_fake_mcp(monkeypatch):
    mcp_pkg = types.ModuleType("mcp")
    server_pkg = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = _FakeFastMCP
    server_pkg.fastmcp = fastmcp_mod
    mcp_pkg.server = server_pkg
    monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server", server_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)


def _result(source=None, doc_id="d1", content="chunk text", score=0.5, method="bm25"):
    meta = {"source": source} if source else {}
    chunk = types.SimpleNamespace(metadata=meta, doc_id=doc_id, content=content)
    return types.SimpleNamespace(chunk=chunk, score=score, retrieval_method=method)


class _FakeRetriever:
    def __init__(self, results=None, exc=None):
        self._results = results if results is not None else [_result()]
        self._exc = exc

    async def retrieve(self, query, *, top_k=5, metadata_filter=None):
        if self._exc:
            raise self._exc
        return self._results


# --------------------------------------------------------------------------- #
# _format_results (pure)
# --------------------------------------------------------------------------- #
class TestFormatResults:
    def test_empty(self):
        assert _format_results([]) == "No passages found for that query."

    def test_renders_source_and_score(self):
        out = _format_results([_result(source="policy.md", content="the answer", score=0.123, method="bm25")])
        assert "[1] [Source: policy.md]" in out
        assert "score: 0.123" in out and "method: bm25" in out
        assert "the answer" in out

    def test_source_falls_back_to_doc_id(self):
        out = _format_results([_result(source=None, doc_id="doc42")])
        assert "[Source: doc42]" in out

    def test_multiple_results_separated(self):
        out = _format_results([_result(source="a"), _result(source="b")])
        assert "[1] [Source: a]" in out and "[2] [Source: b]" in out
        assert "---" in out


# --------------------------------------------------------------------------- #
# server construction + search tool
# --------------------------------------------------------------------------- #
class TestServer:
    def test_missing_mcp_sdk_raises(self):
        # no fake mcp installed -> real import fails -> LLMInvalidRequestError
        with pytest.raises(LLMInvalidRequestError, match="mcp SDK"):
            RagWorkbenchMCPServer(_FakeRetriever())

    async def test_search_tool_happy_path(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        server = RagWorkbenchMCPServer(
            _FakeRetriever(results=[_result(source="s.md", content="the cell")]), top_k_default=3
        )
        search = server.fastmcp.tools["search"]
        out = await search(query="cell", top_k=2)
        assert "[Source: s.md]" in out and "the cell" in out

    async def test_search_tool_swallows_retriever_error(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        server = RagWorkbenchMCPServer(_FakeRetriever(exc=RuntimeError("boom")))
        out = await server.fastmcp.tools["search"](query="x")
        assert out.startswith("Search failed:") and "boom" in out

    def test_run_stdio_delegates(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        server = RagWorkbenchMCPServer(_FakeRetriever())
        server.run_stdio()
        assert server.fastmcp.run_calls == ["stdio"]

    def test_run_http_delegates(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        server = RagWorkbenchMCPServer(_FakeRetriever())
        server.run_http()
        assert server.fastmcp.run_calls == ["streamable-http"]


# --------------------------------------------------------------------------- #
# serve_from_config
# --------------------------------------------------------------------------- #
class TestServeFromConfig:
    def test_none_retriever_raises(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        monkeypatch.setattr(ragi, "build_pipeline", lambda conf, embedder=None: None)
        with pytest.raises(ValueError, match="no retriever"):
            serve_from_config({"enabled": False})

    def test_unknown_transport_raises(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        monkeypatch.setattr(ragi, "build_pipeline", lambda conf, embedder=None: _FakeRetriever())
        with pytest.raises(ValueError, match="unknown transport"):
            serve_from_config({"enabled": True}, transport="ws")

    def test_stdio_serves(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        monkeypatch.setattr(ragi, "build_pipeline", lambda conf, embedder=None: _FakeRetriever())
        serve_from_config({"enabled": True}, transport="stdio")  # no raise; faked run returns

    def test_http_serves(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        monkeypatch.setattr(ragi, "build_pipeline", lambda conf, embedder=None: _FakeRetriever())
        serve_from_config({"enabled": True}, transport="http")
