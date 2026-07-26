"""ragi/export/mcp -- read-only MCP server (official ``mcp`` SDK / FastMCP).

Exposes a tuned :class:`Retriever` as a single **read-only** ``search`` tool over stdio
(Claude Desktop / Cursor / local agents) or Streamable HTTP (remote). SAFE-by-construction
-- only a read tool exists, so there is **no risk gating** (cf. koboi's ``select_exposed_tools``,
which is unnecessary here). The ``[mcp]`` extra gates the SDK import so the core stays zero-dep.

The ``mcp`` SDK (FastMCP) absorbs all wire plumbing -- protocol version negotiation, the
JSON-RPC dispatch loop, the ``content:[{type:text}]`` result envelope, and both transports.
We only provide: one ``@mcp.tool async def search`` that calls the retriever and returns a
deterministic markdown string (koboi's convention -- never raw ``list[dict]``, which doesn't
round-trip through the text envelope).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ragi.errors import LLMInvalidRequestError

if TYPE_CHECKING:
    from ragi.protocols import Retriever

_logger = logging.getLogger(__name__)


def _format_results(results: list) -> str:
    """Render retrieval results as numbered markdown with [Source:] citations."""
    if not results:
        return "No passages found for that query."
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.chunk.metadata.get("source", r.chunk.doc_id)
        header = f"[{i}] [Source: {source}] (score: {r.score:.3f}, method: {r.retrieval_method})"
        blocks.append(f"{header}\n{r.chunk.content}")
    return "\n\n---\n\n".join(blocks)


class RagWorkbenchMCPServer:
    """Wrap a :class:`Retriever` as a read-only MCP server exposing one ``search`` tool."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        name: str = "ragi",
        top_k_default: int = 5,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as e:
            raise LLMInvalidRequestError("(mcp) the mcp SDK is required: pip install 'ragi[mcp]'") from e
        self._retriever = retriever
        self._top_k_default = top_k_default
        self._mcp = FastMCP(name, host=host, port=port)
        self._register_search_tool()

    def _register_search_tool(self) -> None:
        retriever = self._retriever
        top_k_default = self._top_k_default
        format_results = _format_results

        @self._mcp.tool()
        async def search(query: str, top_k: int = top_k_default) -> str:
            """Search the knowledge base for passages relevant to the query.

            Returns ranked passages as numbered markdown blocks, each with a
            [Source: <doc_id>] citation and a relevance score. Use to ground answers
            in the indexed documents.
            """
            try:
                results = await retriever.retrieve(query, top_k=top_k)
            except Exception as exc:  # noqa: BLE001 -- never crash the MCP call
                _logger.warning("search tool failed: %s", exc)
                return f"Search failed: {exc}"
            return format_results(results)

    @property
    def fastmcp(self):
        """The underlying FastMCP instance (for advanced/custom wiring)."""
        return self._mcp

    def run_stdio(self) -> None:
        """Serve over stdio (blocking). For Claude Desktop / Cursor / local agents."""
        self._mcp.run(transport="stdio")

    def run_http(self) -> None:
        """Serve over Streamable HTTP (blocking). host/port set at construction."""
        self._mcp.run(transport="streamable-http")


def serve_from_config(
    conf: dict[str, Any],
    *,
    transport: str = "stdio",
    embedder=None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Build a pipeline from ``conf`` and serve it over MCP (blocking).

    Lexical + rerank configs serve directly. Semantic/hybrid configs need an ``embedder``
    -- ``ragi serve`` does not wire one in v0.3, so they degrade to keyword over MCP;
    pass ``embedder=`` programmatically for full semantic serve.
    """
    import ragi

    ragi.register_builtins()
    retriever = ragi.build_pipeline(conf, embedder=embedder)
    if retriever is None:
        raise ValueError("pipeline produced no retriever (disabled or no documents loaded)")
    server = RagWorkbenchMCPServer(retriever, host=host, port=port)
    if transport == "stdio":
        server.run_stdio()
    elif transport in ("http", "streamable-http"):
        server.run_http()
    else:
        raise ValueError(f"unknown transport {transport!r}; use 'stdio' or 'http'")
