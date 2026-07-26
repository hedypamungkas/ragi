"""End-to-end MCP stdio round-trip: a real ``mcp`` client -> ragworkbench ``search`` tool.

Spawns the server as a subprocess (BM25 over a temp corpus) and drives it with the official
``mcp`` client SDK: initialize -> tools/list -> tools/call. This is a genuine client->server
MCP round-trip, so it is protocol-compatible with Claude Desktop / Cursor by construction.
Skipped when the ``[mcp]`` extra is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src"

# Subprocess server: build a BM25 pipeline over argv[1] corpus dir, then serve stdio.
_SERVER_SCRIPT = (
    "import sys, ragworkbench as rwb; rwb.register_builtins(); "
    "r = rwb.build_pipeline({'enabled':True,'chunker':'paragraph','retriever':'bm25',"
    "'documents':[{'path':sys.argv[1]}]}); "
    "from ragworkbench.export.mcp import RagWorkbenchMCPServer; "
    "RagWorkbenchMCPServer(r).run_stdio()"
)


async def test_mcp_stdio_search_round_trip(tmp_path):
    (tmp_path / "a.txt").write_text("the mitochondrion is the powerhouse of the cell")
    (tmp_path / "b.txt").write_text("photosynthesis converts sunlight into chemical energy in plants")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", _SERVER_SCRIPT, str(tmp_path)],
        env={"PYTHONPATH": str(SRC), "PATH": os.environ.get("PATH", "")},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert "search" in [t.name for t in tools.tools]

            result = await session.call_tool("search", {"query": "photosynthesis sunlight energy", "top_k": 2})
            text = "".join(getattr(c, "text", "") for c in result.content)
            assert "photosynthesis" in text.lower()
            assert "Source: b" in text  # gold doc cited in the markdown
            assert result.isError is not True
