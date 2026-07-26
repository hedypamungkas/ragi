"""ragworkbench/adapters/_dispatch -- shared tool-call dispatch + result formatting.

The OpenAI/Anthropic SDK adapters parse a provider's ``tool_call``/``tool_use`` object into
``{query, top_k, filter}`` then hand off here. Formatting matches the MCP server's
``search`` tool (numbered markdown + ``[Source:]`` citations) so an agent sees the same shape
whether it consumes the lib via MCP, OpenAI function-calling, or Claude ``tool_use``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragworkbench.protocols import Retriever

_logger = logging.getLogger(__name__)


def format_search_result(results: list) -> str:
    """Render retrieval results as numbered markdown with [Source:] citations."""
    if not results:
        return "No passages found."
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.chunk.metadata.get("source", r.chunk.doc_id)
        blocks.append(f"[{i}] [Source: {source}] (score: {r.score:.3f})\n{r.chunk.content}")
    return "\n\n---\n\n".join(blocks)


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("execute_*_tool_call() cannot run from a running event loop; await the *_async variant instead.")


async def dispatch_search_async(retriever: Retriever, query: str, top_k: int = 5, filter: dict | None = None) -> str:
    results = await retriever.retrieve(query, top_k=top_k, metadata_filter=filter)
    return format_search_result(results)


def dispatch_search(retriever: Retriever, query: str, top_k: int = 5, filter: dict | None = None) -> str:
    """Sync entry point (use the ``_async`` variant inside an event loop)."""
    return _run_sync(dispatch_search_async(retriever, query, top_k, filter))
