"""ragi/adapters/anthropic -- Claude ``tool_use`` dispatch.

Lets a Claude / Anthropic SDK agent consume the tuned retrieval: register
:func:`anthropic_search_tool_schema` as the tool, then dispatch the model's ``tool_use`` block
through :func:`execute_anthropic_tool_call`. Pure stdlib (no ``anthropic`` SDK import needed
to *dispatch* a call -- the agent runtime supplies the ``tool_use`` object).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragi.adapters._dispatch import dispatch_search, dispatch_search_async
from ragi.export.toolschema import to_anthropic

if TYPE_CHECKING:
    from ragi.protocols import Retriever


def anthropic_search_tool_schema() -> dict:
    """The Anthropic ``tool_use`` tool spec for ``search`` (re-export of ``to_anthropic``)."""
    return to_anthropic()


def _parse_anthropic_args(tool_use) -> dict:
    """Extract {query, top_k, filter} from a Claude ``tool_use`` block (object or dict)."""
    if isinstance(tool_use, dict):
        return tool_use.get("input") or {}
    return getattr(tool_use, "input", None) or {}


def execute_anthropic_tool_call(retriever: Retriever, tool_use) -> str:
    """Sync: dispatch a Claude ``tool_use`` block -> retrieve -> cited markdown string."""
    args = _parse_anthropic_args(tool_use)
    return dispatch_search(retriever, args.get("query", ""), args.get("top_k", 5), args.get("filter"))


async def execute_anthropic_tool_call_async(retriever: Retriever, tool_use) -> str:
    """Async variant (use inside an event loop)."""
    args = _parse_anthropic_args(tool_use)
    return await dispatch_search_async(retriever, args.get("query", ""), args.get("top_k", 5), args.get("filter"))
