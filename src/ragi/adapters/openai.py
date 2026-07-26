"""ragi/adapters/openai -- OpenAI function-calling tool-call dispatch.

Lets a raw OpenAI SDK / OpenAI-compatible agent consume the tuned retrieval: register
:func:`openai_search_tool_schema` as the tool, then dispatch the model's ``tool_call`` through
:func:`execute_openai_tool_call`. Pure stdlib (no ``openai`` SDK import needed to *dispatch*
a call -- the agent runtime supplies the ``tool_call`` object).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ragi.adapters._dispatch import dispatch_search, dispatch_search_async
from ragi.export.toolschema import to_openai

if TYPE_CHECKING:
    from ragi.protocols import Retriever


def openai_search_tool_schema() -> dict:
    """The OpenAI function-calling tool spec for ``search`` (re-export of ``to_openai``)."""
    return to_openai()


def _parse_openai_args(tool_call) -> dict:
    """Extract {query, top_k, filter} from an OpenAI ``tool_call`` (object or dict)."""
    fn = getattr(tool_call, "function", None)
    if fn is None and isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
    raw = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else {}
    return raw or {}


def execute_openai_tool_call(retriever: Retriever, tool_call) -> str:
    """Sync: dispatch an OpenAI ``tool_call`` -> retrieve -> cited markdown string."""
    args = _parse_openai_args(tool_call)
    return dispatch_search(retriever, args.get("query", ""), args.get("top_k", 5), args.get("filter"))


async def execute_openai_tool_call_async(retriever: Retriever, tool_call) -> str:
    """Async variant (use inside an event loop)."""
    args = _parse_openai_args(tool_call)
    return await dispatch_search_async(retriever, args.get("query", ""), args.get("top_k", 5), args.get("filter"))
