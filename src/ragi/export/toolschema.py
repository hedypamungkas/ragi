"""ragi/export/toolschema -- static tool-schema generator for the search tool.

Emits the canonical 3-key ``{name, description, input_schema}`` spec (mirrors MCP's tool
def shape -- cf. koboi/mcp/server.py:21-26) plus format adapters for the three agent
tool-calling conventions:

- **MCP** (``toolschema.to_mcp``): ``input_schema`` -> ``inputSchema`` (camelCase wire key).
- **OpenAI** function-calling (``to_openai``): ``{type: function, function: {name, description, parameters}}``.
- **Anthropic** tool_use (``to_anthropic``): ``{name, description, input_schema}``.

Pure stdlib (only ``json``) -- so a consumer can wire the tuned retrieval as a tool
WITHOUT importing ragi (or any heavy dep) at the agent layer. The schema is
deliberately static and explicit rather than inferred from a signature: when you're
exporting to a spec, explicit beats inferred.
"""

from __future__ import annotations

import json
from typing import Any

SEARCH_DESCRIPTION = (
    "Search the knowledge base for passages relevant to the query. Returns ranked "
    "passages as numbered markdown blocks, each with a [Source: <doc_id>] citation and a "
    "relevance score. Use this to ground answers in the indexed documents."
)


def search_tool_schema() -> dict[str, Any]:
    """The canonical 3-key ``search`` tool spec (JSON-Schema ``object`` input)."""
    return {
        "name": "search",
        "description": SEARCH_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of passages to return (default 5).",
                },
                "filter": {
                    "type": "object",
                    "description": (
                        "Optional metadata filter (Mongo-style): "
                        '{field: value} for equality, {field: {"$gte": v}}, "$in", etc. '
                        "Relevance scoping, not a security boundary."
                    ),
                },
            },
            "required": ["query"],
        },
    }


def to_mcp(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """MCP wire form (``input_schema`` -> ``inputSchema``, camelCase)."""
    spec = spec if spec is not None else search_tool_schema()
    return {"name": spec["name"], "description": spec["description"], "inputSchema": spec["input_schema"]}


def to_openai(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """OpenAI function-calling form: ``{type: function, function: {name, description, parameters}}``."""
    spec = spec if spec is not None else search_tool_schema()
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["input_schema"],
        },
    }


def to_anthropic(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Anthropic ``tool_use`` form: ``{name, description, input_schema}``."""
    spec = spec if spec is not None else search_tool_schema()
    return {"name": spec["name"], "description": spec["description"], "input_schema": spec["input_schema"]}


def dumps(spec: dict[str, Any]) -> str:
    """Pretty-print a tool spec as JSON (2-space indent)."""
    return json.dumps(spec, indent=2)
