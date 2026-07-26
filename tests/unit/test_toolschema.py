"""Unit tests for the static tool-schema generator (no deps)."""

from __future__ import annotations

from ragi.export.toolschema import (
    search_tool_schema,
    to_anthropic,
    to_mcp,
    to_openai,
)


def test_schema_shape():
    s = search_tool_schema()
    assert set(s.keys()) == {"name", "description", "input_schema"}
    assert s["name"] == "search"
    ischema = s["input_schema"]
    assert ischema["type"] == "object"
    assert ischema["required"] == ["query"]
    assert "query" in ischema["properties"]
    assert "top_k" in ischema["properties"]
    assert "filter" in ischema["properties"]


def test_to_mcp_uses_camelcase_inputschemakey():
    m = to_mcp()
    assert "inputSchema" in m
    assert "input_schema" not in m
    assert m["name"] == "search"
    assert m["inputSchema"]["type"] == "object"


def test_to_openai_function_envelope():
    o = to_openai()
    assert o["type"] == "function"
    fn = o["function"]
    assert fn["name"] == "search"
    assert "parameters" in fn
    assert fn["parameters"]["type"] == "object"
    assert fn["parameters"]["required"] == ["query"]


def test_to_anthropic_shape():
    a = to_anthropic()
    assert set(a.keys()) == {"name", "description", "input_schema"}
    assert a["input_schema"]["type"] == "object"


def test_format_adapters_accept_custom_spec():
    custom = {"name": "my_search", "description": "x", "input_schema": {"type": "object"}}
    assert to_mcp(custom)["name"] == "my_search"
    assert to_openai(custom)["function"]["name"] == "my_search"
    assert to_anthropic(custom)["name"] == "my_search"
