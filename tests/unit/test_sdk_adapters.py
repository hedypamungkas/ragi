"""Unit tests for the OpenAI / Anthropic SDK tool-call adapters (pure stdlib, no SDK)."""

from __future__ import annotations

from types import SimpleNamespace

from ragworkbench.adapters.anthropic import execute_anthropic_tool_call
from ragworkbench.adapters.openai import execute_openai_tool_call
from ragworkbench.retrieval.retriever import KeywordRetriever
from ragworkbench.types import Chunk

CHUNKS = [Chunk("c1", "d1", "the mitochondrion is the powerhouse of the cell", metadata={"source": "bio"})]
RETRIEVER = KeywordRetriever(chunks=CHUNKS)


class _OpenAIToolCall(SimpleNamespace):
    """Mimic the openai SDK ToolCall shape (function.arguments is a JSON string)."""


def test_openai_adapter_parses_json_arguments():
    tc = _OpenAIToolCall(
        id="1",
        function=SimpleNamespace(name="search", arguments='{"query": "mitochondrion powerhouse", "top_k": 2}'),
    )
    out = execute_openai_tool_call(RETRIEVER, tc)
    assert "Source: bio" in out
    assert "mitochondrion" in out.lower()


def test_openai_adapter_accepts_dict_shape():
    tc = {"function": {"name": "search", "arguments": '{"query": "mitochondrion"}'}}
    out = execute_openai_tool_call(RETRIEVER, tc)
    assert isinstance(out, str) and out


def test_anthropic_adapter_parses_input_dict():
    tool_use = SimpleNamespace(name="search", input={"query": "mitochondrion powerhouse", "top_k": 2})
    out = execute_anthropic_tool_call(RETRIEVER, tool_use)
    assert "Source: bio" in out


def test_anthropic_adapter_accepts_dict_shape():
    tool_use = {"name": "search", "input": {"query": "mitochondrion"}}
    out = execute_anthropic_tool_call(RETRIEVER, tool_use)
    assert isinstance(out, str) and out
