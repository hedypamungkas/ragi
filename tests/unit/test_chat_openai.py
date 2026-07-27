"""Unit tests for ragi.chat.openai -- OpenAI-compatible ChatClient (fake transport)."""

from __future__ import annotations

import pytest

from ragi.chat.openai import OpenAIChatClient
from ragi.errors import LLMConnectionError, LLMRateLimitError


class _FakeTransport:
    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc
        self.closed = False
        self.posts = []

    async def post(self, path, body):
        self.posts.append((path, body))
        if self._exc:
            raise self._exc
        return self._payload

    async def close(self):
        self.closed = True


class TestOpenAIChatClient:
    async def test_complete_returns_content(self):
        client = OpenAIChatClient("k")
        client._transport = _FakeTransport({"choices": [{"message": {"content": "hi there"}}]})
        out = await client.complete([{"role": "user", "content": "q"}])
        assert out == "hi there"

    async def test_complete_empty_choices_returns_empty(self):
        client = OpenAIChatClient("k")
        client._transport = _FakeTransport({"choices": []})
        assert await client.complete([]) == ""

    async def test_complete_no_choices_key_returns_empty(self):
        client = OpenAIChatClient("k")
        client._transport = _FakeTransport({})
        assert await client.complete([]) == ""

    async def test_complete_missing_content_returns_empty(self):
        client = OpenAIChatClient("k")
        client._transport = _FakeTransport({"choices": [{"message": {}}]})
        assert await client.complete([]) == ""

    async def test_complete_sends_model_and_temperature(self):
        client = OpenAIChatClient("k", model="m", temperature=0.7)
        t = _FakeTransport({"choices": [{"message": {"content": "x"}}]})
        client._transport = t
        await client.complete([{"role": "user", "content": "q"}])
        path, body = t.posts[0]
        assert path == "/chat/completions"
        assert body["model"] == "m" and body["temperature"] == 0.7

    async def test_close_delegates_to_transport(self):
        client = OpenAIChatClient("k")
        t = _FakeTransport()
        client._transport = t
        await client.close()
        assert t.closed is True

    @pytest.mark.parametrize("exc", [LLMRateLimitError("429"), LLMConnectionError("boom")])
    async def test_transport_error_propagates_uncaught(self, exc):
        # Contract (module docstring): ``complete`` raises the ``LLM*`` hierarchy on
        # transport errors so callers (runner / scorers) can catch and fail-soft. A
        # future broad ``except`` swallowing this would make "transport died" look like
        # "LLM answered empty" -- this pins the propagation.
        client = OpenAIChatClient("k")
        client._transport = _FakeTransport(exc=exc)
        with pytest.raises(type(exc)):
            await client.complete([{"role": "user", "content": "q"}])
