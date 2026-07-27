"""Unit tests for ragi.chat.openai -- OpenAI-compatible ChatClient (fake transport)."""

from __future__ import annotations

from ragi.chat.openai import OpenAIChatClient


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
