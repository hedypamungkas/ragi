"""Unit tests for ragi.embeddings.openai -- OpenAI-compatible embedding client (fail-soft)."""

from __future__ import annotations

from ragi.embeddings.openai import OpenAIEmbeddingClient
from ragi.errors import LLMRateLimitError


class _FakeTransport:
    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc

    async def post(self, path, body):
        if self._exc:
            raise self._exc
        return self._payload

    async def close(self):
        pass


class TestParseAt:
    def test_extracts_embedding(self):
        data = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4]}]}
        assert OpenAIEmbeddingClient._parse_at(data, 0) == [0.1, 0.2, 0.3]
        assert OpenAIEmbeddingClient._parse_at(data, 1) == [0.4]

    def test_missing_data_key(self):
        assert OpenAIEmbeddingClient._parse_at({}, 0) is None

    def test_index_out_of_range(self):
        assert OpenAIEmbeddingClient._parse_at({"data": [{"embedding": [0.1]}]}, 5) is None

    def test_wrong_type_input(self):
        assert OpenAIEmbeddingClient._parse_at("not a dict", 0) is None

    def test_non_list_embedding(self):
        assert OpenAIEmbeddingClient._parse_at({"data": [{"embedding": "nope"}]}, 0) is None

    def test_coerces_to_float(self):
        assert OpenAIEmbeddingClient._parse_at({"data": [{"embedding": [1, 2]}]}, 0) == [1.0, 2.0]


class TestEmbed:
    async def test_happy_returns_vector(self):
        c = OpenAIEmbeddingClient("k")
        c._transport = _FakeTransport({"data": [{"embedding": [0.1, 0.2]}]})
        assert await c.embed("hello") == [0.1, 0.2]

    async def test_fail_soft_on_transport_error(self):
        c = OpenAIEmbeddingClient("k")
        c._transport = _FakeTransport(exc=LLMRateLimitError("rate limited"))
        assert await c.embed("hello") is None


class TestEmbedBatch:
    async def test_empty_input_returns_empty(self):
        c = OpenAIEmbeddingClient("k")
        assert await c.embed_batch([]) == []

    async def test_happy_returns_per_text_vectors(self):
        c = OpenAIEmbeddingClient("k")
        c._transport = _FakeTransport({"data": [{"embedding": [0.1]}, {"embedding": [0.2]}]})
        out = await c.embed_batch(["a", "b"])
        assert out == [[0.1], [0.2]]

    async def test_fail_soft_all_none_on_transport_error(self):
        c = OpenAIEmbeddingClient("k")
        c._transport = _FakeTransport(exc=LLMRateLimitError("down"))
        assert await c.embed_batch(["a", "b", "c"]) == [None, None, None]

    async def test_per_item_parse_failure_is_none_slot(self):
        # fewer embeddings than texts -> the missing slot parses to None
        c = OpenAIEmbeddingClient("k")
        c._transport = _FakeTransport({"data": [{"embedding": [0.1]}]})
        out = await c.embed_batch(["a", "b"])
        assert out == [[0.1], None]


class TestClose:
    async def test_close_does_not_raise(self):
        c = OpenAIEmbeddingClient("k")
        await c.close()  # delegates to transport.close (no-op fake)


def test_construct_with_base_url_override():
    # just ensure custom base_url path doesn't raise and wires a transport
    c = OpenAIEmbeddingClient("k", base_url="https://gateway.example/v1")
    assert c._model == "text-embedding-3-small"
