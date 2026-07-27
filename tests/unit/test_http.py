"""Unit tests for the inlined HTTP transport primitives."""

from __future__ import annotations

import httpx
import pytest

from ragi._internal import http as http_mod
from ragi._internal.http import BearerAuth, HttpTransport, _raise_for_status
from ragi.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
)


class _FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.text = "" if payload is not None else "plain"
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TestHttpPrimitives:
    def test_bearer_auth(self):
        assert BearerAuth("tok").apply({})["Authorization"] == "Bearer tok"

    def test_ok_response_does_not_raise(self):
        _raise_for_status(_FakeResp(200, {"ok": True}))  # no exception

    def test_status_mapping(self):
        with pytest.raises(LLMAuthenticationError):
            _raise_for_status(_FakeResp(401, {"error": {"message": "no"}}))
        with pytest.raises(LLMRateLimitError) as exc_info:
            _raise_for_status(_FakeResp(429, {"error": {"message": "slow"}}))
        assert exc_info.value.retry_after is None
        with pytest.raises(LLMInvalidRequestError):
            _raise_for_status(_FakeResp(400, {"error": {"message": "bad"}}))
        with pytest.raises(LLMServerError):
            _raise_for_status(_FakeResp(503, {"error": {"message": "down"}}))

    def test_retry_after_header_parsed(self):
        resp = _FakeResp(429, {"error": {"message": "slow"}}, headers={"retry-after": "5"})
        with pytest.raises(LLMRateLimitError) as exc_info:
            _raise_for_status(resp)
        assert exc_info.value.retry_after == 5.0

    def test_extract_error_detail_handles_plain_text(self):
        from ragi._internal.http import _extract_error_detail

        resp = _FakeResp(500, payload=None, headers={})  # text="plain", json() raises
        assert _extract_error_detail(resp) == "plain"


class _AsyncResp:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text or ("" if payload is not None else "plain")
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _AsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent_headers = []

    async def post(self, url, json=None, headers=None):
        self.sent_headers.append(headers or {})
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self):
        pass


@pytest.fixture
def no_sleep(monkeypatch):
    async def _noop(_):
        return None

    monkeypatch.setattr(http_mod.asyncio, "sleep", _noop)


def _transport(responses, **kw):
    t = HttpTransport("https://api.example", BearerAuth("tok"), max_retries=2, **kw)
    t._client = _AsyncClient(responses)
    return t


class TestHttpTransportPost:
    async def test_returns_json_on_2xx(self, no_sleep):
        t = _transport([_AsyncResp(200, {"ok": True})])
        assert await t.post("/x", {"q": 1}) == {"ok": True}

    async def test_injects_bearer_and_default_headers(self, no_sleep):
        t = _transport([_AsyncResp(200, {"ok": 1})])
        await t.post("/x", {})
        headers = t._client.sent_headers[0]
        assert headers["Authorization"] == "Bearer tok"
        assert headers["Content-Type"] == "application/json"

    async def test_non_json_2xx_raises_parse_error(self, no_sleep):
        t = _transport([_AsyncResp(200, payload=None, text="not json")])
        with pytest.raises(LLMResponseParseError):
            await t.post("/x", {})

    async def test_non_retryable_4xx_raises(self, no_sleep):
        t = _transport([_AsyncResp(400, {"error": {"message": "bad"}})])
        with pytest.raises(LLMInvalidRequestError):
            await t.post("/x", {})

    async def test_auth_error_raises(self, no_sleep):
        t = _transport([_AsyncResp(401, {"error": {"message": "no"}})])
        with pytest.raises(LLMAuthenticationError):
            await t.post("/x", {})

    async def test_retryable_429_retries_then_succeeds(self, no_sleep):
        t = _transport(
            [_AsyncResp(429, {"error": {"message": "slow"}}, headers={"retry-after": "0"}), _AsyncResp(200, {"ok": 1})]
        )
        assert await t.post("/x", {}) == {"ok": 1}

    async def test_retryable_503_exhausts_raises_server_error(self, no_sleep):
        t = _transport([_AsyncResp(503, {"error": {"message": "down"}})] * 5)
        with pytest.raises(LLMServerError):
            await t.post("/x", {})

    async def test_connect_error_retries_then_succeeds(self, no_sleep):
        t = _transport([httpx.ConnectError("transient"), _AsyncResp(200, {"ok": 1})])
        assert await t.post("/x", {}) == {"ok": 1}

    async def test_connect_error_exhausts_raises_connection_error(self, no_sleep):
        t = _transport([httpx.ConnectError("dead")] * 5)
        with pytest.raises(LLMConnectionError):
            await t.post("/x", {})

    async def test_close_does_not_raise(self):
        t = _transport([])
        await t.close()
