"""Unit tests for the inlined HTTP transport primitives."""

from __future__ import annotations

import pytest

from ragworkbench._internal.http import BearerAuth, _raise_for_status
from ragworkbench.errors import (
    LLMAuthenticationError,
    LLMInvalidRequestError,
    LLMRateLimitError,
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
