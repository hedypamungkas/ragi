"""ragworkbench/_internal/http -- minimal async HTTP transport for provider calls.

Inlined & slimmed from koboi/llm/http_transport.py + koboi/llm/auth.py (only the surface
the rerank backends + embedding client need: POST-with-retry + Bearer auth). Raises the
``ragworkbench.errors`` LLM* hierarchy. ~95 LOC, zero ``koboi.*`` imports.

Rerank/embedding callers wrap ``post()`` in ``try/except`` and fail-soft (return None /
base results), so the exact exception subclass rarely matters downstream -- but we map
status codes faithfully so callers *can* branch on it if they want.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import httpx

from ragworkbench.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
)

_logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_DEFAULT_TIMEOUT = 120.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 529})


class AuthStrategy(ABC):
    """Pluggable auth: mutate outbound headers (e.g. inject a Bearer token)."""

    @abstractmethod
    def apply(self, headers: dict[str, str]) -> dict[str, str]: ...


class BearerAuth(AuthStrategy):
    """``Authorization: Bearer <token>``."""

    def __init__(self, token: str) -> None:
        self.token = token

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers["Authorization"] = f"Bearer {self.token}"
        return headers


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            msg = (data.get("error") or {}).get("message") if isinstance(data.get("error"), dict) else None
            return str(msg or data)[:500]
        return str(data)[:500]
    except Exception:  # noqa: BLE001 -- best-effort detail extraction
        return response.text[:500]


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    detail = _extract_error_detail(response)
    if status in (401, 403):
        raise LLMAuthenticationError(f"{status}: {detail}")
    if status == 429:
        retry_after = response.headers.get("retry-after")
        try:
            ra = float(retry_after) if retry_after else None
        except ValueError:
            ra = None
        raise LLMRateLimitError(f"429: {detail}", retry_after=ra)
    if status >= 500:
        raise LLMServerError(f"{status}: {detail}")
    raise LLMInvalidRequestError(f"{status}: {detail}")  # 4xx incl. 400


class HttpTransport:
    """Async POST client with bounded retry + exponential backoff."""

    def __init__(
        self,
        base_url: str,
        auth: AuthStrategy,
        *,
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._default_headers = {"Content-Type": "application/json", **(default_headers or {})}
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    async def post(self, path: str, body: dict) -> dict:
        url = f"{self._base_url}{path}"
        headers = self._auth.apply(dict(self._default_headers))
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(url, json=body, headers=headers)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                raise LLMConnectionError(f"transport failure after {self._max_retries} retries: {exc}") from exc

            if response.status_code < 400:
                try:
                    return response.json()
                except Exception as exc:  # noqa: BLE001
                    raise LLMResponseParseError(f"non-JSON 2xx response: {response.text[:200]}") from exc

            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                retry_after = response.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else 2**attempt
                except ValueError:
                    wait = 2**attempt
                await asyncio.sleep(wait)
                continue
            _raise_for_status(response)  # non-retryable, or final retryable attempt
        # Unreachable: every loop path either returns, raises, or continues; safety net.
        raise LLMServerError(f"Max retries exceeded ({self._max_retries}) for {url}")

    async def close(self) -> None:
        await self._client.aclose()
