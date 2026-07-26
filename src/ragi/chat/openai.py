"""ragi/chat/openai -- OpenAI-compatible ChatClient (single-turn completion).

Satisfies the :class:`ChatClient` Protocol via the inlined :class:`HttpTransport`
(``POST /chat/completions``). Works against OpenAI or any OpenAI-compatible gateway
(``base_url`` override, e.g. a local vLLM/LiteLLM/Koboi gateway). Used by Mode B eval
(faithfulness judge + answer generation) -- raises the ``LLM*`` hierarchy on transport
errors; callers (the runner / scorers) catch and fail-soft.
"""

from __future__ import annotations

import logging

from ragi._internal.http import BearerAuth, HttpTransport

_logger = logging.getLogger(__name__)


class OpenAIChatClient:
    """OpenAI-compatible chat client (``ChatClient`` Protocol)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
    ):
        self._model = model
        self._temperature = temperature
        self._transport = HttpTransport(base_url or "https://api.openai.com/v1", BearerAuth(api_key), timeout=timeout)

    async def complete(self, messages: list[dict]) -> str:
        data = await self._transport.post(
            "/chat/completions",
            {"model": self._model, "messages": messages, "temperature": self._temperature},
        )
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    async def close(self) -> None:
        await self._transport.close()
