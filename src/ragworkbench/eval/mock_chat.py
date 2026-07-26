"""ragworkbench/eval/mock_chat -- scripted ``ChatClient`` for deterministic Mode B tests.

Zero-key stand-in for a real ``ChatClient`` so the full Mode B loop (decompose → batch-NLI
→ coverage → score → rubric → CI) runs deterministically in pytest. Two feeding modes:

- ``patterns``: a ``{substring: reply}`` dict — the first substring found in the prompt
  wins. Best for scorers whose prompts carry distinctive phrases (e.g. the faithfulness
  decompose vs NLI prompts).
- ``responses``: a FIFO list popped in call order.
``patterns`` takes precedence; falls back to ``responses``; then ``default``.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


class MockChatClient:
    """Deterministic ChatClient for tests. Records every call in ``self.calls``."""

    def __init__(
        self,
        responses: list[str] | None = None,
        patterns: dict[str, str] | None = None,
        default: str = "",
    ):
        self._responses = list(responses or [])
        self._patterns = dict(patterns or {})
        self._default = default
        self.calls: list[list[dict]] = []

    async def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        prompt = "".join(str(m.get("content", "")) for m in messages)
        for substr, reply in self._patterns.items():
            if substr in prompt:
                return reply
        if self._responses:
            return self._responses.pop(0)
        return self._default
