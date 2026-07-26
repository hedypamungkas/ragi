"""ragi/errors -- exception hierarchy.

The ``LLM*`` subset is an inlined, slimmed version of koboi's LLMError hierarchy
(used by the v0.2+ embed/rerank/rewrite adapters). It lives here so the core lib has
zero dependency on any provider SDK.
"""

from __future__ import annotations


class RagError(Exception):
    """Base for all ragi errors."""


class ConfigError(RagError):
    """Invalid pipeline or eval configuration."""


class RetrievalError(RagError):
    """Retrieval-stage failure."""


class EvalError(RagError):
    """Evaluation-stage failure."""


# --- inlined LLM-error subset (v0.2+ adapters) -----------------------------


class LLMError(RagError):
    """Base for LLM / embedding / rerank provider errors."""


class LLMInvalidRequestError(LLMError):
    """Bad request to a provider (e.g. unknown rerank provider, bad model id)."""


class LLMConnectionError(LLMError):
    """Network / transport failure reaching a provider."""


class LLMAuthenticationError(LLMError):
    """Provider rejected credentials (401/403)."""


class LLMRateLimitError(LLMError):
    """Provider rate-limited (429). Carries ``retry_after`` seconds when the header is present."""

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMServerError(LLMError):
    """Provider returned a 5xx, or retries were exhausted."""


class LLMResponseParseError(LLMError):
    """Provider response body could not be parsed as expected."""
