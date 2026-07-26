"""ragworkbench/embeddings/openai -- OpenAI-compatible embedding adapter.

Uses :class:`ragworkbench._internal.http.HttpTransport` (POST-with-retry + Bearer
auth) so an embedding backend can be configured without pulling in the OpenAI SDK.
Satisfies :class:`ragworkbench.protocols.EmbeddingClient`.

Fail-soft contract: :meth:`embed` returns ``None`` on ANY error (transport, auth,
parse, missing key) so the calling retriever degrades to lexical (keyword) without
raising -- a retrieval stage must never break the run over a provider hiccup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ragworkbench._internal.http import BearerAuth, HttpTransport
from ragworkbench.errors import LLMError

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient:
    """OpenAI-compatible ``/embeddings`` client (fail-soft).

    ``base_url=None`` targets the public OpenAI API; point it at an OpenAI-compatible
    gateway (Azure OpenAI, Together, a local server) otherwise. :meth:`embed`
    returns ``None`` on any failure so Semantic/Hybrid retrievers degrade to keyword.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._transport = HttpTransport(
            base_url or "https://api.openai.com/v1",
            BearerAuth(api_key),
            timeout=timeout,
        )

    async def embed(self, text: str) -> list[float] | None:
        """Embed a single text. Returns ``None`` on any failure (fail-soft)."""
        try:
            data = await self._transport.post(
                "/embeddings",
                {"model": self._model, "input": text},
            )
        except LLMError as exc:
            _logger.warning("OpenAIEmbeddingClient: embed failed (%s); returning None", exc)
            return None
        return self._parse_at(data, 0)

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch in one call. Fail-soft per item: a parse failure -> ``None`` for that slot.

        On a transport-level failure the whole batch returns ``[None] * len(texts)``
        (each caller leg degrades independently). OpenAI returns ``data`` as a list
        whose order follows the input list.
        """
        if not texts:
            return []
        try:
            data = await self._transport.post(
                "/embeddings",
                {"model": self._model, "input": texts},
            )
        except LLMError as exc:
            _logger.warning("OpenAIEmbeddingClient: embed_batch failed (%s); returning all-None", exc)
            return [None] * len(texts)
        return [self._parse_at(data, i) for i in range(len(texts))]

    @staticmethod
    def _parse_at(data: object, index: int = 0) -> list[float] | None:
        """Extract the ``index``-th embedding from an OpenAI ``/embeddings`` response.

        Returns ``None`` on any shape drift (missing key, wrong type, short list)
        rather than raising.
        """
        try:
            emb = data["data"][index]["embedding"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            _logger.warning(
                "OpenAIEmbeddingClient: malformed embedding response (index=%s); returning None",
                index,
            )
            return None
        if not isinstance(emb, list):
            return None
        return [float(x) for x in emb]

    async def close(self) -> None:
        await self._transport.close()
