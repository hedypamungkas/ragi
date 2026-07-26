"""ragworkbench/embeddings/mock -- deterministic bag-of-words embedding for tests.

No API key, no network. :meth:`embed` returns a 128-dim bag-of-words vector:
each token's ``hash() % 128`` bucket is incremented (float counts), so identical
or token-overlapping texts score high cosine -- :class:`SemanticRetriever` then
behaves like a deterministic token-overlap ranker. Never returns ``None``.

Satisfies :class:`ragworkbench.protocols.EmbeddingClient`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_DIM = 128


class MockEmbeddingClient:
    """Deterministic bag-of-words embedder (no key, no network).

    ``embed`` never returns ``None``. Identical texts produce identical vectors;
    overlapping tokens raise cosine. Within one process, token->bucket is stable,
    so relative ranking is deterministic (a Semantic/Hybrid retriever using this
    client ranks like a token-overlap counter).
    """

    _DIM = _DIM

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._DIM
        for token in re.findall(r"\w+", text.lower()):
            vec[hash(token) % self._DIM] += 1.0
        return vec

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self._vector(t) for t in texts]
