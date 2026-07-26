"""ragworkbench/vectorstore/faiss -- FAISS-backed VectorStore (cosine via IndexFlatIP).

Import-gated behind the ``vector-faiss`` extra (``faiss-cpu``; ``numpy`` ships with it).
Builds an in-process ``faiss.IndexFlatIP`` over **L2-normalized** vectors -- inner product
on unit vectors equals cosine similarity, so the returned scores are directly comparable
to :class:`InMemoryVectorStore`'s cosine. A flat (brute-force) index is O(n) per query
(same as the memory backend) but benefits from FAISS's SIMD kernels; for sub-linear ANN
swap in an IVF/HNSW index -- the add/search contract stays the same.

FAISS has no native metadata layer, so ``filter`` is applied **post-ANN** in Python via
:func:`ragworkbench.ingest.filters.matches_filter` (same predicate the memory backend uses).
When a filter is present we over-fetch the full ranking (k = ``ntotal``) so the filter-
before-cut semantics match the reference; without a filter we ask FAISS for exactly
``top_k``. Dimension is inferred from the first added chunk if ``dim`` is not supplied.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ragworkbench.errors import LLMInvalidRequestError
from ragworkbench.ingest.filters import matches_filter
from ragworkbench.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from ragworkbench.protocols import EmbeddingClient

_logger = logging.getLogger(__name__)
_METHOD = "vectorstore:faiss"

try:
    import faiss  # type: ignore[import-not-found]
    import numpy  # numpy is bundled with faiss-cpu  # type: ignore[import-not-found]

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - exercised via the construct-time gate
    faiss = None  # type: ignore[assignment]
    numpy = None  # type: ignore[assignment]
    _HAS_FAISS = False


_INSTALL_HINT = "pip install 'ragworkbench[vector-faiss]'"


def _require_faiss() -> None:
    if not _HAS_FAISS:
        raise LLMInvalidRequestError(f"(vector-faiss) faiss-cpu required: {_INSTALL_HINT}")


class FaissVectorStore:
    """FAISS ``IndexFlatIP`` backend. Cosine via L2-normalized inner product.

    Holds a parallel ``list[Chunk]`` so metadata filters can be applied post-ANN --
    FAISS indices are addressable only by integer position.
    """

    def __init__(
        self,
        embedder: EmbeddingClient | None = None,
        dim: int | None = None,
    ):
        _require_faiss()
        self._embedder = embedder
        self._dim = dim
        self._index: Any = None  # built lazily on first add when dim is unknown
        self._chunks: list[Chunk] = []

    @staticmethod
    def _normalize_rows(mat: Any) -> Any:
        """L2-normalize each row in place semantically; guards against zero-norm rows."""
        norms = numpy.linalg.norm(mat, axis=1, keepdims=True)  # type: ignore[union-attr]
        norms[norms < 1e-10] = 1.0
        return mat / norms

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        to_embed = [c for c in chunks if c.embedding is None]
        if to_embed and self._embedder is not None:
            vectors = await self._embedder.embed_batch([c.content for c in to_embed])
            for c, v in zip(to_embed, vectors, strict=True):
                c.embedding = v
        # drop chunks whose embedding still isn't resolvable
        usable = [c for c in chunks if c.embedding is not None]
        if not usable:
            return
        if self._dim is None:
            self._dim = len(usable[0].embedding)  # type: ignore[arg-type]
        mat = numpy.asarray(  # type: ignore[union-attr]
            [c.embedding for c in usable], dtype="float32"
        )
        mat = self._normalize_rows(mat)
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)  # type: ignore[union-attr]
        self._index.add(mat)  # type: ignore[union-attr]
        self._chunks.extend(usable)

    async def search(
        self,
        query: str | list[float] | None,
        *,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        if self._index is None or not self._chunks:
            return []
        if isinstance(query, str):
            if self._embedder is None:
                return []
            query = await self._embedder.embed(query)
        if query is None:
            return []
        qv = numpy.asarray([query], dtype="float32")  # type: ignore[union-attr]
        qv = self._normalize_rows(qv)
        # Filter present -> over-fetch the whole ranking so filter-before-cut matches the
        # reference; FAISS IndexFlatIP is exhaustive either way (no sub-linear loss).
        k = self._index.ntotal if filter else min(top_k, self._index.ntotal)
        if k <= 0:
            return []
        scores, indices = self._index.search(qv, k)  # type: ignore[union-attr]
        out: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if len(out) >= top_k:
                break
            if idx < 0 or idx >= len(self._chunks):
                continue  # FAISS pads with -1 when k > ntotal
            chunk = self._chunks[idx]
            if filter and not matches_filter(chunk.metadata, filter):
                continue
            out.append(RetrievalResult(chunk=chunk, score=float(score), retrieval_method=_METHOD))
        return out

    @property
    def count(self) -> int:
        return len(self._chunks)
