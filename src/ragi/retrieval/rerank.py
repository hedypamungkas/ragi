"""ragi/retrieval/rerank -- pluggable cross-encoder rerank stage.

Lifted from ``koboi/rag/rerank.py`` and decoupled: **zero ``koboi.*`` imports**.
A true cross-encoder reranker that wraps a :class:`BaseRetriever`, over-fetches,
and re-scores the top candidates via one of three backends:

* ``jina``   -- Jina Reranker API (default; per-token billing, large doc capacity).
* ``cohere`` -- Cohere v2 Rerank API (per-call; default model is multilingual).
* ``local``  -- BGE cross-encoder via sentence-transformers (no egress; the
  ``[rerank-local]`` extra, mirrors the optional-extra import-gate pattern).

This is a production pipeline stage -- when enabled it runs on every retrieval.
It reuses the slimmed :class:`HttpTransport` + :class:`BearerAuth` + the
:class:`LLMError` hierarchy that already live in ragi, and co-locates
the whole small concern in one module. Fail-soft like the embedding adapters:
on any provider hiccup the wrapper returns the base retriever's results
truncated to ``top_k``, stamped so the degradation is **always observable**
(``rerank:failed(<provider>,<base>)``) -- never silently reads as healthy base.

A :class:`MockReranker` (deterministic query-word overlap, pure stdlib) ships for
tests so the fail-soft / re-stamp path can be exercised without a key or network.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from ragi._internal.http import BearerAuth, HttpTransport
from ragi.errors import LLMInvalidRequestError
from ragi.retrieval.retriever import BaseRetriever
from ragi.types import RetrievalResult

_logger = logging.getLogger(__name__)

# Per-provider document-per-call caps. :class:`CrossEncoderReranker` clamps its
# over-fetch to the active provider's cap so we never exceed a batch limit
# (v1: no multi-call batching). ``mock`` and any unknown provider fall back to
# the jina cap (2048) at lookup time.
_PROVIDER_MAX_BATCH: dict[str, int] = {"jina": 2048, "cohere": 100, "local": 10_000}


def _clamp01(value: float) -> float:
    """Clamp a score to ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))


class RerankBackend(ABC):
    """Scores ``(query, document)`` pairs via a cross-encoder.

    Implementations MUST return ``None`` on any failure (network, auth, parse)
    so the wrapper can fall back to the base retriever's results -- retrieval
    never breaks on a rerank hiccup.
    """

    #: Short provider label, surfaced in ``retrieval_method`` (e.g. ``rerank:jina(...)``).
    provider: str = "cross"

    @abstractmethod
    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        """Return ``[(original_index, score in [0,1]), ...]`` sorted desc
        (``len <= top_n``), or ``None`` on failure."""
        ...

    async def close(self) -> None:  # noqa: B027 - intentional optional override
        """Release HTTP transports / models. Default no-op; HTTP backends override."""
        ...


class JinaRerankBackend(RerankBackend):
    """Jina Reranker API: ``POST /rerank`` -> ``{results:[{index, relevance_score, document}]}``."""

    provider = "jina"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jina-reranker-v3",
        base_url: str = "https://api.jina.ai/v1",
        timeout: float = 30.0,
    ):
        self._model = model
        self._transport = HttpTransport(base_url, BearerAuth(api_key), timeout=timeout)

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        try:
            data = await self._transport.post(
                "/rerank",
                {"model": self._model, "query": query, "documents": documents, "top_n": top_n},
            )
            return _parse_rerank_results(data)
        except Exception as e:  # noqa: BLE001 - fail-soft, mirror embedding adapters
            _logger.warning("Jina rerank failed: %s", e)
            return None

    async def close(self) -> None:
        await self._transport.close()


class CohereRerankBackend(RerankBackend):
    """Cohere v2 Rerank: ``POST /rerank`` -> ``{results:[{index, relevance_score}]}`` (no doc echo).

    Default model is multilingual -- override with ``rerank-english-v3.0`` for an
    English-specialized edge.
    """

    provider = "cohere"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-multilingual-v3.0",
        base_url: str = "https://api.cohere.com/v2",
        timeout: float = 30.0,
    ):
        self._model = model
        self._transport = HttpTransport(base_url, BearerAuth(api_key), timeout=timeout)

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        try:
            data = await self._transport.post(
                "/rerank",
                {"model": self._model, "query": query, "documents": documents, "top_n": top_n},
            )
            return _parse_rerank_results(data)
        except Exception as e:  # noqa: BLE001 - fail-soft, mirror embedding adapters
            _logger.warning("Cohere rerank failed: %s", e)
            return None

    async def close(self) -> None:
        await self._transport.close()


class LocalBGERerankBackend(RerankBackend):
    """Local BGE cross-encoder via sentence-transformers (logits -> sigmoid).

    Heavy: pulls torch. Import-gated behind the ``[rerank-local]`` extra (mirrors
    the ``[tokenizer]``/tiktoken pattern) so the default install stays lean.
    """

    provider = "local"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        try:
            from sentence_transformers import CrossEncoder  # import-gated ([rerank-local] extra)
        except ImportError as e:
            raise LLMInvalidRequestError(
                "(rerank-local) sentence-transformers required: pip install 'ragi[rerank-local]'"
            ) from e
        self._model_name = model
        self._model = CrossEncoder(model)  # sync; rerank() runs predict() off-loop

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        try:
            pairs = [[query, d] for d in documents]
            # predict() is CPU-bound -> run off the event loop.
            scores = await asyncio.to_thread(self._model.predict, pairs)
            ranked = sorted(enumerate(list(scores)), key=lambda x: x[1], reverse=True)[:top_n]
            return [(i, _sigmoid(float(s))) for i, s in ranked]
        except Exception as e:  # noqa: BLE001 - fail-soft, mirror embedding adapters
            _logger.warning("Local BGE rerank failed: %s", e)
            return None


class MockReranker(RerankBackend):
    """Deterministic reranker for tests (pure stdlib, no key, no network).

    Scores each document by the fraction of distinct query words it contains,
    so the result is in ``[0, 1]`` and the ranking is stable and predictable.
    """

    provider = "mock"

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            # No signal -> preserve original order with a zero score.
            return [(i, 0.0) for i in range(min(top_n, len(documents)))]
        denom = len(query_words)
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            doc_words = set(re.findall(r"\w+", doc.lower()))
            overlap = len(query_words & doc_words)
            scored.append((i, _clamp01(overlap / denom)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


def build_rerank_client(conf: dict | None) -> RerankBackend | None:
    """Build a rerank backend from a config dict.

    Mirrors the embedding client's None-on-unconfigured contract: for HTTP
    providers (jina/cohere) returns ``None`` when no ``api_key`` is set so the
    caller uses the base retriever **unwrapped (no reranking)** with a warning.
    The ``local`` provider is built regardless of ``api_key`` (it needs no key).
    Unknown providers raise :class:`LLMInvalidRequestError` at build time
    (fail-fast, validated BEFORE the ``api_key`` gate so a typo'd provider name
    is caught even when the key is also missing). ``provider`` defaults to
    ``jina``.
    """
    cfg = conf or {}
    if not isinstance(cfg, dict):
        return None
    provider = str(cfg.get("provider") or "jina").lower()
    timeout = float(cfg.get("timeout", 30.0))

    if provider == "local":
        return LocalBGERerankBackend(model=cfg.get("model") or "BAAI/bge-reranker-v2-m3")

    # Validate the provider name BEFORE the api_key gate -- fail-fast on a typo
    # (a bogus/misspelt provider must raise, not silently fall back via the
    # missing-key branch).
    if provider not in ("jina", "cohere"):
        raise LLMInvalidRequestError(f"Unknown rerank provider: {provider!r}. Available: jina, cohere, local.")

    api_key = cfg.get("api_key") or os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not api_key:
        _logger.warning(
            "rerank provider %r has no api_key; cross-encoder rerank disabled (falling back).",
            provider,
        )
        return None

    if provider == "jina":
        return JinaRerankBackend(
            api_key=api_key,
            model=cfg.get("model") or "jina-reranker-v3",
            base_url=cfg.get("base_url") or "https://api.jina.ai/v1",
            timeout=timeout,
        )
    return CohereRerankBackend(
        api_key=api_key,
        model=cfg.get("model") or "rerank-multilingual-v3.0",
        base_url=cfg.get("base_url") or "https://api.cohere.com/v2",
        timeout=timeout,
    )


class CrossEncoderReranker(BaseRetriever):
    """Wraps a base retriever, over-fetches, and re-scores via a cross-encoder backend.

    Same wrapper shape as koboi's heuristic ``RerankerRetriever``: delegates the
    over-fetch to ``self._base.retrieve(...)`` (which carries its own
    ``_chunks``) so metadata filtering stays base-retriever-owned; this wrapper
    does NOT set ``self._chunks``. Fail-soft: if the backend returns
    ``None``/empty, returns the base results (original order, truncated to
    ``top_k``) so retrieval never breaks. The degradation is **always
    observable** -- results are stamped
    ``retrieval_method="rerank:failed(<provider>,<base>)"`` plus a warning log,
    so a persistent outage (bad key / quota / bad model) never silently reads
    as healthy base retrieval. On success each result is stamped
    ``rerank:<provider>(<base_method>)`` so evals can detect the rerank provider.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        backend: RerankBackend,
        fetch_multiplier: int = 3,
        score_threshold: float | None = None,
    ):
        self._base = base_retriever
        self._backend = backend
        self._fetch_multiplier = fetch_multiplier
        self._score_threshold = score_threshold
        self._provider = backend.provider

    async def retrieve(
        self, query: str, *, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[RetrievalResult]:
        cap = _PROVIDER_MAX_BATCH.get(self._provider, 2048)
        fetch_k = min(max(top_k * self._fetch_multiplier, top_k + 5), cap)
        results = await self._base.retrieve(query, top_k=fetch_k, metadata_filter=metadata_filter)

        if not results:
            # Nothing to rank. (We do NOT bail when ``len(results) <= top_k``: a
            # lexical base like KeywordRetriever drops zero-overlap docs, so a
            # small-but-relevant result set still warrants reranking -- it re-scores
            # with the cross-encoder signal and, crucially, re-stamps
            # ``retrieval_method`` so the rerank stage / any fail-soft degradation
            # stays OBSERVABLE to evals. Bailing early here would silently mask
            # the rerank stage whenever the base is sparse.)
            return results

        documents = [r.chunk.content for r in results]
        ranked = await self._backend.rerank(query, documents, top_n=top_k)
        if not ranked:  # backend failed/empty -> fail-soft to base order (ALWAYS observable)
            base_method = results[0].retrieval_method
            _logger.warning(
                "Rerank backend %s returned no usable results; returning base results (fail-soft).",
                self._provider,
            )
            method = f"rerank:failed({self._provider},{base_method})"
            return [RetrievalResult(chunk=r.chunk, score=r.score, retrieval_method=method) for r in results[:top_k]]

        out: list[RetrievalResult] = []
        for idx, score in ranked:
            if idx < 0 or idx >= len(results):
                continue  # defensive: provider returned a stale/out-of-range index
            if self._score_threshold is not None and score < self._score_threshold:
                continue
            base = results[idx]
            out.append(
                RetrievalResult(
                    chunk=base.chunk,
                    score=score,
                    retrieval_method=f"rerank:{self._provider}({base.retrieval_method})",
                )
            )
        # If everything was filtered out by score_threshold, fall back to base order.
        return out[:top_k] if out else results[:top_k]

    async def close(self) -> None:
        """Close the backend's HTTP transport(s). Idempotent; safe to call at shutdown."""
        await self._backend.close()


# ---------------------------------------------------------------------------
# Shared parse / score helpers
# ---------------------------------------------------------------------------


def _int(value: Any) -> int:
    """Coerce a JSON index to int (providers may return it as a numpy/float-ish value)."""
    return int(value)


def _parse_rerank_results(data: dict) -> list[tuple[int, float]]:
    """Parse a Jina/Cohere rerank response into ``[(index, score in [0,1])]``.

    Tolerates a single malformed row (missing/non-numeric ``index`` or
    ``relevance_score``) by skipping it rather than discarding the whole batch
    -- a provider returning 99 good scores + 1 bad row keeps the 99. Both HTTP
    backends share this parser.
    """
    out: list[tuple[int, float]] = []
    for r in data.get("results", []):
        try:
            out.append((_int(r["index"]), _clamp01(r.get("relevance_score", 0.0))))
        except (KeyError, TypeError, ValueError):
            continue  # skip one malformed row; keep the rest
    return out


def _sigmoid(x: float) -> float:
    """Logits -> ``[0,1]`` probability (numerically stable), clamped."""
    if x >= 0:
        z = math.exp(-x)
        return _clamp01(1.0 / (1.0 + z))
    z = math.exp(x)
    return _clamp01(z / (1.0 + z))
