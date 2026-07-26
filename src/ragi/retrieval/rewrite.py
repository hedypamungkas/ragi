"""ragi/retrieval/rewrite -- query rewriting + HyDE as a measurable Retriever stage.

Lifted from ``koboi/rag/rewrite.py`` and decoupled: **zero ``koboi.*`` imports**.
Two surfaces that koboi buries inside its augmentation strategy are exposed here as
first-class, instrumented stages:

- :class:`QueryRewriter` -- rule-based normalization (always-on, deterministic,
  zero-cost) plus optional LLM rewrite / HyDE, with FIFO caching keyed by
  ``(mode, query)`` and graceful fallback to the rule-normalized (or raw) query on
  any failure.
- :class:`RewritingRetriever` -- a :class:`~ragi.retrieval.retriever.BaseRetriever`
  wrapper that rewrites the query, delegates retrieval to the wrapped base, and exposes
  the rewrite decision on ``self.last_rewrite`` for eval/observability. Koboi has no
  equivalent -- it hides the rewrite inside the augmentation; we lift it out so the
  rewrite can be measured, ablated, and compared like any other Retriever stage, WITHOUT
  mutating the inner retriever's ``retrieval_method`` stamp.

A :class:`MockChatClient` is included for tests/offline runs (no API key).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from ragi.retrieval.retriever import BaseRetriever

if TYPE_CHECKING:
    from ragi.protocols import ChatClient
    from ragi.types import RetrievalResult

_logger = logging.getLogger(__name__)

__all__ = [
    "QueryRewriter",
    "RewritingRetriever",
    "MockChatClient",
    "rule_based_rewrite",
    "QUERY_REWRITE_PROMPT",
    "HYDE_PROMPT",
]


# Conservative filler removal -- preserves all content terms. Matches the spirit of the
# query-side synonym expansion already in KeywordRetriever, generalized to every retriever.
_STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be been being do does did "
    "i you he she it we they me him her us them my your his its our their this that "
    "these those with from by as can could would should will shall please hey um like".split()
)

QUERY_REWRITE_PROMPT = (
    "Rewrite the following user question into a concise, search-optimized query for a "
    "document retrieval system. Preserve the core intent and key entities (names, numbers, "
    "terms). Remove greetings, filler, and conversational language. Do NOT answer the "
    "question. Output ONLY the rewritten query.\n\nQuestion: {query}\n\nRewritten query:"
)

HYDE_PROMPT = (
    "Write a short (2-3 sentence) hypothetical answer to the following question, as if you "
    "had the relevant document in front of you. It will be used ONLY to find similar "
    "documents via embedding similarity -- do not hedge, just write a plausible, factual "
    "answer. Output ONLY the answer.\n\nQuestion: {query}\n\nAnswer:"
)

_MAX_QUERY_CHARS = 1000  # cap to bound the rewrite prompt (prompt-injection surface)


def rule_based_rewrite(query: str) -> str:
    """Deterministic normalization: lowercase, drop filler stopwords, collapse whitespace.

    Conservative: only removes obvious filler, keeps every content term. Always safe.
    Falls back to the stripped original query when every token is a stopword (so the
    result is never empty).
    """
    tokens = re.findall(r"\w+", query.lower())
    kept = [t for t in tokens if t not in _STOPWORDS]
    text = " ".join(kept) if kept else query.strip()
    return re.sub(r"\s+", " ", text).strip()


class QueryRewriter:
    """Opt-in query rewriting (rule-based + LLM) and HyDE, with caching + fallback.

    ``client`` follows the :class:`~ragi.protocols.ChatClient` protocol
    (``await client.complete(messages) -> str``). When ``client`` is ``None`` or any
    LLM call fails, the rule-normalized query is used instead (or the raw query if
    ``fallback_to_raw`` is disabled) -- retrieval never breaks because of a rewrite
    hiccup. The returned ``meta`` dict ``{original, rewritten, method}`` is surfaced
    for eval/observability (e.g. stamped onto a result's metadata).
    """

    def __init__(
        self,
        client: ChatClient | None = None,
        config: dict | None = None,
    ) -> None:
        self._client = client
        # Cache is keyed by (mode, query) so the same query under different modes
        # (e.g. "llm" vs "hyde") caches separately -- koboi keyed on query alone.
        self._cache: dict[tuple[str, str], str] = {}
        self._max_cache = int((config or {}).get("query_cache_size", 256))
        self._fallback = bool((config or {}).get("fallback_to_raw", True))

    async def rewrite(self, query: str, *, mode: str = "llm") -> tuple[str, dict]:
        """Return ``(effective_query, meta)``.

        ``mode`` in ``{"rule", "llm", "hyde"}``. ``meta`` = ``{original, rewritten, method}``
        where ``method`` is one of ``rule`` / ``llm`` / ``hyde`` / ``cache`` / ``rule-fallback``.
        """
        normalized = rule_based_rewrite(query)

        if mode == "rule":
            return normalized or query, {"original": query, "rewritten": normalized, "method": "rule"}

        # mode is "llm" or "hyde" -- both need a chat client.
        if self._client is None:
            _logger.warning(
                "rewrite mode %r requested but no chat client configured; using rule-based result",
                mode,
            )
            return normalized or query, {"original": query, "rewritten": normalized, "method": "rule"}

        cache_key = (mode, query)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached, {"original": query, "rewritten": cached, "method": "cache"}

        template = HYDE_PROMPT if mode == "hyde" else QUERY_REWRITE_PROMPT
        prompt = template.format(query=query[:_MAX_QUERY_CHARS])
        try:
            text = await self._client.complete([{"role": "user", "content": prompt}])
        except Exception as exc:  # network / provider -> fallback path handles it
            _logger.warning("query rewrite LLM call failed: %s", exc)
            text = None
        text = text.strip() if text else None

        if not text:
            # LLM unavailable/failed/empty -> fall back to the rule-normalized (or raw) query.
            effective = (normalized or query) if self._fallback else query
            return effective, {"original": query, "rewritten": effective, "method": "rule-fallback"}

        if len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)), None)  # FIFO evict
        self._cache[cache_key] = text
        return text, {"original": query, "rewritten": text, "method": mode}


class RewritingRetriever(BaseRetriever):
    """:class:`~ragi.retrieval.retriever.BaseRetriever` wrapper that rewrites first.

    Rewrites the query via :class:`QueryRewriter`, delegates retrieval to ``base_retriever``,
    and exposes the rewrite decision on ``self.last_rewrite`` (a ``{original, rewritten,
    method}`` dict, or ``None`` before the first call). The base retriever's results and
    ``retrieval_method`` stamps are left UNCHANGED -- rewrite visibility lives entirely on
    ``self.last_rewrite`` so the inner stage's provenance stays honest.

    Koboi has no equivalent: it buries the rewrite inside the augmentation. Exposing it as
    a Retriever-stage wrapper makes the rewrite measurable, ablatable, and comparable
    against a plain base retriever in an ablation.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        chat_client: ChatClient | None = None,
        mode: str = "llm",
        config: dict | None = None,
    ) -> None:
        self._base = base_retriever
        self._rewriter = QueryRewriter(client=chat_client, config=config)
        self._mode = mode
        self.last_rewrite: dict | None = None

    async def retrieve(
        self, query: str, *, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[RetrievalResult]:
        effective, meta = await self._rewriter.rewrite(query, mode=self._mode)
        self.last_rewrite = meta
        return await self._base.retrieve(effective, top_k=top_k, metadata_filter=metadata_filter)


class MockChatClient:
    """Scripted or transform-based :class:`~ragi.protocols.ChatClient` for tests.

    Two modes (mutually exclusive; ``responses`` wins when given):

    - ``responses`` (a list of ``str``): ``complete()`` pops the next response in order.
    - ``transform`` (a ``str -> str`` callable): applied to the last user message's
      content. Default: ``lambda s: s.replace("?", "").strip() + " simplified"``.

    Returns ``str`` directly (matches the :class:`ChatClient` protocol). No API key.
    """

    def __init__(
        self,
        transform=None,
        responses: list[str] | None = None,
    ) -> None:
        if responses is not None:
            self._responses: list[str] | None = list(responses)
            self._transform = None
        else:
            self._responses = None
            self._transform = transform or (lambda s: s.replace("?", "").strip() + " simplified")

    async def complete(self, messages: list[dict]) -> str:
        if self._responses is not None:
            if not self._responses:
                raise IndexError("MockChatClient: no more scripted responses")
            return self._responses.pop(0)
        prompt = messages[-1]["content"] if messages else ""
        return self._transform(prompt)
