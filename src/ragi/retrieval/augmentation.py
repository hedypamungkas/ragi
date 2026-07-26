"""ragi/retrieval/augmentation -- context-injection strategies (agent seam).

Lifted from ``koboi/rag/augmentation.py`` and decoupled: **zero ``koboi.*`` imports**.
The augmentation's job is purely: retrieve context -> format with numbered citations
-> inject into the user message (``in_memory``) or the LLM messages list (``on_the_fly``).

Design change vs koboi: koboi's augmentation ALSO did query-rewrite internally (the
``query_rewrite``/``hyde``/``rewrite_client``/``rewrite_config`` params + an internal
``_maybe_rewrite`` step). In ragi rewrite lives in a separate
``RewritingRetriever`` wrapper (a measurable retrieval stage) -- so this augmentation
does NOT rewrite (would double-rewrite). All rewrite params are dropped.

Two built-in strategies, self-registering on import via
:func:`ragi.registry.register_augmentation` so a plain
``from ragi.retrieval.augmentation import ...`` is enough to populate
:data:`augmentation_registry`:

- ``in_memory``: augments the user message before it's stored in conversation memory
  (the stored row carries the context, so later turns see it).
- ``on_the_fly``: augments the last user message in-place before each LLM call, keeping
  stored memory clean; caches the formatted context per user-content.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC
from typing import TYPE_CHECKING

from ragi.registry import register_augmentation

if TYPE_CHECKING:
    from ragi.retrieval.retriever import BaseRetriever
    from ragi.types import RetrievalResult

_logger = logging.getLogger(__name__)

# Injected into the augmented user message when retrieval returned NO usable context
# (no hits, or the relevance_threshold / dedup collapsed everything to empty). An
# anti-fabrication cue: without it the LLM receives a bare question with zero signal
# that retrieval failed and may fabricate confidently. The marker rides through
# ``_build_augmented_message`` so the LLM sees it inside the standard "Document
# context" block (a stronger abstention cue than a bare prefix, and consistent with
# non-empty turns). Default-ON.
ABSTENTION_MARKER = (
    "[RETRIEVAL_EMPTY] No relevant context was retrieved for this query. "
    "Do not fabricate, infer, or speculate from parametric knowledge. "
    "If you cannot answer from prior conversation, state that you do not have "
    "enough information."
)


class AugmentationStrategy(ABC):  # noqa: B024 - registry type marker; methods have default no-op impls
    """Base class for RAG context-injection strategies.

    Subclasses override :meth:`augment_for_memory` (``in_memory``) and/or
    :meth:`augment_for_llm` (``on_the_fly``). The shared :meth:`_retrieve_and_format`
    runs retrieval, applies the relevance gate + content-hash dedup, and formats the
    surviving chunks as a numbered-citation context block.

    ``last_results`` is overwritten on each retrieval call (multi-turn safe -- an
    assignment, not accumulation) so a caller can stamp it onto a run-result for
    eval/observability.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 3,
        relevance_threshold: float | None = None,
        metadata_filter: dict | None = None,
        logger: logging.Logger | None = None,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.metadata_filter = metadata_filter  # relevance scoping (NOT an ACL boundary)
        self._logger: logging.Logger = logger or _logger
        # Last retrieved chunks: overwritten each call so this reflects the latest
        # retrieval (multi-turn safe). Surfaced for run-result/eval stamping.
        self.last_results: list[RetrievalResult] = []

    async def _retrieve_and_format(self, query: str) -> tuple[str, list[RetrievalResult]]:
        """Retrieve, gate, dedup, and format -> ``(context, results)``.

        Returns ``(ABSTENTION_MARKER, [])`` when no results survive so the LLM
        receives an explicit anti-fabrication cue instead of a bare question.
        """
        results = await self.retriever.retrieve(query, top_k=self.top_k, metadata_filter=self.metadata_filter)

        # Relevance gate: drop results below threshold. Score semantics differ per
        # retrieval method (keyword = TF-IDF cosine, bm25 = Okapi, ...) -- comparable
        # only within a method, not across.
        if self.relevance_threshold is not None:
            results = [r for r in results if r.score >= self.relevance_threshold]

        # Dedup by content hash (keep first occurrence) so overlapping chunks or
        # duplicate files don't inject the same passage twice.
        seen: set[str] = set()
        deduped: list[RetrievalResult] = []
        for r in results:
            h = hashlib.sha256(r.chunk.content.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            deduped.append(r)
        results = deduped

        # Overwrite each call -- reflects the latest retrieval (multi-turn safe).
        self.last_results = list(results)

        if not results:
            return ABSTENTION_MARKER, []

        # Numbered citations [1] [2] ... so the model can echo references back.
        parts: list[str] = []
        for i, r in enumerate(results, start=1):
            source = r.chunk.metadata.get("source", r.chunk.doc_id)
            parts.append(f"[{i}] [Source: {source}]\n{r.chunk.content}")
        context = "\n---\n".join(parts)

        self._logger.info(
            "rag augmentation: query=%r method=%s hits=%d",
            query,
            results[0].retrieval_method,
            len(results),
        )
        return context, results

    @staticmethod
    def _build_augmented_message(user_message: str, context: str) -> str:
        return f"Document context:\n---\n{context}\n---\n\nQuestion: {user_message}"

    async def augment_for_memory(self, user_message: str) -> str:
        """Default no-op. ``in_memory`` strategies override to inject context here."""
        return user_message

    async def augment_for_llm(self, messages: list[dict]) -> list[dict]:
        """Default no-op. ``on_the_fly`` strategies override to rewrite messages here."""
        return messages


@register_augmentation(
    "in_memory",
    description="Augment the user message with retrieved context before storing in memory",
)
class InMemoryAugmentation(AugmentationStrategy):
    """Retrieve context and fold it into the user message before it's stored.

    The stored conversation row carries the context, so subsequent turns see it
    (persistent augmentation). Use ``on_the_fly`` instead to keep memory clean.

    On an empty retrieval (``ABSTENTION_MARKER``) the message is returned unchanged
    so the bare question is stored and the downstream LLM call applies its own
    abstention handling.
    """

    async def augment_for_memory(self, user_message: str) -> str:
        context, _ = await self._retrieve_and_format(user_message)
        if context == ABSTENTION_MARKER:
            return user_message
        return self._build_augmented_message(user_message, context)


@register_augmentation(
    "on_the_fly",
    description="Augment the last user message in-place before each LLM call",
)
class OnTheFlyAugmentation(AugmentationStrategy):
    """Retrieve context and rewrite the last user message in-place per LLM call.

    Keeps stored memory clean (the context never enters conversation history) and
    re-retrieves as the query evolves across turns. Caches the formatted context per
    user-content so a repeated message (e.g. a retried turn) doesn't re-retrieve.

    On an empty retrieval (``ABSTENTION_MARKER``) the messages list is returned
    unmodified so the bare question goes to the LLM (which applies its own
    abstention handling).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 3,
        relevance_threshold: float | None = None,
        metadata_filter: dict | None = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            retriever=retriever,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            metadata_filter=metadata_filter,
            logger=logger,
        )
        self._cache: dict[str, str] = {}

    async def augment_for_llm(self, messages: list[dict]) -> list[dict]:
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return messages

        user_content = messages[last_user_idx].get("content", "")

        if user_content in self._cache:
            context = self._cache[user_content]
        else:
            context, _ = await self._retrieve_and_format(user_content)
            self._cache[user_content] = context

        if context == ABSTENTION_MARKER:
            return messages

        messages[last_user_idx]["content"] = self._build_augmented_message(user_content, context)
        return messages
