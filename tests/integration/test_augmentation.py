"""Integration tests for the augmentation agent seam."""

from __future__ import annotations

import asyncio

from ragi.retrieval.augmentation import (
    ABSTENTION_MARKER,
    AugmentationStrategy,
    InMemoryAugmentation,
    OnTheFlyAugmentation,
)
from ragi.retrieval.retriever import KeywordRetriever
from ragi.types import Chunk

CHUNKS = [Chunk("c1", "d1", "the refund policy allows 30 days", metadata={"source": "policy"})]


def test_inmemory_injects_cited_context():
    aug = InMemoryAugmentation(KeywordRetriever(chunks=CHUNKS), top_k=1)
    out = asyncio.run(aug.augment_for_memory("what is the refund policy?"))
    assert "[Source: policy]" in out
    assert "Document context" in out
    assert len(aug.last_results) == 1


def test_onthefly_rewrites_last_user_message():
    aug = OnTheFlyAugmentation(KeywordRetriever(chunks=CHUNKS), top_k=1)
    msgs = [{"role": "user", "content": "refund policy?"}]
    out = asyncio.run(aug.augment_for_llm(msgs))
    assert "Document context" in out[-1]["content"]


def test_abstention_when_threshold_filters_everything():
    # relevance_threshold above all scores -> empty results -> message returned unchanged.
    aug = InMemoryAugmentation(KeywordRetriever(chunks=CHUNKS), top_k=1, relevance_threshold=999.0)
    out = asyncio.run(aug.augment_for_memory("refund policy?"))
    assert out == "refund policy?"  # unchanged (abstention path)
    assert aug.last_results == []


def test_metadata_filter_scopes_retrieval():
    other = [Chunk("c2", "d2", "refund terms are strict", metadata={"source": "terms"})]
    aug = InMemoryAugmentation(KeywordRetriever(chunks=CHUNKS + other), top_k=5, metadata_filter={"source": "policy"})
    out = asyncio.run(aug.augment_for_memory("refund?"))
    assert "[Source: policy]" in out
    assert all(r.chunk.metadata.get("source") == "policy" for r in aug.last_results)


def test_dedup_collapses_identical_content():
    dupes = [
        Chunk("c1", "d1", "identical refund wording", metadata={"source": "a"}),
        Chunk("c2", "d2", "identical refund wording", metadata={"source": "b"}),
    ]
    aug = InMemoryAugmentation(KeywordRetriever(chunks=dupes), top_k=5)
    out = asyncio.run(aug.augment_for_memory("identical refund wording"))
    # content-hash dedup keeps only the first -> a single citation block
    assert out.count("[Source:") == 1


def test_base_strategy_default_noops():
    base = AugmentationStrategy(KeywordRetriever(chunks=CHUNKS))
    assert asyncio.run(base.augment_for_memory("x")) == "x"
    msgs = [{"role": "user", "content": "x"}]
    assert asyncio.run(base.augment_for_llm(msgs)) is msgs


def test_onthefly_no_user_message_returns_unchanged():
    aug = OnTheFlyAugmentation(KeywordRetriever(chunks=CHUNKS), top_k=1)
    msgs = [{"role": "system", "content": "sys"}]
    assert asyncio.run(aug.augment_for_llm(msgs)) == msgs


def test_onthefly_cache_hit_reuses_formatted_context():
    aug = OnTheFlyAugmentation(KeywordRetriever(chunks=CHUNKS), top_k=1)
    asyncio.run(aug.augment_for_llm([{"role": "user", "content": "refund policy?"}]))
    assert "refund policy?" in aug._cache  # first call cached the formatted context
    out = asyncio.run(aug.augment_for_llm([{"role": "user", "content": "refund policy?"}]))
    assert "Document context" in out[-1]["content"]


def test_onthefly_abstention_leaves_message_unchanged():
    aug = OnTheFlyAugmentation(KeywordRetriever(chunks=CHUNKS), top_k=1, relevance_threshold=999.0)
    msgs = [{"role": "user", "content": "refund policy?"}]
    out = asyncio.run(aug.augment_for_llm(msgs))
    assert out[-1]["content"] == "refund policy?"  # ABSTENTION_MARKER path
    assert "refund policy?" in aug._cache  # the marker itself is cached
    assert aug._cache["refund policy?"] == ABSTENTION_MARKER
