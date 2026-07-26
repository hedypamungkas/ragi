"""Integration tests for the augmentation agent seam."""

from __future__ import annotations

import asyncio

from ragi.retrieval.augmentation import (
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
