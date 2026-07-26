"""Unit tests for the in-memory VectorStore."""

from __future__ import annotations

import asyncio

from ragworkbench.embeddings.mock import MockEmbeddingClient
from ragworkbench.types import Chunk
from ragworkbench.vectorstore.memory import InMemoryVectorStore

CHUNKS = [
    Chunk("c1", "d1", "photosynthesis converts sunlight into energy", metadata={"source": "bio"}),
    Chunk("c2", "d2", "gravity attracts two masses", metadata={"source": "phys"}),
]


def test_add_then_search_ranks_gold():
    store = InMemoryVectorStore(MockEmbeddingClient())
    asyncio.run(store.add(CHUNKS))
    assert store.count == 2
    qv = asyncio.run(MockEmbeddingClient().embed("photosynthesis sunlight energy"))
    results = asyncio.run(store.search(qv, top_k=2))
    assert results[0].chunk.doc_id == "d1"
    assert results[0].retrieval_method == "vectorstore:memory"


def test_search_with_metadata_filter():
    store = InMemoryVectorStore(MockEmbeddingClient())
    asyncio.run(store.add(CHUNKS))
    qv = asyncio.run(MockEmbeddingClient().embed("masses gravity"))
    results = asyncio.run(store.search(qv, top_k=2, filter={"source": "phys"}))
    assert all(r.chunk.metadata["source"] == "phys" for r in results)
    assert results and results[0].chunk.doc_id == "d2"


def test_empty_store_search_returns_empty():
    store = InMemoryVectorStore(MockEmbeddingClient())
    results = asyncio.run(store.search([1.0, 0.0], top_k=2))
    assert results == []
