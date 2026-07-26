"""Unit tests for the VectorStoreRetriever (lazy indexing)."""

from __future__ import annotations

import asyncio

from ragi.embeddings.mock import MockEmbeddingClient
from ragi.retrieval.vectorstore_retriever import VectorStoreRetriever
from ragi.types import Chunk
from ragi.vectorstore.memory import InMemoryVectorStore

CHUNKS = [
    Chunk("c1", "d1", "photosynthesis converts sunlight into energy"),
    Chunk("c2", "d2", "gravity attracts two masses"),
]


def test_lazy_indexes_on_first_retrieve():
    store = InMemoryVectorStore(MockEmbeddingClient())
    retriever = VectorStoreRetriever(CHUNKS, embedder=MockEmbeddingClient(), store=store)
    assert retriever._indexed is False
    results = asyncio.run(retriever.retrieve("photosynthesis sunlight energy", top_k=2))
    assert retriever._indexed is True
    assert results[0].chunk.doc_id == "d1"


def test_no_embedder_returns_empty():
    store = InMemoryVectorStore(None)
    retriever = VectorStoreRetriever(CHUNKS, embedder=None, store=store)
    results = asyncio.run(retriever.retrieve("photosynthesis", top_k=2))
    assert results == []


def test_no_store_raises_on_retrieve():
    retriever = VectorStoreRetriever(CHUNKS, embedder=MockEmbeddingClient(), store=None)
    import pytest

    with pytest.raises(ValueError, match="store"):
        asyncio.run(retriever.retrieve("photosynthesis", top_k=2))
