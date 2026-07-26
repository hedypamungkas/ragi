"""Unit tests for Semantic/Hybrid retrievers (embedding-based)."""

from __future__ import annotations

import asyncio

from ragworkbench.embeddings.cache import clear_embedding_cache
from ragworkbench.embeddings.mock import MockEmbeddingClient
from ragworkbench.retrieval.retriever import HybridRetriever, SemanticRetriever
from ragworkbench.types import Chunk

CHUNKS = [
    Chunk("c1", "d1", "the mitochondrion is the powerhouse of the cell"),
    Chunk("c2", "d2", "photosynthesis converts sunlight into chemical energy"),
]


def test_semantic_ranks_gold_first():
    clear_embedding_cache()
    r = SemanticRetriever(chunks=CHUNKS, embedder=MockEmbeddingClient())
    res = asyncio.run(r.retrieve("photosynthesis sunlight energy", top_k=2))
    assert res, "expected results"
    assert res[0].chunk.doc_id == "d2"


def test_semantic_degrades_to_keyword_without_embedder():
    clear_embedding_cache()
    r = SemanticRetriever(chunks=CHUNKS)  # no embedder
    res = asyncio.run(r.retrieve("photosynthesis", top_k=2))
    assert res and "fallback" in res[0].retrieval_method


def test_hybrid_rrf_stamp():
    clear_embedding_cache()
    r = HybridRetriever(chunks=CHUNKS, embedder=MockEmbeddingClient())
    res = asyncio.run(r.retrieve("photosynthesis energy", top_k=2))
    assert res and res[0].retrieval_method == "hybrid"
