"""Unit tests for the embedding index cache."""

from __future__ import annotations

import asyncio

from ragi.embeddings.cache import _EMBEDDING_CACHE, clear_embedding_cache
from ragi.types import Chunk


async def _embed(text: str):
    return [float(len(text)), 1.0]


async def _embed_none(text: str):
    return None


def test_cache_builds_then_hits():
    clear_embedding_cache()
    chunks = [Chunk("c1", "d1", "hello"), Chunk("c2", "d2", "world")]
    emb, ok = asyncio.run(_EMBEDDING_CACHE.get_or_build(chunks, _embed))
    assert ok is True
    assert emb is not None and set(emb.keys()) == {"c1", "c2"}
    # Second call over the same signature is a cache hit.
    emb2, ok2 = asyncio.run(_EMBEDDING_CACHE.get_or_build(chunks, _embed))
    assert ok2 is True and emb2 is not None


def test_cache_ok_false_when_embed_returns_none():
    clear_embedding_cache()
    chunks = [Chunk("c1", "d1", "hello")]
    emb, ok = asyncio.run(_EMBEDDING_CACHE.get_or_build(chunks, _embed_none))
    assert ok is False  # miss NOT cached (retry-after-recovery works)


def test_clear_resets():
    clear_embedding_cache()
    chunks = [Chunk("c1", "d1", "hello")]
    asyncio.run(_EMBEDDING_CACHE.get_or_build(chunks, _embed))
    clear_embedding_cache()
    # index is empty after clear; a fresh build re-runs embed_fn
    emb, ok = asyncio.run(_EMBEDDING_CACHE.get_or_build(chunks, _embed))
    assert ok is True and emb is not None
