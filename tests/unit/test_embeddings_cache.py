"""Unit tests for the embedding index cache."""

from __future__ import annotations

import asyncio
import json

from ragi.embeddings.cache import (
    _EMBEDDING_CACHE,
    _EmbeddingIndexCache,
    clear_embedding_cache,
    set_embedding_cache_path,
)
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


class TestSignature:
    def test_signature_is_content_based_and_stable(self):
        chunks = [Chunk("c1", "d1", "hello"), Chunk("c2", "d2", "world")]
        a = _EmbeddingIndexCache._signature(chunks)
        b = _EmbeddingIndexCache._signature(chunks)
        assert a == b and len(a) == 64  # sha256 hex

    def test_signature_changes_with_content(self):
        a = _EmbeddingIndexCache._signature([Chunk("c1", "d1", "hello")])
        b = _EmbeddingIndexCache._signature([Chunk("c1", "d1", "goodbye")])
        assert a != b


class TestDiskPersistence:
    def test_save_then_load_roundtrip(self, tmp_path):
        path = tmp_path / "emb.json"
        cache = _EmbeddingIndexCache(cache_path=str(path))
        chunks = [Chunk("c1", "d1", "hello")]
        asyncio.run(cache.get_or_build(chunks, _embed))
        assert path.exists()  # _save_disk wrote it

        # a fresh cache reading the same file hits without re-embedding
        cache2 = _EmbeddingIndexCache(cache_path=str(path))
        calls = {"n": 0}

        async def counting_embed(text):
            calls["n"] += 1
            return [1.0]

        emb, ok = asyncio.run(cache2.get_or_build(chunks, counting_embed))
        assert ok is True and emb is not None
        assert calls["n"] == 0  # served from disk, no re-embed

    def test_load_missing_file_is_noop(self, tmp_path):
        cache = _EmbeddingIndexCache(cache_path=str(tmp_path / "absent.json"))
        cache._load_disk()
        assert cache._index == {}

    def test_load_corrupt_json_starts_empty(self, tmp_path, caplog):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        cache = _EmbeddingIndexCache(cache_path=str(path))
        with caplog.at_level("WARNING"):
            cache._load_disk()
        assert cache._index == {}
        assert any("load failed" in r.message.lower() for r in caplog.records)

    def test_load_non_dict_top_level_skipped(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        cache = _EmbeddingIndexCache(cache_path=str(path))
        cache._load_disk()
        assert cache._index == {}

    def test_load_skips_non_dict_emb_map(self, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps({"good_sig": {"cid": [0.1]}, "bad_sig": "notdict"}), encoding="utf-8")
        cache = _EmbeddingIndexCache(cache_path=str(path))
        cache._load_disk()
        assert "good_sig" in cache._index and "bad_sig" not in cache._index

    def test_load_is_lazy_and_runs_once(self, tmp_path):
        path = tmp_path / "emb.json"
        path.write_text(json.dumps({"sig": {"cid": [0.1]}}), encoding="utf-8")
        cache = _EmbeddingIndexCache(cache_path=str(path))
        cache._load_disk()
        # mutate the file after first load; second call must NOT re-read
        path.write_text(json.dumps({"sig2": {"cid": [0.9]}}), encoding="utf-8")
        cache._load_disk()
        assert "sig2" not in cache._index

    def test_save_failure_is_swallowed(self, tmp_path, monkeypatch, caplog):
        path = tmp_path / "emb.json"
        cache = _EmbeddingIndexCache(cache_path=str(path))

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr("ragi.embeddings.cache.Path.mkdir", boom)
        with caplog.at_level("WARNING"):
            cache._save_disk()
        assert any("save failed" in r.message.lower() for r in caplog.records)

    def test_save_skipped_when_no_path(self):
        cache = _EmbeddingIndexCache(cache_path=None)
        cache._save_disk()  # no raise, no-op


class TestSetEmbeddingCachePath:
    def test_set_path_forces_reload_flag(self, tmp_path):
        set_embedding_cache_path(str(tmp_path / "x.json"))
        assert _EMBEDDING_CACHE._cache_path == str(tmp_path / "x.json")
        assert _EMBEDDING_CACHE._disk_loaded is False
        # reset to default to avoid leaking state into other tests
        set_embedding_cache_path(None)
        assert _EMBEDDING_CACHE._cache_path is None
