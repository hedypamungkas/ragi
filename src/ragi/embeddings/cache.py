"""ragi/embeddings/cache -- process-level shared embedding index.

Lifted near-verbatim from ``koboi/rag/retriever.py`` (``_EmbeddingIndexCache``)
and decoupled: zero ``koboi.*`` imports; stdlib only (a ``TYPE_CHECKING`` import
of :class:`Chunk` annotates the signature helper).

Retrievers built from the same corpus (same chunk ids + contents) reuse one
embedding pass instead of each re-embedding every chunk. This makes semantic /
hybrid retrieval affordable for multi-session deployments: the corpus is embedded
once per process, not per retriever/session.

Only successful builds are cached; an unavailable embedding endpoint yields a miss
that is NOT stored, so a later retry once it recovers still works. Concurrency:
one :class:`asyncio.Lock` per signature dedupes concurrent first-builds.

The signature is content-only and **not model-aware** -- it assumes one embedding
model per process; call :func:`clear_embedding_cache` after changing models (or
restart the process, which clears this module-level cache).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragi.types import Chunk

_logger = logging.getLogger(__name__)


class _EmbeddingIndexCache:
    """Process-level shared embedding index, keyed by a corpus signature.

    The signature is SHA-256 over ``c.id + "\\0" + c.content + "\\0"`` per chunk
    (content-only, NOT model-aware -- see module docstring).
    """

    def __init__(self, cache_path: str | None = None) -> None:
        self._index: dict[str, dict[str, list[float]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Opt-in on-disk persistence (JSON). None = in-memory only (default).
        self._cache_path: str | None = cache_path
        self._disk_loaded = False

    @staticmethod
    def _signature(chunks: list[Chunk]) -> str:
        h = hashlib.sha256()
        for c in chunks:
            h.update(c.id.encode())
            h.update(b"\0")
            h.update(c.content.encode())
            h.update(b"\0")
        return h.hexdigest()

    def _load_disk(self) -> None:
        """Lazy-load the JSON on-disk cache into the in-memory index (once)."""
        if self._disk_loaded:
            return
        self._disk_loaded = True
        if not self._cache_path:
            return
        p = Path(self._cache_path)
        if not p.exists():
            return
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for sig, emb_map in data.items():
                    if isinstance(emb_map, dict) and sig not in self._index:
                        self._index[sig] = emb_map
        except (OSError, ValueError) as exc:  # corrupt/unreadable cache -> start empty
            _logger.warning("Embedding cache load failed (%s); starting empty", exc)

    def _save_disk(self) -> None:
        """Persist the in-memory index to JSON (atomic temp-replace)."""
        if not self._cache_path:
            return
        try:
            p = Path(self._cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._index, f)
            tmp.replace(p)
        except OSError as exc:  # best-effort; never block retrieval on a save failure
            _logger.warning("Embedding cache save failed: %s", exc)

    async def get_or_build(
        self,
        chunks: list[Chunk],
        embed_fn,  # Callable[[str], Awaitable[list[float] | None]]
    ) -> tuple[dict[str, list[float]] | None, bool]:
        """Return ``(id -> embedding, ok)``. ``ok=False`` means unavailable.

        ``ok=False`` is returned ONLY when ``embed_fn`` returned ``None`` (endpoint
        unavailable); the miss is NOT cached so a later retry once the endpoint
        recovers still works. ``embed_fn`` is called one-by-one:
        ``emb = await embed_fn(c.content)``.
        """
        self._load_disk()
        sig = self._signature(chunks)
        cached = self._index.get(sig)
        if cached is not None:
            return cached, True
        # dict.setdefault is atomic in a single-threaded asyncio loop, so the
        # lock is created safely before any await.
        lock = self._locks.setdefault(sig, asyncio.Lock())
        async with lock:
            cached = self._index.get(sig)  # double-check after acquiring lock
            if cached is not None:
                return cached, True
            emb_map: dict[str, list[float]] = {}
            for c in chunks:
                emb = await embed_fn(c.content)
                if emb is None:
                    return None, False  # endpoint unavailable; do not cache
                emb_map[c.id] = emb
            self._index[sig] = emb_map
            self._save_disk()
            return emb_map, True

    def clear(self) -> None:
        self._index.clear()
        self._locks.clear()


#: Process-wide shared embedding index (see :class:`_EmbeddingIndexCache`).
_EMBEDDING_CACHE = _EmbeddingIndexCache()


def clear_embedding_cache() -> None:
    """Reset the shared embedding index (test isolation / model change / forced rebuild)."""
    _EMBEDDING_CACHE.clear()


def set_embedding_cache_path(path: str | None) -> None:
    """Set the on-disk embedding cache path (opt-in) and force a reload.

    When set, successful corpus embeddings persist to JSON across process restarts
    so a redeploy does not re-embed the whole corpus. ``None`` = in-memory only
    (the default).
    """
    _EMBEDDING_CACHE._cache_path = path
    _EMBEDDING_CACHE._disk_loaded = False  # force reload on next get_or_build
