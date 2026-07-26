"""ragi/ingest/chunker -- text chunking strategies for the RAG pipeline.

Lifted from ``koboi/rag/chunker.py`` and decoupled: zero ``koboi.*`` imports. The
``BaseChunker`` ABC + ``FixedSizeChunker`` / ``SentenceChunker`` / ``ParagraphChunker``
/ ``SemanticChunker`` built-ins are registered at module import via ``@register_chunker``
(so importing this module is sufficient to populate ``chunker_registry`` --
``ragi.register_builtins`` just imports it).

The ``SemanticChunker`` is async-aware: it embeds each sentence via
``EmbeddingClient.embed_batch`` (the only chunker here that needs an embedder) and
greedily merges adjacent sentences by cosine similarity. Its sync ``chunk()`` runs the
async path via ``asyncio.run`` and refuses to nest inside a running event loop (call
``chunk_async`` directly instead) -- fixing the upstream koboi bug where the sync
chunker had no access to the async embedding endpoint and silently fell back to
sentence mode.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any

from ragi.registry import chunker_registry, register_chunker
from ragi.types import Chunk, Document

_logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Chunker contract: turn a ``Document`` into an ordered list of ``Chunk``s."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...

    def _make_chunk(self, doc_id: str, index: int, content: str) -> Chunk:
        return Chunk(
            id=f"{doc_id}_c{index}",
            doc_id=doc_id,
            content=content.strip(),
            metadata={"chunk_index": index},
        )


@register_chunker(
    "fixed",
    description="Fixed-size chunks with overlap and sentence-boundary snapping",
)
class FixedSizeChunker(BaseChunker):
    """Fixed-size chunks with overlap and sentence-boundary snapping."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [self._make_chunk(document.id, 0, text)]

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Snap to sentence boundary to avoid breaking mid-sentence
            if end < len(text):
                snap = text.rfind(". ", start, end)
                if snap > start:
                    end = snap + 1
                else:
                    snap = text.rfind("\n", start, end)
                    if snap > start:
                        end = snap

            piece = text[start:end].strip()
            if piece:
                chunks.append(self._make_chunk(document.id, index, piece))
                index += 1

            next_start = end - self.overlap
            if next_start <= start:
                start = end
            else:
                start = next_start

        return chunks


@register_chunker(
    "sentence",
    description="Sentence-aware chunks up to max_chunk_size",
)
class SentenceChunker(BaseChunker):
    """Sentence-aware chunks up to ``max_chunk_size``."""

    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        index = 0

        for sentence in sentences:
            if current_len + len(sentence) > self.max_chunk_size and current:
                content = " ".join(current)
                chunks.append(self._make_chunk(document.id, index, content))
                index += 1
                current = []
                current_len = 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            content = " ".join(current)
            chunks.append(self._make_chunk(document.id, index, content))

        return chunks


@register_chunker(
    "paragraph",
    description="Paragraph-based chunks with heading-aware merging",
)
class ParagraphChunker(BaseChunker):
    """Paragraph-based chunks with heading-aware merging."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self._fallback = FixedSizeChunker(chunk_size=max_chunk_size, overlap=50)

    @staticmethod
    def _is_heading(text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and bool(re.match(r"^#{1,6}\s", stripped))

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge heading-only paragraphs with their following content
        merged: list[str] = []
        pending_heading: str | None = None
        for para in paragraphs:
            if self._is_heading(para):
                if pending_heading is not None:
                    merged.append(pending_heading)
                pending_heading = para
            elif pending_heading is not None:
                merged.append(pending_heading + "\n\n" + para)
                pending_heading = None
            else:
                merged.append(para)
        if pending_heading is not None:
            merged.append(pending_heading)

        chunks: list[Chunk] = []
        index = 0

        for para in merged:
            if len(para) <= self.max_chunk_size:
                chunks.append(self._make_chunk(document.id, index, para))
                index += 1
            else:
                sub_doc = Document(id=document.id, title="", content=para)
                for sub_chunk in self._fallback.chunk(sub_doc):
                    sub_chunk.metadata["chunk_index"] = index
                    sub_chunk.id = f"{document.id}_c{index}"
                    chunks.append(sub_chunk)
                    index += 1

        return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-stdlib cosine similarity (0.0 for zero-norm or orthogonal vectors)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def _run_sync(coro: Coroutine[Any, Any, list[Chunk]]) -> list[Chunk]:
    """Run a coroutine from sync code; refuse to nest inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Inside a running loop -- close the un-awaited coroutine to avoid a RuntimeWarning.
    coro.close()
    raise RuntimeError("SemanticChunker.chunk() cannot embed from a running event loop; use chunk_async()")


@register_chunker(
    "semantic",
    description="Embedding-aware semantic chunking (greedy sentence merge by cosine)",
)
class SemanticChunker(BaseChunker):
    """Embedding-aware semantic chunking.

    Splits into sentences (same regex as :class:`SentenceChunker`), embeds each via
    ``embedder.embed_batch``, then greedily merges adjacent sentences while cosine
    similarity stays above ``threshold`` and the buffer stays under ``max_chunk_size``.
    A merged group whose own length exceeds the cap is hard-split via
    :class:`FixedSizeChunker` (mirrors :class:`ParagraphChunker`). If the embedder
    returns ``None`` for every sentence (fail-soft unavailability), the chunker
    degrades to :class:`SentenceChunker` so semantic mode never silently drops content.
    """

    def __init__(self, embedder, threshold: float = 0.76, max_chunk_size: int = 800):
        self.embedder = embedder
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size
        self._sentence_chunker = SentenceChunker(max_chunk_size=max_chunk_size)
        self._fallback = FixedSizeChunker(chunk_size=max_chunk_size, overlap=50)

    async def chunk_async(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []
        # Same sentence regex as SentenceChunker.chunk (kept as a stored helper per spec).
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return []
        vecs = await self.embedder.embed_batch(sentences)
        if all(v is None for v in vecs):
            # Embedder unavailable -> degrade to size-based sentence chunking.
            return self._sentence_chunker.chunk(document)

        groups: list[str] = []
        buffer: list[str] = []
        buffer_len = 0
        prev_vec: list[float] | None = None

        # embed_batch returns one vector per sentence; strict=False tolerates a mis-sized batch.
        for sentence, vec in zip(sentences, vecs, strict=False):
            merge = (
                vec is not None
                and bool(buffer)
                and prev_vec is not None
                and _cosine(prev_vec, vec) >= self.threshold
                and buffer_len + len(sentence) + 1 < self.max_chunk_size
            )
            if merge:
                buffer.append(sentence)
                buffer_len += len(sentence) + 1
                prev_vec = vec
                continue
            if buffer:
                groups.append(" ".join(buffer))
                buffer = []
                buffer_len = 0
            buffer.append(sentence)
            buffer_len += len(sentence) + 1
            prev_vec = vec
        if buffer:
            groups.append(" ".join(buffer))

        # Hard-cap any oversized group via FixedSizeChunker (mirrors ParagraphChunker).
        chunks: list[Chunk] = []
        index = 0
        for content in groups:
            if len(content) <= self.max_chunk_size:
                chunks.append(self._make_chunk(document.id, index, content))
                index += 1
            else:
                sub_doc = Document(id=document.id, title="", content=content)
                for sub in self._fallback.chunk(sub_doc):
                    sub.metadata["chunk_index"] = index
                    sub.id = f"{document.id}_c{index}"
                    chunks.append(sub)
                    index += 1
        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        return _run_sync(self.chunk_async(document))


def resolve_chunker(name: str) -> BaseChunker:
    """Build a chunker by registry name using its default constructor kwargs.

    Raises ``ValueError`` if no chunker is registered under ``name``. Callers needing
    custom kwargs should instantiate the class directly or use
    ``ragi.registry._build_chunker`` (config-driven).
    """
    entry = chunker_registry.get(name)
    if entry is None:
        raise ValueError(f"Unknown chunker '{name}'. Available: {chunker_registry.list_available()}")
    return entry.cls()
