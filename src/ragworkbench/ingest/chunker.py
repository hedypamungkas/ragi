"""ragworkbench/ingest/chunker -- text chunking strategies for the RAG pipeline.

Lifted from ``koboi/rag/chunker.py`` and decoupled: zero ``koboi.*`` imports. The
``BaseChunker`` ABC + ``FixedSizeChunker`` / ``SentenceChunker`` / ``ParagraphChunker``
built-ins are registered at module import via ``@register_chunker`` (so importing this
module is sufficient to populate ``chunker_registry`` -- ``ragworkbench.register_builtins``
just imports it).

Dropped vs koboi: ``SemanticChunker`` is intentionally omitted. Upstream it always
falls back to ``SentenceChunker`` because the sync chunker has no access to the async
embedding endpoint (``_get_embeddings_sync`` returns ``None`` unconditionally). A real
embedding-aware semantic chunker will land in v0.4 once the async embedder seam is in
place -- shipping a known-broken one would just silently masquerade as sentence mode.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from ragworkbench.registry import chunker_registry, register_chunker
from ragworkbench.types import Chunk, Document

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


def resolve_chunker(name: str) -> BaseChunker:
    """Build a chunker by registry name using its default constructor kwargs.

    Raises ``ValueError`` if no chunker is registered under ``name``. Callers needing
    custom kwargs should instantiate the class directly or use
    ``ragworkbench.registry._build_chunker`` (config-driven).
    """
    entry = chunker_registry.get(name)
    if entry is None:
        raise ValueError(f"Unknown chunker '{name}'. Available: {chunker_registry.list_available()}")
    return entry.cls()
