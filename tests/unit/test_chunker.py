"""Unit tests for the chunker strategies."""

from __future__ import annotations

import asyncio

import pytest

from ragi.ingest import chunker
from ragi.ingest.chunker import (
    FixedSizeChunker,
    ParagraphChunker,
    SemanticChunker,
    SentenceChunker,
    resolve_chunker,
)
from ragi.types import Document


def _doc(text: str) -> Document:
    return Document(id="d1", title="t", content=text)


class TestChunkers:
    def test_paragraph_splits_on_double_newline(self):
        chunks = ParagraphChunker().chunk(_doc("First paragraph.\n\nSecond paragraph."))
        assert len(chunks) == 2
        assert "First" in chunks[0].content and "Second" in chunks[1].content

    def test_paragraph_merges_heading_with_body(self):
        chunks = ParagraphChunker().chunk(_doc("# Heading\n\nBody text here."))
        assert len(chunks) == 1
        assert "Heading" in chunks[0].content and "Body" in chunks[0].content

    def test_fixed_size_produces_multiple_chunks(self):
        chunks = FixedSizeChunker(chunk_size=40, overlap=5).chunk(_doc("word " * 200))
        assert len(chunks) >= 2

    def test_sentence_groups_under_max_size(self):
        chunks = SentenceChunker(max_chunk_size=50).chunk(_doc("Short one. Also short. Plus more text."))
        assert len(chunks) >= 1
        # each chunk content is non-empty
        assert all(c.content for c in chunks)

    def test_chunks_carry_doc_id_and_index(self):
        chunks = ParagraphChunker().chunk(_doc("a\n\nb\n\nc"))
        assert len(chunks) == 3
        assert all(c.doc_id == "d1" for c in chunks)
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]

    def test_empty_content_returns_empty(self):
        assert ParagraphChunker().chunk(_doc("   ")) == []


class TestFixedSizeChunkerBranches:
    def test_empty_returns_empty(self):
        assert FixedSizeChunker().chunk(_doc("")) == []

    def test_single_chunk_when_under_size(self):
        chunks = FixedSizeChunker(chunk_size=1000).chunk(_doc("short text"))
        assert len(chunks) == 1

    def test_snaps_to_newline_when_no_sentence_boundary(self):
        # no ". " inside the window -> falls back to newline snap
        chunks = FixedSizeChunker(chunk_size=7, overlap=0).chunk(_doc("line1\nline2\nline3\nline4"))
        assert len(chunks) >= 2
        assert all(c.content for c in chunks)

    def test_overlap_larger_than_chunk_does_not_loop_forever(self):
        # next_start <= start -> start jumps to end (clamp), progress guaranteed
        chunks = FixedSizeChunker(chunk_size=10, overlap=20).chunk(_doc("word " * 50))
        assert len(chunks) >= 2


class TestSentenceChunkerBranches:
    def test_empty_returns_empty(self):
        assert SentenceChunker().chunk(_doc("")) == []

    def test_flushes_buffer_when_size_exceeded(self):
        chunks = SentenceChunker(max_chunk_size=15).chunk(_doc("Short one. Also short. Plus more text."))
        assert len(chunks) == 3  # each sentence forced a flush
        assert all(c.content for c in chunks)


class TestParagraphChunkerBranches:
    def test_heading_only_at_end_is_emitted(self):
        chunks = ParagraphChunker().chunk(_doc("body text.\n\n# Trailing heading"))
        # heading has no following body -> emitted standalone
        assert any("Trailing heading" in c.content for c in chunks)

    def test_oversize_paragraph_uses_fixed_fallback(self):
        chunks = ParagraphChunker(max_chunk_size=10).chunk(_doc("a" * 120))
        assert len(chunks) >= 2  # hard-split via FixedSizeChunker

    def test_is_heading_detection(self):
        assert ParagraphChunker._is_heading("# title") is True
        assert ParagraphChunker._is_heading("## sub") is True
        assert ParagraphChunker._is_heading("not a heading") is False
        assert ParagraphChunker._is_heading("   ") is False


class TestCosine:
    def test_parallel_vectors_score_one(self):
        assert chunker._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert chunker._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_norm_returns_zero(self):
        assert chunker._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestRunSync:
    def test_runs_coro_when_no_loop(self):
        async def dummy():
            return ["x"]

        assert chunker._run_sync(dummy()) == ["x"]

    def test_refuses_to_nest_in_running_loop(self):
        async def runner():
            async def inner():
                return []

            with pytest.raises(RuntimeError, match="running event loop"):
                chunker._run_sync(inner())

        asyncio.run(runner())


class _FakeEmbedder:
    def __init__(self, vecs):
        self._vecs = vecs

    async def embed_batch(self, texts):
        return self._vecs


class TestSemanticChunker:
    def test_empty_returns_empty(self):
        assert SemanticChunker(_FakeEmbedder([]), max_chunk_size=100).chunk(_doc("")) == []

    def test_all_none_vecs_degrades_to_sentence_mode(self):
        ch = SemanticChunker(_FakeEmbedder([None, None]), max_chunk_size=100)
        chunks = ch.chunk(_doc("first sentence. second sentence."))
        assert len(chunks) == 1 and chunks[0].content

    def test_merges_similar_adjacent_sentences(self):
        # identical vectors -> cosine 1.0 >= default threshold -> merged into one group
        ch = SemanticChunker(_FakeEmbedder([[1.0, 0.0], [1.0, 0.0]]), max_chunk_size=800)
        chunks = ch.chunk(_doc("alpha sentence. beta sentence."))
        assert len(chunks) == 1

    def test_dissimilar_vectors_stay_separate(self):
        # orthogonal vectors -> cosine 0.0 < threshold -> two groups
        ch = SemanticChunker(_FakeEmbedder([[1.0, 0.0], [0.0, 1.0]]), max_chunk_size=800)
        chunks = ch.chunk(_doc("alpha sentence. beta sentence."))
        assert len(chunks) == 2

    def test_oversize_group_hard_split(self):
        long = "word " * 40  # one sentence, longer than max_chunk_size
        ch = SemanticChunker(_FakeEmbedder([[1.0, 0.0]]), max_chunk_size=20)
        chunks = ch.chunk(_doc(long))
        assert len(chunks) >= 2  # FixedSizeChunker hard-split


class TestResolveChunker:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown chunker"):
            resolve_chunker("does-not-exist")

    def test_known_returns_instance(self):
        assert isinstance(resolve_chunker("fixed"), FixedSizeChunker)
        assert isinstance(resolve_chunker("sentence"), SentenceChunker)
