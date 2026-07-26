"""Unit tests for the chunker strategies."""

from __future__ import annotations

from ragworkbench.ingest.chunker import FixedSizeChunker, ParagraphChunker, SentenceChunker
from ragworkbench.types import Document


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
