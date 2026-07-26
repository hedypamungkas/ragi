"""Integration test for the LangChain adapter (langchain_core available)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
from langchain_core.documents import Document  # noqa: E402

from ragi.adapters.langchain import LangChainRetrieverAdapter  # noqa: E402
from ragi.retrieval.retriever import KeywordRetriever  # noqa: E402
from ragi.types import Chunk  # noqa: E402

CHUNKS = [
    Chunk("c1", "d1", "the mitochondrion is the powerhouse of the cell"),
    Chunk("c2", "d2", "photosynthesis converts sunlight into chemical energy"),
]


def test_langchain_adapter_invoke_returns_documents():
    adapter = LangChainRetrieverAdapter(ragi_retriever=KeywordRetriever(chunks=CHUNKS), top_k=2)
    docs = adapter.invoke("photosynthesis sunlight energy")
    assert docs, "expected at least one Document"
    assert isinstance(docs[0], Document)
    assert "photosynthesis" in docs[0].page_content.lower()
    assert docs[0].metadata["doc_id"] == "d2"
    assert "score" in docs[0].metadata


@pytest.mark.asyncio
async def test_langchain_adapter_ainvoke():
    adapter = LangChainRetrieverAdapter(ragi_retriever=KeywordRetriever(chunks=CHUNKS), top_k=2)
    docs = await adapter.ainvoke("photosynthesis sunlight energy")
    assert docs and docs[0].metadata["doc_id"] == "d2"
