"""Integration test: build_pipeline vectorstore branch over a temp corpus."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ragworkbench as rwb
from ragworkbench.embeddings.mock import MockEmbeddingClient


async def test_build_pipeline_vectorstore_memory():
    rwb.register_builtins()
    d = Path(tempfile.mkdtemp())
    (d / "a.txt").write_text("the mitochondrion is the powerhouse of the cell")
    (d / "b.txt").write_text("photosynthesis converts sunlight into chemical energy")
    retriever = rwb.build_pipeline(
        {
            "enabled": True,
            "chunker": "paragraph",
            "retriever": "vectorstore",
            "vectorstore": {"backend": "memory"},
            "documents": [{"path": str(d)}],
        },
        embedder=MockEmbeddingClient(),
    )
    assert type(retriever).__name__ == "VectorStoreRetriever"
    results = await retriever.retrieve("photosynthesis sunlight energy", top_k=2)
    assert results and results[0].chunk.doc_id == "b"
