"""Integration test: build_pipeline over a temp corpus + retrieve."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ragi


async def test_build_pipeline_over_temp_corpus_ranks_gold_first():
    ragi.register_builtins()
    d = tempfile.mkdtemp()
    (Path(d) / "a.txt").write_text("the mitochondrion is the powerhouse of the cell", encoding="utf-8")
    (Path(d) / "b.txt").write_text("photosynthesis converts sunlight into chemical energy in plants", encoding="utf-8")
    retriever = ragi.build_pipeline(
        {"enabled": True, "chunker": "paragraph", "retriever": "bm25", "documents": [{"path": d}]}
    )
    assert retriever is not None
    results = await retriever.retrieve("photosynthesis sunlight energy", top_k=2)
    assert results, "expected at least one result"
    assert results[0].chunk.doc_id == "b"
    assert results[0].retrieval_method == "bm25"


async def test_build_pipeline_keyword_path():
    ragi.register_builtins()
    d = tempfile.mkdtemp()
    (Path(d) / "only.txt").write_text("a quick brown fox jumps over the lazy dog", encoding="utf-8")
    retriever = ragi.build_pipeline(
        {"enabled": True, "chunker": "paragraph", "retriever": "keyword", "documents": [{"path": d}]}
    )
    results = await retriever.retrieve("quick fox", top_k=3)
    assert results and results[0].chunk.doc_id == "only"


def test_build_pipeline_disabled_returns_none():
    ragi.register_builtins()
    assert ragi.build_pipeline({"enabled": False, "documents": []}) is None


def test_build_pipeline_missing_documents_returns_none():
    ragi.register_builtins()
    # enabled but no resolvable documents -> None (warns)
    assert ragi.build_pipeline({"enabled": True, "documents": [{"path": "/no/such/dir/here"}]}) is None
