"""Unit tests for ragi.eval.datasets -- golden-set loaders + resolve + synthetic fixture."""

from __future__ import annotations

import json

import pytest

from ragi.eval.datasets import (
    load_dataset_csv,
    load_dataset_json,
    load_msmarco,
    resolve_dataset,
    synthetic_chunks,
    synthetic_fixture,
)


def _write_json(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


class TestLoadDatasetJson:
    def test_loads_qrels_and_metadata(self, tmp_path):
        p = _write_json(
            tmp_path,
            "g.json",
            {"name": "my", "corpus_dir": "data/c", "qrels": [{"query": "q", "gold_doc": "d1", "gold_needles": ["n"]}]},
        )
        ds = load_dataset_json(p)
        assert ds.name == "my" and ds.corpus_dir == "data/c"
        assert ds.qrels[0].gold_doc == "d1"

    def test_name_falls_back_to_stem(self, tmp_path):
        p = _write_json(tmp_path, "stem.json", {"qrels": [{"query": "q"}]})
        ds = load_dataset_json(p)
        assert ds.name == "stem"

    def test_gold_needles_string_becomes_list(self, tmp_path):
        p = _write_json(tmp_path, "s.json", {"qrels": [{"query": "q", "gold_needles": "single"}]})
        assert load_dataset_json(p).qrels[0].gold_needles == ["single"]

    def test_gold_doc_falls_back_to_doc_id(self, tmp_path):
        p = _write_json(tmp_path, "d.json", {"qrels": [{"query": "q", "doc_id": "p9"}]})
        assert load_dataset_json(p).qrels[0].gold_doc == "p9"

    def test_out_of_scope_flag(self, tmp_path):
        p = _write_json(tmp_path, "o.json", {"qrels": [{"query": "q", "out_of_scope": True}]})
        assert load_dataset_json(p).qrels[0].out_of_scope is True

    def test_empty_qrels(self, tmp_path):
        p = _write_json(tmp_path, "e.json", {"name": "e"})
        assert load_dataset_json(p).qrels == []


class TestLoadDatasetCsv:
    def test_loads_rows_and_splits_needles(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text(
            'query,gold_doc,gold_needles,expected_answer\nq1,d1,"a|b",ans1\nq2,,"c",\n',
            encoding="utf-8",
        )
        ds = load_dataset_csv(p, name="csvd", corpus_dir="c")
        assert ds.name == "csvd" and ds.corpus_dir == "c"
        assert ds.qrels[0].gold_needles == ["a", "b"]
        assert ds.qrels[0].expected_answer == "ans1"
        assert ds.qrels[1].gold_doc is None
        assert ds.qrels[1].gold_needles == ["c"]

    def test_needles_column_alias(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("query,needles\nq1,x|y\n", encoding="utf-8")
        assert load_dataset_csv(p).qrels[0].gold_needles == ["x", "y"]


class TestLoadMsmarco:
    def test_sets_name_and_corpus(self, tmp_path):
        p = _write_json(tmp_path, "m.json", {"qrels": [{"query": "q", "gold_doc": "p"}]})
        ds = load_msmarco(p, corpus_dir="corp")
        assert ds.name == "msmarco" and ds.corpus_dir == "corp"


class TestSynthetic:
    def test_chunks_match_corpus(self):
        chunks = synthetic_chunks()
        assert len(chunks) >= 5
        assert all(c.metadata["source"] for c in chunks)

    def test_fixture_has_qrels(self):
        ds = synthetic_fixture()
        assert ds.name == "synthetic"
        assert len(ds.qrels) >= 5
        assert all(q.gold_needles for q in ds.qrels)


class TestResolveDataset:
    def test_synthetic(self):
        assert resolve_dataset("synthetic").name == "synthetic"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            resolve_dataset("nope")

    def test_byo_requires_path(self):
        with pytest.raises(ValueError, match="requires --qrels"):
            resolve_dataset("byo")

    def test_csv_requires_path(self):
        with pytest.raises(ValueError, match="requires --qrels"):
            resolve_dataset("csv")

    def test_byo_loads(self, tmp_path):
        p = _write_json(tmp_path, "b.json", {"qrels": [{"query": "q"}]})
        assert resolve_dataset("byo", qrels_path=p).qrels[0].query == "q"

    def test_csv_loads(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("query\nq1\n", encoding="utf-8")
        assert resolve_dataset("csv", qrels_path=p).qrels[0].query == "q1"
