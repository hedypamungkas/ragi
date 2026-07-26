"""Unit tests for the TyDi-id dataset loader (fixture-based; builder is datasets-only)."""

from __future__ import annotations

import json

from ragworkbench.eval.datasets import load_tydi_id, resolve_dataset


def test_load_tydi_id(tmp_path):
    qrels = {
        "name": "tydi-id",
        "qrels": [
            {
                "query": "apa ibu kota indonesia",
                "gold_doc": "p123abc",
                "gold_needles": ["ibu kota"],
                "expected_answer": "Jakarta",
            }
        ],
    }
    p = tmp_path / "id_qrels.json"
    p.write_text(json.dumps(qrels), encoding="utf-8")
    ds = load_tydi_id(p)
    assert ds.name == "tydi-id"
    assert len(ds.qrels) == 1
    assert ds.qrels[0].gold_doc == "p123abc"
    assert ds.qrels[0].expected_answer == "Jakarta"


def test_resolve_dataset_tydi_id_alias(tmp_path):
    p = tmp_path / "id.json"
    p.write_text(json.dumps({"qrels": [{"query": "q", "gold_doc": "p1", "gold_needles": ["x"]}]}), encoding="utf-8")
    for alias in ("tydi-id", "tydi_id", "tydi"):
        ds = resolve_dataset(alias, qrels_path=p)
        assert ds.name == "tydi-id"
