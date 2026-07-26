"""ragworkbench/eval/datasets -- golden-set loaders (MS MARCO, BYO, synthetic).

A "golden set" is the retriever-independent contract: a list of ``{query, gold_doc,
gold_needles, expected_answer}``. A retrieved chunk counts as a "hit" if ``gold_doc``
is among the retrieved doc_ids OR a ``gold_needles`` substring appears in retrieved
content (the metric functions use the needle form).

The MS MARCO loader reads the qrels JSON produced by ``scripts/build_ir_corpus.py``
(HF ``microsoft/ms_marco``; passage TEXT stays gitignored -- only the license-light qrels
with 10-word gold snippets are committed). The synthetic fixture is a tiny built-in set so
the closed loop can be exercised offline with no download and no API key.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from ragworkbench.eval.types import GoldenDataset, GoldenQrel
from ragworkbench.types import Chunk

_logger = logging.getLogger(__name__)

DEFAULT_MSMARCO_QRELS = "evals/fixtures/ir_qrels.json"


def _qrel_from_dict(d: dict) -> GoldenQrel:
    gold_needles = d.get("gold_needles") or []
    if isinstance(gold_needles, str):
        gold_needles = [gold_needles]
    return GoldenQrel(
        query=d["query"],
        gold_doc=d.get("gold_doc") or d.get("doc_id"),
        gold_needles=list(gold_needles),
        expected_answer=d.get("expected_answer"),
        out_of_scope=bool(d.get("out_of_scope", False)),
    )


def load_dataset_json(path: str | Path) -> GoldenDataset:
    """Load a golden dataset from a JSON file.

    Schema::

        {"name": "my", "corpus_dir": "data/corpus",
         "qrels": [{"query": "...", "gold_doc": "pid", "gold_needles": ["..."], "expected_answer": "..."}]}
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    qrels = [_qrel_from_dict(q) for q in data.get("qrels", [])]
    return GoldenDataset(
        name=data.get("name", path.stem),
        qrels=qrels,
        corpus_dir=data.get("corpus_dir"),
    )


def load_dataset_csv(path: str | Path, *, name: str | None = None, corpus_dir: str | None = None) -> GoldenDataset:
    """Load a golden dataset from CSV (columns: query, gold_doc?, gold_needles?, expected_answer?)."""
    path = Path(path)
    qrels: list[GoldenQrel] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            needles_raw = row.get("gold_needles") or row.get("needles") or ""
            qrels.append(
                GoldenQrel(
                    query=row["query"],
                    gold_doc=row.get("gold_doc") or None,
                    gold_needles=[s.strip() for s in needles_raw.split("|") if s.strip()],
                    expected_answer=row.get("expected_answer") or None,
                )
            )
    return GoldenDataset(name=name or path.stem, qrels=qrels, corpus_dir=corpus_dir)


def load_msmarco(qrels_path: str | Path = DEFAULT_MSMARCO_QRELS, *, corpus_dir: str | None = None) -> GoldenDataset:
    """Load the MS MARCO golden set produced by ``scripts/build_ir_corpus.py``."""
    ds = load_dataset_json(qrels_path)
    ds.name = "msmarco"
    if corpus_dir:
        ds.corpus_dir = corpus_dir
    return ds


DEFAULT_TYDI_ID_QRELS = "evals/fixtures/id_native_qrels.json"


def load_tydi_id(qrels_path: str | Path = DEFAULT_TYDI_ID_QRELS, *, corpus_dir: str | None = None) -> GoldenDataset:
    """Load the native Indonesian golden set from TyDi QA (``scripts/build_id_native_corpus.py``).

    Schema matches MS MARCO (``{query, gold_doc, gold_needles, expected_answer}``) -- natively
    collected (not translated), so the per-language ID claim is caveat-free.
    """
    ds = load_dataset_json(qrels_path)
    ds.name = "tydi-id"
    if corpus_dir:
        ds.corpus_dir = corpus_dir
    return ds


# --- synthetic fixture (offline, no download) ------------------------------

# A tiny inline corpus + qrels so the closed loop runs with zero network/zero API key.
SYNTHETIC_CORPUS: list[tuple[str, str]] = [
    (
        "doc_mitochondria",
        "The mitochondrion is the powerhouse of the cell, producing ATP via oxidative phosphorylation.",
    ),
    (
        "doc_photosynthesis",
        "Photosynthesis converts sunlight into chemical energy stored as glucose in plant chloroplasts.",
    ),
    (
        "doc_gravity",
        "Gravity is the force that attracts two masses toward each other; it gives objects weight on Earth.",
    ),
    (
        "doc_water_cycle",
        "The water cycle describes evaporation, condensation, and precipitation moving water through the biosphere.",
    ),
    (
        "doc_dna",
        "DNA carries the genetic instructions for life and is composed of four nucleotide bases arranged in a double helix.",
    ),
    (
        "doc_volcano",
        "A volcano is a rupture in the crust that allows molten magma, ash, and gases to escape from below the surface.",
    ),
]


def synthetic_chunks() -> list[Chunk]:
    """Build Chunk objects for the synthetic corpus (1 chunk per passage)."""
    return [
        Chunk(id=f"c_{doc_id}", doc_id=doc_id, content=content, metadata={"source": doc_id})
        for doc_id, content in SYNTHETIC_CORPUS
    ]


def synthetic_fixture() -> GoldenDataset:
    """A tiny built-in golden dataset for offline tests/demos (gold_needles match SYNTHETIC_CORPUS)."""
    qrels = [
        GoldenQrel(
            query="what produces ATP in the cell", gold_doc="doc_mitochondria", gold_needles=["powerhouse of the cell"]
        ),
        GoldenQrel(
            query="how do plants make energy",
            gold_doc="doc_photosynthesis",
            gold_needles=["converts sunlight into chemical energy"],
        ),
        GoldenQrel(
            query="what force attracts two masses together",
            gold_doc="doc_gravity",
            gold_needles=["force that attracts two masses"],
        ),
        GoldenQrel(
            query="how does water move through nature",
            gold_doc="doc_water_cycle",
            gold_needles=["evaporation, condensation"],
        ),
        GoldenQrel(
            query="what molecule stores genetic information", gold_doc="doc_dna", gold_needles=["genetic instructions"]
        ),
    ]
    return GoldenDataset(name="synthetic", qrels=qrels, corpus_dir=None)


def resolve_dataset(
    name: str,
    *,
    qrels_path: str | Path | None = None,
    corpus_dir: str | None = None,
) -> GoldenDataset:
    """Resolve a dataset by name: ``msmarco`` | ``synthetic`` | ``byo`` (needs qrels_path)."""
    name = name.lower()
    if name == "synthetic":
        return synthetic_fixture()
    if name == "msmarco":
        return load_msmarco(qrels_path or DEFAULT_MSMARCO_QRELS, corpus_dir=corpus_dir)
    if name in ("tydi-id", "tydi_id", "tydi"):
        return load_tydi_id(qrels_path or DEFAULT_TYDI_ID_QRELS, corpus_dir=corpus_dir)
    if name in ("byo", "json"):
        if not qrels_path:
            raise ValueError("dataset 'byo' requires --qrels PATH")
        return load_dataset_json(qrels_path)
    if name == "csv":
        if not qrels_path:
            raise ValueError("dataset 'csv' requires --qrels PATH")
        return load_dataset_csv(qrels_path, corpus_dir=corpus_dir)
    raise ValueError(f"Unknown dataset '{name}'. Available: synthetic | msmarco | byo | csv")
