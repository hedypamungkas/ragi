#!/usr/bin/env python3
"""Build a real RAG eval corpus + qrels from MS MARCO (v2.1).

Writes (idempotent; HF-cached after first run):
- ``data/ir_corpus/p<sha>.txt`` -- one file per unique passage (gitignored; MS MARCO is
  research-use, so passage TEXT stays out of the repo).
- ``evals/fixtures/ir_qrels.json`` -- committed, license-light: only query + answer + gold
  passage id + a short distinctive snippet (no passage body). Consumed by
  ``ragi.eval.datasets.load_msmarco``.

    pip install -e ".[eval]"   # needs `datasets`
    python scripts/build_ir_corpus.py --n 120

The first run downloads MS MARCO v2.1 (HF-cached thereafter). ``--config v1.1`` is a
fallback if v2.1 is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "ir_corpus"
QRELS = ROOT / "evals" / "fixtures" / "ir_qrels.json"


def _passage_id(text: str) -> str:
    return "p" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _snippet(text: str, words: int = 10) -> str:
    """A short, likely-unique prefix for content-based recall fallback."""
    return " ".join(text.split()[:words])


def _load_rows(config: str, n: int):
    from datasets import load_dataset

    ds = load_dataset("microsoft/ms_marco", config, split="validation")
    return ds.select(range(min(n, len(ds))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120, help="number of queries to sample")
    ap.add_argument("--config", default="v2.1", choices=["v2.1", "v1.1"])
    ap.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    ap.add_argument("--out", type=Path, default=QRELS)
    args = ap.parse_args()

    args.corpus.mkdir(parents=True, exist_ok=True)
    try:
        rows = _load_rows(args.config, args.n)
    except Exception as e:
        print(f"Failed to load microsoft/ms_marco ({args.config}): {e}", file=sys.stderr)
        print(
            "Tip: pip install -e '.[eval]' for the `datasets` lib; the first run downloads MS MARCO (HF-cached).",
            file=sys.stderr,
        )
        return 2

    qrels: list[dict] = []
    seen_passages: dict[str, str] = {}  # pid -> filename
    written = 0

    for row in rows:
        query = (row.get("query") or "").strip()
        answers = row.get("answers") or []
        answer = answers[0].strip() if answers and answers[0].strip() else ""
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") or []
        selected = passages.get("is_selected") or []
        if not query or not answer or not texts:
            continue
        gold_ids: list[str] = []
        first_gold_text = ""
        for text, is_gold in zip(texts, selected, strict=False):
            text = (text or "").strip()
            if not text:
                continue
            pid = _passage_id(text)
            if pid not in seen_passages:
                fname = f"{pid}.txt"
                (args.corpus / fname).write_text(text + "\n")
                seen_passages[pid] = fname
                written += 1
            if is_gold:
                gold_ids.append(pid)
                if not first_gold_text:
                    first_gold_text = text[:200]
        if not gold_ids:
            continue
        qrels.append(
            {
                "query": query,
                "gold_doc": gold_ids[0],  # chunk doc_id == passage pid (loader strips .txt)
                "gold_needles": [_snippet(first_gold_text)],
                "expected_answer": answer,
            }
        )

    if not qrels:
        print("No qualifying rows (need query + answer + >=1 selected passage).", file=sys.stderr)
        return 3

    data = {
        "name": "msmarco",
        "_comment": (
            "Real RAG eval qrels from MS MARCO v2.1 (validation). Built by "
            "scripts/build_ir_corpus.py. License-light: only query/answer/gold-id/snippet "
            "(no passage body). gold_doc is the data/ir_corpus/<pid>.txt chunk doc_id."
        ),
        "corpus_dir": str(args.corpus.relative_to(ROOT)),
        "qrels": qrels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(qrels)} qrels -> {args.out} | {written} unique passages -> {args.corpus}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
