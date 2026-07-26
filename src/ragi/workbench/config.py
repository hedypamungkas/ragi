"""ragi/workbench/config -- load a pipeline config (YAML file or dict).

A pipeline config is a flat dict consumed by ``ragi.build_pipeline``::

    enabled: true
    chunker: paragraph
    retriever: bm25
    top_k: 10
    k1: 1.5
    b: 0.75
    documents:
      - path: data/ir_corpus
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a pipeline config from a YAML path or accept a dict verbatim."""
    if isinstance(source, dict):
        return dict(source)
    path = Path(source)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping (got {type(data).__name__})")
    return data
