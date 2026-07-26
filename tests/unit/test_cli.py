"""Unit tests for the ``ragi`` CLI subcommands (offline, no key).

Exercises ingest / eval (Mode A, synthetic) / compare / export-tool-schema / serve(error) /
end_to_end(no-key) via ``main([...])``. Covers ``workbench/cli.py`` end-to-end.
"""

from __future__ import annotations

import json

import pytest

from ragi.workbench.cli import main

BM25 = "configs/bm25_baseline.yaml"
KEYWORD = "configs/keyword_baseline.yaml"


def test_ingest(capsys):
    rc = main(["ingest", BM25])
    out = capsys.readouterr().out
    assert rc == 0
    assert "chunkers:" in out and "retrievers:" in out
    assert "BM25Retriever" in out


def test_eval_mode_a_synthetic(capsys):
    rc = main(["eval", BM25, "--dataset", "synthetic", "--show-cases", "0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "EvalReport" in out and "retrieval_recall" in out


def test_eval_strict_exits_nonzero_when_rubric_fails(capsys):
    # synthetic n=5 < min_n=120 -> rubric FAILs -> --strict exits 1.
    rc = main(["eval", BM25, "--dataset", "synthetic", "--strict"])
    assert rc == 1


def test_compare(capsys):
    rc = main(["compare", KEYWORD, BM25, "--dataset", "synthetic", "--metric", "recall@10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "A/B comparison" in out


@pytest.mark.parametrize("fmt", ["base", "mcp", "openai", "anthropic"])
def test_export_tool_schema_formats(capsys, fmt):
    rc = main(["export-tool-schema", "--format", fmt])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    if fmt == "openai":
        assert data["type"] == "function" and "parameters" in data["function"]
    elif fmt == "mcp":
        assert "inputSchema" in data
    elif fmt == "anthropic":
        assert "input_schema" in data
    else:
        assert data["name"] == "search"


def test_eval_end_to_end_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["eval", BM25, "--dataset", "synthetic", "--mode", "end_to_end"])


def test_serve_missing_config_returns_error(capsys):
    rc = main(["serve", "/nonexistent/config.yaml"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "ERROR" in err


def test_no_subcommand_errors(capsys):
    with pytest.raises(SystemExit):
        main([])
