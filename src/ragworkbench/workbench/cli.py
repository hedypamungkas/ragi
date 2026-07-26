"""ragworkbench/workbench/cli -- the ``ragwb`` command: ingest | eval | compare | serve | export-tool-schema.

The closed-loop driver:

    ragwb ingest <config>                          # validate config + index corpus + probe
    ragwb eval <config> --dataset msmarco -n 120    # measure (Mode A, no API key)
    ragwb compare <cfgA> <cfgB> --metric recall@10  # A/B iterate
    ragwb serve <config> --transport stdio          # ship as a read-only MCP server
    ragwb export-tool-schema --format openai        # ship as an OpenAI/Claude tool spec
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import ragworkbench as rwb
from ragworkbench.workbench.config import load_config


def _prepare_corpus(config: dict, dataset) -> tuple[dict, str | None]:
    """Return (config_with_documents, synthetic_tmp_dir|None).

    For the synthetic dataset, materialize the inline corpus to a temp dir so the standard
    ``build_pipeline`` path indexes it uniformly (doc_id == gold_doc). Real datasets keep
    their own ``documents``/``corpus_dir``.
    """
    if dataset.name == "synthetic":
        from ragworkbench.eval.datasets import SYNTHETIC_CORPUS

        tmp = tempfile.mkdtemp(prefix="rwb_synth_")
        for doc_id, content in SYNTHETIC_CORPUS:
            (Path(tmp) / f"{doc_id}.txt").write_text(content + "\n", encoding="utf-8")
        cfg = dict(config)
        cfg["documents"] = [{"path": tmp}]
        return cfg, tmp
    return config, None


def _build_retriever(config: dict, dataset):
    rwb.register_builtins()
    cfg, _ = _prepare_corpus(config, dataset)
    retriever = rwb.build_pipeline(cfg)
    if retriever is None:
        raise SystemExit(
            "ERROR: pipeline produced no retriever (disabled, or no documents loaded at the "
            "configured path). Check `documents:` / `corpus_dir` and run `ragwb ingest`."
        )
    return retriever


def _resolve_metric(metric: str) -> tuple[str, int]:
    """Accept 'recall@10' or 'recall'; return (metric_name, k)."""
    name, _, kstr = metric.partition("@")
    try:
        return name.strip(), int(kstr) if kstr else 10
    except ValueError:
        return name.strip(), 10


# --- subcommands -----------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rwb.register_builtins()
    print(f"chunkers: {rwb.chunker_registry.list_available()}")
    print(f"retrievers: {rwb.retriever_registry.list_available()}")
    print(f"parsers: {rwb.parser_registry.list_available()}")
    # Use the synthetic corpus for a probe if no documents configured (offline-safe).
    from ragworkbench.eval.datasets import resolve_dataset

    dataset = resolve_dataset("synthetic")
    retriever = _build_retriever(config, dataset)
    probe = asyncio.run(retriever.retrieve(dataset.qrels[0].query, top_k=3))
    print(f"\nretriever: {type(retriever).__name__}")
    print(f"probe query: {dataset.qrels[0].query!r}")
    for r in probe:
        print(f"  [{round(r.score, 3)}] {r.chunk.doc_id}: {r.chunk.content[:80]}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ragworkbench.eval.datasets import resolve_dataset
    from ragworkbench.eval.runner import StandaloneEvalRunner, format_report

    config = load_config(args.config)
    dataset = resolve_dataset(args.dataset, qrels_path=args.qrels, corpus_dir=args.corpus)
    retriever = _build_retriever(config, dataset)
    metric_names = [
        m for m in (args.metrics.split(",") if args.metrics else "recall,mrr,ndcg,precision,hit".split(","))
    ]
    if args.mode == "end_to_end":
        report = asyncio.run(_run_end_to_end(retriever, dataset, args, metric_names))
    else:
        runner = StandaloneEvalRunner(metrics=tuple(metric_names), k=args.k)
        report = asyncio.run(runner.run(retriever, dataset, n=args.n))
    print(format_report(report, show_cases=args.show_cases))
    if args.strict:
        return 0 if report.rubric.overall_pass else 1
    return 0


async def _run_end_to_end(retriever, dataset, args, metric_names):
    """Mode B: build an OpenAIChatClient from env + run the EndToEndEvalRunner."""
    import os

    from ragworkbench.chat.openai import OpenAIChatClient
    from ragworkbench.eval.faithfulness import FaithfulnessScorer
    from ragworkbench.eval.generation_scorers import AbstentionScorer, AnswerCorrectnessScorer
    from ragworkbench.eval.runner import EndToEndEvalRunner

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("ERROR: --mode end_to_end requires OPENAI_API_KEY")
    chat = OpenAIChatClient(api_key=key, model=args.chat_model, base_url=os.environ.get("OPENAI_BASE_URL") or None)
    scorers = [FaithfulnessScorer(chat), AnswerCorrectnessScorer(chat), AbstentionScorer()]
    runner = EndToEndEvalRunner(chat, scorers, metrics=tuple(metric_names), k=args.k)
    return await runner.run(retriever, dataset, n=args.n)


def cmd_compare(args: argparse.Namespace) -> int:
    from ragworkbench.eval.compare import compare, format_comparison
    from ragworkbench.eval.datasets import resolve_dataset

    dataset = resolve_dataset(args.dataset, qrels_path=args.qrels, corpus_dir=args.corpus)
    retriever_a = _build_retriever(load_config(args.config_a), dataset)
    retriever_b = _build_retriever(load_config(args.config_b), dataset)
    metric, k = _resolve_metric(args.metric)
    rep = asyncio.run(compare(retriever_a, retriever_b, dataset, metric=metric, k=k, n=args.n))
    print(format_comparison(rep, name_a=Path(args.config_a).stem, name_b=Path(args.config_b).stem))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve a pipeline as a read-only MCP server (stdio default, or --transport http).

    Needs the ``[mcp]`` extra. Lexical + rerank configs serve directly; semantic/hybrid
    degrade to keyword (the CLI does not wire an embedder in v0.3).
    """
    from ragworkbench.export.mcp import serve_from_config

    config = load_config(args.config)
    serve_from_config(config, transport=args.transport, host=args.host, port=args.port)
    return 0


def cmd_export_tool_schema(args: argparse.Namespace) -> int:
    """Print the search tool schema in the requested agent-calling format."""
    from ragworkbench.export.toolschema import (
        dumps,
        search_tool_schema,
        to_anthropic,
        to_mcp,
        to_openai,
    )

    spec = search_tool_schema()
    out = {"mcp": to_mcp(spec), "openai": to_openai(spec), "anthropic": to_anthropic(spec), "base": spec}[args.format]
    print(dumps(out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ragwb", description="rag-workbench closed-loop CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="validate config + index corpus + probe")
    p_ingest.add_argument("config", help="path to pipeline YAML")
    p_ingest.set_defaults(func=cmd_ingest)

    p_eval = sub.add_parser("eval", help="measure a stack (Mode A, no API key)")
    p_eval.add_argument("config", help="path to pipeline YAML")
    p_eval.add_argument("--dataset", default="synthetic", help="synthetic | msmarco | byo | csv")
    p_eval.add_argument("--qrels", default=None, help="qrels path (for byo/csv/msmarco override)")
    p_eval.add_argument("--corpus", default=None, help="corpus dir override")
    p_eval.add_argument("-n", type=int, default=None, help="subsample N qrels")
    p_eval.add_argument("-k", type=int, default=10, help="top_k for retrieval")
    p_eval.add_argument("--metrics", default=None, help="comma list: recall,mrr,ndcg,precision,hit")
    p_eval.add_argument(
        "--mode",
        choices=["retrieval", "end_to_end"],
        default="retrieval",
        help="retrieval-only (Mode A, no key) or end-to-end (Mode B, needs OPENAI_API_KEY)",
    )
    p_eval.add_argument("--chat-model", default="gpt-4o-mini", help="chat/judge model for --mode end_to_end")
    p_eval.add_argument("--show-cases", type=int, default=3, help="print first N per-case rows")
    p_eval.add_argument("--strict", action="store_true", help="exit 1 if rubric FAILs (CI gate)")
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare", help="A/B compare two stacks (paired bootstrap CI)")
    p_cmp.add_argument("config_a")
    p_cmp.add_argument("config_b")
    p_cmp.add_argument("--dataset", default="synthetic")
    p_cmp.add_argument("--qrels", default=None)
    p_cmp.add_argument("--corpus", default=None)
    p_cmp.add_argument("--metric", default="recall@10", help="e.g. recall@10, ndcg@10, mrr@10")
    p_cmp.add_argument("-n", type=int, default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_serve = sub.add_parser("serve", help="serve a pipeline as a read-only MCP server")
    p_serve.add_argument("config", help="path to pipeline YAML")
    p_serve.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio", help="MCP transport (default stdio)"
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="HTTP host (ignored for stdio)")
    p_serve.add_argument("--port", type=int, default=8000, help="HTTP port (ignored for stdio)")
    p_serve.set_defaults(func=cmd_serve)

    p_exp = sub.add_parser("export-tool-schema", help="print the search tool schema (mcp/openai/anthropic)")
    p_exp.add_argument("--format", choices=["mcp", "openai", "anthropic", "base"], default="base")
    p_exp.set_defaults(func=cmd_export_tool_schema)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 -- CLI top-level guard
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
