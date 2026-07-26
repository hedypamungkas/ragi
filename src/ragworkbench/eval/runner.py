"""ragworkbench/eval/runner -- StandaloneEvalRunner (Mode A, retrieval-only).

The wedge. Drives a ``Retriever`` **directly** and scores ranking quality against a golden
qrels set -- NOT an agent loop. This isolates retrieval quality from LLM phrasing variance
(the confound in agent-driving eval runners) and runs with **zero API key**: a lexical-only
stack needs no embeddings, no chat client.

Loop: ``query -> retriever.retrieve -> ranked chunks -> IR metrics vs GoldenQrel ->
bootstrap CI per metric -> rubric PASS/FAIL``.

Mode B (end-to-end, faithfulness via a ChatClient) is layered on in v0.4.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import TYPE_CHECKING

from ragworkbench.eval.ci import bootstrap_ci
from ragworkbench.eval.metrics import compute_ranking_metric
from ragworkbench.eval.rubric import evaluate_rubric, format_rubric
from ragworkbench.eval.types import (
    CaseResult,
    EvalCase,
    EvalReport,
    EvalScore,
    GoldenDataset,
    MetricSummary,
)

if TYPE_CHECKING:
    from ragworkbench.protocols import ChatClient, Retriever

_logger = logging.getLogger(__name__)

_DEFAULT_METRICS = ("recall", "mrr", "ndcg", "precision", "hit")


class StandaloneEvalRunner:
    """Measure a retrieval stack against a golden dataset (retrieval-only, deterministic)."""

    def __init__(
        self,
        metrics: tuple[str, ...] = _DEFAULT_METRICS,
        *,
        k: int = 10,
        seed: int = 42,
    ):
        self.metrics = list(metrics)
        self.k = k
        self.seed = seed

    async def run(
        self,
        retriever: Retriever,
        dataset: GoldenDataset,
        *,
        n: int | None = None,
    ) -> EvalReport:
        """Run all scorers over the dataset (optionally subsampled to ``n`` qrels)."""
        qrels = list(dataset.qrels)
        if n is not None and 0 < n < len(qrels):
            qrels = random.Random(self.seed).sample(qrels, n)

        per_metric: dict[str, list[float]] = {f"retrieval_{m}": [] for m in self.metrics}
        per_case: list[CaseResult] = []

        for qrel in qrels:
            results = await retriever.retrieve(qrel.query, top_k=self.k)
            retrieved_contents = [r.chunk.content for r in results]
            doc_ids = [r.chunk.doc_id for r in results]
            scores: list[EvalScore] = []
            for metric in self.metrics:
                value = compute_ranking_metric(metric, retrieved_contents, qrel.gold_needles, self.k)
                name = f"retrieval_{metric}"
                per_metric[name].append(value)
                scores.append(EvalScore(name, round(value, 3), f"{metric}@{self.k}={value:.3f}"))
            per_case.append(CaseResult(query=qrel.query, retrieved_doc_ids=doc_ids, scores=scores))

        aggregate: dict[str, MetricSummary] = {}
        for name, values in per_metric.items():
            ci = bootstrap_ci(values, seed=self.seed)
            aggregate[name] = MetricSummary(
                mean=round(ci.mean, 3),
                ci_low=round(ci.lower, 3),
                ci_high=round(ci.upper, 3),
                n=len(values),
            )

        rubric = evaluate_rubric(aggregate, n=len(qrels))
        config_hash = _fingerprint(retriever, dataset, self.metrics, self.k, len(qrels))
        return EvalReport(
            dataset=dataset.name,
            n=len(qrels),
            top_k=self.k,
            per_case=per_case,
            aggregate=aggregate,
            rubric=rubric,
            config_hash=config_hash,
        )


def _fingerprint(retriever: Retriever, dataset: GoldenDataset, metrics: list[str], k: int, n: int) -> str:
    """A reproducibility stamp of the measured configuration."""
    payload = json.dumps(
        {
            "retriever": type(retriever).__name__,
            "dataset": dataset.name,
            "metrics": metrics,
            "k": k,
            "n": n,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def format_report(report: EvalReport, *, show_cases: int = 0) -> str:
    """Human-readable eval report (metrics + bootstrap CI + rubric verdict)."""
    lines = [
        f"=== EvalReport  dataset={report.dataset}  n={report.n}  top_k={report.top_k} ===",
        f"config_hash={report.config_hash}",
        "",
        f"{'metric':<22} {'mean':>7} {'95% CI':>20} {'n':>4}",
        "-" * 56,
    ]
    for name, s in report.aggregate.items():
        lines.append(f"{name:<22} {s.mean:>7.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}] {s.n:>4}")
    lines.append("")
    lines.append(format_rubric(report.rubric))
    if show_cases:
        lines.append("")
        lines.append(f"first {show_cases} cases:")
        for cr in report.per_case[:show_cases]:
            top = ",".join(cr.retrieved_doc_ids[:3]) or "-"
            lines.append(f"  q={cr.query[:60]!r:<62} top=[{top}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode B -- end-to-end (retrieve -> generate -> judge)
# ---------------------------------------------------------------------------

_RAG_GENERATION_PROMPT = (
    "Answer the question using ONLY the provided context. If the context does not contain "
    "the answer, say you don't know -- do not fabricate.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)


def _format_context(results: list) -> str:
    """Render retrieved results as a cited context block for the generation prompt."""
    if not results:
        return "(no context retrieved)"
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.chunk.metadata.get("source", r.chunk.doc_id)
        blocks.append(f"[{i}] [Source: {source}]\n{r.chunk.content}")
    return "\n\n".join(blocks)


class EndToEndEvalRunner:
    """Mode B: retrieve -> generate via ``ChatClient`` -> judge faithfulness/correctness/abstention.

    Clones the :class:`StandaloneEvalRunner` skeleton; the inner loop adds a generation step
    and runs the provided ``scorers`` (e.g. ``FaithfulnessScorer`` / ``AnswerCorrectnessScorer``
    / ``AbstentionScorer``) over the generated answer. Retrieval metrics are still computed.
    Needs a ``chat_client`` (a real key for live runs; a ``MockChatClient`` for tests) -- Mode A
    (:class:`StandaloneEvalRunner`) stays zero-key.
    """

    def __init__(
        self,
        chat_client: ChatClient,
        scorers,
        *,
        metrics: tuple[str, ...] = _DEFAULT_METRICS,
        k: int = 10,
        seed: int = 42,
        generation_prompt: str | None = None,
    ):
        if chat_client is None:
            raise ValueError("Mode B (EndToEndEvalRunner) requires a chat_client.")
        self._chat = chat_client
        self.scorers = list(scorers)
        self.metrics = list(metrics)
        self.k = k
        self.seed = seed
        self._prompt = generation_prompt or _RAG_GENERATION_PROMPT

    async def run(
        self,
        retriever: Retriever,
        dataset: GoldenDataset,
        *,
        n: int | None = None,
    ) -> EvalReport:
        """Run retrieval + generation + scoring over the dataset (optionally subsampled to ``n``)."""
        qrels = list(dataset.qrels)
        if n is not None and 0 < n < len(qrels):
            qrels = random.Random(self.seed).sample(qrels, n)

        per_metric: dict[str, list[float]] = {f"retrieval_{m}": [] for m in self.metrics}
        for sc in self.scorers:
            per_metric.setdefault(sc.name, [])
        per_case: list[CaseResult] = []

        for qrel in qrels:
            results = await retriever.retrieve(qrel.query, top_k=self.k)
            retrieved_contents = [r.chunk.content for r in results]
            doc_ids = [r.chunk.doc_id for r in results]
            ctx_dicts = [{"content": c, "doc_id": d} for c, d in zip(retrieved_contents, doc_ids, strict=True)]
            context_str = _format_context(results)
            scores: list[EvalScore] = []

            for metric in self.metrics:
                value = compute_ranking_metric(metric, retrieved_contents, qrel.gold_needles, self.k)
                name = f"retrieval_{metric}"
                per_metric[name].append(value)
                scores.append(EvalScore(name, round(value, 3), f"{metric}@{self.k}={value:.3f}"))

            answer = await self._generate(qrel.query, context_str)
            case = EvalCase(
                query=qrel.query,
                gold_needles=qrel.gold_needles,
                gold_doc=qrel.gold_doc,
                expected_answer=qrel.expected_answer,
                out_of_scope=qrel.out_of_scope,
            )
            for sc in self.scorers:
                s = await sc.score(case, answer, {"rag_results": ctx_dicts})
                per_metric[s.name].append(s.value)
                scores.append(s)
            per_case.append(CaseResult(query=qrel.query, retrieved_doc_ids=doc_ids, scores=scores))

        aggregate: dict[str, MetricSummary] = {}
        for name, values in per_metric.items():
            ci = bootstrap_ci(values, seed=self.seed)
            aggregate[name] = MetricSummary(
                mean=round(ci.mean, 3),
                ci_low=round(ci.lower, 3),
                ci_high=round(ci.upper, 3),
                n=len(values),
            )

        rubric = evaluate_rubric(aggregate, n=len(qrels))
        config_hash = _fingerprint(
            retriever,
            dataset,
            self.metrics + [sc.name for sc in self.scorers],
            self.k,
            len(qrels),
        )
        return EvalReport(
            dataset=dataset.name,
            n=len(qrels),
            top_k=self.k,
            per_case=per_case,
            aggregate=aggregate,
            rubric=rubric,
            config_hash=config_hash,
        )

    async def _generate(self, query: str, context_str: str) -> str:
        try:
            return await self._chat.complete(
                [{"role": "user", "content": self._prompt.format(context=context_str, query=query)}]
            )
        except Exception as exc:  # noqa: BLE001 -- never crash the eval; score the empty answer
            _logger.warning("generation failed (scoring empty answer): %s", exc)
            return ""
