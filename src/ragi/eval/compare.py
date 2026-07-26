"""ragi/eval/compare -- A/B stack comparison with a paired bootstrap CI.

The *iterate* step made rigorous. Runs two retrieval stacks over the SAME golden dataset,
computes the per-query metric for each, then bootstraps the **paired delta** (B - A) to
answer: *"is stack B significantly better than A?"* -- significant when the bootstrap CI on
the delta excludes zero.

Paired (per-query) differencing is more powerful than comparing two independent CIs: it
removes between-query difficulty variance, so a real improvement is detectable at smaller N.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ragi.eval.ci import bootstrap_ci
from ragi.eval.metrics import compute_ranking_metric
from ragi.eval.types import ComparisonReport, GoldenDataset

if TYPE_CHECKING:
    from ragi.protocols import Retriever


async def compare(
    retriever_a: Retriever,
    retriever_b: Retriever,
    dataset: GoldenDataset,
    *,
    metric: str = "recall",
    k: int = 10,
    n: int | None = None,
    seed: int = 42,
    n_boot: int = 2000,
) -> ComparisonReport:
    """Compare two retrieval stacks on one metric; report the paired delta + its bootstrap CI."""
    qrels = list(dataset.qrels)
    if n is not None and 0 < n < len(qrels):
        qrels = random.Random(seed).sample(qrels, n)  # nosec B311 - seeded eval subsampling, reproducible not cryptographic

    values_a: list[float] = []
    values_b: list[float] = []
    deltas: list[float] = []

    for qrel in qrels:
        ra = await retriever_a.retrieve(qrel.query, top_k=k)
        rb = await retriever_b.retrieve(qrel.query, top_k=k)
        va = compute_ranking_metric(metric, [r.chunk.content for r in ra], qrel.gold_needles, k)
        vb = compute_ranking_metric(metric, [r.chunk.content for r in rb], qrel.gold_needles, k)
        values_a.append(va)
        values_b.append(vb)
        deltas.append(vb - va)

    mean_a = sum(values_a) / len(values_a) if values_a else 0.0
    mean_b = sum(values_b) / len(values_b) if values_b else 0.0
    ci = bootstrap_ci(deltas, seed=seed, n_boot=n_boot)
    significant = ci.lower > 0 or ci.upper < 0
    return ComparisonReport(
        metric=metric,
        k=k,
        n=len(qrels),
        mean_a=round(mean_a, 3),
        mean_b=round(mean_b, 3),
        delta=round(mean_b - mean_a, 3),
        ci_low=round(ci.lower, 3),
        ci_high=round(ci.upper, 3),
        significant=significant,
    )


def format_comparison(report: ComparisonReport, *, name_a: str = "A", name_b: str = "B") -> str:
    direction = "improvement" if report.delta > 0 else ("regression" if report.delta < 0 else "no change")
    verdict = "SIGNIFICANT" if report.significant else "not significant"
    return (
        f"=== A/B comparison  metric={report.metric}@{report.k}  n={report.n} ===\n"
        f"  {name_a} mean = {report.mean_a:.3f}\n"
        f"  {name_b} mean = {report.mean_b:.3f}\n"
        f"  delta ({name_b} - {name_a}) = {report.delta:+.3f}  "
        f"95% CI [{report.ci_low:+.3f}, {report.ci_high:+.3f}]  ->  {direction}, {verdict}"
    )
