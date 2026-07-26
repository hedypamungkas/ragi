"""ragworkbench/eval/rubric -- production-readiness PASS/FAIL decision.

A dimension **FAILS if its bootstrap CI lower bound is below threshold** (not the mean) --
this is the discipline that prevents false greens: a point estimate of 0.90 over 5 queries
is meaningless without a CI. v0.1 ships the *retrieval* subset of the 9-dimension rubric
(faithfulness / correctness / abstention / noise arrive in v0.4 with Mode B generation eval).

The weights are the retrieval-applicable projection of the doc's rubric
(docs/rag-production-readiness-eval.md): ranking is the single highest-leverage dimension.
"""

from __future__ import annotations

from ragworkbench.eval.types import (
    DimensionResult,
    MetricSummary,
    RubricDecision,
    RubricDimension,
)

# v0.1 retrieval-focused rubric. Thresholds anchored on the shipped MS MARCO baselines
# (BM25 recall@10 ~0.898, MRR ~0.44, nDCG ~0.55; +rerank ~0.977/0.63/0.72 in v0.2).
DEFAULT_RUBRIC: list[RubricDimension] = [
    # Generation dims (Mode B, v0.4) -- heaviest weight on faithfulness (anti-hallucination).
    RubricDimension("faithfulness", "faithfulness", 0.80, 0.18, "claim-decomposition NLI coverage"),
    # Ranking is one 9-dim dimension (weight 0.17), split across recall/mrr/ndcg.
    RubricDimension("ranking_recall", "retrieval_recall", 0.85, 0.08, "recall@10 CI lower bound"),
    RubricDimension("ranking_mrr", "retrieval_mrr", 0.40, 0.05, "MRR@10 CI lower bound"),
    RubricDimension("ranking_ndcg", "retrieval_ndcg", 0.55, 0.04, "nDCG@10 CI lower bound"),
    RubricDimension("answer_correctness", "answer_correctness", 0.75, 0.13, "factual correctness vs gold answer"),
    RubricDimension("abstention", "abstention", 0.80, 0.09, "OOS negative rejection"),
    # Infra dims -- NA in v0.4 (no scorer populates them); declared for the full 9-dim report.
    RubricDimension("ingestion_fidelity", "ingestion_fidelity", 1.0, 0.10, "parser/chunker fidelity (infra; NA)"),
    RubricDimension("noise_robustness", "noise", 0.80, 0.09, "noise-injection delta (needs fixture; NA)"),
    RubricDimension("robustness", "robustness", 1.0, 0.08, "graceful degradation (infra; NA)"),
    RubricDimension("performance", "performance", 1.0, 0.08, "p95 latency / cost (infra; NA)"),
]
# Confidence (0.08) is the separate min_n check in evaluate_rubric, not a metric dim.
# Weights sum to 0.92 across the 10 metric dims + 0.08 confidence = 1.00.

DEFAULT_MIN_N = 120  # below this a CI is too wide to be defensible; the gate hard-fails


def evaluate_rubric(
    aggregate: dict[str, MetricSummary],
    *,
    n: int,
    rubric: list[RubricDimension] | None = None,
    min_n: int = DEFAULT_MIN_N,
) -> RubricDecision:
    """Evaluate the rubric against a run's aggregate metrics.

    Each dimension: ``PASS`` if ``aggregate[metric_key].ci_low >= threshold``;
    ``NA`` if the metric is absent (e.g. a v0.4-only dimension not yet measured);
    ``FAIL`` otherwise. ``overall_pass`` requires ``min_n_ok`` AND every applicable
    (present) dimension to PASS.
    """
    rubric = rubric if rubric is not None else DEFAULT_RUBRIC
    dims: list[DimensionResult] = []
    weight_sum = 0.0
    pass_weight = 0.0
    for d in rubric:
        summary = aggregate.get(d.metric_key)
        if summary is None:
            dims.append(DimensionResult(d.name, "NA", f"metric '{d.metric_key}' not measured (v0.4)"))
            continue
        weight_sum += d.weight
        if summary.ci_low >= d.threshold:
            status = "PASS"
            pass_weight += d.weight
        else:
            status = "FAIL"
        dims.append(
            DimensionResult(
                d.name,
                status,
                f"{d.metric_key} CI-low {summary.ci_low:.3f} {'>=' if status == 'PASS' else '<'} {d.threshold}",
            )
        )

    weighted_score = round(pass_weight / weight_sum, 3) if weight_sum > 0 else 0.0
    min_n_ok = n >= min_n
    overall_pass = min_n_ok and all(dr.status == "PASS" for dr in dims if dr.status != "NA")
    return RubricDecision(overall_pass=overall_pass, weighted_score=weighted_score, min_n_ok=min_n_ok, dimensions=dims)


def format_rubric(decision: RubricDecision) -> str:
    """Human-readable rubric verdict block."""
    lines = [
        f"  Rubric: {'PASS' if decision.overall_pass else 'FAIL'} "
        f"(weighted {decision.weighted_score:.0%} of applicable dims; "
        f"min_n={'OK' if decision.min_n_ok else 'TOO FEW'})"
    ]
    for dr in decision.dimensions:
        lines.append(f"    [{dr.status:>4}] {dr.name:<18} {dr.detail}")
    return "\n".join(lines)
