"""Unit tests for the full 9-dimension production-readiness rubric."""

from __future__ import annotations

from ragi.eval.rubric import DEFAULT_RUBRIC, evaluate_rubric
from ragi.eval.types import MetricSummary


def test_metric_dim_weights_sum_to_092():
    # 10 metric dims sum to 0.92; +0.08 confidence (the min_n check) = 1.00.
    total = sum(d.weight for d in DEFAULT_RUBRIC)
    assert abs(total - 0.92) < 0.001


def test_full_report_active_vs_na_dims():
    agg = {
        "faithfulness": MetricSummary(0.90, 0.85, 0.95, 120),
        "retrieval_recall": MetricSummary(0.90, 0.86, 0.94, 120),
        "retrieval_mrr": MetricSummary(0.50, 0.45, 0.55, 120),
        "retrieval_ndcg": MetricSummary(0.70, 0.60, 0.80, 120),
        "answer_correctness": MetricSummary(0.85, 0.80, 0.90, 120),
        "abstention": MetricSummary(0.90, 0.85, 0.95, 120),
    }
    decision = evaluate_rubric(agg, n=120)
    statuses = {d.name: d.status for d in decision.dimensions}
    # Mode-B-measurable dims -> PASS (CI lower bounds meet thresholds).
    assert statuses["faithfulness"] == "PASS"
    assert statuses["answer_correctness"] == "PASS"
    assert statuses["abstention"] == "PASS"
    assert statuses["ranking_recall"] == "PASS"
    # Infra / noise dims -> NA (no metric populated).
    assert statuses["ingestion_fidelity"] == "NA"
    assert statuses["noise_robustness"] == "NA"
    assert statuses["robustness"] == "NA"
    assert statuses["performance"] == "NA"
    assert decision.min_n_ok is True
    assert decision.overall_pass is True


def test_retrieval_only_run_leaves_generation_dims_na():
    # A Mode A aggregate (retrieval only) -> generation dims NA, don't block overall_pass.
    agg = {
        "retrieval_recall": MetricSummary(0.90, 0.86, 0.94, 120),
        "retrieval_mrr": MetricSummary(0.50, 0.45, 0.55, 120),
        "retrieval_ndcg": MetricSummary(0.70, 0.60, 0.80, 120),
    }
    decision = evaluate_rubric(agg, n=120)
    statuses = {d.name: d.status for d in decision.dimensions}
    assert statuses["faithfulness"] == "NA"
    assert statuses["ranking_recall"] == "PASS"
    assert decision.overall_pass is True
