"""Integration test: StandaloneEvalRunner drives a BM25 retriever over the synthetic fixture.

This is the v0.1 closed loop, exercised offline with NO API key. Asserts structural
properties (CI shape, rubric dimensions, subsample, min_n honesty) rather than exact
metric values.
"""

from __future__ import annotations

from ragworkbench.eval.compare import compare
from ragworkbench.eval.datasets import synthetic_chunks, synthetic_fixture
from ragworkbench.eval.rubric import evaluate_rubric
from ragworkbench.eval.runner import StandaloneEvalRunner
from ragworkbench.retrieval.retriever import BM25Retriever, KeywordRetriever


async def test_runner_synthetic_closed_loop():
    retriever = BM25Retriever(chunks=synthetic_chunks())
    runner = StandaloneEvalRunner(k=10)
    report = await runner.run(retriever, synthetic_fixture())

    assert report.n == len(synthetic_fixture().qrels)
    assert "retrieval_recall" in report.aggregate
    s = report.aggregate["retrieval_recall"]
    assert 0.0 <= s.ci_low <= s.mean <= s.ci_high <= 1.0
    # The synthetic set is lexically easy -> recall should be high.
    assert s.mean >= 0.8
    # Rubric covers all configured metric dimensions.
    assert len(report.rubric.dimensions) >= 4
    assert report.config_hash  # reproducibility stamp present


async def test_runner_subsample_n():
    retriever = BM25Retriever(chunks=synthetic_chunks())
    runner = StandaloneEvalRunner(k=10)
    report = await runner.run(retriever, synthetic_fixture(), n=2)
    assert report.n == 2


async def test_rubric_fails_on_small_n_honest_ci():
    # n=5 < default min_n=120 -> the gate MUST fail regardless of how good the numbers are.
    retriever = BM25Retriever(chunks=synthetic_chunks())
    runner = StandaloneEvalRunner(k=10)
    report = await runner.run(retriever, synthetic_fixture())
    assert report.rubric.min_n_ok is False
    assert report.rubric.overall_pass is False


def test_rubric_passes_when_ci_lower_bound_meets_threshold():
    # Fabricate a strong aggregate and a large n -> PASS.
    from ragworkbench.eval.types import MetricSummary

    agg = {
        "retrieval_recall": MetricSummary(0.95, 0.90, 0.99, 120),
        "retrieval_mrr": MetricSummary(0.6, 0.5, 0.7, 120),
        "retrieval_ndcg": MetricSummary(0.7, 0.6, 0.8, 120),
        "retrieval_precision": MetricSummary(0.3, 0.25, 0.35, 120),
        "retrieval_hit": MetricSummary(0.95, 0.90, 0.99, 120),
    }
    decision = evaluate_rubric(agg, n=120)
    assert decision.overall_pass is True
    assert decision.min_n_ok is True


async def test_ab_compare_paired_delta_shape():
    a = KeywordRetriever(chunks=synthetic_chunks())
    b = BM25Retriever(chunks=synthetic_chunks())
    rep = await compare(a, b, synthetic_fixture(), metric="recall", k=10)
    assert rep.n == len(synthetic_fixture().qrels)
    assert -1.0 <= rep.delta <= 1.0
    assert rep.ci_low <= rep.delta <= rep.ci_high or rep.ci_low == rep.ci_high  # degenerate when no variance
