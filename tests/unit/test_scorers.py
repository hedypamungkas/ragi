"""Unit tests for ragi.eval.scorers -- BaseScorer adapters (async, no LLM/network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragi.eval.scorers import (
    BootstrapCIScorer,
    CitationScorer,
    RetrievalMetricScorer,
    _extract_content,
)
from ragi.eval.types import EvalCase


class TestExtractContent:
    def test_dict_with_content(self):
        assert _extract_content({"content": "hello"}) == "hello"

    def test_dict_without_content(self):
        assert _extract_content({"source": "x"}) == ""

    def test_object_with_content_attr(self):
        assert _extract_content(SimpleNamespace(content="obj")) == "obj"

    def test_object_without_content_attr(self):
        assert _extract_content(SimpleNamespace(other="x")) == ""


def _case(needles):
    return EvalCase(query="q", gold_needles=needles)


class TestRetrievalMetricScorer:
    async def test_no_gold_returns_zero(self):
        s = await RetrievalMetricScorer().score(_case([]), "out", {"rag_results": [{"content": "x"}]})
        assert s.value == 0.0 and "no gold_needles" in s.reason

    async def test_no_rag_results_returns_zero(self):
        s = await RetrievalMetricScorer().score(_case(["needle"]), "out", {})
        assert s.value == 0.0 and "no rag_results" in s.reason

    async def test_recall_hits(self):
        rag = [{"content": "the powerhouse of the cell"}, {"content": "unrelated"}]
        s = await RetrievalMetricScorer(metric="recall", k=10).score(_case(["powerhouse"]), "out", {"rag_results": rag})
        assert s.value == 1.0
        assert "recall@10" in s.reason

    async def test_reads_object_chunks(self):
        rag = [SimpleNamespace(content="powerhouse of the cell")]
        s = await RetrievalMetricScorer(metric="hit").score(_case(["powerhouse"]), "out", {"rag_results": rag})
        assert s.value == 1.0

    async def test_precision_partial(self):
        rag = [{"content": "powerhouse"}, {"content": "noise"}, {"content": "noise2"}]
        s = await RetrievalMetricScorer(metric="precision", k=2).score(
            _case(["powerhouse"]), "out", {"rag_results": rag}
        )
        # only 1 of top-2 retrieved is gold
        assert s.value == pytest.approx(0.5)


class TestCitationScorer:
    async def test_all_citations_resolve(self):
        rag = [{"source": "a"}, {"source": "b"}]
        s = await CitationScorer().score(_case([]), "see [1] [2]", {"rag_results": rag})
        assert s.value == 1.0 and "citation_precision" in s.reason

    async def test_unresolved_citation(self):
        rag = [{"source": "a"}]
        s = await CitationScorer().score(_case([]), "see [9]", {"rag_results": rag})
        assert s.value == 0.0


class _ConstantScorer(RetrievalMetricScorer):
    """Inner scorer stub that always returns a fixed value (for BootstrapCIScorer)."""

    def __init__(self, value):
        super().__init__()
        self._value = value

    async def score(self, case, output, context):
        from ragi.eval.types import EvalScore

        return EvalScore("constant", self._value, "stub")


class TestBootstrapCIScorer:
    async def test_samples_from_context(self):
        s = await BootstrapCIScorer().score(_case([]), "out", {"samples": [0.4, 0.6, 0.8, 0.5]})
        assert 0.0 < s.value < 1.0
        assert "CI=[" in s.reason and "n=4" in s.reason

    async def test_no_samples_no_inner_returns_zero(self):
        s = await BootstrapCIScorer().score(_case([]), "out", {})
        assert s.value == 0.0 and "no samples" in s.reason

    async def test_no_samples_with_inner_evaluates_single_case(self):
        # N=1 -> the honest full-width CI has lower bound 0.0 (a CI gate can never pass
        # on a single sample -- that's the point of the "grow N" guard).
        s = await BootstrapCIScorer(inner=_ConstantScorer(0.7)).score(_case([]), "out", {})
        assert s.value == 0.0
        assert "grow N" in s.reason  # the n<2 note

    async def test_aggregate_runs_inner_over_items(self):
        items = [(_case([]), "o", {}) for _ in range(5)]
        s = await BootstrapCIScorer(inner=_ConstantScorer(0.8)).aggregate(items)
        assert s.value == pytest.approx(0.8)
        assert "n=5" in s.reason

    async def test_aggregate_without_inner_returns_zero(self):
        s = await BootstrapCIScorer().aggregate([(_case([]), "o", {})])
        assert s.value == 0.0 and "no inner scorer" in s.reason
