"""Unit tests for ragi.eval.registry -- ScorerRegistry (config-driven scorer factories)."""

from __future__ import annotations

import pytest

from ragi.eval.registry import ScorerRegistry, register_default_scorers
from ragi.eval.scorers import BootstrapCIScorer, CitationScorer, RetrievalMetricScorer


@pytest.fixture(autouse=True)
def _reset_registry():
    # Each test starts from the default-registered scorers (imported at module load).
    register_default_scorers()
    yield


class TestRegistryBasics:
    def test_defaults_registered(self):
        names = ScorerRegistry.list_available()
        for expected in ("retrieval_metric", "retrieval_recall", "citation_precision", "bootstrap_ci"):
            assert expected in names

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            ScorerRegistry.create("nope")

    def test_register_and_create_custom(self):
        ScorerRegistry.register("custom", lambda **kw: RetrievalMetricScorer(**kw))
        inst = ScorerRegistry.create("custom", metric="mrr")
        assert isinstance(inst, RetrievalMetricScorer) and inst.metric == "mrr"

    def test_build_list_by_name(self):
        scorers = ScorerRegistry.build(["retrieval_recall", "citation_precision"])
        assert isinstance(scorers[0], RetrievalMetricScorer)
        assert isinstance(scorers[1], CitationScorer)


class TestRetrievalAliases:
    def test_alias_binds_metric_and_ignores_stray_kwarg(self):
        # retrieval_ndcg factory pops a stray `metric` kwarg and binds the alias
        inst = ScorerRegistry.create("retrieval_ndcg", metric="recall", k=5)
        assert isinstance(inst, RetrievalMetricScorer)
        assert inst.metric == "ndcg" and inst.k == 5


class TestFromConfig:
    def test_builds_from_config_dicts(self):
        scorers = ScorerRegistry.from_config(
            [{"name": "retrieval_metric", "metric": "precision", "k": 3}, {"name": "citation_precision"}]
        )
        assert len(scorers) == 2
        assert scorers[0].metric == "precision" and scorers[0].k == 3

    def test_missing_name_skipped(self, caplog):
        with caplog.at_level("WARNING"):
            scorers = ScorerRegistry.from_config([{"metric": "recall"}, {"name": "citation_precision"}])
        assert len(scorers) == 1
        assert any("missing 'name'" in r.message for r in caplog.records)

    def test_bad_kwargs_skipped(self, caplog):
        with caplog.at_level("WARNING"):
            scorers = ScorerRegistry.from_config(
                [{"name": "retrieval_metric", "bogus_kw": 1}, {"name": "bootstrap_ci"}]
            )
        # retrieval_metric with bogus kw fails -> skipped; bootstrap_ci succeeds
        assert len(scorers) == 1 and isinstance(scorers[0], BootstrapCIScorer)
        assert any("Failed to create" in r.message for r in caplog.records)


class TestClear:
    def test_clear_empties_registry(self):
        ScorerRegistry.clear()
        assert ScorerRegistry.list_available() == []
        # restore for the rest of the suite
        register_default_scorers()
