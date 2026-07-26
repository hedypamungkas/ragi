"""Unit tests for the pure IR metric functions (recall/precision/mrr/ndcg/hit)."""

from __future__ import annotations

import pytest

from ragi.eval.metrics import (
    compute_ranking_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestMetrics:
    def test_recall_single_gold_hit(self):
        assert recall_at_k(["hello world", "foo bar"], ["hello"]) == 1.0

    def test_recall_miss(self):
        assert recall_at_k(["foo", "bar"], ["missing"]) == 0.0

    def test_recall_empty_gold_is_vacuously_one(self):
        # No gold needles -> nothing required -> recall = 1.0 (matches upstream semantics).
        assert recall_at_k(["x", "y"], []) == 1.0

    def test_precision_counts_relevant_in_topk(self):
        # 2 relevant in top-4 -> 0.5
        assert precision_at_k(["gold a", "gold b", "x", "y"], ["gold"], k=4) == 0.5

    def test_ndcg_ranks_position_1_above_position_2(self):
        higher = ndcg_at_k(["gold here", "other"], ["gold"])
        lower = ndcg_at_k(["other", "gold here"], ["gold"])
        assert higher > lower

    def test_compute_ranking_dispatch_clamped(self):
        # value clamped to [0,1]
        v = compute_ranking_metric("recall", ["a"], ["a"], 10)
        assert 0.0 <= v <= 1.0

    def test_compute_ranking_alias_hit_rate(self):
        assert compute_ranking_metric("hit", ["gold", "x"], ["gold"], 10) == 1.0
        assert compute_ranking_metric("hit_rate", ["x", "y"], ["gold"], 10) == 0.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            compute_ranking_metric("bogus", ["a"], ["a"])
