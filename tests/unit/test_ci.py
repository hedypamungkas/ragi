"""Unit tests for the bootstrap confidence-interval helper."""

from __future__ import annotations

from ragi.eval.ci import bootstrap_ci


class TestBootstrapCI:
    def test_deterministic_given_seed(self):
        scores = [0.5, 0.7, 0.4, 0.9, 0.6, 0.3, 0.8]
        a = bootstrap_ci(scores, seed=42)
        b = bootstrap_ci(scores, seed=42)
        assert a == b

    def test_n1_is_full_width(self):
        # A single observation carries ~no spread info -> honest conservative CI is [0,1].
        r = bootstrap_ci([0.9])
        assert r.n == 1
        assert r.lower == 0.0
        assert r.upper == 1.0

    def test_bounds_ordering(self):
        r = bootstrap_ci([0.5, 0.6, 0.7, 0.8, 0.4, 0.55, 0.65, 0.75] * 2, seed=1)
        assert r.lower <= r.mean <= r.upper
        assert 0.0 <= r.lower and r.upper <= 1.0

    def test_empty_scores(self):
        r = bootstrap_ci([])
        assert r.n == 0
        assert r.mean == 0.0

    def test_low_variance_tight_ci(self):
        # Near-constant scores -> a tight CI around the mean.
        r = bootstrap_ci([0.81, 0.82, 0.80, 0.81, 0.82, 0.80, 0.81] * 4, seed=7)
        assert (r.upper - r.lower) < 0.05
