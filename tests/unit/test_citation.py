"""Unit tests for ragi.eval.citation -- ALCE-style citation precision (pure stdlib)."""

from __future__ import annotations

import pytest

from ragi.eval.citation import _rag_sources, citation_precision


class TestRagSources:
    def test_collects_source_fields(self):
        srcs = _rag_sources([{"source": "a.md"}, {"source": "b.md"}, {"content": "no source"}])
        assert srcs == {"a.md", "b.md"}

    def test_empty_or_none(self):
        assert _rag_sources([]) == set()
        assert _rag_sources(None) == set()

    def test_ignores_non_dict_items(self):
        assert _rag_sources(["plain string", 42]) == set()

    def test_falsy_source_skipped(self):
        assert _rag_sources([{"source": ""}, {"source": None}]) == set()


class TestCitationPrecision:
    def test_no_markers_is_vacuous_pass(self):
        assert citation_precision("an answer with no cites", [{"source": "x"}]) == 1.0

    def test_positional_marker_resolves(self):
        retrieved = [{"source": "a"}, {"source": "b"}]
        assert citation_precision("see [1] and [2]", retrieved, k=10) == 1.0

    def test_positional_marker_out_of_range_unresolved(self):
        retrieved = [{"source": "a"}]  # only 1 chunk
        assert citation_precision("see [1] [2] [5]", retrieved, k=10) == pytest.approx(1 / 3)

    def test_named_source_resolves(self):
        retrieved = [{"source": "policy.md"}]
        assert citation_precision("per [Source: policy.md]", retrieved) == 1.0

    def test_named_source_unresolved(self):
        retrieved = [{"source": "other.md"}]
        assert citation_precision("per [Source: missing.md]", retrieved) == 0.0

    def test_mixed_markers(self):
        retrieved = [{"source": "a.md"}, {"source": "b.md"}]
        # [1] ok, [Source: a.md] ok, [9] bad -> 2/3
        assert citation_precision("[1] [Source: a.md] [9]", retrieved) == pytest.approx(2 / 3)

    def test_k_limits_pool_for_positional(self):
        retrieved = [{"source": str(i)} for i in range(5)]
        # k=2 -> only [1] and [2] resolve; [3] does not -> 2/3
        assert citation_precision("[1] [2] [3]", retrieved, k=2) == pytest.approx(2 / 3)

    def test_k_zero_uses_full_pool(self):
        retrieved = [{"source": "a"}, {"source": "b"}]
        assert citation_precision("[1] [2]", retrieved, k=0) == 1.0

    def test_markers_but_empty_retrieved(self):
        # positional [1] cannot resolve against an empty pool -> 0.0
        assert citation_precision("see [1]", []) == 0.0
