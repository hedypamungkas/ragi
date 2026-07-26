"""Unit tests for the metadata filter (relevance scoping, NOT an ACL boundary)."""

from __future__ import annotations

from ragworkbench.ingest.filters import matches_filter


class TestMatchesFilter:
    def test_none_filter_matches_all(self):
        assert matches_filter({"a": 1}, None) is True

    def test_equality_match_and_miss(self):
        assert matches_filter({"a": 1}, {"a": 1}) is True
        assert matches_filter({"a": 1}, {"a": 2}) is False

    def test_comparison_ops(self):
        assert matches_filter({"year": 2024}, {"year": {"$gte": 2024}}) is True
        assert matches_filter({"year": 2023}, {"year": {"$gte": 2024}}) is False
        assert matches_filter({"year": 2024}, {"year": {"$lt": 2024}}) is False

    def test_in_operator(self):
        assert matches_filter({"source": "policy"}, {"source": {"$in": ["policy", "handbook"]}}) is True
        assert matches_filter({"source": "other"}, {"source": {"$in": ["policy", "handbook"]}}) is False

    def test_missing_field_excludes_on_doubt(self):
        assert matches_filter({}, {"a": 1}) is False

    def test_unknown_operator_strict_no_match(self):
        assert matches_filter({"a": 1}, {"a": {"$weird": 1}}) is False

    def test_incomparable_types_no_match(self):
        assert matches_filter({"a": "str"}, {"a": {"$gte": 5}}) is False
