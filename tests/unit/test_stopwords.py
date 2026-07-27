"""Unit tests for ragi.retrieval.stopwords -- spec resolution for EN/ID sets."""

from __future__ import annotations

from ragi.retrieval.stopwords import STOPWORDS_EN, STOPWORDS_ID, get_stopwords


class TestGetStopwords:
    def test_none_disables(self):
        assert get_stopwords(None) is None

    def test_false_disables(self):
        assert get_stopwords(False) is None

    def test_true_returns_english(self):
        assert get_stopwords(True) == set(STOPWORDS_EN)

    def test_en_named(self):
        assert get_stopwords("en") == set(STOPWORDS_EN)

    def test_id_named_case_insensitive(self):
        assert get_stopwords("ID") == set(STOPWORDS_ID)
        assert get_stopwords("Id") == set(STOPWORDS_ID)

    def test_unknown_language_warns_and_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert get_stopwords("fr") is None
        assert any("Unknown stopwords language" in r.message for r in caplog.records)

    def test_set_passthrough_lowercased(self):
        assert get_stopwords({"Foo", "BAR"}) == {"foo", "bar"}

    def test_list_passthrough_lowercased(self):
        assert get_stopwords(["Foo", "Bar"]) == {"foo", "bar"}

    def test_returns_a_copy_not_the_shared_set(self):
        s = get_stopwords(True)
        s.add("injected")
        assert "injected" not in STOPWORDS_EN  # caller mutation must not leak
