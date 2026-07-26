"""Unit tests for AnswerCorrectness + Abstention scorers."""

from __future__ import annotations

import asyncio

from ragworkbench.eval.generation_scorers import (
    AbstentionScorer,
    AnswerCorrectnessScorer,
)
from ragworkbench.eval.mock_chat import MockChatClient
from ragworkbench.eval.types import EvalCase


class TestAnswerCorrectness:
    def test_judge_score_parsed(self):
        chat = MockChatClient(default="CORRECTNESS: 0.9\nlooks good")
        score = asyncio.run(
            AnswerCorrectnessScorer(chat).score(
                EvalCase(query="q", expected_answer="42"), "the answer is forty-two", {}
            )
        )
        assert score.value == 0.9

    def test_deterministic_fallback_substring_match(self):
        # No chat client -> deterministic; expected substring present -> 1.0
        score = asyncio.run(
            AnswerCorrectnessScorer().score(EvalCase(query="q", expected_answer="Paris"), "the capital is Paris", {})
        )
        assert score.value == 1.0

    def test_deterministic_fallback_partial_overlap(self):
        score = asyncio.run(
            AnswerCorrectnessScorer().score(
                EvalCase(query="q", expected_answer="Eiffel Tower Paris"), "the Eiffel Tower", {}
            )
        )
        assert 0.0 < score.value < 1.0  # partial token overlap

    def test_deterministic_fallback_miss(self):
        score = asyncio.run(AnswerCorrectnessScorer().score(EvalCase(query="q", expected_answer="Paris"), "London", {}))
        assert score.value == 0.0

    def test_no_expected_answer_is_na(self):
        score = asyncio.run(AnswerCorrectnessScorer().score(EvalCase(query="q"), "ans", {}))
        assert score.value == 1.0


class TestAbstention:
    def test_oos_refused_scores_one(self):
        score = asyncio.run(
            AbstentionScorer().score(EvalCase(query="q", out_of_scope=True), "I don't know the answer to that.", {})
        )
        assert score.value == 1.0

    def test_oos_hallucinated_scores_zero(self):
        score = asyncio.run(
            AbstentionScorer().score(
                EvalCase(query="q", out_of_scope=True),
                "The answer is definitely 42 based on the context.",
                {},
            )
        )
        assert score.value == 0.0

    def test_not_oos_is_na(self):
        score = asyncio.run(AbstentionScorer().score(EvalCase(query="q", out_of_scope=False), "some answer", {}))
        assert score.value == 1.0
