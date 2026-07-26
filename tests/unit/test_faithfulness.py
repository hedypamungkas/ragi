"""Unit tests for the faithfulness scorer (claim-decomposition + single + fail-soft)."""

from __future__ import annotations

import asyncio

from ragi.eval.faithfulness import FaithfulnessScorer, normalize_verdict
from ragi.eval.mock_chat import MockChatClient
from ragi.eval.types import EvalCase

_CTX = {"rag_results": [{"content": "photosynthesis converts sunlight into chemical energy"}]}


def test_claim_decomp_coverage_half():
    chat = MockChatClient(
        patterns={
            "atomic factual claims": '["sunlight is used", "the moon is made of cheese"]',
            "decide for EACH": '["SUPPORTED", "UNSUPPORTED"]',
        }
    )
    score = asyncio.run(FaithfulnessScorer(chat).score(EvalCase(query="q"), "answer", _CTX))
    assert score.name == "faithfulness"
    assert score.value == 0.5


def test_claim_decomp_coverage_full():
    chat = MockChatClient(
        patterns={
            "atomic factual claims": '["a", "b"]',
            "decide for EACH": '["SUPPORTED", "SUPPORTED"]',
        }
    )
    score = asyncio.run(FaithfulnessScorer(chat).score(EvalCase(query="q"), "ans", _CTX))
    assert score.value == 1.0


def test_verdict_normalization_substring_trap():
    assert normalize_verdict("SUPPORTED") == "SUPPORTED"
    assert normalize_verdict("UNSUPPORTED") == "UNSUPPORTED"  # not caught by SUPPORTED substring
    assert normalize_verdict("CONTRADICTED") == "CONTRADICTED"
    assert normalize_verdict("garbage") == "UNSUPPORTED"  # strict default


def test_no_context_is_vacuously_faithful():
    score = asyncio.run(FaithfulnessScorer(MockChatClient()).score(EvalCase(query="q"), "ans", {"rag_results": []}))
    assert score.value == 1.0


def test_single_call_mode_parses_score():
    chat = MockChatClient(default="FAITHFULNESS: 0.8\nreasoning...")
    score = asyncio.run(FaithfulnessScorer(chat, mode="single").score(EvalCase(query="q"), "ans", _CTX))
    assert score.value == 0.8


def test_fail_soft_on_judge_error():
    class BoomChat:
        async def complete(self, messages):  # noqa: ARG002
            raise RuntimeError("judge down")

    score = asyncio.run(FaithfulnessScorer(BoomChat()).score(EvalCase(query="q"), "ans", _CTX))
    assert score.value == 0.0
    assert "judge failed" in score.reason
