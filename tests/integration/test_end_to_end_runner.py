"""Integration test: Mode B end-to-end runner over the synthetic fixture (MockChatClient, no key)."""

from __future__ import annotations

import pytest

import ragi
from ragi.eval.datasets import synthetic_chunks, synthetic_fixture
from ragi.eval.faithfulness import FaithfulnessScorer
from ragi.eval.generation_scorers import AbstentionScorer, AnswerCorrectnessScorer
from ragi.eval.mock_chat import MockChatClient
from ragi.eval.runner import EndToEndEvalRunner
from ragi.retrieval.retriever import BM25Retriever


def _mock_chat() -> MockChatClient:
    # Distinctive substrings route each kind of call to a scripted reply.
    return MockChatClient(
        patterns={
            "Answer the question": "The mitochondrion produces ATP.",  # generation
            "atomic factual claims": '["mitochondrion produces ATP"]',  # faithfulness decompose
            "decide for EACH": '["SUPPORTED"]',  # faithfulness batch-NLI
            "strict grader": "CORRECTNESS: 1.0",  # answer correctness judge
        }
    )


async def test_end_to_end_mode_b_populates_generation_metrics():
    ragi.register_builtins()
    retriever = BM25Retriever(chunks=synthetic_chunks())
    chat = _mock_chat()
    scorers = [FaithfulnessScorer(chat), AnswerCorrectnessScorer(chat), AbstentionScorer()]
    runner = EndToEndEvalRunner(chat, scorers, k=10)
    report = await runner.run(retriever, synthetic_fixture())

    # Generation metrics are populated (flipped from NA).
    assert "faithfulness" in report.aggregate
    assert "answer_correctness" in report.aggregate
    assert "abstention" in report.aggregate
    # Retrieval metrics still computed.
    assert "retrieval_recall" in report.aggregate
    # Rubric carries the generation dims.
    dim_names = [d.name for d in report.rubric.dimensions]
    assert "faithfulness" in dim_names
    assert "answer_correctness" in dim_names
    assert "abstention" in dim_names


async def test_end_to_end_faithfulness_high_when_grounded():
    ragi.register_builtins()
    retriever = BM25Retriever(chunks=synthetic_chunks())
    chat = _mock_chat()
    runner = EndToEndEvalRunner(chat, [FaithfulnessScorer(chat)], k=10)
    report = await runner.run(retriever, synthetic_fixture())
    # Grounded answers -> faithfulness near 1.0.
    assert report.aggregate["faithfulness"].mean >= 0.5


def test_end_to_end_requires_chat_client():
    with pytest.raises(ValueError, match="chat_client"):
        EndToEndEvalRunner(None, [])
