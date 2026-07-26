"""ragi/eval/generation_scorers -- answer-correctness + abstention scorers (Mode B).

Two generation-quality scorers that complement :mod:`faithfulness`:

- :class:`AnswerCorrectnessScorer` — is the answer factually correct vs the golden
  ``expected_answer``? Single-call ``ChatClient`` judge when available, with a deterministic
  substring/token-overlap fallback (mirrors koboi's ``_factual_correctness_fallback``) so the
  metric still produces a number when the judge is gated or absent.
- :class:`AbstentionScorer` — for ``out_of_scope`` qrels, rewards the system REFUSING rather
  than hallucinating (refusal-marker check). Non-OOS qrels are NA.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from ragi.eval.scorers import BaseScorer
from ragi.eval.types import EvalCase, EvalScore

if TYPE_CHECKING:
    from ragi.protocols import ChatClient

_logger = logging.getLogger(__name__)

_CORRECTNESS_PROMPT = (
    "You are a strict grader. Does the answer match the reference answer's fact(s)? "
    "Ignore wording; judge factual equivalence. Reply on the first line: "
    "CORRECTNESS: <number 0.0 to 1.0>.\n\nReference:\n{expected}\n\nAnswer:\n{answer}"
)

# Lowercase refusal cues -- a confident OOS answer that lacks ALL of these failed to abstain.
_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "i cannot",
    "i can't",
    "i'm not sure",
    "i am not sure",
    "no information",
    "not enough information",
    "not enough context",
    "unable to answer",
    "cannot answer",
    "no clear answer",
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (text or "").lower()).strip()


def _deterministic_correctness(answer: str, expected: str) -> float:
    """Substring + token-overlap fallback (no LLM). 1.0 if expected ⊆ answer, else overlap ratio."""
    a = _normalize(answer)
    e = _normalize(expected)
    if not e:
        return 1.0
    if e in a:
        return 1.0
    e_tokens = set(e.split())
    a_tokens = set(a.split())
    if not e_tokens:
        return 0.0
    return _clamp01(len(e_tokens & a_tokens) / len(e_tokens))


class AnswerCorrectnessScorer(BaseScorer):
    """Score answer factual correctness vs ``case.expected_answer`` (judge + deterministic fallback)."""

    name = "answer_correctness"

    def __init__(self, chat_client: ChatClient | None = None):
        self._chat = chat_client

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        expected = (case.expected_answer or "").strip()
        answer = (output or "").strip()
        if not expected:
            return EvalScore("answer_correctness", 1.0, "no expected_answer (NA)")
        if not answer:
            return EvalScore("answer_correctness", 0.0, "empty answer")
        if self._chat is not None:
            try:
                resp = await self._chat.complete(
                    [
                        {
                            "role": "user",
                            "content": _CORRECTNESS_PROMPT.format(expected=expected, answer=answer),
                        }
                    ]
                )
                m = re.search(r"([0-9]*\.?[0-9]+)", resp)
                if m:
                    val = _clamp01(float(m.group(1)))
                    return EvalScore("answer_correctness", round(val, 3), f"judge: {val:.2f}")
            except Exception as e:  # noqa: BLE001 -- fall through to deterministic
                _logger.warning("correctness judge failed, using deterministic fallback: %s", e)
        val = _deterministic_correctness(answer, expected)
        return EvalScore("answer_correctness", round(val, 3), f"deterministic fallback: {val:.2f}")


class AbstentionScorer(BaseScorer):
    """Score negative-rejection: for OOS qrels, reward refusing over hallucinating.

    Non-OOS qrels are NA (return 1.0 + "not OOS"). For ``out_of_scope=True``: 1.0 if the
    answer contains a refusal marker, else 0.0 (hallucinated a confident answer to an
    unanswerable question).
    """

    name = "abstention"

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        if not case.out_of_scope:
            return EvalScore("abstention", 1.0, "not OOS (NA)")
        answer = (output or "").lower()
        refused = any(marker in answer for marker in _REFUSAL_MARKERS)
        if refused:
            return EvalScore("abstention", 1.0, "OOS: correctly refused")
        return EvalScore("abstention", 0.0, "OOS: hallucinated (no refusal detected)")
