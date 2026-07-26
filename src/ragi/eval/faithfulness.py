"""ragi/eval/faithfulness -- faithfulness scorer (NLI claim-decomposition).

Adapted from koboi's ``GroundingGuardrail`` (``koboi/guardrails/grounding.py:58-77, 207-272``)
to the framework-agnostic ``ChatClient`` Protocol — **not** RAGAS. The rubric doc itself
indicts RAGAS (its multi-generation sampling stalls on OpenAI-compatible gateways) and it
drags in ragas + langchain + datasets. Claim-decomposition needs only
``ChatClient.complete(messages) -> str`` (2 calls: decompose + batch-NLI) and yields a
*coverage ratio* that catches partial hallucination (5 claims, 1 unsupported → 0.80).

Loop: decompose answer into atomic claims → batch-NLI each vs retrieved context →
coverage = supported / claims ∈ [0,1]. Fail-soft: any judge error → 0.0 (never crashes the
eval run). ``mode="single"`` is a cheaper one-call judge fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from ragi.eval.scorers import BaseScorer
from ragi.eval.types import EvalCase, EvalScore

if TYPE_CHECKING:
    from ragi.protocols import ChatClient

_logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = (
    "Break the following answer into atomic factual claims. Output ONLY a JSON "
    'array of short strings, no prose. Example: ["claim one", "claim two"].\n\n'
    "Answer:\n{answer}"
)
_NLI_PROMPT = (
    "You are a strict grounding checker. Given the retrieved context, decide if the "
    "claim is SUPPORTED (directly entailed by the context), CONTRADICTED (the context "
    "says the opposite), or UNSUPPORTED (not in the context). Reply with exactly one "
    "word: SUPPORTED, CONTRADICTED, or UNSUPPORTED.\n\n"
    "Context:\n{context}\n\nClaim:\n{claim}"
)
_BATCH_NLI_PROMPT = (
    "You are a strict grounding checker. Given the retrieved context, decide for EACH "
    "claim whether it is SUPPORTED (directly entailed by the context), CONTRADICTED "
    "(the context says the opposite), or UNSUPPORTED (not in the context). "
    "Respond ONLY as a JSON array of verdicts, one per claim, in order. "
    'Example: ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"].\n\n'
    "Context:\n{context}\n\nClaims:\n{claims}"
)
_SINGLE_PROMPT = (
    "You are a strict faithfulness judge. Given the retrieved context, is the answer "
    "FULLY supported by it (no hallucinated/unsupported claims)? "
    "Reply on the first line: FAITHFULNESS: <number 0.0 to 1.0>.\n\n"
    "Context:\n{context}\n\nAnswer:\n{answer}"
)


def normalize_verdict(v: str) -> str:
    """Normalize a judge verdict, guarding the SUPPORTED/UNSUPPORTED substring trap.

    ``"SUPPORTED"`` is a substring of ``"UNSUPPORTED"`` — check the negatives first.
    Unknown verdicts default to UNSUPPORTED (strict).
    """
    v = (v or "").upper()
    if "UNSUPPORTED" in v:
        return "UNSUPPORTED"
    if "CONTRADICTED" in v:
        return "CONTRADICTED"
    if "SUPPORTED" in v:
        return "SUPPORTED"
    return "UNSUPPORTED"


def _extract_content(item) -> str:
    if isinstance(item, dict):
        return str(item.get("content", ""))
    return str(getattr(item, "content", ""))


def _parse_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:  # noqa: BLE001
        pass
    return []


class FaithfulnessScorer(BaseScorer):
    """Score answer faithfulness ∈ [0,1] via NLI claim-decomposition (default) or a single judge call."""

    name = "faithfulness"

    def __init__(self, chat_client: ChatClient, mode: str = "claim_decomp"):
        self._chat = chat_client
        self.mode = mode

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        answer = (output or "").strip()
        rag = context.get("rag_results") or []
        context_str = "\n\n".join(_extract_content(c) for c in rag).strip()
        if not answer:
            return EvalScore("faithfulness", 1.0, "empty answer (vacuously faithful)")
        if not context_str:
            return EvalScore("faithfulness", 1.0, "no retrieved context to check against")
        try:
            if self.mode == "single":
                return await self._score_single(answer, context_str)
            return await self._score_claim_decomp(answer, context_str)
        except Exception as e:  # noqa: BLE001 -- fail-soft: never crash the eval run
            _logger.warning("faithfulness judge failed: %s", e)
            return EvalScore("faithfulness", 0.0, f"judge failed: {e}")

    async def _score_claim_decomp(self, answer: str, context_str: str) -> EvalScore:
        raw = await self._chat.complete([{"role": "user", "content": _DECOMPOSE_PROMPT.format(answer=answer)}])
        claims = [c for c in _parse_json_list(raw) if c.strip()]
        if not claims:
            return EvalScore("faithfulness", 1.0, "no atomic claims extracted")
        raw_v = await self._chat.complete(
            [
                {
                    "role": "user",
                    "content": _BATCH_NLI_PROMPT.format(context=context_str, claims=json.dumps(claims)),
                }
            ]
        )
        verdicts = _parse_json_list(raw_v)
        if len(verdicts) != len(claims):
            # Batch parse mismatch -> per-claim NLI fallback (koboi's pattern).
            verdicts = await self._per_claim_nli(claims, context_str)
        supported = sum(1 for v in verdicts if normalize_verdict(v) == "SUPPORTED")
        coverage = supported / len(claims)
        return EvalScore("faithfulness", round(coverage, 3), f"{supported}/{len(claims)} claims supported")

    async def _per_claim_nli(self, claims: list[str], context_str: str) -> list[str]:
        out: list[str] = []
        for claim in claims:
            v = await self._chat.complete(
                [{"role": "user", "content": _NLI_PROMPT.format(context=context_str, claim=claim)}]
            )
            out.append(v)
        return out

    async def _score_single(self, answer: str, context_str: str) -> EvalScore:
        resp = await self._chat.complete(
            [{"role": "user", "content": _SINGLE_PROMPT.format(context=context_str, answer=answer)}]
        )
        m = re.search(r"([0-9]*\.?[0-9]+)", resp)
        val = max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
        return EvalScore("faithfulness", round(val, 3), f"single-call judge: {val:.2f}")
