"""ragi/eval/citation -- ALCE-style citation precision (pure stdlib).

Citation grounding for the numbered-citation format a RAG augmentation layer emits
(``[1] [Source: company_policy.md]\\n<chunk content>``). Verifies that every citation
marker in the answer resolves to a chunk that was actually retrieved -- a mock-safe
(no-LLM) proxy for citation correctness.

Two marker kinds are supported:
- positional ``[n]``  -> resolves iff ``1 <= n <= min(k, len(retrieved))``
- named ``[Source: x]`` -> resolves iff ``x`` is among the retrieved chunk sources

A live (LLM) leg that NLI-checks each cited span against its chunk is deferred to a
later tier; this is the deterministic format-vs-correctness gate.
"""

from __future__ import annotations

import re

CITATION_NUM = re.compile(r"\[(\d+)\]")
SOURCE_NAMED = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)


def _rag_sources(retrieved: list) -> set[str]:
    sources: set[str] = set()
    for c in retrieved or []:
        if isinstance(c, dict):
            s = c.get("source")
            if s:
                sources.add(str(s))
    return sources


def citation_precision(answer: str, retrieved: list[dict | str], k: int = 10) -> float:
    """Fraction of citation markers in ``answer`` that resolve to a retrieved chunk.

    Positional ``[n]`` resolves iff ``1 <= n <= min(k, len(retrieved))``. Named
    ``[Source: x]`` resolves iff ``x`` is among the retrieved chunk ``source`` fields.
    Returns ``1.0`` when the answer carries no citation markers (vacuous pass --
    nothing to verify).
    """
    pool = retrieved[:k] if (k and k > 0) else (retrieved or [])
    n_pool = len(pool)
    sources = _rag_sources(pool)
    nums = [int(x) for x in CITATION_NUM.findall(answer)]
    named = [m.strip() for m in SOURCE_NAMED.findall(answer)]
    total = len(nums) + len(named)
    if total == 0:
        return 1.0
    resolved = sum(1 for n in nums if 1 <= n <= n_pool)
    resolved += sum(1 for s in named if s in sources)
    return resolved / total
