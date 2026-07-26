"""ragworkbench/eval/scorers -- BaseScorer ABC + thin RAG scorer implementations.

Thin scorer adapters that bind the pure metric/citation/CI functions to the per-case
``BaseScorer.score(case, output, context)`` contract. Reads retrieved chunks from
``context['rag_results']`` (each item may be a dict with ``content`` or an object
exposing ``.content``); reads gold needles directly from ``case.gold_needles``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from ragworkbench.eval.ci import bootstrap_ci
from ragworkbench.eval.citation import citation_precision
from ragworkbench.eval.metrics import _needles, compute_ranking_metric
from ragworkbench.eval.types import EvalCase, EvalScore


class BaseScorer(ABC):
    """Abstract scorer: produces an EvalScore for one (case, output, context)."""

    name: str = "base"

    @abstractmethod
    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        """Evaluate one case and return a named score in [0, 1] with a reason."""


def _extract_content(chunk) -> str:
    """Read chunk content from a dict or an object with a ``.content`` attribute."""
    if isinstance(chunk, dict):
        return str(chunk.get("content", ""))
    return str(getattr(chunk, "content", ""))


class RetrievalMetricScorer(BaseScorer):
    """Score one IR ranking metric over ``context['rag_results']``.

    Reads retrieved chunks (rank order) from ``context['rag_results']`` and gold
    needles directly from ``case.gold_needles``. Returns the metric value in [0, 1].
    """

    name = "retrieval_metric"

    def __init__(self, metric: str = "recall", k: int = 10):
        self.metric = metric
        self.k = k

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        score_name = f"retrieval_{self.metric}"
        rag = context.get("rag_results") or []
        retrieved = [_extract_content(c) for c in rag]
        gold = case.gold_needles
        if not gold:
            return EvalScore(score_name, 0.0, "no gold_needles on case")
        if not retrieved:
            return EvalScore(score_name, 0.0, f"no rag_results in context ({len(rag)} chunks)")
        value = compute_ranking_metric(self.metric, retrieved, gold, self.k)
        reason = (
            f"{self.metric}@{self.k}={value:.3f} over {len(retrieved)} chunks, {len(_needles(gold))} gold needle(s)"
        )
        return EvalScore(score_name, round(value, 3), reason)


class CitationScorer(BaseScorer):
    """Citation precision over ``context['rag_results']`` (ALCE-style marker resolution)."""

    name = "citation_precision"

    def __init__(self, k: int = 10):
        self.k = k

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        rag = context.get("rag_results") or []
        value = citation_precision(output, rag, self.k)
        reason = f"citation_precision={value:.3f} over {len(rag)} retrieved chunk(s) (k={self.k})"
        return EvalScore("citation_precision", round(value, 3), reason)


class BootstrapCIScorer(BaseScorer):
    """Aggregate an inner scorer's per-case scores into a bootstrap CI lower bound.

    Two equivalent feeds:
    - ``score(case, output, context)``: reads ``context['samples']`` (a list of
      per-query metric values, e.g. produced by running ``inner`` over a batch) and
      returns the CI lower bound. If no samples are present but ``inner`` is set,
      evaluates ``inner`` on this single case and yields the honest N=1 full-width
      CI (cannot pass a CI-lower-bound gate on one sample -- grow N).
    - ``aggregate(items)``: the intended batch entry point -- runs ``inner.score``
      on each ``(case, output, context)`` and bootstraps the collected values.

    ``inner`` is optional so the scorer can wrap a pre-computed samples list without
    forcing an inner scorer to be constructed.
    """

    name = "bootstrap_ci"

    def __init__(
        self,
        inner: BaseScorer | None = None,
        n_boot: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ):
        self.inner = inner
        self.n_boot = n_boot
        self.ci = ci
        self.seed = seed

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        samples: list[float] = []
        raw = context.get("samples")
        if isinstance(raw, list):
            samples = [float(x) for x in raw]
        if not samples and self.inner is not None:
            inner_score = await self.inner.score(case, output, context)
            samples = [inner_score.value]
        if not samples:
            return EvalScore("bootstrap_ci", 0.0, "no samples in context and no inner scorer")

        return self._summarize(samples)

    async def aggregate(self, items: list[tuple[EvalCase, str, dict]]) -> EvalScore:
        """Run ``inner`` over each item and bootstrap the per-case values into one score."""
        if self.inner is None:
            return EvalScore("bootstrap_ci", 0.0, "no inner scorer configured")
        scores = await asyncio.gather(*(self.inner.score(c, o, ctx) for (c, o, ctx) in items))
        return self._summarize([s.value for s in scores])

    def _summarize(self, samples: list[float]) -> EvalScore:
        result = bootstrap_ci(samples, confidence=self.ci, n_boot=self.n_boot, seed=self.seed)
        pct = int(self.ci * 100)
        note = " (n<2: uninformative full-width CI - grow N)" if result.n < 2 else ""
        reason = (
            f"{pct}% CI=[{result.lower:.3f}, {result.upper:.3f}] "
            f"hw={result.half_width:.3f} n={result.n} mean={result.mean:.3f}{note}"
        )
        return EvalScore("bootstrap_ci", round(result.lower, 3), reason)
