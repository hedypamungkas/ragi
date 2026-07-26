"""ragi/eval/registry -- named Scorer factory registry.

Allows config-driven eval composition by registering scorer factories by name, then
creating them with keyword arguments from config (or by bare name list via ``build``).
The 3 built-in RAG scorers (retrieval_metric / citation_precision / bootstrap_ci) are
registered as defaults at import time so ``build`` / ``from_config`` work out of the box.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ragi.eval.scorers import (
    BaseScorer,
    BootstrapCIScorer,
    CitationScorer,
    RetrievalMetricScorer,
)

_logger = logging.getLogger(__name__)


class ScorerRegistry:
    """Registry of named scorer factories for config-driven eval composition."""

    _factories: dict[str, Callable[..., BaseScorer]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseScorer]) -> None:
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseScorer:
        if name not in cls._factories:
            raise ValueError(f"Unknown scorer '{name}'. Available: {cls.list_available()}")
        return cls._factories[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._factories.keys())

    @classmethod
    def build(cls, names: list[str]) -> list[BaseScorer]:
        """Build a list of scorers by name, each constructed with its defaults."""
        return [cls.create(n) for n in names]

    @classmethod
    def from_config(cls, scorer_configs: list[dict[str, Any]]) -> list[BaseScorer]:
        """Build scorer list from config dicts.

        Each dict must have a 'name' key. Remaining keys are passed as kwargs:
            [{"name": "retrieval_metric", "metric": "recall", "k": 10}]
        """
        scorers: list[BaseScorer] = []
        for cfg in scorer_configs:
            name = cfg.get("name")
            if not name:
                _logger.warning("Scorer config missing 'name', skipping: %s", cfg)
                continue
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            try:
                scorers.append(cls.create(name, **kwargs))
            except (ValueError, TypeError) as e:
                _logger.warning("Failed to create scorer '%s': %s", name, e)
        return scorers

    @classmethod
    def clear(cls) -> None:
        """Remove all registered factories. Useful for test isolation."""
        cls._factories.clear()


def register_default_scorers() -> None:
    """Register the built-in RAG eval scorers. Idempotent; called once at import time."""
    # RetrievalMetricScorer -- generic + one alias per supported metric.
    ScorerRegistry.register("retrieval_metric", lambda **kw: RetrievalMetricScorer(**kw))

    def _retrieval_factory(metric_name: str):
        def _factory(**kw: Any) -> RetrievalMetricScorer:
            kw.pop("metric", None)  # ignore any stray `metric` kwarg, bind the alias
            return RetrievalMetricScorer(metric=metric_name, **kw)

        return _factory

    for _metric in ("recall", "precision", "hit", "mrr", "ndcg"):
        ScorerRegistry.register(f"retrieval_{_metric}", _retrieval_factory(_metric))

    ScorerRegistry.register("citation_precision", lambda **kw: CitationScorer(**kw))
    ScorerRegistry.register("bootstrap_ci", lambda **kw: BootstrapCIScorer(**kw))


# Auto-register defaults at import so ``build`` / ``from_config`` work without an
# explicit registration step. Call ``ScorerRegistry.clear()`` for test isolation.
register_default_scorers()
