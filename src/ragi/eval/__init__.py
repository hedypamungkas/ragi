"""ragi/eval -- RAG evaluation metrics, scorers, and registry (stdlib-only).

The eval layer is the wedge of this toolkit: deterministic, mock-safe IR ranking
metrics (recall@k / precision@k / MRR / nDCG@k / hit-rate), ALCE-style citation
precision, and a seedable bootstrap CI -- all pure stdlib, no LLM, no embeddings,
no koboi dependency. Compose scorers via :class:`ScorerRegistry`.
"""

from __future__ import annotations

from ragi.eval.ci import CIResult, bootstrap_ci
from ragi.eval.citation import citation_precision
from ragi.eval.metrics import (
    compute_ranking_metric,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragi.eval.registry import ScorerRegistry, register_default_scorers
from ragi.eval.scorers import (
    BaseScorer,
    BootstrapCIScorer,
    CitationScorer,
    RetrievalMetricScorer,
)
from ragi.eval.types import (
    EvalCase,
    EvalScore,
    GoldenDataset,
    GoldenQrel,
    MetricSummary,
)

__all__ = [
    # types
    "EvalCase",
    "EvalScore",
    "GoldenQrel",
    "GoldenDataset",
    "MetricSummary",
    # metrics (pure)
    "compute_ranking_metric",
    "recall_at_k",
    "precision_at_k",
    "hit_rate",
    "mrr",
    "ndcg_at_k",
    # citation (pure)
    "citation_precision",
    # ci (pure)
    "bootstrap_ci",
    "CIResult",
    # scorers
    "BaseScorer",
    "RetrievalMetricScorer",
    "CitationScorer",
    "BootstrapCIScorer",
    # registry
    "ScorerRegistry",
    "register_default_scorers",
]
