"""ragi/eval/metrics -- deterministic IR ranking metrics (pure stdlib).

Pure-stdlib retrieval-quality metrics (recall@k, precision@k, MRR, nDCG@k, hit-rate)
over a ranked list of chunk contents. No LLM, no embeddings, no EvalCase dependency --
the canonical mock-safe CI gate for retrieval ranking quality (the single
highest-leverage RAG production-readiness dimension).

Relevance model (mock-safe): a retrieved chunk is *relevant* if its content contains
any of the ``gold`` needles (case-insensitive substring). This is "gold-coverage"
recall -- for the common single-gold qrel it reduces to standard recall@k in {0,1};
for multi-gold it measures how many distinct gold facts the top-k surfaces.
"""

from __future__ import annotations

import math


def _needles(gold: str | list[str]) -> list[str]:
    if isinstance(gold, str):
        return [gold] if gold else []
    return [g for g in gold if g]


def _relevant(content: str, needles: list[str]) -> bool:
    c = content.lower()
    return any(n.lower() in c for n in needles)


def recall_at_k(retrieved: list[str], gold: str | list[str], k: int = 10) -> float:
    """Fraction of gold needles covered by the top-k retrieved chunks."""
    needles = _needles(gold)
    if not needles:
        return 1.0
    top = retrieved[:k]
    covered = {n for n in needles if any(n.lower() in c.lower() for c in top)}
    return len(covered) / len(needles)


def precision_at_k(retrieved: list[str], gold: str | list[str], k: int = 10) -> float:
    """Fraction of the top-k chunks that are relevant (contain a gold needle)."""
    needles = _needles(gold)
    if not needles or k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant = sum(1 for c in top if _relevant(c, needles))
    return relevant / k


def hit_rate(retrieved: list[str], gold: str | list[str], k: int = 10) -> float:
    """1.0 if any relevant chunk appears in the top-k, else 0.0 (== recall@1 for single gold)."""
    needles = _needles(gold)
    if not needles:
        return 0.0
    return 1.0 if any(_relevant(c, needles) for c in retrieved[:k]) else 0.0


def mrr(retrieved: list[str], gold: str | list[str], k: int = 10) -> float:
    """Mean Reciprocal Rank of the first relevant chunk within the top-k."""
    needles = _needles(gold)
    if not needles:
        return 0.0
    for i, c in enumerate(retrieved[:k]):
        if _relevant(c, needles):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: str | list[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain@k with binary relevance."""
    needles = _needles(gold)
    if not needles:
        return 0.0
    top = retrieved[:k]
    gains = [1.0 if _relevant(c, needles) else 0.0 for c in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_n = min(k, sum(1 for c in retrieved if _relevant(c, needles)))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


_METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit": hit_rate,
    "hit_rate": hit_rate,
    "mrr": mrr,
    "ndcg": ndcg_at_k,
}


def compute_ranking_metric(metric: str, retrieved: list[str], gold: str | list[str], k: int = 10) -> float:
    """Dispatch a named metric over a ranked list of chunk contents.

    ``metric`` is one of recall|precision|hit|mrr|ndcg. ``retrieved`` is the chunk
    contents in RANK ORDER (as stamped on ``rag_results``).
    """
    fn = _METRICS.get(metric.lower())
    if fn is None:
        raise ValueError(f"Unknown ranking metric '{metric}'. Available: {sorted(_METRICS)}")
    return max(0.0, min(1.0, fn(retrieved, gold, k)))
