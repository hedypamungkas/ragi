"""ragi/eval/types -- slimmed eval dataclasses (case, score, golden set)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    query: str
    gold_needles: list[str] = field(default_factory=list)  # substrings for content-based recall
    expected_answer: str | None = None
    gold_doc: str | None = None  # doc_id of the gold passage (optional)
    out_of_scope: bool = False  # OOS queries test abstention (Mode B, v0.4)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalScore:
    name: str
    value: float
    reason: str = ""


@dataclass
class GoldenQrel:
    query: str
    gold_doc: str | None = None
    gold_needles: list[str] = field(default_factory=list)
    expected_answer: str | None = None
    out_of_scope: bool = False


@dataclass
class GoldenDataset:
    name: str
    qrels: list[GoldenQrel]
    corpus_dir: str | None = None  # where passage files live (for the retriever corpus)


@dataclass
class MetricSummary:
    mean: float
    ci_low: float
    ci_high: float
    n: int


# --- run / decision result types ------------------------------------------


@dataclass
class CaseResult:
    """Per-query eval outcome: the ranked doc_ids retrieved + the metric scores."""

    query: str
    retrieved_doc_ids: list[str]
    scores: list[EvalScore]


@dataclass
class RubricDimension:
    """One production-readiness dimension: a metric + a CI-lower-bound threshold."""

    name: str
    metric_key: str  # key in EvalReport.aggregate, e.g. "retrieval_recall"
    threshold: float  # aggregate[metric_key].ci_low must be >= threshold to PASS
    weight: float
    description: str = ""


@dataclass
class DimensionResult:
    name: str
    status: str  # "PASS" | "FAIL" | "NA"
    detail: str


@dataclass
class RubricDecision:
    """The 9-dimension (v0.1: retrieval subset) PASS/FAIL verdict."""

    overall_pass: bool
    weighted_score: float  # weighted fraction of PASS among applicable dimensions
    min_n_ok: bool
    dimensions: list[DimensionResult]


@dataclass
class EvalReport:
    """Full output of a StandaloneEvalRunner run."""

    dataset: str
    n: int
    top_k: int
    per_case: list[CaseResult]
    aggregate: dict[str, MetricSummary]  # metric_name -> mean + bootstrap CI
    rubric: RubricDecision
    config_hash: str  # reproducibility fingerprint of the measured stack


@dataclass
class ComparisonReport:
    """A/B comparison of two stacks on one metric (paired bootstrap CI on the delta)."""

    metric: str
    k: int
    n: int
    mean_a: float
    mean_b: float
    delta: float  # mean_b - mean_a
    ci_low: float  # bootstrap CI on the paired delta
    ci_high: float
    significant: bool  # CI excludes 0
