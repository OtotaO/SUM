from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

SCHEMA_VERSION = "0.3.0"

RegenerationPath = Literal["canonical", "freeform"]
InputKind = Literal["prose", "canonical"]
PerfOperation = Literal["ingest", "encode", "merge", "entail"]

# Polytaxis-aligned honesty label on every metric record.
# "provable"           — mathematically proven (e.g., LCM commutativity).
# "certified"          — verified by external algorithm (α,β-CROWN, SMT, zk-proof).
# "empirical-benchmark"— measured on a corpus; no formal guarantee.
# "expert-opinion"     — human curator judgment.
EpistemicStatus = Literal[
    "provable", "certified", "empirical-benchmark", "expert-opinion"
]


@dataclass(frozen=True)
class ExtractionMetrics:
    corpus_id: str
    precision: float
    recall: float
    f1: float
    n_predicted: int
    n_gold: int
    n_correct: int
    epistemic_status: EpistemicStatus = "empirical-benchmark"


@dataclass(frozen=True)
class PerDocRegeneration:
    """Per-document regeneration attribution.

    Surfaces which specific (s,p,o) claims failed LLM entailment so the
    aggregate FActScore gap can be debugged at the generator-prompt layer
    instead of treated as an opaque corpus-level number.
    """

    doc_id: str
    n_claims: int
    n_supported: int
    per_claim_rate: float
    unsupported_claims: Sequence[tuple[str, str, str]] = field(default_factory=tuple)
    narrative_excerpt: str = ""


@dataclass(frozen=True)
class RegenerationMetrics:
    corpus_id: str
    path: RegenerationPath
    factscore: float
    minicheck_entailment_rate: float
    n_generations: int
    n_supported_claims: int
    n_total_claims: int
    epistemic_status: EpistemicStatus = "empirical-benchmark"
    per_doc: Sequence[PerDocRegeneration] = field(default_factory=tuple)


@dataclass(frozen=True)
class RoundtripMetrics:
    corpus_id: str
    input_kind: InputKind
    axiom_drift_pct: float
    n_roundtrips: int
    source_axioms_avg: float
    reconstructed_axioms_avg: float
    epistemic_status: EpistemicStatus = "empirical-benchmark"


@dataclass(frozen=True)
class PerformanceMetrics:
    operation: PerfOperation
    corpus_size: int
    p50_ms: float
    p99_ms: float
    n_samples: int
    epistemic_status: EpistemicStatus = "empirical-benchmark"


@dataclass(frozen=True)
class BenchReport:
    schema_version: str
    run_id: str
    timestamp_utc: str
    git_sha: str
    host: str
    python_version: str
    model_snapshots: Mapping[str, str]
    extraction: Sequence[ExtractionMetrics] = field(default_factory=tuple)
    regeneration: Sequence[RegenerationMetrics] = field(default_factory=tuple)
    roundtrip: Sequence[RoundtripMetrics] = field(default_factory=tuple)
    performance: Sequence[PerformanceMetrics] = field(default_factory=tuple)
