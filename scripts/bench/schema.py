from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

SCHEMA_VERSION = "0.1.0"

RegenerationPath = Literal["canonical", "freeform"]
InputKind = Literal["prose", "canonical"]
PerfOperation = Literal["ingest", "encode", "merge", "entail"]


@dataclass(frozen=True)
class ExtractionMetrics:
    corpus_id: str
    precision: float
    recall: float
    f1: float
    n_predicted: int
    n_gold: int
    n_correct: int


@dataclass(frozen=True)
class RegenerationMetrics:
    corpus_id: str
    path: RegenerationPath
    factscore: float
    minicheck_entailment_rate: float
    n_generations: int
    n_supported_claims: int
    n_total_claims: int


@dataclass(frozen=True)
class RoundtripMetrics:
    corpus_id: str
    input_kind: InputKind
    axiom_drift_pct: float
    n_roundtrips: int
    source_axioms_avg: float
    reconstructed_axioms_avg: float


@dataclass(frozen=True)
class PerformanceMetrics:
    operation: PerfOperation
    corpus_size: int
    p50_ms: float
    p99_ms: float
    n_samples: int


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
