from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .schema import (
    BenchReport,
    ExtractionMetrics,
    LlmRoundtripMetrics,
    PerDocLlmRoundtrip,
    PerDocRegeneration,
    PerformanceMetrics,
    RegenerationMetrics,
    RoundtripMetrics,
)

REGRESSION_THRESHOLD_F1: float = 0.02
REGRESSION_THRESHOLD_DRIFT_PCT: float = 1.0
REGRESSION_THRESHOLD_FACTSCORE: float = 0.03
REGRESSION_THRESHOLD_MINICHECK: float = 0.03
REGRESSION_THRESHOLD_PERF_P99_RATIO: float = 0.15
# LLM narrative round-trip drift is noisier than the sieve-only prose
# re-extract (stochastic generator + stochastic extractor compound), so the
# threshold is intentionally wider than the canonical/prose drift threshold.
REGRESSION_THRESHOLD_LLM_DRIFT_PCT: float = 5.0

HISTORY_PATH_RELATIVE = "bench_history.jsonl"


@dataclass(frozen=True)
class RegressionFinding:
    metric_path: str
    previous: float
    current: float
    threshold: float
    direction: str  # always "regression" — improvements are not findings


def detect_regressions(
    current: BenchReport,
    previous: BenchReport | None,
) -> Sequence[RegressionFinding]:
    """Compare current report against previous green report. Empty = pass."""
    if previous is None:
        return ()

    findings: list[RegressionFinding] = []

    prev_f1 = {m.corpus_id: m.f1 for m in previous.extraction}
    for em in current.extraction:
        prev = prev_f1.get(em.corpus_id)
        if prev is not None and em.f1 < prev - REGRESSION_THRESHOLD_F1:
            findings.append(
                RegressionFinding(
                    metric_path=f"extraction[{em.corpus_id}].f1",
                    previous=prev,
                    current=em.f1,
                    threshold=REGRESSION_THRESHOLD_F1,
                    direction="regression",
                )
            )

    prev_drift = {
        (m.corpus_id, m.input_kind): m.axiom_drift_pct for m in previous.roundtrip
    }
    for rm in current.roundtrip:
        prev = prev_drift.get((rm.corpus_id, rm.input_kind))
        if (
            prev is not None
            and rm.axiom_drift_pct > prev + REGRESSION_THRESHOLD_DRIFT_PCT
        ):
            findings.append(
                RegressionFinding(
                    metric_path=f"roundtrip[{rm.corpus_id}:{rm.input_kind}].drift_pct",
                    previous=prev,
                    current=rm.axiom_drift_pct,
                    threshold=REGRESSION_THRESHOLD_DRIFT_PCT,
                    direction="regression",
                )
            )

    prev_fs = {(m.corpus_id, m.path): m.factscore for m in previous.regeneration}
    for gm in current.regeneration:
        prev = prev_fs.get((gm.corpus_id, gm.path))
        if prev is not None and gm.factscore < prev - REGRESSION_THRESHOLD_FACTSCORE:
            findings.append(
                RegressionFinding(
                    metric_path=f"regeneration[{gm.corpus_id}:{gm.path}].factscore",
                    previous=prev,
                    current=gm.factscore,
                    threshold=REGRESSION_THRESHOLD_FACTSCORE,
                    direction="regression",
                )
            )

    prev_llm_drift = {m.corpus_id: m.drift_pct for m in previous.llm_roundtrip}
    for lrm in current.llm_roundtrip:
        prev = prev_llm_drift.get(lrm.corpus_id)
        if (
            prev is not None
            and lrm.drift_pct > prev + REGRESSION_THRESHOLD_LLM_DRIFT_PCT
        ):
            findings.append(
                RegressionFinding(
                    metric_path=f"llm_roundtrip[{lrm.corpus_id}].drift_pct",
                    previous=prev,
                    current=lrm.drift_pct,
                    threshold=REGRESSION_THRESHOLD_LLM_DRIFT_PCT,
                    direction="regression",
                )
            )

    prev_p99 = {
        (m.operation, m.corpus_size): m.p99_ms for m in previous.performance
    }
    for pm in current.performance:
        prev = prev_p99.get((pm.operation, pm.corpus_size))
        if prev is not None and prev > 0:
            ratio = (pm.p99_ms - prev) / prev
            if ratio > REGRESSION_THRESHOLD_PERF_P99_RATIO:
                findings.append(
                    RegressionFinding(
                        metric_path=f"performance[{pm.operation}:{pm.corpus_size}].p99_ms",
                        previous=prev,
                        current=pm.p99_ms,
                        threshold=REGRESSION_THRESHOLD_PERF_P99_RATIO,
                        direction="regression",
                    )
                )

    return tuple(findings)


def append_history(report: BenchReport, history_path: str | Path) -> None:
    """Append one JSONL record. Never rewrite history."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dataclasses.asdict(report)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_latest(history_path: str | Path) -> BenchReport | None:
    """Return the most recent report in the history file, or None if empty/missing."""
    path = Path(history_path)
    if not path.exists():
        return None
    last_line = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                last_line = line
    if not last_line:
        return None
    return _report_from_dict(json.loads(last_line))


def _regeneration_from_dict(m: dict[str, Any]) -> RegenerationMetrics:
    per_doc = tuple(
        PerDocRegeneration(
            doc_id=p["doc_id"],
            n_claims=p["n_claims"],
            n_supported=p["n_supported"],
            per_claim_rate=p["per_claim_rate"],
            unsupported_claims=tuple(
                tuple(t) for t in p.get("unsupported_claims", ())
            ),
            narrative_excerpt=p.get("narrative_excerpt", ""),
        )
        for p in m.get("per_doc", ())
    )
    fields = {k: v for k, v in m.items() if k != "per_doc"}
    return RegenerationMetrics(**fields, per_doc=per_doc)


def _llm_roundtrip_from_dict(m: dict[str, Any]) -> LlmRoundtripMetrics:
    per_doc = tuple(
        PerDocLlmRoundtrip(
            doc_id=p["doc_id"],
            n_source_axioms=p["n_source_axioms"],
            n_reconstructed_axioms=p["n_reconstructed_axioms"],
            drift_pct=p["drift_pct"],
            missing_claims=tuple(tuple(t) for t in p.get("missing_claims", ())),
            extra_claims=tuple(tuple(t) for t in p.get("extra_claims", ())),
            narrative_excerpt=p.get("narrative_excerpt", ""),
        )
        for p in m.get("per_doc", ())
    )
    fields = {k: v for k, v in m.items() if k != "per_doc"}
    return LlmRoundtripMetrics(**fields, per_doc=per_doc)


def _report_from_dict(d: dict[str, Any]) -> BenchReport:
    return BenchReport(
        schema_version=d["schema_version"],
        run_id=d["run_id"],
        timestamp_utc=d["timestamp_utc"],
        git_sha=d["git_sha"],
        host=d["host"],
        python_version=d["python_version"],
        model_snapshots=dict(d["model_snapshots"]),
        extraction=tuple(ExtractionMetrics(**m) for m in d.get("extraction", [])),
        regeneration=tuple(
            _regeneration_from_dict(m) for m in d.get("regeneration", [])
        ),
        roundtrip=tuple(RoundtripMetrics(**m) for m in d.get("roundtrip", [])),
        llm_roundtrip=tuple(
            _llm_roundtrip_from_dict(m) for m in d.get("llm_roundtrip", [])
        ),
        performance=tuple(PerformanceMetrics(**m) for m in d.get("performance", [])),
    )
