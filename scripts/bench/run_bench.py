"""
SUM benchmark harness orchestrator.

Invocation (from repo root):
    python -m scripts.bench.run_bench \\
        --out bench_report.json \\
        --corpus scripts/bench/corpora/seed_tiny.json \\
        [--no-llm] [--no-perf] [--fail-on-regression]

Model snapshots MUST be pinned (e.g. "gpt-4o-2024-08-06"); unpinned IDs are a
determinism bug and fail the run.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .ci_contract import (
    HISTORY_PATH_RELATIVE,
    append_history,
    detect_regressions,
    load_latest,
)
from .corpus import JsonCorpus
from .runners.extraction import SumExtractionRunner
from .runners.llm_roundtrip import LlmRoundtripRunner
from .runners.performance import SumPerformanceRunner
from .runners.regeneration import OpenAiRegenerationRunner
from .runners.roundtrip import SumRoundtripRunner
from .schema import (
    SCHEMA_VERSION,
    BenchReport,
    ExtractionMetrics,
    LlmRoundtripMetrics,
    PerformanceMetrics,
    RegenerationMetrics,
    RoundtripMetrics,
)


@dataclass(frozen=True)
class CliArgs:
    out_path: Path
    corpus_path: Path
    history_path: Path
    no_llm: bool
    no_perf: bool
    fail_on_regression: bool
    quick: bool


def parse_args(argv: Sequence[str]) -> CliArgs:
    p = argparse.ArgumentParser(
        prog="sum-bench",
        description="SUM measurement harness (extraction / regeneration / roundtrip / perf)",
    )
    p.add_argument("--out", required=True, help="Output path for bench_report.json")
    p.add_argument("--corpus", required=True, help="Path to corpus JSON file")
    p.add_argument(
        "--history",
        default=HISTORY_PATH_RELATIVE,
        help="JSONL history file (append-only)",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-dependent runners (regeneration, roundtrip)",
    )
    p.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip performance runner",
    )
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if regressions are detected vs previous history entry",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced perf load (sizes=100/500/1000, samples=10). "
        "Defaults target long-form nightly runs and may take 10+ minutes.",
    )
    ns = p.parse_args(list(argv))
    return CliArgs(
        out_path=Path(ns.out),
        corpus_path=Path(ns.corpus),
        history_path=Path(ns.history),
        no_llm=ns.no_llm,
        no_perf=ns.no_perf,
        fail_on_regression=ns.fail_on_regression,
        quick=ns.quick,
    )


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


_UNPINNED_SUSPECTS = frozenset(
    {
        "gpt-4o",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "gemini-pro",
        "gemini-1.5-pro",
    }
)


def _looks_unpinned(model_id: str) -> bool:
    return model_id.strip().lower() in _UNPINNED_SUSPECTS


_ROLE_OVERRIDE_ENV = {
    "factscore": "SUM_BENCH_FACTSCORE_MODEL",
    "minicheck": "SUM_BENCH_MINICHECK_MODEL",
    "generator": "SUM_BENCH_GENERATOR_MODEL",
    "extractor": "SUM_BENCH_EXTRACTOR_MODEL",
}


def _resolve_model_snapshots(no_llm: bool) -> dict[str, str]:
    """Collect and validate pinned model IDs from environment.

    Resolution order (per role):
      1. Role-specific override env var (SUM_BENCH_<ROLE>_MODEL) if set.
      2. Global default SUM_BENCH_MODEL if set.
      3. SystemExit.

    The four role-specific vars remain honored for backwards compat —
    runs that used to set all four keep working — but a single
    SUM_BENCH_MODEL is now sufficient for the common case where all
    four roles should use the same pinned snapshot.
    """
    snapshots: dict[str, str] = {
        "sum.sieve": "deterministic_v1",
    }
    if no_llm:
        return snapshots

    global_default = os.environ.get("SUM_BENCH_MODEL", "").strip()

    for role, override_var in _ROLE_OVERRIDE_ENV.items():
        override = os.environ.get(override_var, "").strip()
        val = override or global_default
        source = override_var if override else "SUM_BENCH_MODEL"

        if not val:
            raise SystemExit(
                f"Model {role!r} not pinned. Set SUM_BENCH_MODEL to a "
                f"specific snapshot id (e.g. 'gpt-4o-mini-2024-07-18') "
                f"to use one model for every role, or set the role-specific "
                f"{override_var} to override that role. Pass --no-llm to "
                f"skip LLM runners entirely."
            )
        if _looks_unpinned(val):
            raise SystemExit(
                f"Model {role!r}={val!r} (from {source}) looks unpinned "
                f"(no version/date suffix). Use an explicit snapshot id "
                f"for deterministic benchmarking."
            )
        snapshots[role] = val
    return snapshots


def build_report(args: CliArgs) -> BenchReport:
    corpus = JsonCorpus.load(args.corpus_path)
    model_snapshots = _resolve_model_snapshots(args.no_llm)

    extraction: list[ExtractionMetrics] = []
    regeneration: list[RegenerationMetrics] = []
    roundtrip: list[RoundtripMetrics] = []
    llm_roundtrip: list[LlmRoundtripMetrics] = []
    performance: list[PerformanceMetrics] = []

    extraction.append(SumExtractionRunner().run(corpus))
    roundtrip.extend(SumRoundtripRunner().run(corpus))

    if not args.no_perf:
        perf_runner = (
            SumPerformanceRunner(
                corpus_sizes=(100, 500, 1000),
                samples_per_op=10,
                warmup=3,
            )
            if args.quick
            else SumPerformanceRunner()
        )
        performance.extend(perf_runner.run())

    if not args.no_llm:
        regen_runner = OpenAiRegenerationRunner(
            generator_model_id=model_snapshots["generator"],
            entailment_model_id=model_snapshots["minicheck"],
        )
        regeneration.extend(regen_runner.run(corpus))

        llm_rt_runner = LlmRoundtripRunner(
            generator_model_id=model_snapshots["generator"],
            extractor_model_id=model_snapshots["extractor"],
        )
        llm_roundtrip.extend(llm_rt_runner.run(corpus))

    return BenchReport(
        schema_version=SCHEMA_VERSION,
        run_id=uuid.uuid4().hex,
        timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        git_sha=_git_sha(),
        host=socket.gethostname(),
        python_version=sys.version.split()[0],
        model_snapshots=model_snapshots,
        extraction=tuple(extraction),
        regeneration=tuple(regeneration),
        roundtrip=tuple(roundtrip),
        llm_roundtrip=tuple(llm_roundtrip),
        performance=tuple(performance),
    )


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    report = build_report(args)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(
        json.dumps(dataclasses.asdict(report), indent=2, sort_keys=False),
        encoding="utf-8",
    )

    previous = load_latest(args.history_path)
    findings = detect_regressions(report, previous)
    append_history(report, args.history_path)

    if findings:
        print(f"REGRESSIONS DETECTED ({len(findings)}):", file=sys.stderr)
        for f in findings:
            print(
                f"  {f.metric_path}: {f.previous:.6f} -> {f.current:.6f} "
                f"(threshold={f.threshold})",
                file=sys.stderr,
            )
        if args.fail_on_regression:
            return 2

    print(f"OK: {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
