"""Provenance-path performance characterization — low vs high traffic.

Measures the M1 provenance path in the shape a production consumer
would actually hit it:

    write:  ProvenanceRecord → ledger.record_provenance  (ingest)
    read:   ledger.get_structured_provenance_for_axiom  (lookup)
    cap:    compute_prov_id in tight loop                (crypto ceiling)

The bench is NOT a pytest test — it's a reporting harness. The point
is to produce the numbers a user or a downstream machine-consumer can
reason about. Three orders of magnitude of corpus size (10, 100, 1000,
5000) give a curve rather than a single point.

The SQLite substrate is the real workload here; the sieve and LLM
paths are pre-deduped noise upstream and are measured separately by
``scripts/bench/run_bench.py`` (ingest/encode/merge/entail). This
harness concentrates on what's NEW with M1: structured provenance
writes + reads.

Usage:
    python -m scripts.bench_provenance_path
    python -m scripts.bench_provenance_path --sizes 10,100,1000,5000

Output: human-readable table + a JSON summary at --out if given.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.infrastructure.provenance import (
    ProvenanceRecord,
    compute_prov_id,
)


@dataclass(frozen=True)
class PhaseMetric:
    phase: str
    corpus_size: int
    n_samples: int
    total_ms: float
    mean_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    throughput_per_sec: float


def _make_record(i: int) -> ProvenanceRecord:
    """Synthesize a realistic ProvenanceRecord. Index-driven so each i
    produces a distinct source span, a distinct prov_id, and a
    distinct axiom_key."""
    return ProvenanceRecord(
        source_uri="sha256:" + f"{i:064x}"[:64],
        byte_start=0,
        byte_end=max(1, 16 + (i % 32)),
        extractor_id="sum.bench:synthetic_v1",
        timestamp="2026-04-19T00:00:00+00:00",
        text_excerpt=f"synthetic excerpt #{i}",
    )


def _axiom_key_for(i: int) -> str:
    return f"s{i}||p||o{i}"


def _percentile_us(samples_ns: list[int], p: float) -> float:
    if not samples_ns:
        return 0.0
    ordered = sorted(samples_ns)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * p)))
    return ordered[idx] / 1_000.0


def _summarize(
    phase: str, size: int, samples_ns: list[int], total_ns: int
) -> PhaseMetric:
    n = len(samples_ns)
    if n == 0:
        return PhaseMetric(phase, size, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = statistics.fmean(samples_ns)
    total_s = total_ns / 1e9
    throughput = (n / total_s) if total_s > 0 else 0.0
    return PhaseMetric(
        phase=phase,
        corpus_size=size,
        n_samples=n,
        total_ms=total_ns / 1_000_000.0,
        mean_us=mean / 1_000.0,
        p50_us=_percentile_us(samples_ns, 0.50),
        p95_us=_percentile_us(samples_ns, 0.95),
        p99_us=_percentile_us(samples_ns, 0.99),
        throughput_per_sec=throughput,
    )


def bench_prov_id(size: int) -> PhaseMetric:
    """Pure compute_prov_id ceiling — no I/O, no SQLite. This is the
    cryptographic floor: JCS canonicalization + SHA-256 per record.
    Real writes include this cost plus SQLite INSERT overhead."""
    records = [_make_record(i) for i in range(size)]
    samples_ns: list[int] = []
    t_start = time.monotonic_ns()
    for rec in records:
        t0 = time.monotonic_ns()
        compute_prov_id(rec)
        samples_ns.append(time.monotonic_ns() - t0)
    total_ns = time.monotonic_ns() - t_start
    return _summarize("prov_id.compute", size, samples_ns, total_ns)


def bench_record_provenance(size: int) -> tuple[PhaseMetric, AkashicLedger, str]:
    """End-to-end write: ProvenanceRecord → SQLite. Returns the populated
    ledger so the read bench can reuse it."""
    import asyncio

    fd, path = tempfile.mkstemp(suffix=".db", prefix="prov_bench_")
    os.close(fd)
    ledger = AkashicLedger(db_path=path)

    records = [_make_record(i) for i in range(size)]
    axiom_keys = [_axiom_key_for(i) for i in range(size)]

    async def _write_loop() -> tuple[list[int], int]:
        samples: list[int] = []
        t_total_start = time.monotonic_ns()
        for rec, key in zip(records, axiom_keys):
            t0 = time.monotonic_ns()
            await ledger.record_provenance(rec, key)
            samples.append(time.monotonic_ns() - t0)
        return samples, time.monotonic_ns() - t_total_start

    samples_ns, total_ns = asyncio.run(_write_loop())
    return (
        _summarize("record_provenance", size, samples_ns, total_ns),
        ledger,
        path,
    )


def bench_record_provenance_batch(
    size: int,
) -> tuple[PhaseMetric, AkashicLedger, str]:
    """Batch-ingest path: one BEGIN IMMEDIATE transaction holding N
    INSERTs. Measures amortised per-record latency (total_ms / size)
    rather than per-op latency since the whole batch is one operation
    from the caller's perspective."""
    import asyncio

    fd, path = tempfile.mkstemp(suffix=".db", prefix="prov_bench_batch_")
    os.close(fd)
    ledger = AkashicLedger(db_path=path)

    records = [_make_record(i) for i in range(size)]
    axiom_keys = [_axiom_key_for(i) for i in range(size)]
    pairs = list(zip(records, axiom_keys))

    async def _run() -> int:
        t0 = time.monotonic_ns()
        await ledger.record_provenance_batch(pairs)
        return time.monotonic_ns() - t0

    total_ns = asyncio.run(_run())
    # For the batch path, every record is the "sample" from the caller's
    # perspective. Report amortised per-record cost as the p50.
    per_record_ns = total_ns // max(1, size)
    samples = [per_record_ns] * size
    return (
        _summarize("record_provenance_batch", size, samples, total_ns),
        ledger,
        path,
    )


def bench_get_provenance(
    ledger: AkashicLedger, size: int, n_queries: int
) -> PhaseMetric:
    """Query latency at size. Lookup random axiom_keys from a ledger
    already populated with `size` records; measure n_queries lookups."""
    import asyncio
    import random

    rng = random.Random(size + 42)
    keys = [_axiom_key_for(rng.randrange(size)) for _ in range(n_queries)]

    async def _read_loop() -> tuple[list[int], int]:
        samples: list[int] = []
        t_total_start = time.monotonic_ns()
        for key in keys:
            t0 = time.monotonic_ns()
            await ledger.get_structured_provenance_for_axiom(key)
            samples.append(time.monotonic_ns() - t0)
        return samples, time.monotonic_ns() - t_total_start

    samples_ns, total_ns = asyncio.run(_read_loop())
    return _summarize(
        "get_structured_provenance_for_axiom", size, samples_ns, total_ns
    )


def _print_table(rows: Iterable[PhaseMetric]) -> None:
    header = (
        f"{'phase':<42} {'size':>6} {'N':>6} "
        f"{'total_ms':>10} {'mean_µs':>9} {'p50_µs':>9} "
        f"{'p95_µs':>9} {'p99_µs':>9} {'ops/sec':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.phase:<42} {r.corpus_size:>6d} {r.n_samples:>6d} "
            f"{r.total_ms:>10.1f} {r.mean_us:>9.1f} {r.p50_us:>9.1f} "
            f"{r.p95_us:>9.1f} {r.p99_us:>9.1f} {r.throughput_per_sec:>10.1f}"
        )


def _parse_sizes(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(prog="bench_provenance_path")
    p.add_argument(
        "--sizes", default="10,100,1000,5000",
        help="comma-separated corpus sizes (default: 10,100,1000,5000)",
    )
    p.add_argument(
        "--queries", type=int, default=200,
        help="number of get_structured_provenance_for_axiom lookups per size",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="if given, write JSON summary to this path",
    )
    args = p.parse_args()

    sizes = _parse_sizes(args.sizes)
    all_metrics: list[PhaseMetric] = []
    db_paths: list[str] = []

    for size in sizes:
        all_metrics.append(bench_prov_id(size))
        write_metric, ledger, db_path = bench_record_provenance(size)
        all_metrics.append(write_metric)
        db_paths.append(db_path)
        read_metric = bench_get_provenance(ledger, size, args.queries)
        all_metrics.append(read_metric)
        batch_metric, batch_ledger, batch_path = bench_record_provenance_batch(size)
        all_metrics.append(batch_metric)
        db_paths.append(batch_path)

    _print_table(all_metrics)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([asdict(m) for m in all_metrics], indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

    # Clean up temp DBs
    for path in db_paths:
        try:
            os.unlink(path)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
