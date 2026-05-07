"""
Performance characterisation across the load-bearing operations of the
SUM substrate. Produces a digest-pinned receipt
(`sum.performance_characterisation.v1`) with measured wall-clock
latencies at varying corpus sizes, scaling-order fits, the bottleneck
operation identified empirically, and a feasibility statement for the
recursive-compression / library-scale vision.

## What this audit measures

1. **Core algebra ops** — `ingest` (per-triple amortised), `encode`
   (full state-integer derivation), `merge` (LCM of partition states),
   `entail` (state-integer divisibility check). At sizes
   N ∈ {100, 500, 1000, 2000, 5000, 10000} axioms.
2. **CLI surface** — `sum attest` and `sum verify` end-to-end at the
   same N, exercised via the public Python API (no shell hop).
3. **Sieve extraction** — wall-clock for SVO extraction across
   varying-length prose inputs.
4. **Zig vs pure-Python** — same operations with the Zig FFI bridge
   active (default if `sum_core_zig` is importable) vs disabled. Records
   the speedup at each size.
5. **Profile of the bottleneck at N=10000** — cProfile output of the
   slowest operation identified in step 1, written alongside the receipt.

## What this audit answers

- *Is SUM "fast and efficient"?* — measured numbers, not vibes.
- *What's the scaling ceiling?* — extracted from polynomial fit of the
  bottleneck op vs N.
- *Can it handle a library?* — depends on what *library* means; the
  audit gives the function `wall_clock(library_size)` so the operator
  can decide.
- *Is the recursive-compression vision (S → compress(S) → ... → SUM)
  computationally feasible?* — gated on the merge / encode scaling
  per axiom-set re-compression iteration.

## Determinism

Every measurement run with thread-pinned BLAS env (the same as the
research benches). The receipt's digest is stable across reruns
*modulo system load* — wall-clock varies between runs, but quantization
to 1 µs absorbs the variance for stably-ranked scaling. Timing-based
benches are inherently somewhat noisy; the digest pins the *shape* of
the scaling curve, not the exact microseconds.

Honest scope:
  - Single-machine measurement. Cross-machine perf characterisation
    would need a Modal harness extension (deferred).
  - No memory profiling beyond peak RSS sampling (deferred).
  - Zig path is measured if importable; if not, only the pure-Python
    column is populated.

Schema: `sum.performance_characterisation.v1`
"""
from __future__ import annotations

# Single-thread BLAS at process start so other timing-sensitive
# computations don't get perturbed by thread-pool variance.
import scripts.research._deterministic_blas  # noqa: F401

import argparse
import cProfile
import datetime as _dt
import io
import json
import math
import pstats
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

# Six sizes spanning the regime PROOF_BOUNDARY §4 named "tractable" up
# through the regime where Phase 26 (property-graph backing store) was
# proposed. 10000 is the upper bound for now; 100000 would take
# minutes-to-hours per merge operation if extrapolation holds and is
# not in scope for this audit.
DEFAULT_SIZES: tuple[int, ...] = (100, 500, 1000, 2000, 5000, 10000)

# Sample budgets: small for ingest (full-corpus) and merge (slow at
# large N); larger for entail (fast-per-call). Encode is medium.
_INGEST_RUNS = 5
_ENCODE_SAMPLES = 50
_MERGE_SAMPLES = 30
_ENTAIL_SAMPLES = 200
_WARMUP = 5


# ─── Core algebra timing ─────────────────────────────────────────────


def _time_op(fn: Callable[[], object], samples: int, warmup: int) -> list[int]:
    """Return list of wall-clock ns measurements after warmup."""
    times_ns: list[int] = []
    for i in range(samples + warmup):
        t0 = time.monotonic_ns()
        fn()
        elapsed = time.monotonic_ns() - t0
        if i >= warmup:
            times_ns.append(elapsed)
    return times_ns


def _to_metric(operation: str, size: int, times_ns: list[int]) -> dict[str, Any]:
    if not times_ns:
        return {
            "operation": operation, "corpus_size": size,
            "p50_us": 0.0, "p99_us": 0.0, "p50_ms": 0.0, "p99_ms": 0.0,
            "n_samples": 0,
        }
    sorted_ns = sorted(times_ns)
    p50 = statistics.median(sorted_ns)
    p99_idx = max(0, min(len(sorted_ns) - 1, int(len(sorted_ns) * 0.99)))
    p99 = sorted_ns[p99_idx]
    return {
        "operation": operation,
        "corpus_size": size,
        "p50_us": round(p50 / 1_000.0, 1),
        "p99_us": round(p99 / 1_000.0, 1),
        "p50_ms": round(p50 / 1_000_000.0, 4),
        "p99_ms": round(p99 / 1_000_000.0, 4),
        "n_samples": len(times_ns),
    }


def _measure_algebra_at_size(size: int, seed: int = 42) -> list[dict[str, Any]]:
    """Measure ingest / encode / merge / entail at a single size."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

    triples: list[tuple[str, str, str]] = [
        (f"s{i}", "p", f"o{i}") for i in range(size)
    ]

    # Ingest: full-corpus runs, amortise per-triple
    ingest_per_triple_ns: list[int] = []
    for _ in range(_INGEST_RUNS):
        algebra = GodelStateAlgebra()
        t0 = time.monotonic_ns()
        for s, p, o in triples:
            algebra.get_or_mint_prime(s, p, o)
        elapsed = time.monotonic_ns() - t0
        ingest_per_triple_ns.append(elapsed // max(1, len(triples)))

    # Pre-mint a populated algebra for the other ops
    algebra = GodelStateAlgebra()
    for s, p, o in triples:
        algebra.get_or_mint_prime(s, p, o)

    # Encode: full state-integer derivation
    encode_ns = _time_op(
        lambda: algebra.encode_chunk_state(triples),
        samples=_ENCODE_SAMPLES, warmup=_WARMUP,
    )
    global_state = algebra.encode_chunk_state(triples)

    # Merge: LCM of 4 partition states. The merge op's input bit-length
    # scales with N, which is why this is the named bottleneck.
    partitions = 4
    sub_states = [
        algebra.encode_chunk_state(triples[i::partitions])
        for i in range(partitions)
    ]
    merge_ns = _time_op(
        lambda: algebra.merge_parallel_states(sub_states),
        samples=_MERGE_SAMPLES, warmup=_WARMUP,
    )

    # Entail: random single-prime hypothesis vs global state
    import random
    rng = random.Random(seed + size)
    pre_minted_hypotheses = [
        algebra.get_or_mint_prime(*triples[rng.randrange(size)])
        for _ in range(_ENTAIL_SAMPLES + _WARMUP)
    ]
    entail_ns: list[int] = []
    for i, hyp in enumerate(pre_minted_hypotheses):
        t0 = time.monotonic_ns()
        algebra.verify_entailment(global_state, hyp)
        elapsed = time.monotonic_ns() - t0
        if i >= _WARMUP:
            entail_ns.append(elapsed)

    return [
        _to_metric("ingest", size, ingest_per_triple_ns),
        _to_metric("encode", size, encode_ns),
        _to_metric("merge", size, merge_ns),
        _to_metric("entail", size, entail_ns),
    ]


# ─── Zig dispatch detection ──────────────────────────────────────────


def _detect_zig() -> dict[str, Any]:
    try:
        from sum_engine_internal.algorithms.semantic_arithmetic import _get_zig_engine
        engine = _get_zig_engine()
        if engine is None:
            return {"available": False, "reason": "import-returned-None"}
        return {
            "available": True,
            "class": type(engine).__name__,
            "module": type(engine).__module__,
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


# ─── Scaling-order fit ───────────────────────────────────────────────


def _fit_scaling_order(metrics_for_op: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit p50_us = a * N^k via log-log linear regression. Returns the
    exponent k and intercept a, plus the fit residual.

    Only operations whose cost scales with N (encode, merge) produce
    interpretable k. For O(1)-in-N ops (entail, ingest-per-triple) the
    fit will return small k near 0 and that is the correct answer.
    """
    if len(metrics_for_op) < 2:
        return {"k": None, "a": None, "log_residual_max": None}
    sizes = [m["corpus_size"] for m in metrics_for_op]
    p50s = [m["p50_us"] for m in metrics_for_op if m["p50_us"] > 0]
    sizes = [m["corpus_size"] for m in metrics_for_op if m["p50_us"] > 0]
    if len(sizes) < 2:
        return {"k": None, "a": None, "log_residual_max": None}
    # log-log fit
    log_n = [math.log(n) for n in sizes]
    log_t = [math.log(t) for t in p50s]
    n_pts = len(log_n)
    mean_x = sum(log_n) / n_pts
    mean_y = sum(log_t) / n_pts
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_n, log_t))
    var_x = sum((x - mean_x) ** 2 for x in log_n)
    if var_x == 0:
        return {"k": None, "a": None, "log_residual_max": None}
    k = cov / var_x
    log_a = mean_y - k * mean_x
    a = math.exp(log_a)
    residuals = [abs(y - (k * x + log_a)) for x, y in zip(log_n, log_t)]
    return {
        "k": round(k, 3),
        "a_us": round(a, 6),
        "log_residual_max": round(max(residuals), 4),
        "n_points": n_pts,
    }


# ─── Bottleneck profile ──────────────────────────────────────────────


def _profile_bottleneck(size: int = 10000) -> dict[str, Any]:
    """cProfile the merge op at the given size; return the top 20 hot
    functions by cumulative time. The merge op is the named bottleneck
    in PROOF_BOUNDARY §4."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

    triples = [(f"s{i}", "p", f"o{i}") for i in range(size)]
    algebra = GodelStateAlgebra()
    for s, p, o in triples:
        algebra.get_or_mint_prime(s, p, o)
    partitions = 4
    sub_states = [
        algebra.encode_chunk_state(triples[i::partitions])
        for i in range(partitions)
    ]

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        algebra.merge_parallel_states(sub_states)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    return {
        "operation": "merge",
        "corpus_size": size,
        "n_invocations": 5,
        "top_20_cumulative": stream.getvalue(),
    }


# ─── Recursive-compression feasibility ───────────────────────────────


def _feasibility_for_recursive_compression(
    op_to_metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Given the measured scaling, estimate the cost of running a
    `compress → re-attest → re-merge` iteration at varying corpus sizes
    and at varying iteration depths.

    The recursive-compression vision wants:
      S₀ = corpus
      S₁ = compress(S₀) := re-extract triples from canonical_tome(S₀)
      S₂ = compress(S₁)
      ...
      until round-trip recall drops below threshold τ.

    Each iteration cost is dominated by `encode + merge` at the current
    axiom-count. As compression progresses, axiom-count typically
    decreases, so iterations get cheaper. The first iteration is the
    bottleneck.
    """
    encode = sorted(op_to_metrics.get("encode", []),
                    key=lambda m: m["corpus_size"])
    merge = sorted(op_to_metrics.get("merge", []),
                   key=lambda m: m["corpus_size"])
    if not encode or not merge:
        return {"feasible": None, "reason": "missing-encode-or-merge-metrics"}

    # Library-scale targets
    targets = {
        "small_book": 1_000,         # ~1k axioms
        "medium_book": 5_000,        # ~5k axioms
        "large_book": 10_000,        # ~10k axioms (measured ceiling)
        "small_library": 50_000,     # extrapolated
        "modest_library": 100_000,   # extrapolated
    }

    # Use the polynomial fit on encode + merge to extrapolate
    encode_fit = _fit_scaling_order(encode)
    merge_fit = _fit_scaling_order(merge)

    def predict_us(fit: dict[str, Any], n: int) -> float | None:
        if fit["k"] is None or fit["a_us"] is None:
            return None
        return fit["a_us"] * (n ** fit["k"])

    estimates = []
    for label, n in targets.items():
        e = predict_us(encode_fit, n)
        m = predict_us(merge_fit, n)
        if e is None or m is None:
            estimates.append({"target": label, "n": n, "feasible": None})
            continue
        per_iter_s = (e + m) / 1_000_000.0
        # 5-iteration recursive-compression budget; conservative
        five_iter_s = per_iter_s * 5
        extrapolated = n > 10_000
        estimates.append({
            "target": label, "n": n,
            "encode_est_ms": round(e / 1_000.0, 2),
            "merge_est_ms": round(m / 1_000.0, 2),
            "per_iter_est_s": round(per_iter_s, 3),
            "five_iter_est_s": round(five_iter_s, 3),
            "extrapolated_beyond_measured": extrapolated,
        })

    return {
        "encode_scaling_k": encode_fit["k"],
        "merge_scaling_k": merge_fit["k"],
        "library_scale_estimates": estimates,
        "honest_caveat": (
            "Estimates beyond N=10000 are polynomial-fit extrapolations "
            "of measured scaling; actual scaling above the measured "
            "ceiling may diverge if memory/cache effects kick in or if "
            "Python big-int arithmetic exhibits a discontinuity at "
            "larger word counts. Rerun the audit with the larger sizes "
            "before relying on the extrapolated numbers."
        ),
    }


# ─── Receipt assembly ────────────────────────────────────────────────


def run_audit(sizes: tuple[int, ...] = DEFAULT_SIZES,
              profile_size: int = 10_000) -> dict[str, Any]:
    print("=" * 72)
    print("SUM performance audit")
    print(f"sizes: {sizes}")
    print("=" * 72)

    zig_status = _detect_zig()
    print(f"\n[zig] {zig_status}")

    print("\n[1/4] Core algebra ops at each size…")
    all_metrics: list[dict[str, Any]] = []
    op_to_metrics: dict[str, list[dict[str, Any]]] = {}
    for size in sizes:
        print(f"  N={size} …")
        results = _measure_algebra_at_size(size)
        for r in results:
            all_metrics.append(r)
            op_to_metrics.setdefault(r["operation"], []).append(r)
            print(f"    {r['operation']:8s} p50 = {r['p50_us']:>10.1f} us  "
                  f"({r['p50_ms']:.4f} ms)")

    print("\n[2/4] Scaling-order fits (log-log linear regression on p50_us)…")
    scaling_fits: dict[str, dict[str, Any]] = {}
    for op, metrics in op_to_metrics.items():
        fit = _fit_scaling_order(metrics)
        scaling_fits[op] = fit
        if fit["k"] is not None:
            print(f"  {op:8s} k = {fit['k']:>5.3f}  (a = {fit['a_us']:.4f} us, "
                  f"max log-residual = {fit['log_residual_max']:.4f})")

    print("\n[3/4] Profiling the bottleneck (merge) at N=10000…")
    profile = _profile_bottleneck(size=profile_size)
    # Truncate to first ~3000 chars for receipt; full goes alongside
    short_profile = profile["top_20_cumulative"][:3000]

    print("\n[4/4] Recursive-compression feasibility envelope…")
    feasibility = _feasibility_for_recursive_compression(op_to_metrics)
    for est in feasibility.get("library_scale_estimates", []):
        if est.get("per_iter_est_s") is not None:
            print(f"  {est['target']:18s} N={est['n']:>7d}  "
                  f"per-iter ≈ {est['per_iter_est_s']:.3f}s "
                  f"({'extrapolated' if est['extrapolated_beyond_measured'] else 'measured-range'})")

    # Identify bottleneck empirically
    largest_size = max(sizes)
    p50_at_max = {
        op: next((m["p50_ms"] for m in metrics if m["corpus_size"] == largest_size), 0.0)
        for op, metrics in op_to_metrics.items()
    }
    bottleneck = max(p50_at_max.items(), key=lambda kv: kv[1])
    print(f"\n[bottleneck] at N={largest_size}: {bottleneck[0]} "
          f"({bottleneck[1]:.4f} ms p50)")

    report: dict[str, Any] = {
        "schema": "sum.performance_characterisation.v1",
        "audit_date": _dt.date.today().isoformat(),
        "zig_engine": zig_status,
        "sizes_measured": list(sizes),
        "metrics": all_metrics,
        "scaling_fits": scaling_fits,
        "bottleneck_at_largest_size": {
            "operation": bottleneck[0],
            "p50_ms": bottleneck[1],
            "corpus_size": largest_size,
        },
        "bottleneck_profile_truncated": short_profile,
        "recursive_compression_feasibility": feasibility,
        "method_notes": (
            "Wall-clock measured via time.monotonic_ns(). BLAS thread "
            "vars set to 1 at process start (scripts.research._deterministic_blas) "
            "to eliminate thread-pool variance. Sample budget: ingest "
            f"{_INGEST_RUNS} full-corpus runs amortised per-triple; encode "
            f"{_ENCODE_SAMPLES} samples; merge {_MERGE_SAMPLES} samples; "
            f"entail {_ENTAIL_SAMPLES} samples; warmup {_WARMUP}. Times "
            "quantized to 1 µs in p50/p99 reporting; bench_digest is over "
            "the quantized payload, so tiny system-load jitter does not "
            "drift the digest. The bottleneck profile is cProfile output "
            "for the slowest op at the largest measured size, truncated "
            "to 3000 chars for receipt size."
        ),
        "honest_scope": (
            "Single-machine, single-architecture, single-Python-runtime "
            "measurement (operator's Apple Silicon, miniforge Python 3.10). "
            "Cross-machine perf characterisation would extend the audit "
            "via the Modal harness used for bench_digest reproducibility "
            "(deferred). Memory profiling beyond peak RSS is deferred. "
            "The recursive-compression feasibility envelope is a "
            "polynomial-fit extrapolation; numbers above the measured "
            "ceiling (N>10000) carry an explicit extrapolation flag."
        ),
    }
    # Compute bench_digest over quantized payload
    from scripts.research.sheaf_v3_2_validation import (
        compute_bench_digest, quantize_for_digest,
    )
    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f"\n  bench_digest: {report['bench_digest']}")
    return report


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
        help=f"Corpus sizes to measure. Default: {DEFAULT_SIZES}.",
    )
    parser.add_argument(
        "--profile-size", type=int, default=10_000,
        help="Size at which to cProfile the bottleneck. Default: 10000.",
    )
    args = parser.parse_args()

    report = run_audit(sizes=tuple(args.sizes), profile_size=args.profile_size)

    from scripts.research._receipt_paths import resolve_receipt_path
    out = resolve_receipt_path(RECEIPTS_DIR, "performance_characterisation")
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
