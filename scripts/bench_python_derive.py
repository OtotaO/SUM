#!/usr/bin/env python3
"""Python-side companion to Tests/benchmarks/browser_wasm_bench.html.

Priority 2 of docs/NEXT_SESSION_PLAYBOOK.md wants cross-runtime
performance numbers on the same input sizes: N ∈ {10, 100, 1000, 10000}
axiom derivations. The browser harness covers the WASM and JS paths;
this script covers the Python path (which itself dispatches to the Zig
shared library when present, pure-sympy fallback otherwise — see
sum_engine_internal.algorithms.semantic_arithmetic._deterministic_prime_v1).

The reported number is wall-clock, median of R trials, per N. Output is
a JSON block with the same schema as the browser harness's output so
docs/WASM_PERFORMANCE.md can aggregate all four (Python, Node,
Browser-WASM, Browser-JS) blocks side-by-side.

**What this does NOT do.** This does not prove anything about speed; it
measures it. Per docs/PROOF_BOUNDARY.md, every number emitted here is
labelled `measured`. If the Zig shared library is unavailable on the
host, the path falls through to sympy.nextprime and the numbers reflect
that — report the fallback honestly rather than hiding it.

Usage:

    python scripts/bench_python_derive.py
    python scripts/bench_python_derive.py --sizes 10,100,1000 --trials 3
    python scripts/bench_python_derive.py --json > bench.json

The output JSON can be pasted directly into the "Python CLI numbers"
row of docs/WASM_PERFORMANCE.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

# Add repo root so imports work whether invoked as a module or script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sum_engine_internal.algorithms.semantic_arithmetic import (  # noqa: E402
    GodelStateAlgebra,
)

AXIOM_KEY_SEED = "sum-bench-v1"
DEFAULT_SIZES = [10, 100, 1000, 10000]
DEFAULT_TRIALS = 5
WARMUP_ITERATIONS = 3


def axiom_triple(i: int) -> tuple[str, str, str]:
    """Deterministic SVO triple matching the browser harness's keys.

    Browser-side key format (godel.js's axiomKey):
        "{s}||{p}||{o}" with s,p,o lowercased + trimmed
    This helper returns the three components; the caller passes them to
    ``get_or_mint_prime`` which joins with "||" under the same convention.
    """
    return (
        f"{AXIOM_KEY_SEED}_subject_{i}",
        "relates_to",
        f"{AXIOM_KEY_SEED}_object_{i}",
    )


def time_one_trial(algebra: GodelStateAlgebra, n: int) -> tuple[float, int]:
    """One trial: derive N primes, LCM them, return (elapsed_ms, state)."""
    state = 1
    t0 = time.perf_counter()
    for i in range(n):
        s, p, o = axiom_triple(i)
        prime = algebra.get_or_mint_prime(s, p, o)
        # Inline LCM to match the browser harness's accounting (prime
        # derivation + LCM bookkeeping is what the state-integer hot
        # path actually does).
        from math import gcd
        state = state * prime // gcd(state, prime) if state else prime
    t1 = time.perf_counter()
    return ((t1 - t0) * 1000.0, state)


def run(sizes: list[int], trials: int) -> dict:
    """Run the benchmark across all N and return a JSON-serialisable dict."""
    algebra = GodelStateAlgebra()

    # Warm-up separate from measurement — primes the Zig shared lib
    # load, any caches, import paths.
    for _ in range(WARMUP_ITERATIONS):
        algebra.get_or_mint_prime(*axiom_triple(0))

    results: list[dict] = []
    for n in sizes:
        ms_list: list[float] = []
        state = None
        for _ in range(trials):
            ms, s = time_one_trial(algebra, n)
            ms_list.append(ms)
            state = s
        results.append(
            {
                "N": n,
                "trials": trials,
                "ms": {
                    "median": round(statistics.median(ms_list), 4),
                    "min": round(min(ms_list), 4),
                    "max": round(max(ms_list), 4),
                    "all": [round(x, 4) for x in ms_list],
                },
                "per_op_us": round(
                    statistics.median(ms_list) * 1000.0 / n, 3
                ),
                "state_integer_first8_hex": f"{state:016x}"[:16]
                if state is not None
                else None,
            }
        )

    # Is the Zig core actually in use? Probe by temporarily monkey-
    # patching and seeing which branch serves. Simpler: just check
    # whether the Zig engine loader resolves.
    try:
        from sum_engine_internal.algorithms.semantic_arithmetic import (
            _get_zig_engine,
        )

        zig_engine = _get_zig_engine()
        zig_path = (
            "zig_core_loaded" if zig_engine is not None else "python_fallback"
        )
    except Exception:  # pragma: no cover — diagnostics only
        zig_path = "unknown"

    return {
        "schema": "sum.wasm_bench.v1",
        "surface": "python_cli",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "derivation_path": zig_path,
        "trials_per_n": trials,
        "axiom_key_seed": AXIOM_KEY_SEED,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default=",".join(str(x) for x in DEFAULT_SIZES),
        help="Comma-separated axiom counts (default: 10,100,1000,10000)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Trials per N (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only the JSON result (for redirection into a file)",
    )
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    payload = run(sizes, args.trials)

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    # Human-readable summary + JSON block (so ad-hoc runs produce both).
    print("SUM Python derivation benchmark")
    print(f"  python: {payload['python_version']}")
    print(f"  platform: {payload['platform']} ({payload['machine']})")
    print(f"  derivation path: {payload['derivation_path']}")
    print(f"  trials per N: {payload['trials_per_n']}")
    print()
    print(
        f"  {'N':>6}  {'median (ms)':>12}  {'min-max (ms)':>18}  {'per-op (µs)':>12}"
    )
    for r in payload["results"]:
        med = r["ms"]["median"]
        lo, hi = r["ms"]["min"], r["ms"]["max"]
        print(
            f"  {r['N']:>6}  {med:>12.3f}  {lo:>8.3f}-{hi:>8.3f}  {r['per_op_us']:>12.2f}"
        )
    print()
    print("Machine-readable (paste into docs/WASM_PERFORMANCE.md):")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
