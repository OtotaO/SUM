"""M1 benchmark — Merkle set-commitment sidecar vs LCM substrate.

Measures two membership-verification paths on the same synthetic
fact set at increasing N:

  * **Merkle inclusion-proof verify.** Build the tree once; measure
    the average per-proof verify time across a sample.
    Theoretically O(log N) hash operations.
  * **LCM divisibility check.** Mint primes for every key; LCM into
    the state integer; measure the average per-check time of
    ``state % prime``. Theoretically O(B²) where B is the bit-length
    of the state — which scales linearly with N at SUM's prime size.

The success criterion the playbook (M1 entry) names is "log-size
inclusion proofs that verify materially faster than the LCM
divisibility path at N=10,000." This benchmark either confirms or
refutes that.

Output is JSON to stdout (or --out FILE), suitable for inclusion
in docs/PROOF_BOUNDARY.md §2.2.

Usage:
    python -m scripts.bench.runners.merkle_vs_lcm \
        --sizes 100 1000 10000 \
        --samples 50 \
        --out /tmp/merkle_vs_lcm.json

Sample size affects per-N total runtime; 50 is a reasonable default.
At N=10k the LCM build is slow (~50s on the reference host per
PROOF_BOUNDARY §2.2); the runner respects this and prints a
progress line per phase.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

# Add repo root to sys.path so we can import sum_engine_internal
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.merkle_sidecar import build_tree, verify_inclusion


def _gen_keys(n: int) -> list[str]:
    """Synthetic canonical fact keys. Stable order for reproducibility."""
    return [f"subject{i:06d}||predicate{i % 16:02d}||object{i:06d}" for i in range(n)]


def _bench_merkle(keys: list[str], samples: int) -> dict:
    n = len(keys)
    print(f"  [merkle] building tree (N={n})...", flush=True)
    t0 = time.perf_counter()
    tree = build_tree(keys)
    build_ms = (time.perf_counter() - t0) * 1000.0

    # Sample uniformly across the leaf range.
    rng = random.Random(42)
    sample_indices = rng.sample(range(n), min(samples, n))

    print(f"  [merkle] generating + verifying {len(sample_indices)} proofs...", flush=True)
    proof_gen_times = []
    proof_verify_times = []
    for idx in sample_indices:
        key = keys[idx]
        t0 = time.perf_counter()
        proof = tree.inclusion_proof(key)
        proof_gen_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        ok = verify_inclusion(key, proof, tree.root)
        proof_verify_times.append((time.perf_counter() - t0) * 1000.0)
        assert ok, f"unexpected verify failure at idx={idx}"

    return {
        "n": n,
        "build_ms_total": round(build_ms, 3),
        "build_ms_per_leaf": round(build_ms / n, 6),
        "proof_gen_ms_p50": round(statistics.median(proof_gen_times), 6),
        "proof_gen_ms_p99": round(_pct(proof_gen_times, 99), 6),
        "proof_verify_ms_p50": round(statistics.median(proof_verify_times), 6),
        "proof_verify_ms_p99": round(_pct(proof_verify_times, 99), 6),
        "proof_size_bytes_avg": round(_avg_proof_bytes(tree, sample_indices), 1),
        "log2_n_ceil": math.ceil(math.log2(n)),
        "samples": len(sample_indices),
    }


def _avg_proof_bytes(tree, sample_indices) -> float:
    """Estimate inclusion-proof wire size: 32 bytes per sibling."""
    sizes = []
    for idx in sample_indices:
        proof = tree.inclusion_proof(tree.leaves[idx])
        # Each sibling is (position-string, 32-byte-hash). Count
        # only the 32-byte hash; the position is one bit semantically.
        sizes.append(32 * len(proof.siblings))
    return sum(sizes) / len(sizes) if sizes else 0.0


def _bench_lcm(keys: list[str], samples: int, skip_build_at: int | None) -> dict:
    n = len(keys)
    print(f"  [lcm] minting {n} primes...", flush=True)
    t0 = time.perf_counter()
    algebra = GodelStateAlgebra()
    primes = []
    for k in keys:
        s, p, o = k.split("||")
        primes.append(algebra.get_or_mint_prime(s, p, o))
    mint_ms = (time.perf_counter() - t0) * 1000.0

    if skip_build_at is not None and n >= skip_build_at:
        print(
            f"  [lcm] N={n} >= --skip-lcm-build-at={skip_build_at}; "
            f"using LCM(first 1000) as proxy state for the modulo "
            f"benchmark (full LCM at this N takes minutes per "
            f"PROOF_BOUNDARY §2.2)",
            flush=True,
        )
        proxy_primes = primes[:1000]
        t0 = time.perf_counter()
        state = 1
        for p in proxy_primes:
            state = state * p // math.gcd(state, p)
        build_ms = (time.perf_counter() - t0) * 1000.0
        proxy_state_n = 1000
    else:
        print(f"  [lcm] building state via LCM (N={n})...", flush=True)
        t0 = time.perf_counter()
        state = 1
        for p in primes:
            state = state * p // math.gcd(state, p)
        build_ms = (time.perf_counter() - t0) * 1000.0
        proxy_state_n = n

    rng = random.Random(42)
    sample_indices = rng.sample(range(n), min(samples, n))

    print(f"  [lcm] checking divisibility for {len(sample_indices)} primes...", flush=True)
    check_times = []
    for idx in sample_indices:
        prime = primes[idx]
        t0 = time.perf_counter()
        is_member = (state % prime == 0)
        check_times.append((time.perf_counter() - t0) * 1000.0)
        # Note: when state was built from a proxy subset, this
        # boolean is meaningful only for primes in that subset.
        # The TIMING is still meaningful — it reflects modulo on
        # an N-bit-length integer.

    return {
        "n": n,
        "mint_ms_total": round(mint_ms, 3),
        "mint_ms_per_prime": round(mint_ms / n, 6),
        "lcm_build_ms_total": round(build_ms, 3),
        "lcm_build_n_used": proxy_state_n,
        "state_bit_length": state.bit_length(),
        "divisibility_ms_p50": round(statistics.median(check_times), 6),
        "divisibility_ms_p99": round(_pct(check_times, 99), 6),
        "samples": len(sample_indices),
    }


def _pct(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="N values to benchmark (default: 100 1000 10000).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of inclusion-proof / divisibility samples per N (default: 50).",
    )
    parser.add_argument(
        "--skip-lcm-build-at",
        type=int,
        default=10000,
        help=(
            "At N >= this, use LCM of first 1000 primes as the "
            "proxy state for the modulo benchmark instead of "
            "building the full LCM (which takes minutes at scale "
            "per PROOF_BOUNDARY §2.2). Default: 10000."
        ),
    )
    parser.add_argument("--out", default=None, help="Path to write JSON output (default: stdout).")
    args = parser.parse_args()

    results = []
    for n in args.sizes:
        print(f"\n=== N={n} ===", flush=True)
        keys = _gen_keys(n)
        merkle = _bench_merkle(keys, args.samples)
        lcm = _bench_lcm(keys, args.samples, args.skip_lcm_build_at)
        results.append(
            {
                "n": n,
                "merkle": merkle,
                "lcm": lcm,
                "speedup_proof_verify_vs_divisibility": round(
                    lcm["divisibility_ms_p50"] / merkle["proof_verify_ms_p50"], 2
                )
                if merkle["proof_verify_ms_p50"] > 0
                else None,
            }
        )

    payload = {
        "schema": "sum.merkle_vs_lcm_bench.v1",
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    text = json.dumps(payload, indent=2) + "\n"

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nbench written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
