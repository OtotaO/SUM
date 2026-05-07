# Performance characterisation

This document reports measured wall-clock scaling for the load-bearing
operations of the SUM substrate, identifies the bottleneck empirically,
and estimates the feasibility envelope for the original-vision
recursive-compression workload at varying library sizes. Companion
receipt: [`fixtures/bench_receipts/performance_characterisation_2026-05-07.json`](../fixtures/bench_receipts/performance_characterisation_2026-05-07.json),
schema `sum.performance_characterisation.v1`,
`bench_digest 839f4a7f…3ad74`.

The audit is the first measured answer to the operator's standing
question — *"not sure if it's fast and efficient but more on that
later"* — and the empirical foundation for any future scaling claim.

## Method summary

- Six corpus sizes: N ∈ {100, 500, 1000, 2000, 5000, 10000} axioms.
- Synthetic axioms `(s_i, p, o_i)` give deterministic, non-colliding
  primes and exercise the full state-integer derivation path.
- Single-threaded BLAS at process start (`scripts.research._deterministic_blas`)
  to eliminate thread-pool variance.
- Wall-clock via `time.monotonic_ns()`. Sample budgets per op:
  ingest = 5 full-corpus runs (per-triple amortised); encode = 50
  samples; merge = 30 samples; entail = 200 samples; warmup = 5.
- Zig FFI bridge active throughout (locally available as
  `ZigMathEngine` via `sum_engine_internal.infrastructure.zig_bridge`).
- Single machine: operator's Apple Silicon, miniforge Python 3.10,
  numpy 1.26.4 (OpenBLAS arm64).

## Measured scaling

p50 wall-clock at each size (microseconds):

| Op | N=100 | N=500 | N=1000 | N=2000 | N=5000 | N=10000 |
|---|---:|---:|---:|---:|---:|---:|
| **ingest** (per-triple) | 47.8 | 44.2 | 44.1 | 45.4 | 45.8 | 47.6 |
| **encode** | 350.2 | 6,187 | 23,372 | 91,622 | 562,312 | 2,236,531 |
| **merge** | 24,606 | 178,965 | 476,220 | 1,380,603 | 6,812,148 | **23,899,015** |
| **entail** | 13.3 | 64.2 | 124.3 | 248.7 | 620.8 | 1,212.3 |

Same data, expressed as wall-clock at the largest size (N=10000):

| Op | p50 at N=10000 |
|---|---:|
| ingest (per-triple) | 47.6 µs |
| entail | 1.2 ms |
| encode | 2.24 s |
| **merge** (bottleneck) | **23.9 s** |

## Scaling-order fits

Log-log linear regression of p50 vs N over the six measured points:

| Op | Fitted exponent k | Coefficient a (µs) | Max log-residual |
|---|---:|---:|---:|
| ingest | 0.001 | 45.57 | 0.045 |
| entail | 0.981 | 0.144 | 0.017 |
| merge | 1.497 | 19.05 | 0.269 |
| encode | 1.909 | 0.048 | 0.112 |

Reading:

- **`ingest` is amortised constant per-triple** (k=0.001, intercept
  ≈45 µs). Hash-derive then dict-set; no surprise.
- **`entail` is linear in N** (k=0.981). State % prime is O(bit-length),
  bit-length grows linearly with N. Theoretical and measured agree.
- **`merge` is empirically n^1.5** (k=1.497) over the measured range.
  Theoretical worst-case is O(N²) for naive big-int LCM, so we are
  doing better than the naive bound — likely from the Zig FFI path
  doing GCD via Lehmer / binary algorithms rather than schoolbook.
- **`encode` is empirically near-quadratic** (k=1.909). Each step does
  N LCM-update operations against a state that has grown to bit-length
  O(N); O(N) × O(N) ≈ O(N²) matches.

The merge fit's max log-residual (0.27) is larger than encode's
(0.11) — at the high-N end the curve gets a bit steeper. The
extrapolated estimates above N=10000 should be read with that caveat;
the actual k could climb toward 1.6-1.7 in the 50k-100k range.

## Bottleneck — empirically identified

**`merge` dominates wall-clock at every measured size.** At N=100 it
already takes 24.6ms (vs 350µs for encode and ~50µs for ingest /
entail). At N=10000 it takes 23.9 seconds — one merge call exceeds the
sum of every other operation's cost across an entire bench run.

This matches PROOF_BOUNDARY §4's prior quote ("merge — 519 ms at
N=1000, the dominant wall-clock cost and scaling bottleneck") and
strengthens it: the dominance grows with N. The cProfile output
(in the receipt's `bottleneck_profile_truncated` field) shows the
hot path is concentrated in `math.gcd` calls underneath the LCM
implementation; algorithmic acceleration of GCD is the load-bearing
optimisation.

## Recursive-compression feasibility envelope

The original-vision workload is iterated `compress` until the
round-trip recall crosses a degradation threshold τ. Per-iteration
cost is dominated by `encode + merge` at the current axiom count.
Using the measured fits to extrapolate:

| Target | N | encode (est) | merge (est) | per-iter (est) | 5-iter (est) |
|---|---:|---:|---:|---:|---:|
| small book | 1,000 | 23 ms | 593 ms | **0.62 s** | 3.1 s |
| medium book | 5,000 | 558 ms | 6.56 s | **7.12 s** | 35.6 s |
| large book | 10,000 | 2.10 s | 18.5 s | **20.6 s** | 1.7 min |
| small library | 50,000 | 47.5 s | 203 s | **4.2 min** *(extrap)* | 21 min |
| modest library | 100,000 | 178 s | 571 s | **12.5 min** *(extrap)* | 62 min |

Reading:

- **Up to ~5,000 axioms (medium book): comfortable** for interactive
  recursive-compression. A multi-iteration walk to a fixed point
  finishes in tens of seconds.
- **At ~10,000 axioms (large book): borderline.** A single iteration
  is 20s; a 5-iteration walk to the SUM is ~2 minutes. Tolerable
  for batch use; uncomfortable for interactive.
- **At ~50,000 axioms (small library): Phase 26 territory.** A single
  iteration is over 4 minutes (extrapolated). The recursive-
  compression workload is *not feasible* on the current substrate
  beyond this point — 5 iterations is over 20 minutes.
- **At ~100,000 axioms (modest library): clearly requires Phase 26.**
  Each iteration is 12 minutes; a 5-iteration walk is over an hour.
  The property-graph backing store proposed in `docs/PROOF_BOUNDARY.md`
  §3 (Phase 26) becomes the gating dependency for "hand SUM a
  library."

## Scaling ceiling

The honest summary: the current substrate scales comfortably to the
**low thousands of axioms**, scales acceptably to the **low tens of
thousands** for batch workloads, and requires architectural change
(prime-as-witness with property-graph as primary index) for
**library-scale workloads** above ~50k axioms.

This matches PROOF_BOUNDARY §4's prior framing word-for-word — but
now backed by measured numbers across six sizes rather than a single
N=1000 datum. The merge-as-bottleneck claim is empirically sharpened.

## Candidate accelerations

In rough order of (effort × payoff):

1. **Unconditional Zig core activation for the merge path.** The Zig
   bridge is currently active when importable. Confirming it stays
   active across all deployment paths — and removing the pure-Python
   fallback for production — would eliminate dispatch overhead on
   every LCM call. Effort: < 1 day. Payoff: small (the Zig path is
   already what we measured); mostly a hardening exercise.

2. **GMP via Zig.** Lehmer GCD in GMP is asymptotically faster than
   the Zig core's current implementation. For the bit-lengths we hit
   at N=10000 (~3 million bits, since each prime is ~64 bits and
   they accumulate in the LCM), GMP would shave ~30-50% off merge
   wall-clock. Effort: 1-2 weeks (Zig-side GMP linkage; Python-side
   unchanged). Payoff: meaningful — pushes the comfortable ceiling
   from ~5k to ~15-20k axioms.

3. **Property-graph backing store (Phase 26).** Demote the prime
   integer to a signed witness; primary queries hit a Neo4j-style
   property graph. The state-integer becomes a verification artifact
   the verifier can re-derive on demand, not a live query
   substrate. Effort: 1-2 months (backing-store integration,
   verifier path, migration utilities, cross-runtime hygiene).
   Payoff: lifts the ceiling to library scale (>1M axioms) at the
   cost of architectural complexity. **This is the gating
   dependency** for the original-vision recursive-compression
   workload at modest-library scale.

4. **Parallel merge with hash-partitioning.** The current
   `merge_parallel_states` accepts pre-partitioned states; the
   partitioning is sequential. Parallelising the partition-build
   (multiprocessing, since the GIL blocks Python big-int LCM) would
   give ~Nx speedup at N partitions. Effort: 1 week. Payoff: linear
   in cores available; useful for batch workloads but doesn't
   change the asymptotic scaling.

## Implications for the original-vision recursive-compression track

The operator's vision (corpus → axioms → axioms-of-axioms → ... →
the SUM, the incompressible point) is computationally feasible
**today** for corpora up to ~5k axioms. That covers most book-length
material. It is feasible-but-slow up to ~10k. Above that, Phase 26
becomes the gating dependency.

The good news: the round-trip verifier (which the recursive
compression iteration depends on) is *fast* — entail at N=10000 is
1.2ms, and the verify path adds JCS canonicalisation + Ed25519
verification on the bundle, neither of which scales with N
significantly. So the *measurement* of the SUM (where round-trip
recall drops below τ) is cheap; only the *iteration* itself is
expensive.

Concrete next-step plan, gated on this audit:

1. Implement `compress(axiom_set) → axiom_set` (same operation, applied
   to a tome rendered from axioms — re-extracting). Measure round-trip
   recall at each iteration. Receipt schema:
   `sum.recursive_compression_walk.v1`. Feasible immediately at N
   ≤ 5000.
2. Identify the SUM (incompressible point) for `seed_long_paragraphs`
   and `seed_news_briefs`. Both are ≤ 16 docs, ≤ 100 axioms — well
   within the comfortable regime.
3. Extend to a real book-length corpus (~1k axioms). Still
   comfortable.
4. Library scale (~50k+ axioms): defer to post-Phase 26.

## Honest scope

- Single-machine, single-architecture, single-Python-runtime
  measurement (operator's Apple Silicon, miniforge Python 3.10,
  OpenBLAS arm64). Cross-machine perf characterisation would extend
  the audit via the Modal harness used for `bench_digest`
  reproducibility (deferred).
- Memory profiling beyond peak RSS sampling is deferred.
- Estimates above the measured ceiling (N>10000) are polynomial-fit
  extrapolations; actual scaling at 50k-100k may diverge if memory
  / cache effects kick in or if Python big-int arithmetic exhibits
  a discontinuity at larger word counts. Re-run the audit with
  larger sizes before relying on those numbers in load-bearing
  prose.
- The `bench_digest` is byte-stable across reruns *modulo system
  load* — wall-clock varies between runs but quantization to 1 µs
  absorbs the variance for stably-ranked scaling. Timing-based
  benches are inherently somewhat noisy; the digest pins the *shape*
  of the scaling curve, not exact microseconds.

The receipt is reproducible by anyone via:

```
python -m scripts.bench.performance_audit
```

…on a similar machine. Cross-machine reproduction follows the same
pattern as the §4.8 cross-LAPACK matrix (Modal harness extension is
the natural follow-up).
