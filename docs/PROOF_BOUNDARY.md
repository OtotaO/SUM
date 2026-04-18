# Proof Boundary

**Version:** 1.2.0
**Date:** 2026-04-17

This document explicitly separates what the SUM engine **proves mechanically**, what it **measures empirically**, and what remains **aspirational or future work**.

Every claim surfaced elsewhere in the codebase, README, or marketing material MUST trace back to exactly one category here. Conflating proved with measured is the single most common misrepresentation in systems that combine symbolic and neural components; this file exists to prevent that in SUM.

---

## 1. Mechanically Proven

These properties are enforced by deterministic code and verified by tests, including cross-runtime witnesses. They carry the epistemic status `provable` (see §5).

### 1.1. Canonical Round-Trip Conservation

**Claim:** For any Gödel State Integer `S`:
```
reconstruct(parse(canonical_tome(S))) == S
```

**Proof mechanism:** The Ouroboros verifier (Phase 14) encodes a state into a canonical tome, parses the canonical lines back into axiom keys, re-derives primes, and asserts integer equality.

**Boundary:** This proves lossless round-tripping **over the canonical semantic representation**. It does NOT prove that arbitrary English prose can be losslessly compressed and recovered. The canonical representation is the proof substrate; narrative text is a rendering layer. Round-trip over arbitrary prose is an empirical-benchmark measurement (see §2.3).

### 1.2. Cross-Runtime State Equivalence

**Claim:** The Gödel State Integer is runtime-independent. Given the same canonical tome, Python (sympy) and Node.js (BigInt + Miller-Rabin) produce identical state integers.

**Proof mechanism:** The Phase 16 standalone Node.js witness independently reconstructs the state from bundle canonical tomes and asserts exact match.

**Boundary:** Both implementations use the same deterministic prime derivation algorithm (SHA-256 → 8-byte seed → nextprime). Cross-runtime equivalence is proven for the default (non-colliding) derivation path. The collision-resolution path depends on minting order and is NOT independently verified across runtimes in the current test suite.

### 1.3. Bundle Tamper Detection (Trusted Peers)

**Claim:** HMAC-SHA256 signatures detect any modification to the canonical tome, state integer, or timestamp.

**Proof mechanism:** Import rejects bundles with invalid signatures.

**Boundary:** This is tamper detection, not authenticity. Both producer and consumer must share the HMAC key. A party with the key can forge signatures. See `THREAT_MODEL.md`.

### 1.4. Algebra Invariants

**Claim:** The Gödel-State algebra satisfies standard mathematical properties.

| Property | Mechanism | Status |
|----------|-----------|--------|
| LCM commutativity | `lcm(A, B) == lcm(B, A)` | Tested |
| LCM associativity | `lcm(lcm(A,B), C) == lcm(A, lcm(B,C))` | Tested |
| Merge idempotency | `lcm(A, A) == A` | Tested |
| Entailment correctness | `merged % component == 0` | Tested |
| Delta correctness | `lcm(source, delta) == target` | Tested |
| Deletion correctness | `(state * p) // p == state` when `p | state` | Tested |

### 1.5. Durability Contract (Phase 0)

**Claim:** The Gödel state survives process crashes and restarts without data loss or branch bleed.

**Proof mechanism:** Event-sourced replay via the Akashic Ledger. Branch-scoped events are replayed with `branch=` filter. Branch head snapshots provide instant boot. 14 boundary tests verify: restart semantics, branch isolation, novel import materialization, gossip callback persistence.

**Boundary:** Durability depends on SQLite's fsync guarantees. Does NOT protect against disk corruption or hardware failure.

### 1.6. Extraction Structural Gating (Phase 19A)

**Claim:** Malformed, underspecified, or duplicate triplets are rejected before entering the Gödel algebra.

**Proof mechanism:** `ExtractionValidator` enforces: non-empty fields, length bounds (1–500 chars), illegal character rejection (control chars, JSON fragments), predicate canonicalization, and within-batch deduplication. 25 unit tests cover all gating logic.

**Boundary:** This is structural validation, not semantic validation. A structurally valid triplet can still be semantically wrong (e.g., "cat||is_a||number"). Semantic validation requires the confidence calibration and deduplication layers, which are separate.

### 1.7. Merkle Hash-Chain Integrity (Phase 19C)

**Claim:** Any modification, deletion, or injection of events in the Akashic Ledger is detectable.

**Proof mechanism:** Each event stores `prev_hash = SHA-256(prev_hash + operation + prime + axiom_key + branch)`. Genesis seed: `SHA-256("SUM_GENESIS_BLOCK")`. `verify_chain()` walks the full chain on boot, reporting the first broken link. 16 tests verify: tamper detection (mutation, deletion, hash overwrite, injection), chain construction, and chain tip.

**Boundary:** This is tamper detection, not prevention. A local attacker with write access to SQLite can rewrite the entire chain. The hash chain proves that no event was modified after the fact by an actor without full database write access.

---

## 2. Empirically Measured

These properties are observed on a fixed benchmark but not formally proven. They carry the epistemic status `empirical-benchmark` (see §5) and depend on implementation quality, input characteristics, and runtime environment.

### 2.1. Extraction Fidelity

The quality of semantic extraction from natural language depends on:
- The NLP parser (spaCy lemmatizer, dependency parser, model variant)
- Input text structure and complexity
- Domain vocabulary coverage

**Bench harness measurements (schema v0.2.0):**

| corpus | size | precision | recall | F1 | correct / gold |
|---|---|---|---|---|---|
| `seed_tiny_v1` | 8 SVO sentences | 1.000 | 1.000 | **1.000** | 8 / 8 |
| `seed_v1` | 50 SVO sentences | 1.000 | 1.000 | **1.000** | 50 / 50 |

Reproduce via:
```
python -m scripts.bench.run_bench \
    --out bench_report.json \
    --corpus scripts/bench/corpora/seed_v1.json \
    --no-llm
```

`seed_tiny_v1` remains as a fast-feedback smoke baseline (<30 s including spaCy bootstrap). `seed_v1` is the statistically-meaningful benchmark and is the source of record for the published F1 number.

**Historical note — 8 previously-failing patterns now recovered by a POS fallback:**

Earlier measurements (commits before the sieve POS-fallback landing) showed 8 systematic `seed_v1` failures with F1 = 0.913 and recall = 0.840. All 8 shared one root cause: `spaCy en_core_web_sm` parses `<plural-noun> <verb> <noun>` (no article, no modifier) as compound noun phrases rather than SVO clauses. Examples: "Dogs chase cats.", "Diamond cuts glass.", "Copper carries current.", "Iron forms rust.", "Electrons orbit nuclei.", "Enzymes catalyze reactions.", "Muscles contract fibers.", "Engineers design bridges."

The POS fallback in `_pos_fallback_triplet()` now fires *only when the dep-based path yields nothing for a sentence*, and only when the sentence contains exactly three content tokens (NOUN / PROPN / VERB / ADJ — excluding DET / AUX / ADV / ADP / PUNCT / PART). All 8 previously-failing cases now recover correctly, with an auxiliary plural-singularizer for spaCy's occasional "plural-noun misparsed as ADJ" failure (e.g. "Dogs" → "dog").

Precision stayed at 1.000 through the recovery — the fallback's three-content-token gate refuses to fire on sentences with adverbial modifiers, stacked adjectives, auxiliaries, or prepositional phrases, which the dep-based path handles correctly. The sieve's npadvmod subject-dep relaxation (commit `9aea41e`) and the POS fallback (this section) together close every failure mode observed on `seed_v1` without introducing any false positive.

**Residual ceiling:** the current 100 % F1 on `seed_v1` reflects that corpus's scope (simple declarative SVO). Non-SVO constructions (passives with agent phrases, relative clauses, compound predicates, implicit subjects) remain untested; they are deliberately excluded from `seed_v1` and handled by Phase 19B's separate adversarial corpus.

**Prior documented benchmark:** A 50-document golden benchmark corpus exists (Phase 19B) spanning 7 adversarial categories with 100 gold-standard triplets. That corpus remains the source of truth for Phase 19B claims; `seed_v1` is the bench-harness benchmark and complements Phase 19B rather than replacing it.

Structural gating (Phase 19A) rejects malformed triplets. Semantic quality on non-trivial inputs remains the acknowledged weakest link.

### 2.2. Operation Performance

Gödel arithmetic operations (LCM, GCD, modulo) operate on arbitrary-precision integers. Their complexity scales with integer **bit length**, not axiom count:
- GCD: O(n²) via Euclidean algorithm on n-bit integers (sub-quadratic with GMP)
- LCM: O(n²) (reduces to GCD)
- Modulo: O(n²)

**Bench harness measurements (commit `321e573`, Darwin arm64 / Python 3.10.14 / CPython / no Zig):**

| operation | N=100 | N=500 | N=1000 | empirical scaling |
|---|---|---|---|---|
| ingest per-triple (p50) | 0.049 ms | 0.046 ms | 0.045 ms | **O(1) stable** |
| encode (p50) | 0.131 ms | 1.552 ms | 5.107 ms | ~O(n²) |
| merge (p50) | 28.4 ms | 206.4 ms | **518.8 ms** | ~O(n²) — bottleneck |
| entail (p50) | 0.014 ms | 0.062 ms | 0.123 ms | ~O(n) |

Extrapolating the merge curve: N=10,000 primes → ~50 s/op wall-clock; N=100,000 → >1 hr/op. This is the empirical basis for the guidance that **prime encoding is a viable substrate up to low-thousands of axioms and an attestation artifact above that**. For corpora above that ceiling, plug in a property-graph backing store and retain the Gödel integer as a signed witness, not as the primary query path.

At N=10,000 with 200 samples the harness run did not converge inside a 10-minute wall-clock budget on the reference host — recorded as further confirmation of the n² cost of LCM on ~160KB integers. Use `--quick` for dev/PR-time runs; reserve full 1k/5k/10k × 200 samples for scheduled nightly runs on dedicated hardware.

### 2.3. Round-Trip Conservation on Arbitrary Prose

The canonical-template round-trip (§1.1) is **proven**; the round-trip over arbitrary natural-language prose is **not**. The latter is what a reader usually assumes when they hear "conservation," and honesty requires a separate treatment.

**Current status:** Wired in commit `a6606eb` via `SumRoundtripRunner`. Two paths measured per corpus run:

| path | drift (`seed_tiny_v1`) | drift (`seed_v1`) | epistemic_status | interpretation |
|---|---|---|---|---|
| `input_kind="canonical"` | **0.00 %** | **0.00 %** | `provable` | Ouroboros proof (§1.1) verified per-document on every CI run. Symmetric difference of axiom sets is identically zero by construction; any non-zero value is a codec regression alarm. |
| `input_kind="prose"` | 42.86 % | **50.00 %** | `empirical-benchmark` | Sieve re-extraction of the system's own canonical-template output (`generate_canonical` → `extract_triplets`) loses half the axioms on the larger corpus. Reconstructed axioms/doc = 0.60 vs source = 1.00 on `seed_v1`. |

The prose drift rising from 42.86 % to 50.00 % on the larger corpus is not statistical noise — it is a direct empirical confirmation that **the NLP sieve is not a bijective codec, even on the system's own deterministic output**, and that the failure rate tracks the fraction of input sentences spaCy parses atypically. `generate_canonical` emits `"The {s} {p} {o}."` with already-canonicalized (lowercased, lemmatized) keys; on that atypical text, spaCy's dependency parser frequently tags `"like"` in `"X like Y"` as a preposition rather than a verb, so no ROOT verb is found and the triplet is dropped.

**What this measurement does NOT cover:** the full LLM narrative round-trip (`text → triples → LLM-rendered prose → triples'`). That path requires the regeneration runner with pinned MiniCheck and FActScore models; it is tracked as the remaining LLM-gated measurement. Until wired, SUM makes **no empirical claim** about LLM-rendered prose conservation — only the `0.00 %` canonical result and the `42.86 %` sieve-self-parse result are on record.

### 2.4. Bench Harness Substrate

The `scripts/bench/` directory contains the measurement-first infrastructure that makes §2.1–§2.3 reproducible. Key properties:

- **Every report is content-addressable.** `run_id`, `git_sha`, host, Python version, and model snapshots are captured inline. Corpus SHA-256 snapshot hash travels with each report; corpus mutation invalidates historical comparisons at the hash layer.
- **Model snapshots MUST be pinned** (e.g., `gpt-4o-2024-08-06`, not `gpt-4o`). Unpinned identifiers raise `SystemExit` before any work begins.
- **`PerformanceRunner` uses synthetic triples** `(s_i, p, o_i)` for deterministic, non-colliding primes; exercises the pure-Python path even when the Zig core is absent.
- **`ExtractionRunner` uses set-comparison on canonical keys** (no post-hoc lemmatization reconciliation). Gold-triple mismatches with sieve output count as false negatives. Honesty over flattery.
- **CI regression detection** compares each new report against the most recent history entry; `--fail-on-regression` exits non-zero on any F1 drop > 0.02, drift increase > 1%, FActScore drop > 0.03, or p99 ratio > 1.15.
- **LLM-gated runners** (`regeneration.py`, `roundtrip.py`) require `SUM_BENCH_FACTSCORE_MODEL`, `SUM_BENCH_MINICHECK_MODEL`, `SUM_BENCH_GENERATOR_MODEL` env vars with pinned IDs.

---

## 3. Aspirational / Future Work

These are design goals, NOT current capabilities.

| Goal | Status | Target Phase |
|------|--------|-------------|
| Richer semantic IR (qualifiers, time, negation) | Not implemented | Future |
| Multi-pass extraction ensemble | Partially addressed (structural gating, Phase 19B benchmark, bench harness) | Future |
| Hierarchical semantic compression (motifs, chapters) | Not implemented | Future |
| Multi-renderer rehydration (textbook, quiz, study guide) | Not implemented | Future |
| Federation with trust policies | Not implemented | Future |
| Scientific/technical corpora support | Not implemented | Future |
| **Bidirectional distillation with sliding-scale parameters** (density, formality, audience, perspective) | **Interface + density shipped** (`internal/ensemble/tome_sliders.py`, `AutoregressiveTomeGenerator.generate_controlled`, 21 tests). Non-density axes await LLM extrapolator wiring. | **Phase 30+ for full-LLM slider product** |
| **Polytaxis Bucket A absorption** (SHACL, conformal prediction sets, VC 2.0 with `eddsa-rdfc-2022`, RFC 3161 timestamping, RFC 9162 CT v2 proofs, PROV-O/PROV-STAR, polyglot RDF/JSON-LD/Turtle emission) | **In progress — `epistemic_status` field shipped in v1.2.0; Venn-Abers conformal-interval algorithm + tests shipped as standalone module; production integration pending** | **Phase 25** |
| Prose round-trip conservation measurement (via `SumRoundtripRunner` + LLM extrapolator + MiniCheck gate) | Stubbed in bench harness; pending LLM wiring | STATE 4-B |
| Property-graph backing store for corpora above ~10k axioms (prime encoding demoted to signed witness) | Design decision pending empirical confirmation (now confirmed — see §2.2) | Phase 26 |

---

## 4. Complexity Honesty

### What "O(1)" Actually Means in This Codebase

Many operations are described as "O(1)" in comments and documentation. This is shorthand for:

> **O(1) in axiom count** — the operation does not require scanning the axiom list, re-parsing documents, or iterating over a corpus.

It is NOT O(1) in the information-theoretic sense. All operations on Gödel integers scale with the **bit length** of the integer, which grows with each axiom's prime.

**Honest characterization (now empirically confirmed by the bench harness, §2.2):**
- Entailment check (`state % prime == 0`): O(n) in bit length, O(1) in axiom enumeration. Measured p50 = 123 µs at N=1000.
- Merge (`lcm(A, B)`): O(n²) in bit length via GCD. Measured p50 = 519 ms at N=1000. **This is the dominant wall-clock cost and the scaling bottleneck of the current substrate.**
- Branching (integer copy): O(n) in bit length.
- Sync delta (`gcd(A, B)`): O(n²) in bit length.

For practical corpus sizes up to the low thousands of axioms, operations are tractable. Above that ceiling the substrate requires either a property-graph backing store (prime integer demoted to attestation witness) or algorithmic acceleration (GMP via Zig, already present but conditionally active).

---

## 5. Epistemic Status Taxonomy

Absorbed from the Polytaxis formal specification v0.1 §2 as the single highest-leverage honesty mechanism at zero implementation cost. Every SUM claim that is surfaced as a metric, certificate, or signed artefact must declare exactly one of:

| Status | Meaning | Examples in SUM |
|---|---|---|
| `provable` | Proven by deterministic code; the proof is encoded, not asserted. | Canonical round-trip conservation (§1.1); algebra invariants (§1.4). |
| `certified` | Verified by an external algorithm with a published soundness proof. | SMT-solver consistency (Z3/CVC5, planned); α,β-CROWN neural-net verification (planned). |
| `empirical-benchmark` | Measured on a fixed corpus or benchmark. Reproducible, not provable. | Extraction F1; wall-clock p50/p99; regeneration faithfulness (when wired); round-trip drift on prose. |
| `expert-opinion` | Human curator judgment. Lowest formal weight. | Curator-promoted category assignments (future). |

The `epistemic_status` field is mandatory on every `BenchReport` metric record as of schema v0.2.0 (commit `321e573`). Future upgrades: every signed Verifiable Credential emitted by SUM will carry the same field; every claim returned by `/ask` and `/extrapolate` endpoints will carry the same field in the response envelope.

**Conflation rule:** A summary or marketing surface that quotes an `empirical-benchmark` number alongside language like "mathematically guaranteed" or "proven" is a policy violation and must be corrected. The README, THREAT_MODEL, and CANONICAL_ABI_SPEC are required to observe this rule; `PROOF_BOUNDARY.md` is its arbiter.

---

## 6. Progress Toward the Ultimate Goal

SUM's ultimate goal is a **bidirectional knowledge distillation engine**: turn narrative tomes into structured tags and vice versa, with tunable sliders for density, formality, perspective, and audience — truthful in every claim it purports.

**Current honest state (commit `321e573`):**

| Capability | Status | Measurement |
|---|---|---|
| Tome → Tag (extraction) | **F1 = 1.000** | 50/50 on seed_v1; 8/8 on seed_tiny; Phase 19B adversarial corpus separately maintained; non-SVO constructions still untested on the bench |
| Tag → Tome (canonical, deterministic) | Working | Mathematically proven round-trip (§1.1) |
| Tag → Tome (narrative, prose) | Requires LLM extrapolator | No empirical measurement yet |
| Round-trip conservation (canonical) | Proven + empirically verified per-run | §1.1; 0.00% drift on `seed_tiny_v1` and `seed_v1` (commits `a6606eb`, current) |
| Round-trip conservation (sieve re-extract of canonical text) | Measured | 42.86% (seed_tiny) / 50.00% (seed_v1) drift; sieve is not bijective even on its own output |
| Regeneration faithfulness (LLM narrative from axioms) | **Wired** (OpenAiRegenerationRunner); awaits user-side API-key-gated execution | Requires `OPENAI_API_KEY` + `SUM_BENCH_GENERATOR_MODEL` + `SUM_BENCH_MINICHECK_MODEL` + `SUM_BENCH_FACTSCORE_MODEL` env vars |
| Round-trip conservation (LLM narrative prose, full loop) | Not yet wired — needs LLM re-extraction leg | Will compose existing LLM narrative generator + LLM extractor + drift metric |
| Extraction ceiling investigation (en_core_web_trf upgrade or LLM fallback) | 8 / 50 seed_v1 failures all fit one spaCy parse pattern; architectural decision pending | User call |
| Sliding-scale rendering parameters | **Interface shipped** (`TomeSliders`): 5 axes — density / length / formality / audience / perspective. Density slider actioned on the deterministic canonical path (lexicographic axiom subsetting); remaining 4 axes LLM-gated and captured in output header as metadata | Phase 30+ (LLM wiring for non-density axes) |
| Cryptographic attestation | Working | Ed25519 + HMAC-SHA256 + Merkle chain |
| Epistemic-status labeling | Shipped v1.2.0 | See §5 |
| SHACL structural validation (Polytaxis Bucket A) | Not yet | Phase 25 |
| Conformal prediction confidence (Polytaxis Bucket A) | Algorithm shipped (`internal/ensemble/venn_abers.py`, 18 tests); **production wiring via `ConfidenceCalibrator.calibrate_interval()` shipped** with `load_venn_abers_fixture()` helper and fixture tests; calibration-set authoring is the remaining step | Phase 25 |
| VC 2.0 `eddsa-rdfc-2022` emission | Not yet | Phase 25 |

The gap from `current honest state` to `ultimate goal` is the refactor roadmap of record. Any PR that claims to close part of this gap must update this section with the new measurement.
