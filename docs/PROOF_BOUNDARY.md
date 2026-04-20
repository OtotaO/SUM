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

| path | drift (`seed_tiny_v1`) | drift (`seed_v1`) | drift (`seed_v2`) | epistemic_status | interpretation |
|---|---|---|---|---|---|
| `input_kind="canonical"` | **0.00 %** | **0.00 %** | **0.00 %** | `provable` | Ouroboros proof (§1.1) verified per-document on every CI run. Symmetric difference of axiom sets is identically zero by construction; any non-zero value is a codec regression alarm. |
| `input_kind="prose"` | 42.86 % | **54.00 %** | **56.25 %** | `empirical-benchmark` | Sieve re-extraction of the system's own canonical-template output (`generate_canonical` → `extract_triplets`) loses axioms on every corpus; the drift is a direct function of the fraction of sentences spaCy parses atypically or that the sieve lemmatizer normalises away. seed_v2's drift fell from 60.00 % to 56.25 % after the passive-voice fix because agentless-passive suppression trimmed one noisy re-extract case. |

The prose drift rising monotonically from `seed_tiny` → `seed_v1` → `seed_v2` is not statistical noise — it is a direct empirical confirmation that **the NLP sieve is not a bijective codec, even on the system's own deterministic output**. `generate_canonical` emits `"The {s} {p} {o}."` with already-canonicalized (lowercased, lemmatized) keys; on that atypical text, spaCy's dependency parser frequently tags function words atypically (e.g. `"like"` in `"X like Y"` as a preposition rather than a verb), so no ROOT verb is found and the triplet is dropped. `seed_v2`'s harder parse patterns (apposition, relative clause, passive, conjunction) amplify this.

**Sieve canonical-invariant guard:** the canonical template parser is `^The (\S+) (\S+) (.+)\.$` — subject and predicate must be single `\S+` tokens, object is `.+` (greedy, accepts whitespace). The sieve had a latent bug where multi-word subjects (e.g. `"Marie Curie"` → `"marie curie"`) were space-joined, which then bled into the parser's second capture group and silently broke the canonical round-trip: on `seed_v2` that manifested as 11.76 % canonical drift on the one affected document (200 % drift on that doc, averaged across 17). The sieve now `"_"`-joins compound modifiers for subject (predicate is always a spaCy lemma — single token by construction; object's `.+` regex tolerates whitespace), restoring the canonical-round-trip provability claim to universal scope. Test: `Tests/test_sieve_canonical_invariant.py`.

**What this measurement does NOT cover:** the full LLM narrative round-trip (`text → triples → LLM-rendered prose → triples'`). That path has both a wired runner (`scripts/bench/runners/llm_roundtrip.py`, composing `LiveLLMAdapter.extract_triplets → generate_text → extract_triplets`) and, as of 2026-04-19, a first end-to-end measurement on `seed_v1` — see §2.5. The short version: **drift = 107.75 %, exact-match recall = 0.12**, interpreted as "facts preserved, keys not" and driven by generator elaboration + unprompted extractor paraphrase. The `0.00 %` canonical result and the `54.00 %` sieve-self-parse result remain the complementary measurements on the deterministic side.

### 2.4. Regeneration Faithfulness (LLM Narrative → Axiom Entailment)

SUM's `tag → tome` direction measured end-to-end: for each source axiom set, `LiveLLMAdapter.generate_text` produces a prose narrative, and `LlmEntailmentChecker` (structured-output entailment via pinned model snapshot) independently judges whether each source axiom is supported by the narrative. FActScore is the mean per-document entailment rate.

**End-to-end runs (both with temporary API keys, rolled immediately after use):**

| date | corpus | generator | entailment model | n_docs | n_claims | supported | FActScore |
|---|---|---|---|---|---|---|---|
| 2026-04-17 | `seed_v1` | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` | 50 / 50 gen | 50 | 48 | **0.960** |
| 2026-04-19 | `seed_v1` | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` | 50 / 50 gen | 50 | 47 | **0.940** |

Run-to-run delta of 0.02 is below the `fail-on-regression` threshold (0.03) and consistent with OpenAI-side non-determinism at the pinned snapshot — the model ID is pinned, but the OpenAI chat-completion endpoint is not deterministic at `temperature` defaults. **Both numbers are load-bearing and both stay on record.**

**2026-04-19 per-doc attribution** (the three failures the aggregate 0.940 hides):

| doc_id | source triple | LLM narrative excerpt | failure mode |
|---|---|---|---|
| `doc_017` | `steel resist corrosion` | "Steel is … also susceptible to corrosion, a process that can …" | generator flipped polarity — narrative says steel *is susceptible to* corrosion instead of *resists* it; entailment checker correctly rejected. |
| `doc_018` | `diamond cut glass` | "Diamonds are renowned for … hardness and brilliance, which stem from their crystal structure …" | generator described diamond's hardness without naming the cut-glass action; entailment did not find the predicate. |
| `doc_030` | `muscle contract fiber` | "Muscles … consist of specialized cells known as muscle fibers. When a muscle needs to contract, the m…" | generator inverted subject/object — narrative has *muscle fibers* as the thing that contracts, not the thing *muscle contracts on*; the SVO triple is read by the checker with `muscle` as subject, so no match.

**Interpretation:** LLM-rendered narratives conditioned on SUM's structured axioms preserve 94–96 % of source claims under independent entailment judgement, sampled across two independent runs one week apart with identical pinned models. The 4–6 % gap is the empirical ceiling of the `LiveLLMAdapter` + `LlmEntailmentChecker` stack on simple SVO inputs; per-document attribution names each failure and makes the gap debuggable at the generator-prompt layer rather than opaque. Each of the three 2026-04-19 failures is a specific, different kind of drift (polarity flip; predicate omission; subject/object inversion) — there is no single fix that would close the gap, which is itself a finding.

**Boundary:** FActScore is empirical, not provable. The generator could be swapped for a stricter constrained-decoding pipeline (XGrammar + WebNLG-fine-tuned T5) that raises this number, and the checker could be swapped for a specialist like MiniCheck-FT5. Both are roadmap items. Until then, the 0.94–0.96 band stands as the honest measurement of the current stack.

### 2.5. LLM Narrative Round-Trip Drift (Full Loop)

The full LLM narrative round-trip — `text → LLM.extract → axioms → LLM.generate → prose' → LLM.extract → axioms'` — now has a measured number, ending an "unmeasured claim" row that stood since §6 was introduced.

**First end-to-end run (2026-04-19, temporary API key, rolled):**

| corpus | generator | extractor | n_docs | source axioms (total) | reconstructed axioms (total) | drift_pct | exact-match recall |
|---|---|---|---|---|---|---|---|
| `seed_v1` | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` | 50 | 50 | 600 | **107.75 %** | **6 / 50 (0.12)** |

**Two numbers, both load-bearing:**

1. **`drift_pct = 107.75 %`** is the mean per-document `100 * |A Δ A'| / max(|A|, |A'|)`. It exceeds 100 % on a majority of documents because the LLM-extracted axiom set from the generated narrative is on average **12× the size** (mean `n_reconstructed = 12.0`, range 4–21) of the single source axiom the generator was asked to preserve. With `|source| = 1` and `|recon| = 12`, one missing plus twelve extra triples over a denominator of 12 gives ~108 %; the formula is doing what it says, not drifting numerically.

2. **`exact-match recall = 0.12`** (6 of 50 documents had their exact source triple appear verbatim in the LLM's re-extraction) is the honest answer to the question the drift number is *asked* to answer. The other 44 documents lost the exact surface form of the source triple through two dominant mechanisms, both visible in the per-doc attribution:
   - **Generator elaboration.** For `alice likes cats`, the generator produces a narrative about companionship and affection, and the extractor — unprompted for faithfulness — reads out ≥4 triples like `alice has_fondness_for cats` / `cats provide companionship` / `cats can_be source_of_joy`. The source triple is *semantically preserved* but not *surface-preserved*.
   - **Entity and predicate paraphrase.** For `newton described gravity` the extractor returns `isaac_newton described gravity`. For `fish eat plankton` it returns `fish consume plankton`. The facts are the same; the string-keyed symmetric-difference is not. This is a *canonicalization* failure at the extractor layer, not a reasoning failure at the generator layer.

**Interpretation:** the two measurements say the same thing from opposite sides: **the `LiveLLMAdapter` generator+extractor pair preserves *facts* but not *keys*.** FActScore (§2.4) judges facts and returns 0.94–0.96. Round-trip drift (§2.5) judges keys and returns 108 %. Neither is wrong; they disagree because the pipeline is not key-stable end-to-end.

SUM's numbers here are a specific instance of a phenomenon the distillation literature has characterised since 2021 — the measurement is new, the pattern is not.

- **Stanton et al., "Does Knowledge Distillation Really Work?"** (NeurIPS 2021): even in self-distillation with identical architectures, students fail to achieve high fidelity to their teachers, and *higher fidelity does not always mean better generalization*. The optimisation layer, not the capacity layer, is the load-bearing constraint. SUM's 12× amplification (generator produces 12 reconstructed triples per source) and 0.12 exact-match recall are what their theoretical framing predicts when the student is unconstrained by a fidelity objective.
- **Menon et al., "A Statistical Perspective on Distillation"** (ICML 2021, PMLR 139:7632–7642): distillation helps because the teacher approximates the Bayes class-probability function; soft labels reduce the variance of the student's objective relative to one-hot targets. The 0.94–0.96 FActScore reflects Bayes-probability preservation (the facts survive); the 108 % key drift reflects the variance that soft-label elaboration introduces into the surface form. Both behaviours are simultaneous predictions of the same bias-variance account.

The §2.5 numbers do not require a new mechanism story — they are a measurement of SUM's specific pipeline against an already-documented regularity. Note also **Saxe et al., "On the Information Bottleneck Theory of Deep Learning"** (ICLR 2018), which constrains how this should be framed: the information-bottleneck objective remains a legitimate *objective* (min I(X;T), max I(T;Y)), but a causal claim that SGD's compression phase produces generalisation is *refuted* — the compression phase is an artefact of saturating nonlinearities like tanh and vanishes under ReLU. SUM's compression is explicit and symbolic (sieve + prime encoding), not emergent from SGD, so this warning does not wound any claim SUM makes — but it should stop any future claim of "SUM generalises via the information bottleneck mechanism" from being written.

**Boundary:** closing this gap is a canonicalization problem, not an LLM problem. An entity-resolution pass that collapses `newton` and `isaac_newton`, a WordNet or lemma-based predicate normaliser, or a prompt that asks the extractor to return triples in a pinned vocabulary would move the 0.12 exact-match recall upward without changing the generator. None of these are shipped. The 107.75 % / 0.12 numbers stand as the honest empirical ceiling of the unprompted, unresolved `LiveLLMAdapter` pipeline — which is what §6 promised to measure, and what §6 is now free to stop promising.

### 2.6. Bench Harness Substrate

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
| **Polytaxis Bucket A absorption** (SHACL, conformal prediction sets, VC 2.0 with `eddsa-jcs-2022`, RFC 3161 timestamping, RFC 9162 CT v2 proofs, PROV-O/PROV-STAR, polyglot RDF/JSON-LD/Turtle emission) | **In progress** — shipped: `epistemic_status` field (v1.2.0), Venn-Abers conformal-interval algorithm + `ConfidenceCalibrator` wiring, PROV-O JSON-LD adapter for Akashic Ledger events, W3C VC 2.0 Data Integrity emission + verification under `eddsa-jcs-2022` (`internal/infrastructure/verifiable_credential.py` + pure-Python RFC 8785 JCS at `internal/infrastructure/jcs.py`, 58 tests). **Pending:** SHACL, RFC 3161 TSA anchor, RFC 9162 CT v2 proofs, full polyglot emission (Turtle/RDF-XML beyond JSON-LD) | **Phase 25** |
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
| Tome → Tag (extraction) | **F1 = 1.000** on seed_v1, **F1 = 0.762** on seed_v2 | seed_v1: 50/50 simple SVO. seed_v2 (difficulty corpus, 20 docs × 7 parse patterns — apposition, passive voice, relative clause, conjunction, negation, hedging, complex PP): **precision 1.000, recall 0.615, TP=16/26, predicted=16**. The sieve now emits ZERO false positives on seed_v2 — every remaining failure is a RECALL miss (dropped fact), not a TRUTH inversion (asserted-wrong fact). Truth-layer classes closed: negation (doc_016, doc_017) and modal hedging (doc_018) suppressed; passive voice with agent (doc_007-009) now swaps the grammatical subject/object via the `agent → pobj` spaCy dep path to recover active-form triples; agentless passive suppressed. Remaining RECALL-reducing failure modes: relative clauses drop the subordinate fact (doc_010-012); apposition drops the "X be Y" fact (doc_004-006); compound subject/object drops all but the first conjunct (doc_013-015); prepositional-complement verbs return empty (doc_020). These are misses, not lies — the Gödel state is no longer corrupted by surface-voice parsing choices. |
| Tag → Tome (canonical, deterministic) | Working | Mathematically proven round-trip (§1.1) |
| Tag → Tome (narrative, prose) | Requires LLM extrapolator | No empirical measurement yet |
| Round-trip conservation (canonical) | Proven + empirically verified per-run | §1.1; 0.00% drift on `seed_tiny_v1` and `seed_v1` (commits `a6606eb`, current) |
| Round-trip conservation (sieve re-extract of canonical text) | Measured | 42.86% (seed_tiny) / 50.00% (seed_v1) drift; sieve is not bijective even on its own output |
| Regeneration faithfulness (LLM narrative from axioms) | **Measured (two runs)** | FActScore = **0.960** (48/50, 2026-04-17) / **0.940** (47/50, 2026-04-19) on seed_v1 with `gpt-4o-mini-2024-07-18` for both generator and entailment checker; delta below regression threshold, both runs on record; see §2.4 |
| Round-trip conservation (LLM narrative prose, full loop) | **Measured** | drift = **107.75 %**, exact-match recall = **0.12** on seed_v1 (2026-04-19), both legs `gpt-4o-mini-2024-07-18`. 50 source axioms → 600 extracted (12× amplification); per-doc attribution confirms the pattern is generator elaboration + extractor paraphrase, not reasoning failure. See §2.5 |
| Extraction ceiling investigation (en_core_web_trf upgrade or LLM fallback) | 8 / 50 seed_v1 failures all fit one spaCy parse pattern; architectural decision pending | User call |
| Sliding-scale rendering parameters | **Interface shipped** (`TomeSliders`): 5 axes — density / length / formality / audience / perspective. Density slider actioned on the deterministic canonical path (lexicographic axiom subsetting); remaining 4 axes LLM-gated and captured in output header as metadata | Phase 30+ (LLM wiring for non-density axes) |
| Cryptographic attestation | Working | Ed25519 + HMAC-SHA256 + Merkle chain |
| Epistemic-status labeling | Shipped v1.2.0 | See §5 |
| SHACL structural validation (Polytaxis Bucket A) | Not yet | Phase 25 |
| Conformal prediction confidence (Polytaxis Bucket A) | Algorithm shipped (`internal/ensemble/venn_abers.py`, 18 tests); **production wiring via `ConfidenceCalibrator.calibrate_interval()` shipped** with `load_venn_abers_fixture()` helper and fixture tests; calibration-set authoring is the remaining step | Phase 25 |
| VC 2.0 `eddsa-jcs-2022` emission + verification | **Shipped** (`internal/infrastructure/verifiable_credential.py` + pure-Python RFC 8785 JCS at `internal/infrastructure/jcs.py`, 58 tests: 30 JCS + 28 VC covering sign/verify round-trip, tamper detection, JSON-on-disk persistence, multibase base58btc round-trip, key-reordering resilience) | Phase 25 |

The gap from `current honest state` to `ultimate goal` is the refactor roadmap of record. Any PR that claims to close part of this gap must update this section with the new measurement.
