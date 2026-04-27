# Proof Boundary

**Version:** 1.4.0
**Date:** 2026-04-27

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

**Claim:** The Gödel State Integer is runtime-independent. Given the same canonical tome, Python (sympy), Node.js (BigInt + Miller-Rabin via `standalone_verifier/math.js`), and in-browser JavaScript (the inlined copy in `single_file_demo/index.html`) all produce byte-identical state integers.

**Proof mechanism:** Four independent harnesses lock the contract in CI (the first three cover valid inputs; the fourth covers adversarial inputs):
- `scripts/verify_cross_runtime.py` — Python mints a CanonicalBundle via `CanonicalCodec.export_bundle`; Node.js reconstructs via `standalone_verifier/verify.js`; state integers must match byte-for-byte. K1 / K1-multiword / K2 / K3 / K4.
- `scripts/verify_godel_cross_runtime.py` — 12 axiom keys (including UTF-8 and multi-word cases) minted in both Python and Node; 6 triple-lists encoded to state integers in both. 18 / 18 fixtures byte-identical.
- Browser-minted bundle → `node standalone_verifier/verify.js` — the inlined JavaScript in the single-file demo produces a CanonicalBundle that validates under the Node verifier unchanged, closing the three-runtime loop.
- `scripts/verify_cross_runtime_adversarial.py` — ADVERSARIAL rejection matrix. Six deliberately-malformed bundles (missing tome, truncated tome, state integer = 0, state integer = -42, canonical_format_version = 99.0.0, Ed25519-signed bundle with tome tampered post-sign). Both verifiers must reject AND classify the rejection equivalently (`structural` / `signature` / `version` / `scheme`). This closes the "agree on invalidity" gap that the first three harnesses left open — see Priority 1 in `docs/NEXT_SESSION_PLAYBOOK.md`.

**Boundary:** All three implementations use the same deterministic prime derivation (`SHA-256(axiom_key) → first 8 bytes big-endian → seed → nextprime(seed)`) via the `sha256_64_v1` scheme. The collision-resolution path depends on minting order; it has cross-*instance* coverage (two `GodelStateAlgebra` instances minting in different orders produce identical primes for identical keys, stress-tested at 1,000 axioms) but is not yet cross-*runtime* collision-verified. Production corpora up to ~2³² axioms have birthday-bound collision probability < 10⁻⁹; the path is not load-bearing at current scale.

### 1.3. Bundle Tamper Detection (Trusted Peers)

**Claim:** HMAC-SHA256 signatures detect any modification to the canonical tome, state integer, or timestamp.

**Proof mechanism:** Import rejects bundles with invalid signatures.

**Boundary:** This is tamper detection, not authenticity. Both producer and consumer must share the HMAC key. A party with the key can forge signatures. See `THREAT_MODEL.md`.

### 1.3.1. Bundle Public-Key Attestation (Any Third-Party Verifier)

**Claim:** Ed25519-signed CanonicalBundles are tamper-detectable by any third party with no shared secret. The same bundle bytes verify identically in Python, Node.js, and modern browsers — the three-runtime trust triangle is byte-symmetric.

**Proof mechanism:** Three cross-runtime gates:
- `sum verify` (Python, `sum_cli/main.py::_verify_ed25519_bundle`) — decodes the embedded `public_key` and `public_signature`, re-computes the `{tome|state|timestamp}` payload, verifies with `cryptography.Ed25519PublicKey.verify`.
- `standalone_verifier/verify.js` (Node ≥ 18.4, `verifyEd25519`) — same payload, same key bytes, `crypto.webcrypto.subtle.verify({name:'Ed25519'})`.
- `single_file_demo/index.html` (Browser Chrome 113+ / Firefox 129+ / Safari 17+, `verifyEd25519InBrowser`) — same payload, same key bytes, `crypto.subtle.verify({name:'Ed25519'})`.

Locked in CI by the cross-runtime harness K3 (positive: Python mints Ed25519 bundle → Node verifies ✓) and K4 (negative: tampered tome → Node reports `✗ INVALID`). K4 is what proves verify.js actually runs the signature check rather than reporting `verified` unconditionally.

**Boundary:** The signature authenticates the Gödel state + tome + timestamp. It does NOT authenticate the source of the prose the tome was extracted from — that's what the `AkashicLedger` provenance layer (feature 101) exists for. Bundles without Ed25519 fields fall back to structural verification only; `--strict` enforces at least one verifiable signature.

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

**Claim:** Any modification, deletion, or injection of events in the Akashic Ledger is detectable, and this property now holds under concurrent writers.

**Proof mechanism:** Each event stores `prev_hash = SHA-256(prev_hash + operation + prime + axiom_key + branch)`. Genesis seed: `SHA-256("SUM_GENESIS_BLOCK")`. `verify_chain()` walks the full chain on boot, reporting the first broken link. 16 tests in `test_merkle_chain.py` verify single-writer tamper detection (mutation, deletion, hash overwrite, injection). An additional 6 concurrency tests in `test_ledger_concurrency.py` verify the invariant holds under 50-200 parallel `append_event` calls.

**Concurrency hardening (commit `9c4139d`):** Until this fix, `append_event` read `prev_hash` in autocommit mode (Python's sqlite3 default) and only began a transaction on the subsequent INSERT. Two concurrent writers could both observe the same `prev_hash`, compute event hashes against the same stale parent, and both commit — leaving `verify_chain()` reporting `is_valid=False` on a perfectly well-behaved multi-writer pipeline. The fix wraps every writer in `BEGIN IMMEDIATE`, acquiring the reserved write-lock before the SELECT and serialising writers at the SQLite boundary. The discipline is now centralised in `AkashicLedger._write_txn` (commit `76ceb40`) so future writers inherit it automatically.

**Boundary:** This is tamper detection, not prevention. A local attacker with full SQLite write access can rewrite the entire chain from genesis. The hash chain proves no event was modified after the fact by an actor without that access.

### 1.8. Render Receipt Cryptographic Binding (Phase E.1 v0.9.A)

**Claim:** Every successful `/api/render` response carries a `render_receipt` whose signed payload binds, under Ed25519, the exact tome bytes (`tome_hash`), the post-density triple set (`triples_hash`), the post-quantize slider position (`sliders_quantized`), the model that actually served (`model`), the provider taxonomy value (`provider`), the issuer's stamping timestamp (`signed_at`), the C2PA `digital_source_type`, and a content-addressed `render_id`. Mutating any signed field invalidates the signature; the verifier rejects with `ERR_JWS_SIGNATURE_VERIFICATION_FAILED`.

**Proof mechanism:** Standard JOSE/JCS bindings, implemented in `worker/src/receipt/sign.ts` and specified end-to-end in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md):
- **Ed25519 (RFC 8032)** signature over the **JCS-canonical (RFC 8785)** UTF-8 byte representation of the `payload` object.
- **Detached JWS (RFC 7515 §A.5)** with `b64: false` per RFC 7797 — the canonical bytes ARE the detached payload; the middle segment of the compact JWS is empty.
- **JWKS distribution (RFC 7517)** at `/.well-known/jwks.json` — single Ed25519 OKP JWK entry, content-type `application/jwk-set+json`, `Cache-Control: public, max-age=3600`. Receipt's `kid` selects the verifying key.
- Protected header pins `alg: "EdDSA"`, the matching `kid`, `b64: false`, and `crit: ["b64"]` so older verifiers fail closed on the unencoded-payload semantics.

The cross-runtime canonicalisation rule (JCS normalises integer-valued floats — `1.0` → `1`, `-0` → `0` — per RFC 8785 §3.2.2.3) is byte-stable across `canonicalize@3` (TypeScript) and `jcs` (Python); 10/10 edge fixtures verified in the v0.9.A research pass.

**Boundary:** What the signature proves and what it does not is canonical in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §5 (Trust Scope). The signature authenticates the *render attestation* (issuer signed this tome / triples / sliders / model / time tuple); it does not authenticate the tome's factual content, the freshness of a cache-HIT response, or the issuer's beliefs about what their configured-default model should have been. Issuer key compromise, freshness replay, and issuer collusion are explicitly out of scope (§5.1 threat model).

The cryptographic binding moves to "proved on adversarial inputs" once the v0.9.B browser verifier and v0.9.C Python verifier land with negative-path fixtures (every signed field tampered → `ERR_JWS_SIGNATURE_VERIFICATION_FAILED`); until then, the negative path is exercised in `worker/`-local TypeScript tests but not yet locked across runtimes.

---

## 2. Empirically Measured

These properties are observed on a fixed benchmark but not formally proven. They carry the epistemic status `empirical-benchmark` (see §5) and depend on implementation quality, input characteristics, and runtime environment.

### 2.1. Extraction Fidelity

The quality of semantic extraction from natural language depends on:
- The NLP parser (spaCy lemmatizer, dependency parser, model variant)
- Input text structure and complexity
- Domain vocabulary coverage

**Bench harness measurements (schema v0.3.0):**

| corpus | size | precision | recall | F1 | correct / gold |
|---|---|---|---|---|---|
| `seed_tiny_v1` | 8 SVO sentences | 1.000 | 1.000 | **1.000** | 8 / 8 |
| `seed_v1` | 50 SVO sentences | 1.000 | 1.000 | **1.000** | 50 / 50 |
| `seed_v2` | 20 difficulty-corpus docs | **1.000** | 0.615 | **0.762** | 16 / 26 |

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

**Residual ceiling (now measured):** `seed_v1`'s 100 % F1 reflects that corpus's scope — simple declarative SVO, one fact per document. `seed_v2` was authored specifically to exercise the parse patterns `seed_v1` deliberately excludes: apposition, passive voice (with and without agent), relative clauses, conjunction (compound subject + object), negation, modal hedging, and prepositional-complement verbs. `seed_v2`'s **0.762 F1 with precision = 1.000** is the honest empirical ceiling of the current sieve on real-prose constructions. Every remaining `seed_v2` failure is a RECALL miss (a fact dropped), never a TRUTH inversion (a fact asserted wrong) — the two truth-layer bug classes that used to corrupt the Gödel state are now closed: negation is intentionally suppressed (commit `ef392cb`) and passive-with-agent is now semantically inverted to active form (commit `b751222`).

**Prior documented benchmark:** A 50-document golden benchmark corpus exists (Phase 19B) spanning 7 adversarial categories with 100 gold-standard triplets. That corpus remains the source of truth for Phase 19B claims; `seed_v1` is the bench-harness benchmark and complements Phase 19B rather than replacing it.

Structural gating (Phase 19A) rejects malformed triplets. Semantic quality on non-trivial inputs remains the acknowledged weakest link.

### 2.2. Operation Performance

Gödel arithmetic operations (LCM, GCD, modulo) operate on arbitrary-precision integers. Their complexity scales with integer **bit length**, not axiom count:
- GCD: O(n²) via Euclidean algorithm on n-bit integers (sub-quadratic with GMP)
- LCM: O(n²) (reduces to GCD)
- Modulo: O(n²)

**Bench harness measurements (commit `9ed49bf` and later, Darwin arm64 / Python 3.10.14 / CPython / no Zig):**

Core algebra operations (via `scripts/bench/runners/performance.py`):

| operation | N=100 | N=500 | N=1000 | empirical scaling |
|---|---|---|---|---|
| ingest per-triple (p50) | 0.049 ms | 0.046 ms | 0.045 ms | **O(1) stable** |
| encode (p50) | 0.131 ms | 1.552 ms | 5.107 ms | ~O(n²) |
| merge (p50) | 28.4 ms | 206.4 ms | **518.8 ms** | ~O(n²) — bottleneck |
| entail (p50) | 0.014 ms | 0.062 ms | 0.123 ms | ~O(n) |

Provenance path (via `scripts/bench_provenance_path.py`, N=100/1000/5000):

| operation | p50 | p99 | steady ops/sec | scaling |
|---|---|---|---|---|
| `compute_prov_id` (JCS + SHA-256) | 35 µs | 45 µs | ~28 k | flat — crypto ceiling |
| `record_provenance` (single-tx write) | 460 µs | 1 ms | ~2 k | flat — SQLite-bound |
| `record_provenance_batch` (single-tx N-insert) | 45 µs amortised | 45 µs | **~22 k** | flat — within 30% of crypto ceiling (**10.2× the single-write path**) |
| `get_structured_provenance_for_axiom` | 128 µs | 170–600 µs | ~7 k | flat — indexed lookup |

The batch path (`record_provenance_batch`, commit `9ed49bf`) lifts sustained ingest from ~2 k/sec to ~22 k/sec on a single ledger handle, closing the gap to the prov_id compute ceiling. For machine-consumer pipelines above ~100 k docs/min, shard by axiom_key hash or move storage off SQLite; for human-scale use cases, the single-write path is already sub-millisecond.

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

### 2.6. Slider Axis Fact-Preservation (Phase E.1 v0.4 → v0.7)

The slider's load-bearing claim — *axis changes do not lose facts* — has been **empirically verified** across two independently-authored corpora, a four-layer fact-preservation substrate, and a deterministic prompt-hardening mechanism that closed the catastrophic-failure mode v0.6 surfaced. [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) is the canonical contract document; this section pins the load-bearing numbers as `empirical-benchmark` and links the failure-mode arc.

**Bench harness measurements:**

| Run | Corpus | LLM-axis cells | Median | p10 | Min | NLI rescue rate | Real losses | Catastrophic outliers (≥5) |
|---|---|---|---|---|---|---|---|---|
| v0.4 | `seed_paragraphs.json` (n=8 short, 4–12 triples/doc) | 160 | **1.000** | **0.818** | 0.727 | 100 % (186/186) | **0** | 0 |
| v0.6 (no hardening) | `seed_long_paragraphs.json` (n=16 long, 9–24 triples/doc) | 320 | 1.000 | 0.769 | 0.111 | 95.7 % (800/836) | 36 | **2** |
| v0.7 (`FACT_PRESERVATION_REINFORCEMENT`) | same long bench | 319 | 1.000 | 0.750 | **0.700** | 99.8 % (653/654) | **1** | **0** |

**Layered fact-preservation metrics** (all reported per cell in the JSONL artifact):

- **Strict** — exact `(s, p, o)` match. Brittle to surface-form drift; retained as regression check.
- **Normalized (A3)** — strips auxiliary-verb prefixes (`was_`, `has_`) + preposition suffixes (`_in`, `_from`) from predicates, articles from entities. Free, deterministic.
- **Semantic (A1)** — greedy one-to-one cosine similarity on triple-as-text embeddings (`text-embedding-3-small`, threshold 0.85).
- **NLI audit (v0.4)** — LLM-as-judge entailment (`LiveLLMAdapter.check_entailment`, Pydantic-enforced). Fires only when semantic < `--audit-threshold` (default 0.7). Load-bearing metric for the slider claim.

**Prompt-hardening mechanism (v0.7, deterministic, no extra LLM cost):** `build_system_prompt` (Python in `tome_sliders.py`; TS mirror in `worker/src/render/axis_prompts.ts`) appends a `FACT_PRESERVATION_REINFORCEMENT` clause when any non-density axis is at ≤ 0.3. Same input → same output; the mechanism is data, not learning.

**MontageLie defence:** order preservation = 1.000 wherever measurable across all benches. Set-based fact preservation alone is exploitable by reordering true facts into a deceptive narrative (Zheng et al. May 2025); pairing NLI audit with `order_preservation` is harder to defeat than either alone.

**LLM self-attestation is NOT a free oracle.** v0.3 added `claim_jaccard` measuring agreement between the LLM's `claimed_triples` and an independent re-extraction; cross-axis median = 0.286. Counts match (n_claimed ≈ n_reextracted ≈ n_source) — surface-form divergence, not list-size mismatch. **Independent re-extraction remains the source of truth**; do not ship a "fast mode" that skips it in favour of `claimed_triples` (the bench data shows that mode would systematically under-report preservation).

**Reproduce:**
```
bash scripts/bench/run_paragraphs.sh        # short, n=8, ~$0.30, ~2 min with NLI
bash scripts/bench/run_long_paragraphs.sh   # long, n=16, ~$1.50, ~10 min with NLI
```

Both runners require `OPENAI_API_KEY`. Pinned model snapshots are mandatory; the harness raises `SystemExit` on unpinned identifiers (see §2.8).

**Boundary:** "median 1.000" describes the LLM-axis cells; the density axis explicitly drops facts at `density < 1.0` (it's the product knob), and density-axis "losses" in the bench summary are loss-by-design, not loss-by-accident. The remaining 1 confirmed real loss across 654 audited cells (v0.7, on the audience axis) is at the LLM's hard ceiling on this corpus, not a contract violation. Future canonicalisation work (QID-keyed triples) would make A1+NLI superfluous; the four-layer substrate is the bridge.

### 2.7. Robustness — `LengthFinishReasonError` Four-Layer Defence (Phase E.1 v0.8)

The v0.7 long-doc bench errored on 1 / 400 cells when re-extraction overflowed the 16384-token completion ceiling. v0.8 lands a four-layer defence and re-runs the same bench: **0 / 400 cells errored.**

| Run | Errored cells | Median | Catastrophic outliers |
|---|---|---|---|
| v0.7 | 1 / 400 | 1.000 | 0 |
| v0.8 | **0 / 400** | 1.000 | 0 |

**The four layers** (in `sum_engine_internal/ensemble/live_llm_adapter.py`):
1. **Prompt-side cap** — system prompt now states `Return at most 64 triplets…`. LLM compliance under structured output is empirically high.
2. **Partial-response salvage** — `salvage_partial_triplets` walks the truncated JSON in `e.completion.choices[0].message.content` and returns whatever complete triplet objects appeared before the cutoff. Pure function; free (same response).
3. **One-shot retry with tighter cap** — when salvage yields nothing, retry once with `cap=32` plus an emphatic note. Bounded to a single extra API call.
4. **Re-raise on retry failure** — terminal; escalates to caller.

**Wild events in the v0.8 bench run:** 1× salvage fired (recovered 19 triplets from a partial response, `cap=64`, `completion_tokens=16384`; free). 1× retry-with-cap=32 fired on a different cell. Both events logged; no errors propagated.

**Pin bump (load-bearing):** `LengthFinishReasonError` was added in `openai-python 1.40.0` alongside structured-outputs support. `pyproject.toml` and `requirements-prod.txt` bumped from `openai>=1.0.0` to `openai>=1.40.0,<3.0.0`; without the bump, fresh installs that pip-resolve to <1.40 would `ImportError` on `from openai import LengthFinishReasonError`.

**Verification:** 60 unit tests pass (51 slider + 9 salvage); 1095 full Python suite pass; cross-runtime gates K1–K4 + A1–A6 green; bench: 400/400 cells succeed.

### 2.8. Bench Harness Substrate

The `scripts/bench/` directory contains the measurement-first infrastructure that makes §2.1–§2.3 reproducible. Key properties:

- **Every report is content-addressable.** `run_id`, `git_sha`, host, Python version, and model snapshots are captured inline. Corpus SHA-256 snapshot hash travels with each report; corpus mutation invalidates historical comparisons at the hash layer.
- **Model snapshots MUST be pinned** (e.g., `gpt-4o-2024-08-06`, not `gpt-4o`). Unpinned identifiers raise `SystemExit` before any work begins.
- **`PerformanceRunner` uses synthetic triples** `(s_i, p, o_i)` for deterministic, non-colliding primes; exercises the pure-Python path even when the Zig core is absent.
- **`ExtractionRunner` uses set-comparison on canonical keys** (no post-hoc lemmatization reconciliation). Gold-triple mismatches with sieve output count as false negatives. Honesty over flattery.
- **CI regression detection** compares each new report against the most recent history entry; `--fail-on-regression` exits non-zero on any F1 drop > 0.02, drift increase > 1%, FActScore drop > 0.03, or p99 ratio > 1.15.
- **LLM-gated runners** (`regeneration.py`, `roundtrip.py`, `llm_roundtrip.py`) require a pinned snapshot ID (e.g. `gpt-4o-mini-2024-07-18`). The harness reads `SUM_BENCH_MODEL` as the single default applied to every role; per-role overrides `SUM_BENCH_FACTSCORE_MODEL`, `SUM_BENCH_MINICHECK_MODEL`, `SUM_BENCH_GENERATOR_MODEL`, and `SUM_BENCH_EXTRACTOR_MODEL` take precedence when set. Unpinned or missing identifiers raise `SystemExit` before any work begins.

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
| **Bidirectional distillation with sliding-scale parameters** (density, length, formality, audience, perspective) | **Shipped end-to-end (Phase E.1 v0.4 → v0.7)** — density on the deterministic canonical path; length / formality / audience / perspective LLM-conditioned via `worker/src/routes/render.ts` + `worker/src/render/axis_prompts.ts` (TS mirror of the Python prompt fragments). Fact-preservation verified at scale: median 1.000, p10 0.769 (long n=16) / 0.818 (short n=8), catastrophic outliers eliminated by v0.7 prompt hardening — see §2.6. | **Measured / production** (was "Phase 30+"); render-receipt attestation per call (§1.8) |
| **Polytaxis Bucket A absorption** (SHACL, conformal prediction sets, VC 2.0 with `eddsa-jcs-2022`, RFC 3161 timestamping, RFC 9162 CT v2 proofs, PROV-O/PROV-STAR, polyglot RDF/JSON-LD/Turtle emission) | **In progress** — shipped: `epistemic_status` field (v1.2.0), Venn-Abers conformal-interval algorithm + `ConfidenceCalibrator` wiring, PROV-O JSON-LD adapter for Akashic Ledger events, W3C VC 2.0 Data Integrity emission + verification under `eddsa-jcs-2022` (`sum_engine_internal/infrastructure/verifiable_credential.py` + pure-Python RFC 8785 JCS at `sum_engine_internal/infrastructure/jcs.py`, 58 tests). **Pending:** SHACL, RFC 3161 TSA anchor, RFC 9162 CT v2 proofs, full polyglot emission (Turtle/RDF-XML beyond JSON-LD) | **Phase 25** |
| Prose round-trip conservation measurement (via `SumRoundtripRunner` + LLM extrapolator + MiniCheck gate) | **Measured** — see §2.5 | STATE 4-B (shipped) |
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

The `epistemic_status` field is mandatory on every `BenchReport` metric record as of schema v0.3.0 (introduced v0.2.0 in commit `321e573`, carried forward through the per-doc regeneration and llm_roundtrip additions in `02b4413` and `9fd232d`). Future upgrades: every signed Verifiable Credential emitted by SUM will carry the same field; every claim returned by `/ask` and `/extrapolate` endpoints will carry the same field in the response envelope.

**Conflation rule:** A summary or marketing surface that quotes an `empirical-benchmark` number alongside language like "mathematically guaranteed" or "proven" is a policy violation and must be corrected. The README, THREAT_MODEL, and CANONICAL_ABI_SPEC are required to observe this rule; `PROOF_BOUNDARY.md` is its arbiter.

---

## 6. Progress Toward the Ultimate Goal

SUM's ultimate goal is a **bidirectional knowledge distillation engine**: turn narrative tomes into structured tags and vice versa, with tunable sliders for density, formality, perspective, and audience — truthful in every claim it purports.

**Current honest state (commit `3ade8c9` and later — see the git log for the running tip):**

| Capability | Status | Measurement |
|---|---|---|
| Tome → Tag (extraction) | **F1 = 1.000** on seed_v1, **F1 = 0.762** on seed_v2 | seed_v1: 50/50 simple SVO. seed_v2 (difficulty corpus, 20 docs × 7 parse patterns — apposition, passive voice, relative clause, conjunction, negation, hedging, complex PP): **precision 1.000, recall 0.615, TP=16/26, predicted=16**. The sieve now emits ZERO false positives on seed_v2 — every remaining failure is a RECALL miss (dropped fact), not a TRUTH inversion (asserted-wrong fact). Truth-layer classes closed: negation (doc_016, doc_017) and modal hedging (doc_018) suppressed; passive voice with agent (doc_007-009) now swaps the grammatical subject/object via the `agent → pobj` spaCy dep path to recover active-form triples; agentless passive suppressed. Remaining RECALL-reducing failure modes: relative clauses drop the subordinate fact (doc_010-012); apposition drops the "X be Y" fact (doc_004-006); compound subject/object drops all but the first conjunct (doc_013-015); prepositional-complement verbs return empty (doc_020). These are misses, not lies — the Gödel state is no longer corrupted by surface-voice parsing choices. |
| Tag → Tome (canonical, deterministic) | Working | Mathematically proven round-trip (§1.1) |
| Tag → Tome (narrative, prose) | **Measured** | FActScore 0.940 / 0.960 on seed_v1 via `LiveLLMAdapter.generate_text` + `LlmEntailmentChecker`, both legs `gpt-4o-mini-2024-07-18`; see §2.4 |
| Round-trip conservation (canonical) | Proven + empirically verified per-run | §1.1; 0.00% drift on `seed_tiny_v1` and `seed_v1` (commits `a6606eb`, current) |
| Round-trip conservation (sieve re-extract of canonical text) | Measured | 42.86 % (seed_tiny) / 54.00 % (seed_v1) / 56.25 % (seed_v2) drift; sieve is not bijective even on its own canonical output — monotonic with corpus difficulty, see §2.3 |
| Regeneration faithfulness (LLM narrative from axioms) | **Measured (two runs)** | FActScore = **0.960** (48/50, 2026-04-17) / **0.940** (47/50, 2026-04-19) on seed_v1 with `gpt-4o-mini-2024-07-18` for both generator and entailment checker; delta below regression threshold, both runs on record; see §2.4 |
| Round-trip conservation (LLM narrative prose, full loop) | **Measured** | drift = **107.75 %**, exact-match recall = **0.12** on seed_v1 (2026-04-19), both legs `gpt-4o-mini-2024-07-18`. 50 source axioms → 600 extracted (12× amplification); per-doc attribution confirms the pattern is generator elaboration + extractor paraphrase, not reasoning failure. See §2.5 |
| Extraction ceiling investigation (en_core_web_trf upgrade or LLM fallback) | seed_v1 at F1 = 1.000 (no remaining failures); seed_v2 at F1 = 0.762 with precision = 1.000 — every remaining failure is a RECALL miss not a TRUTH inversion (apposition secondary, relative-clause subordinate, compound non-head conjuncts). Architectural decision pending on whether to address via `en_core_web_trf` upgrade or LLM fallback at the sieve boundary | User call |
| Sliding-scale rendering parameters | **Shipped end-to-end** — 5 axes (density / length / formality / audience / perspective). Density actioned deterministically via lexicographic axiom subsetting. Length / formality / audience / perspective LLM-conditioned via the Cloudflare Worker render path (`worker/src/routes/render.ts`, Anthropic provider, optional CF AI Gateway). Fact-preservation verified at scale (§2.6); robustness layered (§2.7). Every render carries a signed receipt (§1.8). | **Measured + cryptographically attested** |
| Cryptographic attestation | Working, cross-runtime | Ed25519 + HMAC-SHA256 + Merkle chain. Ed25519 verified in all three shipping runtimes against the same bundle bytes: Python (`sum verify`), Node (`standalone_verifier/verify.js` via WebCrypto), Browser (`single_file_demo/index.html` via SubtleCrypto). Locked by cross-runtime K3/K4 harness + A1–A6 adversarial-rejection matrix in CI. |
| Per-render attestation (Phase E.1 v0.9.A) | **Shipped** | `sum.render_receipt.v1` — Ed25519 (RFC 8032) over JCS-canonical (RFC 8785) payload bytes, wrapped as detached JWS (RFC 7515 §A.5) with public keys distributed via JWKS (RFC 7517) at `/.well-known/jwks.json`. Active kid `sum-render-2026-04-27-1`. Cryptographic binding documented in §1.8; full wire spec in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md). v0.9.B (browser verifier) and v0.9.C (Python verifier) are queued in [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md); they will close the negative-path proof across runtimes. |
| Epistemic-status labeling | Shipped v1.2.0 | See §5 |
| SHACL structural validation (Polytaxis Bucket A) | Not yet | Phase 25 |
| Conformal prediction confidence (Polytaxis Bucket A) | Algorithm shipped (`sum_engine_internal/ensemble/venn_abers.py`, 18 tests); **production wiring via `ConfidenceCalibrator.calibrate_interval()` shipped** with `load_venn_abers_fixture()` helper and fixture tests; calibration-set authoring is the remaining step | Phase 25 |
| VC 2.0 `eddsa-jcs-2022` emission + verification | **Shipped** (`sum_engine_internal/infrastructure/verifiable_credential.py` + pure-Python RFC 8785 JCS at `sum_engine_internal/infrastructure/jcs.py`, 58 tests: 30 JCS + 28 VC covering sign/verify round-trip, tamper detection, JSON-on-disk persistence, multibase base58btc round-trip, key-reordering resilience) | Phase 25 |

The gap from `current honest state` to `ultimate goal` is the refactor roadmap of record. Any PR that claims to close part of this gap must update this section with the new measurement.
