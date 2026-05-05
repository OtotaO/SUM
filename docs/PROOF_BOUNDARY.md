# Proof Boundary

**Version:** 1.6.0
**Date:** 2026-05-03

**v1.6.0 changes (2026-05-03):** §1.10 expanded — six per-regime compliance validators (EU AI Act Art 12, GDPR Art 30, HIPAA § 164.312(b), ISO 27001 A.8.15, SOC 2 CC7.2, PCI DSS v4.0 Req 10) consuming `sum.compliance_report.v1` without shape modification; the regime-agnosticism claim is now empirical fact, not single-instance assertion. §2.9 expanded — v3.2 detector closes F3 STRUCTURAL FAIL at the detector layer (PR #127); F1 / F2 / F3 / F4 / F5 verdict ladder pinned. §2.10 updated — Sprint 1 (PR #135) closed the `PYTHONHASHSEED=0` reproducibility caveat at the substrate layer (single-line `sorted(set(triplets))` fix in the deterministic sieve); bench reproducibility is now unconditional. Reflects substrate state at HEAD `86b2ed3` after PRs #117–#138 (Path 3 actionable layer + bench-substrate intensification arc).

**v1.5.0 changes (2026-05-02):** §1.9 added (audit-log primitive, v0.5.0 substrate), §1.10 added (compliance report shape), §2.9 added (sheaf-Laplacian detector measurements with F3 STRUCTURAL FAIL named explicitly), §2.10 added (`bench_digest` reproducibility under quantization). Reflects substrate state at HEAD `bb7957d` after PRs #117–#125.

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

**`sha256_128_v2` cross-runtime byte-identity (locked 2026-04-29):** the future-default prime scheme (`SHA-256 → first 16 bytes → BPSW-nextprime(seed)`) is now byte-identity-locked across Python and Node via `scripts/verify_godel_v2_cross_runtime.py` — 12 axiom-key fixtures (K1-v2) + 6 state-encoding fixtures (K2-v2), all byte-identical. Wired into CI alongside the v1 K-matrix gate. **The default scheme stays `sha256_64_v1`**; flipping it is a separate operator decision that requires a `bundle_version` minor bump per `docs/COMPATIBILITY_POLICY.md`. This gate proves the migration path is open (any future v1 → v2 cutover will not silently regress); the cutover itself is unshipped.

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

**Cross-runtime negative-path proof (Phase E.1 v0.9.B + v0.9.C, complete):** The shared receipt-fixture set under [`fixtures/render_receipts/`](../fixtures/render_receipts/) — 15 runtime-neutral JSON cases (positive control + 12 tampered-field variants + 2 forward-compat variants) — is consumed unchanged by both:
- The JS verifier in [`single_file_demo/receipt_verifier.js`](../single_file_demo/receipt_verifier.js), exercised by `node single_file_demo/test_render_receipt_verify.js` (CI: `vendor-byte-equivalence` job).
- The Python verifier in [`sum_engine_internal/render_receipt/`](../sum_engine_internal/render_receipt/), exercised by `pytest Tests/test_render_receipt_verifier.py` (CI: same job, parallel step).

Both runtimes MUST produce byte-identical outcomes on every fixture: same error class string for every reject case, signature verifies on the positive control. Cross-runtime divergence on any fixture is a stop-the-line trigger and would invalidate this section's claim. The cryptographic binding is therefore now **proved on adversarial inputs across runtimes** — the K-style equivalence we already have for CanonicalBundle, applied to render receipts.

### 1.9. Audit-Log Schema and Fail-Open Semantics (Path 3 substrate, v0.5.0)

**Claim:** Every CLI operation (`sum attest` / `sum verify` / `sum render`) emits exactly one JSONL row with the `sum.audit_log.v1` schema when `SUM_AUDIT_LOG` is set; the trust loop continues to function regardless of audit-log destination health (fail-open).

**Proof mechanism:** [`Tests/test_audit_log.py`](../Tests/test_audit_log.py) — **17 / 17 pass** (11 baseline + 6 PR-#119 gap-closure). Pins:
- Schema: `schema = "sum.audit_log.v1"`, ISO 8601 UTC `timestamp` ending in `Z`, `operation ∈ {"attest", "verify", "render"}`, `cli_version`.
- Operation-specific row shape (attest carries `signed`/`hmac`/`source_uri`; verify carries `axiom_count`/`state_integer_digits`; render carries `mode` plus worker-mode `render_receipt_kid`/`worker_url`/`render_receipt_schema`).
- Cross-reference: end-to-end attest → verify → render produces three rows whose `axiom_count`/`state_integer_digits` cross-link.
- Fail-open: `SUM_AUDIT_LOG` pointing at an unwritable path does NOT raise; the operation completes; the canonical bundle / receipt remains the load-bearing trust artifact regardless of audit destination state.
- Multi-process O_APPEND atomicity: 8 worker processes × 20 emits = 160 rows; every line parseable JSON, every (worker_id, iteration) pair present exactly once.
- Signed-bundle attest rows: Ed25519, HMAC, dual-signed paths each pinned with their own contract test (PR #119).
- Worker-mode render rows: synthetic-envelope test pins the four positive fields.
- Empty-string semantics: `SUM_AUDIT_LOG=""` treated as unset (no-op), pinned in PR #119.

**Boundary:** The audit log is regime-agnostic substrate. It records *what happened*; it does not enforce any specific compliance regime. Per-regime validation is a downstream consumer (see §2.9 EU AI Act Article 12 validator). The audit log is *advisory* — a non-functional audit destination is a compliance-tooling problem, not a substrate problem. Schema is additive: future minor versions may add fields; consumers MUST ignore unknown keys.

### 1.10. Compliance Report Schema and Regime-Agnostic Aggregation (Path 3 actionable layer)

**Claim:** Every per-regime validator on top of `sum.audit_log.v1` returns a `sum.compliance_report.v1` `ValidationReport` carrying schema, regime id, rows examined, and a list of typed `Violation` records (rule_id, row_index, operation, message, row). The shape is **regime-agnostic** — six per-regime validators across statutorily distinct domains all consume the same shape without modification. Downstream consumers (CLI, dashboard, retention pipeline) ingest reports across regimes without per-regime adapters.

**Six regime instances proving the regime-agnosticism claim** (each with its own statutory anchor + falsifiable rule set; none required schema changes to `sum.compliance_report.v1`):

| Regime | Statute | Rules | Tests |
|---|---|---|---|
| `eu-ai-act-article-12` | Reg (EU) 2024/1689 Art 12 (high-risk AI record-keeping) | 6 (R1–R6 + operation-specific anchors) | 32 |
| `gdpr-article-30` | Reg (EU) 2016/679 Art 30 (Records of Processing Activities) | 5 (per-row floor) | 25 |
| `hipaa-164-312-b` | 45 CFR § 164.312(b) (Audit Controls) | 6 (per-row floor + examination-completeness) | 27 |
| `iso-27001-8-15` | ISO/IEC 27001:2022 A.8.15 (Logging) | 5 (per-row floor) | 19 |
| `soc-2-cc-7-2` | AICPA TSP §100A CC7.2 (System Operations) | 5 (per-row floor) | 19 |
| `pci-dss-4-req-10` | PCI DSS v4.0 Req 10 (Log and Monitor) | 6 (per-row floor + examination-completeness) | 25 |

**Proof mechanism:** Dataclass shape pinned in [`sum_engine_internal/compliance/report.py`](../sum_engine_internal/compliance/report.py) (frozen). Each regime's test file (`Tests/compliance/test_<regime>.py`) carries a `test_validation_report_shape_matches_other_regimes` test that asserts byte-shape parity across all currently-shipped regimes — when PCI DSS landed, that test asserted six-way parity with one assertion. [`Tests/compliance/test_cli_dispatch.py`](../Tests/compliance/test_cli_dispatch.py) pins three cross-regime substrate contracts: (C1) `_COMPLIANCE_REGIMES` description registry keys ≡ `_compliance_validators()` dispatch registry keys; (C2) every regime returns `sum.compliance_report.v1` from `cmd_compliance_check`; (C3) exit codes 0 / 1 / 2 (ok / violations / usage error). Shared timestamp predicate at [`sum_engine_internal/compliance/_predicates.py`](../sum_engine_internal/compliance/_predicates.py) (PR #137 Sprint 3) — six regime modules import from one source; pinned in [`Tests/compliance/test_predicates.py`](../Tests/compliance/test_predicates.py) via object-identity check. Total compliance suite **164 / 164 pass**.

**Boundary:** Adding a new regime is a new module under `sum_engine_internal/compliance/<regime>.py` exposing `validate(rows) -> ValidationReport` plus dispatch entries in both registries. Existing `rule_id` strings are stable; downstream dashboards filter on them. **A green `ValidationReport` says the per-row form floor is satisfied** for that regime's record-keeping requirements; it does NOT prove operational compliance with the regime as a whole. Each regime's wire-spec doc (`docs/COMPLIANCE_<REGIME>.md`) names what's out of scope: organisational policies, retention duration, log file protection, review processes, user identification (PCI DSS structural gap, named explicitly), failure detection / alerting, etc.

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

**Merge cost above N=1 000 — extrapolation, not measurement.** The measured merge p50 at N=1 000 is 518.8 ms; the cost is empirically `O(B²)` where `B` is the state-integer bit-length and `B` itself scales linearly with N. Naive extrapolation along that curve gives **N=10 000 ≈ 50 s/op** and **N=100 000 ≈ 1.4 hr/op** — these numbers are the basis for the substrate guidance, not measured values, and they assume no GMP / sub-quadratic GCD acceleration. The closest direct measurement: at N=10 000 with 200 samples, the harness run did not converge inside a 10-minute wall-clock budget on the reference host (Darwin arm64, CPython 3.10, no Zig). That non-convergence is consistent with the extrapolation but does not pin it. **The guidance — *prime encoding is a viable substrate up to low-thousands of axioms and an attestation artifact above that* — holds on the measured N=100/500/1 000 trend; the >1 000 numbers above carry that caveat.** For corpora above that ceiling, plug in a property-graph backing store and retain the Gödel integer as a signed witness, not as the primary query path.

Use `--quick` for dev/PR-time runs; reserve full 1k/5k/10k × 200 samples for scheduled nightly runs on dedicated hardware.

**Merkle set-commitment sidecar — inclusion-proof verify vs LCM divisibility (M1, prototype):**

Companion measurement to the merge-cost ceiling above. The Merkle sidecar (`docs/MERKLE_SIDECAR_FORMAT.md`, `sum_engine_internal/merkle_sidecar/`) gives external verifiers a `log₂(N)` membership-witness path that bypasses the LCM substrate's bit-length-quadratic merge cost. Run via `scripts/bench/runners/merkle_vs_lcm.py` (50 samples per N, Darwin arm64 / Python 3.10):

| N | Merkle verify p50 | LCM `state % p` p50 | speedup | LCM state bit-length | proof size |
|---:|---:|---:|---:|---:|---:|
| 100    | 4.6 µs | 3.2 µs | **0.7×** | 6 236     | 220 B |
| 1 000  | 5.8 µs | 29.6 µs | **5.15×** | 62 496   | 320 B |
| 5 000  | 7.2 µs | 151.2 µs | **21.1×** | 312 709  | 403 B |

The sidecar's verify cost is empirically flat across the range tested (4.6 → 7.2 µs as N grows 50×) — proof length grows as `log₂(N)` and SHA-256 is constant-time per block, so verify is `O(log N)` hash operations. The LCM divisibility cost grows linearly with state bit-length, which itself grows linearly with N at SUM's prime sizes, giving the substrate-level `O(N)` per-check seen in the table. The crossover where Merkle wins is N ≈ 100 — below that, LCM modulo on small integers is faster than 14 SHA-256 invocations; above it, the substrate-quadratic merge cost dominates.

**N=10 000 row deliberately omitted.** A run on the reference laptop forces the runner's `--skip-lcm-build-at=10000` proxy mode (the full LCM build takes minutes), which would substitute a 62 k-bit divisor for the real ~625 k-bit divisor and produce a misleading 3.95× footnoted "speedup." The 5 000-row figure is the honest production-relevant headline: **21× faster membership verify at N=5 000 with no LCM build required.** A real N=10 000 measurement is gated on running the bench on a host that can hold the full LCM in memory and time-budget; until that lands, no number for that N appears here.

The success criterion the M1 playbook entry names — *"log-size inclusion proofs that verify materially faster than the LCM divisibility path"* — is met at N=5 000 with a **21×** advantage that has been measured on real LCM state, not a proxy. Production wiring (emitting `merkle_root` in `CanonicalBundle` / render receipts, pinned behind a `bundle_version` minor bump per `docs/COMPATIBILITY_POLICY.md`) is gated on review of this measurement plus the leaf-format spec lock.

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

**Boundary:** the prior boundary paragraph hypothesised that **canonicalisation alone** (entity resolution + predicate lemma-fold + pinned vocabulary) could move the 0.12 exact-match recall upward "without changing the generator." That hypothesis has now been measured offline against the cached per-doc data and **is partially refuted**.

**§2.5 canonicalisation-replay receipt (`scripts/bench/runners/canonicalization_replay.py`, run 2026-04-28, no LLM cost — operates on the cached `bench_history.jsonl` per-doc `missing_claims` / `extra_claims` arrays, output at `fixtures/bench_receipts/s25_canonicalization_replay_2026-04-28.json`):**

| Regime | Rules | drift_pct (mean) | exact-match recall (mean) | docs at full recall |
|---|---|---:|---:|---:|
| **L0 baseline** | none — sanity check | **107.75** | **0.12** | 6 / 50 |
| **L1 predicate-only** | lowercase + strip aux prefixes (`was_`, `has_`, …) + strip prep suffixes (`_in`, `_to`, …) + strip verb inflection (`-s`, `-ed`, `-ing`) | **107.75** | **0.12** | 6 / 50 |
| **L2 + subject** | L1 + lowercase / underscore→space / last-word-as-key on subjects (catches `albert_einstein` ≈ `einstein`) | **106.68** | **0.16** | 8 / 50 |
| **L3 aggressive** | L2 + lowercase / strip articles / first-content-word-as-key on objects (CEILING — risks conflating distinct facts) | **106.36** | **0.18** | 9 / 50 |

**The L1 row is the falsification.** Predicate-only canonicalisation moves zero exact matches because the cached `missing_claims` for failed docs do not have a paraphrase pair in `extra_claims` whose only difference is predicate inflection. The dominant failure mode is **generator elaboration**: the LLM produces an average of 12 reconstructed axioms per source axiom and *elaborates around the source claim rather than paraphrasing it*. There is nothing for predicate normalisation to canonicalise *to*.

L2 (subject canonicalisation) recovers 2 docs — exactly the `newton`/`isaac_newton`-shape cases the prior boundary paragraph cited. L3 (aggressive object collapse) recovers 1 more, at the cost of a regime that conflates legitimately-distinct facts and is reported as the ceiling, not a recommendation.

**The receipt:** canonicalisation alone moves exact-match recall by **at most 0.06 absolute (0.12 → 0.18)** under maximally aggressive rules. The headline drift_pct moves only **1.4 points** because the formula is dominated by `|reconstructed| >> |source|` regardless of key alignment. Closing the §2.5 gap requires moving the *generator*, not just the post-hoc key normaliser.

**§2.5 generator-side intervention receipt** (`scripts/bench/runners/s25_generator_side.py`, run 2026-04-28, live OpenAI API ~$0.20, output at `fixtures/bench_receipts/s25_generator_side_2026-04-28.json`). Two interventions, three ablations against the same `seed_v1` corpus and pinned model (`gpt-4o-mini-2024-07-18`):

- **Canonical-first generator prompt** — system prompt requires the generator to surface each source claim verbatim ("The {s} {p} {o}.") *before* elaborating. Pure prompt change.
- **Constrained-decoding extractor** — per-doc Pydantic schema with `Literal` enums pinned to source-axiom vocabulary (subject ∈ source_subjects, predicate ∈ source_predicates ∪ canonical_padding, object ∈ source_objects). OpenAI structured-output enforces the constraint at the API.

| Ablation | drift_pct (mean) | exact-match recall (mean) | recall p10 | docs at full recall |
|---|---:|---:|---:|---:|
| **L0 baseline** (no intervention) | 107.75 | 0.12 | 0.00 | 6 / 50 |
| L3 max canonicalisation (post-hoc) | 106.36 | 0.18 | 0.00 | 9 / 50 |
| A — canonical-first generator only | 94.85 | 0.60 | 0.00 | 30 / 50 |
| B — constrained extractor only | 81.97 | 0.62 | 0.00 | 31 / 50 |
| A + B combined (initial) | 21.00 | 0.90 | 1.00 | 45 / 50 |
| **A + B combined + lemma-exclusion fix** | **0.00** | **1.0000** | **1.00** | **50 / 50** |

**The §2.5 gap is closed on `seed_v1`.** The combined intervention with the lemma-exclusion fix takes exact-match recall to **1.00 across all 50 documents** and drift to **0.00%** — within rounding of the canonical (provable) round-trip on the same corpus.

**Each layer's contribution:**
1. **Canonical-first generator alone:** 0.12 → 0.60. Addresses generator elaboration at the source.
2. **Constrained extractor alone:** 0.12 → 0.62. Addresses surface-form drift at the symptom.
3. **Combined (initial):** 0.90. Two layers compose because they operate on different stages of the failure mode.
4. **Combined + lemma-exclusion fix:** 1.00. The remaining 5/50 residual was a single root cause: the constrained extractor's enum allowed both the source's inflected predicate (`proposed`) and its lemma (`propose`) from the canonical-padding set, and the LLM picked the lemma every time. Removing lemmas of source predicates from the canonical-padding set forces the LLM to use the source surface form. Receipt at `fixtures/bench_receipts/s25_residual_closure_2026-04-28.json`.

**Scaling check on `seed_v2`** (`fixtures/bench_receipts/s25_generator_side_seed_v2_2026-04-28.json`, run 2026-04-28, ~$0.12). The seed_v2 corpus is 20 difficulty-pattern docs across 7 parse patterns (apposition, passive voice, relative clause, conjunction, negation, hedging, complex PP) including multi-fact docs. Same combined intervention, same pinned model:

| Ablation | drift_pct (mean) | exact-match recall (mean) | recall p10 | docs at full recall |
|---|---:|---:|---:|---:|
| canonical_first only | 98.92 | 0.5750 | 0.00 | 11 / 20 |
| constrained_extractor only | 52.08 | 0.8250 | 0.00 | 16 / 20 |
| **combined** | **5.00** | **0.9750** | **1.00** | **19 / 20** |

**The intervention pattern scales.** Combined goes from `seed_v1`'s 1.00 to `seed_v2`'s 0.9750 — a 0.025 absolute drop on a corpus that includes apposition, passive, relative-clause, conjunction, negation, hedging, and complex-PP parse patterns plus multi-fact docs. The single failing doc (doc_015, "Alice and Bob visited Paris.") is **not an intervention failure**: the runner's first-pass `_baseline_extract` returned a malformed source axiom for that doc (`['alice', 'visited', 'paris},{']`); the combined ablation correctly preserved that corrupted source through the round-trip. The fail-mode is an LLM extraction artifact on the source pass, not the intervention.

**Note the per-ablation curve has the opposite shape vs `seed_v1`:** on seed_v2, constrained_extractor alone (0.8250) beats canonical_first alone (0.5750), where on seed_v1 they were nearly identical (0.62 vs 0.60). The reason is corpus form: seed_v2's source predicates are mostly already in lemma form (`win`, `emit`, `orbit`, `visit`), so the lemma-exclusion fix has less work to do and the LLM extractor naturally selects the source form. Conversely, seed_v1's predicates are mostly inflected (`proposed`, `contains`, `discovered`), so the canonical-first generator prompt was the load-bearing fix there. Different corpora, different layers earn their keep — but **combined wins decisively on both** (1.00 on seed_v1, 0.975 on seed_v2).

**Capstone scaling check on `seed_long_paragraphs`** (`fixtures/bench_receipts/s25_generator_side_seed_long_combined_2026-04-28.json`, run 2026-04-28, ~$0.10). The seed_long_paragraphs corpus is 16 hand-authored multi-paragraph documents on disparate technical/historical topics (internet history, quantum mechanics, evolution, Roman empire, solar system, neural networks, climate change, relativity, French revolution, cell biology, WWII, periodic table, human genome, cryptography, industrial revolution, immune system). Source-axiom counts per doc: **11–28** (vs. 1–2 on seed_v1/v2 — an order of magnitude denser).

| Ablation | drift_pct (mean) | exact-match recall (mean) | recall p10 | docs at full recall |
|---|---:|---:|---:|---:|
| canonical_first only † | 69.36 | 0.7045 | n/a | 4 / 16 |
| constrained_extractor only ‡ | n/a | n/a | n/a | n/a |
| **combined** | **0.57** | **0.9972** | **1.00** | **15 / 16** |

† Recorded from a partial earlier run (the all-3-ablations sweep hung on a network call after constrained_extractor's 8th doc; canonical_first had completed cleanly first). The 0.7045 figure is informative — canonical_first alone *improves* on long-form vs seed_v2's 0.5750, suggesting longer prose with denser source axioms gives the LLM more context to anchor the canonical sentences.

‡ Constrained_extractor alone was not measured to completion on seed_long for the same reason. The 8 docs that completed before the hang trended ~0.40 recall — a substantial drop from seed_v2's 0.825 because long-form prose makes the per-doc constrained vocabulary wider and noisier, and the LLM emits far fewer triples under constraint than the unconstrained extractor does (averaged ~10 vs ~25 per doc). This is the only ablation that visibly degrades on long-form. **It does not propagate to the combined ablation.**

**Combined supra-additive composition holds on multi-paragraph multi-fact prose.** Combined recall on seed_long is **0.9972** — within rounding of seed_v1's 1.00 and effectively matching seed_v2's 0.9750. The single failing doc (doc_long_solar_system, recall = 0.9545) is **not an intervention failure**: the LLM source-extract produced two semantically-overlapping axioms `[mars, has_two_moons, phobos and deimos]` and a separate near-paraphrase that admitted `has_known_moons` into the constrained vocabulary; the round-trip's reconstructed extractor picked `has_known_moons` over `has_two_moons` for one mars axiom. 21 of 22 axioms in that doc round-tripped exactly; the one drift is a benign upstream duplication, not a structural failure of the intervention.

**Cross-corpus comparison — the §2.5 closure scales universally:**

| Corpus | n_docs | axioms/doc | combined recall | drift_pct | full recall |
|---|---:|---:|---:|---:|---:|
| seed_v1 (single-fact SVO) | 50 | 1 | 1.0000 | 0.00 | 50 / 50 |
| seed_v2 (7 difficulty parse patterns + multi-fact) | 20 | 1–2 | 0.9750 | 5.00 | 19 / 20 |
| **seed_long (16-topic multi-paragraph)** | **16** | **11–28** | **0.9972** | **0.57** | **15 / 16** |

The combined intervention lands ≥ 0.97 recall and ≤ 5 % drift on every measured corpus shape — single-fact short-form, multi-fact difficulty-pattern, and multi-paragraph dense-prose. **The §2.5 closure is corpus-independent.** The remaining gap on each corpus traces to upstream LLM source-extraction artifacts (corrupted axioms on seed_v2 doc_015, semantically-duplicate axioms on seed_long solar_system), not to the intervention pattern itself.

**Status:** §2.5 closed across all measured corpora. The §6 row in the progress table reflects this. The intervention pattern (canonical-first generator + constrained-decoding extractor + lemma-exclusion of source-predicate lemmas from the canonical-padding set) is the load-bearing engineering finding; the receipt artifacts are the durable proof.

**Status update:** the §2.5 row in §6's progress table moves from `Measured (drift = 107.75%, recall = 0.12)` to `Closed on seed_v1 (combined intervention with lemma-exclusion fix: drift = 0.00%, recall = 1.0000, 50/50 docs at full recall)`. The unprompted pipeline's 107.75/0.12 numbers stand as the *baseline measurement under no intervention*; the post-fix numbers are the load-bearing result.

The receipt schema is `sum.s25_generator_side.v1` (with sibling per-ablation schemas `sum.s25_canonical_first_generator.v1`, `sum.s25_constrained_extractor.v1`, `sum.s25_combined.v1`). Receipts compare cleanly to the prior `sum.s25_canonicalization_replay.v1` receipt — same `seed_v1` corpus, same pinned model, same `n_docs = 50`. Reproducible: `python -m scripts.bench.runners.s25_generator_side --ablation all --out <path>` (requires `OPENAI_API_KEY`).

### 2.5.1. `/api/qid` Resolution Accuracy Floor

The hosted Worker's `/api/qid` route resolves free-text terms to Wikidata QIDs via `wbsearchentities`. The README's "Future developments" historically claimed a "target >95% accuracy floor" but the floor was never measured. Closed 2026-04-28 by `scripts/bench/runners/qid_accuracy.py` against a 30-term hand-curated corpus (8 people / 8 places / 8 concepts / 6 common nouns).

**Receipt** (`fixtures/bench_receipts/qid_accuracy_2026-04-28.json`, schema `sum.qid_resolution_accuracy.v1`, ~$0 cost):

| Metric | Result | Denominator |
|---|---:|---|
| Hit-rate (any non-null QID returned) | **30 / 30 = 1.0000** | all terms |
| Label-substring match | **24 / 24 = 1.0000** | pattern-matchable terms (excludes 6 common-noun rows) |
| Wall-clock p50 | ~200 ms / term | — |

**Boundary on this measurement.** Label-substring match is robust to wbsearchentities's quirks but does not measure semantic accuracy against canonical Q-IDs. The receipt records `relativity` → `Q201607 (Relativity Records)` — a music-label entity, not the physics theory — as a passing label-substring match. The two-tier metric is the floor; canonical-QID accuracy would require hand-verified ground-truth pairs (a follow-on, scoped explicitly in the README). The current resolver is a thin layer over wbsearchentities; SPARQL-driven disambiguation that prefers the most-linked-to entity for ambiguous terms remains an unshipped enhancement — the `relativity` row demonstrates exactly the case SPARQL would address.

Reproducible: `python -m scripts.bench.runners.qid_accuracy --out <path>` (no API key needed).

### 2.6. Slider Axis Fact-Preservation (Phase E.1 v0.4 → v0.7)

The slider's load-bearing claim — *axis changes do not lose facts* — has been **empirically verified** across two independently-authored corpora, a four-layer fact-preservation substrate, and a deterministic prompt-hardening mechanism that closed the catastrophic-failure mode v0.6 surfaced. [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) is the canonical contract document; this section pins the load-bearing numbers as `empirical-benchmark` and links the failure-mode arc.

**Bench harness measurements:**

| Run | Corpus | LLM-axis cells (excl. density) | Median | p10 | Min | NLI rescue rate | Real losses | Catastrophic outliers (≥5) |
|---|---|---|---|---|---|---|---|---|
| v0.4 | `seed_paragraphs.json` (n=8 short, 4–12 triples/doc) | 160 | **1.000** | **0.818** | 0.727 | 100 % (186/186) | **0** | 0 |
| v0.6 (no hardening) | `seed_long_paragraphs.json` (n=16 long, 9–24 triples/doc) | 320 | 1.000 | 0.769 | 0.111 | 95.7 % (800/836) | 36 | **2** |
| v0.7 (`FACT_PRESERVATION_REINFORCEMENT`) | same long bench | 319 | 1.000 | 0.750 | **0.700** | 99.8 % (653/654) | **1** | **0** |

**Reading the v0.7 row:** the v0.7 p10 (0.750) sits slightly below v0.6's (0.769) despite catastrophic outliers being eliminated and the floor lifting (0.111 → 0.700). This is distribution-shape, not regression: the reinforcement clause makes the LLM's surface forms more defensive, so the strict embedding-similarity layer triggers NLI audit on more cells. Audit then rescues every flagged fact (99.8 % rate); cells move from "1.000 by semantic alone" to "1.000 with NLI confirmation," which sits in the 0.7–0.99 band on the strict score. Net: 1 confirmed loss across 654 audit calls, catastrophic outliers gone, p10 nominally lower because the perfect-cells share narrowed (60 % → 52 %). [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) §"Headline result" describes the same trade in product terms.

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

### 2.9. Sheaf-Laplacian Hallucination Detector (v1 / v2.x / v3 / v3.1 / v3.2 / Sprint-7.5 complementary hybrid)

**Claim:** Sheaf-Laplacian quadratic forms over render manifolds detect specific perturbation classes (A1 entity-swap, A2 predicate-flip, A3 off-graph fabrication, A4 triple-drop) at measurable AUC on the `seed_long_paragraphs` corpus (16 docs, 120 source triples). v3 receipt-weighted detector improves on v2.2 baseline; v3.1 boundary deviation has a **structural blind spot** at corpus scale; **v3.2 closes the F3 STRUCTURAL FAIL at the detector layer** by combining v3's weighted Laplacian energy with v3.1's harmonic-extension deviation as a complementary signal. Sprint 7.5 surfaced a *second* structural finding (the cochain-on-source-graph is mathematically blind to entity-set-preserving perturbations) and recovered the detector's competitive position via Borda fusion of (v3.2 + per-rendered-triple V) with B2 entity-set jaccard — the **complementary hybrid** that strictly beats trivial baselines on this corpus's full perturbation space.

**Math foundation:** Hansen-Ghrist 2019 (arXiv:1808.01513), Gebhart-Hansen-Schrater 2023 (AISTATS, arXiv:2110.03789). Mechanically pinned: symmetric-PSD Laplacian, factored quadratic form equivalent to materialized matrix form, ‖δx‖² = x^T L x to floating-point precision; v3.2 subsumption (γ=0 ≡ v3) byte-identical; H16–H20 universal-quantifier claims via Hypothesis property tests. Total research suite covering math + falsifiability **88 / 88 tests pass** (across `Tests/research/test_sheaf_laplacian_v{2,3,32}.py` + `test_sheaf_laplacian_v32_property.py`).

**Measurement (PR #114 v2.2 ROC bench, PR #124 v3 ROC bench, PR #125 F3 diagnostic, PR #127 v3.2 closure, PR #135 substrate-determinism rebase):**

| Detector | A1 | A2 | A3 | A4 | Notes |
|---|---|---|---|---|---|
| v2.2 baseline | 0.62 | 0.50 | 1.00 | 0.86 | A2 predicate-flip is at chance — known v2.x weakness; needs predicate-perturbation negative sampling in training |
| v3 receipt-weighted | 0.66 | 0.50 | — | **0.97** | **+10.7% AUC on A4** is the standout. F1 (trusted-side amplification): MARGINAL (Δ≈+0.034 mean AUC under post-Sprint-1 substrate). F2 (no untrusted collapse): PASS. |
| v3.1 boundary deviation | 0.50 | 0.50 | — | 0.50 | **F3 STRUCTURAL FAIL** (PR #125 diagnostic): 8 / 8 cells of the 2×2×2 hypothesis sweep FAIL. Mathematical blind spot for boundary perturbations when `L_IB = 0`. |
| **v3.2 combined** (γ=0.1) | — | — | — | — | **F4 PASS / F5 PASS at γ ≤ 0.1.** Trusted-mean AUC = **0.659** (vs v3 = 0.663; Δ = −0.004, well within F5's −0.02 tolerance). At γ = 0 reduces byte-identically to v3 (subsumption). At γ → 1.0 the magnitude-matching auto-calibration overweights deviation and F5 fails (Δ = −0.028); optimal γ on this corpus is small. |
| **Sprint-7.5 baselines** (B1 / B2) | 0.97 / 1.00 | 0.50 / 0.50 | — | 1.00 / 1.00 | Trivial entity-set baselines: B1 = 1 − recall of source entities in render; B2 = 1 − jaccard(source_entities, rendered_entities). B2 trusted-mean AUC = **0.833** (across A1+A2+A4 trusted cells). v3.2-cochain-only loses to baselines by Δ = −0.174 — caught BEFORE arXiv submission by the substrate's truth-first discipline. |
| **v3.2 + per-triple** | 0.70 | 0.67 | — | 0.91 | A2 LIFTED 0.500 → 0.671 by integrating the §3.5 per-rendered-triple V channel (Gebhart 2023 §4) — the channel v2.2 §4.3 ROC bench used to hit A1/A2/A3 = 1.000 but the v3.x cochain arc had dropped. Cochain channel alone is *mathematically blind* to entity-set-preserving perturbations because predicate doesn't enter the cochain; per-triple channel scores (h, r, t) directly under F_h(r), F_t(r). Trusted-mean AUC = 0.759. Still loses to B2 alone, but informative about *why*: only detector that catches A2. |
| **Complementary Borda hybrid** ★ | 0.97 | 0.67 | — | 0.99 | Borda(v3.2 + per-triple, B2). **WINS at trusted-mean AUC = 0.876, Δ = +0.043 vs B2 alone** — above the +0.030 "real win" threshold. Two detectors structurally complementary: B2 catches entity-set-changing perturbations (A1/A4) at 1.000; v3.2 + per-triple is the only signal that catches A2. Borda preserves both contributions. **HYBRID_BEATS_BASELINE verdict.** Receipt: `fixtures/bench_receipts/complementary_hybrid_2026-05-04.json` (`bench_digest dc6e0260…`). |

**F1 / F2 / F3 / F4 / F5 verdict ladder pinned across the receipts:**
- **F1** (v3 trusted-mean ≥ v2.2 trusted-mean by ≥ 0.04): MARGINAL — Δ=+0.034 post-Sprint-1.
- **F2** (no class drops by >0.10 from v22 to v3 on untrusted): PASS — 0 / 3 classes collapse.
- **F3** (v3.1 standalone deviation trusted-mean ≥ 0.55): **FAIL by mathematical necessity** — 0.499. The `L_IB = 0` topology defeats the harmonic-extension signal; this is structural, not parametric.
- **F4** (v3.2 combined trusted-mean ≥ 0.55): **PASS at every γ tested** (γ ∈ {0, 0.1, 1.0, auto}) — 0.635 to 0.663.
- **F5** (v3.2 trusted-mean ≥ v3 trusted-mean − 0.02): **PASS at γ ≤ 0.1**, FAIL at γ > 0.5. Calibration finding: deviation's signal-to-noise ratio is below what its magnitude suggests; on this corpus it functions as a small modulator, not a co-leader.

**Boundary — what is and is not proved:**
- **Math primitives** (symmetric-PSD, factored equality, weighted form linearity, harmonic-extension defining property H7, v3.2 subsumption / fall-back / no-λ-double-counting H16–H20) are mechanically pinned in §1-equivalent fashion within the `[research]` extras. The library API is documented at [`docs/SHEAF_LIBRARY_API.md`](SHEAF_LIBRARY_API.md) (PR #138).
- **Detection AUC** is measured on a single fixed corpus (n=16). The numbers above are not generalised claims; they apply to `seed_long_paragraphs` under the current trained sheaf, λ_auto, and seed-controlled perturbation harness. Cross-corpus generalisation is unmeasured.
- **F3 STRUCTURAL FAIL of v3.1 standalone deviation** is named explicitly and remains true at the v3.1 layer — the harmonic-extension signal is mathematically blind under `L_IB = 0` topology. v3.2 closes the *detector-layer* problem by combining with v_laplacian_w (which is informative anywhere on the graph); it does NOT make v3.1's standalone-deviation signal informative.
- **A2 predicate-flip is at chance across every cochain-channel detector** (v22 / v3 / v3.1 / v3.2). Sprint 7.5 confirmed this is *structural* not parametric — adding predicate-perturbation negatives to training (`aa34b6e8…` digest pin) does NOT lift A2 because the cochain construction discards predicate information. A2 detection is recovered by integrating the §3.5 per-rendered-triple V channel (`7025436f…` digest pin lifts A2 to 0.671); the **complementary hybrid** combines this channel with B2 entity-set jaccard via Borda fusion to beat trivial baselines (`dc6e0260…` digest pin, trusted-mean 0.876).
- **Sprint-7.5 recovery-experiment digests** (all pinned in [`Tests/research/test_recovery_experiment_digests.py`](../Tests/research/test_recovery_experiment_digests.py); each reproduces 3× in fresh procs unconditionally):
  - `a7965803…6c2003` — Borda(v3.2_only, B2) loses (Δ=−0.025)
  - `aa34b6e8…c866e7` — predicate-perturbation training fails to lift A2
  - `7025436f…fd4fa` — per-triple integration lifts A2 from 0.500 to 0.671
  - `dc6e0260…343ce` — complementary Borda(v3.2 + per-triple, B2) WINS (HYBRID_BEATS_BASELINE)
- **Receipts:** [`fixtures/bench_receipts/v3_roc_bench_2026-05-03.json`](../fixtures/bench_receipts/v3_roc_bench_2026-05-03.json), [`fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-03.json`](../fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-03.json), [`fixtures/bench_receipts/v3_2_validation_2026-05-03.json`](../fixtures/bench_receipts/v3_2_validation_2026-05-03.json), and the Sprint-7.5 set (`baseline_comparison_2026-05-04.json`, `hybrid_comparison_2026-05-04.json`, `predicate_negatives_experiment_2026-05-04.json`, `per_triple_integration_2026-05-04.json`, `complementary_hybrid_2026-05-04.json`, `cross_machine_verification_2026-05-04.json`). Each carries per-cell AUCs and a `bench_digest` SHA-256 (JCS-canonical, RFC 8785) for reproducibility. Post-Sprint-1 (PR #135) the digests reproduce **unconditionally** across Python invocations; Sprint 7.5 §2.10 below extends the claim cross-machine.

### 2.10. Bench Reproducibility Digest (`bench_digest`, PR #125 + PR #135 substrate-determinism rebase)

**Claim:** Quantized JCS-canonical SHA-256 over a research-bench `DiagnosticReport` is byte-stable across runs on the same machine + same code. Post-Sprint-1 (PR #135) reproducibility is **unconditional** — no environment-variable manipulation required. LAPACK threading inside `np.linalg.lstsq` introduces ~±0.02 AUC jitter; quantization to 3 decimals on AUCs and 4 on diagnostic floats absorbs it.

**Proof mechanism:** Three layers compose to give the unconditional reproducibility:

1. **In-process digest stability** — [`Tests/research/test_sheaf_v3_1_f3_diagnostic.py::test_v3_1_f3_diagnostic_digest_is_quantization_stable`](../Tests/research/test_sheaf_v3_1_f3_diagnostic.py) runs the diagnostic twice in-process, asserts `report1.bench_digest == report2.bench_digest`. Pinned since PR #125.

2. **Cross-invocation digest stability** — three benches (`scripts/research/sheaf_v3_roc_bench.py`, `sheaf_v3_1_f3_diagnostic.py`, `sheaf_v3_2_validation.py`) each produce **identical `bench_digest`** when run three times in fresh Python processes without `PYTHONHASHSEED=0` (verified during PR #135). The earlier `reproducibility_requires: "PYTHONHASHSEED=0"` field on the v3.2 receipt was removed when the load-bearing site (`DeterministicSieve.extract_triplets` returning `list(set(triplets))` whose iteration order was hash-randomized) was fixed by `sorted(set(triplets))`.

3. **Canonicalization** — JCS canonicalization is the project's own [`sum_engine_internal/infrastructure/jcs.py`](../sum_engine_internal/infrastructure/jcs.py) (RFC 8785 — the same canonicalization the trust loop uses for `CanonicalBundle` and `render_receipt.v1`). The digest is computed over a quantized payload; quantization is what makes the digest stable across LAPACK jitter. The on-disk receipt's `bench_digest` field documents the value an external reader can verify.

**Current digests (post-Sprint-1):**
- `fixtures/bench_receipts/v3_2_validation_2026-05-03.json` — `b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a`
- `fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-03.json` — `62b6e1878d1d12f36eb80e301304854a1a2c03386f0e872850d3461b2f733e7c`
- `fixtures/bench_receipts/v3_roc_bench_2026-05-03.json` — no digest field; AUCs reproducible directly via deterministic sieve

**Boundary:** Three intended uses, each with its own surface:
1. **Reproducibility canary** — same machine, same code → same digest. Drift indicates upstream change. **Unconditional** (no env-var requirement).
2. **Cross-machine witness** (Sprint 7.5 — measured). The v3.2 validation digest (`b4d26c01…`) and the complementary-hybrid digest (`dc6e0260…`) **reproduced byte-for-byte** when re-run on Modal x86_64 (Linux 4.4 / glibc 2.31 / Python 3.10.8 / numpy 1.25.0 / OpenBLAS-via-PyPI / AVX2 SIMD) at the pinned commit SHA. The substantive `HYBRID_BEATS_BASELINE` verdict also reproduces cross-machine. Apple Accelerate (operator's Apple Silicon) and OpenBLAS (Modal x86_64) are two distinct LAPACK environments — the digest's reproducibility property holds across the LAPACK boundary that previously made cross-machine reproducibility "unmeasured." Receipt: [`fixtures/bench_receipts/cross_machine_verification_2026-05-04.json`](../fixtures/bench_receipts/cross_machine_verification_2026-05-04.json), schema `sum.cross_machine_verification.v1`. Verification harness: [`scripts/research/cross_machine_verify_modal.py`](../scripts/research/cross_machine_verify_modal.py) — any reader with Modal credits can rerun via `modal run` against the pinned SHA and verify both digests match. Cross-machine reproducibility beyond Apple Accelerate + OpenBLAS-via-PyPI (e.g. Intel MKL, ARM Linux, AMD AOCL) is unmeasured (v0.2 candidates).
3. **Cross-runtime witness** — when (if) a future Node/browser port of v3 / v3.1 / v3.2 reproduces these AUCs, the matching digest is the K-style portability proof. Not yet measured (no Node port exists for these detectors).
4. **Signable bench artifact** — the digest can be Ed25519-signed with the project's existing JWKS keys; the arXiv preprint at `docs/arxiv/sheaf-detector-note-v0.md` (v0.1 + Sprint 7.5) cites the digests as anchors; readers re-running the bench (locally or via the published Modal harness) verify their digest matches.

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
| Property-graph backing store for corpora above ~10k axioms (prime encoding demoted to signed witness) | Design direction confirmed by §2.2 measurements (merge cost makes prime-as-primary-query unviable above low-thousands); implementation not started | Phase 26 |
| Formal-method verification (TLA+ specs, SMT-solver consistency proofs, α,β-CROWN neural-net verification) | Design notes archived at [`docs/archive/FORMAL_MODELS.md`](archive/FORMAL_MODELS.md). Not started; gated on a downstream consumer that requires `certified` epistemic status (§5) rather than `provable` or `empirical-benchmark`. | Future |

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
| Round-trip conservation (LLM narrative prose, full loop) | **Closed across measured corpora** | Baseline (2026-04-19): drift = 107.75 %, recall = 0.12. Combined intervention (canonical-first generator + constrained-decoding extractor with `Literal`-enum vocab pin + lemma-exclusion of source-predicate lemmas from canonical-padding): seed_v1 (50 docs, single-fact SVO) **recall 1.0000 / drift 0.00 / 50-of-50**; seed_v2 (20 docs, difficulty patterns + multi-fact) **recall 0.9750 / drift 5.00 / 19-of-20**; seed_long_paragraphs (16 docs, 11–28 axioms each) **recall 0.9972 / drift 0.57 / 15-of-16**. Each remaining gap is an upstream source-extraction artifact, not an intervention failure. Per-ablation breakdowns + receipts in §2.5. |
| Extraction ceiling investigation (en_core_web_trf upgrade or LLM fallback) | seed_v1 at F1 = 1.000 (no remaining failures); seed_v2 at F1 = 0.762 with precision = 1.000 — every remaining failure is a RECALL miss not a TRUTH inversion (apposition secondary, relative-clause subordinate, compound non-head conjuncts). **Strategic placeholder:** decision deferred until §2.5 LLM round-trip drift attack lands. Kill condition: §2.5 work resolves whether the LLM-as-extractor path is the right fix (in which case the trf upgrade is dropped) or whether the sieve needs to stay primary (in which case trf is the right next step). Decision required before any further sieve-recall work. | Gated on §2.5 |
| Sliding-scale rendering parameters | **Shipped end-to-end** — 5 axes (density / length / formality / audience / perspective). Density actioned deterministically via lexicographic axiom subsetting. Length / formality / audience / perspective LLM-conditioned via the Cloudflare Worker render path (`worker/src/routes/render.ts`, Anthropic provider, optional CF AI Gateway). Fact-preservation verified at scale (§2.6); robustness layered (§2.7). Every render carries a signed receipt (§1.8). | **Measured + cryptographically attested** |
| Cryptographic attestation | Working, cross-runtime | Ed25519 + HMAC-SHA256 + Merkle chain. Ed25519 verified in all three shipping runtimes against the same bundle bytes: Python (`sum verify`), Node (`standalone_verifier/verify.js` via WebCrypto), Browser (`single_file_demo/index.html` via SubtleCrypto). Locked by cross-runtime K3/K4 harness + A1–A6 adversarial-rejection matrix in CI. |
| Per-render attestation (Phase E.1 v0.9.A) | **Shipped** | `sum.render_receipt.v1` — Ed25519 (RFC 8032) over JCS-canonical (RFC 8785) payload bytes, wrapped as detached JWS (RFC 7515 §A.5) with public keys distributed via JWKS (RFC 7517) at `/.well-known/jwks.json`. Active kid `sum-render-2026-04-27-1`. Cryptographic binding documented in §1.8; full wire spec in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md). v0.9.B (browser verifier) and v0.9.C (Python verifier) are queued in [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md); they will close the negative-path proof across runtimes. |
| Epistemic-status labeling | Shipped v1.2.0 | See §5 |
| SHACL structural validation (Polytaxis Bucket A) | Not yet | Phase 25 |
| Conformal prediction confidence (Polytaxis Bucket A) | Algorithm shipped (`sum_engine_internal/ensemble/venn_abers.py`, 18 tests); **production wiring via `ConfidenceCalibrator.calibrate_interval()` shipped** with `load_venn_abers_fixture()` helper and fixture tests; calibration-set authoring is the remaining step | Phase 25 |
| VC 2.0 `eddsa-jcs-2022` emission + verification | **Shipped** (`sum_engine_internal/infrastructure/verifiable_credential.py` + pure-Python RFC 8785 JCS at `sum_engine_internal/infrastructure/jcs.py`, 58 tests: 30 JCS + 28 VC covering sign/verify round-trip, tamper detection, JSON-on-disk persistence, multibase base58btc round-trip, key-reordering resilience) | Phase 25 |

The gap from `current honest state` to `ultimate goal` is the refactor roadmap of record. Any PR that claims to close part of this gap must update this section with the new measurement.
