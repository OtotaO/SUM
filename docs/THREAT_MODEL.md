# Threat Model

**Version:** 1.3.0
**Date:** 2026-04-20

This document describes what the SUM engine's security and integrity mechanisms protect against, and — critically — what they do NOT protect against.

---

## 1. Trust Architecture

### 1.1. Current Model: Shared-Secret Peers

The current system uses **HMAC-SHA256** for bundle signatures. This implies a **symmetric trust model**: both the producer and consumer of a bundle must share the signing key.

```
Producer ──(shared key)──> Bundle ──(shared key)──> Consumer
```

### 1.2. Trust Assumptions

| Assumption | Status |
|-----------|--------|
| Producer and consumer share a key | **Required** |
| Transport channel integrity | **Not assumed** (HMAC covers this) |
| Third-party verification | **Supported** via Ed25519 (self-asserted key) |
| Key secrecy | **Assumed** (key rotation + archive available) |

---

## 2. What Is Protected

### 2.1. Bundle Tampering (✅ Protected)

**Threat:** An attacker intercepts a bundle in transit and modifies the canonical tome, state integer, or timestamp.

**Defense:** HMAC-SHA256 signature covers `canonical_tome|state_integer|timestamp`. Any modification invalidates the signature. The importer rejects bundles with invalid signatures.

**Residual risk:** None, assuming the HMAC key is not compromised.

### 2.2. State Integer Forgery (✅ Protected within trust boundary)

**Threat:** An attacker crafts a bundle with a state integer that doesn't match the canonical tome.

**Defense:** The witness verifier independently reconstructs the state from the canonical tome and compares. Mismatch is detectable without the HMAC key.

### 2.3. Version Mismatch (✅ Protected)

**Threat:** A consumer receives a bundle produced with an incompatible canonical format version.

**Defense:** The `canonical_format_version` field is checked. Unsupported major versions are rejected. See `COMPATIBILITY_POLICY.md`.

### 2.4. Malformed Bundles (✅ Protected)

**Threat:** A consumer receives a structurally invalid bundle (missing fields, corrupt JSON, invalid types).

**Defense:** Required fields are validated before processing. Missing fields raise explicit errors.

---

## 3. What Is NOT Protected

### 3.1. Public Authenticity (✅ Now Protected)

**Threat:** A third party wants to verify that a bundle was produced by a specific author.

**Defense:** Ed25519 public-key signatures. The 32-byte public key is embedded in the bundle. Any party with the public key can verify provenance without the HMAC secret.

**Residual risk:** The embedded public key is self-asserted. A trust-on-first-use (TOFU) model or certificate authority is needed for strong identity binding. The current system proves "this bundle was signed by the holder of this private key" but not "this private key belongs to entity X."

### 3.2. Key Compromise (✅ Now Protected)

**Threat:** The HMAC signing key or Ed25519 private key is leaked or stolen.

**Defense:** Key rotation via `KeyManager.rotate_keypair()`. Old keys are archived to `keys/rotated/` with microsecond timestamps. `list_trusted_public_keys()` returns all historical keys for verifying bundles signed by rotated keys.

**Residual risk:** There is no real-time revocation mechanism. If a key is compromised, bundles signed with it remain valid until the operator manually invalidates them. A future PKI or key revocation list (KRL) would address this.

### 3.3. Extraction Manipulation (⚠️ Partially Protected)

**Threat:** An attacker crafts input text designed to produce misleading extractions (adversarial NLP).

**Impact:** The sieve may extract incorrect or misleading axioms. These become part of the canonical state and appear legitimate.

**Defense (Phase 19A + bench harness):** `ExtractionValidator` provides structural gating: non-empty fields, length bounds, control character rejection, JSON fragment rejection, predicate canonicalization, and batch deduplication. 25 unit tests verify gating logic. A 50-document golden benchmark corpus (Phase 19B) measures extraction precision/recall/F1 across 7 adversarial categories. As of 2026-04-18 the bench harness at `scripts/bench/` provides continuous empirical monitoring — extraction F1 regressions are caught at CI time against a `seed_v1` baseline (current: F1=1.000, precision=1.000), and LLM-narrative regeneration faithfulness is measured via `OpenAiRegenerationRunner` (current: FActScore=0.960).

**Residual risk:** Structural gating catches malformed triplets but cannot detect semantically incorrect or misleading extractions. A structurally valid but factually wrong triplet passes the gate. The bench harness catches degradation over time against a fixed corpus but does not substitute for semantic validation on fresh inputs. Full semantic validation requires multi-pass extraction with confidence scoring (Venn-Abers `ConfidenceInterval` shipped; calibration fixture pending) and human-review lanes.

### 3.4. Semantic Collision Replay (✅ Now Protected)

**Threat:** Two different axiom keys hash to the same SHA-256 8-byte prefix, producing the same seed and potentially the same prime.

**Defense:** Collision resolution advances to the next prime. Cross-*instance* consistency is verified via a 1000-axiom stress test (two independent `GodelStateAlgebra` instances minting in different orders produce identical primes for identical keys). Cross-*runtime* consistency on the non-colliding derivation path is verified by three harnesses: `scripts/verify_godel_cross_runtime.py` (12 axiom-key fixtures + 6 state-encoding fixtures, Python ↔ Node byte-identical), the Browser JS inlined copy in `single_file_demo/index.html` round-tripped through `node standalone_verifier/verify.js`, and the shared primitives in `standalone_verifier/math.js` that both Node consumers import (eliminating the two-implementation drift surface that existed pre-`7ca3e56`).

**Residual risk:** Astronomically unlikely collision (birthday bound ≈ 2⁻³² at ~10⁴ axioms). Resolution path is tested cross-instance; the resolution path has not yet been exercised across runtimes because no natural collision has been triggered in the test corpora. Three independent instances + three runtimes agreeing on the non-colliding path is the load-bearing guarantee.

### 3.5. Contradiction Governance (✅ Now Protected)

**Threat:** Two bundles contain contradictory facts (e.g., "earth orbits sun" and "sun orbits earth").

**Defense:** The `DeterministicArbiter` resolves Level 3 Curvature using SHA-256 lexicographic ordering: for each conflict (subject, predicate, obj_a, obj_b), the winner is whichever object has the lower SHA-256 hash of its canonical triplet key. This guarantees identical resolution on every node without LLM. The `EpistemicArbiter` (LLM-based) remains available as an optional upgrade when richer judgment is desired.

**Residual risk:** SHA-256 ordering is deterministic but not semantically meaningful — it picks a winner, but doesn't judge truth. This is by design: the system does not claim to solve "objective truth."

### 3.6. Denial of Service (⚠️ Partially Protected)

**Threat:** An attacker submits extremely large integers or bundles to exhaust memory or CPU.

**Defense (shipped):** Bundle limits — canonical tome max 10 MB, state integer max 100,000 digits, axiom count max 10,000. These are enforced at the import boundary.

**Defense (not shipped — latent):** `sum_engine_internal/infrastructure/rate_limiter.py` implements a sliding-window per-IP limiter. **This module is not yet wired into `api/quantum_router.py`** — see `docs/MODULE_AUDIT.md` for the full accounting. The implementation works and is tested (`Tests/test_rate_limiter.py`), but the API does not call into it, so volumetric abuse at the request boundary is not blocked today. Wiring is a single-file change; this threat model will be updated when the import lands.

**Residual risk:** Volumetric request-layer DoS is unprotected until the rate limiter is wired. Distributed DDoS additionally requires upstream infrastructure (CDN, WAF). The Cloudflare Pages deployment path for the single-file demo inherits Cloudflare's edge rate-limiting by default, independent of the application layer — see `README.md` "Single-File Deployment" section.

### 3.7. Ledger Tampering (✅ Detectable, concurrency-hardened)

**Threat:** An attacker with database access modifies, deletes, or inserts events in the Akashic Ledger to alter the reconstructed knowledge state.

**Defense (Phase 19C):** SHA-256 Merkle hash-chain on every event. Each event stores `prev_hash = SHA-256(prev_hash + payload)` from a deterministic genesis seed. `verify_chain()` walks the full chain on boot and reports the first broken link. 16 tests in `Tests/test_merkle_chain.py` verify single-writer tamper detection (mutation, deletion, injection, hash overwrite). 6 additional tests in `Tests/test_ledger_concurrency.py` verify the invariant holds under 50–200 concurrent `append_event` calls.

**Concurrency-hardening (commit `9c4139d`):** Before this fix, `append_event` read `prev_hash` in autocommit mode (Python's sqlite3 default starts a transaction only on the first INSERT), so two concurrent writers could both observe the same parent hash, compute their event hashes against the same stale parent, and both commit — leaving `verify_chain()` reporting `is_valid=False` on a perfectly well-behaved multi-writer pipeline. The fix wraps every writer in `BEGIN IMMEDIATE`, acquiring SQLite's reserved write-lock before the SELECT. The discipline is now centralised in `AkashicLedger._write_txn` (commit `76ceb40`) so every future writer inherits it. Tamper-detection now holds universally, not only single-writer.

**Residual risk:** A local attacker with full database write access can recompute the entire chain from genesis. The hash chain protects against partial tampering, not full database replacement.

### 3.8. P2P Mesh Authentication (⚠️ Partial)

**Threat:** An unauthenticated party injects arbitrary axioms via `/sync/state`.

**Defense (current):** In production mode (non-default JWT secret), `/sync/state`
requires JWT authentication. This prevents anonymous state injection but does not
provide mutual peer authentication.

**Residual risk:** Any party with a valid JWT can inject axioms. A compromised peer
node with valid credentials can bloat state indefinitely via LCM (can add axioms
but cannot delete). Full mutual TLS or peer certificate pinning would address this.

---

## 4. Attack Surface Summary

| Attack Vector | Protected | Mechanism |
|--------------|-----------|-----------|
| Bundle tampering in transit | ✅ | HMAC-SHA256 |
| State/tome mismatch | ✅ | Witness verification |
| Version mismatch | ✅ | Version gate |
| Malformed bundles | ✅ | Field validation |
| Public authenticity | ✅ | Ed25519 signatures (self-asserted key) |
| Key compromise | ✅ | Key rotation + archive (no real-time revocation) |
| Adversarial extraction | ⚠️ | Structural gating (19A) + benchmark (19B); semantic validation pending |
| Collision replay | ✅ | 1000-axiom cross-instance stress test |
| Contradiction governance | ✅ | DeterministicArbiter (SHA-256 ordering, no LLM) |
| Resource exhaustion | ✅ | Bundle size limits + sliding window rate limiter |
| Ledger tampering | ✅ | SHA-256 Merkle hash-chain (holds under concurrent writers post `9c4139d`; detect partial tamper) |
| VC 2.0 bundle forgery | ✅ | Ed25519 signature over SHA-256(JCS(proofConfig)) ‖ SHA-256(JCS(document)) per W3C Data Integrity 1.0 + `eddsa-jcs-2022`; 58 tests including tamper detection, key-reordering resilience, JSON-on-disk persistence |
| Request-layer volumetric DoS | ⚠️ | Rate limiter implemented (`sum_engine_internal/infrastructure/rate_limiter.py`) but NOT WIRED into `api/quantum_router.py`; see §3.6 |
| Render-receipt forgery | ✅ | Ed25519 signature over JCS-canonical payload (Phase E.1 v0.9.A); detached JWS w/ JWKS distribution; verifier rejects every tampered signed-field per the 15-fixture cross-runtime matrix. See `docs/RENDER_RECEIPT_FORMAT.md` §5 + `docs/PROOF_BOUNDARY.md` §1.8 |
| Trust-root manifest forgery | ✅ | Same Ed25519/JCS/JWS shape as render receipts (R0.2); 17-test round-trip including 9 tampered-payload variants. See `docs/TRUST_ROOT_FORMAT.md` §6 |
| CI / supply-chain compromise | ✅ | R0.3 hardening: every action SHA-pinned, `permissions: contents: read` defaults, OpenSSF Scorecard advisory, StepSecurity Harden-Runner audit-mode. SHA-pin lint job runs on every push (`scripts/lint_workflow_pins.py`) |
| Receipt key compromise (post-rotation acceptance) | ⚠️ | Rotation-grace-window only today (`docs/RENDER_RECEIPT_FORMAT.md` §6); explicit revocation surface (`/.well-known/revoked-kids.json`) is queued as G3 in `docs/NEXT_SESSION_PLAYBOOK.md`. Operator runbook in `docs/INCIDENT_RESPONSE.md` case 1 |
| JWKS endpoint drift | ⚠️ | Detected via the daily transparency anchor's `jwks_sha256` (R0.5 design-now; implementation pending — see `docs/TRANSPARENCY_ANCHOR.md`). Operator runbook in `docs/INCIDENT_RESPONSE.md` case 2 |
| Cross-runtime verifier divergence | ✅ | 15-fixture matrix consumed unchanged by both Python (`sum_engine_internal.render_receipt`) and JS (`single_file_demo/receipt_verifier.js`) verifiers; cross-runtime byte-identical outcomes locked in CI (`vendor-byte-equivalence` job). PROOF_BOUNDARY §1.8 stands at "proved on adversarial inputs across runtimes" |
| LLM provider silent model drift | ⚠️ | Receipt's `payload.model` records the model the API actually echoed back (not the configured-default); operator detection runbook in `docs/INCIDENT_RESPONSE.md` case 7. `sum.model_call_evidence.v1` sidecar (R0.5 design-now; implementation pending — see `docs/MODEL_CALL_EVIDENCE_FORMAT.md`) is the planned hash-only forensic-binding surface |

---

## Cross-references

The Phase E.1 / R0 doc-pass arc extended SUM's threat surface
substantially. Each new artifact has a corresponding spec + the
operator-side response is documented in `docs/INCIDENT_RESPONSE.md`:

- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §5 +
  §7 — render-receipt trust scope + threat model.
- [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) §6 + §7 —
  trust-root manifest trust scope + threat model.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) — operator
  runbook for the eight named crisis cases (render-key compromise,
  JWKS drift, PyPI release compromise, GHA workflow compromise,
  Worker deploy compromise, benchmark claim later wrong, LLM
  provider model drift, canonicalisation bug).
- [`SECURITY.md`](../SECURITY.md) — researcher → operator
  vulnerability disclosure policy.
- [`docs/TRANSPARENCY_ANCHOR.md`](TRANSPARENCY_ANCHOR.md) — R0.5
  design for daily transparency anchoring; the post-hoc-tampering
  detection surface for cases 2, 3, 5.
- [`docs/MODEL_CALL_EVIDENCE_FORMAT.md`](MODEL_CALL_EVIDENCE_FORMAT.md)
  — R0.5 design for the optional hash-only model-call provenance
  sidecar; addresses LLM provider drift forensically.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.8 + §2.6 + §2.7 —
  what's proved vs measured on the new layers.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) G3, P5,
  P7 — open hardening tracks: revocation surface, threat-model-to-
  test traceability, supply-chain attestation. Each closes a row in
  the table above.
