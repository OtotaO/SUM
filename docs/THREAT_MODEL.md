# Threat Model

**Version:** 1.1.0
**Date:** 2026-03-25

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

**Defense (Phase 19A):** `ExtractionValidator` provides structural gating: non-empty fields, length bounds, control character rejection, JSON fragment rejection, predicate canonicalization, and batch deduplication. 25 unit tests verify gating logic. A 50-document golden benchmark corpus (Phase 19B) measures extraction precision/recall/F1 across 7 adversarial categories.

**Residual risk:** Structural gating catches malformed triplets but cannot detect semantically incorrect or misleading extractions. A structurally valid but factually wrong triplet passes the gate. Full semantic validation requires multi-pass extraction with confidence scoring and human-review lanes.

### 3.4. Semantic Collision Replay (✅ Now Protected)

**Threat:** Two different axiom keys hash to the same SHA-256 8-byte prefix, producing the same seed and potentially the same prime.

**Defense:** Collision resolution advances to the next prime. Cross-instance consistency is now verified: a 1000-axiom stress test confirms that two independent `GodelStateAlgebra` instances minting axioms in different orders produce identical primes for identical axiom keys. Order independence is mathematically guaranteed.

**Residual risk:** Astronomically unlikely collision (birthday bound ≈ 2⁻³² at ~10⁴ axioms). Resolution path is tested and cross-verified.

### 3.5. Contradiction Governance (✅ Now Protected)

**Threat:** Two bundles contain contradictory facts (e.g., "earth orbits sun" and "sun orbits earth").

**Defense:** The `DeterministicArbiter` resolves Level 3 Curvature using SHA-256 lexicographic ordering: for each conflict (subject, predicate, obj_a, obj_b), the winner is whichever object has the lower SHA-256 hash of its canonical triplet key. This guarantees identical resolution on every node without LLM. The `EpistemicArbiter` (LLM-based) remains available as an optional upgrade when richer judgment is desired.

**Residual risk:** SHA-256 ordering is deterministic but not semantically meaningful — it picks a winner, but doesn't judge truth. This is by design: the system does not claim to solve "objective truth."

### 3.6. Denial of Service (✅ Now Protected)

**Threat:** An attacker submits extremely large integers or bundles to exhaust memory or CPU.

**Defense:** Multi-layer protection:
- **Bundle limits:** Canonical tome max 10 MB, state integer max 100,000 digits, axiom count max 10,000
- **Rate limiting:** In-memory sliding window rate limiter (configurable per-IP, default 60 req/min) blocks volumetric abuse

**Residual risk:** In-memory rate limiter does not persist across restarts. Distributed DDoS requires upstream infrastructure (CDN, WAF).

### 3.7. Ledger Tampering (✅ Detectable)

**Threat:** An attacker with database access modifies, deletes, or inserts events in the Akashic Ledger to alter the reconstructed knowledge state.

**Defense (Phase 19C):** SHA-256 Merkle hash-chain on every event. Each event stores `prev_hash = SHA-256(prev_hash + payload)` from a deterministic genesis seed. `verify_chain()` walks the full chain on boot and reports the first broken link. 16 tests verify tamper detection (mutation, deletion, injection, hash overwrite).

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
| Ledger tampering | ✅ | SHA-256 Merkle hash-chain (detect partial tamper) |
