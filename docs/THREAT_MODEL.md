# Threat Model

**Version:** 1.0.0
**Date:** 2026-03-22

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
| Third-party verification | **Not supported** (requires public-key crypto) |
| Key secrecy | **Assumed** (if key is compromised, all guarantees void) |

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

### 3.1. Public Authenticity (❌ Not Protected)

**Threat:** A third party wants to verify that a bundle was produced by a specific author.

**Current status:** HMAC is a shared-secret scheme. Anyone with the key can produce a valid signature. There is no way for a third party to verify provenance without the key.

**Mitigation (future):** Ed25519 or similar public-key signatures would allow anyone with the public key to verify the producer's identity. This is targeted for a future workstream.

### 3.2. Key Compromise (❌ Not Protected)

**Threat:** The HMAC signing key is leaked or stolen.

**Impact:** An attacker can forge valid-looking bundles. All bundles signed with the compromised key become untrustworthy.

**Mitigation:** Key rotation. Currently no key rotation mechanism exists. Future work should add key identifiers and rotation/revocation support.

### 3.3. Extraction Manipulation (❌ Not Protected)

**Threat:** An attacker crafts input text designed to produce misleading extractions (adversarial NLP).

**Impact:** The sieve may extract incorrect or misleading axioms. These become part of the canonical state and appear legitimate.

**Mitigation:** Extraction is the weakest link. Future phases target multi-pass extraction with confidence scoring and human-review lanes. Currently, extraction output is trusted without independent verification.

### 3.4. Semantic Collision Replay (⚠️ Partially Protected)

**Threat:** Two different axiom keys hash to the same SHA-256 8-byte prefix, producing the same seed and potentially the same prime.

**Defense:** Collision resolution advances to the next prime. However, this depends on minting order — two instances minting axioms in different orders may assign different primes to the same axiom key if a collision occurs.

**Risk assessment:** Astronomically unlikely (birthday bound ≈ 2⁻³² at ~10⁴ axioms). The collision resolution path is tested but not independently verified across runtimes.

### 3.5. Contradiction Governance (❌ Not Protected)

**Threat:** Two bundles contain contradictory facts (e.g., "earth orbits sun" and "sun orbits earth").

**Impact:** The system detects contradictions via exclusion zones (same subject+predicate, different objects). However, automated resolution (EpistemicArbiter) uses LLM judgment, which is non-deterministic and non-auditable.

**Honest status:** Contradiction detection is mechanical. Contradiction *resolution* involves judgment and is NOT purely mathematical. The system does not solve "objective truth" by arithmetic alone.

### 3.6. Denial of Service (❌ Not Protected)

**Threat:** An attacker submits extremely large integers or bundles to exhaust memory or CPU.

**Impact:** Arbitrary-precision integer operations scale with bit length. A sufficiently large integer could consume excessive resources.

**Mitigation:** No rate limiting or size bounds are currently enforced on bundle import. Future hardening should add explicit size limits.

---

## 4. Attack Surface Summary

| Attack Vector | Protected | Mechanism |
|--------------|-----------|-----------|
| Bundle tampering in transit | ✅ | HMAC-SHA256 |
| State/tome mismatch | ✅ | Witness verification |
| Version mismatch | ✅ | Version gate |
| Malformed bundles | ✅ | Field validation |
| Public authenticity | ❌ | Needs public-key crypto |
| Key compromise | ❌ | Needs key rotation |
| Adversarial extraction | ❌ | Needs extraction hardening |
| Collision replay | ⚠️ | Resolution exists, not cross-verified |
| Contradiction governance | ❌ | Detection only, resolution is non-deterministic |
| Resource exhaustion | ❌ | Needs size limits |
