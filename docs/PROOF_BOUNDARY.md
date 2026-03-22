# Proof Boundary

**Version:** 1.0.0
**Date:** 2026-03-22

This document explicitly separates what the SUM engine **proves mechanically**, what it **measures empirically**, and what remains **aspirational or future work**.

---

## 1. Mechanically Proven

These properties are enforced by deterministic code and verified by tests, including cross-runtime witnesses.

### 1.1. Canonical Round-Trip Conservation

**Claim:** For any Gödel State Integer `S`:
```
reconstruct(parse(canonical_tome(S))) == S
```

**Proof mechanism:** The Ouroboros verifier (Phase 14) encodes a state into a canonical tome, parses the canonical lines back into axiom keys, re-derives primes, and asserts integer equality.

**Boundary:** This proves lossless round-tripping **over the canonical semantic representation**. It does NOT prove that arbitrary English prose can be losslessly compressed and recovered. The canonical representation is the proof substrate; narrative text is a rendering layer.

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

---

## 2. Empirically Measured

These properties are observed but not formally proven. They depend on implementation quality and input characteristics.

### 2.1. Extraction Fidelity

The quality of semantic extraction from natural language depends on:
- The NLP parser (spaCy lemmatizer, dependency parser)
- Input text structure and complexity
- Domain vocabulary coverage

**Current status:** No formal extraction benchmark exists. Extraction quality is the acknowledged weakest link. Phase 18 (future) targets a multi-pass extraction ensemble with precision/recall measurement.

### 2.2. Operation Performance

Gödel arithmetic operations (LCM, GCD, modulo) operate on arbitrary-precision integers. Their complexity scales with integer **bit length**, not axiom count:
- GCD: O(n²) via Euclidean algorithm on n-bit integers (sub-quadratic with GMP)
- LCM: O(n²) (reduces to GCD)
- Modulo: O(n²)

For practical corpus sizes (< 10,000 axioms, < 10,000-bit integers), operations complete in microseconds. But they are NOT literally O(1).

### 2.3. Compression Ratio

The ratio of source text to canonical representation to integer form has not been formally benchmarked across diverse corpora. Phase 19 (future) targets this.

---

## 3. Aspirational / Future Work

These are design goals, NOT current capabilities.

| Goal | Status | Target Phase |
|------|--------|-------------|
| Public-key bundle provenance | Not implemented | Phase 17+ |
| Richer semantic IR (qualifiers, time, negation) | Not implemented | Phase 17+ |
| Multi-pass extraction ensemble | Not implemented | Phase 18+ |
| Hierarchical semantic compression (motifs, chapters) | Not implemented | Phase 19+ |
| Multi-renderer rehydration (textbook, quiz, study guide) | Not implemented | Phase 20+ |
| Curriculum sequencing / learning OS | Not implemented | Phase 21+ |
| Federation with trust policies | Not implemented | Phase 22+ |
| Scientific/technical corpora support | Not implemented | Phase 23+ |

---

## 4. Complexity Honesty

### What "O(1)" Actually Means in This Codebase

Many operations are described as "O(1)" in comments and documentation. This is shorthand for:

> **O(1) in axiom count** — the operation does not require scanning the axiom list, re-parsing documents, or iterating over a corpus.

It is NOT O(1) in the information-theoretic sense. All operations on Gödel integers scale with the **bit length** of the integer, which grows logarithmically with each axiom's prime.

**Honest characterization:**
- Entailment check (`state % prime == 0`): O(n) in bit length, O(1) in axiom enumeration
- Merge (`lcm(A, B)`): O(n²) in bit length via GCD
- Branching (integer copy): O(n) in bit length
- Sync delta (`gcd(A, B)`): O(n²) in bit length

For practical corpus sizes, these operations are extremely fast. The key insight is that they avoid corpus-scale document scanning, not that they are literally constant-time.
