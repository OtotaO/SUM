# Stage 3 Design Doc — 128-Bit Prime Derivation

**Status:** DRAFT — Not approved for implementation  
**Scheme ID:** `sha256_128_v2`  
**Date:** 2026-03-23  
**Author:** ototao  
**Prerequisite:** Stage 2 (scheme versioning) FROZEN  

---

## 1. Problem Statement

The current `sha256_64_v1` scheme derives primes from a 64-bit seed space (2⁶⁴ ≈ 1.8×10¹⁹). At scale, this creates two risks:

| Risk | Severity | Trigger Threshold |
|------|----------|-------------------|
| Birthday-bound collision on 8-byte SHA-256 prefix | Low | ~2¹⁶ axioms (65K) for 50% collision probability |
| Prime density in 64-bit range too sparse for large corpora | Medium | Corpus > 10⁶ axioms |
| Collision resolution order-dependence | Medium | Any collision between distributed nodes |

Moving to a 128-bit seed space (2¹²⁸ ≈ 3.4×10³⁸) makes birthday-bound collisions require ~2⁶⁴ axioms — effectively impossible for any realistic corpus. It eliminates collision resolution as a practical concern completely.

## 2. Proposed Scheme: `sha256_128_v2`

```
Input:   axiom_key (UTF-8 string)
Step 1:  H = SHA-256(axiom_key)
Step 2:  seed = int.from_bytes(H[:16], byteorder='big')     # 128-bit seed
Step 3:  prime = nextprime(seed)                              # using BPSW
Output:  prime (128-bit+ integer)
```

### 2.1. What Changes From v1

| Component | v1 (`sha256_64_v1`) | v2 (`sha256_128_v2`) |
|-----------|---------------------|----------------------|
| Seed extraction | `H[:8]` → u64 | `H[:16]` → u128 |
| Primality test | Deterministic M-R, 12 witnesses | BPSW |
| Prime size | ~64 bits | ~128 bits |
| Collision resolution | Still needed (rare) | Effectively eliminated |
| State integer size | Product of 64-bit primes | Product of 128-bit primes |

### 2.2. What Does NOT Change

- Hash function: still SHA-256
- Key normalization: still `subject||predicate||object` lowercase trimmed
- State composition: still LCM of primes
- Entailment: still modulo check
- Canonical tome format: unchanged
- Bundle structure: unchanged (string representation of state_integer)

## 3. Primality Method Decision

This is the consensus-critical core of Stage 3.

### 3.1. Option A: BPSW (Baillie-PSW) — RECOMMENDED

**Algorithm:** Miller-Rabin base 2 + Strong Lucas test with Selfridge parameters.

| Property | Value |
|----------|-------|
| Correctness claim | No known BPSW pseudoprime exists (verified to 2⁶⁴ by Feitsma/Galway, exhaustive search ongoing) |
| False positive rate | Conjectured zero; bounty of $30+ for counterexample since 1980 |
| Deterministic? | Yes — no random witnesses, no probabilistic behavior |
| Performance | ~4× slower than 12-witness M-R on 128-bit inputs |
| Cross-runtime availability | Python: `sympy.isprime()` already uses BPSW. Zig: must implement. Node: must implement. |

**Why BPSW for 128-bit:**
The 12-witness deterministic M-R set {2,3,5,7,11,13,17,19,23,29,31,37} is only proven correct up to 3.3×10²⁴ (≈ 81 bits). For 128-bit inputs, M-R with these witnesses is NO LONGER PROVABLY DETERMINISTIC. BPSW has no known failure at any range.

### 3.2. Option B: Extended Deterministic Miller-Rabin

Use a larger witness set proven for 128-bit range. Currently:
- No known witness set has been proven sufficient for all 128-bit composites.
- One could use the first ~20 primes as witnesses (heuristically sound but unproven).

**Verdict:** Unproven for this range. Rejected for identity-critical math.

### 3.3. Option C: Hybrid (BPSW + Trial Division)

Use trial division for small factors (< 10⁶) then BPSW for the remaining candidates.

**Verdict:** This is an optimization of Option A, not a fundamentally different choice. Can be adopted as an implementation detail without affecting the scheme definition.

### 3.4. Decision

> **BPSW (Option A) with optional trial-division prefilter (Option C optimization).**
>
> Rationale: It is the only primality test that is both (a) deterministic and (b) has a plausible zero-error claim at 128-bit scale. The 12-witness M-R is provably correct only up to ~81 bits.

## 4. Cross-Runtime Determinism Requirements

Three runtimes must produce identical primes for identical axiom keys. This is NON-NEGOTIABLE for the scheme to function.

### 4.1. Runtime Inventory

| Runtime | Current File | v1 Method | v2 Required Change |
|---------|-------------|-----------|-------------------|
| **Python** | `semantic_arithmetic.py` L141-143 | `h[:8]` → `sympy.nextprime(seed)` | `h[:16]` → `sympy.nextprime(seed)` (sympy uses BPSW internally) |
| **Zig** | `core-zig/src/main.zig` L105-117 | `hash[0..8]` → u64 → `nextPrime` (12-witness M-R) | `hash[0..16]` → u128 → new `nextPrime128` (BPSW) |
| **Node.js** | `standalone_verifier/verify.js` L106-113 | BigInt, same M-R | BigInt, `nextPrime` updated to BPSW |

### 4.2. Parity Test Requirement

Before any v2 code is merged:

```
For each axiom_key in REFERENCE_VECTORS:
    assert python_prime(axiom_key) == zig_prime(axiom_key) == node_prime(axiom_key)
```

Reference vectors MUST include:
- At least 3 existing v1 vectors (to prove v1 still works)
- At least 5 new v2 vectors with 128-bit seeds
- At least 1 edge case near 2¹²⁸ boundary
- The results must be frozen in `Tests/fixtures/`

### 4.3. Implementation Ordering

```
1. Python (sympy already does BPSW)        ← easiest, do first
2. Node.js (BigInt-native, straightforward) ← second
3. Zig (u128, must hand-write BPSW)         ← hardest, do last
```

## 5. Mixed-Scheme Behavior

### 5.1. Live Sync Rules

| Scenario | Behavior | HTTP Code |
|----------|----------|-----------|
| v1 node receives v1 sync | ✅ Accept | 200 |
| v1 node receives v2 sync | ❌ Reject | 409 |
| v2 node receives v2 sync | ✅ Accept | 200 |
| v2 node receives v1 sync | ❌ Reject | 409 |
| No scheme in request | Default to sender's own scheme | 200 |

> **v1 and v2 nodes MUST NOT merge state.** The primes are fundamentally different — LCM of mixed primes produces a state that neither side can interpret correctly.

### 5.2. Bundle Import/Export Rules

| Scenario | Behavior |
|----------|----------|
| v2 node exports bundle | Bundle carries `prime_scheme: "sha256_128_v2"` |
| v2 node imports v2 bundle | ✅ Accept, verify, reconstruct |
| v2 node imports v1 bundle | ❌ Reject with `ValueError` |
| v1 node imports v2 bundle | ❌ Reject with `ValueError` (already works via Stage 2) |

### 5.3. Read-Only Interop?

**No.** Even read-only operations (entailment checks, search) depend on the prime mapping. A v1 prime for "earth||orbits||sun" is different from a v2 prime for the same axiom. No cross-scheme operation is safe.

### 5.4. Branch-Level Mixing

**Forbidden.** A single KOS instance MUST run exactly one scheme. All branches within that instance use the same scheme. There is no per-branch scheme polymorphism.

## 6. Migration Strategy

### 6.1. Can a v1 Ledger Be Upgraded?

**No.** State upgrade would require:
1. Enumerating all axioms in the v1 ledger
2. Re-deriving primes under v2
3. Recomputing all branch states
4. Invalidating all existing bundles, witnesses, and ZK proofs

This is equivalent to rebuilding the universe. A v2 instance starts fresh.

### 6.2. Data Preservation

A migration tool MAY:
1. Export all axiom triples (subject, predicate, object) from a v1 ledger
2. Re-ingest them into a v2 instance
3. The v2 instance derives new primes — the state integers will be entirely different

This preserves **semantic content** but not **state identity**. Bundle signatures, Gödel State Integers, and ZK proofs from v1 are invalidated.

### 6.3. Rollback

If v2 migration is interrupted:
- The v1 ledger is unchanged (it was never modified)
- Discard the v2 instance and continue on v1
- No state corruption is possible because v1 and v2 ledgers are fully independent

### 6.4. Fixture Handling

| Fixture | Action |
|---------|--------|
| `Tests/fixtures/golden_vectors.json` | Keep v1 vectors, ADD v2 vectors |
| `Tests/fixtures/reference_bundle.json` | Keep as v1 reference, ADD v2 reference bundle |
| ABI spec reference vectors (§4.4) | Keep existing, ADD v2 section |

## 7. Runtime Surfaces Impacted

Complete enumeration of files that must change for v2:

### 7.1. Core Algorithm
| File | Change |
|------|--------|
| `internal/algorithms/semantic_arithmetic.py` | `h[:16]` seed extraction, conditional on active scheme |
| `internal/infrastructure/scheme_registry.py` | `CURRENT_SCHEME` → `sha256_128_v2` when activated |

### 7.2. Zig Core
| File | Change |
|------|--------|
| `core-zig/src/main.zig` | New `modPow128`, `isPrimeBPSW`, `nextPrime128` functions |
| `internal/infrastructure/zig_bridge.py` | New ctypes binding for `sum_get_deterministic_prime_128` |

### 7.3. Node.js Verifier
| File | Change |
|------|--------|
| `standalone_verifier/verify.js` | BPSW `isPrime` function, 128-bit seed extraction |

### 7.4. Bundle/Transport
| File | Change |
|------|--------|
| `internal/infrastructure/canonical_codec.py` | No change (scheme field already present) |
| `api/quantum_router.py` | No change (scheme validation already works) |
| `internal/infrastructure/p2p_mesh.py` | No change (scheme negotiation already works) |
| `docs/CANONICAL_ABI_SPEC.md` | Add §4.5 for v2 derivation rule + reference vectors |

### 7.5. Tests/Fixtures
| File | Change |
|------|--------|
| `Tests/test_scheme_registry.py` | Update compatibility expectations when v2 activates |
| `Tests/fixtures/` | Add v2 golden vectors and reference bundle |
| New: `Tests/test_128bit_parity.py` | Cross-runtime parity test |

### 7.6. Documentation
| File | Change |
|------|--------|
| `docs/CANONICAL_ABI_SPEC.md` | v2 derivation section |
| `README.md` | Update scheme description |
| This document | Mark as APPROVED |

## 8. Acceptance Criteria (Implementation Gates)

No v2 code may be merged until ALL of the following are satisfied:

| Gate | Requirement |
|------|-------------|
| **G1** | BPSW algorithm chosen and documented (this doc) |
| **G2** | Python v2 prime derivation produces correct results for all reference vectors |
| **G3** | Node.js v2 prime derivation matches Python exactly |
| **G4** | Zig v2 prime derivation matches Python exactly |
| **G5** | Cross-runtime parity test exists in `Tests/` and passes |
| **G6** | Mixed-scheme rejection tested end-to-end (v1↔v2 reject, v2↔v2 accept) |
| **G7** | v2 reference bundle exported, independently verified by Node.js verifier |
| **G8** | v2 golden vectors frozen in `Tests/fixtures/` |
| **G9** | ABI spec updated with v2 derivation rule |
| **G10** | Migration tool (optional) documented if built |

## 9. Open Questions

These MUST be resolved before marking this doc APPROVED:

1. **BPSW variant:** Use Selfridge parameters (D, P, Q) = first D in {5, −7, 9, −11, ...} where Jacobi(D|n) = −1? This is the standard Selfridge Method A. Confirm all three runtimes use identical parameters.

2. **Strong Lucas vs Standard Lucas:** Strong Lucas is more restrictive (fewer pseudoprimes). Recommend Strong Lucas. Confirm sympy's `isprime()` uses Strong Lucas internally.

3. **Trial division bound:** How many small primes to check before BPSW? Recommend primes up to 1000 (168 primes) for performance. This is an optimization, not a correctness concern.

4. **Collision resolution in v2:** With a 128-bit seed space, collisions are effectively impossible (birthday bound at ~2⁶⁴ axioms). Should we REMOVE the collision resolution loop entirely for v2, or keep it for belt-and-suspenders safety?

5. **WASM target:** The Zig WASM build currently exports u64-returning functions. u128 is not a native WASM type. Strategy: return the prime as two u64 halves, or as a hex string in WASM linear memory?

---

> [!CAUTION]
> **This document is DRAFT.** No implementation work should begin until all open questions are resolved and this document is marked APPROVED.
