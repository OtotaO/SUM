# Canonical Semantic ABI Specification

**Version:** 1.1.0
**Status:** Normative
**Date:** 2026-03-22

---

## 1. Scope

This document specifies the Canonical Semantic ABI: the deterministic format, derivation rules, and transport protocol for Gödel-State knowledge bundles in the SUM engine.

A conforming implementation MUST be able to:
1. Parse a canonical tome and extract axiom keys
2. Derive the same prime for a given axiom key as any other conforming implementation
3. Reconstruct the same Gödel State Integer from a canonical tome
4. Validate bundle structure and version compatibility

## 2. Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

| Term | Definition |
|------|-----------|
| Axiom | An irreducible semantic triple: (subject, predicate, object) |
| Axiom Key | Normalized string: `subject\|\|predicate\|\|object` (all lowercase, trimmed) |
| Gödel State Integer | Product of all axiom primes via LCM: `∏ lcm(prime_i)` |
| Canonical Tome | Deterministic text rendering of a Gödel State Integer |
| Bundle | Self-contained JSON transport unit with tome, state, metadata, and signature |

## 3. Canonical Format

### 3.1. Version Header

A canonical tome MUST begin with a version header line:

```
@canonical_version: <major>.<minor>.<patch>
```

Implementations MUST parse this header. If the major version does not match the implementation's supported major version, the implementation MUST reject the tome with a clear error.

### 3.2. Canonical Tome Grammar

```
tome        = version_header NEWLINE title NEWLINE sections
version_header = "@canonical_version: " VERSION
title       = "# " TEXT
sections    = section*
section     = section_header NEWLINE fact_lines NEWLINE
section_header = "## " TEXT
fact_lines  = fact_line*
fact_line   = "The " SUBJECT " " PREDICATE " " OBJECT "."
```

Where:
- `SUBJECT`, `PREDICATE`, `OBJECT` are single whitespace-free tokens (no internal spaces)
- All tokens are lowercase
- Sections are sorted lexicographically by subject
- Fact lines within a section are sorted lexicographically by full axiom key

### 3.3. Canonical Line Extraction Rule

A conforming parser MUST extract axiom keys only from lines matching:

```
/^The\s+(\S+)\s+(\S+)\s+(\S+)\.$/
```

All other lines (headers, blank lines, comments) MUST be ignored during state reconstruction.

### 3.4. Axiom Key Normalization

Given a canonical fact line `The {s} {p} {o}.`, the axiom key is:

```
lowercase(s) + "||" + lowercase(p) + "||" + lowercase(o)
```

Implementations MUST normalize to lowercase. Leading and trailing whitespace MUST be trimmed.

## 4. Deterministic Prime Derivation

### 4.1. Algorithm

Given an axiom key string `K`:

1. Compute `H = SHA-256(K)` where `K` is encoded as UTF-8
2. Extract the first 8 bytes of `H` as a big-endian unsigned 64-bit integer: `seed`
3. Find the next prime number strictly greater than `seed`: `prime = nextprime(seed)`

### 4.2. Collision Handling

If `prime` is already assigned to a different axiom key within the same instance:

1. Advance: `prime = nextprime(prime)`
2. Repeat until `prime` is either unassigned or assigned to the same axiom key

> [!NOTE]
> **Informative.** Collision probability is astronomically low for SHA-256 over an 8-byte prefix (≈ 2⁻³² for birthday-bound on 64-bit space at ~10⁴ axioms). In practice, the collision resolution path is almost never exercised. However, a conforming implementation MUST handle it.

### 4.3. Cross-Implementation Caveat

Collision resolution depends on minting order. Two implementations that mint axioms in different orders may produce different primes for a colliding pair. In practice this does not affect bundle verification because:
- Bundles contain the canonical tome AND the state integer
- The witness verifier mints primes in canonical-line order (deterministic)
- Both producer and verifier derive from the same canonical tome

### 4.4. Reference Vectors

| Axiom Key | SHA-256 First 8 Bytes (hex) | Seed | Prime |
|-----------|----------------------------|------|-------|
| `alice\|\|likes\|\|cats` | `c6d380e53c64fca9` | `14326936561644797097` | `14326936561644797201` |
| `bob\|\|knows\|\|python` | `b37d3c2b55c0b019` | `12933559861697884185` | `12933559861697884259` |
| `earth\|\|orbits\|\|sun` | `8e3176e1eae59d0a` | `10246101339925224714` | `10246101339925224733` |

Reference LCM state for all three: `1898585074409907150524167558344558620554613878579045806247`

### 4.5. v2 Derivation Rule (`sha256_128_v2`) — **SHADOW / EXPERIMENTAL**

> [!IMPORTANT]
> **`sha256_128_v2` is NOT the active default scheme.** It exists in shadow mode only.
> The active scheme remains `sha256_64_v1`. No production system should emit v2 primes
> unless explicitly configured to do so.

Given an axiom key string `K`:

1. Compute `H = SHA-256(K)` where `K` is encoded as UTF-8
2. Extract the first **16 bytes** of `H` as a big-endian unsigned 128-bit integer: `seed`
3. Find the next prime number strictly greater than `seed`: `prime = nextprime(seed)`

Primality testing for 128-bit candidates uses **BPSW** (Baillie–PSW):
- Phase 1: Strong base-2 Miller–Rabin
- Phase 2: Strong Lucas probable prime test (Selfridge Method A, P=1, Q=(1−D)/4)

> [!WARNING]
> BPSW has no known pseudoprimes but is not a formal proof of primality.
> This is an engineering trust assumption. If a BPSW counterexample is ever
> discovered, the v2 scheme would need to be superseded.

### 4.6. v2 Collision Policy

**v2 does NOT use a collision-resolution loop.** If two distinct axiom keys produce
the same 128-bit prime, the implementation MUST raise a hard error (e.g., `PrimeCollisionError`).

Rationale: the birthday bound for a 128-bit seed space is ~2⁶⁴ distinct axioms.
Collisions are astronomically unlikely. A collision-resolution loop introduces
order-dependent nondeterminism that is consensus-unsafe for distributed systems.

### 4.7. Non-Interoperability Statement

v1 (`sha256_64_v1`) and v2 (`sha256_128_v2`) produce **completely different primes**
for the same axiom key. They are non-interoperable:

- Sync between v1 and v2 nodes MUST be rejected (HTTP 409)
- Bundle import across schemes MUST be rejected
- No read-only, bridge, or fallback mode is permitted

Migration from v1 to v2 is a **fresh-universe operation**: semantic content may be
re-ingested, but state identity (primes, Gödel integers, bundles, ZK proofs) is invalidated.

### 4.8. v2 Reference Vectors

| Axiom Key | SHA-256 First 16 Bytes (hex) | Seed | Prime |
|-----------|------------------------------|------|-------|
| `alice\|\|likes\|\|cats` | `c6d380e53c64fca99927bc755a37c1d9` | `264285332112933860981052902103273947609` | `264285332112933860981052902103273947671` |
| `bob\|\|knows\|\|python` | `b37d3c2b55c0b0195fccb24db0ac8e75` | `238582068730743173113692744107846045301` | `238582068730743173113692744107846045503` |
| `earth\|\|orbits\|\|sun` | `8e3176e1eae59d0a3c4f748c53c015f3` | `189007209170893135023962148948466996723` | `189007209170893135023962148948466996823` |
| `quantum\|\|entangles\|\|photon` | `391d904d931d2bad8cff08a1d8c68ea1` | `75919499181718715751351316207293075105` | `75919499181718715751351316207293075217` |
| `water\|\|contains\|\|hydrogen` | `165d98c13189e1510a1a17e1f2a459d9` | `29728997747738460826164635775038413273` | `29728997747738460826164635775038413403` |

## 5. State Reconstruction

Given a set of axiom keys `{K₁, K₂, ..., Kₙ}`:

```
state = LCM(prime(K₁), prime(K₂), ..., prime(Kₙ))
```

Since all primes are distinct for distinct axiom keys (modulo collision resolution), and LCM of distinct primes equals their product:

```
state = prime(K₁) × prime(K₂) × ... × prime(Kₙ)
```

Deduplication is inherent: `LCM(p, p) = p`.

## 6. Bundle Schema

### 6.1. Required Fields

A conforming bundle MUST contain:

| Field | Type | Description |
|-------|------|-------------|
| `bundle_version` | string | Bundle format version (currently `"1.1.0"`) |
| `canonical_format_version` | string | Canonical tome grammar version (currently `"1.0.0"`) |
| `canonical_tome` | string | The canonical tome text |
| `state_integer` | string | Decimal string representation of the Gödel State Integer |
| `timestamp` | string | ISO 8601 timestamp with timezone |
| `signature` | string | Signature string (see §7) |

### 6.2. Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `branch` | string | Branch identifier |
| `axiom_count` | integer | Number of active axioms |
| `is_delta` | boolean | Whether this is a delta bundle |
| `public_signature` | string | Ed25519 signature: `"ed25519:<base64>"` (see §7.2) |
| `public_key` | string | Ed25519 public key: `"ed25519:<base64>"` (see §7.2) |

### 6.3. State Integer Encoding

The state integer MUST be encoded as a decimal string to avoid JSON integer overflow. Implementations MUST use arbitrary-precision integer types for reconstruction.

## 7. Signature

### 7.1. Current: HMAC-SHA256

The current signature format is:

```
hmac-sha256:<hex_digest>
```

The HMAC payload is:

```
canonical_tome + "|" + state_integer + "|" + timestamp
```

The HMAC key is a shared secret between the producer and consumer.

> [!WARNING]
> HMAC-SHA256 provides **tamper detection between trusted peers** sharing a secret key. It does NOT provide public third-party authenticity verification. See `THREAT_MODEL.md` for details.

### 7.2. Ed25519 Public-Key Signatures

When a signing keypair is available, bundles include dual signatures:

```
"public_signature": "ed25519:<base64_signature>",
"public_key": "ed25519:<base64_32byte_pubkey>"
```

The Ed25519 payload is identical to the HMAC payload:

```
canonical_tome + "|" + state_integer + "|" + timestamp
```

Encoded as UTF-8 before signing.

**Key format:** Ed25519 private keys are stored as PEM (PKCS8). Public keys are 32 bytes, base64-encoded in bundles.

**Verification:** Any party with the 32-byte public key can verify the signature without possessing the HMAC secret. Both signature types are verified independently on import.

**Backward compatibility:** `public_signature` and `public_key` are OPTIONAL. Bundles without these fields are valid v1.0.0 bundles and MUST be accepted by v1.1.0 consumers.

## 8. Import/Export Semantics

### 8.1. Export

An exporter MUST:
1. Generate the canonical tome from the state integer
2. Serialize the state integer as a decimal string
3. Generate a timestamp
4. Compute the signature
5. Assemble the bundle with all required fields

### 8.2. Import

An importer MUST:
1. Validate all required fields are present
2. Verify the signature
3. Parse the state integer from the decimal string
4. Optionally: reconstruct the state from the canonical tome and compare

### 8.3. Verification (Witness Mode)

A witness verifier:
1. MUST parse the canonical tome
2. MUST independently derive primes from axiom keys
3. MUST reconstruct the state integer via LCM
4. MUST compare the reconstructed integer to `state_integer`
5. SHOULD verify the Ed25519 signature if `public_key` is embedded

## 9. Delta Bundles

A delta bundle contains only the novel axioms between a source and target state:

```
delta_state = target_state // gcd(target_state, source_state)
```

Delta bundles use the same schema as full bundles. The `is_delta` field SHOULD be set to `true`.
