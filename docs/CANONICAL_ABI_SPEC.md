# Canonical Semantic ABI Specification

**Version:** 1.0.0
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
| `bundle_version` | string | Bundle format version (currently `"1.0.0"`) |
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

### 7.2. Future: Public-Key Signatures

A future version MAY add public-key signatures (e.g., Ed25519) for independent provenance verification. When introduced, both signature types MAY coexist in a dual-signature model.

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
5. SHOULD NOT verify the HMAC signature (requires shared secret)

## 9. Delta Bundles

A delta bundle contains only the novel axioms between a source and target state:

```
delta_state = target_state // gcd(target_state, source_state)
```

Delta bundles use the same schema as full bundles. The `is_delta` field SHOULD be set to `true`.
