# Compatibility Policy

**Version:** 1.0.0
**Date:** 2026-03-22 (confirmed current as of `sum-engine 0.1.0` — 2026-04-22)

> This policy governs on-wire artifacts (CanonicalBundles, canonical tome
> format, prime scheme). It is unchanged by the v0.1.0 PyPI release:
> `sum-engine 0.1.0` ships the existing `sha256_64_v1` scheme and
> `canonical_format_version 1.0.0` exactly as specified below. The `sum`
> CLI itself follows the Python-package semver track independently — its
> breaking-change commitments live in [`CHANGELOG.md`](../CHANGELOG.md).

---

## 1. Version Numbering

Both `canonical_format_version` and `bundle_version` follow Semantic Versioning (SemVer):

```
MAJOR.MINOR.PATCH
```

| Component | Meaning |
|-----------|---------|
| MAJOR | Breaking change to the canonical grammar, prime derivation, or bundle schema |
| MINOR | Backward-compatible additions (new optional fields, informative sections) |
| PATCH | Bug fixes, documentation corrections, no behavioral change |

## 2. Compatibility Rules

### 2.1. Canonical Format Version

The `canonical_format_version` governs the canonical tome grammar and prime derivation algorithm.

| Change Type | Version Impact | Consumer Behavior |
|------------|---------------|-------------------|
| New canonical line grammar | MAJOR bump | MUST reject if major version unsupported |
| New optional tome sections | MINOR bump | SHOULD ignore unrecognized sections |
| Derivation algorithm change | MAJOR bump | MUST reject; primes are incompatible |
| Reference vector corrections | PATCH bump | No behavioral change |

**Rule:** A consumer MUST reject a tome whose `canonical_format_version` major version is greater than its supported major version. It SHOULD accept tomes with the same major version and a higher minor or patch version.

### 2.2. Bundle Version

The `bundle_version` governs the bundle JSON schema.

| Change Type | Version Impact | Consumer Behavior |
|------------|---------------|-------------------|
| New required field | MAJOR bump | MUST reject if missing required field |
| New optional field | MINOR bump | SHOULD ignore unrecognized fields |
| Field semantics change | MAJOR bump | MUST reject if major version unsupported |
| Schema clarification | PATCH bump | No behavioral change |

### 2.3. Signature Algorithm

Signature algorithm changes are implicitly handled by the signature prefix:
- `hmac-sha256:...` — current
- Future algorithms will use distinct prefixes (e.g., `ed25519:...`)

A consumer that encounters an unrecognized signature prefix MUST reject the bundle or skip signature verification with a clear warning.

## 3. Backward Compatibility Guarantees

### 3.1. Current Guarantees (v1.0.0)

| Guarantee | Status |
|-----------|--------|
| v1.0.0 canonical tomes remain parseable | ✅ Guaranteed |
| v1.0.0 bundles remain importable | ✅ Guaranteed |
| Reference vectors (§4.4 of spec) remain valid | ✅ Guaranteed |
| New implementations can verify v1.0.0 bundles | ✅ Guaranteed |

### 3.2. What MAY Change in Minor Versions

- Additional optional bundle fields
- Additional informative tome sections
- Additional standalone witness implementations
- Documentation updates

### 3.3. What Requires a Major Version Bump

- Changes to the canonical line grammar
- Changes to the prime derivation algorithm
- Changes to axiom key normalization rules
- Changes to the HMAC payload format
- Removal of required bundle fields
- Changes to collision resolution semantics

## 4. Migration Path

When a major version bump is necessary:

1. The new version MUST be published with a migration guide
2. A compatibility layer SHOULD exist that can read both old and new formats
3. The standalone witness MUST be updated to support both versions during the transition period
4. Test fixtures for both versions MUST exist in the test suite
5. A minimum deprecation period of one release cycle SHOULD be observed before removing old version support

## 5. Frozen Artifacts

The following artifacts are considered frozen for v1.0.0 and MUST NOT change without a major version bump:

- Reference vectors in `CANONICAL_ABI_SPEC.md` §4.4
- Canonical line regex: `/^The\s+(\S+)\s+(\S+)\s+(\S+)\.$/`
- SHA-256 seed derivation (first 8 bytes, big-endian)
- Bundle required field set: `{canonical_tome, state_integer, timestamp, signature, bundle_version, canonical_format_version}`
- HMAC payload format: `canonical_tome|state_integer|timestamp`
