# Trust-root manifest format (`sum.trust_root.v1`)

**Status:** spec landing alongside Phase R0.2 tooling.
**Schema identifier:** `sum.trust_root.v1`
**Cryptography:** Ed25519 (RFC 8032) over JCS-canonical bytes (RFC 8785), wrapped as a detached JWS (RFC 7515 §A.5) with public keys distributed via JWKS (RFC 7517). Same primitives the render-receipt format ([`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md)) uses; the trust-root manifest reuses them so consumers fielding either artifact use the same verifier shape.

A trust-root manifest is the **single signed artifact a downstream consumer verifies first** when adopting a SUM release. It binds, in one verifiable object: the source commit, the release version, the published artifact hashes, the JWKS state, and the algorithm registry. Without it, a consumer composes the trust chain by hand from multiple surfaces (GitHub release page, PyPI page, JWKS endpoint, etc.) and there's no single thing they can re-verify if any one surface is suspect.

The manifest is **TUF-inspired, not full TUF**: TUF's required metadata roles (root / targets / snapshot / timestamp) are appropriate for systems that publish updates continuously. SUM publishes per-release manifests; the simpler TUF-style root + targets is sufficient at this scale. If SUM's deployment shape ever needs continuous-update semantics, the manifest grows toward a full TUF repository — until then, this lighter shape is honest about what it is.

The bar for this document: a downstream consumer should be able to implement a working trust-root verifier from this spec without reading SUM's source. If you find yourself reaching for `scripts/build_trust_manifest.py` or `sum_engine_internal/trust_root/verifier.py`, that's a defect in this spec — please file an issue.

---

## 1. Wire format

A signed trust-root manifest is a JSON object with this shape:

```json
{
  "schema": "sum.trust_root.v1",
  "kid": "sum-trust-root-2026-04-27-1",
  "payload": {
    "issued_at": "2026-04-27T18:00:00.000Z",
    "repo": "OtotaO/SUM",
    "commit": "<release_commit_sha>",
    "release": "v0.3.1",
    "artifacts": [
      {
        "name": "sum_engine-0.3.1-py3-none-any.whl",
        "kind": "pypi-wheel",
        "sha256": "<hex>",
        "size_bytes": 184320,
        "pypi_provenance": "present",
        "github_attestation": "present",
        "cosign_bundle": "absent"
      },
      {
        "name": "sum_engine-0.3.1.tar.gz",
        "kind": "pypi-sdist",
        "sha256": "<hex>",
        "size_bytes": 92160,
        "pypi_provenance": "present",
        "github_attestation": "present",
        "cosign_bundle": "absent"
      }
    ],
    "render_receipt_jwks": {
      "current_kids": ["sum-render-2026-04-27-1"],
      "revoked_kids": [],
      "jwks_sha256": "<hex>"
    },
    "algorithm_registry": {
      "prime_scheme_current": "sha256_64_v1",
      "prime_scheme_next": "sha256_128_v2"
    }
  },
  "jws": "<protected-b64>..<signature-b64>"
}
```

The four top-level keys are siblings, not nested:

| Field | Type | Purpose |
|---|---|---|
| `schema` | string | Schema identifier. v1 today; future versions add new identifiers, never mutate this value's meaning. |
| `kid` | string | Key ID matching one entry in the issuer's trust-root JWKS. The verifier picks the right public key by this kid. **Distinct from the render-receipt kid** — render-key compromise does not invalidate trust-root manifests. |
| `payload` | object | The signed-over object. Verifier JCS-canonicalises this exact object and verifies `jws` against the resulting bytes. |
| `jws` | string | Detached JWS: `<protected-header-base64url>..<signature-base64url>`. Middle segment is empty; the JCS-canonical payload bytes are the detached payload. |

### 1.1 Payload field semantics

#### Top-level

| Field | Type | Source of truth |
|---|---|---|
| `issued_at` | string | ISO-8601 UTC timestamp at manifest construction time. |
| `repo` | string | Canonical `OWNER/REPO` form (e.g. `OtotaO/SUM`); MUST match the GitHub Trusted Publisher claim on the release artifacts. |
| `commit` | string | Git commit SHA (40-char lowercase hex) of the source tree the release was built from. Cross-checks against `git log <release>` on the published repo. |
| `release` | string | The release tag (e.g. `v0.3.1`). MUST match the SemVer of the artifacts in `artifacts[]`. |
| `artifacts` | array | One entry per published artifact (wheel, sdist, standalone verifier bundle, Worker bundle). See §1.1.1. |
| `render_receipt_jwks` | object | State of the render-receipt JWKS at issuance time. See §1.1.2. |
| `algorithm_registry` | object | Currently-active and next-planned cryptographic schemes. See §1.1.3. |

#### 1.1.1 `artifacts[]`

Each entry binds a published artifact to a hash + a set of attestation-status flags so consumers can immediately see what trust paths exist:

| Field | Type | Meaning |
|---|---|---|
| `name` | string | The exact filename downstream consumers will see (e.g. `sum_engine-0.3.1-py3-none-any.whl`). |
| `kind` | string | One of `pypi-wheel`, `pypi-sdist`, `standalone-verifier-js`, `worker-bundle`. Forward-compat: consumers MUST treat unknown values as opaque. |
| `sha256` | string | Lowercase hex SHA-256 of the artifact bytes. The consumer verifies this matches whatever they actually downloaded. |
| `size_bytes` | integer | Byte length of the artifact. Defence-in-depth against a hash-collision attack restricted to a different size. |
| `pypi_provenance` | string | `"present"` if the artifact has a verifiable PEP 740 PyPI attestation, `"absent"` otherwise. PyPI-only kinds. |
| `github_attestation` | string | `"present"` if the artifact has a corresponding GitHub Artifact Attestation, `"absent"` otherwise. |
| `cosign_bundle` | string | `"present"` if a Sigstore Cosign bundle was published, `"absent"` otherwise. |

The three attestation-status fields are **string-valued** rather than boolean so future values like `"revoked"` or `"superseded"` can land without a schema bump.

#### 1.1.2 `render_receipt_jwks`

| Field | Type | Meaning |
|---|---|---|
| `current_kids` | array of strings | The render-receipt kids actively signing at issuance time. Typically one; multiple during a rotation grace window. |
| `revoked_kids` | array of strings | Kids that MUST NOT be trusted past their `effective_revocation_at` time. The actual `effective_revocation_at` per kid lives in `/.well-known/revoked-kids.json` (R0.2's sibling track G3); this manifest just names the kids. |
| `jwks_sha256` | string | Lowercase hex SHA-256 of the JCS-canonicalised JWKS document at `/.well-known/jwks.json` at issuance time. Pins the JWKS contents so downstream consumers can detect tampering. |

#### 1.1.3 `algorithm_registry`

| Field | Type | Meaning |
|---|---|---|
| `prime_scheme_current` | string | The CanonicalBundle prime scheme actively in use (`sha256_64_v1` today). |
| `prime_scheme_next` | string\|null | The next-planned scheme (`sha256_128_v2` per Priority 3 in [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md)) or `null` if none planned. Future fields can extend this object to register signature algorithms (per G3 crypto-agility) without a schema bump. |

### 1.2 Forward-compat policy

The v1 schema is stable. Future revisions:
- MAY add new fields to `payload` or any sub-object. Verifiers MUST treat unknown fields as opaque-but-signed metadata.
- MUST NOT remove or rename existing fields.
- MUST bump the `schema` identifier to `sum.trust_root.v2` (etc.) for breaking changes.
- New protected-header fields that change verification semantics MUST land in the JWS `crit` array (RFC 7515 §4.1.11), forcing older verifiers to fail closed.

A v1-aware verifier will continue to verify v1 manifests forever. A v1-aware verifier that encounters a v2 manifest MUST reject closed via the schema check (Step 0.5 of the verifier algorithm below).

---

## 2. Verifier algorithm

The verifier algorithm is identical to the render-receipt verifier from [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §2.1, with one substitution: `SUPPORTED_SCHEMA = "sum.trust_root.v1"` instead of `sum.render_receipt.v1`. The Python implementation is [`sum_engine_internal.trust_root.verify_trust_manifest`](../sum_engine_internal/trust_root/verifier.py); a future JS implementation can mirror the receipt verifier's shape one-for-one.

### 2.1 Six-step procedure

```
1. Parse manifest.kid. Look up the matching key in trust-root jwks.keys.
   If no match → REJECT (unknown_kid).

2. Take manifest.payload (the object, not the string). JCS-canonicalise it
   (RFC 8785). Encode to UTF-8 bytes. This is the detached payload.

3. Split manifest.jws on "." → [protected_b64, middle, signature_b64].
   The middle segment MUST be empty (detached payload encoding).
   If middle is non-empty → REJECT (malformed_jws).

4. Construct the flattened JWS object:
   { protected: protected_b64, payload: <JCS bytes from step 2>, signature: signature_b64 }

5. Verify with the trust-root key. Catch signature failure → REJECT
   (signature_invalid).

6. Inspect the verified protected header. It MUST contain:
     alg: "EdDSA"
     kid: <matching manifest.kid>
     b64: false
     crit: ["b64"]
   Any deviation → REJECT (header_invariant_violated).
```

### 2.2 Forward-compat gates (steps 0.5 and 3.5)

Same as the render-receipt verifier:
- **Step 0.5 (schema gate).** `manifest.schema` MUST equal the supported schema (`sum.trust_root.v1`). v1 verifiers reject v2 manifests closed.
- **Step 3.5 (crit gate).** Decode the protected header BEFORE signature verification. Any extension named in `crit` that the verifier doesn't understand → REJECT (`crit_unknown_extension`). Per RFC 7515 §4.1.11, this fail-closed behaviour is mandatory.

The Python verifier's error-class taxonomy is identical to the render-receipt verifier's: `signature_invalid`, `malformed_jws`, `unknown_kid`, `kid_mismatch`, `schema_unknown`, `crit_unknown_extension`, `header_invariant_violated`, `malformed_receipt` (renamed `malformed_manifest` for the trust-root surface), `malformed_jwks`. A consumer that ships both verifiers can share error-class handling code.

---

## 3. Trust-root JWKS

The trust-root signing key is **distinct from the render-receipt signing key**. The two are separate keypairs, separate kids, separate JWKS endpoints (or separate entries in a combined endpoint). Render-key compromise does not invalidate trust-root manifests.

The recommended deployment shape:

| Endpoint | Purpose |
|---|---|
| `/.well-known/jwks.json` | render-receipt signing keys (Phase E.1 v0.9.A surface, unchanged) |
| `/.well-known/sum-trust-root.json` | the **current release's** signed trust-root manifest |
| `https://github.com/OtotaO/SUM/releases/download/v<version>/sum.trust_root.v1.json` | the per-release immutable manifest (canonical asset URL) |

The per-release URL is the durable form — it never changes once published. The `/.well-known/sum-trust-root.json` form is a convenience for verifiers fetching from the same origin as JWKS without dereferencing the GitHub redirect.

The **trust-root JWKS** itself can either:
- Live in a separate endpoint (`/.well-known/trust-root-jwks.json`) for clean separation; OR
- Be included in the same `/.well-known/jwks.json` document under a different `kid` (with a `use` field distinguishing the keys' purpose).

Either is acceptable; the second is simpler operationally and is the recommended default unless an operator has a specific reason to separate.

---

## 4. Producing a manifest

Operators run `scripts/build_trust_manifest.py` then `scripts/sign_trust_manifest.py` at release time:

```bash
# Step 1: gather facts from the local release artifacts.
python scripts/build_trust_manifest.py \
    --release v0.3.1 \
    --commit "$(git rev-parse HEAD)" \
    --dist-dir dist \
    --jwks-url https://sum-demo.ototao.workers.dev/.well-known/jwks.json \
    --algorithm-current sha256_64_v1 \
    --algorithm-next sha256_128_v2 \
    --out unsigned_manifest.json

# Step 2: sign the manifest with the trust-root private key.
# The private key is a Ed25519 OKP JWK; a pending publisher / KMS / one-time
# generated keypair all work. SUM's CI uses an Ed25519 key stored in a
# GitHub Actions Trusted Publisher integration mirroring the PyPI flow,
# but any Ed25519 OKP JWK on disk works.
python scripts/sign_trust_manifest.py \
    --in unsigned_manifest.json \
    --signing-jwk /path/to/trust_root_private.jwk \
    --kid "sum-trust-root-$(date -u +%Y-%m-%d)-1" \
    --out sum.trust_root.v1.json
```

Output is the signed manifest. Upload to:
- The GitHub release as an asset under the canonical name `sum.trust_root.v1.json`.
- (Optional, recommended) the Worker's `/.well-known/sum-trust-root.json`.

A single manifest covers a single release; new release → new manifest with bumped `commit`, `release`, and `artifacts[].sha256` values.

---

## 5. Consuming a manifest

The minimal trust-loop a downstream consumer should run:

```python
from sum_engine_internal.trust_root import verify_trust_manifest, VerifyError

# 1. Fetch the manifest
import urllib.request, json
manifest_url = "https://github.com/OtotaO/SUM/releases/download/v0.3.1/sum.trust_root.v1.json"
manifest = json.loads(urllib.request.urlopen(manifest_url).read())

# 2. Fetch the trust-root JWKS (same origin or GitHub release asset)
trust_root_jwks_url = "https://sum-demo.ototao.workers.dev/.well-known/jwks.json"
trust_root_jwks = json.loads(urllib.request.urlopen(trust_root_jwks_url).read())

# 3. Verify
try:
    result = verify_trust_manifest(manifest, trust_root_jwks)
    payload = result.payload
    print(f"verified manifest for {payload['repo']} {payload['release']}")
except VerifyError as e:
    print(f"REJECTED: {e.error_class} — {e}")
    raise

# 4. Use the payload to verify the actual artifact you downloaded.
import hashlib
local_wheel_sha = hashlib.sha256(open("sum_engine-0.3.1-py3-none-any.whl","rb").read()).hexdigest()
expected = next(a for a in payload["artifacts"] if a["name"] == "sum_engine-0.3.1-py3-none-any.whl")
assert local_wheel_sha == expected["sha256"], "downloaded wheel doesn't match manifest"
```

The manifest doesn't relieve the consumer of verifying the artifacts themselves — it gives the consumer **a single signed source of truth** for what hashes to expect. PEP 740 PyPI provenance + GitHub Artifact Attestation + Cosign bundle remain orthogonal verification paths; the manifest names which paths are present so the consumer knows what cross-checks are available.

---

## 6. Trust scope

A verified trust-root manifest PROVES:

| Claim | Defence mechanism |
|---|---|
| The issuer (holder of the trust-root signing key) attested to this release tuple. | Ed25519 signature; verified with the public JWK at the kid. |
| The expected artifact hashes are these. | Signed; consumer rehashes downloaded artifact and compares. |
| The expected source commit is this. | Signed; consumer can cross-check against `git log <release>` on the public repo. |
| The render-receipt JWKS at issuance time was this snapshot. | `jwks_sha256` is signed; consumer rehashes the JWKS document and compares to detect tampering between the manifest and the JWKS endpoint. |
| The algorithm registry's currently-active scheme is this. | Signed; protects against an attacker silently downgrading the prime scheme on a subset of clients. |

A verified trust-root manifest DOES NOT PROVE:

| Non-claim | Why not |
|---|---|
| The bytes downstream consumers downloaded are byte-identical to what was signed. | The consumer MUST hash their downloaded artifact and compare to `artifacts[].sha256`. The manifest names the expected hash; the cross-check is the consumer's. |
| The artifact's source code is bug-free. | The manifest is a trust artifact, not a code-quality artifact. SUM's [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) is the canonical place for what's proved vs measured vs designed about the code itself. |
| The render-receipt signing key is uncompromised. | A separate concern. The trust-root manifest names which kids are CURRENT and which are REVOKED at issuance time; an attacker who compromises the render-receipt key after manifest issuance can sign render receipts under a still-listed-as-current kid until the next manifest is published or `/.well-known/revoked-kids.json` lists the compromised kid (G3). |
| The manifest is fresh. | Like render receipts, the manifest's `issued_at` is at issuance time. A consumer requiring freshness MUST reject manifests older than their freshness window. |

---

## 7. Threat model

**Defends against:**
- **Artifact substitution.** A different artifact under the same name and version fails the consumer's `sha256 == manifest.artifacts[].sha256` check.
- **Source-tree drift.** If a downstream consumer wants to audit the source, the manifest's `commit` field tells them which exact tree to clone; tampering with the visible source doesn't affect what the manifest committed to.
- **JWKS substitution.** The `jwks_sha256` field lets a consumer detect a manipulated JWKS endpoint at any point post-issuance.
- **Algorithm downgrade.** An attacker can't silently shift `prime_scheme_current` for a subset of clients; what was signed is what was signed.
- **Single-surface compromise.** A consumer who verifies the manifest first and downloads only what it lists doesn't have to trust the GitHub release page, the PyPI page, the JWKS endpoint, and the Worker URL independently — one signed object covers all of them.

**Does NOT defend against:**
- **Trust-root key compromise.** If the trust-root signing key leaks, an attacker can forge manifests. Mitigation: rotate (same JWKS-rotation pattern as the render-receipt key) + monitor for anomalous manifest issuance. Keeping the trust-root key separate from the render-receipt key means render-key compromise alone doesn't break this layer.
- **Issuer collusion.** If the issuer signs intentionally false claims, the manifest is internally consistent but externally wrong. This is the trust-the-issuer assumption every signature scheme makes.
- **Pre-issuance tampering.** The manifest pins what's true at issuance time; if the artifact was already malicious when the issuer signed it, the manifest signs a malicious artifact's hash. Defence against this is the build-time CI gates (PEP 740 PyPI provenance + GitHub Artifact Attestation + Cosign bundle), not the manifest itself.

---

## Appendix A: example end-to-end

A worked example showing the full pipeline — generate keypair → build → sign → verify — lives as the round-trip pytest at `Tests/test_trust_root.py`. Reading that test gives a verifier author concrete byte-level expectations for every step of §2.1 and §4 above.
