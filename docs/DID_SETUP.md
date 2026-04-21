# DID Setup for SUM Issuance

SUM emits W3C Verifiable Credentials 2.0 signed under the `eddsa-jcs-2022` cryptosuite (commit `e007f94`). The default `make_credential(..., issuer="did:example:issuer")` pattern is placeholder-only — no real verifier can resolve `did:example:*`. This document explains how to replace it with a resolvable DID so SUM-issued credentials verify under every W3C-compliant VC 2.0 verifier on the market (DIF Universal Resolver, Digital Bazaar, Spruce ssi, Veramo, Mattr, Microsoft Entra Verified ID).

Two paths. Both are shipped primitives in `internal/infrastructure/verifiable_credential.py`.

---

## Path A — `did:key` (self-resolving, zero hosting)

Best for: ephemeral issuance, air-gapped operation, CLI automation, test fixtures.

The entire public key is encoded in the DID identifier. A verifier with only the DID can reconstruct the public key and verify any credential the corresponding private key signed. No domain to register, no file to host, no DNS to configure.

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from internal.infrastructure.verifiable_credential import (
    ed25519_to_did_key,
    make_credential,
    sign_credential,
)

sk = Ed25519PrivateKey.generate()    # rotate this; don't commit
pk = sk.public_key()
did = ed25519_to_did_key(pk)          # did:key:z6Mk...
vm = f"{did}#{did.split(':')[-1]}"    # the did:key verificationMethod convention

cred = make_credential(
    subject={"axiom": "alice||like||cat"},
    issuer=did,
)
signed = sign_credential(cred, sk, vm)
# → signed["proof"]["verificationMethod"] points at a self-resolving DID;
#   any VC 2.0 verifier derives the public key from the DID itself.
```

Verifier workflow:

1. Parse `signed["proof"]["verificationMethod"]` → extract the `did:key:z...` portion.
2. Decode the multibase-base58btc part → strip the `0xed 0x01` multicodec prefix → 32-byte Ed25519 public key.
3. Re-compute `hashData = SHA-256(JCS(proofConfig)) ‖ SHA-256(JCS(document))`.
4. Verify the Ed25519 signature over `hashData`.

No network request. Perfect for CLI-only flows and for the `sum attest` pipeline when no domain is available.

---

## Path B — `did:web` (domain-resolving, hosted)

Best for: hosted SUM deployments (Cloudflare Pages, GitHub Pages, any static host). The DID is tied to a domain you control; verifiers resolve it by fetching `https://{domain}/.well-known/did.json`.

### Bootstrap

Run once per deployment:

```bash
python -m scripts.generate_did_web \
    --domain sum-demo.pages.dev \
    --out-dir single_file_demo/.well-known \
    --private-key-out keys/did_web_issuer.pem
```

Output:

```
did:web issuer bootstrap complete.
  DID:              did:web:sum-demo.pages.dev
  did:key (AKA):    did:key:z6Mk...
  DID document:     single_file_demo/.well-known/did.json
  did-key sidecar:  single_file_demo/.well-known/did-key.txt
  private key:      keys/did_web_issuer.pem  (DO NOT COMMIT)
```

### What gets committed, what doesn't

| File | Commit? | Notes |
|---|---|---|
| `single_file_demo/.well-known/did.json` | ✅ yes | The public DID document. Served by Cloudflare Pages at `https://{domain}/.well-known/did.json`. |
| `single_file_demo/.well-known/did-key.txt` | ✅ yes | The self-resolving sibling DID (already present as `alsoKnownAs` in `did.json`; the sidecar is for convenience). |
| `keys/did_web_issuer.pem` | ❌ never | Private key. Already covered by `.gitignore` (`*.pem`, `keys/`). Store in Cloudflare Pages environment variables or a secret manager. |

### Cloudflare Pages deployment — enabling `/.well-known/`

Cloudflare Pages serves every file in the output directory (`single_file_demo/` per `README.md`) including dotfile paths like `.well-known/did.json`. No `_redirects` config needed — the path just works.

Verify after deploy:

```bash
curl -sL https://sum-demo.pages.dev/.well-known/did.json | jq .id
# → "did:web:sum-demo.pages.dev"
```

### Signing with the `did:web` private key

```python
import json
from cryptography.hazmat.primitives import serialization
from internal.infrastructure.verifiable_credential import (
    did_web_verification_method,
    make_credential,
    sign_credential,
)

with open("keys/did_web_issuer.pem", "rb") as f:
    sk = serialization.load_pem_private_key(f.read(), password=None)

cred = make_credential(
    subject={"axiom": "alice||like||cat"},
    issuer="did:web:sum-demo.pages.dev",
)
signed = sign_credential(
    cred,
    sk,
    did_web_verification_method("sum-demo.pages.dev"),
)
print(json.dumps(signed, indent=2))
```

The `proof.verificationMethod` is now `did:web:sum-demo.pages.dev#key-1`. Any DIF-conformant verifier will:

1. Resolve the DID by fetching `/.well-known/did.json` from the domain.
2. Extract the `publicKeyMultibase` entry matching the `#key-1` fragment.
3. Decode it back to a 32-byte Ed25519 public key.
4. Verify the signature.

### Rotating keys

Run the generator with a new `--key-id` and append the new verification method to the DID document manually (or re-run generate and add both method entries). Keep the old key active until no outstanding credentials need it; then remove from `verificationMethod` and from `assertionMethod`/`authentication`. The `alsoKnownAs` list can carry multiple `did:key` siblings across rotations.

---

## Verifier compatibility notes (2026)

Tested mentally against the following reference implementations based on the 2026 ecosystem survey (`docs/FEATURE_CATALOG.md`):

| Implementation | `did:web` | `did:key` | Notes |
|---|---|---|---|
| DIF Universal Resolver | ✅ | ✅ | dev.uniresolver.io; hosts both resolvers |
| Digital Bazaar VC-JS | ✅ | ✅ | Reference implementation for `eddsa-jcs-2022` |
| Spruce ssi | ✅ | ✅ | Rust library; CLI verifier |
| Veramo | ✅ | ✅ | TypeScript SDK |
| Microsoft Entra Verified ID | ✅ | ✅ | Enterprise-tier; accepts both |
| Mattr Global | ✅ | ✅ | Commercial VC verifier |

The `DID_CONTEXT` SUM embeds (`https://www.w3.org/ns/did/v1` + `https://w3id.org/security/multikey/v1`) is the W3C-canonical form that all six of the above parse without additional configuration.

---

## What SUM does NOT do

- **DID:web resolution client.** SUM emits DIDs; verifiers resolve them. The `verify_credential(credential, public_key)` surface requires the caller to supply the public key, which is intentional — SUM stays out of HTTP-dependency territory. For end-to-end verification with automatic key resolution, use DIF Universal Resolver or Spruce ssi's CLI and pass the resulting key to `verify_credential`.
- **did:key's `controllerDocument` pseudodocument generation.** The `did:key` spec defines a notional document a resolver reconstructs from the DID. SUM neither generates it nor requires it; the public key is directly derivable from the DID string.
- **DID status lists, revocation, credential-status**. These are VC 2.0 optional and can be added by callers via `make_credential(..., extra_fields={...})`. Not in scope for the issuer helpers.
- **Key management at rest.** The private key sits in a file on disk; encrypting it is your deployment's responsibility. Cloudflare Pages environment variables, Vercel secrets, `sops`, or a hardware-backed keystore are all reasonable.

---

## Security posture

- **Private key disclosure is game over.** Anyone with the `keys/did_web_issuer.pem` can sign bundles as if they were your deployment. Protect accordingly: do not commit, do not share over insecure channels, rotate on any suspected compromise.
- **Domain takeover enables forgery.** `did:web` trusts whoever controls the domain's `/.well-known/` path. If the domain is lost or hijacked, all credentials signed under it become unverifiable or (worse) forgeable. Consider pairing with `did:key` as an `alsoKnownAs` backup so verifiers can fall back to self-resolution.
- **No revocation path.** If the key is compromised, you cannot invalidate already-issued credentials retroactively. Rotate the key, update the DID document to remove the compromised `verificationMethod`, and notify downstream consumers. This is the cost of the design's simplicity — VC 2.0 has optional `credentialStatus` primitives for revocation that SUM does not ship by default.
