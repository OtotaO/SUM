# Security policy

SUM is a verifiability project. The integrity of its trust surfaces — render-receipt signing, the cross-runtime verifier triangle, the PyPI release attestation, the JWKS rotation contract — is the product. A real vulnerability in any of these is the kind of finding we want to fix faster than we want to hide.

## Supported versions

| Version | Supported |
|---|---|
| `sum-engine` ≥ 0.3.x on PyPI | ✅ |
| Earlier `0.x` releases | best-effort; please upgrade |
| `main` branch in this repo | ✅ |
| Cloudflare Worker at `sum-demo.ototao.workers.dev` | ✅ |
| Single-file demo committed under `single_file_demo/` | ✅ |

The cross-runtime trust surfaces — Python `sum verify`, Node `standalone_verifier/verify.js`, the browser receipt verifier in `single_file_demo/`, and the Python receipt verifier in `sum_engine_internal/render_receipt/` — are all in scope.

## How to report

**Preferred: GitHub private security advisory.**
Open one at https://github.com/OtotaO/SUM/security/advisories/new. The advisory thread keeps the report private until coordinated disclosure, and lets us track the fix + CVE assignment in one place. This is the canonical channel.

**Alternate: email.**
Send a report to `ototao@pm.me` with subject prefix `[SUM-SECURITY]`. Plain text is fine; PGP encryption is welcome but not required (request the public key in a separate, unencrypted message if you want it).

**Please do NOT:**
- Open a public GitHub issue describing an unfixed vulnerability.
- Tweet, post, or otherwise publicly disclose before the embargo lifts.
- Run intrusive tests against `sum-demo.ototao.workers.dev` (see "Out of scope" below).

## What to include

Whatever you have. A minimal report we can act on contains:
1. **What you found.** One paragraph describing the vulnerability class (signature bypass, canonicalisation divergence, key compromise, CSRF on the demo, etc.).
2. **How to reproduce.** Specific bytes, fixture, or curl command. The smaller the repro, the faster we fix.
3. **What you think the impact is.** The trust contract this breaks (e.g., "verifier accepts a tampered receipt as valid" vs. "demo loads jose from a non-vendored source").
4. **Whether you've disclosed it elsewhere.** If yes, where; if no, please hold disclosure until the embargo period.

We will respond — even if just to acknowledge — within **5 business days** for the initial intake. Typical investigation + fix windows:

- Hot vulnerabilities (signing-key compromise, accepted-but-tampered receipt, canonicalisation divergence): patch + advisory within **14 days**.
- Lower-severity issues (best-practice deviations, infrastructure hardening gaps, doc drift): triaged within **30 days**, fix on the next normal cycle.

## What happens after you report

1. We confirm we received your report and either accept it as a security issue or explain why it isn't.
2. We open (with your permission) a private GitHub Security Advisory and add you as a collaborator if you want commit attribution.
3. We work on a fix. You're welcome but not required to participate.
4. We coordinate a disclosure date. Default embargo is 90 days from confirmation; we'll shorten or extend with reasonable justification.
5. We publish the advisory + fix together. Credit to you in the advisory unless you prefer otherwise.

## What's in scope

- Cryptographic invariants of the cross-runtime trust triangle: any fixture or input that causes Python ↔ Node ↔ browser verifiers to disagree on validity (positive **or** negative path).
- Render-receipt verifier flaws: any tampered receipt that any of the three verifiers accepts as valid; any well-formed receipt any verifier rejects.
- PyPI release-attestation flaws: any path by which a published artifact's PEP 740 attestation passes verification but doesn't actually trace back to the expected `OtotaO/SUM` repo + workflow.
- Worker deployment flaws: any path by which a `/api/render` or `/.well-known/jwks.json` response could be served by something other than the published Worker code.
- Canonicalisation flaws: any input that causes the Python `jcs` module and the JS `canonicalize` library to produce different bytes for the same JSON value (within the input shapes SUM uses).
- Cross-site issues against the live demo (`sum-demo.ototao.workers.dev`).

## What's out of scope

- Findings against unsupported / earlier `0.x` releases (please upgrade and re-test).
- Reports about the cryptographic primitives themselves (Ed25519, SHA-256). If those break, the world has bigger problems than SUM.
- Unproven theoretical concerns ("Ed25519 is not quantum-safe" — yes, see [`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md) G3 crypto-agility).
- Bug-bounty-style intrusive testing against `sum-demo.ototao.workers.dev`. The Worker has no rate-limit gate today (Priority 5 in the playbook); please don't be the reason it gets one urgently. Read-only probes (curl, fetch with normal browser UA) are fine; volumetric or fuzzing probes are not.
- Issues with third-party dependencies that don't manifest in SUM. Report those upstream.

## Operator response runbook

For SUM operators (the person running `sum-demo.ototao.workers.dev` or publishing `sum-engine` releases) handling a confirmed incident, see [`docs/INCIDENT_RESPONSE.md`](docs/INCIDENT_RESPONSE.md). It covers the eight failure modes the existing trust surface admits are possible, with a four-field response per case (Detection / Containment / Revocation-or-correction / Public disclosure wording).

## Our honesty contract

A signed render receipt asserts that the issuer signed a specific (tome, triples, sliders, model, time) tuple. It does **not** assert factual truth, freshness, or issuer honesty. SUM's whole project is being precise about what is proved, what is measured, and what is asserted but unproved (see [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md)). A vulnerability that breaks the proved layer is severity-1; one that breaks a measured number is severity-2; one that affects only an asserted-but-unproved layer is a finding worth filing but not a security issue.

If you're unsure whether your finding is a security issue or an honesty issue, file it anyway — we'd rather over-respond than under-respond on this surface.
