# `sum_verify` — the stable, dependency-light receipt verifier

> The small package an integrator pins to **check** SUM receipts —
> without the CLI, the research extras, or a numeric stack.

This is the answer to the most-asked thing in the 30-guest adoption
simulation that wasn't "show me my document": *"give me a small, stable,
non-`[research]` verify surface I can depend on — I'm not going to pin a
research format buried in a 3000-line CLI."* (`sum_verify` is that
surface; the per-document readout is [`sum meaning-diff`](MEANING_LOSS_FRONTIER.md).)

## Install

```bash
pip install "sum-engine[verify]"
```

That pulls `joserfc` (the detached-JWS / RFC 7797 machinery) on top of the
package's base `cryptography` (for Ed25519) and `sympy` (used by the
state-integer path, not imported by `sum_verify`). **No numpy, scipy, or torch** — verifying a meaning-risk
receipt *replays its conformal bound offline* through a pure-Python
re-derivation of the same inequality (`sum_verify/_conformal.py`). The
property is pinned in a clean subprocess by `Tests/test_sum_verify_sdk.py`.

## Use it as a library

```python
import json
from sum_verify import verify

receipt = json.load(open("receipt.json"))
jwks    = json.load(open("jwks.json"))     # issuer's /.well-known/jwks.json

# Signature + structural (disclosure) checks only:
payload = verify(receipt, jwks)

# Meaning-risk receipts ALSO replay the bound offline when handed the
# committed per-pair losses side-band (bare list or {"losses": [...]}):
losses  = json.load(open("losses.json"))
payload = verify(receipt, jwks, losses=losses)
```

`verify()` dispatches on the envelope's `schema` field. It returns the
verified payload dict (meaning-risk) or a `VerifyResult` whose `.payload`
carries the body (render / transform). On failure it raises — see
*Errors* below.

## Explicit offline trust policy

The default `verify()` remains a signature/structure check against supplied
keys, preserving historical receipt acceptance and return types. A relying
application can add `trust_policy=TrustPolicy(...)`, or use `verify_report()`
for individual `passed` / `not_checked` outcomes. Reports snapshot their payload
and check metadata at verification time; their `result` field preserves the
legacy verifier return object. Failures raise
`TrustPolicyError` (also a `SumVerifyError`), with a machine-readable `check`.

```python
from sum_verify import (
    RevocationSnapshot, TrustPolicy, key_fingerprint, verify_report,
)

# These three values come from the relying application's trusted configuration
# and recorded acquisition process, not the untrusted receipt packet:
# approved_jwks, revocation_document, revocations_retrieved_at
policy = TrustPolicy(
    trusted_keys={
        key["kid"]: key_fingerprint(key) for key in approved_jwks["keys"]
    },
    revocations=RevocationSnapshot.from_document(
        revocation_document, retrieved_at=revocations_retrieved_at,
    ),
    require_revocations=True,
    max_revocation_age_seconds=3600,
    max_age_seconds=300,
)
report = verify_report(receipt, jwks, trust_policy=policy)
print(report.to_dict())
```

The trusted catalog pins both Ed25519 public key material and its approved
key ID; an attacker-chosen `kid` cannot rename a revoked key into acceptance.
An empty catalog rejects every key. Revocation checks reject every known alias
of revoked material. Retain historical keys in the trusted catalog: a snapshot
with unresolved revoked IDs fails closed with `revocation_key_unresolved`,
because the verifier cannot otherwise determine their material or aliases. The caller
must authenticate its approved keys and revocation document; the SDK does not
fetch them or authenticate an organization's identity. A receipt packet's own
JWKS is useful for signature checking but cannot establish that independent
trust decision.

`revocations=None` means **not checked**. An explicitly supplied empty snapshot
means the key was absent from that snapshot. `require_revocations=True` or a
snapshot age limit fails closed if the snapshot is missing. Snapshot freshness
uses the caller-recorded retrieval time, not a field supplied by the signer;
that clock and the snapshot's authenticity remain caller responsibilities.
Older cached snapshots cannot establish whether a newer revocation exists.

`archival=True` deliberately leaves receipt freshness **not checked** and is
mutually exclusive with an age limit. It does not prove historical existence
or waive revocation: the policy rejects every listed key even when `signed_at`
claims a pre-compromise time. A compromised key can sign a backdated receipt.
Independent timestamp/transparency evidence and organizational archival policy
are outside this verifier. Direct render and transform `revoked_kids=` remain
legacy effective-time checks; use `TrustPolicy` for the stronger policy above.

For artifact commitments, use `expected_bindings={"input_hash": computed_hash}`
and optionally `required_bindings={"input_hash"}`. Values must be independently
computed using that receipt family's canonicalization rules, not copied from
the receipt. Required bindings need supplied comparison values; merely having
a signed hash is insufficient. Supported names are signed top-level `*_hash`
fields; using a field absent from that family fails. This checks hash equality,
not source retrieval, conversion fidelity, model remeasurement, or truth.

The generic chain path applies policy to the outer envelope. For supplied
hops, verify each with its own policy and use
`verify_chain_receipt(hop_envelopes=...)` to check the chain's commitments.
The report marks absent hop checks explicitly. Meaning-risk loss replay is
reported independently from source remeasurement and sampling assumptions,
which remain **not checked**. Historical receipts without `statistical_scope`
report `not_declared`; their signed bytes are not rewritten.

## Use it from the shell

```bash
python -m sum_verify receipt.json --jwks jwks.json [--losses losses.json]
# → {"verified": true, "schema": "...", "replayed": true, ...}   (exit 0)
# → {"verified": false, "error": "...", "detail": "..."}          (exit 1)
```

For richer output (perspective cohorts, the layered explanation) use the
full `sum verify-meaning` CLI; `python -m sum_verify` is deliberately
tiny.

## Supported schemas

`sum_verify.SUPPORTED_SCHEMAS`:

| schema | what it carries | offline bound replay |
| --- | --- | --- |
| `sum.meaning_risk_receipt.v1` | signed, conformal bound on a named meaning-loss proxy (the flagship) | ✅ with `losses=` |
| `sum.render_receipt.v1` | signed render provenance | — (no replayable bound) |
| `sum.transform_receipt.v1` | signed transform provenance | — |
| `sum.chain_receipt.v1` | ordered chain of meaning-risk receipts + integer-exact Bonferroni budget + optional direct end-to-end bound | ✅ hops via `verify_chain_receipt(hop_envelopes=…)`; end-to-end leg via `end_to_end_losses=…` (the `verify()` dispatcher runs the always-on checks; full side-band goes through `verify_chain_receipt` / `python -m sum_verify --hops`) |

Group-conditional **perspective** receipts
(`sum.perspective_risk_receipt.v1`) are verified by the full
`sum verify-meaning` CLI today; folding them into this SDK is the natural
next increment.

## Errors

| exception | meaning |
| --- | --- |
| `JoseEnvelopeError` | cryptographic / structural failure on a meaning-risk envelope |
| `ReceiptVerifyError` | cryptographic / structural failure on a render / transform receipt |
| `ChainReceiptReplayError` / `ChainReceiptDisclosureError` | chain receipt: side-band does not reproduce the committed hashes/sums/bound, or a required disclosure (`not_covered` / `disclosure` / `budget_scope`) is missing |
| `MeaningReceiptDisclosureError` | signature valid, but the receipt omits a required disclosure (`not_covered` / `disclosure`) — a bare bound is refused |
| `MeaningReceiptReplayError` | signature valid, but the supplied losses don't reproduce the committed hash / bound / `n` / `controlled` |
| `TrustPolicyError` | valid signed receipt rejected by a requested key, revocation, freshness, or artifact-binding check |
| `UnsupportedSchemaError` | the envelope's `schema` is not one this SDK accepts |

## What a verified receipt proves — and does NOT

**Proves:** the payload was signed by the holder of `kid`'s private key;
the envelope is well-formed and (optionally) unexpired; and — for a
meaning-risk receipt replayed with its losses — that the committed losses
hash to the anchor and re-certify to the stated bound by *exact integer
equality* on the micro-unit wire grid.

**Does NOT prove that meaning was preserved.** A meaning-risk receipt
bounds a **named proxy** for meaning-loss, **marginally** (the average
over the calibration corpus, never per-document). Interpreting the bound
as a population statement requires the named method's assumptions, including
independent calibration draws from the target distribution under a fixed
evaluation policy; exchangeability alone is insufficient. Arithmetic replay
does not validate those assumptions. It says nothing
about the layers its `not_covered` field declares out of scope —
arrangement, sound, connotation, implicature. The verifier *enforces*
that those disclosures are present; it does not let a bare bound through.
See [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) and
[`docs/MEANING_RISK_RECEIPT_FORMAT.md`](MEANING_RISK_RECEIPT_FORMAT.md).

## Stability promise

`sum_verify.__version__` (SemVer) tracks **this module's public surface**
and the receipt wire formats it accepts — independent of the engine's
release version. An engine release that doesn't change a supported wire
format does not bump it. The names in `sum_verify.__all__` are the
pinnable contract; a backwards-incompatible change to any of them, or to
an accepted format, is a major bump.

## Why a separate package (the design note)

The verification path is intentionally a near-clean-room reimplementation
of the conformal arithmetic, *not* a thin re-export of the generator's
kernels. Two implementations of the same bound is a divergence hazard, so
the discipline here is to make divergence **loud**: the parity grid and
golden-equivalence tests in `Tests/test_sum_verify_sdk.py` assert the
pure-Python kernels agree with the canonical numpy/scipy ones to far
inside the 1e-6 wire grid, and that the committed golden receipts replay
*identically* through both paths. The cryptographic trust root (RFC-8785
JCS, Ed25519/JWS) is shared verbatim from
`sum_engine_internal.infrastructure` — reimplementing *that* would be real
risk with no upside. The wire contract (integer micro-units, exact-integer
replay comparison) is the stable interface both sides honour.
