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

That pulls exactly two runtime dependencies: `cryptography` (already a
SUM core dep, for Ed25519) and `joserfc` (the detached-JWS / RFC 7797
machinery). **No numpy, scipy, or torch** — verifying a meaning-risk
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

Group-conditional **perspective** receipts
(`sum.perspective_risk_receipt.v1`) are verified by the full
`sum verify-meaning` CLI today; folding them into this SDK is the natural
next increment.

## Errors

| exception | meaning |
| --- | --- |
| `JoseEnvelopeError` | cryptographic / structural failure on a meaning-risk envelope |
| `ReceiptVerifyError` | cryptographic / structural failure on a render / transform receipt |
| `MeaningReceiptDisclosureError` | signature valid, but the receipt omits a required disclosure (`not_covered` / `disclosure`) — a bare bound is refused |
| `MeaningReceiptReplayError` | signature valid, but the supplied losses don't reproduce the committed hash / bound / `n` / `controlled` |
| `UnsupportedSchemaError` | the envelope's `schema` is not one this SDK accepts |

## What a verified receipt proves — and does NOT

**Proves:** the payload was signed by the holder of `kid`'s private key;
the envelope is well-formed and (optionally) unexpired; and — for a
meaning-risk receipt replayed with its losses — that the committed losses
hash to the anchor and re-certify to the stated bound by *exact integer
equality* on the micro-unit wire grid.

**Does NOT prove that meaning was preserved.** A meaning-risk receipt
bounds a **named proxy** for meaning-loss, **marginally** (the average
over the calibration corpus, never per-document), and only **under
exchangeability** between that corpus and deployment. It says nothing
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
