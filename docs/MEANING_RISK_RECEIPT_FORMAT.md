# `sum.meaning_risk_receipt.v1` — wire format

*Status: research (behind the `[research]` extra). Companion to
`sum_engine_internal/research/meaning/receipt.py` and the frontier chart
in `docs/MEANING_LOSS_FRONTIER.md`.*

A meaning-risk receipt is a **signed, replayable certificate** that the
expected *meaning-loss* of a transform — measured by a named proxy,
bounded distribution-free — does not exceed a stated ceiling. It reuses
the render/transform-receipt trust stack wholesale (RFC 8785 JCS, RFC
8032 Ed25519, RFC 7515 detached JWS, RFC 7517 JWKS) and adds exactly one
thing the other receipts don't have: a **replay anchor**.

## 1. Envelope

Identical four-key shape to `sum.transform_receipt.v1`:

```json
{
  "schema": "sum.meaning_risk_receipt.v1",
  "kid": "<key id, also in the JWS protected header>",
  "payload": { ... see §2 ... },
  "jws": "<protected>..<signature>"
}
```

The JWS is detached (RFC 7515 §A.5, RFC 7797 `b64:false`), computed over
the JCS-canonical bytes of `payload`. Verification is the shared
six-step algorithm in `infrastructure/jose_envelope.py` (the same one
the render-receipt and trust-root verifiers use).

## 2. Payload fields

| Field | Type | Meaning |
|---|---|---|
| `scorer` | string | name of the meaning-loss proxy (e.g. `"lexical-coverage-bidirectional"`, `"bidirectional-entailment[<judge>]"`). The bound is conditional on this. |
| `scorer_version` | string | pinned version of the proxy. Any change to its internals (stop-words, tokenisation, judge) is a version bump. |
| `loss_definition` | string | one-line human semantics of the proxy's `[0,1]` number. |
| `n` | int | calibration sample size (number of `(source, transform)` pairs). |
| `delta` | float | miscoverage allowance; confidence = `1 − delta`. |
| `method` | string | `"hoeffding"` or `"clopper_pearson"`. |
| `point_estimate` | float | observed mean meaning-loss on the calibration sample (rounded to 6 dp). |
| `risk_upper_bound` | float | the certified `(1−delta)` **upper** bound on `E[loss]` (rounded to 6 dp). The headline number. |
| `losses_hash` | string | `"sha256-<hex>"` over JCS-canonical bytes of the rounded per-pair loss vector. **The replay anchor.** |
| `corpus_id` | string | names the calibration envelope — the exchangeability scope the bound is valid within. |
| `transform` | string | free label of what produced the pairs (e.g. `"slider:density=0.5"`). |
| `not_covered` | string[] | layers the proxy structurally cannot bound. Default: `["arrangement","sound","connotation","implicature"]`. **Required and non-empty.** |
| `disclosure` | string | the proxy/marginal/exchangeability caveat in prose. |
| `signed_at` | string | ISO-8601 UTC, millisecond precision + `Z` (byte-identical to the transform-receipt stamp). |
| `alpha_target` | float | *(optional)* the risk level the operator wanted controlled. |
| `controlled` | bool | *(present iff `alpha_target` is)* whether `risk_upper_bound ≤ alpha_target`. |

## 3. Verification — two stages

**Stage A — cryptographic + disclosure (always).** Run
`verify_jose_envelope` with
`supported_schema="sum.meaning_risk_receipt.v1"`. Confirms the signature,
schema gate, header invariants, and (opt-in) the `signed_at` replay
window. The verifier then enforces the **disclosure invariants** —
`not_covered` must be a non-empty list and `disclosure` a non-empty
string (`MeaningReceiptDisclosureError` otherwise) — so a signed-but-
disclosure-free receipt cannot pass as a bare bound. On success the
payload is *authentic* and *self-disclosing*.

**Stage B — replay (when the losses are supplied side-band).** This is
what this receipt adds. The loss vector is **rounded to 6 dp**
(`_round_losses`) — the exact vector `losses_hash` commits and
`build_payload` certified over — then:

1. **Hash anchor.** Recompute `losses_hash(losses)`; it must equal
   `payload.losses_hash`. *(Confirms the side-band evidence is the
   evidence the receipt committed to.)*
2. **Re-certify.** Re-run `certify_meaning_risk` on the **rounded**
   vector with the payload's `delta` / `method`.
3. **Bound match.** The reproduced `risk_upper_bound` and
   `point_estimate` must equal the payload's (to 6 dp).
4. **Sample-size match.** `payload.n` must equal the number of committed
   losses — an inflated `n` misrepresents finite-sample confidence even
   when the bound is honest.
5. **Decision match.** When `alpha_target` is present, `controlled` is
   recomputed from the replayed bound (`risk_upper_bound ≤ alpha_target`)
   and must equal `payload.controlled` — the operational pass/fail flag
   cannot ride a valid signature while contradicting the bound.

Because `certify_meaning_risk` is deterministic and both producer and
verifier certify over the **same rounded vector**, Stage B reproduces
the bound **byte-for-byte on the same commit** (the rounding is the
single source of truth — re-certifying over raw losses would false-
reject an honest producer who ships the rounded loss file the hash
commits). A receipt whose author hand-edited `risk_upper_bound`,
`controlled`, or `n` to a stronger claim passes Stage A — they signed
their own statement — but **fails Stage B**. That separation
(`SIGNATURE_INVALID` vs `MeaningReceiptReplayError`) is deliberate: it
distinguishes *tampered-in-transit* from *overclaimed-at-issue*.

## 4. Trust scope — what a verified receipt does and does NOT prove

**Proves** (Stage A + Stage B): authentic signature; required disclosure
fields present; the committed losses hash to `losses_hash`; the named
certifier reproduces the bound, `point_estimate`, `n`, and `controlled`
on those losses.

**Does NOT prove:**

- **that meaning was preserved.** Only that a *named proxy* for
  meaning-loss is bounded *on average*. The proxy is named in the
  payload precisely so this is never ambiguous.
- **anything per-document.** The bound is **marginal**, not conditional.
- **which scorer produced the losses.** The `scorer` / `scorer_version`
  fields are **producer-asserted, not attested**. The bound is a pure
  function of the loss *numbers*; nothing cryptographic binds them to
  the scorer that generated them (that would require re-running the
  scorer over the source/transform pairs, not just re-certifying over
  losses). A receipt records *which proxy the issuer claims* — trust in
  the scorer label is trust in the issuer, not in the math.
- **anything about `not_covered` layers** — arrangement (*naẓm*), sound,
  connotation, implicature. The proxy is blind to them; the field says
  so. Validity also rests on **exchangeability** with `corpus_id`.

See `docs/MEANING_LOSS_FRONTIER.md` §5 and `docs/PROOF_BOUNDARY.md`. The
"no guarantee-language without a same-commit replay receipt" rule from
the bench-hardening plan applies here verbatim — and Stage B is that
receipt.

## 5. Forward-compat

Adding or removing a payload field is a schema bump (`...v2`).
Verifiers fail closed on an unknown `schema` (Step 0.5 of the shared
algorithm). The `not_covered` list may *grow* within v1 (declaring more
out-of-scope layers is always safe); it must never be empty.
