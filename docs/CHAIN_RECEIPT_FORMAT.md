# `sum.chain_receipt.v1` — the certified chain (wire format)

**Status: shipped 2026-07-10, `[research]` issuance / dependency-light
verification.** The chain-level capstone the drift-budget work seeded
([`docs/DRIFT_BUDGET.md`](DRIFT_BUDGET.md)): one signed, replayable
certificate over an ORDERED sequence of `sum.meaning_risk_receipt.v1`
certificates — what a multi-hop transformation chain (summarised →
translated → re-summarised; agent → agent → agent) did to meaning, as far
as that can honestly be certified.

Issue: `sum mint-chain --hop hop1.json --hop hop2.json ... --out chain.json`
(self-verifies before handing you the file).
Verify: `sum_verify.verify_chain_receipt(...)` or
`python -m sum_verify chain.json --jwks jwks.json --hops hop1.json hop2.json
[--losses e2e_losses.json]`.

## 1. Envelope

The family-standard four-key envelope (`{schema, kid, payload, jws}`) —
Ed25519 (RFC 8032) over RFC 8785 JCS-canonical payload bytes, detached JWS
(RFC 7515 / 7797 `b64=false`), JWKS key distribution. Nothing new
cryptographically; see [`RECEIPT_FAMILY_SPEC.md`](RECEIPT_FAMILY_SPEC.md) §2.

## 2. Payload

All quantities are float-free (integer micro-units, 1e-6), per the family
wire rule.

| Field | Type | Meaning |
|---|---|---|
| `chain_id` | str | First 16 hex of sha256 over the JCS bytes of the ordered `receipt_hash` list — binds the ORDER |
| `n_hops` | int | == `len(hops)`, ≥ 2 |
| `hops[]` | list | One entry per hop, in transformation order |
| `hops[].index` | int | 1-based, strictly sequential |
| `hops[].receipt_hash` | str | `sha256-<hex>` over the JCS-canonical bytes of the FULL hop envelope |
| `hops[].schema` | str | `sum.meaning_risk_receipt.v1` (v1 chains are meaning-risk hops only) |
| `hops[].risk_upper_bound_micro`, `.delta_micro`, `.n`, `.method`, `.corpus_id`, `.transform`, `.scorer` | mirrors | Copied verbatim from the hop payload; replay proves each mirror equals the referenced payload exactly |
| `composition_rule` | str | `bonferroni_additive.v1` |
| `budget_micro` | int | == Σ `hops[].risk_upper_bound_micro` (integer-exact) |
| `joint_delta_micro` | int | == Σ `hops[].delta_micro` |
| `budget_scope` | str | MANDATORY honesty statement (below); verifier fails closed without it |
| `end_to_end` | obj? | OPTIONAL direct source→final certification: `scorer, scorer_version, loss_definition, n, method, delta_micro, point_estimate_micro, risk_upper_bound_micro, losses_hash` — same machinery as a meaning-risk bound, with its own replay anchor |
| `not_covered` | list | non-empty, fail-closed |
| `disclosure` | str | visible text, fail-closed |
| `signed_at` | str | ms-precision ISO-8601 Z |

## 3. The two legs, and why they are never conflated

**The additive budget (provable).** By the Bonferroni union bound, with
confidence ≥ `1 − joint_delta`, the SUM of per-hop expected proxy losses is
≤ `budget`. This is exactly
`drift_budget.compose_drift_budget_from_payloads` — integer-exact against
the hop payloads, re-summed on every verify.

**What the budget does NOT bound: end-to-end loss.** The meaning-loss
proxy is a directed loss, not a metric — no triangle inequality holds in
either direction. Empirically both regimes occur: chains recover meaning
(additive over-counts) and chains compound brittleness (additive
under-counts); [`DRIFT_BUDGET.md`](DRIFT_BUDGET.md) measures both. So the
end-to-end claim, when wanted, is MEASURED DIRECTLY: the optional
`end_to_end` leg certifies source→final losses with its own hash anchor
and bound, replayable exactly like a meaning-risk receipt.

The `budget_scope` field carries this statement inside the signed payload,
and every verifier surface fails closed if it is missing — composition is
where a reader will most want to over-read, so the honesty line is
structural, not decorative.

## 4. Verification

Stage A (always): JOSE verification (signature, schema, header
invariants, optional replay window); disclosure invariants (`not_covered`,
`disclosure`, `budget_scope`); payload-internal replay — ordered indices,
`n_hops`, integer-exact `budget_micro` / `joint_delta_micro` re-sums,
`chain_id` re-derivation.

Stage B (side-band, any combination):

- **`hop_envelopes` (ordered):** each hashes to its committed
  `receipt_hash`; each hop envelope itself VERIFIES (signature +
  disclosures, WITHOUT the replay window — `max_age_seconds` applies to
  the chain envelope only, since hops predate the chain by construction)
  against the supplied JWKS — one JWKS may carry multiple
  kids, so multi-issuer chains verify with a merged JWKS; every mirrored
  field equals the hop payload exactly. (Each hop's own LOSS replay stays
  independently available via `verify_meaning_risk_receipt` with that
  hop's side-band losses — a chain receipt adds structure and composition,
  it does not re-litigate per-hop evidence.)
- **`end_to_end_losses`:** hash anchor match, re-certification over the
  quantised committed vector, integer-exact bound / point-estimate / n.

Errors: `ChainReceiptReplayError` / `ChainReceiptDisclosureError`
(both `SumVerifyError` subclasses). Cross-runtime: the JS verifier does
not implement this schema in v1 and fails closed on it
(`SCHEMA_UNKNOWN`), per the family's unknown-schema rule.

## 5. Trust scope

A verified chain receipt proves: the issuer's key signed exactly this
chain structure; the named hops (by exact bytes) are the chain's hops in
this order; the budget and joint-delta are the true integer sums of the
hop bounds; when replayed with side-band, the hops are genuine verified
certificates and the end-to-end leg's bound reproduces from its committed
losses.

It does NOT prove: anything about meaning beyond what each hop's named
proxy bounds (all per-hop caveats apply unchanged — proxy, marginal,
exchangeability); that the additive budget bounds end-to-end loss (§3);
that any single document stayed within any bound (marginal, never
per-document); or that the hop corpora/transforms actually connect into a
semantically meaningful pipeline — the chain binds what the issuer
CLAIMED is a pipeline; whether hop k's output distribution is hop k+1's
input distribution is an exchangeability question outside the signature.
