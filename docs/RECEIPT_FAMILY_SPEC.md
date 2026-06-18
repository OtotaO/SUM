# SUM Receipt Family — unified specification (v1 overview)

*Status: consolidating overview, 2026-06-08. This document is the single
entry point to SUM's four transformation-receipt schemas — what they share,
how they differ, and what each does and does not prove. It is an **overview
that references the per-schema format docs as the authoritative source of
truth**; it deliberately does not restate their exact field algorithms, to
avoid becoming a second, drifting copy. Where this doc and a per-schema
format doc disagree, the format doc wins. Positioning defers to
[`PRODUCT_VISION.md`](PRODUCT_VISION.md) §0; the proved/measured boundary
defers to [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md).*

---

## 1. What this family is

SUM emits a **signed, portable, offline-verifiable receipt for each
transformation of text** — chain-of-custody for AI-transformed text,
robust to rewriting in a way image-centric provenance (C2PA/SynthID) is
not. The receipt does not assert the output is *true*; it attests *what
transformation was performed and what it preserved*, so any party — with
no access to SUM's servers — can verify the claim. The standing line:
**attest, don't detect** (`PRODUCT_VISION.md` §0). General "is this AI?"
detection is a paraphrase-defeated liability; a signature is not.

The family has **four schemas**, in two tiers:

| Schema | Tier | Attests |
|---|---|---|
| `sum.render_receipt.v1` | provenance | a tome was rendered from triples under named slider settings, by a named model/provider |
| `sum.transform_receipt.v1` | provenance | a named transform mapped an input (hash) to an output (hash) under named parameters (hash) |
| `sum.meaning_risk_receipt.v1` | provenance **+ measured bound** | the above **plus** a distribution-free upper bound on expected meaning-loss under a *named proxy*, over a named corpus |
| `sum.perspective_risk_receipt.v1` | provenance **+ measured bound** | the meaning-risk bound **per declared cohort** (novice / expert / regulator / …), optionally with a joint (Bonferroni) guarantee |

The **provenance tier** is a cryptographic attestation (what happened).
The **measured-bound tier** adds a conformal certificate that is *replayable*
but bounds a *named proxy marginally under exchangeability* — never a
per-document truth claim. The two tiers must never be conflated; §6.

Supporting infrastructure (own specs, out of scope here): the trust root
([`TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md)), the audit log
([`AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md)), and Merkle inclusion
sidecars ([`MERKLE_SIDECAR_FORMAT.md`](MERKLE_SIDECAR_FORMAT.md)).

## 2. The shared cryptographic model

Every receipt in the family is the **same envelope shape** over a
schema-specific payload:

- **Signature.** Ed25519 (RFC 8032) over the **JCS-canonical** bytes
  (RFC 8785) of the payload, carried as a **detached JWS** (RFC 7515).
- **Key distribution.** Public keys as a **JWKS** (RFC 7517) at
  `/.well-known/jwks.json`; the signing key identified by `kid` in the JWS
  header. The current live key is `sum-render-2026-04-27-1`.
- **Trust root.** A signed manifest at
  `/.well-known/sum-trust-root.json` binds the JWKS, key-rotation cadence,
  and revocation surface (`/.well-known/revoked-kids.json`); see
  `TRUST_ROOT_FORMAT.md`.
- **Envelope.** The wire object carries `{schema, kid, payload, jws}` (or
  the receipt's documented equivalent); verifiers gate on `schema` first
  and **fail closed on an unknown schema**.

This is the same trust triangle the render receipts established (Python ↔
Node ↔ browser, byte-identical), now shared by all four. A verifier that
can check one can check all four's signatures with the same primitive.

### 2.1 The float discipline (the load-bearing canonicalisation rule)

JCS normalises integer-valued floats (`1.0` → `1`), and the meaning-receipt
JS canonicaliser **rejects floats outright**. The family therefore splits:

- **Conformal receipts (`meaning_risk`, `perspective`) are strictly
  float-free** — every numeric quantity is an **integer micro-unit**
  (resolution 1e-6: `delta_micro`, `risk_upper_bound_micro`, …). This is
  required because their replay (§5) demands **exact integer equality** of
  a re-certified bound, and because the float-free wire is what
  canonicalises byte-identically across Python and Node.
- **Provenance receipts (`render`, `transform`)** carry JCS-normalised
  quantized slider floats in the payload; their replay is **hash-based**
  (`input_hash` / `output_hash` / `parameters_hash` / `triples_hash` /
  `tome_hash`), not exact-rate-equality, so JCS float normalisation is
  safe for them.

**Rule for any future numeric field in a signed payload:** encode it
float-free (integer micro-units or a hash), or a Node verifier cannot
canonicalise it. (Exact canonicalisation rules: `CANONICAL_ABI_SPEC.md`,
`RENDER_RECEIPT_FORMAT.md` §4.)

## 3. The four payloads (overview — format docs are authoritative)

Field names are reproduced for orientation; the cited format doc is the
source of truth for types, semantics, and optionality.

### 3.1 `sum.render_receipt.v1` → [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md)

`render_id`, `sliders_quantized{density,length,formality,audience,perspective}`,
`triples_hash`, `tome_hash`, `model`, `provider`, `signed_at`,
`digital_source_type` (C2PA, v2.4).

### 3.2 `sum.transform_receipt.v1` → [`TRANSFORM_RECEIPT_FORMAT.md`](TRANSFORM_RECEIPT_FORMAT.md)

`transform_id`, `transform`, `parameters_hash`, `input_hash`, `output_hash`,
`model`, `provider`, `signed_at`, `digital_source_type`, optional
`source_chain_hash` (T4 — binds the receipt to specific source byte ranges).
Generalises the render receipt to any registered transform
([`TRANSFORM_REGISTRY.md`](TRANSFORM_REGISTRY.md)).

### 3.3 `sum.meaning_risk_receipt.v1` → [`MEANING_RISK_RECEIPT_FORMAT.md`](MEANING_RISK_RECEIPT_FORMAT.md)

`scorer`, `scorer_version`, `loss_definition`, `n`, `delta_micro`, `method`
(∈ `hoeffding` / `clopper_pearson` / `empirical_bernstein`),
`point_estimate_micro`, `risk_upper_bound_micro`, `losses_hash`,
`corpus_id`, `transform`, `not_covered` (non-empty), `disclosure`
(non-empty), `signed_at`, optional `alpha_target_micro` + `controlled`.

### 3.4 `sum.perspective_risk_receipt.v1` → [`MEANING_RISK_RECEIPT_FORMAT.md`](MEANING_RISK_RECEIPT_FORMAT.md) §6

The meaning-risk fields **per cohort**: `simultaneous`, `evidence_hash`,
`marginal_point_estimate_micro`, `marginal_risk_upper_bound_micro`,
`groups[]` (`{group_id, n, point_estimate_micro, risk_upper_bound_micro
[, controlled]}`), optional `controls_all`. `simultaneous=true` ⇒ each
cohort certified at `delta/G` (Bonferroni) so all cohort bounds hold
jointly at ≥ 1−δ.

### 3.5 `sum.study_artifact.v1` — a container, not a receipt

Emitted by `sum study` (the verifiable cheatsheet; see
[`MACHINE_STUDYING_APPLICABILITY.md`](MACHINE_STUDYING_APPLICABILITY.md)).
It is **not itself a signed wire object** — it carries the studied corpus's
`state_integer` + `axiom_count`, the rendered `cheatsheet`, the
`RenderFrontier` it sits on, an `expertise` MEASUREMENT, and — when
`--certify` is used — an **embedded** `sum.meaning_risk_receipt.v1` under
`receipt` (the only certified element; verify it with the §3.3 path). The
`expertise` scalar and frontier losses are per-run measurements, never
guarantees; the `measurement_note` field says so. A consumer gates trust on
the embedded receipt, not on the container.

## 4. Verification model

Two stages, applied uniformly; the second only exists for the conformal
tier.

- **Stage A — cryptographic + disclosure (all four, every runtime).**
  Verify the JWS signature; gate the `schema`; check header invariants;
  for the conformal tier, enforce the **disclosure invariants**
  (`not_covered` non-empty, `disclosure` non-empty). On success the
  payload is *authentic* (and, for the conformal tier, *self-disclosing*).
- **Stage B — replay (conformal tier only, when evidence is supplied
  side-band).** Recompute the evidence hash (`losses_hash` /
  `evidence_hash`) over the integer-micro vector; re-run the certifier;
  require the reproduced bound(s), `point_estimate`, `n`, and
  `controlled`/`controls_all` to match by **exact integer equality**. This
  catches an *overclaimed-at-issue* receipt (valid signature over a
  hand-edited bound) — distinct from a *tampered-in-transit* one
  (`MeaningReceiptReplayError` vs `SIGNATURE_INVALID`).

### 4.1 Cross-runtime matrix (honest scope)

| Receipt | Stage A (sig/schema/disclosure) | Stage B (replay) |
|---|---|---|
| `render` / `transform` | Python · Node (`standalone_verifier`) · browser (`single_file_demo`) | hash comparison, any runtime |
| `meaning_risk` / `perspective` | Python · Node/browser (`single_file_demo/meaning_receipt_verifier.js`) | **Python only** (`verify_meaning_risk_receipt`); for a **model-judge** scorer, additionally **machine-pinned** (cross-hardware float drift in the judge) |

"Verifiable in every runtime" means the **signature and format**, not
necessarily the **meaning recomputation**. The conformal receipts state
this in their own scope banners.

## 5. Trust scope — what the family does and does NOT prove

**Proves** (cryptographic, [provable]): the named issuer signed this exact
payload; the bound evidence hashes to its anchor; for the conformal tier,
the named certifier reproduces the bound on those losses. Nobody
paraphrases around a signature — this is the provenance moat and the
EU AI Act Art 50 disclosure surface.

> **Two load-bearing preconditions (state these wherever "tamper-evident,
> offline-verifiable" is claimed):**
> 1. **Verification reduces to trusting the JWKS.** Receipts do *not* embed
>    their own keys; the verifier checks the signature against a
>    caller-supplied JWKS. As with any JWS system, a receipt forged with an
>    attacker's key *will* verify against an attacker-supplied JWKS — so the
>    JWKS MUST be obtained from a trusted root **out-of-band**
>    (`/.well-known/sum-trust-root.json` → JWKS; see
>    [`TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md)), never from the receipt
>    bundle. "Offline-verifiable" means *verifiable given a trusted JWKS*.
> 2. **Stage A attests the issuer; only Stage B attests the bound.** The
>    Node/browser verifiers are Stage-A (signature + schema + disclosure).
>    A self-signed receipt that commits an honest `losses_hash` but a
>    *fabricated* `risk_upper_bound_micro` passes Stage A — it is caught
>    only by **Stage-B replay** (Python, re-certifying over the side-band
>    loss vector). So a conformal bound is trustworthy only after Stage B on
>    a matching judge stack; a JS-only consumer gets issuer-attestation, not
>    bound-attestation.

**Does NOT prove** (the boundary that keeps the family honest):

- **Output truth / accuracy / freshness.** A receipt attests a
  transformation, not that its output is correct.
- **Issuer honesty.** A verified signature means *this issuer said this*,
  not that the issuer is trustworthy. `scorer` / `model` / `provider`
  labels are **producer-asserted, not attested**.
- **That meaning was preserved.** The conformal tier bounds a *named
  proxy* for meaning-loss, **marginally**, under **exchangeability** with
  the named corpus — never per-document, never the layers `not_covered`
  declares out of reach (arrangement, sound, connotation, implicature).
- **AI-vs-human authorship.** SUM does not ship a detection number; any
  such signal is **advisory — measured, not proved** (`PRODUCT_VISION.md`
  §0). The family is provenance, not detection.

Full discipline: `PROOF_BOUNDARY.md` (§1.8 render binding; §2.11 the
meaning/perspective family).

## 6. Conformance & versioning

- A conformant **verifier** MUST implement Stage A for the schemas it
  accepts, MUST fail closed on an unknown `schema`, and MUST enforce the
  disclosure invariants before honouring a conformal-tier receipt.
- A conformant **issuer** MUST sign over JCS-canonical bytes, MUST keep
  conformal payloads float-free, and MUST populate `not_covered` /
  `disclosure` non-empty for the conformal tier.
- **Versioning.** Adding or removing a payload field is a schema bump
  (`…v2`); `not_covered` may *grow* within v1. Verifiers fail closed on an
  unknown schema, so old verifiers never silently mis-accept new receipts.

## 7. References

- RFC 8032 (Ed25519), RFC 8785 (JCS), RFC 7515 (JWS), RFC 7517 (JWKS).
- C2PA `digitalSourceType` taxonomy, spec.c2pa.org v2.4.
- Per-schema format docs: `RENDER_RECEIPT_FORMAT.md`,
  `TRANSFORM_RECEIPT_FORMAT.md`, `MEANING_RISK_RECEIPT_FORMAT.md` (incl. §6
  perspective).
- `PROOF_BOUNDARY.md` (claim arbiter), `PRODUCT_VISION.md` §0
  (positioning), `TRUST_ROOT_FORMAT.md`, `CANONICAL_ABI_SPEC.md`.

---

*This overview changes no claim and adds no schema. It exists so a newcomer
— a standards reader, a grant reviewer, an agent developer — can see the
whole chain-of-custody family at once and know exactly where the proved /
measured line falls, then descend to the authoritative per-schema doc.*
