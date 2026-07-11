# The Meaning Assertion — an interoperability profile (DRAFT v0.1, 2026-07-10)

**Status: DRAFT proposal.** This document is an interoperability *profile*, not
a ratified standard and not yet submitted to any standards body. It exists to
plant a precise, citable definition of the one assertion the forming
receipt/provenance ecosystem does not carry: a **signed, replayable,
statistically-bounded statement about what a transformation did to a text's
MEANING**. Reference implementation: `sum-engine` (`pip install
"sum-engine[verify]"`), wire format `sum.meaning_risk_receipt.v1`
([`docs/MEANING_RISK_RECEIPT_FORMAT.md`](MEANING_RISK_RECEIPT_FORMAT.md),
[`docs/RECEIPT_FAMILY_SPEC.md`](RECEIPT_FAMILY_SPEC.md)).

## 1. Why this profile exists

The 2026 receipt ecosystem signs **occurrence**: C2PA 2.4 binds asset bytes
(including unstructured text) and identity; agent-receipt Internet-Drafts
(ACTA: `draft-farley-acta-signed-receipts`; ASQAV:
`draft-marques-asqav-compliance-receipts`) sign that a decision or action
happened, with result digests and EU-AI-Act compliance profiles; TEE gateways
attest which model ran. All of these — correctly and explicitly — disclaim
the *semantics* of the content they cover. C2PA's own documentation states a
valid manifest does not certify that assertions are true of the content's
meaning.

That leaves the question every disputant actually asks — *"is this faithful
to what the source said?"* — outside every existing wire format. This profile
defines how a meaning assertion rides **inside** those formats rather than
competing with them: occurrence layers carry custody of bytes and events; the
meaning assertion carries custody of meaning; each is independently
verifiable and neither claims the other's scope.

## 2. The assertion, precisely

A **meaning assertion** is a detached-JWS-signed statement with ALL of the
following properties (each is load-bearing; a payload missing any of them
MUST NOT be presented as a meaning assertion under this profile):

1. **A named proxy.** The payload names the scorer and version that computed
   the loss (`scorer`, `scorer_version`, `loss_definition`). "Meaning" is
   never claimed directly — only a named, reproducible proxy for it.
2. **A statistical bound with printed scope.** The payload carries a
   one-sided upper bound on mean proxy loss (`risk_upper_bound_micro`) with
   its method (`method` ∈ hoeffding / clopper_pearson / empirical_bernstein),
   confidence (`delta_micro`), and sample size (`n`). The bound is
   **marginal, under exchangeability, over the named proxy** — never
   per-document, never a guarantee of meaning.
3. **A replay anchor.** `losses_hash` commits the exact integer-micro loss
   vector; a verifier handed the side-band losses MUST be able to recompute
   the bound to exact integer equality (float-free wire, RFC 8785 JCS
   canonical bytes).
4. **Mandatory negative disclosure.** `not_covered` (non-empty) lists the
   meaning dimensions the proxy is blind to (e.g. arrangement, sound,
   connotation, implicature); `disclosure` carries visible-text honesty
   prose. A verifier MUST fail closed if either is missing or empty — the
   disclosure is a structural invariant, not decoration.
5. **Calibration by reference, never by value.** Proxy-vs-human validity
   numbers (see [`docs/CALIBRATION_CARDS.md`](CALIBRATION_CARDS.md)) are
   corpus-specific measurements about the instrument. They MUST NOT be
   embedded as signed payload values (a cross-corpus overclaim); consumers
   discover them via the scorer identity.

Signature suite: Ed25519 (RFC 8032) over JCS-canonical payload bytes
(RFC 8785), detached JWS (RFC 7515 / RFC 7797 `b64=false`), keys via JWKS
(RFC 7517) — deliberately the same commodity suite the occurrence drafts use,
so embedding requires no new cryptography anywhere.

## 3. Profile A — C2PA custom assertion

C2PA 2.4 manifests bind the asset's bytes (hard binding; for unstructured
text, byte-exact binding). This profile adds the meaning layer as a custom
assertion:

- **Assertion label:** `org.sumengine.meaning_risk` (reverse-domain custom
  assertion naming per C2PA §"labels"). Future family members follow the
  same pattern (`org.sumengine.perspective_risk`, `org.sumengine.chain`).
- **Assertion data:** the complete SUM envelope, verbatim JSON:
  `{"schema": "sum.meaning_risk_receipt.v1", "kid": ..., "payload": {...},
  "jws": ...}` plus an optional `jwks_uri` hint. The envelope is embedded
  whole — a consumer without SUM tooling still sees a well-formed C2PA
  manifest; a consumer with the `sum_verify` SDK (or the JS verifier)
  verifies the meaning layer independently of C2PA trust lists.
- **Relationship to the C2PA action:** the assertion SHOULD accompany a
  `c2pa.actions` entry describing the transformation (e.g.
  `c2pa.edited`, with `digitalSourceType` per SUM's existing alignment,
  [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md)). The C2PA
  layer proves *these exact bytes, this actor, this action*; the SUM
  assertion proves *this named proxy, this bound, replayable*.
- **Verification order:** C2PA validation first (byte custody), SUM
  verification second (meaning custody). Failure of either layer fails only
  its own scope; a consumer MUST NOT read a valid C2PA manifest as evidence
  of meaning preservation, nor a valid meaning assertion as evidence of
  byte custody.

## 4. Profile B — extension claim on occurrence receipts (ACTA/ASQAV shape)

Occurrence-receipt drafts sign decision/action records with content digests.
This profile adds one OPTIONAL field to any such record:

```json
{
  "...": "the occurrence receipt's own fields (its digests bind the
          source and rendered texts it covers)",
  "meaning_assertion": {
    "schema": "sum.meaning_risk_receipt.v1",
    "kid": "...",
    "payload": { "...": "as specified in MEANING_RISK_RECEIPT_FORMAT" },
    "jws": "..."
  }
}
```

Rules: (a) the embedded envelope is verified by SUM tooling exactly as if it
were standalone — embedding changes nothing about its verification; (b) the
occurrence layer's digests SHOULD cover the same source/rendered texts the
meaning assertion's `corpus_id`/`transform` describe, and a consumer SHOULD
check that correspondence; (c) an occurrence verifier that does not know the
field ignores it (additive, forward-compatible); a consumer that *claims* to
verify the meaning layer MUST fail closed on an unknown `schema` value.

## 5. What this profile does NOT provide (the honest boundary, normative)

- **Not truth.** A verified meaning assertion is a cryptographic +
  statistical fact about a named proxy on a calibration sample. It is not
  evidence the content is true, nor a per-document guarantee.
- **Not detection.** Nothing here claims to detect AI authorship —
  detection collapses under paraphrase; attestation is the survivable form.
- **Not a validity oracle.** The proxy's correlation with human judgment is
  modest and corpus-dependent (measured, with committed artifacts, in the
  calibration cards). The assertion's value is that it makes this *checkable*,
  not that it makes it large.

## 6. IPR, licensing, and process

Wire format, reference implementation, and this profile: Apache-2.0. The
intended maturation path, in order: (1) circulate this profile with Paper-1
(arXiv, cs.CR); (2) raise it as a discussion item where the occurrence
drafts live (IETF; and CAWG/C2PA for Profile A) once ONE external
implementer or disputant-bearing user exists; (3) submit as an
Internet-Draft profile only when a second independent implementation is
plausible — a standards submission with zero adopters would be the
waiting-room trap wearing a tie.

## 7. References

C2PA Technical Specification v2.4 · `draft-farley-acta-signed-receipts` ·
`draft-marques-asqav-compliance-receipts` · RFC 8032, 8785, 7515, 7797, 7517
· AEX (arXiv:2603.14283, nearest prior art — same suite, disclaims meaning) ·
Gabbay, *Cryptographic certificates of validity for AI* (arXiv:2606.23768,
deterministic-predicate neighbor) ·
[`docs/MEANING_RISK_RECEIPT_FORMAT.md`](MEANING_RISK_RECEIPT_FORMAT.md) ·
[`docs/RECEIPT_FAMILY_SPEC.md`](RECEIPT_FAMILY_SPEC.md) ·
[`docs/CALIBRATION_CARDS.md`](CALIBRATION_CARDS.md) ·
[`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md).
