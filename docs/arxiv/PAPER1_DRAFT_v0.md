# Chain-of-custody for AI-transformed text: signed, replayable, distribution-free receipts for what a transformation preserved

*Draft v0 — 2026-06-09. Structure per [`SUBMISSION_OUTLINE_2026-06-07.md`](SUBMISSION_OUTLINE_2026-06-07.md)
(Option C: the provenance + meaning-receipt flagship). Every number in §6 is
cited from a committed, signed, independently-re-derived receipt; every claim
is scoped to the proof boundary in §7. Targeted: `cs.CR` primary, `cs.LG`
secondary. This is an operator-review draft — framing, voice, and emphasis are
the operator's to shape.*

---

## Abstract

Two questions about AI-transformed text have no portable, offline-verifiable
answer today: *who transformed this, and what did the transformation
preserve?* Provider-side disclosure (e.g. EU AI Act Article 50) and
image-centric content provenance (C2PA, SynthID) do not cover text that has
been paraphrased, summarized, or translated — the manifest detaches on copy,
and a watermark is defeated by rewriting. We present a receipt family that
answers both questions for text: a **signed, offline-verifiable** attestation
of a transformation (Ed25519 over RFC 8785 JCS-canonical bytes, detached JWS,
JWKS key distribution), extended with a **distribution-free, replayable
certificate** bounding the *expected meaning-loss* of the transformation under
a **named proxy**, computed by a local judge. The certificate replays offline
over a committed loss vector — a third party re-runs the conformal certifier
and reproduces the bound to the bit — while the proof boundary stays explicit:
it bounds a named proxy marginally under exchangeability, never per-document
truth, and never "meaning" itself. We demonstrate the mechanism on two real
public-domain corpora: certified expected meaning-loss ≤ 0.645 for
abstractive summarization of US Congressional bills (BillSum, CC0; n=64), and
≤ 0.412 for EN→FR translation (opus-100; n=64) — with 39/64 faithful
translations scoring exactly zero meaning-loss *despite near-zero lexical
overlap*, the property no watermark or lexical scheme can certify. The thesis
is **attest, don't detect**: a signature survives an adversary with a
thesaurus; a statistical "is-this-AI" classifier does not.

## 1. Introduction

The regulatory and commercial pressure on AI-generated and AI-transformed text
is converging on two needs. First, **disclosure**: EU AI Act Article 50(2)/(4)
(applicable 2 August 2026) obliges providers to mark AI-generated content in a
machine-readable, robust way, and the Commission's own Code of Practice asks
for robustness "to paraphrasing [and] character deletions." Second, and less
served, **accountability for transformation**: when a document is summarized,
re-leveled for a different audience, or translated, *what survived?*

The dominant provenance tools answer neither well for text. C2PA binds a
cryptographic manifest to a media asset, but for plain text the manifest
detaches the moment the text is copied, pasted, or paraphrased. SynthID-Text
and statistical detectors embed or infer a generation signal that is, by
construction, degraded by the rewriting that text invites — every general
"is-this-AI" classifier has collapsed under paraphrase, distribution shift, or
bias against non-native writers, and one major provider retired its own.

We take a different stance: **attest, don't detect.** Rather than infer
whether text is AI-generated, we let any participating transformer *attest*
what it did and what it preserved, in a receipt anyone can verify offline,
robust to rewriting because a signature does not care about surface form. The
contribution is the *composition*: a signed transformation receipt carrying a
**distribution-free certificate of meaning-loss under a named judge**, made
**replayable** so the certificate is reproducible rather than merely asserted.

Concurrent work (§8) has independently arrived at the same cryptographic
substrate for LLM-API provenance but explicitly disclaims semantic
correctness; our contribution is exactly the semantic delta they disclaim,
made into a finite-sample, replayable certificate.

## 2. The substrate

The receipt family shares one envelope: a payload signed with **Ed25519**
(RFC 8032) over its **JCS-canonical bytes** (RFC 8785), carried as a
**detached JWS** (RFC 7515) and verified against a **JWKS** (RFC 7517)
distributed from a signed trust root. Four schemas instantiate it
(`render`, `transform`, `meaning_risk`, `perspective`); the spec is
[`RECEIPT_FAMILY_SPEC.md`](../RECEIPT_FAMILY_SPEC.md). The provenance tier
(`render`/`transform`) attests *what happened* — a named transform mapped an
input (hash) to an output (hash) under named parameters, by a named
model/provider. The measured-bound tier (`meaning_risk`/`perspective`) adds
the conformal certificate of §3–4.

Two substrate properties are load-bearing for the paper's claims and are
independently testable:

- **Cross-runtime byte-determinism.** A signature is verifiable in another
  runtime only if that runtime canonicalizes the payload to identical bytes.
  We verify Python, Node, and browser canonicalizers agree byte-for-byte on
  the receipt payloads, including the RFC 8785 number-serialization edge cases
  (ECMAScript `Number::toString` form); the meaning/perspective payloads are
  **float-free** (all rate quantities are integer micro-units) precisely so
  that exact-equality replay is well-defined across runtimes.
- **Tamper-evidence with explicit preconditions.** A verified signature proves
  the named issuer signed *this exact payload*. Two preconditions must be
  stated wherever "tamper-evident, offline-verifiable" is claimed: (i)
  verification reduces to trusting the JWKS, which must be obtained from a
  trusted root out-of-band, never from the receipt bundle — a forgery verifies
  against an attacker-supplied JWKS, as for any JWS system; (ii) the
  signature attests the *issuer*, while the conformal *bound* is attested only
  by the Stage-B replay of §3.

## 3. Meaning-risk receipts

Given pairs (source, transformed-output) and a **named, versioned judge**
`entails(premise, hypothesis) → bool`, we score each pair's meaning-loss in
[0,1] as one minus a bidirectional sentence-entailment coverage (recall of
source assertions, fidelity against fabrication). Over a calibration sample of
*n* such losses we certify a **one-sided distribution-free upper bound** on
the expected meaning-loss at confidence 1−δ, via Hoeffding / Clopper–Pearson /
empirical-Bernstein (the variance-adaptive bound, tighter for low-variance
batches at larger *n*). The receipt commits the integer-micro loss vector's
hash; verification has two stages:

- **Stage A** (every runtime): signature, schema gate, and the enforced
  disclosure invariants (a non-empty `not_covered` list and `disclosure`
  string — a receipt that discloses nothing cannot pass as a bare bound).
- **Stage B** (offline, deterministic): given the side-band loss vector, the
  verifier recomputes the hash and **re-runs the certifier**, requiring the
  reproduced bound, point estimate, and *n* to match by exact integer
  equality. This is what turns "measured" into "measured and independently
  reproducible." It catches an *overclaimed-at-issue* receipt — a valid
  signature over a hand-edited bound — distinct from a tampered-in-transit one.

The float-free integer-micro wire makes Stage B deterministic across hardware;
re-*deriving* the losses from raw text, however, requires the named model
judge, whose forward pass is machine-pinned (cross-hardware floating-point
drift). So the **certificate** is reproducible everywhere; the **loss
computation** is reproduced only on a matching stack — and the receipt
discloses this rather than hiding it. (Wire spec:
[`MEANING_RISK_RECEIPT_FORMAT.md`](../MEANING_RISK_RECEIPT_FORMAT.md).)

## 4. Perspective receipts

A single average-over-everything bound hides the cohort it is worst on. The
**perspective receipt** certifies the meaning-loss bound *per declared cohort*
(novice / expert / regulator / language / genre) — the discrete-covariate case
of group-conditional risk control — and, with `simultaneous` set, splits δ
across cohorts (Bonferroni) so all cohort bounds hold *jointly* at ≥ 1−δ. The
marginal bound and the per-cohort family are separate 1−δ statements; the
joint guarantee is the simultaneous one. Each cohort pays its own
finite-sample radius (a small cohort gets a wide bound — honest, not a defect).

## 5. Cross-runtime verification

Stage A is cross-runtime: signature, schema, and disclosure invariants verify
in Python and in Node/browser over the same JCS-canonical bytes. Stage B
(replay) is performed where the certifier runs (Python), and for a model-judge
scorer is additionally machine-pinned. "Verifiable in every runtime" therefore
means the *signature and format*, not the *meaning recomputation* — a
distinction the receipts state in their own scope fields, and which a paper
claiming offline verification must not blur.

## 6. Empirical demonstration

We issue two real receipts over public-domain corpora; each is committed
(signed golden + loss vector + deterministic generator), replays offline, and
was independently re-derived to the exact micro-unit and adversarially
audited prior to publication.

**6.1 Compression (BillSum, CC0).** Over the first 64 bills of the BillSum
test split (US Congressional bills + reference summaries; CC0-1.0 as
US-government works), with the local MiniLM-cosine entailment judge, the
abstractive bill→summary transform certifies **expected meaning-loss ≤ 0.6454
at 95% confidence** (n=64, mean 0.4925), controlled against a 0.70 threshold.
Aggressive summarization loses about half the named proxy on average — and the
receipt's job is to *certify how much*, not to claim little was lost.

**6.2 Translation (opus-100).** Over the first 64 length-aligned EN→FR pairs
of opus-100, with a local multilingual NLI judge (mDeBERTa-v3-xnli), the
translation transform certifies **expected meaning-loss ≤ 0.4124 at 95%**
(n=64, mean 0.2594), controlled against 0.50. The headline is the
distribution: **39 of 64 faithful translations score exactly zero
meaning-loss despite near-zero lexical overlap** between English and French.
A lexical or watermark scheme cannot credit a faithful translation — there is
almost no surface to match — whereas the meaning judge does. This is the
chain-of-custody-for-meaning property, made into a signed certificate.

**6.3 Honest scope.** Each receipt is certified under its *own* named judge
(the two judges differ), so the two means are not a single comparable "grading
scale"; the bounds are over the *named, filtered* calibration corpora under
exchangeability, not over all summarization or all translation; the
length-alignment filter on the translation corpus plausibly biases its proxy
low relative to the unfiltered split. The certificate is a **batch** primitive
— a single document uses the per-document measurement, not the bound.

## 7. Proof boundary and bounded claims

A verified receipt **proves** (cryptographic): the named issuer signed this
exact payload; the committed losses hash to the anchor; the named certifier
reproduces the bound on those losses. It **does not prove**: output truth,
issuer honesty (`scorer`/`model` labels are producer-asserted, not attested),
that meaning was preserved (only that a *named proxy* is bounded, marginally,
under exchangeability), or that text is human- vs AI-authored — we ship no
detection number, and any such signal is advisory, never a guarantee. The
layers the proxy structurally cannot reach (arrangement, sound, connotation,
implicature) are declared in a required, enforced `not_covered` field. The
repository's claim arbiter is [`PROOF_BOUNDARY.md`](../PROOF_BOUNDARY.md).

## 8. Position versus prior and concurrent work

- **AEX** (arXiv:2603.14283) independently uses JCS + SHA-256 + Ed25519 signed
  transformation-receipt chains for LLM APIs, but **explicitly disclaims**
  that an accepted transform is "semantically correct, reasonable, or
  minimal." Our contribution is precisely that disclaimed delta, made into a
  finite-sample replayable certificate.
- **C2PA / Content Credentials** bind provenance to media; for text the
  manifest is a soft binding that detaches under copy/paraphrase. **SynthID-Text**
  and statistical detectors target generation, not transformation-preservation,
  and degrade under rewriting.
- **EU AI Act Article 50** obliges the *generator* to mark; SUM is an adjacent,
  complementary instrument (it attests what a *transform* preserved), not a
  50(2) marker substitute.
- **Conformal risk control** (Angelopoulos et al., ICLR 2024) and the
  CP-for-NLP literature (incl. concurrent conformal-factuality work,
  arXiv:2603.27403) are our framing anchors; our contribution is the
  *composition* — a signed, replayable receipt over a named meaning-loss proxy
  — not a new inequality.

## 9. Limitations and future work

The certificate's validity rests on exchangeability between calibration and
deployment, which is assumed, not sampled; model-judge replay is machine-pinned
(de-pinning via integer/fixed-point CPU inference is named future work); the
proxy is a proxy, and the receipt's honesty is the discipline that keeps that
visible. The corpora here are demonstrations of the mechanism, not a benchmark
suite. The binding open problem is not a tighter inequality but adoption: one
external party issuing and verifying a receipt it did not author.

---

*Open drafting TODOs (operator): tighten the abstract to arXiv length; decide
whether §2's substrate detail (bench_digest, compliance validators, threat
model) gets its own section or stays compressed; choose the running example;
add the formal bound statements + the coverage-validation figure; reconcile
voice. The sheaf-detector honest-negative results are deliberately NOT here —
they are the separate `cs.LG` Paper 2.*
