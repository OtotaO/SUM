# Chain-of-Custody for AI-Transformed Text: Signed, Replayable, Distribution-Free Receipts for What a Transformation Preserved

*Draft (final-form v0) — 2026-06-09. Targeted venue: `cs.CR` (primary),
`cs.LG` (secondary). Every empirical number is cited from a committed, signed,
independently re-derived receipt; every claim is scoped to the proof boundary
(§8). Operator-review draft — voice, emphasis, and final framing are the
operator's to shape; drafting notes are at the foot.*

---

## Abstract

Two questions about AI-transformed text lack a portable, offline-verifiable
answer: *who transformed this, and what did the transformation preserve?*
Provider disclosure (EU AI Act Article 50) and image-centric content
provenance (C2PA, SynthID) do not cover text that has been paraphrased,
summarized, or translated — a manifest detaches on copy, a watermark is
defeated by rewriting. We present a receipt family that answers both questions
for text. A signed, offline-verifiable receipt attests a transformation
(Ed25519 over RFC 8785 JCS-canonical bytes, detached JWS, JWKS keys); on top of
it, a **distribution-free, replayable certificate** bounds the expected
*meaning-loss* of the transformation under a **named judge**. The certificate
replays offline over a committed integer loss vector — a third party re-runs
the conformal certifier and reproduces the bound to the bit — while the proof
boundary stays explicit: it bounds a named proxy *marginally*, under
*exchangeability*, never per-document truth and never "meaning" itself. We
demonstrate on two public-domain corpora: certified expected meaning-loss
≤ 0.645 (95%) for abstractive summarization of US Congressional bills
(BillSum, CC0; n=64) and ≤ 0.412 for EN→FR translation (opus-100; n=64), with
39/64 faithful translations scoring *exactly zero* meaning-loss despite
near-zero lexical overlap — the property no watermark or lexical scheme can
certify. The thesis is **attest, don't detect**: a signature survives an
adversary with a thesaurus; a statistical "is-this-AI" classifier does not.

**Contributions.** (1) A unified, cross-runtime-verifiable receipt family for
text-transformation provenance. (2) A *meaning-risk certificate*: a signed,
same-commit-replayable, distribution-free upper bound on expected meaning-loss
under a named judge, with its proof boundary enforced in the wire format.
(3) A group-conditional ("perspective") extension with a Bonferroni
simultaneous guarantee. (4) Two real, audited, public-domain demonstrations
and an empirical coverage validation. (5) An explicit threat model and the two
load-bearing verification preconditions (trusted out-of-band keys;
issuer-attestation vs. bound-attestation) that any "offline-verifiable" claim
must state.

## 1. Introduction

Pressure on AI-generated and AI-transformed text is converging on two needs.
First, **disclosure**: Article 50(2)/(4) of the EU AI Act (applicable
2 August 2026) requires machine-readable, robust marking of AI-generated
content, and the associated Code of Practice asks for robustness "to
paraphrasing [and] character deletions." Second, and far less served,
**accountability for transformation**: when a document is summarized,
re-leveled for a different audience, or translated, *what survived the
operation, and can it be proven to anyone, offline?*

The dominant provenance tools answer neither well for text. C2PA binds a
cryptographic manifest to a media asset, but for plain text the binding is
*soft*: the manifest detaches the moment text is copied, pasted, or
paraphrased. SynthID-Text and statistical "AI detectors" target *generation*,
not transformation-preservation, and degrade under exactly the rewriting text
invites — every general detector has failed under paraphrase, distribution
shift, or bias against non-native writers, and a major provider retired its
own classifier.

We take a different stance: **attest, don't detect.** Rather than infer whether
text is AI-generated, we let any participating transformer *attest* what it did
and what it preserved, in a receipt anyone verifies offline — robust to
rewriting because a signature is indifferent to surface form. The technical
contribution is a *composition*: a signed transformation receipt carrying a
**distribution-free certificate of meaning-loss under a named judge**, made
**replayable** so the certificate is reproducible rather than merely asserted.
Concurrent work (§9) independently arrived at the same cryptographic substrate
for LLM-API provenance but explicitly disclaims semantic correctness; our
contribution is precisely that disclaimed delta, turned into a finite-sample,
replayable certificate.

## 2. Threat model

The receipts defend a specific surface; we state it so the guarantees are not
over-read.

| Attacker capability | Outcome | Mechanism |
|---|---|---|
| Tamper with a signed field in transit (bound, corpus, disclosure, …) | **Rejected** | Ed25519 over the full JCS-canonical payload |
| Strip/empty the disclosure (`not_covered`/`disclosure`) | **Rejected** | Stage-A disclosure invariants |
| Algorithm downgrade (`alg:none`), `crit`/`b64` downgrade, non-detached JWS | **Rejected** | protected-header pinning |
| Self-sign a receipt with a *fabricated* bound but an honest loss-hash | **Rejected by Stage B** (passes Stage A) | re-certification over the committed losses |
| Forge a receipt and present it with an **attacker-controlled JWKS** | **Out of scope** — verification trusts the JWKS | keys must come from a trusted root, out-of-band |
| Replay an expired/stale receipt | Rejected when a max-age window is enforced | `signed_at` window check |

Two preconditions follow and must accompany any "tamper-evident,
offline-verifiable" claim: **(P1)** verification reduces to *trusting the
JWKS*, obtained from a signed trust root out-of-band (never from the receipt
bundle) — a forgery verifies against an attacker's own JWKS, as for any JWS
system; **(P2)** the signature attests the *issuer*, while the conformal
*bound* is attested only by the Stage-B replay (§5). Out of scope: the honesty
of the issuer's labels (`scorer`/`model`/`provider` are producer-asserted),
the correctness of the transformed output, and any claim about human-vs-AI
authorship.

## 3. The substrate and the receipt family

All four schemas (`render`, `transform`, `meaning_risk`, `perspective`) share
one envelope: a payload signed with **Ed25519** (RFC 8032) over its
**JCS-canonical bytes** (RFC 8785), carried as a **detached JWS** (RFC 7515)
and verified against a **JWKS** (RFC 7517). The provenance tier
(`render`/`transform`) attests *what happened*: a named transform mapped an
input (hash) to an output (hash) under named parameters, by a named
model/provider. The measured-bound tier (`meaning_risk`/`perspective`) adds the
certificate of §4–5.

**Cross-runtime determinism.** A signature is verifiable in another runtime
only if that runtime canonicalizes the payload to *identical bytes*. JCS fixes
key ordering and string/number serialization; the one subtlety is number
formatting, which RFC 8785 §3.2.2.3 mandates be the ECMAScript
`Number::toString` form. We verify Python, Node, and browser canonicalizers
agree byte-for-byte on the receipt payloads, including across the full
sub-10⁻⁴ float band where a naïve `repr`-based encoder diverges. The
conformal-tier payloads are additionally **float-free** — every rate quantity
is an integer "micro-unit" (value × 10⁶, ≤ 10⁶ ≪ 2⁵³) — so the exact-equality
replay of §5 is well-defined across runtimes.

## 4. The meaning-risk certificate

**Meaning-loss proxy.** Fix a *named, versioned* judge
$e:(\text{premise},\text{hypothesis})\to\{0,1\}$ (a local NLI model or
embedding-entailment model; never assumed to equal "meaning"). For a (source
$S$, output $T$) pair with sentence units $\mathrm{sent}(\cdot)$, define
$$\mathrm{recall}=\tfrac{1}{|\mathrm{sent}(S)|}\!\!\sum_{s\in\mathrm{sent}(S)}\!\! e(T,s),\quad
\mathrm{fid}=\tfrac{1}{|\mathrm{sent}(T)|}\!\!\sum_{t\in\mathrm{sent}(T)}\!\! e(S,t),$$
$$\ell(S,T)=1-\big(w_r\,\mathrm{recall}+w_f\,\mathrm{fid}\big)\in[0,1],\quad w_r+w_f=1.$$
Recall penalizes omission, fidelity penalizes fabrication; $\ell(S,S)=0$. The
judge is named and versioned in the receipt, so the certified bound is
*explicitly conditional on it*.

**Certificate.** Let $(S_i,T_i)_{i=1}^n$ be a calibration sample exchangeable
with deployment, $\ell_i=\ell(S_i,T_i)$, preservation $p_i=1-\ell_i\in[0,1]$.
By Hoeffding's inequality, the one-sided $(1-\delta)$ lower confidence bound on
$\mathbb{E}[p]$ is $\bar p-\sqrt{\ln(1/\delta)/(2n)}$, hence the
$(1-\delta)$ **upper** bound on expected meaning-loss is
$$U_\delta \;=\; \min\!\Big(1,\; \bar\ell + \sqrt{\tfrac{\ln(1/\delta)}{2n}}\Big),
\qquad \Pr\big[\mathbb{E}[\ell]\le U_\delta\big]\ge 1-\delta. \tag{1}$$
Three estimators ship: **Hoeffding** (1) for $\ell\in[0,1]$;
**Clopper–Pearson** (exact) for the binary preserved/lost view; and the
variance-adaptive **empirical-Bernstein** bound (Maurer & Pontil, 2009), whose
deviation scales with the sample variance and is therefore tighter for
low-variance batches at larger $n$ (its additive $O(1/(n{-}1))$ term makes it
*looser* than Hoeffding at small $n$ — a regime fact, not a tuning knob). The
shipped default selects Clopper–Pearson for binary data and Hoeffding
otherwise; the choice is made a priori, never by comparing realized bounds.

**Wire and disclosure.** The receipt commits the SHA-256 of the integer-micro
loss vector and carries `n`, `delta_micro`, `method`,
`point_estimate_micro`, `risk_upper_bound_micro`, the judge identity, the
calibration `corpus_id`, and two *required, enforced* fields: a non-empty
`not_covered` list (the layers the proxy structurally cannot reach —
arrangement, sound, connotation, implicature) and a non-empty `disclosure`
string. A receipt that discloses nothing cannot pass as a bare bound.

## 5. Verification: two stages

Verification separates *issuer-attestation* (everywhere) from
*bound-attestation* (deterministic, offline, where the certifier runs).

> **Algorithm 1 — verify(envelope, jwks, [losses], [max_age]).**
> **Stage A (any runtime).** (a) JOSE-verify the detached JWS over
> `JCS(payload)` against `jwks`; (b) gate `schema` (fail closed on unknown);
> (c) check header invariants (`alg=EdDSA`, matching `kid`, `b64:false`,
> `crit:["b64"]`); (d) enforce disclosure (`not_covered`≠∅, `disclosure`≠"");
> (e) if `max_age` given, check the `signed_at` window. → *authentic +
> self-disclosing.*
> **Stage B (when the loss vector is supplied side-band).** (f) recompute the
> loss-vector hash and require equality; (g) re-run the certifier of §4 over
> the committed integer-micro losses at the payload's `method`/`delta`; (h)
> require the reproduced `risk_upper_bound_micro`, `point_estimate_micro`, and
> `n` to match by **exact integer equality**. → *bound reproduced.*

Stage B is what turns "measured" into "measured and independently
reproducible," and it is what distinguishes a *tampered-in-transit* receipt
(signature fails) from an *overclaimed-at-issue* one (signature valid, bound
does not replay). The certificate replays on any machine; *re-deriving* the
losses from raw text additionally requires the named model judge, whose
forward pass is machine-pinned (cross-hardware floating-point drift) — a fact
the receipt discloses. So "verifiable in every runtime" means the signature
and bound *replay*, not the *meaning recomputation*.

## 6. Perspective receipts (group-conditional)

A marginal bound hides the cohort it is worst on. Given cohort labels
$g_i$, the **perspective receipt** certifies (1) per declared cohort over only
that cohort's losses — the discrete-covariate case of group-conditional risk
control (Gibbs, Cherian & Candès, 2023). The marginal and the per-cohort
family are *separate* $(1-\delta)$ statements; with `simultaneous` set, each
cohort is certified at $\delta/G$ (Bonferroni) so that *all* $G$ cohort bounds
hold **jointly** at $\ge 1-\delta$. Each cohort pays its own finite-sample
radius — a small cohort gets a wide bound, which is honest, not a defect.

## 7. Empirical demonstration

We issue two real receipts over public-domain corpora. Each is committed (a
signed golden, the integer loss vector, a deterministic generator), replays
offline via Algorithm 1 Stage B, was **independently re-derived to the exact
micro-unit** and adversarially audited before release.

**7.1 Compression (BillSum, CC0).** First 64 bills of the BillSum test split
(US Congressional bills + reference summaries; CC0-1.0 as US-government works),
local MiniLM-cosine entailment judge. The abstractive bill→summary transform
certifies **expected meaning-loss $\le 0.6454$ at 95%** ($n=64$, mean
$0.4925$), controlled against a 0.70 threshold. Aggressive summarization loses
about half the named proxy on average; the receipt *certifies how much*, it
does not claim little was lost.

**7.2 Translation (opus-100).** First 64 length-aligned EN→FR pairs of
opus-100, local multilingual NLI judge (mDeBERTa-v3-xnli). The translation
transform certifies **expected meaning-loss $\le 0.4124$ at 95%** ($n=64$, mean
$0.2594$), controlled against 0.50. The distribution is the headline: **39 of
64 faithful translations score exactly zero meaning-loss despite near-zero
lexical overlap** between English and French. A lexical or watermark scheme
cannot credit a faithful translation — there is almost no surface to match —
whereas the meaning judge does; this is the chain-of-custody-for-meaning
property as a signed certificate.

**7.3 Coverage validation.** The bounds are only meaningful if valid. We
measure empirical coverage of the $(1-\delta)$ upper bound — the fraction of
trials in which the certified bound does not fall below the true loss rate —
on synthetic data with known ground truth ($n=64$, $2\times10^4$ trials):

| method | tl=0.1, δ=.05 | tl=0.3, δ=.05 | tl=0.5, δ=.05 | tl=0.5, δ=.10 |
|---|---|---|---|---|
| target | ≥ 0.95 | ≥ 0.95 | ≥ 0.95 | ≥ 0.90 |
| Hoeffding | 1.000 | 0.997 | 0.992 | 0.984 |
| empirical-Bernstein | 1.000 | 1.000 | 1.000 | 1.000 |
| Clopper–Pearson | 0.961 | 0.972 | 0.970 | 0.919 |

All estimators clear their target in every regime (Hoeffding and eB
conservatively, Clopper–Pearson near-nominal). The Bonferroni simultaneous
path's *joint* (all-cohorts) coverage was separately validated at 0.958 against
a 0.95 target. At the receipts' $n=64$ and observed variance, Hoeffding is the
tighter honest estimator than empirical-Bernstein (BillSum 0.645 vs 0.650;
translation 0.412 vs 0.518), confirming the a-priori default was not a
favorable cherry-pick.

**7.4 Honest scope.** Each receipt is certified under its *own* named judge
(the two judges differ), so the two means are not a single comparable grading
scale. The bounds are over the *named, filtered* calibration corpora under
exchangeability — not over all summarization or all translation; the
length-alignment filter on the translation corpus plausibly biases its proxy
*low* relative to the unfiltered split. The certificate is a **batch**
primitive: a single document uses a per-document measurement, not the bound.

## 8. Proof boundary

A verified receipt **proves** (cryptographic, given P1–P2): the named issuer
signed this exact payload; the committed losses hash to the anchor; the named
certifier reproduces the bound on those losses. It **does not prove**: output
truth or freshness; issuer honesty; that *meaning* was preserved (only that a
*named proxy* is bounded, marginally, under exchangeability); or anything about
human-vs-AI authorship — we ship no detection number, and any such signal is
advisory, never a guarantee. The proxy's structural blind spots are declared
in the enforced `not_covered` field. This is the discipline that keeps the
gap between "a bounded named proxy" and "preserved meaning" continuously
visible rather than rhetorically closed.

## 9. Position versus prior and concurrent work

- **AEX** (arXiv:2603.14283) independently uses JCS + SHA-256 + Ed25519 signed
  transformation-receipt chains for LLM APIs but **explicitly disclaims** that
  an accepted transform is "semantically correct, reasonable, or minimal." Our
  contribution is precisely that disclaimed delta, as a finite-sample,
  replayable certificate.
- **C2PA / Content Credentials** bind provenance to media; for text the binding
  is soft and detaches under copy/paraphrase. **SynthID-Text** (Dathathri et
  al., 2024) and statistical detectors target generation, not
  transformation-preservation, and degrade under rewriting.
- **EU AI Act Article 50** obliges the *generator* to mark; this work is an
  adjacent, complementary instrument (it attests what a *transform* preserved),
  not a 50(2) marker substitute.
- **Distribution-free guarantees.** Hoeffding (1963); the empirical-Bernstein
  bound (Maurer & Pontil, 2009); conformal prediction (Vovk et al., 2005;
  Angelopoulos & Bates, 2023) and conformal risk control (Angelopoulos et al.,
  ICLR 2024); group-conditional coverage (Gibbs, Cherian & Candès, 2023);
  and concurrent conformal-factuality work for LLMs (arXiv:2603.27403). Our
  contribution is the *composition* — a signed, replayable receipt over a named
  meaning-loss proxy — not a new inequality.

## 10. Limitations and future work

Validity rests on exchangeability between calibration and deployment, which is
assumed, not sampled — a deliberately disclosed boundary common to all
conformal guarantees. Model-judge replay is machine-pinned; de-pinning via
integer/fixed-point CPU inference (so a meaning-judge forward pass is
hardware-independent) is named future work. The proxy is a proxy; stronger
faithfulness judges (e.g. MiniCheck-class) are a drop-in upgrade, and a tighter
betting/empirical-Bernstein confidence sequence is a no-wire-change tightening.
The corpora here demonstrate the mechanism; they are not a benchmark suite. The
binding open problem is not a sharper inequality but **adoption**: one external
party issuing and verifying a receipt it did not author.

## References (to be formatted; primary sources verified)

- D. J. Bernstein et al. *Ed25519.* RFC 8032.
- A. Rundgren et al. *JSON Canonicalization Scheme (JCS).* RFC 8785.
- M. Jones et al. *JSON Web Signature (JWS).* RFC 7515; *JSON Web Key (JWK).* RFC 7517.
- W. Hoeffding. *Probability inequalities for sums of bounded random variables.* JASA, 1963.
- A. Maurer, M. Pontil. *Empirical Bernstein bounds and sample variance penalization.* COLT 2009.
- V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.* Springer 2005.
- A. Angelopoulos, S. Bates. *Conformal Prediction: A Gentle Introduction.* FnT ML 2023.
- A. Angelopoulos et al. *Conformal Risk Control.* ICLR 2024.
- I. Gibbs, J. Cherian, E. Candès. *Conformal prediction with conditional guarantees.* 2023.
- S. Dathathri et al. *Scalable watermarking for identifying large language model outputs (SynthID-Text).* Nature, 2024.
- *AEX: Non-Intrusive Multi-Hop Attestation and Provenance for LLM APIs.* arXiv:2603.14283, 2026.
- *Conditional Factuality-Controlled LLMs with Generalization Certificates.* arXiv:2603.27403, 2026.
- C2PA. *Coalition for Content Provenance and Authenticity, Technical Specification* (digitalSourceType taxonomy, v2.4).
- European Union. *Artificial Intelligence Act, Article 50*, and the GPAI/transparency Code of Practice (2026).
- ECMA-262. *ECMAScript Language Specification* (§ Number::toString).

---

*Drafting notes for the operator (next pass): (i) cut the abstract to the
arXiv 1,920-char limit if submitting the abstract field verbatim; (ii) decide
whether §3's substrate (bench_digest reproducibility, the six compliance
validators) earns its own subsection or stays compressed; (iii) add the
coverage figure (the §7.3 table as a plot) and a system diagram; (iv) pick the
single running example used end-to-end; (v) fetch full bibliographic details +
DOIs and convert to the venue's bib style; (vi) reconcile first-person /
voice. The sheaf-detector honest-negative results are intentionally absent —
they are the separate `cs.LG` Paper 2.*
