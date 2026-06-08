# arXiv submission outline — drafted vs. missing, and the framing decision

*Status: editorial map, 2026-06-07. Not a new draft — a reconciliation of
the existing v0.1 preprint
([`sheaf-detector-note-v0.md`](sheaf-detector-note-v0.md), 2026-05-04)
with five weeks of post-draft work (PRs #270–#288: the meaning-loss
frontier + conformal/empirical-Bernstein guarantees + the locked
provenance-first positioning). Ends with the one decision only the
operator can make.*

---

## TL;DR

The existing preprint is **not stale and not broken.** It is complete,
substrate-first (`cs.CR` primary), honest (it already absorbs the
detector-WIN-is-Goodhart re-read in §7.2), and reviewer-anticipated
(`REVIEWER_ANTICIPATION.md`, 15 Qs). What changed is the *project's centre
of gravity*: the detector's headline is a **bounded/negative** result
(`STRUCTURAL_GAP_NO_MODEL_BEATS` at n ≥ 16), while a genuinely novel
**affirmative** result — signed, replayable, distribution-free
meaning-loss receipts — now exists and is **not in the draft.**

So the decision is not "edit the draft." It is **"what is SUM's *first*
arXiv paper?"** My recommendation: **split (Option C)** — lead with a new
provenance + meaning-receipt paper that matches the locked positioning,
and keep the sheaf-detector note as a separate, later, honest-negative
`cs.LG` submission.

> **DECISION (operator, 2026-06-07): Option C — split.** Paper 1 (`cs.CR`)
> = provenance substrate + meaning-receipt family (flagship, first
> imprint). Paper 2 (`cs.LG`, later) = the sheaf-detector honest-negative
> note. The binding gate for Paper 1 is §6's one real committed receipt
> over a real corpus (no-`$`). Remaining sub-decisions (§8.2–8.5) still
> open.

## 1. State of the existing draft — what's drafted and solid

Keep all of this; it is load-bearing and verified (§ refs into the v0.1
note):

| Block | §| Status |
|---|---|---|
| Cross-runtime render receipts (Ed25519 / JCS, Python↔Node↔browser byte-identical, K-matrix gated) | 3.1, 4.8, 7.1 | **Solid, CI-pinned** |
| `bench_digest` cross-machine reproducibility (3 environments, 2 LAPACK stacks) | 4.8, 7.1 | **Solid, CI-pinned** |
| Six audit-grade compliance validators (EU AI Act Art 12 / GDPR / HIPAA / ISO / SOC2 / PCI) | abstract, 6.1 | **Solid** |
| Explicit threat model (T1–T4 attacker capabilities → defence/OOS) | 3.0 | **Solid** |
| Position vs. reproducibility primitives (input *and* result, not just input) | 6.1 | **Solid** |
| Bounded-claims discipline (holds / corrected / narrow / limits) | 7.1–7.4 | **Exemplary — the paper's spine** |

## 2. What's drafted but now reads differently

- **The detector arc (§3.2–3.9, §4.1–4.7).** Months of real work, and the
  honesty is its strength — but the *headline* is now negative. §7.2
  already states it: the synthetic WIN (Δ=+0.043) is a **Goodhart
  artifact** (Manheim & Garrabrant), and at controlled n ≥ 16 across
  corpora × LLM families the joint finding is
  `STRUCTURAL_GAP_NO_MODEL_BEATS`. This is a *fine honest-negative-results
  contribution* — it is a *weak flagship.*
- **The framing tension.** The locked positioning is **"attest, don't
  detect"** ([`PRODUCT_VISION.md`](../PRODUCT_VISION.md) §0). The draft's
  title is *"…for hallucination detection…"*. Leading the project's first
  arXiv paper with a detector — even an honest-negative one — imprints the
  opposite thesis on the exact audience (reviewers, grant evaluators) you
  most want to imprint with provenance.

## 3. What's missing — the affirmative contribution the draft predates

None of the post-#270 work is in the preprint. This is the genuinely
novel, defensible, positioning-aligned material:

- **`sum.meaning_risk_receipt.v1`** — a signed, *same-commit-replayable*
  certificate over a **named meaning-loss proxy**, in checkable text space
  (not model internals). Required, enforced `not_covered` disclosure.
  ([`MEANING_LOSS_FRONTIER.md`](../MEANING_LOSS_FRONTIER.md),
  [`MEANING_RISK_RECEIPT_FORMAT.md`](../MEANING_RISK_RECEIPT_FORMAT.md))
- **Conformal meaning-risk bounds** — distribution-free upper bound on
  E[meaning-loss], the dual of the slider rate kernel; **Hoeffding /
  Clopper–Pearson / empirical-Bernstein** (PR #288, variance-adaptive,
  tighter at batch n), with Monte-Carlo coverage as the validity receipt.
- **Group-conditional (Perspective) receipts** — per-cohort bounds +
  **Bonferroni simultaneous joint-coverage** (the all-cohorts guarantee,
  Monte-Carlo-validated, PR #288). The zenith Perspective-Receipts concept
  as a real artifact.
- **Cross-runtime JS verifier** for both schemas (`meaning_receipt_verifier.js`).
- **The provenance thesis itself** — chain-of-custody for AI-transformed
  text, robust to rewriting where image-centric C2PA/SynthID structurally
  are not; EU AI Act Art 50 disclosure framing; detection-as-advisory.
- **The dogfood honesty** (F21–F26) — batch-vs-single, scorer-determinism,
  cross-lingual machine-pinning — the kind of limits section reviewers
  respect.

## 4. The framing decision (operator call)

| Option | Shape | Pro | Con |
|---|---|---|---|
| **A. Ship the sheaf note as-is** | substrate + honest-negative detector, `cs.CR/cs.LG` | Ready now; reviewed-in-anticipation; honest | Headline is negative; reads as "reproducibility plumbing + a detector that didn't beat baselines"; **off-message** vs. attest-don't-detect |
| **B. Fold meaning-receipts into the same paper** | substrate → app 1 (detector, negative) → app 2 (meaning-receipts, affirmative) | One artifact; the affirmative result rescues the headline | Scope creep; two unrelated method families (sheaf-Laplacian + conformal) in one paper is hard to review; delays submission most |
| **C. Split** *(recommended)* | **Paper 1** (`cs.CR`): provenance substrate + meaning-receipt family — *the flagship, on-message.* **Paper 2** (`cs.LG`, later): the sheaf-detector honest-negative note | Paper 1 matches the locked thesis + grant story + has a defensible *positive* result; Paper 2 is a clean honest-negative-results contribution in its right venue | Two write-ups; Paper 1 needs one real committed receipt first (§6) |

## 5. Recommended structure — Paper 1 (`cs.CR` primary, `cs.LG` secondary)

> **Working title:** *"Chain-of-custody for AI-transformed text: signed,
> replayable, distribution-free receipts for what a transformation
> preserved."*

1. **Introduction** — the gap: provider *disclosure* (Art 50) and "what
   did this transformation lose?" have no portable, offline-verifiable,
   rewriting-robust receipt. Image-centric provenance can't cover text.
   *Attest, don't detect.*
2. **The substrate** *(lifted from the v0.1 note, §3.0–3.1, §4.8, §6.1)* —
   cross-runtime render receipts, `bench_digest`, threat model, compliance
   validators. The cryptographic + reproducibility spine.
3. **Meaning-risk receipts** *(new)* — named bounded proxies; the
   conformal upper bound (dual of the rate kernel); **empirical-Bernstein**
   for batch-tight bounds; float-free integer-micro wire for cross-runtime;
   replay = the "measured → measured-and-reproducible" step.
4. **Perspective receipts** *(new)* — group-conditional bounds; Bonferroni
   simultaneous joint coverage; the novice/expert/regulator framing.
5. **Cross-runtime verification** *(new + §3.1)* — Python ↔ Node ↔ browser;
   Stage A (signature/schema/disclosure) everywhere, Stage B (replay)
   where the scorer runs; the machine-pinning honesty for model judges.
6. **Empirical spine** — the committed golden receipt(s) + the Monte-Carlo
   coverage validations as *receipts of validity* (not just plots).
7. **Proof boundary / bounded claims** *(adapt §7)* — proxy, marginal,
   exchangeability; `not_covered`; measured-vs-certified; detection is
   advisory. The paper's honesty is the contribution.
8. **Position vs. work** — C2PA/SynthID (image), AEX (same RFC stack),
   conformal-prediction lineage, NLI/faithfulness judges (MENLI,
   MiniCheck); reproducibility primitives (§6.1).
9. **Limits & future work** — F21–F26; the real-corpus / paid-judge gates;
   one-real-adopter as the honest "not yet ubiquitous."

## 6. What Paper 1 still needs before submit (gating work, mostly no-$)

- **One real committed receipt over a real corpus** — product-vision rung
  2: a public-domain parallel-translation (or extractive) corpus + a
  deterministic judge → a real `sum.meaning_risk_receipt.v1` + cross-
  runtime golden. The self-authored extractive fixture is a smoke test,
  not an empirical spine a reviewer respects. **This is the binding gate.**
- **Fold meaning-risk + perspective receipts into
  [`PROOF_BOUNDARY.md`](../PROOF_BOUNDARY.md)** (the arbiter is currently
  silent on them) — so the paper's claims trace to the repo's claim ledger.
- **Doc-currency:** scope the README "three-runtime" claim (now true for
  all four receipt families), fix the duplicate `2.5.1` heading, correct
  the CLAUDE.md feature count — small, named in the meaning-loss memory's
  backlog.
- **Decide the empirical-Bernstein presentation** — it is a clean,
  citable tightening (Maurer–Pontil 2009) with an honest regime caveat;
  good for §3.

## 7. Disposition of the sheaf-detector note (Paper 2)

It does **not** get thrown away. It becomes a standalone `cs.LG`
honest-negative-results note — *"Sheaf-Laplacian signals on signed render
bundles: an honest negative result and a Goodhart post-mortem."* It is
nearly submission-ready already; the open items are the ones its own
§8 / `REVIEWER_ANTICIPATION.md` Q13 name (LM-based baselines —
sequence log-prob, MiniCheck-FT5 — and deeper corpus sampling). Submit it
*after* Paper 1 plants the provenance flag, so the project's first
imprint is the affirmative thesis, not the negative one.

## 8. Operator decisions to lock

1. **A / B / C?** (recommendation: C.)
2. If C: **does Paper 1 wait for the one real committed receipt** (§6), or
   submit on the self-authored fixture with an explicit "illustrative
   corpus" caveat? (Recommendation: wait — it is cheap, no-$, and it is
   the difference between a respected and a hand-wavy empirical section.)
3. **Venue:** `cs.CR` primary for Paper 1 — confirm, or `cs.LG`?
4. **Reviewers / pre-circulation:** reuse the `PRE_CIRCULATION_COVER_NOTE`
   list (ACT discourse, Topos, HF forum, direct email)?
5. **Timing vs. the NLnet decision (~Sept):** an arXiv preprint *helps* a
   grant narrative; is there a reason to hold, or does landing Paper 1
   before the decision strengthen it?

## 9. Honest boundary

This outline does not write the paper or change any claim. It records that
the existing draft is *good and honest but off-message as a flagship*, that
the affirmative result now exists outside it, and that the cheapest path to
a strong first submission is to split — with one real receipt as the only
hard gate. The recommendation is mine; the framing call is the operator's.
