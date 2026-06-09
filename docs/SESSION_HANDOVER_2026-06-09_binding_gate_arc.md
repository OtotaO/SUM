# Session handover — the binding-gate + paper-readiness arc (2026-06-01 → 2026-06-09)

*Covers roughly PRs #261–#298. Picks up where
[`SESSION_HANDOVER_2026-06-01_post_v0.7.0_arc.md`](SESSION_HANDOVER_2026-06-01_post_v0.7.0_arc.md)
(#246–#260) left off. Read §0, §5, §6, §7 first if you are cold.*

---

## 0. State at head — the one-paragraph version

The meaning-loss frontier matured from a mechanism into a **demonstrated,
audited, publication-ready outcome.** The arXiv Paper-1 **binding gate is
CLOSED**: two *real* signed `meaning_risk` receipts over *real public-domain
corpora* are committed (BillSum compression ≤ 0.6454 @95%; opus-100 EN→FR
translation ≤ 0.4124 @95%, with 39/64 faithful translations at exactly zero
meaning-loss). The receipt family is unified + spec'd, has a cross-runtime JS
verifier, and survived **two adversarial pre-publication audits** (one on the
receipts, one on the paper-supporting surfaces) plus the full 2,564-test
suite + independent re-derivation. One real cross-runtime bug was found and
fixed (JCS float canonicalization, #297). A **final-form Paper-1 draft** is
written and **open at PR #298 for operator shaping.** `pyproject` is `0.7.1`
(live on PyPI); everything this arc is `[research]`-flagged except the jcs
substrate fix. **The binding constraint has not moved: adoption — one external
party issuing/verifying a receipt it did not author.**

## 1. What shipped (by theme)

**Meaning-loss frontier → Perspective receipts (#282–#288).** Group-conditional
meaning-risk (`certify_meaning_risk_by_group`) → the signed
`sum.perspective_risk_receipt.v1` (per-cohort bounds + optional Bonferroni
`simultaneous` joint guarantee); off-grid-delta replay-symmetry hardening
(#284, the recurring bug class — quantize the *scalar* delta in build);
**empirical-Bernstein** as a variance-adaptive third certifier (#288) with a
Monte-Carlo coverage receipt + the Bonferroni joint-coverage test (CP joint
0.958 vs 0.95). Honest regime fact: eB is *looser* than Hoeffding at small n /
moderate variance — it is a batch instrument, chosen a priori, never by
comparing realized bounds.

**Adoption on-ramp + cross-runtime (#285–#286).** JS verifier for the new
family (`single_file_demo/meaning_receipt_verifier.js`, Stage-A cross-runtime);
the `sum verify-meaning` CLI (closed the F21 "no runnable on-ramp" gap).

**The unified spec (#292).** `docs/RECEIPT_FAMILY_SPEC.md` — the four schemas
(render/transform/meaning_risk/perspective) as one family; *references* the
per-format docs as authoritative (not a second copy). Now also carries the two
trust-scope preconditions the audit established (§2 below).

**THE BINDING GATE (#294 BillSum, #295 opus-100).** The arc's headline. Two
real receipts replace the self-authored extractive smoke-test fixture:
- `fixtures/meaning_receipts_billsum/` — first 64 BillSum bills (**CC0**, US-gov
  public domain; full text committed), MiniLM-cosine judge, abstractive
  summarization, certified ≤ 0.6454 @95% (mean 0.4925), controlled.
- `fixtures/meaning_receipts_translation/` — first 64 length-aligned opus-100
  EN→FR pairs, multilingual mDeBERTa NLI judge, certified ≤ 0.4124 @95% (mean
  0.2594), **39/64 at exactly zero loss** (the moat: meaning preserved despite
  near-zero lexical overlap). opus-100 is mixed-licence → raw text **not**
  redistributed; committed as a sha256-pinned pointer + the loss vector.
- Honest split, baked into both: the *certificate* replays offline over the
  committed integer-micro losses (CI-checked, torch-free); the *loss
  computation* is machine-pinned (model-judge float drift, F23/F26), disclosed.

**The paper (#298, OPEN).** `docs/arxiv/PAPER1_DRAFT.md` — final-form v0,
~330 lines, formal bounds + threat model + Algorithm 1 + a real coverage table
+ references. Every number traces to a committed receipt. **Left open for the
operator to shape** (voice, arXiv-length abstract, figures, bib, running
example). The sheaf-detector negative results are deliberately Paper 2 (cs.LG).

## 2. The verification discipline (two audits + the fix)

The operator's standard this arc was **"beyond certain in every way before we
publish."** Two multi-agent adversarial audits delivered it:

- **Audit 1 (the receipts).** 5 skeptics tried to break statistical validity,
  corpus integrity, honesty/overclaim, reproducibility, code correctness.
  Core sound; **one blocking honesty defect**: the translation receipt's
  *signed disclosure* said "zero lexical overlap" — literally false → "near-zero",
  re-signed (#296). Also fixed: a cross-judge grading overclaim (the BillSum
  MiniLM mean and the translation NLI mean are *different judges* — never
  compare), "meaning-preserving" → "meaning-loss-bounded", + disclosure of the
  discrete judge values / filtered-subset bias / illustrative alpha.
- **Audit 2 (paper-supporting surfaces).** Crypto tamper-resistance (no
  accepted forgery in Python or Node), conformal validity (coverage ≥ 1−δ
  everywhere), prior-art accuracy (AEX/C2PA/Art-50 confirmed against primary
  sources), proof-boundary integrity — all sound. **One blocking bug**:
  reachable cross-runtime JCS divergence (#297, see §0). Two trust-scope
  preconditions surfaced and added to `RECEIPT_FAMILY_SPEC` + the paper:
  **(P1)** verification reduces to trusting an *out-of-band* JWKS (a forgery
  verifies against an attacker-supplied JWKS, as for any JWS); **(P2)** Stage A
  attests the *issuer*, only Stage-B replay attests the *bound*.

The pattern, stated honestly: each audit's *core verified sound* and each
surfaced *one real defect + honesty framing*. Diminishing, converging findings
→ a third pass would polish nits, not find blockers.

## 3. The research scans (decisions, not code)

- **Bellard adoption scan** (2 workflows, breadth + adversarial deep dive):
  verdict **adopt NOTHING as code** — the thematically-closest projects
  (ts_zip/ts_sms/NNCP-determinism via LibNC) are closed-binary; the MIT ones
  are domain-mismatched or substrate-velocity. Yield: one citation (NNCP v2
  §2.4 corroborates F23/F26 machine-pinning, #293) + the verified-and-de-escalated
  note that SUM's integer-micro wire is the right determinism call.
  See [[reference_bellard_adoption_scan_2026-06-08]].
- **Online sweep** (6 axes): found **BillSum** (the binding-gate corpus);
  confirmed **AEX (arXiv:2603.14283)** is the nearest prior-art — *same RFC
  stack, explicitly disclaims meaning preservation* → the cryptographic
  substrate is now **commodity**, the moat is the meaning-preservation +
  paraphrase-robustness + conformal delta. Assembled the paper's related-work
  spine. See [[project_online_sweep_2026-06-08]].

## 4. Documentation state — current as of this handover

- **CHANGELOG `[Unreleased]`** refreshed (perspective receipts, JS verifier,
  verify-meaning, receipt family spec, the two binding-gate receipts, + a
  `### Fixed` entry for the jcs RFC-8785 fix). Attestation refreshed.
- **PROOF_BOUNDARY** §2.11 covers the meaning-loss family; §1.8 fixture count
  corrected (20); the cross-runtime canonicalization claim corrected + scoped
  (now true after #297); NNCP-v2 corroboration cited.
- **Doc audit (Batch A #290 + Batch B #291)** earlier in the arc aligned ~13
  living docs with reality (the demoted quantum_router, the real CLI/MCP
  surface, version pins) + added the CI spaCy-download retry (the recurring
  504 flake). The "142-vs-170 feature count" was confirmed a *non-issue*.
- **RECEIPT_FAMILY_SPEC, MEANING_LOSS_FRONTIER, the two fixture READMEs** all
  carry the post-audit honest framing.
If you change README/CHANGELOG/PROOF_BOUNDARY/FEATURE_CATALOG/RENDER_ or
TRANSFORM_RECEIPT_FORMAT, run `python -m scripts.attest_repo_docs` + commit
`meta/self_attestation.*` or the drift gate reds CI (see
[[feedback_self_attestation_gate]]).

## 5. The operator queue (the actual rate-limit)

Nothing below needs more substrate; these are the operator's to move:
1. **Shape + submit Paper-1** (PR #298 is the draft). The TODOs are the
   author's: arXiv-length abstract, the running example, a coverage figure +
   system diagram, full DOIs/bib, voice. *This is the highest-leverage item.*
2. **One real external adopter** — the binding constraint the charter keeps
   naming. One party issuing/verifying a receipt they did not author. No
   amount of audit substitutes for this.
3. **NLnet grant decision (~Sept)** — a preprint *helps* the narrative; lean
   toward landing Paper-1 before the decision.
4. **MCP web dogfood** — still blocked on the Chrome extension being connected
   (claude.ai/chrome) with the same account; then the agent can drive the live
   `sum-demo.ototao.workers.dev` surface as an external user.

## 6. Where the next session should focus

Per the charter, **adoption is the binding constraint; substrate-velocity is
the #1 auto-pivot trap.** This arc was disciplined about it (the Bellard scan
*correctly* adopted nothing; the receipts/audits/paper serve the grant +
standards path). The next session's defensible work:
- **With the operator: drive Paper-1 to submission** (their shaping + your
  drafting passes on #298). Highest leverage.
- **The deferred $0 backlog** (all gated *behind* the now-closed gate, none are
  velocity traps now that they have a paper to feed): MiniCheck-class judge as
  an opt-in scorer; the Waudby-Smith–Ramdas betting bound (tighter, no wire
  change); Learn-Then-Test (jointly certify threshold + judge); the
  **deterministic CPU-INT8 judge spike** to de-pin F23/F26 (the real research
  frontier — would make a model-judge receipt hardware-independent). See
  [[project_online_sweep_2026-06-08]] for the concrete handles.
- **Do NOT** start the `translate` transform, new detectors, or more substrate
  unless a named buyer/grant/dream pulls — that is the trap.

## 7. Disciplines / anti-patterns this arc memorialised

- **Never let an unverified descriptive claim ride in a signed field / README
  headline / user summary.** "Zero overlap", "preserving", cross-judge
  comparison — *measure the descriptive claim, the exact math being right does
  not make the prose true.* ([[feedback_overclaim_in_signed_fields_2026-06-08]])
- **An "out of scope" comment in load-bearing code (canonicalization, signing)
  is a hazard — verify REACHABILITY before deferring.** The jcs `repr(f)`
  fallback was marked out-of-scope and was reachable (unsnapped density slider).
- **Run the adversarial-audit workflow BEFORE publishing claim-heavy
  artifacts, not after** — it caught what 217+ passing tests and my own
  re-derivation did not (tests check what they assert; they don't check that a
  prose claim is true).
- **Quantize the scalar delta in any receipt that re-certifies on verify**
  (off-grid-delta bug class, now seen 3×). **Pull `--ff-only` before
  `checkout -b`** (meta/* refresh guarantees a rebase conflict otherwise).

## 8. Open threads

- **PR #298** — Paper-1 draft, OPEN, awaiting operator shaping (do not merge
  without them).
- Memory is current: `project_arxiv_framing_2026-06-07` (the gate, the audits,
  the draft), `project_online_sweep_2026-06-08` (the deferred backlog +
  corpus), `reference_bellard_adoption_scan_2026-06-08` (adopt nothing),
  `feedback_overclaim_in_signed_fields_2026-06-08` (the discipline).
- Shipping surface unchanged from 0.7.1; the jcs fix is the only non-research
  substrate change this arc and is a strict correctness improvement.

---

## 9. CONTINUATION (post-#299, same session — read this if picking up cold)

After the handover above was written, the session kept going and produced the
**adoption arc** (PRs #300–#302 + a 30-guest simulation). This is the live
frontier; start here.

**The 30-guest adoption simulation** (mass-parallel, full findings in memory
[[project_adoption_sim_2026-06-09]]). 30 diverse prospective adopters evaluated
SUM's real surfaces with their real need, told not to flatter. Mean pull
**4.7/10; ZERO "adopt now", ZERO hostile.** THE finding: **the honesty IS the
moat, empirically** — the warmest guests are the rigorous skeptics (the
conformal statistician sent to find an invalid bound scored it HIGHEST, 8; the
hostile security engineer ran tamper attacks, all rejected, 6). Lead with the
proof-boundary discipline; it's the go-to-market, not a caveat.

**The demand has a SHAPE (ranked):** (1) **per-document readout** (~9 guests,
"what changed in MY text?" — the receipt certifies a corpus, the human has one
doc); (2) a **small, stable, non-`[research]` verify/SDK package** (integrators
won't pin a research format inside a 3000-line CLI); (3) a **validated judge**
(MiniLM-cosine isn't trustworthy); (4) **multi-hop composition / drift budget**;
(5) domain/standards adapters (C2PA-on-wire, OpenLineage, Art-50 profile);
(6) the **generate** experience (drop one text → actually RENDER it, not just
score); (7) one real adopter.

**Shipped this arc (all merged):**
- **#300** — `verify-meaning --losses` accepts the committed metadata-wrapped
  losses file. THE on-ramp bug 8 guests hit: "the certificate replays offline"
  was broken on our own flagship golden. Fixed + regression-pinned.
- **#301** — `python -m sum_cli` now works (added `sum_cli/__main__.py`).
- **#302 — `sum meaning-diff` = the per-document readout (demand #1).** Prints
  what a transform KEPT / DROPPED / ADDED in plain language ("argument
  preserved" vs "3 dropped, 1 unsupported, here they are"). Honest: a
  per-document MEASUREMENT, not a certified bound (scope printed every time).
  **Built so its `loss` EQUALS exactly what EntailmentScorer.loss certifies**
  (decomposed to claims) — so a per-doc readout and a corpus bound never
  disagree about the number. This is the **atomic unit the multi-hop drift
  budget will compose.** Building it VALIDATED demand #3: the embedding-cosine
  judge produces FALSE drops at the claim level (merged claim → falsely
  flagged dropped), so the command defaults to the **NLI** judge.
  Core: `meaning_loss.MeaningReadout` + `explain_meaning_loss` +
  `EntailmentScorer.explain`. Torch-free test (stub judge) in the CI meaning gate.

**External-tech scan** (memory has detail): **DSPy** (latest May 2026; GEPA
reflective-prompt optimizer) — SUM's meaning-loss certificate is a natural
DSPy/GEPA metric + DSPy is a distribution channel. **Karpathy autoresearch**
(Mar 2026, 66k★, the propose→run→measure→commit loop) — the strategic TAILWIND:
the "loopy era" makes meaning-drift-through-AI-chains urgent = SUM's problem.

**THE CONVERGENCE (the next move):** guest demand #4 + the Karpathy-autoresearch
trend + the "complete flavor" civilizational layer ALL point at ONE frontier:
**compose the meaning-loss certificate across a CHAIN of transforms — a
verifiable "drift budget" over N hops.** The atomic unit (the per-doc readout,
#302) now exists to stack. This is the deepest dream, the loudest tailwind, and
a named guest blocker — the same vector.

**NEXT-SESSION CHOICE (operator's standing direction = onward):**
- **(A) Compose the multi-hop drift budget** — the research move; the
  convergence. Stack per-hop readouts/receipts into a chain-level bound.
- **(B) Ship the small, stable, non-`[research]` verify/SDK package** — the
  adoption move; unblocks the warm-edge integrators (demand #2).
Both are no-`$`, charter-aligned (adoption is still the binding constraint).
Plus standing: PR #298 (Paper-1 draft) awaits operator shaping — fold in
Adler's slogan fix ("custody of a TRANSFORM", not "of meaning").
