# SUM session handover — Sprint 7.5 + arXiv preprint v0.1 (2026-05-05)

## Mission elevation (read first; this is the lens for every decision)

Inherited from the prior handover (`docs/SESSION_HANDOVER_2026-05-04_intensification_arc.md`):

> Each artifact in the substrate should be so good that no one ever
> questions the reason for its existence.

Sprint 7.5 was the operational test of that standard. The first
published v3.x detector (sheaf-Laplacian cochain channel only) was
caught **losing to trivial entity-set baselines by Δ=−0.174
trusted-mean** before pre-circulation, by the substrate's own
baseline-comparison gate. Naming that loss STOP-THE-LINE rather than
"future work" produced the recovery arc that yielded the published WIN
(complementary Borda hybrid at trusted-mean AUC 0.876, Δ=+0.043 vs
B2 alone). The substrate's truth-first discipline functioned as
designed.

The aspirational standard remains: every claim must resolve to
either (a) a mechanically pinned property in `Tests/`, (b) a measured
value in a `fixtures/bench_receipts/*.json` receipt with a
reproducible `bench_digest`, or (c) an explicit "what this does NOT
pin" boundary. **The new property added in Sprint 7.5: cross-machine
reproducibility** — the load-bearing digests reproduce byte-for-byte
across two distinct LAPACK environments (Apple Accelerate on Apple
Silicon and OpenBLAS-via-PyPI on Modal x86_64), and the substantive
WIN verdict reproduces alongside.

When deciding what to ship, keep asking: would an external reader,
looking at this artifact, find any reason to doubt it? If yes, fix
the reason, then ship. If no, the artifact justifies its own existence.

---

## State at HEAD (post-Sprint-7.5 merges)

Substrate is in a clean, fully-consistent state. **Five PRs landed
in this arc (#142 / #146 / #144 / #145 / #147):** four for the preprint
+ recovery work, one for in-repo doc sync. The arXiv preprint is at
v0.1 with the substrate-first reframe and the complementary-hybrid
WIN; the in-repo documentation surface (PROOF_BOUNDARY, spec doc,
library API, CHANGELOG) is in lockstep with the preprint.

| Metric | Value |
|---|---|
| HEAD SHA | (post-#147 merge — fill in after pull) |
| Catalogued features | **156** (137 production / 18 scaffolded / 1 designed) — unchanged this arc; Sprint 7.5 added research artifacts under `[research]` extras which are by convention not cataloged |
| Compliance regimes shipped | **6** (unchanged) |
| Compliance suite tests | **169** (unchanged) |
| Audit-log suite tests | **25** (unchanged) |
| Research (sheaf detector) tests | **102** — was 88 pre-Sprint-7.5 (+9 from baseline_comparison polarity/schema/digest tests; +5 from recovery_experiment digest pins) |
| Full suite | **1781+ passing** (1767 pre-Sprint-7.5 + 14 new); 0 collection errors, 0 outstanding failures |
| `bench_digest` reproducibility | **Cross-machine** (Apple Accelerate ↔ OpenBLAS-via-PyPI on Modal x86_64) for the two load-bearing digests; same-machine for the rest |
| Current bench digests (Sprint-7.5 additions) | baseline_comparison: `cb32c617a3c692bc03bff49d85ae20e424c46cbb9ff47f9ea02285a90fd34e3b` · hybrid_comparison: `a7965803ccf2e703d80364dc21b3ac410491db9768cdfcf91bfefd29356c2003` · predicate_negatives_experiment: `aa34b6e8640621da07823f985ddf35196a85047a64f942493854e09b75c866e7` · per_triple_integration: `7025436f3c010e681bfbd06a04730d017e031df2b376e8e2bb5b404df81fd4fa` · **complementary_hybrid (WIN)**: `dc6e0260f14042fa0b6151a6ca6b443bb0910eabb996f6876f854633969343ce` |
| Pinned earlier digests still hold | v3.2 validation: `b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a` (cross-machine MATCH) · F3 diagnostic: `62b6e1878d1d12f36eb80e301304854a1a2c03386f0e872850d3461b2f733e7c` |
| `PROOF_BOUNDARY.md` | v1.7.0 (this arc — §2.9 detector table extended; §2.10 cross-machine measurement landed) |
| arXiv preprint | `docs/arxiv/sheaf-detector-note-v0.md` v0.1 (1500 lines, substrate-first reframe complete); cover note at `docs/arxiv/PRE_CIRCULATION_COVER_NOTE.md` |

---

## Initial self-check (first 5 minutes — DO NOT SKIP)

```bash
cd "/Users/ototao/Github Projects/SUM/SUM"

# 1. Confirm clean tree at expected HEAD
git status                    # expect: clean
git log --oneline -1          # expect: most-recent merged PR from this arc

# 2. Drift check
python3 -m pytest Tests/test_self_attestation.py -q   # expect: 7 passed

# 3. Recovery-experiment digest pins (load-bearing for the WIN claim)
python3 -m pytest Tests/research/test_recovery_experiment_digests.py -v
# expect: 4 passed (hybrid_comparison, predicate_negatives, per_triple_integration, complementary_hybrid)

# 4. Baseline comparison schema + digest pins
python3 -m pytest Tests/research/test_sheaf_baseline_comparison.py -q
# expect: 9 passed (schema pin, digest stability, scorer polarity, pinned-digest regression)

# 5. Full research + compliance + audit-log suites
python3 -m pytest Tests/research/ Tests/compliance/ Tests/test_audit_log.py -q
# expect: 102 + 169 + 25 = 296 passed

# 6. Bench reproducibility — unconditional, no env var
for i in 1 2 3; do
  python3 -m scripts.research.sheaf_v3_2_validation 2>/dev/null | grep '"bench_digest"'
done
# expect: same digest 3x (b4d26c01...ed5a)

for i in 1 2 3; do
  python3 -m scripts.research.sheaf_complementary_hybrid_experiment 2>/dev/null | grep '"bench_digest"'
done
# expect: same digest 3x (dc6e0260...343ce)

# 7. Mechanical catalog count
grep -cE "^### .*✅" docs/FEATURE_CATALOG.md   # expect: 137
grep -cE "^### "    docs/FEATURE_CATALOG.md   # expect: 156

# 8. Cross-regime registry consistency
python3 -c "
from sum_cli.main import _COMPLIANCE_REGIMES, _compliance_validators
assert set(_COMPLIANCE_REGIMES) == set(_compliance_validators()), 'registry drift'
print(f'OK: {len(_COMPLIANCE_REGIMES)} regimes wired')
"
# expect: OK: 6 regimes wired

# 9. Cross-machine bench_digest verification (optional; requires Modal)
modal run scripts/research/cross_machine_verify_modal.py
# expect: BRANCH_A_DIGESTS_MATCH for both v3.2 validation + complementary hybrid
# Cost: ~$0.01 in Modal credits; runtime ~3-5 min
```

If all eight (or nine, with Modal) checks pass, the substrate matches
this handover and you can begin work. Any fail = stop-the-line,
investigate.

---

## What landed 2026-05-04 / 05 (the five-PR arc)

| PR | Title | Headline |
|------|-------|----------|
| #142 | docs(arxiv): fold v3 / v3.1 / v3.2 / F3 / bench_digest into preprint v0 → v0.1 | Sprint 7 prose fold-in |
| #146 | arxiv(sprint-7.5): T2 baseline + T4 threat model + T2.5 recovery (rebased post-#142) | STOP-THE-LINE finding (B2 beats v3.2 by Δ=−0.174); four recovery experiments; complementary-hybrid WIN at trusted-mean 0.876 (Δ=+0.043) |
| #144 | arxiv(sprint-7.5): T3.M Modal cross-machine bench_digest verification — BRANCH A both digests MATCH | v3.2 validation + complementary hybrid both reproduce byte-for-byte across LAPACK environments |
| #145 | arxiv(sprint-7.5): T5 substrate-first reframe with complementary-hybrid story | Title shift; cs.CR primary; §3.9 hybrid section; §4.7.1 recovery arc; §6 reorganised; §7 bounded claims rewritten |
| #147 | docs: in-repo sync for Sprint 7.5 (CHANGELOG + PROOF_BOUNDARY + SHEAF_*.md) | PROOF_BOUNDARY §2.9/§2.10 + spec doc §3.4.5/§3.4.6 + library API additions + CHANGELOG entry |

### The recovery arc, in a single paragraph

The first v3.x detector (cochain-on-source-graph, §3.6–§3.8 of the
preprint) was caught losing to trivial entity-set baselines (B1
entity-presence-deficit, B2 jaccard-distance) by Δ=−0.174 trusted-mean.
A predicate-perturbation training experiment was tried first
(`aa34b6e8…`); it FAILED to lift A2 from chance, surfacing a
**structural finding**: the cochain is mathematically blind to
entity-set-preserving perturbations because predicate doesn't enter
the cochain construction. Adding training negatives can't fix what
the scoring path discards — same shape as F3 STRUCTURAL FAIL. The
fix was to integrate the §3.5 per-rendered-triple V channel
(`7025436f…`), which lifted A2 from 0.500 → 0.671. Still losing to
B2 alone, but informative: v3.2 + per-triple is the *only* detector
that catches A2. The complementary Borda(v3.2 + per-triple, B2)
hybrid (`dc6e0260…`) finally beat B2 at trusted-mean AUC 0.876,
Δ=+0.043 — **HYBRID_BEATS_BASELINE.** Cross-machine verification
on Modal x86_64 confirmed both the digest and the substantive verdict
reproduce.

---

## What remains before arXiv submit

Three operator-only items. None are technical work I can drive
autonomously; all require operator hands.

1. **(Optional) `v0.6.0` tagged release + PyPI publish.** The
   preprint cites `sum-engine v0.5.0`. The new substrate (recovery
   experiments, hybrid detector, cross-machine harness) would
   justify a `v0.6.0` minor release. Pre-circulation readers with
   only `pip install sum-engine` would otherwise get the older
   state. Out-of-scope for me per the operator's standing PR
   merge authority ("Authority stops at reversible actions; no
   tag pushes, no production deploys").

2. **Pre-circulation packet to 1–2 readers.** Cover note at
   `docs/arxiv/PRE_CIRCULATION_COVER_NOTE.md` is ready. Operator
   picks readers (ACT discourse / Topos Institute / HuggingFace
   forum / direct email candidates suggested in the cover note),
   replaces the placeholder SHA in the reproducibility section
   with the post-#147 merge SHA, sends.

3. **arXiv submit.** Categories `cs.CR` (primary) / `cs.LG`
   (secondary). Submit only after pre-circulation feedback
   absorbed into a v0.2 revision (or v0.1 polish if feedback is
   non-substantive).

---

## Discipline reminders (the rules that produced this arc's clean state)

The prior handover named ten discipline reminders. All ten held
through Sprint 7.5. Two are worth elevating because they were
load-bearing in this specific arc:

**#1 elevated: Truth-first labeling.** The B2-beats-v3.2 finding
could have been buried under "future work" or absorbed into a
"substrate-only paper" reframe (face-saving). Naming it STOP-THE-LINE
forced the recovery experiments. The recovery experiments produced
a structural finding (cochain blindness) that became itself a
load-bearing test. The published preprint includes both the loss
and the recovery — and is *stronger* for it.

**#9 elevated: Synthetic-vs-corpus discipline.** Predicate-perturbation
training was a plausible-but-wrong move because the training
distribution doesn't determine A2 detection — the scoring-path
architecture does. The synthetic intuition ("predicate negatives in
training should help predicate-flip detection at scoring") is exactly
the kind of move that necessary-but-not-sufficient testing catches:
the experiment ran, A2 stayed at chance, and the failure pointed at
the right load-bearing site (the cochain construction).

**New for Sprint 7.5 — #11: When a baseline-comparison gate fires,
treat it as load-bearing finding, not embarrassment.** The substrate
caught its own application failing against trivial baselines BEFORE
publication. That is the methodology working. The right response is
engineering recovery (try options 2, 2.5, 3, 4 from the recovery
plan) before publication-shape decision (substrate-only reframe vs
detector-as-headline). Carmack-mode: fix the system before rewriting
the spec.

---

## Pointers (read order if rebuilding context)

1. `CLAUDE.md` — current handover pointer is to this doc.
2. `docs/PROOF_BOUNDARY.md` v1.7.0 — §2.9 detector table now includes
   the complementary-hybrid WIN row; §2.10 includes cross-machine
   measurement
3. `docs/SHEAF_HALLUCINATION_DETECTOR.md` — §3.4.5 (recovery arc) and
   §3.4.6 (complementary hybrid) added in Sprint 7.5
4. `docs/SHEAF_LIBRARY_API.md` — "Sprint-7.5 additions" subsection
   documents `score_v32_with_per_triple`, `borda_fuse`,
   `score_b1_entity_presence_deficit`, `score_b2_jaccard_distance`,
   plus the Modal cross-machine harness
5. `docs/arxiv/sheaf-detector-note-v0.md` v0.1 — the published-shape
   preprint with substrate-first reframe; §3.0 threat model;
   §3.9 hybrid; §4.7.1 recovery arc; §4.8 cross-machine
6. `docs/arxiv/PRE_CIRCULATION_COVER_NOTE.md` — the template for
   1–2 pre-circulation readers
7. `Tests/research/test_recovery_experiment_digests.py` — pins for
   all four Sprint-7.5 recovery experiments. **Two byte-digest pins**
   (`per_triple_integration` `7025436f…`; `complementary_hybrid`
   `dc6e0260…` — both verified 5× in fresh procs unconditionally,
   the latter also cross-machine MATCH on Modal x86_64) and **two
   behavior-shape pins** (`predicate_negatives` — Python-version-
   sensitive digest, structural finding `A2_STILL_CHANCE` invariant;
   `hybrid_comparison` — cochain-only Borda has LAPACK-jitter
   tie-shuffle sensitivity, structural finding `BORDA_LOSES_TO_B2`
   with Δ ∈ [−0.10, −0.02] invariant). The verdict label
   `HYBRID_BEATS_BASELINE` is asserted alongside the
   `complementary_hybrid` byte-digest pin.
8. `Tests/research/test_sheaf_baseline_comparison.py` — pins for B1/B2
   schema, digest stability, scorer polarity
9. `CHANGELOG.md` `[Unreleased]` — Sprint 7.5 entry at the top covers
   the full arc

Read 1 → 2 → 5 in that order if cold; 3, 4, 6, 7, 8, 9 are reference.

---

## Closing frame

The substrate as of HEAD post-#147 merge is in a state where every
claim resolves to either a test, a reproducible receipt, a
cross-machine measurement, or an explicit boundary. The five PRs
that landed 2026-05-04 / 05 closed the only remaining load-bearing
question before arXiv submit: **does the substrate's discipline
catch real problems before they ship?** Yes. Both at the math
layer (cochain blindness surfaced via failed predicate-perturbation
training) and at the engineering layer (baselines beating v3.2
caught by the baseline-comparison gate). The preprint at v0.1
includes both findings honestly, and the recovery (complementary
Borda hybrid + cross-machine reproducibility) holds against
external scrutiny because every claim is anchored to a digest, a
test, or a cross-machine match.

The aspirational standard — "so good that no one ever questions
the reason for their existence" — applies now to operator-driven
publication steps: pre-circulation packet, optional v0.6.0 release,
arXiv submit. The substrate is ready; the operator decides on
timing and venue.

Begin with the self-check. Once it passes, you have ground truth
and can start on whatever the operator queues next.
