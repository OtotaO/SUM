# Pre-circulation cover note for `docs/arxiv/sheaf-detector-note-v0.md`

**For:** the 1–2 readers the operator selects from the categorical-AI /
applied-ML / security-cryptography communities.

**Status of the artifact:** v0.1 draft (2026-05-04), Sprint 7.5 hardening
complete. Targeted submission `cs.CR` (primary) / `cs.LG` (secondary).
Not yet submitted; circulating for review before submission.

---

## What this note asks of the reader

I'd like a focused read for the following four questions. A few sentences
each is enough; I'm not asking for a full peer review.

### Q1. Math correctness — §2.3, §2.4, §3.7, §3.8

The weighted sheaf Laplacian (§2.3, Hansen-Ghrist 2019 §3.2) and the
harmonic extension (§2.4, Hansen-Ghrist 2019 Prop. 4.1 / Thm. 4.5) are
the load-bearing math. The detector sections build on them via:

- **§3.7 v3.1** (boundary deviation as standalone signal) is named
  STRUCTURAL FAIL at corpus scale because of an `L_IB = 0` topology
  argument. Does the §4.6 diagnostic + §3.7 / §4.5 prose convince you
  that this is a mathematical blind-spot, not a parameter-sweep gap?
- **§3.8 v3.2** (cochain V + γ·deviation_w + λ·v_deficit) claims to
  subsume v3 byte-identically at γ=0 (H16). Pinned in
  `Tests/research/test_sheaf_laplacian_v32.py` plus a Hypothesis
  property test. Reasonable framing?
- **§3.9 hybrid** uses Borda rank-fusion of (v3.2 + per-triple) with B2
  jaccard. Per-cell AUC of components is in §4.7.1. Is the fusion
  framing honest about what each component contributes?

### Q2. Threat model coverage — §3.0

We name four attacker capabilities (T1 adversarial render, T2
adversarial source bundle, T3 stolen signing key, T4 compromised
verifier) and map each to a defence component or to OUT OF SCOPE.

- Are there capabilities we should have named that we didn't?
- Are any of our defence claims overconfident given the substrate's
  current state?
- T2 is "partial" defence — does the prose make clear what is and isn't
  defended?

### Q3. Baseline choice and the WIN claim — §4.7.1

The hybrid's competitive claim rests on beating B1 entity-presence-
deficit (trusted-mean 0.824) and B2 jaccard-distance (0.833) by a
margin of $\Delta = +0.043$ (hybrid trusted-mean 0.876).

- Are B1 and B2 strong enough baselines for v0.1, or should we delay
  submission until LM-based baselines (sequence log-prob, MiniCheck-FT5)
  are in?
- Is "trusted-mean across A1+A2+A4 trusted cells" the right summary
  metric, given that A2 is at chance for B1/B2 and v3.x's win is
  concentrated there?
- Does the "STRUCTURAL_BLINDNESS finding" framing in §4.7.1 land for you?
  We name it as load-bearing because the per-triple channel restoration
  was a direct consequence of diagnosing the cochain's blindness.

### Q4. Substrate-first framing — §1, §6.1

The v0.1 categories shifted from `cs.AI/math.CT` (in v0) to
`cs.CR/cs.LG` (this v0.1) after the empirical work showed the substrate
was the load-bearing contribution and the detector was a worked example
that needed composition with entity-set baselines to be competitive.

- Is `cs.CR` the right primary venue?
- Does the §6.1 vs-published-reproducibility-primitives positioning
  hold? Specifically the claim that DOI-anchored evaluation bundles
  cover input but not result, and that we cover both?
- Does the §6.2 detector positioning concede enough? Is there a stronger
  cs.LG reviewer rebuttal we should preempt?

---

## Reproducibility for the reader

Every digest cited in the preprint is anchored. Two paths to verify
locally:

**Path A — PyPI (preferred once `sum-engine==0.6.0` is on PyPI):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install 'sum-engine[research,sieve]==0.6.0'
python -m spacy download en_core_web_sm

# scripts/ is excluded from the wheel dist (see CLAUDE.md packages.find rule);
# clone the repo for the bench scripts:
git clone --depth 1 --branch v0.6.0 https://github.com/OtotaO/SUM.git
cd SUM
```

**Path B — git clone at pinned SHA (works today, before PyPI publish):**

```bash
git clone https://github.com/OtotaO/SUM.git
cd SUM
git checkout 49b9686   # or any later main HEAD post-v0.6.0-prep
pip install -e '.[research,sieve]'
python -m spacy download en_core_web_sm
```

**Then, regardless of path:**

```bash
# Check all four recovery digest tests pass
python3 -m pytest Tests/research/test_recovery_experiment_digests.py -v
# Expected: 4 passed (3 byte-digest pins + 1 shape pin for hybrid_comparison)

# Check the v3.2 + complementary_hybrid digests directly
for i in 1 2 3; do
    python3 -m scripts.research.sheaf_v3_2_validation 2>/dev/null \
        | grep '"bench_digest"'
done
# Expected: b4d26c01... 3x

for i in 1 2 3; do
    python3 -m scripts.research.sheaf_complementary_hybrid_experiment \
        2>/dev/null | grep '"bench_digest"'
done
# Expected: dc6e0260... 3x
```

To verify cross-machine on Modal (requires Modal credits, ~$0.02
per full run):

```bash
pip install modal
modal token new   # one-time, browser auth
modal run scripts/research/cross_machine_verify_modal.py
```

This runs all 3 load-bearing benches × 2 Python versions (3.10 + 3.12)
and writes `fixtures/bench_receipts/cross_machine_verification_<DATE>.json`.
Expected outcome: `BRANCH_A_THREE_ENVIRONMENTS_DIGESTS_MATCH` — all
3 benches × 3 environments (your machine + Modal Py 3.10 + Modal
Py 3.12) reproduce byte-identically.

The one shape-pinned digest (`hybrid_comparison`) has irreducible
LAPACK-jitter cell-AUC sensitivity; same-machine reruns produce
two equally-valid `BORDA_LOSES_TO_B2` outcomes (`a7965803…` and
`7fac833a…`). The substantive loss-margin claim (Δ ≈ −0.025) is
invariant across both. See `Tests/research/test_recovery_experiment_digests.py::test_hybrid_comparison_loss_finding_holds`
for the rationale; the v0.3+ deterministic-LAPACK alternatives are
named in the docstring.

---

## What we are NOT asking

- Full peer review (this is pre-circulation for sanity-check level
  feedback, not gatekeeping)
- Editorial rewriting
- Suggestions for additional benchmarks beyond the v0.2 candidates
  already named in §8 future work

---

## Turnaround

We'd appreciate feedback within ~1 week if possible. If you need longer,
let us know and we'll hold the arXiv submission until you've had a look.

Pre-circulation venues we're comfortable with: ACT discourse, Topos
Institute mailing list, HuggingFace forum, Lobsters, direct email.

For corrections / contributions: https://github.com/OtotaO/SUM/issues

— SUM project (2026-05-04)
