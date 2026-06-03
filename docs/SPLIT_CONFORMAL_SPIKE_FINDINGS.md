# Split-conformal substrate spike — quick win #1 findings

The wide-net survey's #1 statistics pick (Vovk-Gammerman-Shafer
2005; Angelopoulos & Bates 2023) implemented as a substrate-shaped
quick win. Distribution-free, finite-sample prediction intervals
for any point predictor — the kernel needed to upgrade SUM's
empirical readouts from point estimates to calibrated CIs.

## What landed

  - `sum_engine_internal/research/conformal/split_conformal.py` —
    `SplitConformal` class with `calibrate(predictions, targets)`
    + `predict(point) → ConformalInterval`. Two non-conformity
    score types (absolute, signed). The Angelopoulos-Bates
    `⌈(n+1)(1-α)⌉/n` finite-sample quantile correction is in
    place.
  - Diagnostics: `empirical_coverage`, `average_interval_width`.
  - `scripts/research/conformal_substrate_spike.py` — two-experiment
    measurement harness emitting `sum.split_conformal_spike.v1`
    receipt.
  - `Tests/test_conformal.py` — 19 contract tests covering the
    coverage guarantee on synthetic exchangeable data, score-type
    variants, determinism, edge cases, and diagnostics.

## Experiment 1 — synthetic coverage sweep (PROVABLE kernel)

Setup: `y = 2x + ε`, ε ~ N(0,1), naive constant-mean predictor.
n=2000 calibration + n=2000 test, averaged over 5 seeds.

| α    | target  | empirical             | mean width |
|-----:|--------:|----------------------:|-----------:|
| 0.05 |    0.95 | 0.9521 ± 0.0050       |      19.68 |
| 0.10 |    0.90 | 0.9060 ± 0.0078       |      18.33 |
| 0.20 |    0.80 | 0.8060 ± 0.0119       |      16.18 |

**All three α values hit target within 1 SD.** The
finite-sample-coverage guarantee (Angelopoulos-Bates Theorem 1,
2023) verified on our implementation. The point predictor is
deliberately bad (constant mean for a linear y vs x); conformal
hits coverage anyway — that's the *distribution-free* property
in action.

## Experiment 2 — substrate triple-quality classifier

Demonstrates the wrap-any-predictor pattern at substrate scale.
Workflow:
  - Use the deterministic sieve to extract triples from
    `seed_v1` and `seed_v2` documents
  - Target y_i ∈ {0, 1}: 1 if extracted triple matches a
    `gold_triple`, 0 otherwise
  - Predictor: ridge regression on a 9-feature vector (string
    lengths, vocab membership flags, casing flags)
  - 3-way split: 40% train / 40% cal / 20% test
  - Wrap with `SplitConformal(alpha=0.1)`

Receipt: `fixtures/bench_receipts/split_conformal_spike_*.json`.

| corpus  | n_train | n_cal | n_test | target | empirical | width |
|---------|--------:|------:|-------:|-------:|----------:|------:|
| seed_v1 |      20 |    20 |     10 |   0.90 |    1.000  | 0.031 |
| seed_v2 |    n/a  |  n/a  |   n/a  |   n/a  | skipped — only 16 labeled triples | — |

Coverage hits 100 % at n_test=10 (above-target due to
finite-sample over-coverage at very small N — expected behaviour
that doesn't violate the guarantee). Width is 0.031 on a [0, 1]
target — sharp enough to be useful as a confidence flag.

The substrate experiment scope is small because the existing
labeled corpus is small (`gold_triples` per doc in
`seed_v1.json`). Doesn't matter for the spike's purpose: the
wrap pattern is verified end-to-end on real SUM data flowing
through the existing extractor, and the math kernel is verified
at scale via Experiment 1.

## What this unblocks

This is the kernel needed to upgrade any of SUM's existing
readouts from "point estimate" → "calibrated CI":

  - **Slider-axis readouts** (density / length / formality /
    audience / perspective) — wrap each axis output with
    `SplitConformal(alpha=…)` calibrated on a held-out
    distillation corpus. Users see "axiom confidence ∈ [0.71, 0.89]
    @ 90 %" instead of "0.80".
  - **Sheaf-Laplacian hallucination scores** — the existing
    detector emits a scalar; conformal would give it a CI.
    (The wider research arc's "multiplier bootstrap" is the
    spectral-specific complement.)
  - **Per-axiom corruption_score** (from the RPCA spike) — bound
    "axiom is corrupt" decisions with finite-sample false-positive
    guarantees.

The wiring of any specific readout is left as a follow-on PR
per readout. This PR ships the kernel + tests + receipt; future
PRs wrap individual surfaces.

## Honest tier table

| component                                               | tier        |
|---------------------------------------------------------|-------------|
| Coverage guarantee (Vovk 2005, Angelopoulos-Bates 2023) | [provable]  |
| `⌈(n+1)(1-α)⌉/n` finite-sample correction               | [provable]  |
| Synthetic coverage hits 1-α to ±0.01 across 5 seeds     | [certified] |
| Substrate triple-quality wrap end-to-end                | [empirical, scope-limited by corpus size] |
| Wired into production slider readouts                   | [not yet]   |

## Update 2026-06-02 — rate-guarantee extension (method gap closed)

The split-conformal kernel gives a two-sided *interval* around a point
predictor. The slider contract + bench-hardening T3 actually ask a
one-sided **rate** question: *"with confidence ≥ 1-δ, fact preservation
≥ X?"*. That method gap is now closed (zero API cost):

  - `sum_engine_internal/research/conformal/risk_control.py` — a
    finite-sample, distribution-free lower confidence bound on a
    preservation rate. **Hoeffding** (any [0,1] data) +
    **Clopper–Pearson** (exact, binary per-fact). `certify_rate(...)`
    auto-dispatches; returns a `RateGuarantee`.
  - `Tests/test_conformal_risk_control.py` — 30 tests; the load-bearing
    layer is empirical coverage ≥ 1-δ across (rate, n, δ, method).
  - `scripts/research/conformal_rate_guarantee_spike.py` →
    `sum.conformal_rate_guarantee_spike.v1` receipt. Exp 1: synthetic
    coverage sweep (8/8 hit ≥ 1-δ). Exp 2 (real SUM data, no LLM):
    seed_v1 sieve triples certify *"≥ 94.2% gold at 95% confidence,
    n=50"*.

**What this still does NOT do:** certify the *slider's LLM-axis* fact
preservation. That needs the T2 calibration corpus (per-cell
source→render→re-extract labels) — the same `$ + OPENAI_API_KEY` gate
that blocks T2/T3 today. This wrapper is the analysis layer that sits on
top of that corpus; it does not replace the need to generate it.
Honest boundary: the guarantee holds *within the tested envelope* (the
T2 capability region) under exchangeability — state the envelope with
the rate. See memory `project_conformal_slider_guarantee_2026-06-01`.

| Rate lower bound (Hoeffding / Clopper–Pearson)          | [provable]  |
| Empirical coverage ≥ 1-δ across the synthetic sweep     | [certified] |
| Real-data rate certification on seed_v1 sieve triples   | [empirical, scope-limited by corpus size] |
| Slider LLM-axis fact-preservation certified             | [not yet — needs T2 corpus] |
