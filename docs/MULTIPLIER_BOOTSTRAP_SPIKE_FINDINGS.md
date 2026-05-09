# Multiplier-bootstrap substrate spike — research arc PR #1

The wide-net survey's MED-HIGH statistics pick (Chernozhukov-
Chetverikov-Kato, *Annals of Statistics* 2013), implemented as a
substrate-shaped quick win that surfaces both a verified math
kernel and an honest methodological caveat for the substrate
application.

## What landed

  - `sum_engine_internal/research/bootstrap/multiplier_bootstrap.py`
    — `multiplier_bootstrap(samples, statistic_fn, B)` returning
    `(point, replicates)`; `bootstrap_ci(point, replicates, alpha)`
    extracting per-component intervals; Gaussian + Rademacher
    multiplier helpers.
  - `scripts/research/multiplier_bootstrap_substrate_spike.py` —
    three-experiment harness emitting
    `sum.multiplier_bootstrap_substrate_spike.v1`.
  - `Tests/test_multiplier_bootstrap.py` — 16 contract tests
    covering coverage on a known mean, multi-component statistics,
    determinism via rng, edge cases.

## Experiment 1 — synthetic mean coverage (PROVABLE kernel)

Bootstrap a CI on the sample mean across 100 trials per α
setting, n=200 per trial, B=300:

| α    | target | empirical (±SE)  | mean width |
|-----:|-------:|-----------------:|-----------:|
| 0.05 |   0.95 | 0.950 ± 0.022    |      0.286 |
| 0.10 |   0.90 | 0.890 ± 0.030    |      0.244 |
| 0.20 |   0.80 | 0.760 ± 0.040    |      0.190 |

**All three α values land within 1 SE of target.** The CCK 2013
multiplier bootstrap works correctly on its native domain — iid
samples + a smooth statistic.

## Experiment 2 — eigenvalue CIs on substrate Laplacian

Bootstrap the top-5 eigenvalues of the Laplacian built from
real corpus axiom graphs. **Result is mixed and honest:**

| corpus               | eigenvalue | full value | CI                | width  |
|----------------------|-----------:|-----------:|------------------:|-------:|
| seed_long_paragraphs | λ_1        |    5.0861  | [3.6527, 5.6314]  | 1.979  |
| seed_long_paragraphs | λ_2        |    5.0000  | [3.3195, 4.5934]  | 1.274  |
| seed_long_paragraphs | λ_3        |    3.0000  | [3.2229, 4.1013]  | 0.879  |
| seed_long_paragraphs | λ_4        |    3.0000  | [3.1393, 3.7327]  | 0.593  |
| seed_long_paragraphs | λ_5        |    2.4280  | [3.0679, 3.6152]  | 0.547  |
| seed_news_briefs     | λ_1        |    4.1701  | [3.3715, 5.0472]  | 1.676  |

For λ_1 (top eigenvalue), the CI contains the full value. For
λ_3 / λ_4 / λ_5, **the full value falls OUTSIDE the bootstrap CI**.
This is an honest negative result that surfaces a methodological
caveat: row-resampling an adjacency matrix doesn't preserve the
graph's iid structure (the rows aren't iid — they encode each
node's connectivity). Resampling inflates degrees of repeated
rows, which shifts the spectrum systematically.

## Experiment 3 — CI on von Neumann entropy

Same caveat manifests starkly:

| corpus               | full S | bootstrap mean | CI                 | width  |
|----------------------|-------:|---------------:|-------------------:|-------:|
| seed_long_paragraphs | 4.7510 |        4.7510  | [5.0189, 5.0481]   | 0.029  |
| seed_news_briefs     | 4.1383 |        4.1383  | [4.3731, 4.4245]   | 0.051  |

**Tight CIs, but the full entropy is below the lower bound.**
Same root cause as Experiment 2: row-resampling biases the
graph's degree distribution upward, so the bootstrap distribution
is concentrated around an inflated entropy estimate.

## What this tells us

The CCK 2013 multiplier bootstrap is provably valid for sup-norm
functionals of high-dimensional means under iid sampling. **It
is NOT valid out-of-the-box for graph-spectral statistics**
because the iid-rows assumption fails for adjacency matrices.
The math kernel works (Experiment 1); the simplest substrate
application is a methodological mismatch.

The article-survey agent's pitch ("multiplier bootstrap on
sheaf-Laplacian spectrum gives CIs instead of magic thresholds")
needs more nuance than the agent surfaced. Three honest paths
forward:

1. **Edge bootstrap** — resample edges (the "iid" units in the
   Erdős-Rényi-style model) rather than rows. The graphon
   bootstrap literature (Bickel-Chen-Levina, *Annals of Statistics*
   2011; Green & Shalizi, *Statist. Sci.* 2017) is the right
   theoretical home.
2. **Subsample bootstrap** — resample node subsets without
   replacement at size m << n (Bickel-Sakov 2008). Avoids the
   degree-inflation issue.
3. **Different problem framing** — instead of CIs on graph
   spectra, use the bootstrap for what it's *good at*: scalars
   that can be expressed as iid sample means. E.g., bootstrap
   the per-document axiom-count distribution; bootstrap a
   ridge readout's residuals; bootstrap any statistic of a
   document-level (not graph-level) sample.

## Honest tier table

| component                                                     | tier        |
|---------------------------------------------------------------|-------------|
| Multiplier bootstrap convergence (CCK 2013)                   | [provable]  |
| Coverage on iid mean: empirical = target ±1 SE                | [certified] |
| Coverage on graph eigenvalues via row-resample                | **[FAILS — methodological mismatch]** |
| Coverage on graph entropy via row-resample                    | **[FAILS — same caveat]** |
| Path forward via edge / subsample / graphon bootstrap         | [open question — iteration 2] |

## Compounds with PRs #183, #184

Despite the graph-application caveat, the kernel is directly
useful for the **scalar / per-document** detectors already in
SUM:

  - **RPCA corruption_score** (PR #182) — per-row scalar that's
    iid across documents → bootstrap is well-defined here.
  - **Slider-axis residuals** — when conformal prediction is
    wired into a specific axis (PR #183 follow-on), the
    bootstrap can give an alternative CI for cross-validation.
  - **Per-document NCD distances** (substantive-math agent's #1
    pick from earlier) — would also be iid-bootstrap-friendly.

This PR ships the kernel + tests + receipt + honest findings.
The graph-spectrum-specific bootstrap is an iteration-2 task.
