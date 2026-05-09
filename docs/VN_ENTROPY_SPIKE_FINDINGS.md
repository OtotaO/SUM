# Von Neumann graph entropy spike — quick win #2 findings

The wide-net survey's #1 physics pick implemented as a
substrate-shaped quick win. A single scalar drift detector for
the axiom graph, complementing (not duplicating) the existing
sheaf-Laplacian hallucination check.

## What landed

  - `sum_engine_internal/research/spectral_entropy/vn_entropy.py`
    — the kernel:
    `build_axiom_graph(triples) → (nodes, A)`,
    `normalized_laplacian(A) → L`,
    `density_matrix(L) → ρ = L/Tr(L)`,
    `von_neumann_entropy(ρ) → S(ρ) = -Σ λ_i log λ_i`,
    plus `graph_entropy(triples)` one-shot wrapper.
  - `scripts/research/vn_entropy_substrate_spike.py` —
    three-experiment measurement harness emitting
    `sum.vn_entropy_substrate_spike.v1` receipt.
  - `Tests/test_vn_entropy.py` — 19 contract tests covering
    bounds, determinism, drift sensitivity, pipeline correctness,
    numerical robustness.

## Experiment 1 — synthetic K_n upper bound (PROVABLE kernel)

K_n (complete graph on n nodes) is the maximally-mixing graph.
The De Domenico-Biamonte density matrix has theoretical maximum
entropy `log(N-1)` on K_N. Our implementation must hit this
exactly.

| n  | edges | entropy        | log(n-1)       | abs error |
|---:|------:|---------------:|---------------:|----------:|
|  3 |     3 | 0.6931471806   | 0.6931471806   |  ≈ 0      |
|  5 |    10 | 1.3862943611   | 1.3862943611   |  ≈ 0      |
| 10 |    45 | 2.1972245773   | 2.1972245773   |  ≈ 0      |
| 20 |   190 | 2.9444389792   | 2.9444389792   |  ≈ 0      |
| 50 |  1225 | 3.8918202981   | 3.8918202981   |  ≈ 0      |

**All five sizes hit the theoretical max within numerical
precision.** The implementation matches De Domenico & Biamonte
*Phys. Rev. X* 6:041062 (2016).

## Experiment 2 — substrate corpus entropy

Single scalar per corpus, computed on the
deterministic-sieve-extracted axiom graph.

| corpus               | triples | nodes | edges | S        |
|----------------------|--------:|------:|------:|---------:|
| seed_long_paragraphs |     120 |   215 |   119 |   4.7510 |
| seed_news_briefs     |      66 |   120 |    66 |   4.1383 |
| seed_paragraphs      |      20 |    35 |    20 |   2.9203 |

Each corpus has a single comparable signature. Stable across
runs. Receipt-attachable. Cross-machine reproducible (depends
only on triple set + lex sort).

## Experiment 3 — drift sensitivity (substrate use case)

Inject N off-corpus triples (`junk_X / glorp_X / frobozz_X`) into
each corpus's axiom graph. Measure ΔS = S(graph + junk) − S(clean).

| corpus               | S_clean | slope per corruption | ΔS @ 1 | ΔS @ 5 | ΔS @ 10 | ΔS @ 20 |
|----------------------|--------:|---------------------:|-------:|-------:|--------:|--------:|
| seed_long_paragraphs |  4.7510 |             +0.00791 | +0.009 | +0.042 |  +0.083 |  +0.159 |
| seed_news_briefs     |  4.1383 |             +0.01365 | +0.016 | +0.077 |  +0.148 |  +0.277 |
| seed_paragraphs      |  2.9203 |             +0.03514 | +0.052 | +0.238 |  +0.431 |  +0.731 |

**Drift response is monotonic and approximately linear in
corruption count.** Slope varies inversely with corpus size —
smaller corpora are more sensitive, as expected (one new edge is
a larger fraction of the graph). The substrate-monitor application
is direct: alert when `|ΔS| > k σ` from a corpus's baseline.

A subtle validation: across 5 random seeds per corruption level,
the std of ΔS was 0 — different junk-string labels, same graph
structure, identical entropy. That's the same relabelling-invariance
the contract test
`test_entropy_is_invariant_under_predicate_relabelling` pins,
showing up in the bench data.

## What this unblocks

  - **Drift monitor receipt-line**: nightly bundle snapshots can
    emit S(ρ) as a single scalar; cross-bundle |ΔS| > 2σ becomes
    a pre-alert before downstream tasks regress.
  - **Substrate-version A/B**: comparing two extraction pipelines
    over the same corpus reduces to a scalar entropy delta —
    cheaper than full sheaf-Laplacian recomputation.
  - **Compounds with conformal prediction (PR #183)**: entropy is
    a real-valued scalar; wrap it with `SplitConformal(alpha=0.1)`
    calibrated on a held-out drift baseline → calibrated tripwire
    instead of magic threshold. (Follow-on PR.)

## Honest tier table

| component                                                | tier        |
|----------------------------------------------------------|-------------|
| von Neumann entropy formulation (vN 1932)                | [provable]  |
| Density matrix from Laplacian (De Domenico-Biamonte 2016)| [provable]  |
| K_n upper bound `log(N-1)` matched at n ∈ {3,...,50}     | [certified] |
| Substrate corpus entropy single scalar                   | [empirical] |
| Drift response monotonic in corruption count             | [empirical] |
| Wired into bundle-snapshot CI pipeline                   | [not yet]   |
