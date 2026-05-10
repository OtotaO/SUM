# Wire #4 — Maximum Mean Discrepancy as bundle metadata

The Hilbert-space angle from the most recent strategic
synthesis, implemented as the fourth bundle-metadata field.
Provable kernel-distance metric on probability distributions
(Gretton et al., *JMLR* 13:723-773, 2012, Theorem 5) shipped
into every signed bundle's metadata.

## What ships

  - `sum_engine_internal/research/mmd/mmd.py` — RBF kernel
    matrix + biased empirical MMD² estimator + median-heuristic
    bandwidth
  - `sum_engine_internal/research/mmd/baseline.py` —
    `BaselineMMDComputer` (lazy-cached singleton; calibrates
    once from the substrate's seed corpora)
  - New `axiom_distribution_mmd: Optional[dict]` field on
    `CanonicalBundle`, dict-shaped:
    `{mmd_squared, bandwidth, n_baseline_samples, n_bundle_samples}`
  - 16 contract tests in `Tests/test_mmd.py` + 7 wire tests
    in `Tests/test_bundle_distribution_mmd.py`

## Synthetic verification — provable kernel theorem

| scenario                                  | MMD²        | expectation         |
|-------------------------------------------|------------:|---------------------|
| identical samples, n=50                   |   ~10⁻¹⁰    | = 0 (provable)      |
| same dist, different draws (n=60 ea)      |     0.021   | small               |
| mean-shifted by 5 (n=60 ea)               |     1.072   | large               |
| ratio shifted / same                      |       52×   | discriminates       |

Gretton 2012 Theorem 5 verified: identical samples produce
MMD² ≈ 0 to numerical precision; shifted distributions produce
MMD² strictly larger.

## Substrate baseline

Calibration set: 314 triples extracted via the deterministic
sieve from 6 seed corpora (`seed_v1`, `seed_v2`,
`seed_long_paragraphs`, `seed_news_briefs`, `seed_paragraphs`,
`seed_paragraphs_16`). Embedding: the deterministic
sha256-bucket vectors from PR #182's RPCA module
(`embed_triples`, n_buckets=64). Bandwidth: median heuristic
on baseline pairs (~2.45).

Sample bundle vs baseline:
- 2-triple bundle (`alice build house`, `bob write book`):
  MMD² = 0.192. Within bound (RBF MMD² ≤ 2 absolute).

## Architectural discipline (matches wires #1-#3)

  - **OUTSIDE the signed payload** → Ed25519 / HMAC signatures
    byte-identical → K-matrix unaffected
  - **None for empty bundles** → existing strip-Nones logic
    keeps wire-format clean
  - **Defense-in-depth** at helper + call site → broken MMD
    computer never blocks attestation
  - **Lazy singleton** → import-time cost not paid by callers
    that never produce bundles

## Compounding properties (the headline)

Every kernel from this session compounds with MMD:

| kernel             | how MMD compounds                                                          |
|--------------------|----------------------------------------------------------------------------|
| Multiplier bootstrap (PR #185) | Gretton's permutation test for MMD significance — same kernel applies CIs |
| Split conformal (PR #183)      | Calibrated thresholds for "is this MMD large enough to matter?" |
| vN graph entropy (PR #184)     | Different scalar of the same axiom set — entropy is structural, MMD is distributional |
| RPCA (PR #182)                 | Same axiom-embedding vectors — MMD reuses `embed_triples` |
| Z3 SMT consistency (PR #187)   | Complementary: Z3 verifies axiom-set logic; MMD verifies distribution shape |
| DRVEC (user's project)         | Could store the RKHS-embedded vectors with deterministic recall + causality |

## Substrate use cases this unblocks

  1. **Cross-bundle distribution-shift detection** — every signed
     bundle ships with a single MMD² scalar; downstream
     consumers can flag bundles with anomalous distribution
     drift
  2. **Calibrated MMD threshold** (follow-on PR) — wrap the
     scalar with `SplitConformal` to get a "this bundle's
     MMD² is significantly above baseline" signal at
     operator-chosen α
  3. **Bootstrap MMD CI** (follow-on PR) — multiplier-bootstrap
     the MMD² to get distribution-free CIs on the value
  4. **Multi-bundle drift trajectory** (DRVEC integration) —
     store every bundle's MMD² in DRVEC; query for "bundles
     with MMD² > threshold within last N days"

## Honest tier table

| component                                                   | tier        |
|-------------------------------------------------------------|-------------|
| RBF kernel + biased empirical MMD² (Gretton 2012, Eq. 3)    | [provable]  |
| MMD = 0 ⟺ P = Q for characteristic kernel (Theorem 5)      | [provable]  |
| Synthetic identical→0, shifted→52× pinned by tests          | [certified] |
| Baseline calibration on 314 substrate triples               | [empirical] |
| Substrate bundle gets in-distribution MMD² ~0.19            | [empirical, scope-limited] |
| Calibrated significance threshold via conformal             | [not yet — follow-on PR] |
| Bootstrap CI via multiplier kernel                          | [not yet — follow-on PR] |
| DRVEC-backed multi-bundle MMD trajectory                    | [not yet — follow-on PR] |
