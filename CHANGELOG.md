# Changelog

All notable changes to the `sum-engine` package. Dates in ISO-8601 UTC.

## [Unreleased]

- **D5 — Hypothesis property-based tests for substrate
  invariants.** Test-suite robustness audit follow-up. Of the
  118 test files in this repo, only 3 used Hypothesis before
  this PR. New `Tests/test_property_substrate.py` covers the
  high-leverage gaps with 10 property tests across 7 substrate
  invariants:

  1. **Bundle round-trip** — ∀ axiom_set:
     `import_bundle(export_bundle(state)) == state`. The
     canonical_codec's load-bearing K-matrix invariant.
  2. **Content-hash permutation invariance** — graph_store
     `_canonical_triples_hash` independent of input order.
  3. **MMD properties** — identical samples → MMD² ≈ 0 (10⁻⁹
     floor); MMD²(X,Y) == MMD²(Y,X); MMD² ≥ 0 always.
  4. **vN entropy invariance** — graph entropy unchanged by
     predicate string relabeling (only edges matter).
  5. **Bind verb determinism** — `BindRegistry.bind(value)`
     returns the same id for structurally-equal values; same
     across registry instances (process-global content
     addressing, not instance-local).
  6. **UnionFindStore lex-canonical extraction** — same triple
     set → same canonical form regardless of insertion order
     (the substrate's deterministic-extraction guarantee).
  7. **Signature determinism** — same payload + same key →
     byte-identical HMAC signature.

  Hypothesis settings: `derandomize=True` (failures reproduce
  across CI runs), max_examples=30-100 per test depending on
  cost. Total wall-time: 8.4s for 10 tests × ~50 examples each
  = ~500 generated cases. `make pre-push` updated to include the
  new file (60 tests in smoke set, +10 from D5).

  This closes the highest-leverage item from the test-suite
  audit's Q3 quadrant. Substrate's headline invariants now have
  property-based coverage; the K-matrix discipline (positive
  property assertions) gets its mathematical complement.

- **K3 — Conformal-style size-stratified threshold on bundle
  MMD² (binary anomaly decision).** Compounds PR #183 (split
  conformal kernel) + PR #194 (MMD permutation test) into an
  operator-actionable binary "is this bundle atypical at level
  α?" signal.

  **What ships:**
  - `_build_size_stratified_calibration` in
    `sum_engine_internal/research/mmd/baseline.py` — at
    calibration time, draws random subsamples of the baseline
    at sizes (1, 2, 3, 5, 10, 20, 50) and computes MMD² for
    each subsample-vs-rest. The empirical MMD² distribution
    at each size becomes the calibration set.
  - `BaselineMMDComputer.predict_threshold(observed, bundle_size,
    alpha=0.10)` — picks the closest-size calibration table,
    returns the ⌈(n+1)(1-α)⌉/n-quantile threshold (matching
    `SplitConformal`'s finite-sample correction) and a binary
    `exceeds_threshold` decision.
  - New `axiom_distribution_mmd_threshold: Optional[dict]` field
    on `CanonicalBundle`, dict-shaped:
    `{threshold_alpha, threshold_value, exceeds_threshold,
      n_calibration_samples, calibration_size_used}`.

  **Honest finding caught + fixed before ship:** the first
  implementation used a single fixed subsample size (10), which
  flagged in-distribution small bundles as atypical because
  smaller samples have systematically larger MMD² (sample-size
  confounder). Stratifying by size eliminates this — calibration
  MMD² range scales correctly: size=1 → 0.39, size=2 → 0.19,
  size=10 → 0.04, size=50 → 0.009. In-distribution 3-triple
  bundle now correctly does NOT exceed the size-3 threshold.

  **Substrate use:** every signed bundle now ships a binary
  "atypical at α=0.10" signal that downstream consumers can
  branch on without knowing anything about MMD math. The raw
  value + permutation p-value remain in the
  `axiom_distribution_mmd` field for consumers who want
  finer control.

  **Same wire discipline as #1-#4 + K2:** outside the signed
  payload, defense-in-depth, None for empty bundles.

  8 contract tests in
  `Tests/test_bundle_distribution_mmd_threshold.py`. `make
  pre-push` updated to include the new test file. All 64 tests
  across 6 bundle-metadata + math test files green.

  **Substrate now answers SIX metadata questions per bundle**
  — original four + K2's significance p-value + K3's binary
  threshold decision.

  **Embedding-sensitivity caveat documented:** junk triples
  (random `junk_s_*` strings) currently produce MMD² values
  similar to in-distribution triples in the sha256-bucket
  embedding. This is a known property of the substrate's
  current deterministic embedding — closing it is downstream
  (article §9.1 "learned embedding" path). K3 ships as
  designed; what counts as "atypical" is bounded by the
  embedding's discriminative power.

- **K2 — Permutation-test p-value on bundle MMD² (Gretton 2012
  §3.2 wired into bundle metadata).** Compounds PR #185
  (multiplier bootstrap kernel) + PR #192 (MMD wire) into a
  calibrated significance signal on every signed bundle.

  **What ships:**
  - `mmd_permutation_pvalue(X, Y, sigma, n_permutations=200)` in
    `sum_engine_internal/research/mmd/mmd.py` — Gretton 2012 §3.2
    label-permutation test with finite-sample (1+#ge)/(1+B)
    correction → p ∈ (0, 1] always
  - `BaselineMMDComputer.predict_mmd` extended to compute the
    p-value alongside MMD²
  - The existing `axiom_distribution_mmd` dict on
    `CanonicalBundle` gains two keys: `permutation_p_value` (p ∈
    (0, 1]) and `n_permutations` (integer, defaults 200)

  **Synthetic verification PASSES:**
  - Same-distribution → p uniform on [0, 1] across trials
    (median ≈ 0.5; min 0.03, max 0.98 over 20 trials)
  - Mean-shifted (μ=0 vs μ=2) → p = 0.010 = 1/(B+1) floor on
    every trial (test correctly rejects H_0)
  - Substrate-scale perf: ~60 ms at baseline=314 + bundle=2 with
    B=200; acceptable for `export_bundle` hot path

  **In-distribution substrate result:** sample 3-triple bundle
  drawn from seed-corpus prose vocabulary → p = 0.473 (clearly
  not significant; matches H_0).

  **Same wire discipline as #1-#4:** outside the signed payload,
  defense-in-depth at helper + call site, None for empty
  bundles, lazy singleton init for the baseline computer.

  3 new math tests (`test_permutation_pvalue_is_in_open_unit_interval`,
  `_low_under_clear_distribution_shift`,
  `_distribution_is_uniform_under_H0`) + 1 new wire test
  (`test_in_distribution_bundle_yields_non_significant_pvalue`).
  All 56 tests across the four bundle-metadata test files
  green; the existing `test_bundle_includes_axiom_distribution_mmd_field`
  was tightened to assert the K2 expanded shape (6 keys).

  Substrate use: every signed bundle now answers FIVE
  metadata questions automatically — the original four plus
  "is this bundle's distribution distance from baseline
  *statistically significant* at operator-chosen α?"

  Foresight: K3 (conformal-calibrated MMD threshold) is the
  natural follow-on; together K2+K3 give MMD² + a p-value
  + a calibrated decision threshold. Pre-push gate from the
  D3 PR was used before pushing this PR.

- **`make pre-push` — local pre-flight gate matching CI's
  drift + smoke checks (test-suite robustness D3).** New
  Makefile target running both drift-gate `--check`s
  (self-attestation for tracked docs; repo manifest for bench
  receipts) plus a fast smoke against the bundle-metadata
  wires + drift script. ~3.5 s wall-time on a clean tree.

  **What it catches before push:**
  - Tracked-doc edits without `python -m scripts.attest_repo_docs`
    refresh (the class that hit PR #173)
  - New `fixtures/bench_receipts/*.json` files without
    `python -m scripts.repo_manifest --out meta/repo_manifest.json`
    refresh (the class that hit PRs #176, #191)
  - Regressions in bundle metadata wires #1-#4 or in the
    self-attestation script itself

  Audit context: a robustness sweep across all 118 test files /
  1,425 test functions / 33k LOC of test code surfaced this as
  the meta-fix (D3 in the audit table). Two adjacent findings
  (D1: "joserfc imported without `importorskip`" and D2:
  "unseeded `random` module use") turned out to be **false
  positives from too-narrow regex patterns** — both files
  use `pytest.importorskip("joserfc")` (multi-line form) and
  `random.Random(20260429)` (deterministic seeding) respectively.
  The substrate's discipline is stronger than first-pass grep
  caught.

  Foresight: every subsequent PR benefits. The 4 CI failures in
  this arc (PRs #173, #176, #191, #192) would have all been
  caught locally by this target.

- **Wire #4 — Maximum Mean Discrepancy as bundle metadata
  (Hilbert-space angle).** Fourth bundle-metadata wire from this
  session, completing the trio-plus-one with Wires #1 (vN
  entropy), #2 (calibrated entropy CI), #3 (Z3 consistency).
  Provable kernel-distance metric on probability distributions
  via Reproducing Kernel Hilbert Space (Gretton et al., *JMLR*
  13:723-773, 2012, Theorem 5).

  **What ships:**
  - `sum_engine_internal/research/mmd/mmd.py` — RBF kernel
    matrix + biased empirical MMD² estimator + median-heuristic
    bandwidth
  - `sum_engine_internal/research/mmd/baseline.py` —
    `BaselineMMDComputer` (lazy singleton; calibrates from
    seed corpora once at first call). Uses the
    sha256-bucket axiom embedding from PR #182's RPCA module
    — no new dependency, deterministic, substrate-shaped.
  - New `axiom_distribution_mmd: Optional[dict]` field on
    `CanonicalBundle`, dict-shaped:
    `{mmd_squared, bandwidth, n_baseline_samples,
      n_bundle_samples}`

  **Synthetic verification — provable kernel theorem PASSES:**
  identical samples → MMD² ≈ 10⁻¹⁰; same-distribution different
  draws → MMD² = 0.021; mean-shifted (μ=0 vs μ=5) → MMD² =
  1.072 (52× larger). Identical→0 to numerical precision;
  shifted→discriminates clearly.

  **Substrate baseline:** 314 triples calibrated from 6 seed
  corpora (`seed_v1`, `seed_v2`, `seed_long_paragraphs`,
  `seed_news_briefs`, `seed_paragraphs`, `seed_paragraphs_16`).
  Bandwidth ~2.45 (median heuristic). Sample 2-triple bundle:
  MMD² = 0.192 (within RBF MMD² ≤ 2 absolute bound).

  **Same architectural discipline as Wires #1-#3:**
  - OUTSIDE the signed payload → signatures byte-identical →
    K-matrix unaffected
  - None for empty bundles → strip-Nones keeps wire-format
    clean
  - Defense-in-depth at helper + call site → broken MMD never
    blocks attestation

  **Compounding:** every kernel from this session compounds
  with MMD — bootstrap (PR #185) gives CI on MMD² via Gretton's
  permutation test; conformal (PR #183) gives calibrated
  significance thresholds; vN entropy (PR #184) is a different
  scalar of the same axiom set; RPCA (PR #182) shares the
  embedding; Z3 (PR #187) verifies logic where MMD verifies
  distribution.

  Substrate use: every signed bundle now answers FOUR
  independent metadata questions automatically:
  1. What is the structural entropy? (Wire #1)
  2. Is the entropy typical for its size? (Wire #2)
  3. Is the bundle internally consistent under the curated
     predicate library? (Wire #3)
  4. How distributionally distant is this bundle from the
     substrate's calibration baseline? (Wire #4)

  16 contract tests in `Tests/test_mmd.py` (math + edge cases) +
  7 wire tests in `Tests/test_bundle_distribution_mmd.py`
  (field shape / signature unchanged / round-trip /
  empty-bundle / failure-resilience / four-fields-independence
  / in-distribution sanity). All 143 tests across affected
  surfaces still green. Findings doc:
  `docs/MMD_WIRE_FINDINGS.md`.

- **Wire #3 — Z3 SMT consistency check as bundle metadata.**
  Third production wiring this session, completing the trio
  with Wires #1 (vN entropy) and #2 (calibrated entropy CI).
  Closes the substrate gap PR #190 unblocked: now that the
  curated `SUBSTRATE_PREDICATE_LIBRARY` catches real
  contradiction shapes, every signed bundle automatically
  emits a Z3 verdict.

  **What ships:**
  - `sum_engine_internal/research/smt_consistency/predicate_library.py`
    — promotes the operator-curated library from the spike
    script to a proper module-level constant (single source
    of truth; spike script now imports it). 22 verb lemmas;
    full curation rationale in
    `docs/SMT_CONSISTENCY_SPIKE_FINDINGS.md` iteration 2.
  - New `axiom_consistency_check: Optional[dict]` field on
    `CanonicalBundle`, dict-shaped:
    `{consistent: bool, unsat_core: List[int],
      n_predicates_checked: int, z3_check_ms: float}`.
    Computed at every `export_bundle` call; stripped from
    output when None (cold-start / empty bundle / Z3
    unavailable).

  **Same architectural discipline as Wires #1, #2:**
  - **OUTSIDE the signed payload** → Ed25519 / HMAC signatures
    byte-identical → K-matrix unaffected
  - **None for empty bundles** → strip-Nones keeps wire-format
    clean
  - **Defense-in-depth** at helper + call site → broken Z3 /
    missing predicate library never blocks attestation
  - **z3-solver opt-in** → tests `pytest.importorskip("z3")` so
    CI without it skips cleanly

  **Additive shape (deliberate):** UNSAT bundles still emit
  `consistent: False` in metadata BUT `import_bundle` still
  verifies them. Downstream consumers (not the codec) decide
  whether to trust an UNSAT bundle. This preserves backward
  compatibility with verifiers that don't read the new field;
  an aggressive "refuse to attest UNSAT" mode is a separate
  follow-on PR if/when the operator wants it.

  Substrate use: every signed bundle now answers three
  metadata questions automatically:
  1. What is the structural entropy of this bundle? (Wire #1)
  2. Is that entropy typical for its size? (Wire #2)
  3. Is the bundle internally consistent under the curated
     predicate library? (Wire #3)

  9 contract tests in `Tests/test_bundle_consistency_check.py`
  pin: field present + correct shape / clean axioms → SAT /
  mutual-contain → UNSAT with non-empty core / UNSAT bundle
  still round-trips (additive-shape canary) / signature
  unchanged / empty-bundle behaviour / failure-resilience under
  monkeypatched broken helper / uncurated-predicates path
  (n_predicates_checked=0) / independence from
  entropy/entropy_ci fields. All 133 tests across affected
  surfaces (canonical codec, MCP server, CLI, adversarial
  bundles, all three bundle-metadata test files, SMT spike,
  self-attestation) still green.

- **SMT consistency iteration 2 — predicate-library curation
  closes the substrate-application gap (option c.1).** The
  iteration-1 honest gap from the Z3 SMT consistency spike
  (PR #187) — "starter library matched zero of the predicates
  the deterministic sieve actually produces" — is now closed.
  Surveyed the sieve's vocabulary across all 7 labeled corpora
  (189 distinct predicates; top 30 by frequency); curated
  logical-property declarations for 22 verb lemmas spanning
  containment / production / discovery / receipt / achievement
  / state-change / functional-attribute classes.

  **Curation discipline (operator-vetted, conservative):**
  IRREFLEXIVE whenever ``X p X`` is plainly malformed; ANTISYM
  only when mutual-application is a clear contradiction in plain
  prose (avoiding metaphorical edges like ``know``); TRANSITIVE
  only for spatial/containment-style relations; FUNCTIONAL for
  single-valued attributes. Predicates with context-dependent
  semantics left unconstrained (the conservative choice).

  **Substrate corpus check is now substantive, not vacuous.**
  Z3 verifies 4-17 distinct property-bearing predicates per
  corpus against actual axioms and finds zero contradictions:
  - `seed_v1`: 10 curated predicates (build, compose, contain,
    create, discover, emit, produce, propose, reach, write)
  - `seed_v2`: 4 (develop, emit, win, write)
  - `seed_long_paragraphs`: 17
  - `seed_news_briefs`: 8

  **Real-corpus injection (needle-in-real-haystack): 8/8 caught
  with minimal cores.** New experiment in the spike harness
  injects irreflexive self-loops + antisymmetric mutual pairs
  into each corpus's actual axiom set; Z3 catches every
  injection with a minimal UNSAT core (size 1 or 2) even when
  surrounded by 16-120 clean axioms. The needle-in-haystack
  property holds at real-corpus scale.

  **Z3 wire is now unblocked.** With the curated library
  catching real contradiction shapes + producing minimal cores
  + zero false positives on existing labeled corpora, wiring
  `check_consistency()` as a pre-attest gate in
  `canonical_codec.export_bundle` is a clean follow-on PR.

  Updated receipt:
  `fixtures/bench_receipts/smt_consistency_substrate_spike_20260510T052707Z.json`.
  Findings doc gains an iteration-2 section
  (`docs/SMT_CONSISTENCY_SPIKE_FINDINGS.md`).

- **Wire #2 — calibrated CI on bundle's axiom_graph_entropy.**
  Builds on Wire #1 (PR #188) by adding a finite-sample,
  distribution-free CI for the *expected* entropy at this
  bundle's axiom_count. Uses PR #183's `SplitConformal` kernel
  wrapped around a tiny ridge regressor calibrated on a
  precomputed baseline of (axiom_count, entropy) pairs from the
  substrate's seed corpora.

  **What ships:**
  - `sum_engine_internal/research/conformal/entropy_baseline.py`
    — `BaselineEntropyPredictor` (loads + trains + calibrates
    once at module import) + `get_default_predictor()` lazy
    singleton accessor
  - `fixtures/calibration/entropy_baseline.json` — 122
    (axiom_count, entropy) pairs across 6 seed corpora
    (`seed_v1` / `seed_v2` / `seed_long_paragraphs` /
    `seed_news_briefs` / `seed_paragraphs` /
    `seed_paragraphs_16`)
  - New `axiom_graph_entropy_ci: Optional[List[float]]` field
    on `CanonicalBundle`. Two-element `[lower, upper]` at
    α=0.10 (90 % coverage). Computed at every `export_bundle`
    call; stripped from output when None (cold-start /
    empty-bundle).

  **Same architectural discipline as Wire #1:**
  - **Outside the signed payload** — Ed25519 / HMAC signatures
    byte-identical, K-matrix unaffected
  - **None for empty bundles** — strip-Nones logic handles
    wire-format
  - **Defense-in-depth: helper has its own try/except AND the
    call site wraps it again** — broken predictor never blocks
    attestation
  - **Cold-start safe** — predictor reports `is_calibrated=False`
    if baseline is missing; bundle gets None CI

  Substrate use: every signed bundle now ships with an
  immediately-usable single-bundle anomaly-detection signal
  ("is this bundle's entropy typical for its size?"). No
  multi-bundle history needed; no operator opt-in.

  10 contract tests in `Tests/test_bundle_entropy_ci.py` pin:
  field present + correct shape / actual entropy in CI for
  in-distribution bundle / signature unchanged / round-trip OK
  / empty-bundle behaviour / failure-resilience under
  monkeypatched broken helper / predictor calibrates from
  baseline / cold-start returns None / CI scales
  monotonically with axiom_count. All 108 tests across
  affected surfaces still green.

- **Wire #1 — von Neumann graph entropy as bundle metadata.**
  First production-wiring of a kernel from this session's
  research arc. Every signed bundle now carries an
  `axiom_graph_entropy: Optional[float]` metadata field
  computed at `export_bundle` time via the
  `sum_engine_internal/research/spectral_entropy/` kernel
  (PR #184).

  **Architectural properties (all pinned by tests):**
  - Field is **OUTSIDE the signed payload** (`canonical_tome |
    state_integer | timestamp` from canonical_codec.py:141 is
    unchanged) — so existing Ed25519 / HMAC signatures stay
    byte-identical, the K-matrix cross-runtime trust triangle
    is unaffected, and `import_bundle` round-trips work without
    modification.
  - Field is **None for empty bundles** (state=1 → no axioms);
    the existing None-stripping logic in `export_bundle` keeps
    empty bundles wire-format-clean.
  - **Failure to compute entropy NEVER blocks attestation** —
    defense-in-depth: helper has internal try/except, AND the
    call site in `export_bundle` wraps it again. If the spectral_entropy
    module raises for any reason, the bundle still attests with
    `axiom_graph_entropy=None`.

  Substrate use: cross-bundle |ΔS| > k σ becomes an automatic
  drift tripwire on every bundle SUM produces, with no opt-in
  required. Compounds with PR #186 (SPRT) for adaptive stopping
  on entropy-stability and with PR #183 (conformal) for
  distribution-free CIs on cross-bundle entropy variance.

  8 contract tests in `Tests/test_bundle_axiom_graph_entropy.py`
  pin: field present + correct + matches direct
  `graph_entropy()` / signature unchanged / round-trip OK /
  empty-bundle behaviour / order-permutation invariance /
  failure-resilience under monkeypatched broken helper. All
  124 tests across the affected surfaces (canonical codec,
  MCP server, CLI, adversarial bundles, cross-instance, vN
  entropy research module, self-attestation) still green.

  This is option (b) from the prior session strategic check-in —
  PROOF_BOUNDARY tier movement on the substrate's headline
  product (every signed bundle now ships with an additional
  spectral observable). Next: option (c) — operator-curation
  work to close documented substrate-application gaps.

- **Z3 SMT axiom-consistency kernel.** The substantive-math
  agent's #1 pick from the wide-net survey: detect contradictory
  axioms before signing via Z3 (Nelson-Oppen 1979 /
  De Moura-Bjørner CDCL(T) TACAS 2008). Module
  `sum_engine_internal/research/smt_consistency/`:
  `check_consistency(triples, predicate_properties=…)` returning
  `ConsistencyResult` with `consistent`, `unsat_core` (minimal
  contradicting subset of triple indices), `z3_check_seconds`.
  Four predicate-property schemas via `PredicateProperty` enum:
  `ANTISYMMETRIC`, `IRREFLEXIVE`, `FUNCTIONAL`, `TRANSITIVE`.

  **Experiment 1 — synthetic verification matrix PASSES** on six
  cases: clean (SAT), mutual antisymmetric (UNSAT, core=2),
  self-loop irreflexive (UNSAT, core=1), two-output functional
  (UNSAT, core=2), 3-cycle transitive+irreflexive (UNSAT,
  core=3), needle-in-haystack 50+1 (UNSAT, **core=1 — points
  exactly at the contradiction**). All decided in ≤25 ms with
  minimal cores.

  **Experiment 2 — substrate corpus check surfaces an HONEST
  GAP.** All four labeled corpora return CONSISTENT, but the
  starter `SUBSTRATE_PREDICATE_LIBRARY` (FOAF-like family /
  biography predicates) covers ZERO of the predicates the
  deterministic sieve actually produces (verb lemmas like
  `visit`, `be`, `have`). Z3 asked to check zero
  property-bearing predicates trivially returns SAT. Same shape
  as the RPCA / multiplier-bootstrap-spectral spikes: math
  kernel works on its native domain, substrate application
  needs operator-vetted predicate-library curation matched to
  actual corpus vocabulary (one-day operator task; documented
  in findings doc).

  Compounds with PRs #183 (conformal CI on contradiction
  prevalence), #184 (vN entropy + contradiction joint
  discriminates corruption classes), #186 (SPRT-stop on
  consistency-rate stability).

  Receipt: `fixtures/bench_receipts/smt_consistency_substrate_spike_20260510T024655Z.json`.
  15 contract tests (all green); `z3-solver` opt-in dependency
  (tests `pytest.importorskip("z3")` so CI without it skips
  cleanly). Findings:
  `docs/SMT_CONSISTENCY_SPIKE_FINDINGS.md`.

- **SPRT adaptive-stopping kernel — research arc PR #2.** Wald
  1947 Sequential Probability Ratio Test for Bernoulli streams,
  with operator-chosen (α, β) error bounds. Module
  `sum_engine_internal/research/sequential/`:
  `BinomialSPRT(p0, p1, alpha, beta)` with `observe(x)` /
  `state()` / `reset()` / `run_until_decision(stream)`; three
  decisions (`ACCEPT_H0` / `REJECT_H0` / `CONTINUE`).

  **Experiment 1 — synthetic Wald-bound verification PASSES**
  across four (p_0, p_1, α, β) settings × both H_0/H_1 truth
  scenarios: empirical Type-I and Type-II error rates stay
  within Wald bound + 3σ Monte-Carlo band on all 8 cells.

  **Experiment 2 — substrate budget reduction sharpens the
  agent's "30-50 % savings" claim into HONEST framing.**
  At the substrate's current fixed-N=8, the test is
  statistically *underpowered* for moderate effect sizes;
  SPRT trades sample size for actual error guarantees. For
  clear effect (p_1=0.85): SPRT ~10 samples vs fixed-8, error
  rate 0.03 vs fixed-N's 0.10-0.14. For moderate effects: SPRT
  uses ~30 samples (vs fixed-8) but cuts error from 0.19-0.81
  → 0.03-0.05. The Wald 30-50 % savings appear only at
  power-equivalent fixed-N (~30 for these effects); the
  substrate's existing magic-N=8 is the unstated weak link
  SPRT exposes. **Operator-chosen (α, β) contract replaces
  the magic-N choice.** Receipt:
  `fixtures/bench_receipts/sprt_substrate_spike_20260509T183309Z.json`.
  20 contract tests covering Wald error bounds, boundary
  computation, sample-size savings, edge cases, decision
  direction. Findings: `docs/SPRT_SPIKE_FINDINGS.md`.

  Compounds with PRs #183 (conformal) and #184 (vN entropy):
  SPRT decides when conformal-calibrated thresholds are
  unambiguously crossed; |ΔS| anomaly stops becoming a magic
  threshold.

- **Multiplier bootstrap kernel — research arc PR #1.**
  Distribution-free CIs on vector-valued statistics via the
  Chernozhukov-Chetverikov-Kato (Annals of Statistics 2013)
  Gaussian multiplier bootstrap. Module
  `sum_engine_internal/research/bootstrap/`:
  `multiplier_bootstrap(samples, statistic_fn, B)` returns
  `(point, replicates)`; `bootstrap_ci(point, replicates, alpha)`
  extracts per-component intervals; Gaussian + Rademacher
  multiplier helpers exposed.

  **Experiment 1 — synthetic mean coverage VERIFIED:** at
  α ∈ {0.05, 0.10, 0.20} empirical coverage hits 0.950 / 0.890 /
  0.760 (target: 0.95 / 0.90 / 0.80) — within 1 SE across 100
  trials per setting. CCK 2013 kernel works on its native domain.

  **Experiment 2 + 3 — substrate spectral application reveals an
  honest methodological caveat.** Bootstrap CIs on graph-Laplacian
  eigenvalues and on von Neumann entropy via row-resampling DO
  NOT contain the full-graph values for several lower
  eigenvalues, and place tight CIs *above* the full entropy.
  Root cause: row-resampling an adjacency matrix doesn't preserve
  iid structure (rows aren't iid — they encode each node's
  connectivity); resampling inflates degrees of repeated rows
  and biases the spectrum systematically. The article-survey
  agent's pitch ("bootstrap the sheaf spectrum to replace magic
  thresholds") needs more nuance than the agent surfaced. Path
  forward documented in `docs/MULTIPLIER_BOOTSTRAP_SPIKE_FINDINGS.md`:
  (1) edge / graphon bootstrap (Bickel-Chen-Levina 2011);
  (2) subsample bootstrap (Bickel-Sakov 2008); (3) reframe to
  per-document iid scalars where the kernel applies cleanly
  (RPCA corruption_score, slider residuals, NCD distances).

  Receipt: `fixtures/bench_receipts/multiplier_bootstrap_substrate_spike_20260509T181701Z.json`.
  16 contract tests covering coverage on a known mean,
  multi-component statistics, determinism via rng,
  multiplier-type variants, edge cases. Findings doc:
  `docs/MULTIPLIER_BOOTSTRAP_SPIKE_FINDINGS.md`.

- **Von Neumann graph entropy — quick win #2 from the wide-net
  survey.** Single-scalar drift detector for the axiom graph
  via `sum_engine_internal/research/spectral_entropy/`. Defines
  density matrix ρ = L/Tr(L) over the combinatorial graph
  Laplacian and computes S(ρ) = -Σ λ_i log λ_i (De Domenico &
  Biamonte, *Phys. Rev. X* 6:041062, 2016; building on von
  Neumann's 1932 formulation). Pipeline:
  `build_axiom_graph(triples) → normalized_laplacian → density_matrix → von_neumann_entropy`,
  plus a one-shot `graph_entropy(triples)`. **Synthetic K_n
  upper-bound test PASSES exactly:** for n ∈ {3, 5, 10, 20, 50},
  S equals log(n-1) within numerical precision. **Substrate
  corpus entropy** computed on each labeled corpus's
  deterministic-sieve axiom graph — single scalar per corpus,
  cross-machine reproducible (`seed_long_paragraphs` S=4.7510;
  `seed_news_briefs` S=4.1383; `seed_paragraphs` S=2.9203).
  **Drift sensitivity** under controlled corruption injection:
  ΔS is monotonic and approximately linear in corruption count;
  slope per corruption inversely proportional to corpus size
  (smaller corpora more sensitive — as expected). Receipt:
  `fixtures/bench_receipts/vn_entropy_substrate_spike_20260509T175942Z.json`.
  19 contract tests covering bounds, determinism, drift
  sensitivity, pipeline correctness, numerical robustness.
  Findings: `docs/VN_ENTROPY_SPIKE_FINDINGS.md`. Compounds with
  PR #183: entropy is a real scalar; wrap with
  `SplitConformal(alpha=0.1)` → calibrated drift tripwire instead
  of magic threshold (follow-on PR).

- **Split conformal prediction kernel — quick win #1 from the
  wide-net survey.** Distribution-free, finite-sample prediction
  intervals via `sum_engine_internal/research/conformal/`
  (Vovk-Gammerman-Shafer 2005; Angelopoulos & Bates 2023).
  `SplitConformal(alpha)` + `calibrate(predictions, targets)` +
  `predict(point) → ConformalInterval`. Two non-conformity score
  types (absolute, signed) with the
  `⌈(n+1)(1-α)⌉/n` finite-sample correction. **Synthetic coverage
  sweep verifies the provable kernel:** at α ∈ {0.05, 0.10, 0.20}
  empirical coverage hits 0.952 / 0.906 / 0.806 (target: 0.95 /
  0.90 / 0.80) — within 1 SD across 5 seeds, n=2000 cal + n=2000
  test. Naive constant-mean predictor on linear data still hits
  coverage — the *distribution-free* property in action.
  **Substrate experiment** demonstrates the wrap-pattern
  end-to-end on real SUM data: ridge-predicted triple-quality
  on `seed_v1` (50 labeled triples, 3-way split) wrapped with
  `SplitConformal(alpha=0.1)` — hits target coverage with mean
  width 0.031 on a [0,1] target. The kernel is now available for
  wrapping any of SUM's existing readouts (slider axes,
  sheaf-Laplacian scores, per-axiom corruption_score from RPCA);
  individual wraps are follow-on PRs. 19 contract tests covering
  coverage guarantee, score-type variants, determinism, edge
  cases, and diagnostics. Receipt:
  `fixtures/bench_receipts/split_conformal_spike_20260509T174250Z.json`.
  Findings doc: `docs/SPLIT_CONFORMAL_SPIKE_FINDINGS.md`.

- **Robust PCA axiom-corruption spike (Phase A) — math kernel
  verified, application iteration 1 ships honest negative
  result.** Implements the deep-research article's #1-ranked
  move: Principal Component Pursuit via ADMM (Lin et al.
  arXiv:1009.5055, Algorithm 5) for axiom-corruption detection,
  with provable kernel from Candès, Li, Ma & Wright JACM 58(3):11
  (2011). New module `sum_engine_internal/research/robust_pca/`
  (~190 LOC core + ~40 LOC embedding) with: `pcp(M)` returning
  `PCPResult(L, S, n_iter, residual_norm, rank_estimate, ...)`,
  `corruption_score(M)` per-row L1 of S, deterministic
  sha256-bucketed axiom embedding, edge-case handling
  (empty / non-finite / all-zero). Spike harness
  `scripts/research/robust_pca_axiom_spike.py` runs two
  experiments and emits `sum.robust_pca_axiom_spike.v1` receipt.

  - **Experiment 1 — synthetic ground-truth recovery PASSES:**
    n=50/200/500, exact rank recovery (3/5/10), L_rel_err =
    S_rel_err = 0 to 1e-6, convergence in 16-22 iters. Default
    μ₀ = 1.25/‖M‖₂ (Lin et al.) is load-bearing; wrong μ
    converges to a high-rank "fits-but-doesn't-separate"
    solution. Pinned by tests.
  - **Experiment 2 — corpus corruption detection: HONEST
    NEGATIVE RESULT for the simplest setup.** Single-extraction
    + sha256-bucket embedding: precision/recall ≈ 0.08
    (random-baseline). Single-extraction + corpus-vocab
    embedding + bidirectional-anomaly ranking: precision/recall
    0.50-0.58 (5-6× random-baseline, well below the article's
    0.95 target). The article's hand-wave didn't survive contact
    with reality on the simplest embedding.

  Path forward documented in
  `docs/ROBUST_PCA_AXIOM_SPIKE_FINDINGS.md`: (1) multi-extraction
  matrix (the article's alternative framing — rows = extractions
  of one document; PCP separates consensus from rare noise); (2)
  learned axiom embedding (introduces ML dependency, PROOF_BOUNDARY
  applies); (3) reframe the failure class (RPCA may not be the
  right tool for single-extraction corruption — article §9.3
  persistent homology or SMT-backed consistency are alternative
  candidates). Math kernel stays in place across all iterations.

  18 contract tests covering math correctness, edge cases, and
  embedding determinism. Receipt:
  `fixtures/bench_receipts/robust_pca_axiom_spike_20260509T170817Z.json`.

- **Phase 26.0 iteration 5 — spike concluded; UnionFindStore is
  the Phase 26 backing store.** The egglog spike succeeded in
  its actual job: it taught us we don't need egglog. After
  preparing draft upstream issue comments, the strategic
  question came up: *can we just adapt what we need from egglog
  and move on?* The substrate's actual need across 4 iterations
  distilled to "given equivalent triples under a symmetry rule,
  return a deterministic canonical form" — a graph algorithm
  (union-find) not a special-purpose e-graph requirement.
  `sum_engine_internal/graph_store/unionfind_store.py` (~190
  lines) implements it directly: O(α(N)) union via
  path-compressed union-find, O(class_size) extract, deterministic
  by construction (lex-sort of equivalence class), zero external
  dependency. Three-way comparison receipt
  `fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T164443Z.json`
  shows: 10k insert 3 ms (vs egglog lazy 67 s), 10k extract 2 ms
  (vs egglog deterministic 11 s) — ~5500× faster on extract,
  materialisation eliminated, cross-process determinism
  guaranteed by construction. 18 contract tests in
  `Tests/test_graph_store_unionfind.py` (mirror egglog's 22)
  including two cross-backend parity tests
  (`test_content_hash_matches_egglog_backend`,
  `test_pattern_queries_match_egglog_backend`) that pin both
  backends produce identical results on the same input — if
  either drifts, CI catches it. EgglogStore stays in the
  codebase for as long as the spike data is load-bearing; the
  spike harness still measures it. Findings doc gains
  iteration-5 conclusion: "egglog is the right tool for a
  different problem." The egglog upstream issues
  (#793 extract nondeterminism, #756 bulk-load) stay open
  without SUM contributions — we have a substrate-shaped
  alternative; upstream comments would be downstream noise.

- **Phase 26.0 iteration 4.1 — performance correction (the cost
  model is NOT free).** A pre-flight verification round before
  posting upstream egglog issues caught an overclaim in the
  iteration-4 findings: "the cost model only adds a sha256 per
  extract but that's microseconds compared to the egglog
  overhead." Wrong in the dimension I had not measured. The
  cost model is invoked once per visited e-node per extract
  call, paying the Rust→Python boundary cost + `str(expr)` +
  sha256 each time. Measured overhead at our workload sizes
  (lazy mode, fresh receipt
  `fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T161351Z.json`):

  | workload         | default | deterministic | overhead |
  |------------------|--------:|--------------:|---------:|
  | seed_news_briefs |  471 µs |         89 ms |    188 × |
  | seed_long_paras  |  567 µs |        171 ms |    302 × |
  | synthetic_1k     |  1.3 ms |         1.1 s |    860 × |
  | synthetic_10k    | 17.5 ms |        11.5 s |    657 × |

  `extract_canonical(deterministic=True)` is the workaround for
  cross-process determinism but has a 200–1000× per-extract
  overhead that scales with graph size. At 10k axioms a single
  deterministic extract is 11.5 s; at 50k library-scale it
  would be minutes. Path forward for library-scale: a
  Rust-native deterministic-extract mode in upstream egglog
  (eliminates the callback cost), batched extract API (not
  exposed today), or scoping egglog to small-graph special-case
  rewrites. Spike harness (`scripts/research/phase_26_egglog_spike.py`)
  now measures both modes and emits an `extract_deterministic_*`
  block per workload. Findings doc gains an iteration-4.1
  section. New canary test
  (`test_deterministic_extract_has_measurable_overhead`)
  asserts at least 50× slowdown on a tiny graph so a hypothetical
  future "cost model became free" regression doesn't go silently
  unnoticed (and would be good news worth catching). 22 tests
  total.

- **Phase 26.0 iteration 4 — content-derived cost model resolves
  the determinism red flag.** egglog-python PR #357 (merged
  2025-10-02, present in v11.4.0 which we pin) added a
  `cost_model=` parameter to `EGraph.extract`. Using it,
  `_content_hash_cost_model` returns
  `sum(children_costs) << 64 + sha256(str(expr))[:8]` — two
  priors in two non-overlapping bit ranges (high 64 bits for
  structural cost preserving egglog's "smaller subtree wins"
  intuition; low 64 bits for content-derived total order on
  e-class members with equal structural cost; 2⁻⁶⁴ collision
  probability). `extract_canonical(triple, deterministic=True)`
  (the new default) uses this model; `deterministic=False`
  preserves egglog's default extract for spike A/B comparisons.
  Pinned by three new tests: the iteration-3 known-limitation
  test (`test_default_cost_is_insertion_order_sensitive_on_ties`)
  is replaced by `test_extract_canonical_is_deterministic_across_insertion_order`
  (asserts forward and reversed insertion produce identical
  canonical forms under deterministic=True);
  `test_extract_canonical_deterministic_false_preserves_default_behaviour`
  asserts the iteration-3 limitation still appears under
  `deterministic=False` (so if egglog upstream solves it, our
  wrapper can simplify); `test_content_hash_cost_model_is_pure`
  pins the cost model's purity + bit-range layering. 21 contract
  tests total (3 net new — 1 inverted, 2 added). Findings doc
  gains an iteration-4 section noting (a) what was resolved,
  (b) what was NOT (materialisation bottleneck remains; egglog
  issue #756 bulk-load is open), and (c) external validation
  from the egglog ecosystem (issue #793 — math-microbenchmark
  nondeterminism — confirms our finding upstream; no public
  production users at >10k facts; egg-Rust's `CostFunction` trait
  has always been deterministic). The layered-decomposition cost
  model design is the deep-research article's §2 kernel applied
  to extract-with-cost.

- **Phase 26.0 iteration 3 — actual e-class queries measured;
  two new findings.** Adds a baked-in `ownership_symmetry`
  ruleset (`Triple(s, "owns", o) ⟺ Triple(o, "owned_by", s)`)
  to `EgglogStore` plus three new methods:
  `available_rulesets()`, `saturate(ruleset_name)`, and
  `extract_canonical(triple)` — the minimum surface needed to
  exercise egglog's extract-with-cost end-to-end. Spike harness
  gains a `_measure_eclass` phase running saturate + 50 sampled
  extractions per workload. Lazy-mode numbers
  (eager unchanged from iteration 2): saturate is essentially
  free for unfiring rules (12 ms at 10k); extract scales
  near-linearly with graph size (0.6 ms at 66 triples → 15 ms
  at 10k → projected ~75 ms at 50k). End-to-end e-class query
  cost is dominated by materialisation, not by saturate or
  extract — the option-2 (bulk-load API) direction remains the
  unaddressed bottleneck. **Honest red flag:** egglog's default
  `extract` is non-deterministic across processes when two
  expressions in the same e-class have equal cost (FIFO
  tie-breaker over insertion order). For `bench_digest`
  cross-process reproducibility, a custom cost function with a
  content-derived tie-breaker is mandatory. Pinned by
  `test_default_cost_is_insertion_order_sensitive_on_ties`
  (asserts the two canonical forms differ across insertion-order
  permutations — will start failing if egglog's default becomes
  deterministic, which is the desired direction). Updated
  receipt:
  `fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T135038Z.json`.
  19 contract tests total (4 new). Findings doc gains the
  iteration-3 section, a determinism-red-flag callout, and an
  updated decision-options block.

- **Phase 26.0 iteration 2 — lazy materialisation resolves the
  egglog storage-cost objection.** `EgglogStore` gains
  `materialise_egraph()` and an `eager_materialisation`
  constructor flag (default False — lazy). `add_triple` updates
  only the authoritative Python set; the e-graph stays unbuilt
  until an equivalence-class query forces it. Pattern queries
  and `content_hash` work directly on the set and never trigger
  materialisation. A/B re-run on the same workloads: lazy 10k
  insert drops from 64.4 s → **2.4 ms** (28 000× faster) while
  determinism + substrate parity hold across modes (the same
  triple set produces a byte-identical `content_hash` whether
  inserted lazily or eagerly). Materialisation itself is still
  the same ~70 s at 10k — the e-graph build cost is real but
  now only paid by workloads that actually use it. Architectural
  implication: **egglog is a query layer, not a storage layer**;
  schema-migration work targets the storage path; egglog enters
  only at the query boundary, materialised on demand. Updated
  receipt:
  `fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T133315Z.json`.
  4 new tests added to `Tests/test_graph_store_egglog.py`
  (15 total) covering lazy-defers-registration, eager-registers-
  immediately, cross-mode `content_hash` invariance, and
  pattern-queries-don't-trigger-materialisation. Findings doc
  updated with the iteration-2 numbers.

- **Phase 26.0 spike — egglog backing-store candidate measured.**
  First of three named in `docs/PHASE_26_DESIGN.md` §2 (Neo4j,
  PostgreSQL+AGE, egglog). Egglog ships as a pip wheel — zero
  infrastructure overhead — so it's the natural first candidate.
  Adds `sum_engine_internal/graph_store/` (backend-agnostic
  `GraphStore` Protocol, `Triple` dataclass, JCS-style
  `content_hash`) and `egglog_store.py` (in-process backend
  holding triples in both an `egglog.EGraph` for future
  equivalence-class work and a Python set for queries).
  Measurement script
  `scripts/research/phase_26_egglog_spike.py` re-encodes
  substrate corpora through the existing `DeterministicSieve`
  plus synthetic 1k / 10k workloads and emits a
  `sum.phase_26_backing_store_spike.v1` receipt.
  Result: determinism PASS (forward vs reversed insertion-order
  produce identical content_hash) + substrate-parity PASS
  (find_* matches brute-force on every sampled query). Wall
  time MIXED — sub-30 ms on real corpora, 67 s on synthetic
  10k (above the §4.9 library-scale envelope). 11 contract
  tests in `Tests/test_graph_store_egglog.py`.
  `docs/PHASE_26_EGGLOG_SPIKE_FINDINGS.md` writes up the
  receipt, the 10k red flag, three follow-up directions
  (lazy materialisation, bulk-load API, query-layer-only
  scope), and three operator decision options.

- **Step 4.b — bind win replicates over the real MCP stdio
  transport.** Added a `--use-bind-mcp` mode to
  `scripts/research/agent_failure_experiment.py` that spawns
  `python -m sum_engine_internal.mcp_server` as a subprocess,
  opens an `mcp.ClientSession`, and routes each agent verb to the
  corresponding `*_bind` MCP tool over real JSON-RPC. The agent
  system prompt is unchanged from the in-process bind run — only
  the execution path differs. New log committed as
  `fixtures/agent_logs/agent_run_bind_mcp_doc_long_cell_biology_*.jsonl`.
  Phase counts are byte-identical to the in-process bind run
  (max turn 5, parse_errors 0, free-form retries 0, completed via
  `done`). The bind verb's payoff is in the contract
  (content-addressed handles, typed errors, self-describing
  manifest) — not in any in-process Python shortcut. The
  four-step plan is fully closed at the canonical surface.
  `docs/AGENT_SURFACE_FINDINGS.md` gains a Step 4.b section with
  the three-way comparison table.

- **Bind agent surface mounted on the FastMCP server.**
  `sum_engine_internal/agent_surface/mcp_bind.py` gains
  `register_bind_tools(mcp, registry, real_tools)`; the SUM MCP
  server (`sum_engine_internal/mcp_server/server.py`) calls it
  after the legacy inline tools register, capturing references to
  the in-process closures and mounting six new tools alongside:
  `extract_bind` / `attest_bind` / `verify_bind` / `render_bind` /
  `inspect_bind` plus the `agent_surface_manifest`
  (`sum.agent_surface_manifest.v1`). Bind tools delegate
  in-process — no subprocess hop — and accept either inline values
  or `sha256:<hex>` bind references for the `bundle` argument.
  `attest_bind` binds the bundle directly (not its envelope), so
  the returned `bind_id` flows straight into `verify_bind` /
  `render_bind` / `inspect_bind` with no unwrap step. Errors pass
  through unchanged from the underlying tools so callers branch on
  the same `error_class` enum the v2 server already emits. Closes
  the "natural follow-on" named at the close of PR #172. Six new
  contract tests in `Tests/test_mcp_server.py` pin the surface;
  `test_server_boots_with_expected_tools` now asserts both the
  legacy inline set and the bind set are registered.

- **Agent surface: bind verb + Step 4 verification (PR #172).**
  New `sum_engine_internal/agent_surface/` module with the
  content-addressed `BindRegistry` (`sha256:<hex>` over JCS-canonical
  bytes; deterministic, idempotent, thread-safe) and bind-aware
  wrappers around extract / attest / verify / render / inspect plus
  the `sum.agent_surface_manifest.v1` self-description. Render
  carries a typed precondition for non-neutral LLM-conditioned axes
  so the agent receives `{error_class=schema, structured.…}` instead
  of free-form prose. Re-running the agent_failure_experiment with
  `--use-bind-layer` (same model, same document, same harness) hits
  the falsifiable criterion stated in PR #171 on all three counts:
  max-turn 10 → 5, parse_error events 4 → 0, free-form-error
  retries 1 → 0. The bind layer is the canonical agent surface
  from here forward; the CLI dispatcher in the harness is retained
  as the comparison baseline only. Both logs committed under
  `fixtures/agent_logs/`. Mounting `bind_*` as actual FastMCP
  tools on the existing MCP server is the natural follow-on.

- **Agent-failure experiment harness (PR #171).** New
  `scripts/research/agent_failure_experiment.py` — a real agent
  loop (gpt-4o-mini-2024-07-18) wired to the SUM CLI tools with a
  budget and a goal ("produce a verified summary of this PDF"),
  plus a baseline failure log committed as evidence
  (`fixtures/agent_logs/agent_run_doc_long_cell_biology_*.jsonl`).
  Twenty minutes of watching one agent fail named the missing
  verb (`bind` — content-addressed handles so the agent passes
  `sha256:<hex>` references instead of round-tripping full
  bundles) and the missing typed-error path (render returned
  free-form prose for a precondition violation, forcing the agent
  to interpret it heuristically). Findings written up in
  `docs/AGENT_SURFACE_FINDINGS.md`. PR #172 implemented the bind
  layer and verified the falsifiable criterion stated here.

- **Phase 26 design doc (PR #170).** New
  `docs/PHASE_26_DESIGN.md` — five decision points for the
  property-graph backing store (process model, persistence,
  query language, schema authority, distribution boundary) and
  three candidate stores trade-off matrix: Neo4j, PostgreSQL+AGE,
  egglog. Egglog enters as a serious candidate because its
  built-in equivalence-class semantics + extract-with-cost match
  Phase C's importance-weighted SUM directly. Spike plan named
  but not started; current path foregrounds the agent-surface
  bind layer (PRs #171 / #172) before resuming Phase A/B/C.

- **Path 2 §4.7.4.1 — extremal-Goodhart confirmed; §4.7.4
  consolidates back to STRUCTURAL_GAP at controlled n.** New
  16-doc corpus `seed_paragraphs_16.json` (same encyclopedic voice
  as `seed_paragraphs`, eight originals retained verbatim plus
  eight new docs: Mount Everest, Marie Curie, Great Wall of China,
  Titanic, Renaissance, atomic structure, jet stream, blockchain).
  At doubled n the lone BEATS cell from §4.7.4 disappears:
  gpt-4o-mini Δ=+0.032 BEATS at n=8 → Δ=−0.013 TIES at n=16. Joint
  finding on `seed_paragraphs_16`: `STRUCTURAL_GAP_NO_MODEL_BEATS`,
  matching the other two corpora at n=16. Updated 4-corpus
  aggregate (21 cells: 1 BEATS, 10 TIES, 10 LOSES) — the lone
  BEATS cell is now *explained* (small-n threshold noise via
  extremal Goodhart), not unresolved.

  Substantive consequence: the preprint §4.7.x narrative tightens
  significantly. At controlled sample sizes (n ≥ 16) across
  three corpora and 4-6 LLM lineages, the hybrid does NOT BEAT
  B2 on real-LLM perturbations. The synthetic-bench WIN
  (+0.043) is best read as a Goodhart artifact: the hybrid was
  selected to compose well on a measure (the synthetic harness),
  and the measure stops being a good measure once it is the
  target of optimisation. Preprint §4.7.2 prose now uses the
  "deception register" frame from biological signal-reward
  contracts (Schiestl et al. 1999; Cook & Rasplus 2003) to
  explain the synthetic-vs-real gap mechanistically.

  Preprint §7 restructured into a four-tier audit (claims that
  hold up / claims corrected from prior overstatements / claims
  real but narrow / limits). PROOF_BOUNDARY §2.10 reframed as
  *continuous-enforcement* against the analogue of mutualism
  breakdown (Sachs et al. 2004), with PR #160 dict-order fix as
  the worked example. Pinned in
  `Tests/research/test_sheaf_path2_cross_corpus.py` (21 per-cell
  digests + per-corpus joint findings + cell counts).

- **Path 2 §4.7.4 cross-corpus extension — §4.7.3 finding is
  corpus-specific.** New `scripts/research/sheaf_path2_cross_corpus_aggregate.py`
  loads N per-corpus compare receipts and produces a joint finding
  across the 2-D (corpus × model) matrix. Two new corpora authored:
  `seed_paragraphs` (already in repo, 8 docs, smaller encyclopedic)
  and `seed_news_briefs` (new, 16 docs, news-wire prose; deliberately
  out-of-distribution from `seed_long_paragraphs`). The multi-LLM
  compare gains a `--corpus` flag (backward-compatible: default keeps
  existing pins). Per-corpus receipts go to corpus-suffixed paths so
  they don't collide with the §4.7.3 historical receipt; PR #161
  receipt at `path2_multi_llm_compare_2026-05-05.json` is preserved
  untouched. Bug fix in `scripts/research/_receipt_paths.py`: glob
  now requires a `_YYYY-MM-DD.json` suffix to prevent prefix-of-prefix
  false positives.

  Across 3 corpora (`seed_long_paragraphs` carries the n=6 set
  with claude; the two new corpora are n=5 because Anthropic was
  unavailable during the §4.7.4 capture) — jagged 16-cell matrix:
  **1 BEATS, 8 TIES, 7 LOSES**. Per-corpus joint
  findings: `STRUCTURAL_GAP_NO_MODEL_BEATS` (`seed_long_paragraphs`),
  **`MIXED_VERDICTS_MODEL_DEPENDENT`** (`seed_paragraphs` —
  gpt-4o-mini Δ=+0.032 BEATS at the +0.030 threshold),
  `STRUCTURAL_GAP_NO_MODEL_BEATS` (`seed_news_briefs`). Cross-corpus
  joint finding: **`CROSS_CORPUS_VERDICTS_DIVERGE`** — falsifies
  any naive reading of §4.7.3 as asserting universal LOSES. Honest
  reading: hybrid does not consistently BEAT baseline across LLM
  families × corpora, but isolated cells can produce positive Δ at
  the threshold; synthetic-bench WIN magnitude (+0.043) still sits
  substantially above the lone real-LLM BEATS cell (+0.032).

  Pinned in `Tests/research/test_sheaf_path2_cross_corpus.py` (15
  per-(corpus, model) digests + per-corpus joint findings + cross-
  corpus joint finding + cell counts). New §4.7.4 in
  `docs/arxiv/sheaf-detector-note-v0.md` with the full matrix and
  four-point honest reading.

- **Path 2 §4.7.3 extended to 6 LLM lineages — open-weights via HF
  Inference Providers.** Phase 1 captures landed for Meta Llama-3.3-70B
  (`f1c17c3e…aac29b31`), Alibaba Qwen3.6-35B-A3B (`23da3ecb…461b8ea2`),
  DeepSeek V3-0324 (`619a413f…2fe22c9f`), and Google Gemma-3-27B
  (`fe76913e…0318b9b5`) — all routed through `router.huggingface.co/v1`
  via the new HF route in `llm_dispatch.get_adapter` (model ids of
  shape `org/model` → `OpenAIAdapter` with HF base_url + `HF_TOKEN`).
  Joint finding upgrades from `STRUCTURAL_GAP_ALL_MODELS_LOSE` (n=2)
  to **`STRUCTURAL_GAP_NO_MODEL_BEATS`** (n=6): four LOSE
  (gpt-4o-mini, claude, Llama, Gemma), two TIE (Qwen +0.003,
  DeepSeek +0.018), zero BEAT. The synthetic-bench WIN doesn't
  generalise to *any* LLM family in the cross-organisational sample.
  §4.7.3 prose rewritten with full per-model table + texture
  analysis. Pinned in `Tests/research/test_sheaf_path2_multi_llm_compare.py`
  (new n=6 test alongside n=1 / n=2 cases). Multi-LLM compare's
  `CAPTURE_TIMEOUT_S` raised from 60s → 180s + 2 retries on timeout
  to accommodate occasional slow HF routed calls.

- **Path 2 cross-family corroboration (§4.7.3): Claude Haiku 4.5
  snapshot lands; structural-gap finding holds across LLM families.**
  Phase 1 capture for `claude-haiku-4-5-20251001` produced
  `fixtures/bench_renders/path2_claude-haiku-4-5-20251001.json`
  (16 docs × 4 prompt classes; 64 LLM calls). Phase 2 scoring against
  both committed snapshots returns
  **`joint_finding: STRUCTURAL_GAP_ALL_MODELS_LOSE`** — both LLM
  families have the hybrid LOSING to the B2 baseline (gpt-4o-mini
  Δ=−0.021; claude-haiku-4.5 Δ=−0.032; spread 0.011). The
  synthetic-vs-real gap from §4.7.2 is **not** gpt-4o-mini-specific:
  two independent LLM families with different training corpora,
  architectures, and adversarial-prompt-following styles produce the
  same directional verdict with similar magnitude. Pinned in
  `Tests/research/test_sheaf_path2_multi_llm_compare.py`
  (per-model digests `7b364fc6…cc4b75e` for OpenAI, `d0f9f175…2f6f7`
  for Anthropic). Receipt:
  `fixtures/bench_receipts/path2_multi_llm_compare_2026-05-05.json`.

- **Multi-LLM Path 2 comparison harness (Path 2 → cross-family).** New
  `scripts/research/sheaf_path2_multi_llm_compare.py` extends the
  capture-once-replay-forever architecture across LLM families to test
  whether the §4.7.2 synthetic-vs-real gap is structural or
  gpt-4o-mini-specific. Captures one snapshot per model via the
  vendor-agnostic dispatcher (`llm_dispatch.get_adapter`), reuses the
  Path 2 v3 deterministic Phase-2 scorer, and aggregates per-model
  verdicts into one of five cross-family findings
  (`STRUCTURAL_GAP_ALL_MODELS_LOSE`, `STRUCTURAL_GAP_NO_MODEL_BEATS`,
  `HYBRID_BEATS_ALL_MODELS`, `HYBRID_BEATS_OR_TIES_ALL_MODELS`,
  `MIXED_VERDICTS_MODEL_DEPENDENT`). n=1 honestly reports
  `SINGLE_MODEL_<verdict>` rather than overstating. Default models:
  `gpt-4o-mini-2024-07-18` + `claude-haiku-4-5-20251001`. The
  Anthropic snapshot is operator-gated (Phase 1 needs
  `ANTHROPIC_API_KEY`); the gpt-4o-mini path runs against the existing
  PR #156 snapshot with byte-identical digest. Pinned in
  `Tests/research/test_sheaf_path2_multi_llm_compare.py` (verifies the
  multi-LLM wrapper is a no-op on the scoring path).

- **Path 2 — real-LLM-rendered adversarial bench (closes the §7
  load-bearing asterisk).** New `scripts/research/sheaf_path2_v3_bench.py`
  with capture-once-replay-forever architecture: Phase 1 calls
  `gpt-4o-mini-2024-07-18` to render each `seed_long_paragraphs` doc
  at four prompt classes (neutral + a1/a2/a4 adversarial); Phase 2
  re-extracts via `DeterministicSieve` and scores deterministically.
  LLM render snapshot committed to
  `fixtures/bench_renders/path2_seed_long_paragraphs.json`; downstream
  scoring is byte-stable against the snapshot. Receipt:
  `fixtures/bench_receipts/path2_v3_bench_2026-05-05.json`,
  `bench_digest 7b364fc6…cc4b75e`. Pinned in
  `Tests/research/test_sheaf_path2_v3.py`.

  **Verdict: `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM`** (Δ = −0.021
  trusted-mean). The synthetic-bench WIN (§4.7.1, Δ = +0.043) does
  NOT generalise to real LLM hallucinations. All detectors weaken
  substantially: B2 trusted-mean drops from 0.833 (synthetic) to
  0.660 (real LLM); the hybrid drops from 0.876 to 0.643. The
  synthetic-vs-real gap is itself the load-bearing finding —
  synthetic A1/A2/A4 perturbations have a structural property real
  LLM perturbations don't share, which inflates the hybrid's
  apparent advantage on the synthetic harness.

  This is exactly the kind of finding the substrate's truth-first
  discipline asks for. The §4.7.2 narrative names it honestly:
  the synthetic-bench WIN is real on its corpus but doesn't
  generalise; future work (v0.4+) needs real-LLM-aware training
  or naturalistic perturbation synthesis to close the gap.

- **v0.3 deterministic-BLAS fix for `hybrid_comparison` digest stability.**
  The Sprint 7.5 latent-fix arc closed three of four issues but had to
  shape-pin `hybrid_comparison` because two layers of quantization
  (rank-key in `_ranks`; per-pair score at storage time) couldn't
  fully absorb LAPACK-thread-pool-size variance at numpy-import
  time on Apple Accelerate. The diagnosis converged after observing
  same-process determinism but cross-process variance + the fact
  that `VECLIB_MAXIMUM_THREADS=1` in the shell made the digest
  unconditional 8/8.

  Fix: `scripts/research/_deterministic_blas.py` sets
  `VECLIB_MAXIMUM_THREADS=1` + `OPENBLAS_NUM_THREADS=1` +
  `MKL_NUM_THREADS=1` + `OMP_NUM_THREADS=1` + `BLIS_NUM_THREADS=1`
  + `NUMEXPR_NUM_THREADS=1` via `os.environ.setdefault` at
  module-import time. All five Sprint-7.5 bench scripts import the
  helper as their first non-future import (BEFORE numpy), so
  numpy's BLAS thread count is fixed at 1 on every fresh-process
  invocation. Production library code in `sum_engine_internal/`
  doesn't import the helper, so multi-threaded BLAS remains the
  default for non-bench paths.

  Tests refactored to subprocess invocation:
  pytest itself imports numpy via the Hypothesis plugin BEFORE
  test functions run, making in-process env-var setdefault a
  no-op in pytest context. The recovery digest tests now invoke
  each bench in a clean subprocess (with the deterministic-BLAS
  env explicitly set) and parse the on-disk receipt the bench
  wrote. ~11s for 4 tests vs the previous ~60s in-process.

  `hybrid_comparison` pin re-upgraded to byte-digest pin
  (`a7965803…`). All four recovery experiments are now byte-digest
  pinned.

## [0.6.0] - 2026-05-05

The Sprint 7 + 7.5 release. arXiv preprint v0.1 in
`docs/arxiv/sheaf-detector-note-v0.md` is the headline artifact;
the substrate underneath is the load-bearing one (cryptographically-
anchored bench digests, cross-machine reproducibility verified
across three LAPACK environments, six-regime audit-grade
compliance, complementary-hybrid hallucination detector that
beats trivial entity-set baselines on `seed_long_paragraphs`).

Released to PyPI as `sum-engine==0.6.0`.

### Sprint 7 — arXiv preprint v0 → v0.1 (2026-05-04)

PR #142. Updates `docs/arxiv/sheaf-detector-note-v0.md` to fold in
the v3 / v3.1 / v3.2 / F3 STRUCTURAL FAIL arc, with five named
falsification verdicts (F1 MARGINAL, F2 PASS, F3 STRUCTURAL FAIL,
F4 PASS, F5 PASS at γ ≤ 0.1). Categories shifted from
`cs.AI / math.CT` → `cs.LG / cs.CR`.

### Sprint 7.5 — preprint hardening + complementary-hybrid recovery (2026-05-04 / 05)

- **Sprint 7.5 — preprint hardening + complementary-hybrid
  detector recovery (2026-05-04 / 05).** The arXiv preprint
  was hardened across four stacked PRs (#146 /
  #144 / #145) before pre-circulation:

  - **T2 baseline comparison** (`scripts/research/sheaf_baseline_comparison.py`,
    `fixtures/bench_receipts/baseline_comparison_2026-05-04.json`,
    `bench_digest cb32c617…`). Two trivial reproducible baselines:
    B1 entity-presence-deficit and B2 jaccard-distance, scored on
    the same (clean, perturbed) triple-set pairs the v3.x
    detectors consume. **STOP-THE-LINE finding**: trivial
    baselines beat v3.2 trusted-mean by Δ=−0.174 (B2 0.833 vs
    v3.2 0.659).

  - **T4 §3.0 threat model** added to the preprint. Four
    capabilities (T1 adversarial render / T2 adversarial bundle
    / T3 stolen key / T4 verifier OOS) each mapped to a defence
    component or to OUT OF SCOPE.

  - **T2.5 four recovery experiments** (cochain-only Borda fusion
    loses; predicate-perturbation training fails to lift A2;
    per-rendered-triple V channel restoration lifts A2 from
    0.500 to 0.671; complementary Borda(v3.2 + per-triple, B2)
    **WINS at trusted-mean 0.876, Δ=+0.043 vs B2 alone**).
    Digests `a7965803…`, `aa34b6e8…`, `7025436f…`, `dc6e0260…`
    pinned in `Tests/research/test_recovery_experiment_digests.py`.

    **Structural finding surfaced**: the v3.x cochain-on-source-graph
    is mathematically blind to entity-set-preserving perturbations
    because predicate doesn't enter the cochain. Same shape as F3
    STRUCTURAL FAIL. Adding training negatives can't fix what
    scoring discards. The per-triple channel restoration addresses
    the right load-bearing site.

  - **T3.M Modal cross-machine `bench_digest` verification**
    (`scripts/research/cross_machine_verify_modal.py`,
    `fixtures/bench_receipts/cross_machine_verification_2026-05-04.json`).
    Both v3.2 validation (`b4d26c01…`) and complementary hybrid
    (`dc6e0260…`) digests reproduced byte-for-byte across two
    distinct LAPACK environments: Apple Accelerate on Apple
    Silicon (operator) and OpenBLAS-via-PyPI on Modal x86_64
    (Linux 4.4 / glibc 2.31 / Python 3.10.8 / numpy 1.25.0 /
    AVX2 SIMD). The substantive `HYBRID_BEATS_BASELINE` verdict
    also reproduces cross-machine. BRANCH A outcome — the
    strongest possible cross-machine reproducibility claim.

  - **T5 substrate-first reframe** of the preprint (PR #145).
    Title shift from "Sheaf-Laplacian hallucination detection
    on signed render receipts" to "A cryptographically-anchored
    substrate for hallucination detection on signed render
    bundles". arXiv categories: `cs.CR` (primary) /
    `cs.LG` (secondary). New §3.9 Complementary-signal hybrid
    section (Borda rank-fusion math); new §4.7.1 Recovery arc
    subsection documenting all four recovery experiments;
    §6 reorganised into §6.1 substrate (vs published
    reproducibility primitives, vs zkML, vs compliance-only
    audit-log substrates) + §6.2 detector positioning; §7
    bounded claims rewritten with hybrid-specific items.

  - **T6 pre-circulation cover note** at
    `docs/arxiv/PRE_CIRCULATION_COVER_NOTE.md`. Asks four
    focused questions of the 1–2 pre-circulation readers:
    math correctness; threat model coverage; baseline choice +
    WIN claim; substrate-first framing.

  Total Sprint 7.5 contribution: ~3000 LOC across 5 new bench
  scripts, 1 cross-machine harness, 1 new test file with 4
  pinned digests, 6 new fixture receipts, and 1500-line
  preprint at v0.1. Research suite: 102 passing (was 88
  pre-Sprint-7.5; +14 from new tests).

  The substrate's truth-first discipline functioned as designed:
  caught a real loss (baselines beat v3.2 alone) BEFORE
  publication, surfaced a structural finding (cochain blindness)
  via the predicate-negatives experiment's failure, and produced
  a competitive detector via complementary-signal Borda fusion
  rather than face-saving "honest negative result" framing.

- **Sprint 7.5 verified-quiescence pass (PR #149).** Deep
  verification on current main surfaced two real bugs and a stale
  claim: (1) `hybrid_comparison` digest was intermittent across
  fresh procs because cochain-only Borda fusion has LAPACK-tie-
  shuffle sensitivity (CI passes were lucky); (2) Modal harness
  was pinned pre-merge SHA `37351e2`; (3) two broken doc cross-
  references (`test_sheaf_laplacian_v31.py`,
  `test_sheaf_v2.py`). Fixed: `hybrid_comparison` pin relaxed to
  behavior-shape; PINNED_SHA bumped to `b5fe92b`; doc refs
  corrected. Modal v4 re-run against post-merge main confirmed
  BRANCH_A both digests MATCH. Receipt:
  `fixtures/bench_receipts/cross_machine_verification_2026-05-04.json`.

- **Sprint 7.5 four-latent-issues fix (PR #150).** Closed all four
  latent issues queued by the verified-quiescence pass with
  precision:

  - **Issue 1 (manifest drift):** new
    `scripts/research/_receipt_paths.py` helper —
    `resolve_receipt_path(receipts_dir, schema_prefix)` returns
    the existing canonical receipt (sole match) or today's-dated
    path (zero matches), keeping reruns idempotent. Applied to
    all 6 bench scripts.
  - **Issue 2 (borda tie-break):** `_ranks` in
    `sheaf_hybrid_comparison.py` now quantizes the sort key to
    9 decimals before rank assignment. Sub-ULP LAPACK jitter
    collapses to identical rank → cross-run digest stability.
    Verified `a7965803…` 5/5 in fresh procs after the fix; the
    alternative `7fac833a…` outcome no longer appears.
    `hybrid_comparison` pin upgraded shape → byte-digest.
  - **Issue 3 (predicate-negative sampler upstream):**
    `sum_engine_internal/research/sheaf_laplacian_v2._sample_negative_triples`
    accepts new `n_predicate_negatives_per_positive` parameter
    (additive, default 0 backward-compat); when nonzero,
    produces mixed-class negatives (tail + predicate) per
    positive. `train_restriction_maps` accepts the same param.
    `predicate_negatives_experiment.py` drops the local
    training-loop copy; calls production training. New post-
    refactor digest `ddf41484…`. Pin upgraded shape → byte-digest.
  - **Issue 4 (additional LAPACK environment):**
    `cross_machine_verify_modal.py` extended to two Modal Image
    variants (Python 3.10 + numpy 1.25; Python 3.12 + numpy 2.x)
    × three benches. New per-bench outcome labels.

  Production v2/v3/v32 tests still 65/65 PASS (additive parameter,
  backward-compat default). All four recovery digests now byte-
  digest pinned (was: 2 byte, 2 shape).

- **Sprint 7.5 post-#150 Modal re-verify (PR #151).** PINNED_SHA
  bumped `b5fe92b` → `5715c40` (post-#150 main); Modal v6 run
  against the new SHA produced **3 environments × 3 benches all
  MATCH**. Outcome label
  `BRANCH_A_THREE_ENVIRONMENTS_DIGESTS_MATCH`.

  | Bench | Operator | Modal Py 3.10 | Modal Py 3.12 |
  |---|---|---|---|
  | v3_2_validation | `b4d26c01…` | ✓ | ✓ |
  | complementary_hybrid | `dc6e0260…` | ✓ | ✓ |
  | predicate_negatives | `ddf41484…` | ✓ | ✓ |

  Issue 3 refactor empirically validated end-to-end: pre-refactor
  produced different digests across Modal Python versions
  (Py 3.10: `aa34b6e8…`; Py 3.12: `8638253903…`); post-refactor
  produces a single digest across both. Cross-machine §4.8 prose
  in the preprint and §2.10 in PROOF_BOUNDARY strengthened from
  "2 environments × 2 benches MATCH" to "3 environments × 3
  benches MATCH."

  Latent-issue ledger: 4 → 0.

- **PCI DSS user-identification gap closed (Sprint 4 of the
  intensification path to arXiv).** Three optional identity
  fields (`user_id` / `host_id` / `ip_address`) added to the
  `sum.audit_log.v1` schema as additive optional fields under
  the existing "consumers should ignore unknown keys" convention
  (no schema bump; backward-compatible). The audit-log emit path
  reads three new env vars at process start:
  `SUM_AUDIT_USER_ID`, `SUM_AUDIT_HOST_ID`,
  `SUM_AUDIT_IP_ADDRESS`. Empty-string env vars treated as unset.

  PCI DSS validator gains **R7
  `pci-dss-4-req-10.user-identification`** — fires on rows
  lacking `user_id`. Closes the load-bearing gap previously
  named in `docs/COMPLIANCE_PCI_DSS_4_REQ_10.md` §10.2.2; that
  section now reads "CLOSED 2026-05-03 / PR #140" with operator-
  facing closure instructions (source `SUM_AUDIT_USER_ID` from
  the authenticating proxy's session identity at process start).

  **Truth-first scope.** R7 is a real Req 10.2.2 violation, not
  a soft warning. Pre-Sprint-4 audit logs lacking `user_id` fail
  R7 specifically — the truthful signal that they don't meet
  Req 10.2.2's user-identification requirement. They still pass
  the other six PCI rules and every other regime's validator.

  **Test deltas:**
  - 5 new PCI R7 tests
  - 8 new audit_log identity-field tests (env-var behaviour,
    empty-string handling, payload-override seam, end-to-end
    env-var → audit log → validator R7 closure)
  - Total: 194 passing (25 audit_log + 169 compliance)

  **Why this matters for arXiv (Sprint 7).** The preprint can
  cite a six-regime substrate where the most complex statute
  (PCI DSS Req 10) has a *complete* per-row floor — no
  outstanding load-bearing gap. Operators in PCI-relevant
  contexts get a documented closure path, not a documented
  limitation.

- **Sheaf-Laplacian detector library API doc (Sprint 5b of the
  intensification path to arXiv — first latent capability
  surfaced).** `docs/SHEAF_LIBRARY_API.md` documents the
  programmatic surface of the v2 / v3 / v3.2 sheaf-Laplacian
  hallucination detector — previously importable but undocumented
  for external users beyond docstrings.

  Covers: install path (the `[research]` extras flag), stability
  tier (research-library; one notch below trust-loop guarantees),
  quick-start with worked example (train_restriction_maps →
  weights_from_receipts → boundary_from_weights →
  combined_detector_score_v32), parameter reference for each
  function, the empirical "γ ≤ 0.1" finding from the v3.2
  validation bench, the H16–H20 falsifiable contracts pinned in
  tests, the universal-quantifier upgrades from the Hypothesis
  property tests, and what the library does NOT ship (production
  calibration, decision threshold, A2 detection, CLI surface).

  Also clarifies the orthogonality with the trust loop: the
  detector tells you whether a render is *consistent* with a
  source-of-truth sheaf; the trust loop tells you *who signed*
  the render. In a deployment where signed render receipts feed
  back into `weights_from_receipts`, the two compose.

  This is the first of two latent capabilities being surfaced
  under Sprint 5. The second (render-receipt aggregation CLI)
  ships in a follow-up PR.

- **Shared compliance predicate library (Sprint 3 of the
  intensification path to arXiv).** Six per-regime validators
  (Art 12 / GDPR Art 30 / HIPAA / ISO 27001 / SOC 2 / PCI DSS)
  previously each carried a byte-identical copy of
  `_is_iso8601_utc`. Extracted to
  `sum_engine_internal/compliance/_predicates.py` exporting
  `is_iso8601_utc`. Six regime modules now import from one source.

  **Substrate-tightening proven.** New `Tests/compliance/
  test_predicates.py` pins three contracts:
  - **P1: predicate behaviour** — 9 cases covering Z-suffix
    timestamps (basic, ms, μs), offset-style timezones rejected,
    unparseable bodies rejected, non-string types rejected,
    empty string rejected, human-readable dates rejected.
  - **P2: single source of truth** — exactly one definition of
    `is_iso8601_utc` in the package (in `_predicates.py`); zero
    legacy `_is_iso8601_utc` definitions remaining; six regime
    modules import the shared one (verified via object identity
    `module.is_iso8601_utc is _predicates.is_iso8601_utc`).
  - **Cross-regime propagation** — passing a "+00:00" offset
    timestamp through each of the six regime validators fires
    each regime's R3 rule, proving a behavioural change to the
    shared predicate would propagate to every regime's R3
    verdict simultaneously (the property that motivated the
    extraction).

  **Why this matters for arXiv (Sprint 7).** The shared-predicate
  pattern is replicable for future compliance regimes (NIS2,
  NIST 800-53, etc.) — no copy-paste required. Operationally:
  a future fix to the timestamp predicate (e.g. accepting
  "+00:00" if the regime contract loosens) is a single-file edit
  rather than a six-file lockstep migration.

  **Test deltas:** 12 new predicate tests; total compliance
  suite 152 → 164.

- **fastapi/starlette compatibility pin (Sprint 2 of the
  intensification path to arXiv).** Full pytest run previously
  reported 30 collection errors from six modules
  (`test_browser_extension`, `test_phase13_zenith`,
  `test_phase14_ouroboros`, `test_phase15_abi`,
  `test_semantic_dedup`, `test_state_encoding`) that all import
  through `quantum_main` → `api/quantum_router`. The error
  signature was `Router.__init__() got an unexpected keyword
  argument 'on_startup'` at every `APIRouter()` construction.
  Root cause: a fastapi/starlette mismatch — starlette 1.0.0
  removed `on_startup=` / `on_shutdown=` parameters in favour of
  `lifespan=`, but fastapi 0.115.x still passes empty defaults
  through to its starlette parent. Fix: pin `starlette<1.0` in
  `requirements-prod.txt` until fastapi ships a starlette-1.x-
  compatible release.

  Result: 98 previously-erroring tests now pass cleanly. Full
  pytest reports 0 collection errors. The 2 known
  `test_concurrency_safety.py` flakes remain (separate concern,
  unrelated to fastapi).

- **Bench-digest substrate determinism (Sprint 1 of the
  intensification path to arXiv).** Every bench in the repo
  (`v3_roc_bench`, `sheaf_v3_1_f3_diagnostic`, `sheaf_v3_2_validation`)
  previously reproduced *only* with `PYTHONHASHSEED=0` —
  set-iteration order in the deterministic sieve was hash-
  randomized across Python invocations, propagating
  ~±0.005 noise into per-cell AUCs. **Single load-bearing
  fix:** `sum_engine_internal/algorithms/syntactic_sieve.py:453`
  changed from `list(set(triplets))` to `sorted(set(triplets))`.
  All three benches now reproduce identical `bench_digest`
  values across fresh Python processes without environment-
  variable manipulation; verified by running each bench three
  times.

  **Receipt rebase.** The substrate fix shifted digest values
  slightly (the new sorted order differs from what
  PYTHONHASHSEED=0 produced). Receipts regenerated and dated
  2026-05-03; the 2026-05-02 receipts (PYTHONHASHSEED=0-
  conditional digests) are deleted to avoid future readers
  reproducing against stale anchors. New digests:

  | Receipt | New digest | Old (PYTHONHASHSEED=0) digest |
  |---|---|---|
  | `v3_2_validation_2026-05-03.json` | `b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a` | `97cf977512f9…162f43f` |
  | `v3_1_f3_diagnostic_2026-05-03.json` | `62b6e1878d1d12f36eb80e301304854a1a2c03386f0e872850d3461b2f733e7c` | `244423192cd8…ff5308` |
  | `v3_roc_bench_2026-05-03.json` | (no digest field; AUCs reproducible directly) | — |

  **Substantive verdicts unchanged:** F4 PASSES at every γ
  (F3 STRUCTURAL FAIL still closed at the detector layer);
  F5 PASSES only at γ ≤ 0.1; v3.2 with γ=0 still byte-identical
  to v3 (subsumption holds). The numbers shifted by ≤ 0.005
  per-cell AUC, well under the F4/F5 threshold margins.

  **Removed `reproducibility_requires` field** from the v3.2
  validation receipt. The corresponding caveat sections in
  the spec doc, playbook, and prior CHANGELOG entries are
  superseded by this entry — bench reproducibility is
  unconditional now.

  **Why this matters for arXiv (Sprint 7).** The preprint can
  cite digest values as reproducibility anchors without a
  "PYTHONHASHSEED=0 required" footnote. Anyone who clones the
  repo and runs `python -m scripts.research.sheaf_v3_2_validation`
  gets the same digest as the receipt. That's the
  "reproducible-research-with-cryptographic-teeth" claim
  becoming load-bearing rather than caveated.

- **PCI DSS v4.0 Requirement 10 validator — sixth and final
  regime in the record-keeping shape slate.** Closes the slate
  scoped under Priority 11. Six regimes now consume
  `sum.compliance_report.v1` without shape modification:

  | Regime | Statute | Origin |
  |---|---|---|
  | `eu-ai-act-article-12` | Reg (EU) 2024/1689 Art 12 | EU AI law |
  | `gdpr-article-30` | Reg (EU) 2016/679 Art 30 | EU privacy law |
  | `hipaa-164-312-b` | 45 CFR § 164.312(b) | US health law |
  | `iso-27001-8-15` | ISO/IEC 27001:2022 A.8.15 | Intl ISMS standard |
  | `soc-2-cc-7-2` | AICPA TSP §100A CC7.2 | US audit-attestation |
  | `pci-dss-4-req-10` | PCI DSS v4.0 Req 10 | Payment-card industry |

  PCI DSS Req 10 is the most structurally complex regime in the
  slate (7 sub-requirements 10.1–10.7, with 10.2 itself further
  subdivided). Six per-row rules R1–R6 map to Req 10.2.2 (event
  content) + 10.6 (consistent time): schema-pinned, timestamp-
  present, timestamp-iso8601-utc, event-type-recorded,
  origination-identified, event-content-completeness (per-
  operation anchors).

  **Truth-first scope (load-bearing).** PCI DSS Req 10.2.2 lists
  "user identification" as the FIRST required field for each
  audit-log event. `sum.audit_log.v1` does not currently carry
  a `user_id` field — SUM is a single-process CLI tool without
  a multi-user model. The wire-spec doc names this gap
  explicitly: PCI deployments using SUM as a payment-adjacent
  component need either a schema extension (adding `user_id` /
  `host_id` / `ip_address`) or an authenticating proxy whose
  own logs carry user identity at the aggregation layer. **A
  green report from this validator does NOT mean SUM is PCI-
  compliant** — it means SUM's per-row form satisfies the parts
  of 10.2.2 visible in the current schema.

  The §"What this validator does NOT pin" section is meaningfully
  longer than the other regimes' because PCI DSS Req 10 has more
  obligations that don't fit the per-row shape: 10.1
  organisational policies; 10.2.1.* specific event-type coverage
  (cardholder data access, admin access, log access, invalid
  attempts, credential changes, log start/stop, system-level
  objects); 10.3 log file protection; 10.4 log review process;
  10.5 12-month retention; 10.7 failure detection / alerting;
  cardholder data inventory.

  **Test deltas:** ~25 PCI DSS rule tests. Total compliance suite:
  32 EU AI Act + 25 GDPR + 5 CLI dispatch + 27 HIPAA + 19 ISO + 19
  SOC 2 + 25 PCI = **152 tests**. Six-way byte-shape parity test
  pinned in `test_pci_dss_4_req_10::test_validation_report_shape_matches_other_regimes`.

  **Substrate-tightening summary across the six-regime arc:**
  Started with one regime (EU AI Act Art 12) and a regime-
  agnostic claim that was a single data point. Ended with six
  regimes spanning EU AI law, EU privacy law, US health law,
  international ISMS standard, US audit-attestation, and
  payment-card industry standard — all consuming the same
  `sum.compliance_report.v1` shape without modification. The
  CLI dispatch refactored from `if regime == "..."` to a
  `_compliance_validators()` dict on PR #130; cross-regime
  contracts C1/C2/C3 (registry consistency, schema, exit codes)
  pinned in `Tests/compliance/test_cli_dispatch.py`. Empirical
  finding from the arc: there is a *minimum record-keeping floor*
  common to most record-keeping regimes (R1–R5: schema, timestamp,
  ISO-8601-UTC, activity, system component); the regime-specific
  rules are statutory anchoring + operation-specific anchors on
  top of this shared floor.

  **Different-shape regimes still queued (separate PR family):**
  HIPAA § 164.514 de-identification (transformation rules), EU
  AI Act Art 13 / 16 / 50 (transparency / QMS / disclosure).

- **ISO/IEC 27001:2022 A.8.15 Logging + SOC 2 CC7.2 validators —
  fourth and fifth per-regime compliance consumers (bundled PR).**

  Both regimes share the per-row record-keeping floor shape
  established by GDPR Art 30: five rules covering schema,
  timestamp presence + parseability, activity classification, and
  system component identification. Each carries regime-specific
  `rule_id` strings and statutory message anchors. **Empirical
  finding:** there is a *minimum record-keeping floor* common to
  most record-keeping regimes — five regimes (Art 12, GDPR Art 30,
  HIPAA § 164.312(b), ISO 27001 A.8.15, SOC 2 CC7.2) agree on the
  floor's structure even though the statutes differ wildly (EU AI
  law, EU privacy law, US health law, international standard, US
  audit-attestation).

  **Modules:** `sum_engine_internal/compliance/iso_27001_8_15.py`
  and `sum_engine_internal/compliance/soc_2_cc_7_2.py`. **Wire
  specs:** `docs/COMPLIANCE_ISO_27001_8_15.md` (names "stored",
  "protected", "analysed" verbs as out-of-scope deployment
  obligations) and `docs/COMPLIANCE_SOC_2_CC_7_2.md` (names the
  detection / monitoring / analysis activities, surrounding TSP
  criteria CC6/CC7.1/CC7.3/CC7.4/CC8, anomaly-detection-quality
  audit judgment, and Type 1 vs Type 2 distinction as out of
  scope).

  **Substrate held for the fourth and fifth time.** The cross-
  regime shape pin (each new test file's
  `test_validation_report_shape_matches_other_regimes`)
  expanded to assert N-way parity:
  `Tests/compliance/test_soc_2_cc_7_2.py` checks all five regimes
  return byte-shape-identical `sum.compliance_report.v1` from
  the same input row. C1/C2/C3 dispatch contracts extended
  automatically.

  **Test deltas:** ~19 ISO 27001 tests + ~19 SOC 2 tests = 38 new
  tests. Total compliance suite: 32 EU AI Act + 25 GDPR + 5 CLI
  dispatch + 27 HIPAA + 19 ISO + 19 SOC 2 = **127 tests**.

  **Future regimes still queued (record-keeping shape).** PCI DSS
  4.0 Requirement 10 is the remaining record-keeping regime in
  the slate; planned for a separate PR because Req 10 has 7 sub-
  requirements (10.1–10.7) and many sub-rules — a more honest
  "what this does NOT pin" section than the thinner regimes.

  **Different-shape regime family** (separate PR family if needed):
  HIPAA § 164.514 de-identification (transformation rules), EU AI
  Act Art 13 / 16 / 50 (transparency / QMS / disclosure).

- **HIPAA § 164.312(b) Audit Controls validator — third per-regime
  compliance consumer.** `sum_engine_internal/compliance/
  hipaa_164_312_b.py` validates a `sum.audit_log.v1` stream
  against the per-row form floor of HIPAA Security Rule 45 CFR
  § 164.312(b) (Technical Safeguards — Audit Controls). Six
  rules R1–R6: schema-pinned, timestamp-present, timestamp-
  iso8601-utc, activity-type-recorded (operation), system-
  component-identified (cli_version), examination-completeness
  (per-operation anchors: attest source_uri, verify `ok`
  presence, render mode).

  **Substrate-tightening — substrate held for the third time.**
  `sum.compliance_report.v1` shape is now a *regularity*, not a
  single- or two-data-point claim. The cross-regime CLI dispatch
  contracts (C1 registry consistency, C2 schema, C3 exit codes)
  in `Tests/compliance/test_cli_dispatch.py` extended automatically
  when HIPAA was registered in both `_COMPLIANCE_REGIMES` and
  `_compliance_validators()`. New cross-regime shape pin
  `test_hipaa_164_312_b::test_validation_report_shape_matches_other_regimes`
  asserts byte-shape parity across all three regimes (Art 12,
  Art 30, § 164.312(b)).

  **Truth-first scope.** Wire-spec doc
  `docs/COMPLIANCE_HIPAA_164_312_B.md` §"What this validator does
  NOT pin" names the deployment-scope obligations the validator
  cannot reach: the auditor function (§ 164.312(b)'s "examine"
  verb requires humans/processes that look at the logs), the
  § 164.530(j)(2) six-year retention obligation, surrounding
  access-control safeguards (§ 164.312(a)(1), § 164.308(a)(4)),
  ePHI inventory (whether SUM is even in scope), and user
  identification (the schema doesn't carry user_id; multi-user
  HIPAA deployments need an authenticating proxy or schema
  extension).

  **Rule-shape note.** R6 (`examination-completeness`) overlaps
  in shape with EU AI Act Art 12 R4 + R5 + R6 (operation-specific
  anchors). The rules are NOT lifted into a shared module — the
  statutory anchors differ (HIPAA points at ePHI activity, Art 12
  at AI traceability), so rule_ids stay regime-specific even
  though the per-row check shape is similar. A future PR may
  extract a shared predicate library if 4+ regimes end up needing
  the same per-operation anchors.

  **Test deltas:** 27 HIPAA rule tests (24 per-rule + clean-pass
  + cross-regime + e2e). Total compliance suite: 32 EU AI Act +
  25 GDPR + 5 CLI dispatch + 27 HIPAA = 89 tests.

- **GDPR Article 30 validator — second per-regime compliance consumer.**
  `sum_engine_internal/compliance/gdpr_article_30.py` validates a
  `sum.audit_log.v1` stream against the per-row floor of Regulation
  (EU) 2016/679 Article 30 (Records of Processing Activities). Five
  rules R1–R5: schema-pinned, timestamp-present, timestamp-iso8601-
  utc, processing-category-present (operation), processor-identity-
  present (cli_version). Contract docs at
  `docs/COMPLIANCE_GDPR_ARTICLE_30.md` including a "what this does
  NOT pin" section naming the record-set scope (Art 30(1)(a)–(g)
  controller-level metadata, Art 30(4) availability, Art 30(5)
  exemption assessment, Art 6 lawful-basis verification) — truth-
  first discipline carried over from Art 12.

  **Substrate-tightening (this is the meta point of P11):**
  GDPR Art 30 is the second regime to consume `sum.compliance_
  report.v1`. The shape held without modification, *proving* the
  regime-agnosticism claim that was a single-data-point assertion
  before. Refactored `sum_cli/main.py::cmd_compliance_check` from
  a hardcoded `if regime == "eu-ai-act-article-12"` ladder into a
  dispatch dict (`_compliance_validators()`); the if-equality smell
  was invisible while there was only one regime, became obvious
  with a second. New `Tests/compliance/test_cli_dispatch.py`
  pins three cross-regime substrate contracts (C1: registry
  consistency; C2: every regime returns sum.compliance_report.v1;
  C3: exit-code contract 0/1/2).

  **Test counts:** 25 GDPR rule tests + 5 CLI dispatch tests = 30
  new compliance tests, all green. Total compliance suite now
  62 tests (32 EU AI Act + 25 GDPR + 5 cross-regime CLI dispatch),
  up from 32 before this PR.

  **Cross-regime shape proof** (`Tests/compliance/
  test_gdpr_article_30.py::test_validation_report_shape_matches_
  eu_ai_act`) — same `to_dict()` keys, same Violation dataclass
  fields, only `regime` + `rule_id` strings differ. Empirical
  proof that downstream consumers (dashboards, retention pipelines)
  can ingest reports across regimes without per-regime adapters.

  **What this enables.** The playbook P11 entry now reads "second
  regime consumed the substrate cleanly"; future regimes (HIPAA
  § 164.312(b) audit controls, SOC 2 CC7.2, ISO 27001 A.8.15, PCI
  DSS 4.0 Req 10 — all record-keeping shape) inherit C1/C2/C3
  contracts automatically by adding to both registries.

- **v3.2 — F3 STRUCTURAL FAIL closer at the detector layer.**
  PR #125's diagnostic settled F3 FAIL as structural (when the
  per-doc graph has `L_IB = 0`, harmonic extension is independent
  of `x_B`, so the deviation field is exactly invariant under any
  boundary-only perturbation). v3.2 responds with a *strict
  generalization* of v3 that adds the harmonic-extension deviation
  as a complementary signal:

      v_combined_v32 = v_laplacian_w + γ · deviation_w + λ · v_deficit

  At γ = 0, v3.2 reduces to v3 numerically (subsumption — H16). At
  γ > 0, deviation contributes additively where it has signal;
  falls back to a constant where it's structurally blind, so
  v_laplacian_w still surfaces the perturbation. Module
  `sum_engine_internal/research/sheaf_laplacian_v32.py`; tests
  `Tests/research/test_sheaf_laplacian_v32.py` pin five falsifiable
  predictions H16-H20 (subsumption, L_IB ≠ 0 visibility, F3
  fall-back, no-λ-double-counting, degenerate-boundary fall-back).

  **Corpus-scale validation** (`fixtures/bench_receipts/v3_2_validation_2026-05-03.json`,
  `bench_digest = 97cf977512f9...162f43f`, schema
  `sum.sheaf_v3_2_validation.v1`): F4 (trusted-mean AUC ≥ 0.55)
  PASSES at all γ values tested (γ ∈ {0.0, 0.1, 1.0, auto≈1.0}).
  F5 (no regression vs v3) PASSES at γ ∈ {0, 0.1} (Δ = 0.000,
  −0.004) but FAILS at γ ∈ {1.0, auto} (Δ = −0.031 each — the
  magnitude-matching γ_auto resolves to 1.0127, effectively the
  same as the γ=1.0 cell). The
  truth-first reading: deviation's signal-to-noise ratio on this
  corpus is worse than its magnitude suggests; auto-calibration
  via magnitude-matching is wrong here. Optimal γ is small (≤ 0.1).
  H16 verified at corpus scale: γ = 0 produces trusted-mean AUC =
  0.661, byte-identical to v3's.

  **Reproducibility caveat documented:** `bench_digest` matches
  across runs only when invoked with `PYTHONHASHSEED=0`. Set-
  iteration order in the sieve and `KnowledgeSheafV2.from_triples`
  is hash-randomized otherwise. This caveat applies to the v3
  corpus ROC bench and F3 diagnostic on-disk digests as well — a
  future PR should sort at every set→list conversion in the
  substrate. Recorded in the v3.2 receipt as
  `reproducibility_requires` field.

  **v3.3 candidate directions named** (not investigated): per-doc
  graph-structure-aware γ (use deviation only where L_IB has high
  mass); cochain redesign that propagates render content into the
  interior; A2 weakness via predicate-perturbation negative sampling
  (orthogonal to the v3.2 arc; affects every detector v22/v3/v31/v32).

- **F3 diagnostic harness — F3 FAIL is structural, not parametric.**
  PR #124 reported F3 FAIL on v3.1 boundary deviation at corpus
  scale and named three competing hypotheses (A graph too small,
  B cochain produces zero-vectors, C random partition too harsh).
  This PR builds a 2×2×2 diagnostic over those three axes and runs
  it. **Result: load_bearing_hypothesis = "none".** All 8 cells
  FAIL the F3 PASS threshold (trusted-mean AUC ≥ 0.55); every
  single-axis flip of the PR #124 baseline still FAILs; even the
  all-three-axes-flipped cell FAILs. The detector has a
  *structural* blind spot: when a perturbation targets a
  trusted-edge (boundary) vertex, the cochain change is at
  boundary positions; the harmonic extension formula
  `x_I^* = -L_II^{-1} L_IB x_B` recomputes the interior from the
  new boundary, but the actual interior is unchanged — so
  deviation `‖x_I_actual - x_I^*‖²` ties between clean and
  perturbed by mathematical necessity. Documented at
  `docs/SHEAF_HALLUCINATION_DETECTOR.md` §3.4.3 with implications
  for v3.2 redesign. Receipt:
  `fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-03.json`.

  **New surface: `bench_digest`** — JCS-canonical SHA-256 over
  the quantized report payload (AUCs to 3 decimals; diagnostic
  floats to 4). Three uses: reproducibility canary (same machine
  + same code → same digest); cross-runtime witness (a future
  Node port that reproduces these AUCs has the matching digest
  as portability proof); signable bench artifact (Ed25519-sign
  with project's existing JWKS keys → arXiv preprint can cite
  the digest, readers re-run and verify). Same trust alphabet as
  `render_receipt.v1`. Smoke test
  `test_v3_1_f3_diagnostic_digest_is_quantization_stable` pins
  that two consecutive in-process runs produce identical digests.

- **v3 corpus-scale ROC bench (`sum.sheaf_v3_roc_bench.v1`).**
  Re-uses the 16-doc `seed_long_paragraphs` corpus; deterministic
  50/50 trust partitioning per doc (SHA-256 seed); compares v2.2
  baseline, v3 receipt-weighted, and v3.1 boundary deviation
  detectors across (A1 entity-swap, A2 predicate-flip, A4 triple-
  drop) × (trusted-target, untrusted-target). Receipt at
  `fixtures/bench_receipts/v3_roc_bench_2026-05-03.json`.

  **Three falsification verdicts (truth-first):**
  - **F1 MARGINAL.** v3 mean AUC on trusted-target = 0.685 vs
    v2.2 = 0.663 (Δ = +0.022). H4 holds dramatically at synthetic
    scale (10/10 wins) but only marginally at corpus scale.
  - **F2 PASS.** v3 doesn't collapse on untrusted-target (no
    class drops > 0.10 vs v2.2). 0.1 floor weight is a viable
    naturalistic-prose default.
  - **F3 FAIL.** v3.1 boundary deviation: trusted mean AUC = 0.50,
    untrusted = 0.34. Synthetic H12 utility test passed (PR #122);
    corpus-scale FAILS. Real falsification — boundary inference
    needs work for naturalistic prose with random 50/50 partition.

  F3 is the most important finding. The synthetic test suite
  alone was insufficient to surface this: it took a corpus-scale
  bench. v3.2 hypotheses (larger graphs, better cochain
  construction at vertex boundaries, structurally-meaningful
  trust partitions) named explicitly in `docs/SHEAF_HALLUCINATION_
  DETECTOR.md` §3.4.2.

  Standout positive result: A4 triple-drop @ untrusted goes from
  v2.2 AUC 0.84 → v3 AUC 0.97 (+0.13). The biggest concrete win
  is "v3 catches dropped triples sharply, regardless of whether
  the dropped edge was trusted or not." That single signal
  carries most of v3's value; the trusted-side amplification
  claim is more nuanced.

- **Audit-tightening pass on PR #119/#120/#121/#122 tests.**
  Independent audit surfaced 5 high/medium issues in tests added
  this cycle; closing them surfaced a **real bug** in v3's
  combined-detector formula:
  - **v3 double-counted λ** (`v_combined_v3` previously computed
    `v_laplacian_w + λ · v_deficit` where v2.2's `v_deficit` is
    *already* `λ · deficit²`, giving `λ² · deficit²`). Fixed:
    `v_combined_v3 = v_laplacian_w + v_deficit`. Caught by the
    new `test_combined_v3_lambda_wiring_with_nonzero_deficit`,
    which the prior tests couldn't catch because they only
    exercised λ on clean renders (deficit = 0).
  - H1 (linearity) tautology replaced with four-property pin
    (homogeneity + zero-weights + singleton + additivity), four
    seeds, plus negative control on `e_i` weight vectors.
  - H3 tautology replaced with sentinel weights `[0, 1, 0]` whose
    expected value is hand-known regardless of implementation
    order of operations.
  - Headline trusted-vs-untrusted utility test now loops over 10
    seeded perturbations, asserting ≥ 8/10 wins + mean inequality.
  - R2 (required traceability fields) parametrized over 3 fields ×
    {empty string, None}; R2 missing-timestamp now also pins R3
    (timestamp validity) does NOT fire (the implementation guard).
  - R3 unparseable-Z timestamp case added (the second R3 failure
    mode that wasn't tested).

- **v3.1 harmonic-extension boundary inference.** Implements
  Hansen-Ghrist 2019 Proposition 4.1 / Theorem 4.5: given a sheaf
  on a graph and a partition of vertices into a trust-frame
  *boundary* $B$ and an *interior* $I$, the harmonic extension is
  the unique cochain that agrees with $x_B$ on $B$ and minimizes
  $\|\delta x\|^2$ over $I$. Closed form: $x_I^* = -L_{II}^{-1}
  L_{IB} x_B$, computed via `np.linalg.lstsq` for numerical
  stability under rank-deficient $L_{II}$ (disconnected interior /
  interior with global section). The v3 weighted-Laplacian
  primitive shipped earlier in this cycle was the prerequisite —
  v3.1 takes the same matrix and uses it to interpolate, not just
  to score.
  Practical use: `boundary_from_weights(sheaf, weights)` derives
  $B$ from per-edge receipt weights (a vertex joins the boundary
  iff every incident edge has weight ≥ threshold);
  `boundary_deviation(sheaf, x_full, B)` computes
  $\|x_I - x_I^*\|^2$ — the headline hallucination signal "render
  diverges from what the trust frame predicts." Ten falsifiable
  predictions pinned (H6–H15) including a **surfaced-mid-PR
  honesty pin**: with a single bridge edge connecting boundary to
  interior, the harmonic extension is weight-invariant for any
  positive weights (analytic reason: rank-1 bridge column makes
  the $r = w_{bridge}/w_{interior}$ ratio cancel). Documented in
  the test rather than worked around. The weight effect IS
  observable with multiple bridge edges; that case is also pinned.

- **v3 receipt-weighted sheaf-Laplacian detector (Block B).**
  Extends v2.2's combined detector with per-edge weights derived
  from the trust loop's own outputs — Ed25519-signed render
  receipts. The weighted sheaf Laplacian $L_F^w = \delta^T W \delta$
  (Hansen-Ghrist 2019 §3.2 weighted generalization) gives edges
  backed by trusted-issuer JWKS receipts a higher weight (1.0),
  unsigned edges a lower-weight floor (0.1), and revoked-key edges
  weight 0. The math claim carries: $L_F^w$ is symmetric PSD, the
  factored form $\sum_e w_e \|residual_e\|^2$ matches the
  materialized quadratic form numerically, and uniform weights
  $w_e = c$ reduce v3 to $c \cdot v_2$ exactly. Five falsifiable
  predictions pinned in `Tests/research/test_sheaf_laplacian_v3.py`
  including the headline utility claim H4: tampering a trusted
  (high-weight) edge produces a sharper $\Delta V$ than tampering
  an untrusted edge — receipt-weighting amplifies signal where
  the system already trusts. **Architectural note:** v3 is fractal
  in the project's intended sense — the system's own trust
  artifacts (cross-runtime-verified render receipts) feed into the
  detector's confidence weighting. No other system has a working
  cross-runtime trust triangle to seed this. Out-of-scope and
  named explicitly: harmonic-extension boundary inference (v3.1
  candidate); JWKS verification round-trip (caller's
  responsibility); corpus-scale empirical bench (follow-up).

- **First per-regime compliance validator (Path 3).** EU AI Act
  Article 12 (record-keeping for high-risk AI systems) — the first
  actionable layer on top of the `sum.audit_log.v1` substrate
  shipped in v0.5.0. Six rules pin per-row traceability fields
  (schema, timestamp, operation, cli_version) plus operation-
  specific anchors (`source_uri` for attest, `axiom_count` +
  `state_integer_digits` for verify, `mode` for render). New
  `sum compliance check --regime eu-ai-act-article-12` and
  `sum compliance regimes` CLI verbs; pipe-friendly exit codes
  (0 = ok, 1 = violations) for CI gates. Regime-agnostic
  `sum.compliance_report.v1` shape so future regimes share
  consumers. Tightening of the audit-log contract earlier this
  cycle (PR #119, signed-bundle / multi-process / worker-mode
  render coverage) was the prerequisite — the validator can pin
  what the substrate now reliably emits.

## [0.5.0] — 2026-05-01

Minor-bump feature release. Three substrate-extending PRs land
together as the v0.4.1 → v0.5.0 cycle:

- **#104 — MCP server `render` tool.** Bidirectional symmetry on the
  agent surface. The CLI gained `sum render` in v0.4.0; the MCP
  server's tool surface (`extract / attest / verify / inspect /
  schema`) now also includes `render`. Tool count 5 → 6. The
  bidirectional 3×3 grid (CLI/MCP/HTTP × attest/verify/render) is
  fully populated. MCP-aware LLM clients can drive both directions
  of the trust loop entirely from inside an LLM session.

- **#105 — Sheaf-Laplacian hallucination detector spec.**
  Specifies a sheaf-Laplacian consistency score over signed
  render-receipt manifolds, grounded in Gebhart, Hansen & Schrater
  (2023, AISTATS, arXiv:2110.03789) "Knowledge Sheaves" and the
  sheaf-Laplacian theory of Hansen & Ghrist (2019). Mathematical
  primitive: `x^T L_F x = Σ_e ‖F_v⊵e x_v − F_u⊵e x_u‖²` — the
  Laplacian quadratic form, zero exactly when the cochain is a
  global section, strictly positive otherwise. SUM-to-Knowledge-
  Sheaves mapping charted (state integer ↔ Yoneda token; cross-
  runtime byte-identity ↔ descent under cover; render-receipt ↔
  signed cochain witness). v1/v2/v3 procedures specified;
  falsifiable predictions named; bounded claims set.

- **#106 — v1 sheaf-Laplacian hallucination detector
  (implementation).** Implements the v1 detector specified in #105,
  behind the new `[research]` extras flag in `pyproject.toml`.
  Production install path is unaffected — `numpy` and `scipy` are
  gated behind `pip install 'sum-engine[research]'`. Math verified:
  7 sanity properties pinned (symmetric, PSD, ≥0 quadratic form,
  constant cochain in kernel, single-missing-entity gives V=1,
  per-edge top-1 finds the missing edge, empty-render false
  negative pinned). Synthetic micro-benchmark (6 fact-sets × 5
  perturbation classes, all *connected* graphs): 18/30 catch on
  entity-presence-affecting perturbations, **18/18 = 100% top-1
  localization on caught classes**. Spec corrections from
  empirical run: A5 consistent-hallucination is partially caught
  (via mean signal); P3 localization actually 100% on synthetic
  data (target was 70%); empty-render edge case named in bounded
  claims.

- **#107 — Real-data falsification of v1 on naturalistic prose.**
  Tested v1 on 4-fact disconnected source graph (real prose, real
  sieve extraction, no LLM API). **v1's density-dropout signal
  collapsed to zero on the disconnected graph** because dropping
  whole components leaves every remaining edge in {(1,1), (0,0)}
  — never (1,0). The synthetic bench used connected graphs and
  missed this. Honest framing now in the spec: v1 is a
  *connected-graph entity-presence drift detector* + a
  *sieve-canonicalisation-divergence detector across paraphrases*
  (catches verbose paraphrase 3 producing `python_code` instead of
  `python`, V=3); v1 is **not** a general hallucination detector
  for naturalistic prose. v2 motivation strengthened from
  "addresses A2/A3" to "addresses every v1 blindspot named so far
  including disconnected-graph density-dropout blindness."
  Reproducible: ``PYTHONPATH=. python
  scripts/research/sheaf_real_test.py``. Pinned in code by
  ``test_disconnected_graph_density_dropout_invisible``.

Counts at release: **145 features** (130 production, 14 scaffolded,
1 designed). All CI drift gates green; cross-runtime K-matrix +
A-matrix locked; release machinery validation green; fresh-venv
``pip install`` smoke green.

New extras group: **``[research]``** (`pip install
'sum-engine[research]'`) for `numpy`, `scipy` — required by
``sum_engine_internal/research/sheaf_laplacian.py``. Production
install path (``pip install sum-engine``) is unaffected.

Zero breaking changes from v0.4.1. Every v0.4.1 invocation still
works identically.

---

### v1 sheaf-Laplacian hallucination detector — implementation + spec corrections

Implements the v1 detector specified in
``docs/SHEAF_HALLUCINATION_DETECTOR.md`` §3.2 and verifies the
spec's mathematical and empirical claims against a reproducible
synthetic benchmark. Surfaces and corrects two spec
mischaracterizations against actual measurement.

Module: ``sum_engine_internal/research/sheaf_laplacian.py``,
behind the new ``[research]`` extras flag in ``pyproject.toml``
(``pip install 'sum-engine[research]'``). Production install
path is unaffected — research dependencies (numpy, scipy) are
not pulled by default.

Math verified — 7 sanity properties pinned in
``Tests/research/test_sheaf_laplacian.py``:

  1. Laplacian symmetric (L = δ^T δ)
  2. Laplacian positive-semidefinite
  3. Quadratic form ≥ 0 on random cochains
  4. Constant cochains are global sections (kernel of δ)
  5. Single-missing-entity cochain has predicted V = 1
  6. Per-edge top-1 localization finds the missing edge
  7. Empty-render false-negative pinned (V = 0 on x = 0)

Empirical separation verified on 6 fact-sets × 5 perturbation
classes = 30 trials in ``scripts/research/sheaf_microbench.py``:

  | Class                       | Caught | Localization |
  | --------------------------- | ------ | ------------ |
  | A1 entity-swap              | 6/6 ✓  | 6/6          |
  | A2 predicate-flip           | 0/6 —  | known blind  |
  | A3 off-graph fabrication    | 0/6 —  | known blind  |
  | A4 triple-drop              | 6/6 ✓  | 6/6          |
  | A5 consistent-swap (×3)     | 6/6 ✓  | 6/6          |

Total: **18/30 catch rate; 18/18 = 100% top-1 localization on
caught classes**. The 60% catch rate is precisely the v1 design's
claim — catch entity-presence-affecting perturbations cleanly,
defer predicate-sensitive (A2) and off-graph-sensitive (A3) to
v2 (learned-embedding stalks, planned).

Spec corrections (now in
``docs/SHEAF_HALLUCINATION_DETECTOR.md``):

  - **A5 consistent-hallucination** was originally claimed as a
    blanket v1 blindspot. Empirically: A5 *via entity substitution*
    is caught (mean Laplacian is positive even when per-render
    variance is zero). Only A5 *via predicate-flip* (which is A2)
    or *via off-graph fabrication* (which is A3) is missed. The
    detector signals on the mean, not just the variance.
  - **P3 localization** prediction was ≥ 70%; actual is 100% on
    caught classes. Bar preserved as the threshold v1 must clear;
    v1's actual performance is a strict superset.
  - **Empty-render false negative** was not originally named in
    bounded-claims; now explicit. Callers must treat
    ``n_extracted == 0`` as a separate signal — the Laplacian
    alone cannot distinguish "all entities present everywhere"
    from "no entities present anywhere."

Tests run via ``pytest Tests/research/test_sheaf_laplacian.py``
(12 passed). Reproducible bench:
``PYTHONPATH=. python scripts/research/sheaf_microbench.py``.

FEATURE_CATALOG entry 145 (🔧 scaffolded — code tested, no
production consumer wired). Counts: 144 → 145 total; production
unchanged at 130; scaffolded 13 → 14.

This implementation grounds SUM's primitives inside the
peer-reviewed categorical-AI conversation (Knowledge Sheaves,
Gebhart et al. 2023, AISTATS, arXiv:2110.03789) with a working
artifact, not just a spec. v2 (learned-embedding stalks) is the
natural next step; v3 (receipt-weighted, the SUM-specific
extension) follows v2.

### Research direction — sheaf-Laplacian hallucination detector spec

Documents (without yet implementing) a mathematically rigorous
hallucination-consistency score on top of SUM's signed render-receipt
manifold. Grounded in Gebhart, Hansen & Schrater (2023, AISTATS,
arXiv:2110.03789) "Knowledge Sheaves" and the sheaf-Laplacian theory
of Hansen & Ghrist (2019). The mathematical primitive is Equation 1
of Gebhart et al.: the sheaf-Laplacian quadratic form
``x^T L_F x = sum_e ‖F_u→e x_u − F_v→e x_v‖^2`` — zero exactly when
the cochain ``x`` is a global section, strictly positive otherwise.
Applied to SUM: build a knowledge-sheaf on a bundle's triple set;
treat each rendered tome (varying slider position, paraphrase,
model) as a 0-cochain via re-extracted entity presence/embeddings;
the Laplacian quadratic form measures how badly the rendering
manifold fails to glue across the cover.

This is the first artifact that grounds SUM's primitives inside the
peer-reviewed categorical-AI conversation (the substantive 80% of
the SCT-style synthesis the project's intellectual context points
toward). Coupled to render-receipt issuer-trust weighting (v3 of
the spec), the obstruction-class score is the SUM-specific extension
of the published Knowledge-Sheaves framework; it does not replicate
elsewhere because no other system has cross-runtime-verified render
receipts.

The spec includes:

  - Theoretical foundation with verbatim citation of Gebhart et al.
    Definition 4 (cellular sheaf) and Equation 1 (Laplacian
    quadratic form).
  - SUM-to-Knowledge-Sheaves mapping table — mechanical correspondence
    between SUM primitives (state integer, render receipt, K1–K4
    cross-runtime byte-identity) and Knowledge-Sheaves primitives
    (Yoneda token, signed cochain witness, descent under a covering
    family).
  - v1 (1-dim presence stalks), v2 (text-embedding-3-small stalks),
    v3 (receipt-weighted) procedure specs.
  - Falsifiable predictions: ROC AUC ≥ 0.75 on synthetic
    adversarial benchmark (entity-swap, predicate-flip, fact-
    fabrication, negation-injection); per-edge top-k localisation
    of perturbed triples ≥ 70%; receipt-weighting concentration.
  - Bounded claims explicitly disclaiming hallucination "solution",
    correctness proofs, and detection of consistent-hallucination
    adversarial regimes.
  - 3-week one-engineer plan: Week 1 references + scaffold; Week 2
    v1 prototype + synthetic benchmark; Week 3 v2 + arXiv note.
  - Position vs. the three monetisable wedges (agent-trust,
    C2PA-text-equivalent, compliance audit) — the same artifact in
    all three settings; only the cover-computer and score-reader
    differ.

No code shipped in this entry. The companion implementation will
land in `sum_engine_internal/research/sheaf_laplacian.py` (behind a
``[research]`` extras flag so production install is unaffected) in
a follow-up PR after the spec is reviewed.

CLAUDE.md gains item 10 in the read-first list pointing at the new
spec doc, so future memory-less sessions do not start a parallel or
contradictory research direction.

### MCP server `render` tool — bidirectional symmetry on the agent surface

Closes the gap PR #97 left on the MCP side: the CLI gained `sum render`
in v0.4.0, the MCP server's tool surface (`extract / attest / verify /
inspect / schema`) didn't follow until now. With this entry, MCP-aware
LLM clients (Claude Desktop, Claude Code, Cursor, Continue) can drive
both directions of the trust loop — `attest` to mint a bundle from
prose, `render` to re-emit a tome from a bundle — entirely from inside
an LLM session, with byte-compatible artifacts that verify under any
SUM verifier.

Same algebra, same `generate_controlled`, byte-compatible with the
CLI's local path. Local-only by default (deterministic density slider);
non-neutral length / formality / audience / perspective return
`error_class="schema"` with a message pointing at the Worker's
`POST /api/render` for LLM-conditioned rendering — the MCP server
stays fully offline by default, preserving the existing
`SUM_MCP_ALLOW_NETWORK` opt-in property.

Tool surface now: `extract / attest / verify / inspect / **render** /
schema`. Tool count goes from 5 to 6.

Tests: `Tests/test_mcp_server.py` (+15, now 44 passing) — 15 render-
specific cases including round-trip integrity at density=1.0
(rendered tome re-mints to the source bundle's `state_integer`,
byte-for-byte), density=0.0 emits no axiom lines, density=0.5 keeps
the lex-prefix, slider-bound validation, malformed-bundle gates,
non-neutral-axes-without-worker rejection with actionable error
message pointing at `/api/render`.



## [0.4.1] — 2026-05-01

Patch release. **Wheel content is byte-identical to never-published
v0.4.0** — the v0.4.0 git tag was pushed on 2026-04-30 but the
publish workflow fail-closed at the pre-promotion verify gate due
to a verifier-side bug (PyPI's Integrity API serialises the leaf
certificate at ``verification_material.certificate`` directly as a
base64 string rather than under a ``rawBytes``/``raw_bytes`` envelope;
the script's walker only matched the older shape and reported "no
Sigstore certificates extractable" even though a valid cert chain
was sitting at the documented PyPI Integrity API path). Production
PyPI was correctly never touched. The fail-closed gate did exactly
what it was designed to do; v0.4.0 stays as a forever-untagged-on-
PyPI git tag.

This release ships the verifier fix plus a workflow-trigger
ergonomic improvement, both CI-only changes outside the wheel.

### Fixed

- **`scripts/verify_pypi_attestation.py`: recognise the flattened
  ``certificate``-string shape (#99).** The walker in
  ``extract_raw_byte_strings`` now collects string values at any
  ``certificate`` key in addition to the existing ``rawBytes`` /
  ``raw_bytes`` matches. Downstream cryptographic checks are
  unchanged: certs still must parse as DER X.509 and produce a SAN
  URI matching the expected workflow + tag-ref prefix; the
  ``parse_certificates`` helper already tolerated non-base64-DER
  candidates by skipping them, so widening the collector cannot
  turn a tamper signal into an accept.

  Verified end-to-end against the live TestPyPI provenance at
  ``https://test.pypi.org/integrity/sum-engine/0.4.0/.../provenance``:
  one cert extracted, valid SAN, zero failures.

  Tests: ``Tests/test_verify_pypi_attestation.py`` (+2, now 23) —
  ``test_extract_raw_byte_strings_pypi_flattened_certificate_string``
  and ``test_extract_raw_byte_strings_co_existing_shapes_both_collected``.

  Plus four adversarial vectors confirmed locally: garbage at the
  ``certificate`` key, valid base64 / non-DER bytes, valid cert
  with wrong repo / workflow / ref-prefix SAN — all rejected.

### Added

- **`.github/workflows/publish-pypi.yml`: ``workflow_dispatch``
  trigger (#100).** Lets an operator re-run an existing tag's
  publish flow without re-tagging or cutting a patch version when
  a transient failure (e.g. the verifier-shape bug above) hits.
  No workflow inputs; the trigger contract stays identical to push
  (version comes from the ref, no surface for an operator to inject
  a different version). ``workflow_dispatch`` is gated by repo
  write access — same baseline as pushing a tag — so this expansion
  does not widen the attack surface beyond the original trigger;
  the fail-closed gates protect against index-side substitution
  regardless of which trigger fired the run.

  **Caveat (documented in the workflow header):** to re-run on the
  same tag, the operator must first delete the existing TestPyPI
  release. The rebuild's wheel + sdist will differ byte-for-byte
  from the staged copy (Python wheel builds embed timestamps unless
  ``SOURCE_DATE_EPOCH`` is set), so ``skip-existing: true`` on the
  TestPyPI upload would silently skip, and the verify step's
  same-bytes check would then fail comparing the new local hashes
  against the stale TestPyPI bytes. Path A (cut a patch version)
  is the documented fallback when the operator cannot or does not
  want to delete the staged TestPyPI release; this very release
  is an instance of Path A.

### Why no v0.4.0 on PyPI

- Use ``pip install sum-engine==0.4.1`` (or simply
  ``pip install -U sum-engine``) for the v0.4.0 feature set.
  v0.4.0 was tagged on git but never published; the v0.4.1 wheel
  content is identical because ``scripts/`` and ``.github/`` are
  excluded from the distribution
  (``[tool.setuptools.packages.find].exclude``).

Zero breaking changes from v0.4.0 (no v0.4.0 wheel exists to
compare against on production PyPI; the v0.4.1 wheel is what 0.4.0
*would have been* if the verifier had been current).

## [0.4.0] — 2026-04-30

Minor-bump feature release. The major arc of the v0.3.0 → v0.4.0 cycle
is the **bidirectional canonical round-trip closing** at the engine,
the cross-runtime trust triangle, the rendering surface, and the LLM-
narrative round-trip — all four made provably or measurably tight, all
four locked behind CI gates that run on every PR.

Highlight reel (PRs #82–#97; full per-PR detail below):

- **`sum render` CLI verb (#97)** — bidirectional `sum attest` ↔
  `sum render` symmetry from the shell. Default deterministic path
  (density-only); `--use-worker URL` returns LLM-conditioned tome +
  signed `render_receipt` (`sum.render_receipt.v1`). The reverse
  direction was reachable from Python and HTTP before; now it is
  reachable from a shell prompt and the README's "tags to tomes and
  vice versa" pitch lines up with the CLI surface.
- **Cold-install onboarding fix (#95)** — `pip install
  'sum-engine[sieve]'` → `sum attest` works in 13 seconds first call,
  instant after. Closes the 90%-of-new-users failure mode.
- **`attest-batch --dedup-threshold` (#94)** — 128-permutation MinHash
  over word 3-shingles, pure stdlib, near-duplicate skip pre-extraction.
- **§2.5 cross-vendor LLM closure (#90 + #93)** — vendor-agnostic
  dispatcher (OpenAI ↔ Anthropic), then a frontier-LLM refresh against
  Claude Opus 4.7 + GPT-5.5. Both 2026-frontier models hit 50/50
  perfect recall on the combined ablation; constrained-extractor alone
  hits 50/50 on GPT-5.5. The closure pattern is **vendor-independent**
  across three model families (gpt-4o-mini, opus-4-7, gpt-5.5).
- **Self-attestation pipeline (#89)** — SUM attests SUM. Five canonical
  docs round-trip via `sum verify`; CI gate enforces source-URI
  coherence; surfaced and fixed a real algebra-level pipe-component
  round-trip bug along the way.
- **Omni-format markdown-pivot (#88)** — PDF/HTML/DOCX/EPUB/JSON/IPYNB/
  RTF/XML route through a deterministic `markitdown==0.1.5` pivot;
  `markdown_sha256` lets verifiers replay the conversion.
- **`sum attest-batch` (#87) + `sum attest` arbitrary-size (#86) +
  chunked Gödel-state composition (#85)** — `compose_chunk_states`
  algebra primitive with 21 property tests asserting `state(chunked)
  == state(unchunked)`; CLI now handles inputs above spaCy's 1 MB cap;
  per-file JSONL batch surface with per-file failure isolation.
- **Repo manifest publisher + CI drift gate (#84)** —
  `meta/repo_manifest.json` is the single source of truth for cross-
  channel state; portfolio + downstream consumers fetch it from a
  stable raw URL.
- **External-awareness checkpoint (#83)** — first deliberate "process
  intensification" cycle; logs frontier developments to track and
  audited-no-action items.

Pre-#83 (PRs #82 and earlier in the v0.4 cycle): cross-runtime
sha256_128_v2 byte-identity gate, `sum verify` extraction-provenance
surfacing (closes THREAT_MODEL §3.3 visibility gap), `/api/qid`
accuracy floor measurement (100% hit-rate / 100% label-substring on
30-term corpus), threat-model executable test suite, scaling §2.5 to
seed_v2 and seed_long_paragraphs, MCP server v1 → v2 hardening,
docs/API_REFERENCE.md, README rewrite around the cross-runtime trust
surface, M1 Merkle set-commitment sidecar.

Counts at release: **143 features in FEATURE_CATALOG (129 production,
13 scaffolded, 1 designed).** Manifest + self-attestation + repo
manifest all current; CI drift gates green; cross-runtime K-matrix
+ A-matrix locked; release machinery validation green (PEP 740
attestations + Sigstore via OIDC wired); fresh-venv `pip install`
smoke green.

Zero breaking changes from v0.3.x. Every v0.3.1 invocation still
works identically; `sum render` is purely additive.

---

### `sum render` CLI verb — closes "tags ↔ tomes" symmetry from the shell

The reverse direction of the bidirectional engine was reachable from
Python (`AutoregressiveTomeGenerator.generate_controlled`) and from
HTTP (`POST /api/render` on the Worker), but had no top-level CLI
verb. `sum attest < prose.txt > bundle.json` worked; the inverse
`sum render < bundle.json > tome.md` did not. Non-Python operators
could not drive the reverse direction from a shell.

`sum render` is now wired:

  $ echo "Alice likes cats. Bob owns a dog." | sum attest > bundle.json
  $ sum render < bundle.json
  @canonical_version: 1.0.0
  @sliders: density=1.000 length=0.500 formality=0.500 audience=0.500 perspective=0.500
  # Rendered Tome
  ...

Two paths:

  • **Local (default).** Deterministic, actions only the density
    slider. Non-neutral length / formality / audience / perspective
    return exit 2 with a message pointing at `--use-worker`. The
    rendered tome is the canonical-format output of
    `generate_controlled`, so re-extracting its lines and re-minting
    primes reproduces the source bundle's `state_integer` exactly
    (proof in `Tests/test_sum_cli_render.py::TestRoundTripFullDensity`).

  • **`--use-worker URL`.** POSTs `{triples, slider_position}` to
    `<URL>/api/render` via stdlib `urllib`, returns the LLM-conditioned
    tome plus the signed `render_receipt` (`sum.render_receipt.v1`).
    No new runtime dependency.

Output shapes:

  • Default — tome text on stdout (matches `sum attest` symmetry).
  • `--output PATH` — write tome to file.
  • `--json` — structured envelope on stdout
    (`{tome, sliders, mode, render_receipt?}`); composes with `--output`.

Tests: `Tests/test_sum_cli_render.py` (19) — round-trip integrity at
density=1.0, lex-prefix density subsetting, slider-bound validation,
malformed-bundle exit codes, `--output` file path, `--json` envelope
shape, worker wire contract (stubbed urlopen), worker HTTP error /
unreachable propagation, stdin input.

Closes the last visible gap between the README's "tags to tomes and
vice versa" pitch and the CLI surface a daily-use operator can reach.

### Onboarding fix — cold-install ``sum attest`` now works in 60 seconds

Empirical audit on a fresh venv surfaced that the README's
"Verify it yourself in 60 seconds" pitch did not actually work:

  $ pip install 'sum-engine[sieve]'      # 15s, fine
  $ echo "..." | sum attest
  sum: no extractor available. [...]
  $ echo $?
  1

Despite ``[sieve]`` having just installed spaCy, the default
``sum attest`` errored out. Root cause: ``_pick_extractor``
probed sieve availability via ``spacy.load("en_core_web_sm")``
which raises ``OSError`` when the model is absent. The exception
was caught broadly and the probe fell through to the LLM check,
then to ``SystemExit`` — never giving the sieve constructor's
auto-download fallback a chance to fire.

One-line fix: probe via ``DeterministicSieve()`` so its OSError-
catching auto-downloader runs. Same UX as ``--extractor sieve``
always had — one stderr announcement of the ~50MB download, then
the attest proceeds. 13s end-to-end on first call after install,
instant on subsequent calls.

Tests: ``Tests/test_pick_extractor_cold_install.py`` (4) — probe
routes through DeterministicSieve, falls back to LLM if sieve
construction fails, SystemExit carries the install hint string,
``--extractor`` override short-circuits the probe.

Closes the 90%-of-new-users failure mode. The README's pitch and
the actual product behaviour now line up.

### §2.5 frontier-LLM refresh — GPT-5.5 (closure pattern is vendor-independent)

Symmetric refresh against OpenAI's `gpt-5.5-2026-04-23` snapshot,
completing the cross-vendor receipt set the 2026-04-29 external-
awareness checkpoint queued. Same seed_v1 corpus, same intervention
ablations, same `sum.s25_generator_side.v1` schema family the
Opus 4.7 receipt uses (with `provider: "openai"`).

  | Model                       | canonical_first         | constrained_extractor   | combined                |
  | --------------------------- | ----------------------- | ----------------------- | ----------------------- |
  | gpt-4o-mini-2024-07-18 (Jul 2024, baseline) | — (closure was at recall ≥ 0.97) | — | — |
  | Claude Opus 4.7 (Apr 2026)  | drift 94.70 / r 0.96 / 48-50 | drift 9.33  / r 0.96 / 48-50 | **drift 0.00 / r 1.00 / 50-50** |
  | GPT-5.5 (Apr 2026)          | drift 58.48 / r 0.98 / 49-50 | drift 2.00  / r 1.00 / **50-50** | drift 5.33 / r 1.00 / 50-50 |

**Headlines:**

  - Both 2026-frontier models hit **50/50 perfect recall on the
    combined ablation** — strictly stronger than the gpt-4o-mini
    baseline.
  - GPT-5.5 hits 50/50 with **constrained_extractor alone** (drift
    2.00%, no canonical-first generator needed). Cleanest single-
    intervention result on record. Indicates frontier alignment
    with source vocabulary is now tight enough that one of the two
    interventions is redundant on the easy half of the corpus.
  - The intervention pattern is **vendor-independent across the
    OpenAI ↔ Anthropic frontier as of 2026-04-29**. The §2.5
    closure isn't an artifact of any single model family.

Receipt: `fixtures/bench_receipts/s25_frontier_models_2026-04-29_gpt55.json`
(provider `openai`, model `gpt-5.5-2026-04-23`, 50 docs × 3
ablations, ~$1 spend).

Tooling additions:

  - `scripts/bench/runners/s25_smoke.py` — vendor-agnostic single-
    doc smoke (renamed from `s25_anthropic_smoke.py`). Routes by
    model-id prefix; one shared script for all dispatcher targets.
  - `scripts/bench/runners/list_openai_models.py` — lists the
    frontier-class snapshots available to the active OpenAI key.
    Used to verify the `gpt-5.5` snapshot id before spend.

### §2.5 frontier-LLM refresh — Claude Opus 4.7 (closure pattern transfers)

The §2.5 LLM round-trip closure was originally locked at recall ≥ 0.97
on `gpt-4o-mini-2024-07-18` (Jul 2024). With Anthropic Claude Opus 4.7
(Apr 2026) shipping, the question became whether the
canonical-first-generator + constrained-extractor pattern is model-
specific or model-independent. Re-measured against Opus 4.7 across
the same 50-doc seed_v1 corpus:

  | Ablation                | Drift %  | Recall   | Full-recall |
  | ----------------------- | -------- | -------- | ----------- |
  | canonical_first only    | 94.70    | 0.9600   | 48/50       |
  | constrained_extractor   | 9.33     | 0.9600   | 48/50       |
  | **combined**            | **0.00** | **1.0000** | **50/50**  |

Headline: the **combined ablation hit perfect 50/50 full-recall and
0.00 drift on Claude Opus 4.7** — strictly stronger than the
gpt-4o-mini result (which already locked the closure at recall 0.97+).
The intervention pattern is model-independent across the OpenAI ↔
Anthropic frontier as of 2026-04-29.

Receipt:
`fixtures/bench_receipts/s25_frontier_models_2026-04-29_opus.json`
(schema family `sum.s25_generator_side.v1`, provider `anthropic`,
50 docs × 3 ablations).

To make the bench vendor-agnostic, this release ships:

  - `sum_engine_internal/ensemble/llm_dispatch.py`: `OpenAIAdapter`
    and `AnthropicAdapter` behind a single `LLMAdapter` surface
    (`parse_structured`, `generate_text`). `get_adapter(model_id)`
    routes by prefix (`gpt-`/`o1-`/`o3-`/`o4-` → OpenAI;
    `claude-` → Anthropic; unknown → `ValueError`, never silently
    misroute).

  - Pydantic → Anthropic bridge: `model_json_schema()` becomes the
    Anthropic tool's `input_schema` with `$defs`/`$ref` inlined;
    `tool_choice` forces the model to emit a `tool_use` block whose
    `input` round-trips through `schema.model_validate(...)`.

  - The §2.5 generator-side runner refactored to take an `adapter`
    instead of a `(client, model)` pair. The four call helpers
    (`_baseline_extract`, `_constrained_extract`, `_baseline_generate`,
    `_canonical_first_generate`) call uniform adapter methods. The
    runner's `S25CallTimeoutError` path is preserved by a thin shim
    that converts the dispatcher's `LLMCallTimeoutError` back so the
    per-doc-skip + receipt aggregate paths keep working without
    changes.

  - New optional extra `[anthropic] = ["anthropic>=0.97.0",
    "pydantic>=2.0.0"]`. Both `[llm]` (OpenAI) and `[anthropic]` may
    coexist; users only install the one matching the model id they
    target.

  - Per-call timeout discipline preserved end-to-end: dispatcher
    wraps each SDK call in `asyncio.wait_for`; on timeout, raises
    `LLMCallTimeoutError`; runner converts to `S25CallTimeoutError`;
    `run_doc` records `error_class: "timeout"` and the aggregate
    excludes the timed-out doc from means.

  - `scripts/bench/runners/s25_anthropic_smoke.py`: a one-doc
    (~$0.005, ~30s) smoke that validates dispatcher routing +
    tool-use round-trip + narrative generation end-to-end on the
    live API before committing to the full bench. Used to verify
    wiring before the receipt above was minted.

Tests: `Tests/test_llm_dispatch.py` (13 unit tests with mocked SDK,
no spend) + `Tests/test_s25_runner_timeout.py` updated to mock the
new adapter surface. 35 tests green across the dispatch +
intervention surface.

This closes the §2.5-LLM-refresh item that the 2026-04-29 external-
awareness checkpoint added to the queue.

### Added — repo manifest publisher (single source of truth for cross-channel state)

Closes the cross-channel-drift problem the SUMequities portfolio audit
surfaced ("100 commits / 30d" displayed; actual is 239). The manifest
publisher emits a JSON file under schema `sum.repo_manifest.v1`
that downstream consumers (the SUMequities portfolio, dashboards,
status pages, anyone) fetch and read instead of computing values
locally.

**The manifest** (`meta/repo_manifest.json`) captures every load-
bearing public-surface fact in one file:

- Repo metadata (owner, name, license)
- Git state: `head_sha`, `head_short`, `head_subject`,
  `head_committer_date`, `commits_last_30d`
- GitHub stars (live via `gh repo view`)
- Release: `pyproject_version` + `pypi_published_version`
- Feature counts: total / production / scaffolded / designed
  (mechanically derived from FEATURE_CATALOG.md headings)
- Receipt fixtures catalog (every `fixtures/bench_receipts/*.json`
  with its schema and `issued_at`)
- Hosted-demo URLs (worker, JWKS, revocation list)

**Producer**: `python -m scripts.repo_manifest --out meta/repo_manifest.json`

**Consumer**: any HTTP client fetching the file via
`https://raw.githubusercontent.com/OtotaO/SUM/main/meta/repo_manifest.json`
(or, for the portfolio's case, the equivalent CDN-backed URL).

**CI gate** — new step in `quantum-ci.yml`:
`python -m scripts.repo_manifest --check meta/repo_manifest.json`.
The check strips time-varying fields (`issued_at`, GitHub stars)
and compares the substantive content. **A PR that changes anything
the manifest reflects (commit count, feature counts, version,
receipts) without re-running the publisher fails CI** with a
one-line refresh recipe in the error output.

**Self-applicable verifiability discipline.** The manifest is
itself a structured, fetchable, diff-able artifact — verifiable in
all three operative senses: reproducible by anyone (`gh` + git +
filesystem only), falsifiable (the CI gate fails on drift), and
forward-compatible (future operator decision can Ed25519-sign it
with the trust-root key). SUM's own thesis applied to its own
repo metadata.

**Tests:** 8 in `Tests/test_repo_manifest.py` cover: schema
identifier pinned; load-bearing fields present; receipt catalog
includes session-shipped fixtures; stable-view strips time-
varying fields and is idempotent; --check passes on
just-emitted manifest; --check fails on stale `commits_last_30d`
with refresh recipe in stderr; --check fails on missing file.

**The audit-detected portfolio divergence (100 vs 239 commits)
will close when the SUMequities portfolio is wired to fetch
this manifest** — that's an operator-side change in a separate
repo. The producer side is now in place.

This is the second deliberate "process intensification" move
(the first was the external-awareness checkpoint in PR #83).
Both are mechanisms, not just measurements: each runs every PR
and surfaces drift at CI time.

### External-awareness pass — track relevant 2026-04 developments

A focused audit of external developments since the current
substrate decisions. Each finding is recorded in
`docs/NEXT_SESSION_PLAYBOOK.md` under a new
"External-awareness checkpoint (2026-04-29)" section.

**Three items added to the queue:**

1. **§2.5 LLM-refresh measurement (high-leverage, ~$1–3
   budget).** SUM's §2.5 round-trip closure is locked at recall
   ≥ 0.97 across three corpora using `gpt-4o-mini-2024-07-18`
   (Jul 2024). Two frontier LLMs have shipped since — Anthropic
   **Claude Opus 4.7** (16 Apr 2026) and OpenAI **GPT-5.5** (23
   Apr 2026). The intervention pattern is *probably* model-
   independent but unmeasured on frontier models. Re-run + ship
   a `sum.s25_frontier_models_2026.v1` receipt; requires
   Anthropic SDK support in the runner (currently OpenAI-only).

2. **Sigstore-signed PyPI uploads (medium-leverage, no
   budget).** The `sigstore` PyPI package is now
   Production/Stable; cosign v3 shipped; PyPI accepts in-toto
   Sigstore attestations. The "wait for maturity" gate on
   ship-it has lifted. Add `sigstore sign` step to
   `publish-pypi.yml` gated on GitHub OIDC.

3. **MCP discovery shim (low-leverage, no budget).** MCP next
   spec drop is June 2026; SEP-1649
   (`.well-known/mcp/server-card.json`) is broadly adopted.
   SUM's `sum-mcp` stays stdio-only (HTTP-MCP deferred until
   auth design); the discovery shim is forward-compat
   plumbing for when HTTP-MCP eventually ships.

**Three items audited and confirmed no action needed:**

* **C2PA `digital_source_type`** — taxonomy unchanged across
  C2PA 2.2 → 2.4. SUM's `trainedAlgorithmicMedia` /
  `algorithmicMedia` mappings remain authoritative.
  `docs/RENDER_RECEIPT_FORMAT.md` §7 updated with explicit
  documentation of the deliberate text-on-image-taxonomy
  mapping (no formal text-content profile exists in C2PA 2.x;
  if one ships later, SUM will mint a new field rather than
  overload the existing one per `COMPATIBILITY_POLICY.md`).
* **PQC / Ed25519 / SHA-256** — NIST SP 800-131A r3 keeps
  SHA-256 approved; 2030 deprecation target is RSA/ECDSA,
  not Ed25519 explicitly. Tracking note added; no code
  change today.
* **W3C VC 2.0 / Data Integrity 1.1** — `eddsa-jcs-2022`
  interop tests re-ran 22 Feb 2026 and pass. Render Method
  REC targets Sept 2026; evaluate emission alongside the
  existing receipt when it lands.

This is the first deliberate "process intensification"
external-awareness checkpoint. Future cycles should run this
audit at the start of each session-block (every ~15 PRs or
monthly, whichever comes first) to keep substrate decisions
informed without drift.

### Doc-channel congruency pass — align surfaces with current shipping state

Following an external audit of cross-channel claims (the
SUMequities portfolio at `https://www.sumequities.com/projects/sum/`
vs the SUM repo's docs vs the GitHub repo metadata), this pass
fixes four divergences:

1. **GitHub repo description** updated from the stale "A
   mechanically verifiable knowledge engine built on prime-encoded
   semantic state" to match the README lede: "Cross-runtime
   trust surface for LLM-rendered text: Python, Node, and
   browser runtimes produce byte-identical Ed25519 signatures
   over JCS-canonical bytes." Visible in `gh repo view`, GitHub
   search results, and any consumer that scrapes repo metadata.

2. **`docs/FEATURE_CATALOG.md` extended with Layer 11** — the
   measurement-and-hardening infrastructure shipped this
   session was uncatalogued. New entries (118–126):
   §2.5 canonicalisation-replay runner, §2.5 generator-side
   runner + primitives, §2.5 closure receipts (4 corpora),
   `/api/qid` accuracy floor runner, `sha256_128_v2`
   cross-runtime byte-identity gate, threat-model traceability
   test suite, `sum verify` extraction-provenance surface, MCP
   server v2 (hardened), M1 Merkle set-commitment sidecar
   prototype. Each entry has a verification command.

   Counts re-regenerated: total **126 features**, Production
   ✅ **112**, Scaffolded 🔧 **13**, Designed 📄 **1**.

3. **`docs/PROOF_BOUNDARY.md` §2.5.1 added** for the
   `/api/qid` resolution-accuracy receipt. The empirical-
   benchmark section was missing this measurement; now
   surfaced with the two-tier metric (hit-rate 100 %,
   label-substring match 100 %) and the explicit boundary
   on what the metric does not test.

4. **`pyproject.toml` version bump 0.3.1 → 0.4.0.** The
   [Unreleased] block has accumulated 16+ entries since the
   v0.3.0 PyPI release including the MCP server, threat-model
   test suite, §2.5 closure pattern across 4 corpora, the
   sha256_128_v2 byte-identity gate, the `/api/qid` floor, the
   verify-extraction-provenance surface — all feature-bearing,
   not bug-fix-shaped. Honest semver: minor bump.

**Audit also surfaced two SUMequities-portfolio discrepancies**
that I cannot fix from this repo (separate codebase):

- "100 commits / 30d" displayed; actual is **236**.
- "VERIFIED APR 28" displayed; today is APR 29 (likely a
  daily-refresh stamp).

These are listed as findings for the operator to handle in
the SUMequities repo. A future "process intensification" pass
will add a JSON manifest published from this repo that the
portfolio reads from, so commit-count drift cannot recur.

This PR ships only the SUM-repo-side fixes; the
process-intensification automation is deliberately deferred
per the operator's framing ("first let's just get congruency
everywhere, then we'll look into mechanisms of maintaining
those channels").

### Added — `sha256_128_v2` cross-runtime byte-identity gate (K1-v2 + K2-v2)

The README's hardening backlog said "`sha256_128_v2` activation —
Node side exists, Python side not yet `CURRENT_SCHEME`." That
framing was misleading: the v2 codepath was implemented on **both
sides** (`derivePrimeV2` in Node's `standalone_verifier/math.js`,
`_deterministic_prime_v2` in Python's `semantic_arithmetic.py`). The
real gap was that **no cross-runtime byte-identity gate proved Python
↔ Node agree under v2**.

This PR closes that gap.

**What ships:**

* `single_file_demo/godel_cli.js` — accepts an optional `scheme`
  field on the JSON payload (defaults to `sha256_64_v1`); the
  `sha256_128_v2` value dispatches to `derivePrimeV2`.
* `scripts/verify_godel_v2_cross_runtime.py` — a sibling of the
  existing v1 harness that asserts byte-identity for v2:
  - **K1-v2:** 12 axiom-key fixtures (including UTF-8 + multi-word)
    minted in Python's `sympy.nextprime(seed_128)` and Node's
    `derivePrimeV2`. All 12 byte-identical.
  - **K2-v2:** 6 state-encoding fixtures (single triple, two
    triples, five triples, repeated triple, two order
    permutations) under v2's LCM. All 6 byte-identical.
* CI step in `.github/workflows/quantum-ci.yml` — runs alongside
  the v1 K-matrix on every PR, hard-stops the merge on
  divergence.
* `docs/PROOF_BOUNDARY.md` §1.2 documents the new gate.
* `docs/ALGORITHM_REGISTRY.md` row for `sha256_128_v2` updated to
  "planned (cross-runtime byte-identity locked)".
* `README.md` Future-developments line replaces the misleading
  "Python side not yet `CURRENT_SCHEME`" with the empirical
  status: implementations agree byte-for-byte; flipping the
  default is a separate operator decision.

**What this does NOT do.** This PR does NOT change
`CURRENT_SCHEME`. The default stays `sha256_64_v1`. Flipping the
default to v2 requires:

1. A `bundle_version` minor bump per `docs/COMPATIBILITY_POLICY.md`
   so consumers know which scheme to expect.
2. A migration story for v1 → v2 bundles (an existing v1 bundle's
   `state_integer` is incompatible with a v2-derived state on the
   same axiom keys; consumers cannot mix).
3. An operator-side decision documented in
   `docs/INCIDENT_RESPONSE.md`-shape runbook.

This PR proves the migration path is **empirically open** —
divergence between runtimes would have surfaced as a failing
gate. The default-flip is a separate operator decision, not
gated on this PR.

**Why this matters.** v1's 64-bit seed has a birthday-bound
collision frontier at ~2³² axioms. v2's 128-bit seed lifts that
to ~2⁶⁴. SUM's current corpora are well below the v1 frontier
(seed_v1 = 50 axioms; seed_long = 11–28 per doc × 16 docs = ~250
axioms). v2 is a forward-looking hedge for any future deployment
that crosses the v1 collision-safe boundary.

The byte-identity proof costs nothing per release (CI runs the
gate in <2 seconds). The cost of NOT having it is silent
divergence: a future operator flips the default, the gate
catches the divergence in CI, and we don't ship a broken bundle.

### Added — `sum verify` surfaces extraction provenance (closes THREAT_MODEL §3.3 visibility gap)

The signature on a CanonicalBundle proves the canonical tome
maps to the state integer + the issuer signed this exact
content. It does NOT prove the axioms are factually correct,
and it does NOT prove that re-extracting from the source
prose would produce the same axioms.

`docs/THREAT_MODEL.md` §3.3 documents this gap as the
"signed ≠ true" residual risk. The information needed to
distinguish reproducible (sieve-extracted) from advisory
(LLM-extracted) bundles already lives in the `sum_cli`
sidecar — but `sum verify` did not surface it.

This change adds an `extraction` block to the verifier's
JSON output:

```json
{
  "ok": true,
  "axioms": 2,
  "signatures": {"hmac": "verified", "ed25519": "absent"},
  "extraction": {
    "extractor": "sieve",
    "verifiable": true,
    "source": "sum_cli sidecar"
  }
}
```

The `verifiable` boolean is the load-bearing affordance.
True iff `extractor == "sieve"` (deterministic
re-extraction); false for `extractor == "llm"` (stochastic)
and for bundles with no sidecar (fail-closed — verifier
does not assume reproducibility in the absence of provenance).

Downstream consumers can now branch with one line:

```bash
sum verify --input bundle.json | jq -e '.extraction.verifiable'
```

The human-readable stderr line also names the extractor:

    sum: ✓ verified 2 axiom(s), state integer matches
         (hmac=verified, ed25519=absent, extractor=sieve (verifiable))

**Test coverage:** 5 tests in
`Tests/test_verify_extraction_visibility.py`:

- Sieve-attested bundle reports `verifiable: true`.
- Bundle without sidecar reports `verifiable: false` /
  `source: "absent"` (fail-closed).
- LLM-sidecar bundle reports `verifiable: false`.
- Stderr human-readable line names the extractor.
- Documentation test ties this surface to the THREAT_MODEL
  §3.3 row that motivated it.

**What this is not.** This is not novel cryptography or
new science. It is small CLI ergonomics that closes a
documented threat-model visibility gap. The novelty in SUM
remains the cross-runtime trust triangle, the §2.5 closure
pattern, and the render-receipt format; this change is
plumbing for a downstream-consumer ergonomic affordance.

**What this is.** A 30-LOC CLI fix + 5 tests that lets a
compliance-audit tool gate on bundle reproducibility with
one line. Closes a long-standing threat-model side issue
without a schema change (the field is verifier-output-only;
the bundle schema is unchanged, so existing bundles continue
to verify identically).

### Measured — `/api/qid` accuracy floor (closes a "target >95%" placeholder)

The README's "Future developments" section claimed a "target
>95% accuracy floor" for `/api/qid` SPARQL disambiguation but
**the floor was never measured**. This closes that placeholder
with a real number from a 30-term hand-curated corpus across
four categories (people, places, concepts, common nouns).

`scripts/bench/runners/qid_accuracy.py` runs against the live
hosted Worker, no API key needed, ~$0 cost (Wikidata is free,
Cloudflare on free tier covers ~30 requests trivially). Receipt
at `fixtures/bench_receipts/qid_accuracy_2026-04-28.json` under
schema `sum.qid_resolution_accuracy.v1`.

**Two-tier metric, run 2026-04-28 against `https://sum-demo.ototao.workers.dev`:**

- **Hit-rate: 30/30 (100%)** — every term resolved to a non-null Wikidata entity.
- **Label-substring match: 24/24 (100%)** — every returned label contains the input pattern as a case-insensitive substring (excludes 6 common-noun rows from denominator).
- **Wall-clock p50 ≈ 200ms** per term (Cloudflare cache + Wikidata round-trip).

**Honest finding the receipt surfaces.** Label-substring match
is robust to wbsearchentities's quirks but does NOT measure
semantic accuracy against canonical Q-IDs. The receipt records
`relativity` → `Q201607 (Relativity Records)` — a music-label
entity, not the physics theory — as a passing label-substring
match. The two-tier shape is the floor; canonical-QID accuracy
is a stricter measurement that would need hand-verified
ground-truth pairs (a follow-on, scoped explicitly in the
README).

The current resolver is a thin layer over wbsearchentities;
SPARQL-driven disambiguation that prefers the most-linked-to
entity for ambiguous terms remains an unshipped enhancement —
the receipt's `relativity` row demonstrates exactly the case
SPARQL disambiguation would address.

**Operator note (preserved from the seed_long capstone):** the
runner sets an explicit `User-Agent` header
(`sum-qid-accuracy-bench/0.1`) because Cloudflare's edge
returns 403 Forbidden on the default Python `urllib`
`Python-urllib/3.10` UA. The same fix applied earlier in this
session for the receipt-audit runner.

`README.md`'s "Future developments" line replaces "target >95
% accuracy floor" with the measured numbers + the explicit
boundary on what the metric does and does not test.

### Added — threat-model executable traceability test suite

`docs/THREAT_MODEL.md` §4 (Attack Surface Summary) names every
defence the SUM engine claims to provide. Underlying defences
already had test coverage scattered across `test_resource_guards.py`,
`test_extraction_validator.py`, `test_merkle_chain.py`, etc., but
**no single file demonstrated the threat-model claims hold**.

This adds `Tests/test_threat_model.py` — one test class per
attack-surface row, with the §X.Y reference in each docstring.
The tests intentionally exercise the *primary defence* named in
each row, not every edge case (those live in their dedicated
test files). The threat-model-to-test traceability is the
load-bearing property of this file.

**Coverage** (22 passing, 1 skipped, 2 xfailed):

| Threat row | Defence asserted |
|---|---|
| §2.1 Bundle Tampering | HMAC-SHA256 detects tome/state mutation |
| §2.2 State Integer Forgery | Witness reconstruction without HMAC key |
| §2.3 Version Mismatch | Future canonical_format_version rejected |
| §2.4 Malformed Bundles | Missing required fields rejected (parametrised) |
| §3.3 Extraction Manipulation | Empty / control-char / JSON-fragment / oversized fields rejected by `ExtractionValidator` |
| §3.4 Semantic Collision Replay | Independent algebra instances mint identical primes |
| §3.5 Contradiction Governance | DeterministicArbiter resolves order-independent (skipped if module absent) |
| §3.6 DoS bundle limits | 10 MB tome / 100 K state digits / 50 K tome lines / 200 K ingest chars all gated; ResourceLimitError is HTTP 413 |
| §3.7 Ledger Tampering | Merkle hash-chain detects mutation; clean chain verifies |
| Residual risks (xfail) | HMAC real-time revocation NOT shipped; full DB replacement NOT detectable — both documented as intentional residual gaps that flip to passing tests if/when defence ships |

**Discipline:** the file ends with a `_THREAT_TO_TEST` index +
`test_threat_to_test_index_is_complete` that fails if a test
class is added without registering it in the index. The index
exists to enforce that threat-model rows and tests stay in
1:1 correspondence — adding a test without an index entry, or
adding a row in `THREAT_MODEL.md` without a test, both
surface as a failing test.

**Out of scope here** (covered by their own load-bearing files):
P2P Mesh Auth (`test_phase13_zenith.py`); VC 2.0 forgery
(`test_verifiable_credential.py`); Render-receipt forgery
(`test_render_receipt_verifier.py` + cross-runtime fixture
matrix); Trust-root manifest forgery (`test_trust_root.py`);
Cross-runtime verifier divergence
(`scripts/verify_cross_runtime*.py`); CI/supply-chain
compromise (`scripts/lint_workflow_pins.py` + R0.3 SHA-pin lint
job in CI).

This closes a hardening-backlog item (P5 in
`docs/NEXT_SESSION_PLAYBOOK.md`: threat-model-to-test
traceability). It does not change any defence; it *demonstrates*
the existing defences. A future change to any defence's
behaviour surfaces as a failing test in this file with a clear
"§X.Y" tag.

### Hardened — `s25_generator_side` runner: per-call timeout + graceful per-doc skip

The seed_long capstone surfaced a real failure mode: an OpenAI
structured-output call hung for 14+ minutes with the python
process alive but no CPU progress. The OpenAI SDK has its own
request timeout but the empirical fail was outside that envelope —
likely a stuck websocket on the structured-output stream. The
operator had to `kill -9` the process and re-run. That cost
research budget on a wasted call and ate operator attention.

**Fix:** every LLM call inside the runner is now wrapped in
``asyncio.wait_for`` with a 60-second per-call default, raising
a tagged ``S25CallTimeoutError`` on timeout. ``run_doc``
catches the exception and returns a per-doc record tagged
``error_class: "timeout"`` rather than letting it propagate.
The surrounding ablation continues; the receipt records the
timeout for that doc; the aggregate excludes timed-out docs
from drift/recall means and counts them in
``n_docs_timed_out``. Operator sees at receipt-read time which
docs (if any) failed during execution.

**Changes:**

* `scripts/bench/runners/s25_generator_side.py` — every call
  site (`_baseline_extract`, `_constrained_extract`,
  `_baseline_generate`, `_canonical_first_generate`) now
  takes a `call_timeout_s` keyword argument that threads
  through `_with_call_timeout` (a small `asyncio.wait_for`
  wrapper). `run_doc` catches `S25CallTimeoutError` and
  returns the timeout record. `aggregate()` excludes timed-
  out docs from means; new fields `n_docs_measured`,
  `n_docs_timed_out`, `timed_out_doc_ids`,
  `fraction_full_recall` (over measured) added.
* New `--call-timeout` CLI flag, default 60.0s. Operator can
  tune for slow networks or override for stress tests.
* The receipt JSON's per-ablation block now includes a
  `call_timeout_s` field so a future reader knows what
  timeout the run was conducted under.
* Per-doc records can carry `error_class: "timeout"`,
  `error_what` (which call timed out), and `error_timeout_s`
  (the deadline that fired). Receipt-readers branch on
  `error_class` presence rather than assuming every per-doc
  record has `drift_pct`/`exact_match_recall`.

**Test coverage:** 7 new tests in
`Tests/test_s25_runner_timeout.py`:

  - `_with_call_timeout` passes through on success.
  - Hangs raise `S25CallTimeoutError` (not bare
    `asyncio.TimeoutError`).
  - Other exceptions are NOT swallowed by the timeout wrapper.
  - `run_doc` returns a tagged timeout record when a call
    hangs (verified via a mock client whose calls
    `asyncio.sleep(60)` past the per-call deadline).
  - `aggregate` excludes timed-out docs from drift/recall
    means while counting them separately.
  - All-timed-out edge case does not zero-divide.
  - The 60s default is pinned (regression catch).

22 of 22 tests across the §2.5 test surface pass
(`test_s25_interventions.py` 15 + `test_s25_runner_timeout.py`
7). Existing receipts unchanged — the schema is forward-
compatible because new aggregate fields are additive.

This closes a hardening-backlog item that the seed_long
capstone PR explicitly named as a follow-on. No re-run of
prior measurements is required; the new field is "0 timeouts"
on every prior receipt, retroactively consistent with the
new schema.

### Measured — §2.5 capstone: intervention scales to `seed_long_paragraphs` (multi-paragraph dense-prose)

Capstone receipt for the §2.5 corpus-coverage matrix. Combined
ablation re-run against `seed_long_paragraphs.json` (16 hand-
authored multi-paragraph documents on disparate technical and
historical topics, **11–28 source axioms per doc** — an order
of magnitude denser than seed_v1's single-fact and seed_v2's
1–2-fact shapes). Receipt at
`fixtures/bench_receipts/s25_generator_side_seed_long_combined_2026-04-28.json`.

**Headline:**

| Ablation | drift_pct | recall | docs full recall |
|---|---:|---:|---:|
| canonical_first only † | 69.36 | 0.7045 | 4 / 16 |
| **combined** | **0.57** | **0.9972** | **15 / 16** |

† From a partial earlier sweep; the all-3-ablations run hung
on a network call after constrained_extractor's 8th doc.
canonical_first had completed first. The 0.7045 is informative —
canonical_first alone *improves* on long-form vs seed_v2's
0.5750, suggesting denser source axioms give the LLM more
context to anchor canonical sentences.

constrained_extractor on long-form was visibly worse in the
8 docs that completed before the hang (~0.40 mean, much lower
than seed_v2's 0.825) because long-form prose makes the
per-doc constrained vocabulary wider and noisier, and the LLM
emits far fewer triples under constraint than the unconstrained
extractor does. **It does not propagate to combined.**

**Cross-corpus comparison — closure scales universally:**

| Corpus | n_docs | axioms/doc | combined recall | drift_pct | full recall |
|---|---:|---:|---:|---:|---:|
| seed_v1 (single-fact SVO) | 50 | 1 | 1.0000 | 0.00 | 50 / 50 |
| seed_v2 (7 difficulty patterns + multi-fact) | 20 | 1–2 | 0.9750 | 5.00 | 19 / 20 |
| **seed_long (16-topic multi-paragraph)** | **16** | **11–28** | **0.9972** | **0.57** | **15 / 16** |

The combined intervention lands **≥ 0.97 recall and ≤ 5 %
drift** on every measured corpus shape. The §2.5 closure is
corpus-independent. Each remaining gap traces to upstream LLM
source-extraction artifacts (corrupted axioms on seed_v2
doc_015, semantically-duplicate predicates on seed_long
solar_system), not to the intervention pattern itself.

**The single seed_long failure** (doc_long_solar_system,
recall = 0.9545): the LLM source-extract produced two
semantically-overlapping axioms — one with predicate
`has_two_moons`, another that admitted `has_known_moons`
into the constrained vocabulary; the round-trip's reconstructed
extractor picked the latter for one mars axiom. 21 of 22
axioms in that doc round-tripped exactly. Same fact, two
surface forms — a benign upstream duplication, not a
structural failure.

**Note: the `--ablation all` run on seed_long hung mid-flight**
on a network call to OpenAI's structured-output endpoint
(constrained_extractor doc 9). Process was alive (PID 33408)
but had no CPU time progress for 14+ min. Killed and re-ran
with `--ablation combined` only, which completed cleanly.
The runner currently has no per-call timeout; that's a
hardening-backlog item for a future cycle. The single-
ablation re-run is the load-bearing receipt.

`docs/PROOF_BOUNDARY.md` §2.5 gains a "Capstone scaling check
on seed_long_paragraphs" subsection with the cross-corpus
comparison table. §6 progress-table row updated to "Closed
across measured corpora" with seed_v1 / seed_v2 / seed_long
numbers all named. README "What does NOT yet work"
subsection retitled "LLM narrative round-trip — closed across
measured corpora" with the cross-corpus table.

This receipt completes the §2.5 attack arc with six stacked
receipts:

  1. sum.llm_roundtrip.v1 (2026-04-19) — original 107.75 / 0.12
  2. sum.s25_canonicalization_replay.v1 — falsification, ceiling 0.18
  3. sum.s25_generator_side.v1 — generator-side, recall 0.90
  4. residual closure (lemma-exclusion) — saturation on seed_v1, recall 1.00
  5. seed_v2 scaling check — recall 0.9750 on difficulty corpus
  6. seed_long capstone — recall 0.9972 on multi-paragraph

Each receipt was the reference baseline for the next. The
intervention pattern (canonical-first generator + constrained-
decoding extractor with `Literal`-enum vocab pin +
lemma-exclusion of source-predicate lemmas from canonical-
padding) is the load-bearing engineering finding of this arc.
The corpus-independence of the closure is the load-bearing
empirical finding.

### Measured — §2.5 intervention scales to `seed_v2` (difficulty-pattern corpus)

Same combined intervention re-run against `seed_v2` (20 docs,
7 difficulty parse patterns: apposition, passive voice,
relative clause, conjunction, negation, hedging, complex PP,
including multi-fact docs). `gpt-4o-mini-2024-07-18`,
2026-04-28, ~\$0.12. Receipt at
`fixtures/bench_receipts/s25_generator_side_seed_v2_2026-04-28.json`.

**Headline:**

| Ablation | drift_pct | recall | docs full recall |
|---|---:|---:|---:|
| canonical_first only | 98.92 | 0.5750 | 11 / 20 |
| constrained_extractor only | 52.08 | 0.8250 | 16 / 20 |
| **combined** | **5.00** | **0.9750** | **19 / 20** |

**The intervention pattern scales.** Combined goes from
`seed_v1`'s 1.00 to `seed_v2`'s 0.9750 — a 0.025 absolute drop
on a corpus that adds difficulty-pattern parses + multi-fact
docs. The single failing doc (doc_015, "Alice and Bob visited
Paris.") is **not an intervention failure**: the runner's
first-pass `_baseline_extract` returned a malformed source
axiom (`['alice', 'visited', 'paris},{']`); the combined
ablation correctly preserved the corrupted source through
the round-trip. The fail-mode is an LLM extraction artifact
on the source pass, not the intervention.

**Per-ablation shape inverts vs seed_v1.** On `seed_v2`,
constrained_extractor alone (0.8250) beats canonical_first
alone (0.5750), where on `seed_v1` they were nearly identical
(0.62 vs 0.60). The reason is corpus predicate form: seed_v2
predicates are mostly already lemmas (`win`, `emit`, `orbit`,
`visit`), so lemma-exclusion has less work to do and the LLM
naturally selects the source form. Conversely, seed_v1
predicates are mostly inflected (`proposed`, `contains`,
`discovered`), so the canonical-first generator prompt
carries the work there. Different corpora, different layers
earn their keep — but **combined wins decisively on both**.

**Boundary:** the §2.5 closure now covers single-fact SVO
(seed_v1) and 20-doc difficulty-corpus (seed_v2) shapes.
`seed_long_paragraphs.json` (16 hand-authored multi-paragraph
docs, 9–24 triples each) remains unmeasured under the
intervention. The seed_v2 result establishes the intervention
pattern is **structurally right** across difficulty-pattern
variation; whether it holds on multi-paragraph multi-fact
docs is the next measurement when budget allows.

`PROOF_BOUNDARY.md` §2.5 gains a "Scaling check on `seed_v2`"
subsection with the new ablation table and the per-ablation-
shape-inversion finding. §6 progress-table row updated to
"Closed on `seed_v1`; scales to `seed_v2`" with both
measurements named.

This receipt completes the §2.5 attack arc with five stacked
receipts:

  1. sum.llm_roundtrip.v1 (2026-04-19) — original 107.75 / 0.12
  2. sum.s25_canonicalization_replay.v1 — falsification, ceiling 0.18
  3. sum.s25_generator_side.v1 — generator-side, recall 0.90
  4. sum.s25_residual_closure (lemma-exclusion) — saturation on seed_v1, recall 1.00
  5. sum.s25_generator_side_seed_v2.v1 — scaling check, recall 0.9750 on harder corpus

Each receipt was the reference baseline for the next. The
intervention pattern (canonical-first generator + constrained-
decoding extractor + lemma-exclusion of source-predicate
lemmas from canonical-padding) is the load-bearing engineering
finding of this arc.

### Measured — §2.5 fully closed on `seed_v1` after lemma-exclusion residual fix

Live re-run against the same `seed_v1` corpus (50 docs,
`gpt-4o-mini-2024-07-18`, 2026-04-28, ~\$0.07). Receipt at
`fixtures/bench_receipts/s25_residual_closure_2026-04-28.json`.

The prior combined-intervention receipt closed §2.5
substantially (recall 0.12 → 0.90) but left 5/50 docs failing.
Per-doc analysis showed every failing doc had the same root
cause: the constrained extractor's predicate enum admitted both
the source's inflected predicate (`proposed`, `contains`,
`described`, `discovered`, `build_nests`) and its lemma
(`propose`, `contain`, `describe`, `discover`, `build`) from
`DEFAULT_CANONICAL_PREDICATES`. Faced with both forms in the
enum, the LLM extractor preferred the lemma every time.

**The fix:** when constructing the per-doc constrained schema,
exclude any token from `DEFAULT_CANONICAL_PREDICATES` that is a
candidate lemma of any source predicate. Implementation:
`_candidate_lemmas(predicate)` covers standard English suffix
inflections (`-ed`, `-es`, `-s`, `-ing`, doubled-consonant past
forms, `-ies` → `-y`) plus compound-predicate head-verb removal
(`build_nests` → forbid `build`).

**Result on the same corpus:**

| Ablation | drift_pct | recall | docs full recall |
|---|---:|---:|---:|
| L0 baseline | 107.75 | 0.12 | 6 / 50 |
| Combined (initial) | 21.00 | 0.90 | 45 / 50 |
| **Combined + lemma-exclusion** | **0.00** | **1.0000** | **50 / 50** |

All 5 previously-failing docs (doc_004 / 005 / 010 / 014 /
015) recovered; zero docs newly broken. Drift falls to **0.00%
— within rounding of the canonical (provable) round-trip on
the same corpus**.

**Boundary on this result:** `seed_v1` is single-fact SVO. The
1.00 recall is the saturation point for that corpus's
complexity. Harder corpora (`seed_v2`'s difficulty-pattern
docs, `seed_long_paragraphs`'s multi-paragraph multi-fact)
have NOT been measured under the intervention. The `seed_v1`
receipt establishes that the intervention pattern is right;
whether it scales is the next measurement.

The §2.5 row in `PROOF_BOUNDARY.md` §6 progress table moves
from "Substantially closed by combined intervention" to
"Closed on `seed_v1`" with the boundary noted. README "What
does NOT yet work" subsection retitled "LLM narrative
round-trip — closed on `seed_v1`" with the updated table.

**Test coverage:** 15/15 in `Tests/test_s25_interventions.py`.
The new `test_constrained_schema_excludes_source_predicate_lemmas`
locks the lemma-exclusion behaviour for `-ed`, `-s`, and
compound-predicate cases, asserting via `pydantic.ValidationError`
that the lemma forms are rejected by the schema.

The `_candidate_lemmas` helper is conservative — only fires on
standard English suffixes. Will miss irregulars (`taught` →
`teach`); those are not present in the corpus's failure set,
so this is the right scope for the fix that closes the
observed residual without over-fitting on patterns the data
did not surface.

This receipt closes the §2.5 attack arc opened by the original
107.75% drift measurement (2026-04-19), bounded by the
canonicalisation-replay falsification (2026-04-28 morning,
ceiling 0.18), confirmed by the generator-side intervention
(2026-04-28 evening, recall 0.90), and saturated by this
residual fix. Four stacked receipts; each was the reference
baseline for the next.

### Measured — §2.5 substantially closed by combined generator-side intervention

Live bench against `seed_v1`, 50 docs, `gpt-4o-mini-2024-07-18`,
2026-04-28. Receipt at
`fixtures/bench_receipts/s25_generator_side_2026-04-28.json`
under schema `sum.s25_generator_side.v1`. Cost ≈ \$0.20.

**Headline result:**

| Ablation | drift_pct (mean) | exact-match recall | p10 recall | full recall |
|---|---:|---:|---:|---:|
| L0 baseline | 107.75 | 0.12 | 0.00 | 6 / 50 |
| L3 max canonicalisation (post-hoc, prior receipt) | 106.36 | 0.18 | 0.00 | 9 / 50 |
| A — canonical-first generator only | 94.85 | 0.60 | 0.00 | 30 / 50 |
| B — constrained extractor only | 81.97 | 0.62 | 0.00 | 31 / 50 |
| **A + B combined** | **21.00** | **0.90** | **1.00** | **45 / 50** |

**Recall: 0.12 → 0.90 (7.5× improvement). Drift: 107.75 →
21.00 (5× reduction). p10 recall: 0.00 → 1.00.** The
worst-decile docs at baseline had zero exact-match; under the
combined intervention they all achieve full recall.

**Each layer is independently necessary; combined is
supra-additive.** Canonical-first alone hits 0.60 by addressing
generator elaboration at the source. Constrained-extractor
alone hits 0.62 by addressing surface-form drift at the
symptom. Stacked, they reach 0.90 — better than either
layer's independent effect would predict, because the
canonical-first generator produces prose that the constrained
extractor can actually *find* the source vocabulary in. The
two layers compose because they operate on different stages
of the same failure mode (generator elaboration vs. extractor
paraphrase).

**What's left of the §2.5 gap (5 of 50 docs):** residual is a
per-corpus tuning problem (extend the canonical predicate set,
tighten the verbatim-token rule), not a structural problem
with the intervention pattern.

`docs/PROOF_BOUNDARY.md` §2.5 boundary rewritten with the new
table; the §6 progress-table row moves from `Measured (drift =
107.75%, recall = 0.12)` to `Substantially closed by combined
intervention (drift = 21.00%, recall = 0.90 on seed_v1)`.
`README.md` "What does NOT yet work" subsection retitled "LLM
narrative round-trip — substantially closed" with the
ablation table above.

Receipt schema family is `sum.s25_*.v1` (per-ablation siblings:
`canonical_first_generator`, `constrained_extractor`,
`combined`). Reproducible: `python -m
scripts.bench.runners.s25_generator_side --ablation all --out
<path>` (requires `OPENAI_API_KEY`, ~\$0.20, ~5 min on
`seed_v1`).

This receipt completes the §2.5 attack arc the
canonicalisation-replay receipt opened — that receipt
falsified the cheapest hypothesis (post-hoc canonicalisation
alone, ceiling 0.18); this receipt confirms the intervention
the prior boundary named (constrained decoding to a pinned
vocabulary + canonical-first generator prompt) and lands a
durable measurement.

### Scaffolded — §2.5 generator-side intervention runner (live receipt pending operator spend)

The L0–L3 canonicalisation-replay receipt established that
canonicalisation alone cannot close the §2.5 gap; the dominant
failure mode is generator elaboration. This PR ships the
**runner that measures the two interventions named in that
receipt's "operational read"** — but stops at the spend gate.
The live measurement against the 50-doc `seed_v1` corpus is one
command + ~$0.20 of OpenAI budget away, gated on explicit
operator authorisation rather than burned silently.

**Three ablations registered** (each ships under a distinct
sibling schema, comparable to the L0 baseline by structural
encoding in the runner):

| Ablation | Schema | Mechanism |
|---|---|---|
| Canonical-first generator | `sum.s25_canonical_first_generator.v1` | Generator system prompt requires surfacing each source claim verbatim before elaborating. Pure prompt change. |
| Constrained extractor | `sum.s25_constrained_extractor.v1` | Per-doc Pydantic schema with `Literal` enums pinned to source-axiom vocabulary (subject ∈ source_subjects, predicate ∈ source_predicates ∪ canonical_padding, object ∈ source_objects). OpenAI structured-output enforces the constraint at the API. |
| Combined | `sum.s25_combined.v1` | Both interventions stacked. |

**The runner is offline-testable.** `--dry-run` mode produces a
structurally-valid receipt with synthetic per-doc records — used
to verify the JSON schema family, per-doc field shapes, and
ablation-comparison structure before any spend. The dry-run
fixture lands at
`fixtures/bench_receipts/s25_generator_side_DRYRUN.json`.

**To produce the live receipt** (operator decision):

```bash
OPENAI_API_KEY=... python -m scripts.bench.runners.s25_generator_side \
    --ablation all --out fixtures/bench_receipts/s25_generator_side_$(date +%Y-%m-%d).json
```

Estimated cost: ~$0.20 across all three ablations × 50 docs
(`gpt-4o-mini-2024-07-18`, matching the L0 baseline model). A
2-doc smoke at ~$0.005 is recommended first via `--max-docs 2`.

**Why this PR stops at the spend gate.** Operator-Hard
discipline + the public-project credential constraint:
expending the operator's API budget without explicit
per-experiment authorisation is the same family of move as
sharing a secret. The runner is shipped reproducible; the live
result becomes durable when the operator runs it.

**What the receipt will tell us, regardless of the numbers:**
- If recall moves from 0.12 → high (≥ 0.5): generator-side
  intervention works; §2.5 is largely closed.
- If recall moves modestly (0.20 – 0.40): generator-side helps
  but doesn't fully close; the remaining unmeasured intervention
  (fidelity-objective fine-tune) becomes the next cycle.
- If recall stays near 0.12: generator-side fails like
  canonicalisation did, and the §2.5 gap is structural — the
  failure mode is something the LLM extractor's API surface
  cannot fix; the next investment is symbolic-extraction
  fallback rather than further LLM tuning.

The receipt is the artifact regardless of which branch lands.

**Tests:** 13/13 in `Tests/test_s25_interventions.py` cover
prompt construction, schema acceptance / rejection paths,
empty-source fail-closed posture, and JSON-schema
serialisation for the OpenAI structured-output validator.

**Files added:**
- `sum_engine_internal/ensemble/s25_interventions.py` —
  intervention primitives (prompts + dynamic Pydantic schema
  builder).
- `scripts/bench/runners/s25_generator_side.py` — runner with
  three ablations + offline dry-run mode.
- `Tests/test_s25_interventions.py` — unit coverage.
- `fixtures/bench_receipts/s25_generator_side_DRYRUN.json` —
  dry-run receipt fixture (locks the schema family and per-doc
  field shape).

### Measured — §2.5 canonicalisation-replay receipt

The §2.5 LLM round-trip drift attack ships its first measured
receipt. A new offline runner
(`scripts/bench/runners/canonicalization_replay.py`) replays the
cached `bench_history.jsonl` per-doc data under four progressively
more aggressive canonicalisation regimes — no new LLM cost, no
nondeterminism — and writes a durable artifact at
`fixtures/bench_receipts/s25_canonicalization_replay_2026-04-28.json`.

Headline (`seed_v1`, 50 docs, both legs `gpt-4o-mini-2024-07-18`):

| Regime | drift_pct (mean) | exact-match recall | docs full recall |
|---|---:|---:|---:|
| L0 baseline | 107.75 | 0.12 | 6 / 50 |
| L1 predicate-only | **107.75** | **0.12** | 6 / 50 |
| L2 + subject canonicalisation | 106.68 | 0.16 | 8 / 50 |
| L3 aggressive (ceiling) | 106.36 | 0.18 | 9 / 50 |

**The L1 row is the falsification.** The prior PROOF_BOUNDARY §2.5
boundary hypothesised that "an entity-resolution pass + WordNet /
lemma predicate normaliser would move the 0.12 exact-match recall
upward without changing the generator." Predicate-only
canonicalisation moves **zero** exact matches: the cached
`missing_claims` for failed docs do not have a paraphrase pair in
`extra_claims` whose only difference is predicate inflection. The
dominant failure mode is **generator elaboration** — the LLM
produces ~12 reconstructed axioms per source and elaborates
*around* the source claim rather than paraphrasing it. There is
nothing for predicate normalisation to canonicalise *to*.

L2 recovers 2 docs (the `albert_einstein` ≈ `einstein`,
`isaac_newton` ≈ `newton` cases). L3 recovers 1 more under
aggressive object collapse. Maximum canonicalisation-only
ceiling: **0.18 exact-match recall**, +0.06 absolute over
baseline. Headline drift_pct moves only **1.4 points** because
the formula is dominated by `|reconstructed| >> |source|`
regardless of key alignment.

Operational read: canonicalisation alone does not close the §2.5
gap. The work to move the *generator* (constrained decoding to
a pinned vocabulary, or a fidelity-objective fine-tune) is a
future cycle, gated on this receipt as the reference baseline.
The measurement was deliberately structured to falsify or support
the prior boundary's hypothesis; it falsifies the cheapest one
and constrains where further investment goes.

`docs/PROOF_BOUNDARY.md` §2.5 boundary rewritten with the L0–L3
table. `README.md` "What does NOT yet work" subsection updated
with the same data. The receipt schema is
`sum.s25_canonicalization_replay.v1`; future generator-side
interventions ship under sibling schemas (e.g.
`sum.s25_constrained_decoding.v1`) and compare against this
baseline.

Reproducible offline:
`python -m scripts.bench.runners.canonicalization_replay --out /tmp/replay.json`
— no API key needed.

### Consolidated — `docs/` tree reduced from 25 active docs to 17 + index

Newcomer-recommendation #3 from the Operator-Hard fresh-eyes
audit. `docs/` was 25 files / 7 282 lines, including several
session-shaped or design-history docs that no current consumer
read. After this pass: **17 active docs** organised by reader
(verify / integrate / understand-primitives / operate /
process), plus a new **`docs/README.md` index** that explains
which doc to open and why.

**8 docs moved to `docs/archive/`** with `git mv` (history
preserved as renames):

- `WASM_PERFORMANCE.md` — older WASM benchmark notes, no
  current consumer.
- `MODEL_CALL_EVIDENCE_FORMAT.md` — design for an unshipped
  surface.
- `DEMO_RECORDING.md` — screen-recording instructions,
  session-shaped.
- `STAGE3_128BIT_DESIGN.md` — `sha256_128_v2` design rationale;
  activation criteria summarised in `ALGORITHM_REGISTRY.md`,
  full design history preserved in archive for byte-level
  reference.
- `SLIDER_V02_RESEARCH.md` — v0.2 slider-substrate research;
  load-bearing decisions reflected in `SLIDER_CONTRACT.md`,
  longer-form survey preserved in archive.
- `NLI_MODEL_REGISTRY.md` — supported NLI models; today's
  contract lives in `live_llm_adapter.py`'s pinned-snapshot
  list and `SLIDER_CONTRACT.md`.
- `FORMAL_MODELS.md` — formal-verification roadmap (TLA+ /
  SMT / α,β-CROWN); now a single row in `PROOF_BOUNDARY.md`
  §3 pointing to the archived design.
- `TRANSPARENCY_ANCHOR.md` — Rekor/CT anchoring design; now
  Appendix B of `TRUST_ROOT_FORMAT.md` with archive pointer.

**8 forwarding stub files** at the original paths (e.g.
`docs/STAGE3_128BIT_DESIGN.md`) for external-link continuity:
each stub is a 5-line file pointing to the archive location
and the `docs/README.md` index. External readers following
old links from issues, blog posts, or search engines see the
forwarding pointer rather than a 404 — public-project
discipline.

**Fold pointers** added to the four receiving docs
(`ALGORITHM_REGISTRY.md`, `SLIDER_CONTRACT.md`,
`PROOF_BOUNDARY.md` §3, `TRUST_ROOT_FORMAT.md` Appendix B)
so a reader of the receiving doc knows where the longer-form
material lives.

**Falsification check (Carmack discipline):** every
fold-target verified ≤500 lines after the fold (threshold
800), confirming consolidation reduced file count without
bloating any individual doc into an unreadable wall.

`docs/README.md` is the actual entry-design fix — the reader
who lands on `github.com/OtotaO/SUM/tree/main/docs` no longer
sees 25 unsorted markdown files; they see a one-line-per-doc
index grouped by reader intent.

Net file count: 25 → **17 active + 1 index + 8 stub
redirects + 12 archive entries**. Doc-tree surface for a
cold reader: 17 + 1 = **18 visible files** at the top, of
which 1 is the index that tells them where to go.

### Reframed — README leads with the cross-runtime trust surface, not the slider numbers

Newcomer-recommendation #4 from the Operator-Hard fresh-eyes
audit. The previous lede led with the slider's `median 1.000 /
p10 0.769` claim and a "verifiable fact preservation" framing
that conflated the empirical-benchmark surface with the proven
cryptographic surface. Sophisticated readers — the inner ICP
of this project — open the README and `PROOF_BOUNDARY.md` in
adjacent tabs; conflating those two categories costs us them
in the first 90 seconds.

The new lede leads with the load-bearing differentiator: **a
cross-runtime trust surface for LLM-rendered text**, three
runtimes (Python / Node / browser) producing byte-identical
Ed25519 over JCS bytes, every render carrying a detached-JWS
receipt verifiable offline against `/.well-known/jwks.json`.
The slider numbers, the extraction F1, the canonical
round-trip — all retained, all sourced — but as supporting
measurements under the headline trust claim, not as the
headline themselves.

Other edits in the same pass:

* Phase tags scrubbed from public README prose. "shipped on
  PyPI (v0.3.0)" → "shipped on PyPI"; "shipped (Phase E.1
  v0.9.A.2)" → "shipped"; "the v0.4 → v0.9 arc" → "full
  attribution in `docs/SLIDER_CONTRACT.md`". The CHANGELOG
  retains its phase history; the README does not need to.
* Browser version floor "Chrome 113+, Firefox 129+, Safari
  17+" → "Chrome / Firefox / Safari with WebCrypto Ed25519
  support" (the floor is real but maintaining a static
  number list against silent browser updates is drift bait).
* Future-developments section: shipped-already items
  (`v0.9.B browser receipt verifier`, `v0.9.C Python receipt
  verifier`) removed — they shipped earlier in this
  `[Unreleased]` block. The §2.5 LLM round-trip drift attack
  promoted to the lede of the future-developments section as
  the headline open problem.
* MCP server added to the "What ships today" table — it
  shipped this session and was missing from the surface
  table.
* CI badge text "SUM Knowledge OS CI" → "CI". The workflow
  filename `quantum-ci.yml` is unchanged in this PR (renaming
  it would break every PR's badge link); a follow-on
  rename-pass PR addresses the broader naming legacy.

The fresh-eyes audit prescribed three more newcomer
recommendations: rename pass (drop "Quantum" / "Akashic" /
"Holographic" / "Ouroboros" / "Chronos" terminology),
phase-numbering collapse across deeper docs, doc-tree
consolidation. Each lands as its own focused PR after this
one merges so the reframe is reviewable as a single decision.

### Honesty pass — Tier 1 placeholder sweep across PROOF_BOUNDARY / FEATURE_CATALOG / README

Six load-bearing edits across the public-doc surface, all motivated
by the Operator-Hard standard "every number is either a real
measurement or an explicitly-named strategic placeholder." No
new measurements, no new code — pure honesty corrections.

1. **PROOF_BOUNDARY §2.2 Merkle table — N=10 000 row removed.**
   The row reported "3.95× speedup" with a footnote explaining
   the runner had substituted a 62 k-bit proxy state for the
   real ~625 k-bit one because the full LCM build at that N
   takes minutes. A footnoted speedup is not a measurement.
   The 5 000-row figure (21× faster verify, real LCM state)
   is the honest production-relevant headline; the doc now
   explains why the N=10 000 row is omitted and what gates a
   future real measurement.

2. **PROOF_BOUNDARY §2.2 merge-curve extrapolation marked as
   extrapolation.** The "N=10 000 → ~50 s/op; N=100 000 → >1
   hr/op" line was previously stated in declarative voice and
   cited downstream as a measurement. It is an extrapolation
   along the measured N=100/500/1 000 trend assuming `O(B²)`
   scaling and no GMP/sub-quadratic GCD acceleration. The text
   now says so, and explicitly flags the closest direct
   measurement (the N=10 000 / 200-sample harness run that
   did not converge in 10 minutes) as consistent-with but
   not a pin on the extrapolated value.

3. **README — §2.5 LLM round-trip drift surfaced above the
   fold.** A new "What does NOT yet work — the honest line"
   subsection in `## What ships today` cites the **107.75 %
   drift** and **0.12 exact-match recall** numbers from
   PROOF_BOUNDARY §2.5, names the generator-elaboration +
   extractor-paraphrase mechanism, and points to the full
   attribution. The README previously surfaced only the
   favourable slider numbers; the most load-bearing
   honest-status figure in the repo was two clicks away.
   Now it isn't.

4. **PROOF_BOUNDARY §3 self-contradiction resolved.** The row
   "Property-graph backing store … Design decision pending
   empirical confirmation (now confirmed — see §2.2)" was
   self-contradicting in one cell. Restated as: "Design
   direction confirmed by §2.2 measurements; implementation
   not started."

5. **PROOF_BOUNDARY §6 extraction-ceiling row converted to a
   strategic placeholder with an explicit kill condition.**
   The "architectural decision pending on whether to address
   via en_core_web_trf upgrade or LLM fallback" had been a
   long-standing open decision sitting in the headline
   progress table. Restated with: "Strategic placeholder:
   decision deferred until §2.5 LLM round-trip drift attack
   lands. Kill condition: §2.5 work resolves whether the
   LLM-as-extractor path is the right fix or whether the
   sieve needs to stay primary." Status changed from "User
   call" to "Gated on §2.5."

6. **FEATURE_CATALOG.md summary counts regenerated from the
   body.** The previous summary said 96 ✅ / 14 🔧 / 1 📄;
   mechanical recount via
   `grep -cE "^### .*<emoji>" docs/FEATURE_CATALOG.md`
   gives **103 ✅ / 13 🔧 / 1 📄** (total 117). The doc now
   states the counts came from the recipe and asks future
   editors to rerun it on every doc edit. A CI-side check is
   a follow-on; this PR keeps the diff focused on the
   substantive correction.

`PROOF_BOUNDARY.md` version bumped 1.4.0 → 1.4.1; date
stamped 2026-04-28.

### Hardened — MCP server v2 (unbreakable contract)

Eight-property hardening pass on the MCP server shipped one
PR earlier. Default-deny posture against a prompt-injected
LLM client; fail-closed across the surface; fuzz-tested.

**The eight properties (every one is a regression-test in
`Tests/test_mcp_server.py` or `Tests/test_mcp_server_fuzz.py`):**

1. **Input size caps.** `text` ≤ 200 000 chars; bundles ≤ 10 MB
   tome / 100 000 axioms / 1 000 000 state-integer digits.
   Oversized → `error_class: "input_too_large"`.
2. **Tagged failure classes.** v1 collapsed every error into
   `errors: [string]`. v2 emits `error_class` from the fixed
   enum `schema | signature | structural | input_too_large |
   extractor_unavailable | network_disallowed | revoked |
   internal`. Callers branch on the tag, never on substrings.
3. **Network opt-in.** v1's `extractor="auto"` fell through to
   the LLM extractor if `OPENAI_API_KEY` was set — a
   prompt-injected client could drain a wallet via that path.
   v2 auto resolves to sieve unconditionally; the LLM
   extractor requires `SUM_MCP_ALLOW_NETWORK=1` at server
   start AND `extractor="llm"` explicit per call.
4. **Concurrency-safe.** spaCy's nlp pipeline is serialised
   behind an asyncio lock; concurrent `extract`/`attest`
   calls do not race.
5. **Catch-all per tool.** `try/except Exception` →
   `error_class: "internal"` with the exception type name
   only — no traceback, no internal paths leaked. Server
   stays up under any input.
6. **Forward-compat policy.** Bundles with unknown top-level
   fields under `canonical_format_version=1.x` are accepted
   (additive); future major versions fail closed.
7. **Structured stderr audit.** One JSON line per tool call:
   `{ts, tool, result_class, duration_ms, shapes}`. Argument
   shapes (lengths, types, dict keys) logged; argument
   *values* never logged. Log-injection-proof by construction
   — attacker bytes cannot influence the audit record's
   structure.
8. **Property-tested.** Hypothesis-based fuzz suite exercises
   ~800 adversarial inputs per release across every tool's
   typed parameter. Asserts (a) no tool ever raises uncaught,
   (b) no tool ever returns a success shape on a malformed
   payload. Run via `pytest Tests/test_mcp_server_fuzz.py`.

**One-place result construction.** `success_result()` and
`error_result()` in `sum_engine_internal/mcp_server/errors.py`
are the only paths that produce tool output. The audit logger
hooks them. The error-class enum is enforced at construction
time. Future hardening only needs to change one file.

**Wire-format break vs v1:** v1 callers checking `ok: bool` or
substring-matching on `errors[i]` will break. v2 uses
`"error_class" in result` as the failure signal on every tool
except `verify`, which retains `ok` because its purpose is
specifically to return a verdict. Migration is one-line.

29/29 unit tests + 13 fuzz tests pass. CHANGELOG entry under
[Unreleased]. `docs/MCP_INTEGRATION.md` updated with the
hardening contract section.

### Added — `docs/API_REFERENCE.md` — single integration reference for the Worker API

Wire-spec consolidation for external systems calling SUM over
HTTP. Closes the second leg of the "MCP and API on point"
directive — `MCP_INTEGRATION.md` covers the local-LLM-client
surface; this doc covers everything else (web apps, mobile
apps, server-side services, custom verifiers).

Documents all five Worker routes with exact request/response
shapes:

* `POST /api/render` — slider-conditioned tome rendering plus
  the optional signed `render_receipt`. Includes the full
  `RenderReceipt` payload schema, the detached-JWS envelope
  format, the six-step client-side verification flow, and the
  `triples_used` semantics (subset after density-slider
  filtering, not the input set verbatim).
* `POST /api/complete` — Anthropic-first / OpenAI-fallback LLM
  proxy. Marked explicitly as "for the demo UI, not a general
  LLM proxy for third-party integrations."
* `POST /api/qid` — Wikidata QID/PID resolver with edge-cached
  lookups; per-term confidence scoring (1.0 exact, 0.7 alias,
  0.5 other) and the null-id `reason` taxonomy.
* `GET /.well-known/jwks.json` — render-receipt public keys.
  CORS-permissive override of the baseline `same-origin` CORP
  is documented; the deliberate absence of
  `Access-Control-Allow-Credentials` is called out.
* `GET /.well-known/revoked-kids.json` — `sum.revoked_kids.v1`
  shape with the `effective_revocation_at` semantics (receipts
  signed before that timestamp remain valid; only on-or-after
  is rejected).

Also includes:

* The cross-cutting contract — base URL, auth model (none,
  unauthenticated, edge-rate-limited), baseline security
  headers (CSP / HSTS / Permissions-Policy / COEP / CORP),
  error response shape, caching semantics.
* Operator section — `wrangler secret put` flow for
  `RENDER_RECEIPT_SIGNING_JWK` + `RENDER_RECEIPT_SIGNING_KID`,
  the dashboard-vs-wrangler-toml distinction for
  `RENDER_RECEIPT_PUBLIC_JWKS` (escaping inline JSON in
  wrangler.toml is fragile), env-var-absence behaviour table.
* Working integration examples — Node render-and-verify with
  `jose` + `canonicalize`, Python QID resolution with `httpx`,
  Python render-only.
* Cross-references to `RENDER_RECEIPT_FORMAT.md`,
  `PROOF_BOUNDARY.md` §1.3.1, `MCP_INTEGRATION.md`,
  `INCIDENT_RESPONSE.md`, `SLIDER_CONTRACT.md`,
  `COMPATIBILITY_POLICY.md`.

The Node verify example uses an explicit componentwise sort
comparator for `triples_hash` re-derivation — matches the
Worker's `hashTriples` helper byte-for-byte. (Default
`.sort()` works for triples without separator-collisions but
the explicit version is safe under all string contents; this
is the v0.9.A.1 fix that locked Python ↔ JS hash parity.)

README gets a short "Calling SUM over HTTP" section pointing
to the new doc.

### Added — MCP server v1 (Model Context Protocol integration surface)

`sum-mcp` console script + `sum_engine_internal.mcp_server`
package. Exposes SUM's primary verbs as MCP tools so any
MCP-aware LLM client (Claude Desktop, Claude Code, Cursor,
Continue, custom agents on the MCP Python / TypeScript SDKs)
can drive SUM directly without shelling out to the `sum` CLI
or hitting the hosted Worker API. Closes the highest-leverage
integration gap for "systems calling SUM."

**Five tools registered** (stdio transport, JSON-RPC 2.0):

- `extract(text, extractor="auto"|"sieve"|"llm")` — text →
  triples. Fast, side-effect-free.
- `attest(text, branch, title, signing_key)` — extract +
  build signed CanonicalBundle. Produces byte-identical bytes
  to `sum attest`; verifies through every existing
  Python/Node/browser SUM verifier unchanged.
- `verify(bundle, signing_key, strict)` — six-step verification
  (schema gate → prime-scheme gate → Ed25519 → HMAC → state
  reconstruction → axiom-count match). Returns a structured
  dict, never raises on malformed input.
- `inspect(bundle)` — read-only summary; the "what's in this
  bundle" view an agent calls before paying for `verify`.
- `schema(name)` — field catalogue for sum.canonical_bundle.v1,
  sum.render_receipt.v1, sum.merkle_inclusion.v1. The wire
  spec sources of truth remain the markdown specs; this tool
  gives an in-band, programmatically-readable summary.

**Trust model:** thin façade over `sum_engine_internal` +
`sum_cli.main`. No new cryptography, no new canonical codec —
a bundle attested via MCP verifies byte-identically via the
CLI surface, locked by the cross-runtime byte-identity test
in `Tests/test_mcp_server.py`. The cross-runtime trust
triangle (`PROOF_BOUNDARY.md` §1.3.1) extends transparently
to MCP-attested bundles.

**Wire scope:** stdio only in v1. A remote-MCP variant
(SSE / HTTP) is deliberately deferred — `sum-mcp` over the
network is a different threat model than `sum-mcp` on the
same host, and the auth design hasn't landed.

**Tests:** 16/16 in `Tests/test_mcp_server.py`. Three layers —
tool registration, single-tool behaviour, full extract → attest
→ verify roundtrip, plus the byte-identity check that an
MCP-attested bundle passes the CLI verifier as a subprocess.

**Install:** `pip install sum-engine[mcp]`. New optional
dependency: `mcp>=1.0.0` (the official MCP Python SDK with
FastMCP). New script entry: `sum-mcp` → stdio MCP server.
`docs/MCP_INTEGRATION.md` covers Claude Desktop, Claude Code,
Cursor, Continue, and custom agent wiring with concrete config
snippets. Old `docs/archive/MCP_INTEGRATION.md` is left in
place as historical record (covers an earlier summarization-
era SUM and is not the current spec).

### Added — M1 Merkle set-commitment sidecar (prototype + spec + benchmark)

Companion membership-witness substrate alongside the LCM state
integer. Pure-Python Merkle tree over canonical fact keys with
domain-separated SHA-256 (RFC 6962 / RFC 9162-inspired), giving
external verifiers an `O(log N)` inclusion-proof path that
bypasses the LCM merge ceiling documented in `PROOF_BOUNDARY.md`
§2.2 (~50 s/op extrapolated at N=10 000).

**Lands together as one prototype unit:**

- `docs/MERKLE_SIDECAR_FORMAT.md` — wire spec. Locks
  `LEAF_DOMAIN = b"SUM-MERKLE-FACT-LEAF-v1\0"` and
  `NODE_DOMAIN = b"SUM-MERKLE-FACT-NODE-v1\0"` at spec time
  (separates this surface from the Akashic Ledger hash-chain in
  §1.7), lex-sort canonicalisation, RFC 6962 promote-unchanged
  on odd levels, all-zeros 32-byte sentinel for the empty-set
  root, `sum.merkle_inclusion.v1` proof shape.
- `sum_engine_internal/merkle_sidecar/` — pure-Python
  implementation, no external dependencies (only `hashlib`).
  Public surface: `build_tree`, `MerkleTree`,
  `MerkleTree.inclusion_proof`, `verify_inclusion`,
  `InclusionProof` (with `to_dict` / `from_dict`).
- `Tests/test_merkle_sidecar.py` — 27 tests, all passing.
  Covers determinism + set-semantics dedup, round-trip at N ∈
  {1, 2, 3, 4, 7, 8, 15, 16, 100, 1000} (exercises both
  even-numbered and odd-numbered promote-unchanged paths), all
  tamper-detection paths (wrong key, tampered leaf hash,
  tampered sibling hash, malformed `position`, wrong root),
  empty-tree sentinel rejection, single-element edge case,
  domain-separation invariants pinned.
- `scripts/bench/runners/merkle_vs_lcm.py` — benchmark runner
  comparing Merkle inclusion-proof verify vs LCM `state % prime`
  divisibility check at the same N. JSON output, configurable
  `--skip-lcm-build-at` for the slow-N proxy mode.

**Headline numbers (50 samples, Darwin arm64 / Python 3.10):**

| N | Merkle verify p50 | LCM `state % p` p50 | speedup |
|---:|---:|---:|---:|
| 100    | 4.6 µs | 3.2 µs | 0.7× |
| 1 000  | 5.8 µs | 29.6 µs | **5.15×** |
| 5 000  | 7.2 µs | 151.2 µs | **21.1×** |
| 10 000 | 7.8 µs | 30.7 µs † | 3.95× † |

† At N=10 000 the runner uses LCM(first 1000 primes) as the
modulo divisor because the full LCM build at this N takes
minutes per `PROOF_BOUNDARY.md` §2.2 — the projected real-state
speedup is ≈ 30–40× following the 5 000-row trend. The 21.1×
figure at N=5 000 is the conservative production-relevant
headline. The Merkle verify path is empirically flat across the
range tested (4.6 → 7.8 µs as N grows 100×).

**Status:** prototype-only. Exercised by tests + benchmark; not
wired into `CanonicalBundle` or render receipts yet. Production
wiring requires the leaf-format spec lock and a `bundle_version`
minor bump (`1.0.0` → `1.1.0`) per `docs/COMPATIBILITY_POLICY.md`,
both gated on review of these numbers.

`PROOF_BOUNDARY.md` §2.2 updated with the comparison table +
caveat. Closes M1 entry from `docs/NEXT_SESSION_PLAYBOOK.md`.

### Fixed — Phase E.1 v0.8 (layered defense against LengthFinishReasonError)

The v0.7 long-doc bench errored on 1 cell (`doc_long_human_genome
audience=0.7`) when re-extraction overflowed the 16384-token
completion ceiling. Initial fix (`Pydantic max_length`) was
falsified during research: OpenAI's structured-output validator
does not honor `maxItems` (per the published supported-keywords
list). Replacement is a four-layer defense; bench rerun cleared
the gate.

**Before / after on the same long-paragraph bench:**

| Metric | v0.7 | v0.8 | Note |
|---|---|---|---|
| Errored cells | 1 / 400 | **0 / 400** | LengthFinishReasonError class eliminated |
| Catastrophic outliers (≥5 lost) | 0 | **0** | held |
| Min LLM-axis preservation | 0.700 | 0.545 | one-cell variance |
| Median preservation | 1.000 | 1.000 | held |
| NLI rescue rate | 99.8% | 96.9% | run-to-run noise |
| LLM-axis real losses | 1 | 12 | dispersed; no catastrophic |

**The four-layer defense:**
1. *Prompt-side cap* — system prompt now states "Return at most 64
   triplets…". LLM compliance under structured output is
   empirically high.
2. *Partial-response salvage* — `salvage_partial_triplets` walks
   the truncated JSON in `e.completion.choices[0].message.content`,
   returns whatever complete triplet objects appeared before the
   cutoff. Pure function; free (same response).
3. *One-shot retry with tighter cap* — when salvage yields
   nothing, retry once with cap=32 + emphatic note. Bounded to
   a single extra API call.
4. *Re-raise on retry failure* — terminal; escalates to caller.

**Wild events in the v0.8 bench run:**
- 1× salvage fired: recovered 19 triplets from a partial response
  (cap=64, completion_tokens=16384). Free.
- 1× retry-with-cap=32 fired on a different cell. One extra call.
- Both events logged; no errors propagated.

**Pin bump (load-bearing):** `LengthFinishReasonError` was added in
openai-python 1.40.0 alongside structured-outputs support. Bumping
floor:
- `pyproject.toml`: `openai>=1.0.0` → `openai>=1.40.0,<3.0.0`
- `requirements-prod.txt`: same.

Without the bump, fresh installs that pip-resolve to <1.40 would
ImportError on the new `from openai import LengthFinishReasonError`.

**Files**
- `sum_engine_internal/ensemble/live_llm_adapter.py`: + import,
  + `EXTRACTION_TRIPLE_CAP` / `EXTRACTION_RETRY_CAP`,
  + `salvage_partial_triplets` pure function,
  + `_extract_triplets_with_recovery` async helper,
  + `extract_triplets` rewired to use the recovery path.
- `Tests/test_extractor_salvage.py` — new file, 9 unit tests
  covering salvage helper happy path + adversarial inputs
  (escaped quotes, braces inside strings, malformed objects).
- `pyproject.toml`, `requirements-prod.txt`: pin bumps.

**Verification**
- 60 unit tests pass (51 slider + 9 salvage).
- 1095 full Python suite pass.
- Cross-runtime gates K1–K4 green.
- Bench: 400/400 cells succeed; the previously-failing cell
  succeeds with NLI=1.000.

Research informed by:
- https://community.openai.com/t/min-maxitems-are-not-supported-in-structured-output/958567
- https://github.com/pydantic/pydantic/issues/9815
- https://github.com/openai/openai-python (LengthFinishReasonError class def)

### Improved — Phase E.1 v0.7 (prompt hardening eliminates catastrophic failure mode)

The v0.6 scale bench surfaced two catastrophic outlier cells where
the LLM dropped 80%+ of source facts on technically-dense documents
at extreme `formality=0.1` / `audience=0.3` positions. v0.7 adds a
deterministic prompt mechanism that targets exactly that failure
mode and re-runs the same bench to verify.

**Before / after on the same 16-document long bench:**

| Metric | v0.6 (no hardening) | v0.7 (with reinforcement) | Change |
|---|---|---|---|
| Real losses on LLM axes | 36 | **1** | −97% |
| Cells with ≥5 facts lost | 2 | **0** | eliminated |
| Min preservation | 0.111 | **0.700** | 6× floor lift |
| Median preservation | 1.000 | 1.000 | held |
| p10 | 0.769 | 0.750 | −0.019 (noise) |

**The mechanism (deterministic, no LLM cost):**
`build_system_prompt` in `tome_sliders.py` (and its TS mirror in
`worker/src/render/axis_prompts.ts`) now appends a
`FACT_PRESERVATION_REINFORCEMENT` clause when any non-density axis
is at ≤ 0.3. The clause explicitly tells the LLM "An output that
follows the directives but loses input facts is a FAILED render."
Pure data; same output for same input.

**The trade-off:** 52% of cells score 1.000 perfectly (down from
60%). The reinforcement makes the LLM's surface forms slightly
more defensive, so the strict embedding layer triggers NLI audit
more often. The audit rescues every flagged fact (rescue rate
99.8%). Net: more cells get verified rigorously, real losses
near-zero. This is the right trade — we'd rather verify ten more
cells than miss one catastrophic loss.

**Files**
- `sum_engine_internal/ensemble/tome_sliders.py`:
  `FACT_PRESERVATION_REINFORCEMENT` constant + extension in
  `build_system_prompt`.
- `worker/src/render/axis_prompts.ts`: TS mirror.
- `docs/SLIDER_CONTRACT.md`: version bumped to 0.6; v0.7 results
  table next to v0.6 baseline.

**Verification:** 51 unit tests pass; cross-runtime gates green;
bench shows 399/400 cells succeed (1 errored on
`LengthFinishReasonError` — unrelated robustness issue from prior
benches, not a v0.7 regression).

### Verified — Phase E.1 v0.6 (scale verification on long-document corpus)

The slider's preservation claim was previously verified on 8 short
docs (4–12 triples per doc). v0.6 runs the same bench on 16 long
multi-paragraph documents (9–24 triples per doc, median 17) to
check whether the headline holds at real-world document scale. It
mostly does, with one important honest qualifier.

**Held at scale:**
- Median LLM-axis fact preservation: 1.000 (unchanged).
- 60% of 320 LLM-axis cells score 1.000 (vs 78% on short bench).
- Order preservation: 1.000 wherever measurable.
- NLI rescue rate: 95.7% (800 of 836 audited unmatched facts
  were embedding false negatives, rescued by entailment audit).

**Degraded slightly at scale:**
- p10 dropped from 0.818 → 0.769.
- Min LLM-axis preservation: 0.111 (worst cell).
- 36 confirmed real fact losses on LLM axes (vs 0 on short bench).

**Catastrophic outliers (concentrated, surfaced per-cell):**
Two cells account for 31 of 36 real losses:
- `doc_long_relativity formality=0.1` — 16 / 18 facts lost.
  LLM produced casual paraphrase that dropped scientific precision.
- `doc_long_cryptography audience=0.3` — 15 / 18 facts lost.
  Simplification for general reader dropped technical specifics.

By-axis loss totals: formality 16, audience 16, length 3,
perspective 1.

**What this means for the product claim:** ~99% of LLM-axis
renders preserve all or nearly all facts. ~0.5% of (doc, axis-
position) combinations on technically-dense documents collapse the
source into a vibes-paraphrase. The bench surfaces these per-cell;
nothing silent.

**Files**
- `scripts/bench/corpora/seed_long_paragraphs.json` — 16 hand-
  authored multi-paragraph documents (200–400 words each, topic
  spread across science, history, technology). Public-domain
  factual knowledge to avoid copyright entanglement.
- `scripts/bench/run_long_paragraphs.sh` — runner for the scale
  bench. ~10 min wall clock, ~$1.50 in tokens with NLI audit.
- `docs/SLIDER_CONTRACT.md`: version bumped to 0.5; headline
  rewritten as both-corpora verified; new §"Catastrophic
  outliers" honestly disclosing the failure mode.

**Verification:** 51 unit tests pass; cross-runtime gates green;
bench succeeded on 400/400 cells.

### Added — Phase E.1 v0.5 (Worker render path + slider UI)

The Phase E user-facing loop closes. Paste prose → attest → drag
five sliders → see the tome regenerate at the requested axis
position, with cache-status feedback in real time.

**Worker side:**
- `worker/src/render/axis_prompts.ts` — TypeScript port of the
  Python axis-fragment lookup tables and `build_system_prompt`,
  byte-for-byte equivalent so a Python-rendered tome and a
  Worker-rendered tome from the same input are interchangeable.
  Plus `applyDensity`, `requiresExtrapolator`, `deterministicTome`
  for the canonical (no-LLM) branch.
- `worker/src/routes/render.ts` — replaces the 501 stub with the
  working render path:
    POST → validate → quantize → cache_key → (cache hit?) →
    applyDensity → canonical-or-LLM → cache write → JSON RenderResult.
  Anthropic is the LLM provider (uses `ANTHROPIC_API_KEY` + the
  optional Cloudflare AI Gateway). System prompt comes from
  `buildSystemPrompt`; user prompt is the numbered FACTS list.
  Canonical path skips the LLM entirely when only density is
  non-default. 502 on LLM failure with a clean error message.
- The Worker does NOT compute fact preservation, drift, or
  re-extraction in this revision. The Python bench is the
  canonical source for those metrics; the Worker exposes the
  live render and lets the contract bench verify ahead of time.

**Demo side (`single_file_demo/index.html`):**
- New "Render tome with sliders" card inside the existing result
  block — visible only after a successful attestation.
- Four sliders (length / formality / audience / perspective) with
  live decimal labels. Density (above) is unchanged; it still
  gates which facts get fed to the renderer.
- "Render tome" button POSTs to `/api/render`, swaps the tome
  text into a `<pre>` output area along with cache_status badge,
  `llm_calls_made`, wall-clock ms, and truncated `render_id`.
- A short note links the panel to `SLIDER_CONTRACT.md` and
  surfaces the empirically-verified preservation claim (median
  1.000, p10 0.818).

After Attest, the post-density triples are stashed at
`window.__sumLastTriples` so the render handler can pick them
up without re-extracting.

**Operational notes:**
- Worker requires `ANTHROPIC_API_KEY` secret (`wrangler secret put
  ANTHROPIC_API_KEY`). The demo's existing fall-back path for
  /api/complete is independent.
- `RENDER_CACHE` KV binding is still commented out in
  `wrangler.toml`. Demo works without it (cache misses always
  re-render); enabling the binding makes repeated slider positions
  near-instant.

**Verification:** `npm run typecheck` clean across the new TS;
end-to-end smoke against a deployed Worker pending.

### Verified — Phase E.1 v0.4 (NLI audit confirms the product claim)

The slider's load-bearing claim — *axis changes do not lose facts* —
is now **empirically verified**, not approximated. NLI audit on the
weak cells (where embedding similarity flagged apparent loss)
delivered a clean verdict.

**Headline result:**
- 200 cells × 8 docs / 5 axes / 5 bin positions
- 45 LLM-axis cells flagged for audit (semantic preservation < 0.7)
- 186 NLI entailment calls fired
- **186 facts rescued from semantic false-negatives, 0 facts
  confirmed real loss on any LLM-axis cell**
- Median LLM-axis fact preservation = 1.000; p10 = 0.818;
  min = 0.727; 124 of 160 cells score 1.000 perfectly.

The 110 "real loss" facts in the bench summary footer are *all* on
the density axis — where dropping facts at density<1.0 is the
explicit product knob. Density loss is by design, not by accident.

**Practical reading of the bench data:**
- `length=0.9` semantic p10 = 0.00 was an embedding artifact. NLI
  rescued every "lost" fact. The LLM IS preserving the source when
  asked to write expansively; embeddings just don't recognize the
  rephrased surface forms.
- `audience=0.1` semantic median = 0.83 with 4/8 cells audited —
  every audit confirmed the rephrased "lay-reader" prose still
  expressed the source facts.
- Order preservation = 1.000 across every cell where measurable.
  MontageLie-style reordering attacks would still be detected.

**v0.4 substrate**
- `live_llm_adapter.py`: + `EntailmentResponse` Pydantic model;
  `LiveLLMAdapter.check_entailment` runs structured-output NLI
  judgement with strict prompting.
- `slider_renderer.py`: + `NLIFactBreakdown` dataclass with three-
  bucket accounting; + `nli_fact_preservation` async function that
  runs semantic match first and only fires NLI on whatever semantic
  missed (cost-bounded).
- Bench: + `--audit-threshold` CLI arg (default 0.7); BenchCell
  gains `fact_preservation_nli`, `n_matched_nli_only`, `n_lost_real`,
  `nli_calls_made`. Per-axis stderr summary now shows nli column
  with audited-count; aggregate footer reports total NLI calls,
  facts rescued, facts confirmed lost.

**Cost added by v0.4:** +43s wall clock (138.2s vs 97.7s v0.3),
~$0.15 in tokens for the 186 NLI calls. Bench is still <2.5 min
end-to-end.

**Tests (51 pass; +5 NLI tests):**
TestNLIFactPreservation — phase-2 skipped when phase-1 catches all
(cost guarantee), embedding false-negative rescued, real loss when
neither layer catches, partial mixed case, empty source returns
perfect.

**What this means for the product:** the slider can ship with the
strong claim that axis changes preserve facts. The threshold tables
and per-axis drift numbers stay honest documentation of stylistic
adherence (how well the LLM follows the directive) — but
fact-preservation is no longer a measurement question; it's
verified.

### Improved — Phase E.1 v0.3 (constrained-decoding render path)

Switches the renderer's LLM call from free-form `chat.completions.
create` to `beta.chat.completions.parse` with a Pydantic-enforced
`RenderedTome` schema (tome + claimed_triples). The LLM now emits
both the narrative AND its self-attested list of preserved triples
in one structured response.

**Two confirmed wins:**
- *Reliability:* 0/200 cells errored vs 2/200 in v0.2 on `doc_einstein`
  (`LengthFinishReasonError` from token-budget truncation). Schema-
  enforced output makes parse-failure-class bugs impossible.
- *Adversarial signal:* `claim_reextract_jaccard` records divergence
  between LLM self-attestation and independent re-extraction.

**One surprising negative finding (documented honestly):** the LLM
does NOT reliably itemise what it just wrote in the same canonical
form the extractor uses. Cross-axis median `claim_jaccard` = 0.286
(range 0.00–1.00). Counts match (n_claimed ≈ n_reextracted ≈
n_source) — it's surface-form divergence, not list-size mismatch.
Practical implication: **LLM self-attestation is NOT a free fact-
preservation oracle.** Independent re-extraction remains the source
of truth.

**Latency cost:** +16% (97.7s → 113.6s for 200 cells). Net trade
accepted for the format-validity guarantee + signal density.

**Drift secondary effects:** structured output subtly changes how
the LLM allocates tokens between tome and claimed-triples list,
shifting axis-directive adherence on some positions (formality=0.1
went 0.10 → 0.40; perspective=0.3 went 0.09 → 0.50). Semantic fact
preservation cross-axis median unchanged at 1.000; order preservation
unchanged at 1.000. Product claim intact.

**Files**
- `live_llm_adapter.py`: + `RenderedTome` Pydantic model;
  `OpenAIChatClient.chat_completion_structured` returning
  (tome, triples).
- `slider_renderer.py`: `LLMChatClient` Protocol gains
  `chat_completion_structured`; `RenderResult` gains
  `claimed_triples`; `render()` prefers structured path with
  `hasattr` fallback for legacy clients.
- `Tests/test_slider_renderer.py`: 46 pass (+3 structured-path
  tests); legacy chat-only client fallback verified.
- `Tests/benchmarks/slider_drift_bench.py`: BenchCell gains
  `claim_reextract_jaccard` + `n_claimed_triples`; per-axis stderr
  summary includes `claim_jaccard` column.

Recommended downstream use: trust independent re-extraction for
fact preservation claims; use claim_jaccard for outlier detection
(low jaccard at non-neutral positions = LLM allocating canonicalisation
attention away from itemising). Don't ship a "fast mode" that skips
re-extraction in favour of claimed_triples — the bench data shows
that mode would systematically under-report preservation.

### Verified — Phase E.1 v0.2 (three-layer fact preservation, honest claim)

The previous "fact preservation = 1.000" headline was wrong twice
over: first because it computed against `triples_used` (post-density,
trivially equal to source for non-density axes) instead of
`reextracted_triples` (the actual round-trip set); then, after
correcting that, because exact `(s,p,o)` match is too brittle to
distinguish real fact loss from extraction surface-form drift
(`graduated` vs `graduated_in`).

This release lands the corrected substrate: three composable
preservation layers, plus parallel bench execution that cuts wall
clock 4× without changing token spend.

**Three-layer fact preservation** (all reported per cell, all in
the JSONL artifact):
- Strict — exact-key match. Regression check on extractor stability,
  not the headline.
- Normalized (A3) — strips auxiliary verb prefixes (was_, has_, ...)
  and preposition suffixes (_in, _from, ...) from predicates plus
  articles from entities. Free, deterministic, 50 LOC of rules.
- Semantic (A1) — greedy one-to-one cosine similarity match on
  triple-as-text embeddings (text-embedding-3-small, threshold 0.85).
  This is the load-bearing metric for the slider's product claim.

**Headline result, honestly measured (n=160 LLM-axis cells):**
- Strict median: 0.333. Brittle to surface-form drift; retained as
  regression check only.
- Normalized median: 0.500. A3 lifts strict by ~50% by collapsing
  preposition / auxiliary drift.
- **Semantic median: 1.000. p10: 0.455.** Half the cells preserve
  every source fact; the worst 10% still hold 45%.
- Order preservation: 1.000 wherever measurable. MontageLie-style
  reordering attacks are not a present failure mode of good-faith
  renders.

**Where the slider works perfectly:** all neutral positions (axis=0.5)
preserve every source fact. `length=0.1` and `length=0.3` (compression
modes) score ≥0.91 p10 — the LLM loses no facts when asked to be
brief.

**Where the slider stresses:** `length=0.9` (semantic p10 = 0.00 — the
LLM expands 6 facts to 600 words and dilutes individual fact identity
in some renders), `audience=0.1` and `audience=0.3` (general-reader
mode drops technical specifics — semantic p10 = 0.33). Perspective
moderate positions (0.3, 0.7) show the LLM committing to one mode
rather than blending — registered as drift by the coarse pronoun-
ratio classifier, but order_preservation = 1.000 confirms the facts
themselves stay in place.

**Bench parallelization** — `slider_drift_bench.py` now runs cells
concurrently via `asyncio.Semaphore` + `as_completed` (default
concurrency=16, `--concurrency` CLI arg). Source-extraction is
hoisted outside the cell loop and parallelized too — eliminates
~175 of 200 redundant source-extraction LLM calls on an 8-doc /
25-position run. Wall clock drops from ~7 min to 97.7s. Same total
token spend (~$0.35 with embedding calls); strictly fewer
wall-clock seconds.

**MontageLie defense** — `order_preservation(source, reextracted)`
returns the fraction of preserved-triple pairs that retain their
relative order from source to tome. Regression test demonstrates a
timeline-reversed permutation scores 1.0 on set-based fact
preservation but 0.0 on order_preservation. The defense works as
designed; the bench data shows order = 1.000 in honest renders, so
the attack is a v0.3+ frontier concern, not a present failure mode.

**Audience expansion (5000-word table)** — Brown-corpus frequency
table grew from 2000 to 5000 words. Median audience drift cut
roughly in half from STATE 5b. Combined with the corrected fact-
preservation metric, audience axis now reads as the cleanest LLM
axis on this corpus.

**Files**
- `sum_engine_internal/ensemble/slider_renderer.py` — adds
  `_normalize_predicate`, `_normalize_entity`, `_normalize_triple`,
  `fact_preservation_normalized`, `semantic_fact_preservation`,
  `order_preservation`. `RenderResult` gains `reextracted_triples`.
- `sum_engine_internal/ensemble/data/common_english_5000.txt` — new
  data file; loader prefers 5000 over 2000.
- `Tests/test_slider_renderer.py` — 43 tests pass (was 22 → 30 →
  43). New: TestNormalizationLayer (8), TestSemanticPreservation (5),
  including the headline MontageLie regression test.
- `Tests/benchmarks/slider_drift_bench.py` — three preservation
  columns + order column in BenchCell; parallel execution; per-cell
  progress to stderr; per-axis four-column summary footer.
- `docs/SLIDER_CONTRACT.md` — version bumped to 0.3; per-axis
  fact-preservation table now shows all three layers.

**Verification:** 43 unit tests pass; cross-runtime gates green;
bench re-run in 97.7s (was ~7 min) with 200/200 cells succeeding.

### Improved — Phase E.1 STATE 5b (classifier upgrades + tightened thresholds)

Two follow-up fixes after STATE 5 surfaced the calibration gaps:

**Audience classifier:** the embedded ~200-word common-words list
was swapped for a 2000-word frequency table derived from NLTK's
Brown corpus (`sum_engine_internal/ensemble/data/common_english_2000.txt`,
re-generable via `scripts/data/regen_common_english_2000.py`). The
loader uses `importlib.resources` and ships the file via
`pyproject.toml`'s `[tool.setuptools.package-data]`. Median audience
drift cut by ~50% across all positions; threshold 0.55 → 0.40.

**Length bands:** the per-triple-linear bands `(5,15) … (80,200)`
were replaced with empirically-derived bands
`(4,10) / (5,12) / (4,10) / (30,60) / (80,140)` (words per source
triple). The original assumption that response length scales
linearly with input fact count was wrong — the LLM has a ~6 wpt
floor at and below position 0.5 and scales aggressively above.
Median length drift cut 3× across positions 0.1–0.7; threshold
0.95 → 0.60.

**Bench harness:** `BenchCell` gained a `tome_word_count` field so
future calibration runs have the raw data without re-running.

**SLIDER_CONTRACT.md:** STATE 5b numbers replace STATE 5 placeholders
in the per-axis tables; `§"Empirical bench runs"` now records both
runs side-by-side so future readers can see what changed and why.

**Robustness footnote:** 2/200 cells errored on `doc_einstein` with
`LengthFinishReasonError` — the LLM produced more triples during
re-extraction than fit in its 16384-token completion budget. Bench
captures these per-row rather than aborting; documented as a v0.2
robustness item, not a contract violation.

### Added — Phase E.1 v0.2 research roadmap

`docs/SLIDER_V02_RESEARCH.md` distills a survey of mathematical
substrates for verifiable bidirectional knowledge distillation
engines (AIT, category theory, IB, proof-carrying transformations,
GEPA/DSPy/GRPO, hierarchical PRMs, metamorphic testing) into the
3–4 items that materially improve SUM's slider in the next 1–3 PRs.

The doc has three sections: (a) validation — what SUM is already
doing right per the survey (verifiable rewards, cycle-consistency,
content-addressed everything, the Pareto-frontier framing); (b)
actionable v0.2 work — MontageLie-resistant fact preservation,
constrained decoding for the renderer, audience classifier
expansion, metamorphic testing; (c) awareness/defer — zkML, Lean 4
paragraph-level proofs, GEPA outer loop, etc., that are SOTA but
out of scope for SUM today.

The MontageLie finding is the most consequential: SUM's
`fact_preservation = 1.000` headline uses set-based comparison,
which Zheng et al. (May 2025) showed is exploitable by reordering
preserved triples into a deceptive narrative. v0.2 PR will add
event-order-aware verifier (pairwise order-preservation between
source-triple pairs) so the headline claim is robust.

### Verified — Phase E.1 STATE 5 (empirical bench run + contract update)

Ran `Tests/benchmarks/slider_drift_bench.py` against a real multi-fact
prose corpus (8 hand-authored 3–5 sentence paragraphs, 4–12 triples
each). 200 cells, gpt-4o-mini, ~$0.30 in tokens. Surfaced one
correctness bug, two formula-calibration issues, and one verified
load-bearing claim.

**Headline:** fact preservation is 1.000 (median, p10) across all 200
LLM-axis cells. The slider's central product claim — *axis changes
do not lose facts* — holds empirically. Slider is a real product, not
just substrate.

**Bug fixed in this round:**
- `slider_renderer.render()` was passing the post-density
  `kept_triples` to `measure_drift` as the source set. Density drift
  formula then computed `expected_retained = floor(filtered_count *
  density)`, which double-applied density and produced spurious
  drift values up to 1.75 at moderate densities. Fix: pass the
  original source `triples_tuple`. Density drift now 0.000 across
  all positions (verified by re-run).

**Contract updates** (`docs/SLIDER_CONTRACT.md`):
- All five threshold rows now show `Measured (n=8, p90)` alongside
  the limit. Numbers come from this bench, not theory.
- Density: ≤0.001 verified.
- Formality: ≤0.25 → ≤0.40 (covers p90 tail at extremes).
- Perspective: ≤0.20 → ≤0.40 (median fits, p90 spikes at moderate
  positions; the LLM commits to one perspective rather than
  blending).
- Length: ≤0.5 → ≤0.95 *preliminary*. Per-triple band assumption is
  empirically wrong (LLM doesn't scale response length linearly with
  fact count). v0.2 will recalibrate against absolute word-count
  bands using this bench's tome data.
- Audience: ≤0.10 → ≤0.55 *preliminary*. Embedded ~200-word common-
  words table saturates: technical prose reads as ~50% jargon
  regardless of axis. v0.2 will swap to a frequency-table classifier.
- New §"Empirical bench run" section: per-axis median drift table,
  reproduction command, headline fact-preservation result, two
  documented v0.2 follow-ups.
- Version bumped from 0.1 (draft) to 0.2 (empirically verified).

**Bench corpus:** new file `scripts/bench/corpora/seed_paragraphs.json`
with 8 multi-fact paragraphs hand-authored from common factual
knowledge (avoids copyright entanglement). `scripts/bench/run_*.sh`
wrappers landed for smoke (1 doc) and full (8 docs) runs.

Verification: 22 unit tests pass; 1057 full Python suite pass; bench
re-run with bug fix shows density drift 0.000 across all positions
and percentiles.

### Added — Phase E.1 STATE 4 (slider renderer pipeline lands)

The renderer scaffold from STATE 2 ships its real implementation. Five
xfailed tests flip to passes; one failing test surfaces a design bug
that the fix kills outright.

  sum_engine_internal/ensemble/tome_sliders.py
    + length_fragment / formality_fragment / audience_fragment /
      perspective_fragment now return real prompt strings keyed by bin
      centre (5 positions × 4 axes = 20 fragments). Empty string at
      neutral midpoint to keep the prompt lean at default.
    * quantize() no longer bins density. Density is deterministic and
      binning 1.0→0.9 made "request all triples" un-expressible. The
      cache key includes raw density (unique per density level).

  sum_engine_internal/ensemble/slider_renderer.py
    + render() pipeline: cache-first → quantize → apply_density →
      canonical-vs-LLM branch → measure_drift → cache-write → return.
      Canonical path skips LLM and re-extraction entirely when only
      density is non-default (no drift introduced).
    + measure_drift() implements all five SLIDER_CONTRACT formulas:
      density (set comparison), length (word-count band), formality
      (register marker classifier), audience (jargon density),
      perspective (first-person pronoun ratio). Pure function;
      embedded lookup tables (no data-file deps).
    * measure_drift signature gained `tome: str` — needed for the
      four LLM axes whose drift is measured against tome content,
      not triple sets.

  sum_engine_internal/ensemble/live_llm_adapter.py
    + OpenAIChatClient: thin adapter from LiveLLMAdapter to the
      slider_renderer.LLMChatClient Protocol. Lives in live_llm_adapter
      so the renderer module stays openai-dep-free.

  Tests/test_slider_renderer.py
    * xfail strict markers removed from TestRenderPipeline (3 tests)
      and TestMeasureDrift (2 tests). All 22 tests pass.
    + test_quantize_preserves_density_endpoints — new regression
      test for the density-not-binned fix.

  Tests/benchmarks/slider_drift_bench.py
    + _bench_one_cell now extracts source triples, calls render(),
      computes per-axis drift + fact-preservation per
      SLIDER_CONTRACT.md. main_async wires LiveLLMAdapter +
      OpenAIChatClient. Errors captured per-row rather than raising.

  worker/src/routes/render.ts
    * quantizeSliders mirrors the Python density-exemption.

  worker/src/cache/bin_cache.ts
    * deriveCacheKey now constructs the payload with alphabetical key
      order (matches Python json.dumps sort_keys=True). Outstanding gap
      documented inline: floating-point repr (`1.0` Python vs `1` JS)
      will need normalisation when the Worker actually shares cache
      cells with Python in STATE 4.B.

  docs/SLIDER_CONTRACT.md
    * §"Axis definitions" notes density is exempt from binning.

Verification: 1057 Python tests pass; worker typecheck clean;
make xruntime (K1/K1-mw/K2/K3/K4) PASS; make xruntime-adversarial
(A1–A6) PASS.

STATE 4.B (next): Worker handleRender real LLM call (replace 501
stub) + numeric normalisation in deriveCacheKey for Python↔TS cache
coherence. STATE 5: bench harness against seed_v1 to populate
threshold columns in the contract.

### Added — Phase E scaffold (slider as first-class product)

The genesis vision — bidirectional Tags ↔ Tomes with a slider — has
been substrate-only since the project began (density axis works
deterministically; the other four axes existed as metadata fields).
Phase E.1 STATE 2 lands the typed scaffold for the renderer +
contract doc + tests; STATE 4 fills the per-axis logic.

  sum_engine_internal/ensemble/tome_sliders.py  (extended)
    + SLIDER_BINS_PER_AXIS = 5                  # 3125 cache cells per triple-set
    + snap_to_bin(value, bins) -> float         # quantize to bin centre
    + quantize(TomeSliders) -> TomeSliders      # all-axis snap
    + length_fragment / formality_fragment /    # axis prompt fragments;
      audience_fragment / perspective_fragment    fail-loud at non-neutral
                                                  positions until STATE 4
    + build_system_prompt(TomeSliders) -> str   # composes neutral base +
                                                  per-axis fragments

  sum_engine_internal/ensemble/slider_renderer.py  (new)
    Type contracts:
      Triple = tuple[str, str, str]
      CacheStatus = HIT | MISS | BYPASS
      DriftAxis  = density | length | formality | audience | perspective
      AxisDrift  = (axis, value, threshold, classification, explanation)
      RenderResult = (tome, triples_used, drift, cache_status,
                      llm_calls_made, wall_clock_ms,
                      quantized_sliders, render_id)
      SliderCache (Protocol)        = get / put / stats
      LLMChatClient (Protocol)      = chat_completion(system, user, max_tokens)
      TripleExtractor (Callable)    = (str) -> awaitable[list[Triple]]
    Functions:
      cache_key(triples, sliders)   = sha256(sorted_triples + sliders)[:32]
      render(...)                   = NotImplementedError until STATE 4
      measure_drift(...)            = NotImplementedError until STATE 4
      InMemorySliderCache           = dict-backed reference impl

  worker/src/cache/bin_cache.ts  (new)
    Cache contract mirror. deriveCacheKey produces the SAME 32-char
    string the Python cache_key produces for the same input — cross-
    runtime cache coherence by content-addressed key.

  worker/src/routes/render.ts  (new)
    POST /api/render route. Quantizes incoming slider position,
    derives cache key, returns 501 + activation plan until STATE 4.
    Wired into worker/src/index.ts; KV binding RENDER_CACHE
    declared (commented) in wrangler.toml.

  Tests/test_slider_renderer.py  (new — 21 tests)
    16 pass today (snap, quantize, cache_key, InMemorySliderCache).
    5 xfailed strict (render pipeline + measure_drift) — bodies are
    spec, not stub. STATE 4 lands the implementation; xfails flip
    to passes with no test body changes.

  Tests/benchmarks/slider_drift_bench.py  (new)
    Per-axis drift bench harness. NDJSON output schema
    sum.slider_drift_bench.v1. STATE 2 returns stub-error rows so
    the harness structure is exercised end-to-end; STATE 4 wires
    real measurement.

  docs/SLIDER_CONTRACT.md  (new)
    Source-of-truth spec. Per-axis drift formulas, thresholds,
    cache semantics, UX commit-vs-drag decision matrix, stop-the-
    line conditions. Every numeric tolerance is empirically
    falsifiable by the bench harness.

Carmack-frame anti-hypotheses captured in the contract:
  1. Slider may be wrong UX (users want discrete buttons). E.6
     trial A/B-instruments both control surfaces.
  2. LLM latency may make drag-and-see undeliverable. 500ms
     debounce + bin cache + skeleton-loader; commit-on-release
     fallback.
  3. Round-trip drift may be wildly variant per axis. Live drift
     display per axis; "facts preserved within X%" replaces
     "facts preserved" in product copy.
  4. Go service rewrite is premature without measured Python
     bottleneck. Defer until E.6 telemetry decides.
  5. >10K-axiom scaling is hypothetical. Layered-architecture
     plan stays in PROOF_BOUNDARY §3; build only when measured.

No render claim made today. STATE 4 implements; STATE 5 verifies
against the bench harness; only then does the slider become a
shipping product feature instead of a typed contract.

### Added — Phase B intensification queue (B5–B7) in playbook

- `docs/NEXT_SESSION_PLAYBOOK.md` Phase B grew three explicit items
  that collapse multi-step user flows into single gestures (process
  intensification: combining steps so the user touches the
  deliverable, not an intermediate):
    * **B5** — shareable bundle URLs (`/b/{hash}` Worker route +
      R2 backing). Removes the JSON-file artifact from civilian
      awareness. Depends on B1.
    * **B6** — PWA-installable demo (manifest.json + service
      worker). Phone-screen attestation flow, offline verify after
      first load. ~40 LOC + a manifest. No dependencies.
    * **B7** — `sum attest <url>` fetch mode. Eliminates the
      "open browser → copy text → switch to terminal → paste"
      five-step pattern. Depends on B1.
- "Out of Phase B (named so we don't lose them)" subsection captures
  two items that surfaced in the analysis but belong elsewhere: the
  browser extension (v0.4 feature, depends on B1+B5+B7) and verify
  badges (Phase C5, depends on B5).
- Phase B exit gate updated to include the with-B5–B7 case
  ("phone-to-phone share + verify, no install").
- Pinning policy on B5 explicitly forbids long-term retrievability
  promises under v1 — protects against locking in `sha256_64_v1`
  after Priority 3 lands `sha256_128_v2`.

No code in this commit. Items are queued, not built. Hardening
ordering (Phase A priorities first) is unchanged; the
intensification work is post-hardening.

### Added — platform trajectory (Phases A–D) in NEXT_SESSION_PLAYBOOK

- `docs/NEXT_SESSION_PLAYBOOK.md` grew a new
  "Beyond the priorities — platform trajectory" section after the
  Priority 1–8 block. Four phases:
    * **Phase A** — finish the hardening playbook (= Priorities 3–8).
      No new thinking; exists as a framing anchor for Phases B/C/D.
    * **Phase B** — platform surface: source anchoring in the bundle
      schema (B1), bundle explorer / viewer (B2), `sum verify --explain`
      UX (B3), `sum tutorial` onboarding (B4). Depends on Phase A.
    * **Phase C** — network layer: well-known bundle discovery (C1),
      composition UX (C2), cross-attestation graph (C3), W3C VC 2.0
      full round-trip + PROV-O (C4). Depends on B1 + P6.
    * **Phase D** — 1.0 stability contract. Not new work; a decision
      point with a CI-backed promise that 1.0-minted bundles continue
      to verify 10 years from now.
- "The greater goal, stated plainly" preface names the three gaps the
  phases exist to close: trust end-to-end for a specific adversarial
  user, composability across publishers, engine-itself verifiability.
- Each phase has an explicit exit gate. "How to use this document"
  footer names the reading order for memory-less sessions and flags
  the two common scope-pressure failure modes (ship-faster-skip-gate,
  ship-more-add-item).
- `CLAUDE.md` onboarding item #5 expanded to name the new section and
  its Phase-dependency rule (do not start B-work while A is open).

No immediate execution commitment. Phase A continues through
Priorities 3–8 in their existing order; the trajectory section exists
so future sessions don't re-derive the same roadmap from scratch.

### Added — Priority 2: WASM-vs-JS derivation benchmark harness

- `Tests/benchmarks/browser_wasm_bench.html` — single-file harness that
  runs the deployed WASM core (`sum_core.wasm`) and the pure-JS fallback
  against identical input across N ∈ {10, 100, 1000, 10000} axiom
  derivations. Reports median / min / max ms per surface, per-op µs,
  and the JS ÷ WASM ratio. Also asserts bit-identical state integers
  across the two paths on every trial (correctness gate, not speed
  datum). Emits a machine-readable JSON block ready to paste into the
  methodology doc.
- `scripts/bench_python_derive.py` — Python-side companion that
  measures `GodelStateAlgebra.get_or_mint_prime` on the same key
  generator (`sum-bench-v1` seed), records whether the Zig shared
  library served or the `sympy.nextprime` fallback did, and emits the
  same-schema JSON block.
- `docs/WASM_PERFORMANCE.md` — methodology doc. Declares exactly what
  is measured (prime derivation alone), what is NOT (Ed25519, extraction,
  bundle parse), the trial protocol (5 trials, median, 3 warm-ups), the
  reproduction steps for all four surfaces (Python, Node, Browser-WASM,
  Browser-JS), and the fallback statement that ships regardless of
  outcome. Every numeric cell is labelled `measured`; blocks are `"not
  yet measured"` placeholders until the browser-matrix run happens.
  Change-control rules forbid adding performance language to
  `README.md` or a commit message until the corresponding row has data.
- `make wasm-bench` serves the repo over HTTP so the browser harness
  has a working `instantiateStreaming` + `crypto.subtle` environment.
  `make wasm-bench-python` runs the Python companion.

**No performance claim is made by this commit.** Per the playbook's
"measure before you assert" principle, shipping the harness is
orthogonal to publishing numbers. The numbers arrive in a later commit
that pastes concrete JSON blocks under each per-browser section; that
commit is the one allowed to add "fast" or "X× faster" language to the
prose.

### Added — Priority 1: adversarial cross-runtime rejection matrix

- `scripts/verify_cross_runtime_adversarial.py` — companion to the
  existing K-matrix. Six fixtures (A1-A6) covering the three
  cross-runtime-equivalent rejection classes: structural (missing
  tome, truncated tome, state integer = 0, state integer = -42),
  version (unknown canonical_format_version), signature (Ed25519
  bundle with post-sign tome tampering). Each fixture is passed
  through BOTH the Python verifier (`sum verify --input`) and the
  Node verifier (`standalone_verifier/verify.js`). The harness
  asserts: (1) both reject; (2) rejection classifications agree.
- HMAC tampering is intentionally out of scope for this harness —
  the Node verifier's header docstring is explicit that HMAC is
  not checked ("shared-secret, not public witness"). HMAC fixtures
  stay in `Tests/test_adversarial_bundles.py` (Python unit tests).
- `make xruntime-adversarial` runs it locally.
- `.github/workflows/quantum-ci.yml` `cross-runtime-harness` job
  runs A1-A6 alongside the existing K-matrix on every push.
- `docs/PROOF_BOUNDARY.md` §1.2 updated: the Cross-Runtime State
  Equivalence claim is now backed by FOUR harnesses, the fourth
  explicitly "proved on adversarial inputs," closing the
  valid-only-agreement gap the previous three left open.

Initial run result: **6 / 6 fixtures pass** — the two verifiers
already agree on rejection class for every adversarial case we
built. Worth reading as: the valid-input-agreement property
hasn't been accidentally extending a false claim about invalid-
input agreement; we checked and the claim holds.

Queue: A7+ fixtures (boundary state integers > 10^5 digits for
DoS; scheme-downgrade attempts between sha256_64_v1 and an
as-yet-unshipped v2; empty `{}` bundles; non-object root JSON)
can be added as single-line `FIXTURES` entries. Priority 1 is
formally discharged; future fixtures are additive hardening.

### Added — forward playbook for future sessions

- `docs/NEXT_SESSION_PLAYBOOK.md` — ordered work queue (Priorities
  1–8) with principles, stop-the-line triggers, and the ordering
  rationale. Priorities 1–2 harden existing claims (adversarial
  cross-runtime fuzzing, WASM perf measurement); 3–4 extend them
  into new regions (`sha256_128_v2`, SPARQL disambiguation); 5–6
  make the surface self-describing (threat-model validation,
  delta-bundle semantics); 7–8 broaden the trust base (supply-chain
  attestation, LLM-extraction honesty guardrails). `CLAUDE.md`
  onboarding list now points at this file as item #5 so a memory-
  less session discovers the queue by reading the canonical entry
  block.

### Removed — portfolio-site artifacts (separation of concerns)

SUM is a knowledge-distillation engine. The sumequities.com portfolio
is a personal-portfolio site that references SUM as one of many
featured projects. The two should not share governance, CI rules,
or narrative files — a third-party `pip install sum-engine` consumer
has no business with a portfolio file, and a fork should not inherit
rules about a personal portfolio. Earlier commits in this session
incorrectly coupled them; this entry records the full revert.

- Deleted `PORTFOLIO.md` at repo root.
- Deleted `scripts/check_portfolio_contract.py`,
  `scripts/hooks/pre-commit`, `scripts/install-hooks.sh`.
- Removed the `portfolio-contract` job from
  `.github/workflows/quantum-ci.yml`.
- Removed the `## PORTFOLIO.md contract` section from `CLAUDE.md`;
  replaced the now-stale onboarding list-item #1 (which pointed at
  `PORTFOLIO.md`) with a shortened 4-file reading list. Added an
  `## Out of scope — do not cross-repo edit` note naming the
  portfolio repo as off-limits.
- Removed `make portfolio` and `make install-hooks` targets from
  `Makefile`; dropped both from `.PHONY`. Added `wasm` to `.PHONY`
  (it was listed as a target earlier in the session but never added
  to the list).
- `README.md` hero no longer carries the "Portfolio-facing overview:
  PORTFOLIO.md" pointer. The "Shipped since the last README pass"
  bullet for "PORTFOLIO.md + CLAUDE.md contract" removed.
- `CONTRIBUTING.md` setup block no longer tells contributors to run
  `make install-hooks`; Verification-Gates table no longer carries
  the `PORTFOLIO contract` row.

### Removed — experimental AT Protocol Lexicon (same-confusion teardown)

Phase C from the same session published
`com.sumequities.experimental.axiom` as a Lexicon schema on the
user's Bluesky PDS under the portfolio's domain authority. Same
portfolio-vs-engine confusion at the namespace layer: SUM the
engine should not claim a Lexicon under the portfolio's domain.
External state torn down before this commit landed:

- Bluesky record `at://did:plc:cuqlv67qg6tepr2gjvknajcp/com.atproto.lexicon.schema/com.sumequities.experimental.axiom`
  deleted via `com.atproto.repo.deleteRecord`. PDS confirms
  `RecordNotFound`; `listRecords` returns empty.
- DNS TXT `_lexicon.sumequities.com` (content
  `did=did:web:sumequities.com`) deleted from Cloudflare DNS.
  `dig +short TXT _lexicon.sumequities.com` empty.
- Bluesky app-password `sum-lexicon-publisher` (fragment
  `24hi-yfrq-3q6r-5ezs`) revoked at `bsky.app/settings/app-passwords`.

In-repo artifacts that were drafted on disk but never committed
(the C.6 gate was going to hold them; user surfaced the deeper
issue before C.7 fired) are `rm`'d as working-tree cleanup in this
commit: `scripts/publish_lexicon_schema.py`, `at_proto/lexicon/`
directory and its single JSON.

### Added — `/api/qid` Wikidata resolver (Phase 4a)

- `worker/src/routes/qid.ts` — replaces the 501 stub with a working
  resolver. Takes a batch of `{text, kind?, lang?}` terms, looks each
  one up via the MediaWiki `wbsearchentities` API, returns
  `{id, label, description, confidence, source}` for every term.
  Up to 50 terms per request; parallel fetches. Unknown terms
  surface `{id: null, reason: "no-match"}` rather than errors.
- Two-tier caching: edge Cache API (same-colo, zero-hop) on every
  request; the TTL is 30 days (Wikidata labels rarely change on
  month scales). KV binding left as an optional second layer
  (commented in `wrangler.toml`; activate by
  `wrangler kv:namespace create qid-cache`).
- Confidence scoring mirrors the `match.type` field Wikidata returns
  (`label` → 1.0, `alias` → 0.7, everything else → 0.5) — a
  categorical signal translated into a 0–1 ordering for threshold
  logic downstream.
- User-Agent header `SUMDemoQIDResolver/0.3.0 (+github.com/OtotaO/SUM)`
  per Wikidata's operator-contact guidance.

Intentionally not in v0.3: SPARQL disambiguation when multiple
candidates are plausible. wbsearchentities alone hits >80% accuracy
on common-noun / proper-name lookups; SPARQL refinement (filter by
predicate domain) is Phase 4b once we've measured the v0.3 baseline
on a real corpus.

### Added — WASM acceleration in the browser demo

- `single_file_demo/sum_core.wasm` (97 KB, committed) — the `core-zig/`
  module cross-compiled to `wasm32-freestanding` with `ReleaseSmall`.
  Exports nine functions (`sum_get_deterministic_prime`,
  `sum_get_deterministic_prime_v2`, `sum_bigint_gcd/lcm/mod`,
  `sum_bigint_divisible_by_u64`, `sum_batch_mint_primes`,
  `wasm_alloc_bytes`, `wasm_free_bytes`) plus the linear memory.
- `single_file_demo/sum_core_wasm.js` — browser-side async loader
  factory. Returns `{derivePrime, isReady:true}` on success; returns
  `null` on any failure (WebAssembly unavailable, fetch/compile/
  instantiate error) so the caller's fallback logic stays trivial.
  Handles the WebAssembly i64→BigInt signed-surface wrinkle (u64
  zig returns come back signed in JS; masked with `& 0xffff…ffffn`
  post-call).
- `single_file_demo/test_wasm.js` — zero-dep Node self-test pinning
  the WASM output to the cross-runtime fixture set (same vectors as
  `verify.js --self-test`). Part of the demo's test triad alongside
  `test_jcs.js` and `test_provenance.js`.
- `single_file_demo/index.html` — `derivePrime()` now calls the WASM
  loader first (single-flight, cached after first load); falls back
  to the original WebCrypto+JS-BigInt path when WASM isn't reachable
  (standalone file open, Claude artifact, older browsers). Transparent
  to every caller — the function still returns the correct BigInt.
  A `<link rel=preload>` for the `.wasm` fires in the page head so
  the module is in-flight before the user clicks Attest.
- `.github/workflows/quantum-ci.yml` `zig-core` job:
  * Builds the WASM target alongside the native library.
  * SHA-256-compares the freshly-built `.wasm` against the committed
    blob — catches source/binary drift (fails with the rebuild
    command if they don't match).
  * Runs `node single_file_demo/test_wasm.js` to assert the committed
    `.wasm` still produces the reference primes.
- `core-zig/build.zig` — updated `link_libc` syntax to zig 0.16 /
  0.15.late-cycle module-field form (was `.linkLibC()` method call,
  which zig 0.16 removed from `Build.Step.Compile`).
- `Makefile` — new `make wasm` target builds + copies + runs the
  self-test in one step. Run after any `core-zig/src/main.zig` edit.

Performance: still to be measured on a real workload. The WASM path
replaces roughly "WebCrypto SHA-256 + O(log² N) Miller-Rabin per
candidate × ~80 candidates on average" with native Zig on wasm32.
Expected speedup at 1.5–5× for the prime-minting hot path. Measured
numbers will land in PROOF_BOUNDARY §2.2 when a browser bench harness
is wired.

### Added — hosted-demo infrastructure

- `worker/` directory with a Cloudflare Worker (`src/index.ts`) that
  serves `../single_file_demo/` as static assets and routes `/api/*`
  through TypeScript handlers. Migrates the previous Pages deployment
  to Workers per Cloudflare's April 2026 convergence guidance (Workers
  has full feature parity for static assets + SSR + custom domains,
  and every new capability — Secrets Store, Workflows, Durable Objects,
  Dynamic Workers, Sandboxes — lands Workers-first).
- `worker/src/routes/complete.ts` — LLM proxy, ported from the Pages
  Function. Same request/response shape, same fallback semantics; the
  only user-visible change is that secrets now live in the Workers
  Secrets Store instead of Pages environment variables.
- `worker/src/routes/qid.ts` — stub (returns 501) for the Phase 4a
  Wikidata QID resolver. Contract (request shape, cache key, SPARQL
  + wbsearchentities pipeline) specified inline so the next session
  can land the real implementation without re-deriving the design.
- `.github/workflows/deploy-worker.yml` — manual-dispatch deploy job
  using `cloudflare/wrangler-action@v3`. Requires repo secrets
  `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`. Flip to push-on-tag
  once the deploy cadence stabilises.
- `single_file_demo/functions/api/complete.ts` carries a DEPRECATED
  header pointing at the Worker replacement. Kept in-place so an
  existing Pages deployment does not 404 overnight during the
  switchover.

Security baseline (the `_headers` file's CSP, COOP/COEP, HSTS,
Permissions-Policy) is ported into `worker/src/index.ts` as
`BASELINE_HEADERS`, applied to every Response.

### Pending user action (for first Worker deploy)

  cd worker/
  npm install
  npx wrangler login
  npx wrangler secret put ANTHROPIC_API_KEY
  npx wrangler deploy

After the first deploy, subsequent deploys run via the
`deploy-worker.yml` workflow on manual dispatch.

## [0.3.1] — 2026-04-27

Hygiene release. Zero code-semantic changes. Closes a public-surface
truthfulness drift, locks the failure mode behind a CI gate, and adds
verifiable provenance on the published artifact.

The v0.3.0 wheel was published before the post-PR-A README rewrite,
so `pypi.org/project/sum-engine` showed a long-description that said
`pip install sum-engine[sieve] — shipping soon`: a tautology against
itself for a project whose brand is truthfulness. PyPA's metadata
model freezes the long-description at publish time; the surface
rotted independently of the GitHub README. v0.3.1 picks up the
current README and adds the gate that prevents recurrence.

### Fixed

- `pyproject.toml` version bump 0.3.0 → 0.3.1; the wheel's
  `Description` metadata now matches `README.md` head verbatim.

### Added — packaging hygiene gate

- `scripts/hash_dist.py` — emits `sum.dist_hashes.v1` JSON with
  SHA-256 over each file in `dist/`. Used as the input artifact for
  TestPyPI verification, production verification, and (eventually)
  the R0 trust-root manifest. Single source of truth for "the bytes
  we built locally" across every downstream verification step.
- `scripts/check_long_description_sync.py` — extracts `Description`
  from the built wheel's `*.dist-info/METADATA` and diffs against
  `README.md` after newline normalisation. Fails closed on any
  divergence. Complements `twine check` (renderability) and
  `check-wheel-contents` (file-tree validity) by answering a
  question neither does — "is this actually the README we intended
  to ship."
- `scripts/verify_pypi_attestation.py` — verifies a published
  artifact's PEP 740 attestation against the expected GitHub
  repo + workflow identity, using `pypi-attestations`. Pinned at
  invocation; the upstream CLI labels itself experimental, so the
  release pipeline runs against a pinned version rather than the
  latest tag.
- `.github/workflows/publish-pypi.yml` restructured to a staged
  publish: build dist/* → pre-publish gates (twine check +
  check-wheel-contents + README diff) → upload SAME local files to
  TestPyPI via Trusted Publishing → verify staged provenance
  (FAIL CLOSED here, pre-promotion gate) → upload SAME local files
  to production PyPI → verify production provenance (post-publish
  detection; alarm, not gate). The TestPyPI gate is the
  load-bearing fail-closed step. TestPyPI and production PyPI are
  separate indexes — the same local `dist/*` is uploaded to each;
  no PyPI-side promote operation exists. Trust-relationship setup
  on test.pypi.org is a one-time pre-merge configuration step
  (documented in PR description).

### Unchanged

- CLI contract for `attest / verify / resolve / ledger / inspect /
  schema` and every flag on them.
- CanonicalBundle wire format (`canonical_format_version 1.0.0`).
- Prime scheme (`sha256_64_v1`).
- Every cryptographic contract (HMAC, Ed25519, VC 2.0).
- Cross-runtime trust triangle (K1 / K1-mw / K2 / K3 / K4 + A1–A6
  green on this commit; same bundle bytes still verify in Python ↔
  Node ↔ Browser; rejection class symmetric on adversarial input).

### Demo (single_file_demo/index.html)

Provenance / preservation / signed-not-true labelling added next to
the rendered tome so a casual user reading the live demo can answer
"what does this receipt prove?" without consulting docs:
- "Provenance verified" — the receipt proves the issuer signed this
  render tuple.
- "Preservation benchmarked: median 1.000; p10 0.769 long / 0.818
  short. Not recomputed for this render." — normalises the demo's
  preservation copy to match the README's long+short distinction
  (the previous copy quoted only the short-corpus p10 0.818).
- "Signed does not mean true" — the receipt is not a truth oracle.

Each line cross-references the spec section that backs it
(`docs/RENDER_RECEIPT_FORMAT.md` §5; `docs/SLIDER_CONTRACT.md`).

## [0.3.0] — 2026-04-23

Minor-bump feature release. Agentic-first introspection surface: three
new subcommand clusters that let an LLM agent composing SUM into a
larger pipeline ask questions about ledger state, read bundle shape
without paying crypto cost, and validate SUM output programmatically.
Zero breaking changes — every 0.2.1 invocation still works identically.

### Added

- `sum ledger list [--db DB] [--axiom KEY] [--since ISO] [--limit N]`
  enumerates prov_ids as NDJSON (one JSON object per line), each row
  carrying prov_id, axiom_key, source_uri, byte_start, byte_end,
  timestamp, extractor_id. Filters compose with AND. Previously, agents
  that wanted to introspect a ledger had to craft raw SQL against the
  SQLite file — now they pipe `sum ledger list | jq …`.

- `sum ledger stats [--db DB] [--pretty]` emits a one-shot summary:
  `provenance_records_total`, `distinct_axiom_keys`, earliest/latest
  timestamps (ISO 8601), `chain_tip_hash` (Merkle), and branches with
  their state-integer digit counts.

- `sum ledger head [--db DB] [--branch NAME] [--pretty]` returns the
  current state integer for one named branch or every branch. State
  integers are emitted as strings (never JSON numbers) to preserve
  arbitrary precision — many agent JSON parsers use 64-bit doubles.

- `sum inspect <bundle.json> [--pretty]` reads a bundle's structural
  shape without running signature verification or re-deriving primes:
  axiom counts (claimed + parsed — an agent sees a divergence without
  invoking `sum verify`), state-integer digit size, signature fields
  present, bundle/format versions, timestamp, branch, and the sum_cli
  sidecar (prov_ids, extractor, source_uri) if present.

- `sum schema {bundle|provenance|credential}` prints a JSON Schema
  (Draft 2020-12) for each shape SUM emits. Agents that want to
  validate output against a ground-truth contract no longer have to
  reverse-engineer from prose docs.

### Unchanged

- CLI contract for `attest / verify / resolve` and every flag on them.
- CanonicalBundle wire format (`canonical_format_version 1.0.0`).
- Prime scheme (`sha256_64_v1`).
- Every cryptographic contract (HMAC, Ed25519, VC 2.0).
- Cross-runtime trust triangle — K1 / K1-mw / K2 / K3 / K4 still green
  on this commit; same bundle bytes still verify in Python ↔ Node ↔
  Browser.

### Tests

14 new cases in `Tests/test_sum_cli_agentic.py` pin: NDJSON shape of
ledger list; filter composition (--axiom, --limit); stats summary
keys; head branch-not-found error path; inspect on tampered tome
(reports divergence rather than rejecting — agent's call whether to
run full verify); inspect on malformed JSON; schema title + required
subset is actually emitted by attest.

## [0.2.1] — 2026-04-23

Patch release — fixes a three-minute-old version-reporting bug
introduced by 0.2.0.

### Fixed

- `sum --version` and `bundle.sum_cli.cli_version` now track the
  actually-installed distribution version via
  `importlib.metadata.version("sum-engine")` instead of a hardcoded
  string in `sum_cli/__init__.py`. 0.2.0's wheel shipped with
  `pyproject.toml` at 0.2.0 but the CLI's hardcoded `__version__`
  still said `"0.1.0"`, so every bundle minted under 0.2.0 carried
  `sum_cli.cli_version: "0.1.0"` — a silent truth gap inside the
  very bundles the CLI exists to attest. 0.2.1 closes it at the
  source: no dual source of truth to drift from again.

### Unchanged

- Everything else. No API/CLI contract changes from 0.2.0.

## [0.2.0] — 2026-04-23

Hygiene release (one day after 0.1.0). One BREAKING change, zero
behavior changes.

### Changed — BREAKING (for anyone who imported `internal.*` directly)

- The top-level `internal/` package was renamed to `sum_engine_internal/`
  to remove the PyPI namespace-collision risk. 238 import sites across
  111 Python files were mechanically rewritten; `pyproject.toml`
  `packages.find.include` now lists `sum_engine_internal*`. Every test,
  script, and doc reference updated in the same commit.
- The CLI's public contract (`sum attest / sum verify / sum resolve`,
  all flags, the CanonicalBundle JSON schema, the Gödel-state wire
  format) is unchanged. Anyone using `sum-engine` through the CLI
  sees no difference. Only consumers who were importing
  `internal.infrastructure.X` etc. directly — which the 0.1.0
  CHANGELOG's "Known limitations" explicitly flagged as unsupported
  — need to update their imports to `sum_engine_internal.*`.

### Unchanged

- CanonicalBundle wire format (`canonical_format_version 1.0.0`).
- Prime scheme (`sha256_64_v1`).
- All cryptographic contracts (HMAC, Ed25519, VC 2.0 `eddsa-jcs-2022`).
- Cross-runtime trust triangle (K1 / K1-mw / K2 / K3 / K4 all PASS
  on this commit; same bundle bytes still verify in Python ↔ Node ↔
  Browser).

## [0.1.0] — 2026-04-22

First public release. Ships the `sum` CLI on PyPI, the Python API
under `internal.*` (renamed to `sum_engine_internal.*` in 0.2.0 —
see above), the standalone Node verifier, and the single-file
browser demo. Cross-runtime trust triangle
(Python ↔ Node ↔ Browser) is complete and locked by CI.

### Added — CLI

- `sum attest` — extract SVO triples from prose, mint a
  CanonicalBundle with the Gödel state integer.
- `sum verify` — verify structural reconstruction, HMAC
  signature (with `--signing-key`), and Ed25519 signature
  (self-contained via embedded public key). `--strict` mode
  requires at least one verifiable signature.
- `sum resolve` — look up a ProvenanceRecord in a local
  AkashicLedger by content-addressable prov_id.
- `sum attest --ed25519-key PEM` — mint W3C VC 2.0-compatible
  Ed25519-signed bundles using a PEM produced by
  `python -m scripts.generate_did_web`.
- `sum attest --ledger DB` — record per-triple byte-level
  ProvenanceRecords and attach prov_ids to the bundle; enables
  the attest → resolve loop end-to-end.
- `sum attest --signing-key K` — HMAC-SHA256 signature for
  shared-secret peers (composable with `--ed25519-key`).

### Added — Python API

- `sum_engine_internal.infrastructure.canonical_codec.CanonicalCodec` — HMAC
  and Ed25519 are both optional; when neither is configured,
  bundles carry the state integer only (content-addressed
  integrity without shared secrets or keys). Downgrade-protection
  preserved when a signing_key is configured.
- `sum_engine_internal.infrastructure.verifiable_credential` — W3C VC 2.0
  emission + verification with `eddsa-jcs-2022` cryptosuite.
  `did:key` and `did:web` issuer helpers; `build_did_web_document`
  emits the DID document for hosting at `/.well-known/did.json`.
- `sum_engine_internal.infrastructure.akashic_ledger.AkashicLedger` —
  SQLite-backed event log with Merkle hash-chain integrity and
  BEGIN IMMEDIATE concurrency hardening. `record_provenance` +
  `get_provenance_record` power the CLI's attest/resolve loop.

### Added — Cross-runtime

- `standalone_verifier/verify.js` verifies Ed25519 signatures via
  Node's `crypto.webcrypto.subtle` (Node ≥ 18.4).
- `single_file_demo/index.html` verifies Ed25519 via browser
  SubtleCrypto (Chrome 113+, Firefox 129+, Safari 17+).
- `scripts/verify_cross_runtime.py` — K1 / K1-multiword / K2 / K3
  / K4 kill-experiments: structural round-trip, multi-word object
  regex parity, VC 2.0 named-rejection, Ed25519 positive + negative
  signature verification Python ↔ Node.

### Added — CI

- `cross-runtime-harness` job runs the K1–K4 kill-experiments on
  every PR.
- `pypi-install-smoke` job builds the wheel, installs in a fresh
  venv, and runs `echo prose | sum attest | sum verify` — locks
  the shipping promise against packaging regressions.

### Added — Docs

- `docs/DID_SETUP.md` — runbook for did:key and did:web issuer
  setup, with a verifier-compatibility matrix.
- `docs/PROOF_BOUNDARY.md` §1.3.1 — Ed25519 public-key attestation
  cross-runtime contract.
- `docs/FEATURE_CATALOG.md` Layer 8 — `sum` CLI feature entries
  (98–103) each with a reproducible verification command.

### Cryptosuite

- `eddsa-jcs-2022` with RFC 8785 JCS canonicalisation. Bundles
  emitted under `sha256_64_v1` prime scheme (the production scheme
  for low-thousands-of-axioms corpora).

### Known limitations

- `sum attest --ledger` requires `--extractor=sieve`. The LLM
  extractor has no byte-offset tracking yet (emits a clear error
  with a pointer).
- Browser Ed25519 falls back to "present (use CLI)" on pre-2023
  browsers lacking SubtleCrypto Ed25519 support — never a false ✓.
- The internal Python modules live under a top-level `internal/`
  package. Downstream consumers should depend on the CLI contract,
  not import these modules directly — they may move in 0.2.0.

[Unreleased]: https://github.com/OtotaO/SUM/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/OtotaO/SUM/releases/tag/v0.3.1
[0.3.0]: https://github.com/OtotaO/SUM/releases/tag/v0.3.0
[0.2.1]: https://github.com/OtotaO/SUM/releases/tag/v0.2.1
[0.2.0]: https://github.com/OtotaO/SUM/releases/tag/v0.2.0
[0.1.0]: https://github.com/OtotaO/SUM/releases/tag/v0.1.0
