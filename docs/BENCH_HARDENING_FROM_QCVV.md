# BENCH_HARDENING_FROM_QCVV.md

Standalone brief for Claude Code. Distilled from Hashim et al., *Practical Introduction to Benchmarking and Characterization of Quantum Computers*, PRX Quantum 6, 030202 (2025) — a 132-page tutorial on quantum characterization, verification, and validation (QCVV). You do not need to read the paper. This document is the actionable surface for SUM.

## Scope

QCVV is the most methodologically mature tradition of benchmarking stochastic computational systems. The cryptographic surface of SUM (§1 of `docs/PROOF_BOUNDARY.md` — the K-matrix, A-matrix, JCS/Ed25519, hash-chain integrity) is mechanically proven and has no quantum analog; do not touch it. The empirical-benchmark surface (§2 of `docs/PROOF_BOUNDARY.md` and `docs/SLIDER_CONTRACT.md`) is where this document applies. Four concrete transfers, plus one hygiene task, all of which tighten what is currently the weakest face of the truthfulness contract.

## Non-goals — do not do these

- Do not import quantum vocabulary (Pauli, Choi, diamond distance, T1/T2, dephasing, twirling, gauge freedom) into SUM docs or code. The analogies are speculative; the engineering cost is high; the payoff does not beat the five tasks below.
- Do not attempt Fisher-information-style optimal experiment design for corpus construction. Out of scope.
- Do not translate process tomography or randomized-Clifford benchmarking. Different ontology.
- Do not introduce a "fidelity" alias for fact preservation. Keep SUM's existing names.

## Task sequencing

T1 produces data that T4 consumes. T2 and T3 are independent and can land in parallel. T5 should land before any of the above are cited as load-bearing in a release.

Recommended order: T5 → T1 → T4 → T2 → T3.

---

## T1 — Iterated round-trip (amplificational sensitivity) — CLOSED 2026-05-21

**Status:** **CLOSED.** Runner shipped at `scripts/bench/runners/s25_iterated_round_trip.py` 2026-05-18; K=10 receipts landed across all three seed corpora 2026-05-21. **All three corpora return composition verdict STABLE** (`max-vs-K=1 drift delta = 0.00pp ≤ ε=1.0pp`). PROOF_BOUNDARY §2.5.1 carries the receipts and the structured per-corpus tables. The §2.5 closure claim is empirically composition-stable on every measured corpus shape — first acceptance criterion of this task is met. Receipts:

- `fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json` (50 docs × K=10, median=mean=0.00 across K)
- `fixtures/bench_receipts/s25_iterated_K10_seed_v2_2026-05-21.json` (20 docs × K=10, median=mean=0.00 across K)
- `fixtures/bench_receipts/s25_iterated_K10_seed_long_paragraphs_2026-05-21.json` (16 docs × K=10, median=12.50 flat across K)

The classifier emits one of: **stable** (Δmax ≤ ε from K=1), **accumulating** (monotone growing), **saturating** (grows then plateaus), **noisy**, **insufficient_data**.

Original design below for context.

---


**Concept.** The single most powerful idea in QCVV experiment design: shot-count scales benchmark sensitivity as `1/√N`; *repetition of the noisy operation inside one experiment* scales it as `1/L`. Repeating the operation amplifies whatever drift exists, making it detectable far below the noise floor of single-shot measurement. The §2.5 closure result on `seed_v1` (drift 0.00%, exact-match recall 1.000) is a single-step measurement and tells you almost nothing about whether canonicalisation is closed under iteration or merely closed at the first fixed-point neighbourhood.

**Intervention.** New runner `scripts/bench/runners/s25_iterated_round_trip.py`. Parameters: corpus, K (default 10), pinned model snapshot (raise on unpinned, per `docs/PROOF_BOUNDARY.md` §2.6).

Algorithm per document:
```
axioms_0 = extract(text)                          # combined-intervention extractor
for k in 1..K:
    prose_k    = generate(axioms_{k-1})            # canonical-first generator prompt
    axioms_k   = extract(prose_k)                  # vocab-pinned Literal + lemma-exclusion
    drift_k    = 1 - exact_match_recall(axioms_k, axioms_0)
    record { k, drift_k, |axioms_k|, set_diff(axioms_k, axioms_0) }
```

Output: NDJSON receipt under new schema `sum.iterated_round_trip_drift.v1`, written to `fixtures/bench_receipts/s25_iterated_K10_<corpus>_<YYYY-MM-DD>.json`. Per-doc, per-k row; aggregate rows per (corpus, k) with median / p10 / max drift.

Run on all three corpora that §2.5 closed against: `seed_v1`, `seed_v2`, `seed_long_paragraphs`.

**Acceptance.** One of two outcomes, both informative:
- **Drift stays flat across k** (e.g., max drift_k ≤ max drift_1 + ε for ε ≤ 1pp). §2.5 closure is genuinely a fixed point. PROOF_BOUNDARY.md §2.5 gains a sentence: "closure is stable under K-step iteration; receipt at `fixtures/bench_receipts/s25_iterated_K10_*`."
- **Drift accumulates with k** (e.g., drift_k grows monotonically or super-linearly). §2.5 is qualified: the single-step result is a local neighbourhood, not a global fixed point. PROOF_BOUNDARY.md §2.5 gains an explicit composition caveat. This is *the* claim that needs qualification before any release that cites §2.5 as load-bearing.

**Cost.** ~10× the per-corpus cost of `s25_generator_side_combined` (i.e., ~$0.70–$2.00 per corpus). One sitting.

---

## T2 — Volumetric capability regions for the slider bench

**Concept.** Aggregate metrics (median, p10) hide the *shape* of failure. The right object is the metric as a function over a 2D shape parameter — what QCVV calls a "volumetric plot" and what its derivative "capability regions" formalise: contiguous regions of the shape plane where the system meets a threshold. The slider bench currently reports per-axis medians; the operationally meaningful artifact is a heatmap over `(corpus_complexity × slider_displacement)`.

**Intervention.** Extend the slider bench (currently invoked via `scripts/bench/run_paragraphs.sh` and `scripts/bench/run_long_paragraphs.sh`) to grid over two axes simultaneously:

- **Corpus complexity**: bucket by axioms-per-doc, e.g., `{1, 2–4, 5–9, 10–19, 20+}`.
- **Slider displacement**: for each LLM-conditioned axis (length / formality / audience / perspective), bucket `|Δ from 0.5|` into `{0.1, 0.25, 0.5, 0.75, 0.9}`.

Per cell: median strict-match fact preservation, NLI-audited weak-cell rate, n_observations. Emit under new schema `sum.slider_capability_region.v1` at `fixtures/bench_receipts/slider_capability_<axis>_<YYYY-MM-DD>.json`.

Compute the **capability region** per axis: the maximal contiguous set of (complexity_bucket × displacement_bucket) cells where median fact preservation ≥ threshold (default 0.95, configurable). Output the region as a list of cell coordinates plus a single ASCII heatmap in the receipt for human-readability.

**Acceptance.** `docs/SLIDER_CONTRACT.md` gains a "Capability region" section per axis, replacing the bare median/p10 summary as the headline. The headline becomes operational: "length-axis fact preservation ≥ 0.95 for documents with ≤19 axioms and `|Δ length| ≤ 0.75`" rather than "median 1.000 / p10 0.769". The latter stays as supporting detail.

**Cost.** Marginal over existing bench runs — same shots, different aggregation.

---

## T3 — Worst-case tail bounds for the render receipt's trust scope

**Concept.** Average-case and worst-case error tell you operationally different things. Any guarantee a downstream system can actually rely on — i.e., what a render receipt's trust scope should attest — is a worst-case high-probability bound, not a tail percentile of an empirical distribution. "Median 1.000" is a marketing claim; "fact preservation ≥ X with 95% confidence over the tested envelope" is a guarantee. The DKW inequality is the cheapest tool for the job and requires no assumptions beyond i.i.d. sampling within a (corpus × axis) cell.

**Intervention.** Post-processing pass over existing `sum.slider_drift_bench.v1` receipts. For each (corpus × axis) cell with n observations and empirical fact-preservation CDF F̂(x):

```
ε(n, δ) = sqrt(ln(2/δ) / (2n))          # DKW bound, two-sided
worst_case_95 = inf { x : F̂(x) ≥ 0.05 } - ε(n, 0.05)
```

Report `worst_case_preservation_95ci` per cell. Bump schema to `sum.slider_drift_bench.v2`; preserve v1 alongside per `docs/COMPATIBILITY_POLICY.md` (this is a minor bump — additive field). Receipt at `fixtures/bench_receipts/s25_slider_v2_<YYYY-MM-DD>.json`.

Update `docs/RENDER_RECEIPT_FORMAT.md` §5 (trust scope) to reference this bound, not the median. The receipt's attestation language becomes: *"the issuer attests, with 95% confidence over the tested slider envelope as defined by `sum.slider_capability_region.v1`, fact preservation ≥ X"* — citing the actual measured X per axis, not a hand-wave.

**Acceptance.** §5 of `docs/RENDER_RECEIPT_FORMAT.md` cites a numeric worst-case bound per axis. README's slider claim block ("Slider fact preservation: median 1.000, p10 0.769 ...") gains a third number: the worst-case 95% lower bound. PROOF_BOUNDARY.md §5 (truthfulness contract) explicitly permits the new language as `empirical-benchmark` because it is a measured bound, not an absolute guarantee.

**Cost.** Pure post-processing. No new LLM calls.

---

## T4 — Compositional metric audit for `drift_pct`

**Concept.** Some metrics compose cleanly across independent stages; others do not. The classical example: total variation distance does not compose under N independent samples, but classical (Hellinger) fidelity does — `F(p⊗N, p⊗N) = F(p,p)^N`. SUM reports `drift_pct` per corpus without analysis of its composition law. Before §2.5's single-step result can be cited as a load-bearing claim across multi-stage pipelines (extraction → generation → re-extraction → downstream consumer extracting again), the composition law of `drift_pct` must be either derived or empirically bounded.

**Intervention.** Write `docs/DRIFT_METRIC_COMPOSITION.md`. Three subsections:

1. **Definition.** Pin the exact computation of `drift_pct` currently used in the §2.5 receipts. Currently it appears to be `1 - exact_match_recall(axioms_predicted, axioms_truth)`; confirm or correct against `scripts/bench/runners/s25_generator_side.py`.

2. **Composition (empirical).** Using T1's iterated-round-trip data, fit `drift_pct(K)` as a function of K. Test against candidate composition laws: additive (`drift_K = K · drift_1`), multiplicative-survival (`(1 - drift_K) = (1 - drift_1)^K`), and saturating (`drift_K = drift_∞ · (1 - exp(-K/τ))`). Report best-fit and goodness-of-fit per corpus.

3. **Alternative metric.** Compute Hellinger fidelity over the multinomial distribution of axiom keys (treating each axiom set as a sparse categorical) on the same data. Report whether its empirical composition under K iterations matches the analytical `F^K` law within DKW bounds. If yes, propose Hellinger-on-axioms as the *secondary* metric in `docs/SLIDER_CONTRACT.md` and PROOF_BOUNDARY.md §2.5, retaining `drift_pct` as the primary for backward compatibility but with an explicit cross-reference.

**Acceptance.** `docs/PROOF_BOUNDARY.md` §2.5 cites either (a) an empirical composition bound for `drift_pct` with confidence interval, or (b) a switch to a compositional metric, with full justification. No load-bearing multi-stage claim in any release until this lands.

**Cost.** Pure analysis over T1's receipts. No new LLM calls.

---

## T5 — Negative-control corpus (assumption-violation detector)

**Concept.** A benchmark with no documented failure mode is not a benchmark; it cannot distinguish "the system passed because it is good" from "the system passed because the test is too easy." Every QCVV protocol is paired with an explicit out-of-model regime where it is expected to break, and the protocol is run against that regime as a control. SUM has no such corpus.

**Intervention.** New corpus `seed_negative_control_v1` under `corpora/` (or wherever `seed_v1`, `seed_v2`, `seed_long_paragraphs` live — confirm path). Hand-write 20–40 documents engineered to violate the canonicalisation pipeline's assumptions:

- **Ambiguous coreference**: "Alice told Beth she had won" — `won(Alice)` and `won(Beth)` are both syntactically defensible.
- **Predicate aliases that resolve inconsistently**: documents where the same relation is expressed with two predicate phrases, where the choice of canonical form is genuinely arbitrary.
- **Contradictory axioms within one document**: "Alice was born in 1990. Alice was born in 1991."
- **Entity-resolution adversarial**: surface forms that map to multiple Q-IDs under `/api/qid` with comparable scores.
- **Non-extractable assertions**: hedges, counterfactuals, questions phrased as statements.

Each document is annotated with its expected-failure mode (one of the five above) plus expected behaviour: which step (extract / generate / re-extract) should fail and how.

Run the full §2.5 pipeline, slider bench, and `/api/qid` resolver against `seed_negative_control_v1`. Add a runner that exits 0 if observed failures match annotations and exits 1 otherwise (i.e., the negative control is a *test* the bench must pass, with the success criterion being correct failure detection).

Wire into `make bench` and CI.

**Acceptance.** `seed_negative_control_v1` lives in the repo with annotations. `make bench` includes a `negative-control` target. CI fails when the bench either (a) succeeds on inputs it should fail on, or (b) fails on inputs it should succeed on. PROOF_BOUNDARY.md §2 gains a "Negative controls" subsection citing the corpus.

**Cost.** One sitting of corpus authoring (~half day). Bench wiring is mechanical.

---

## Receipt schema summary

New or modified schemas introduced by this plan:

| Schema | Status | Source of truth |
| --- | --- | --- |
| `sum.iterated_round_trip_drift.v1` | new (T1) | `docs/PROOF_BOUNDARY.md` §2.5 (extended) |
| `sum.slider_capability_region.v1` | new (T2) | `docs/SLIDER_CONTRACT.md` (extended) |
| `sum.slider_drift_bench.v2` | minor bump from v1 (T3) | `docs/SLIDER_CONTRACT.md` |
| `sum.qid_resolution_accuracy.v1` | existing — no change | already shipped |

All new receipts follow the existing convention: NDJSON, pinned-model-snapshot field mandatory, `fixtures/bench_receipts/<schema_short>_<corpus?>_<YYYY-MM-DD>.json`, reproducible with one command.

## Truthfulness-contract compliance

Every claim added to SUM docs by this plan must carry its epistemic status per `docs/PROOF_BOUNDARY.md` §5:

- T1's iteration-stability result: `empirical-benchmark`.
- T2's capability regions: `empirical-benchmark`.
- T3's DKW worst-case bound: `empirical-benchmark` (it is a measured statistical bound, not an absolute guarantee).
- T4's composition law: `empirical-benchmark` if measured; `provable` only if derived analytically and the derivation is in-doc.
- T5's negative-control detection: `certified` (it is a property of the test suite itself, not the system under test).

Do not use the word "guarantee" anywhere in this plan's outputs without a same-commit benchmark receipt or a §5-compliant proof citation.
