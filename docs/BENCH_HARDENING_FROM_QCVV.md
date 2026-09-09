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

## T1 - Iterated round-trip - data collected; interpretation corrected 2026-09-09

**Status:** Three K=10 receipts landed on 2026-05-21. Their recorded median drift is unchanged across K=1..10. That aggregate does not establish document-level stability or composition invariance. The historical `STABLE` labels describe the runner's aggregate rule only. The corrected T4 analysis in [DRIFT_METRIC_COMPOSITION.md](DRIFT_METRIC_COMPOSITION.md) reports endpoint pairing and supersedes the stronger closure interpretation.

- `fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json`: 50 documents, median 0% at every K; mean 4% at K1 and K10.
- `fixtures/bench_receipts/s25_iterated_K10_seed_v2_2026-05-21.json`: 20 documents, median 0% at every K; mean 15% at K1 and K10; one document worsens by 100 percentage points while another improves by 100 points.
- `fixtures/bench_receipts/s25_iterated_K10_seed_long_paragraphs_2026-05-21.json`: 16 documents, median 12.5% at every K; mean rises from 16.7584% to 21.0987%; five documents worsen and the observed maximum rises from 42.8571% to 50%.

These are reaggregations of existing proxy measurements, with no new model calls or independent human assessment. The reference is the initial extracted axiom set; upstream extraction omissions are not measured.

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

**Acceptance, corrected 2026-09-09.** Publish paired per-document changes and per-K counts, medians, means, observed maxima, and additions. Flat aggregate curves may be described as flat on the measured sample. A fixed-point or noninferiority claim additionally requires a predeclared practical margin, independent document sampling, a justified paired inference design, and correction for selected or multiple comparisons. The existing three receipts meet the descriptive reporting requirement; they do not meet that inferential requirement.

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

## T3 - Population-tail bounds for the render receipt's trust scope - OPEN

**Correction, 2026-09-09.** The former formula subtracted DKW epsilon from a measured preservation value, mixing probability and value units. It must not be implemented or cited. A population quantile bound is also not an absolute worst-case guarantee for every document.

**Concept.** For n independent, identically distributed document observations under a fixed policy, DKW bounds the difference between the population and empirical CDFs. A proposed lower bound on the population q-quantile must shift the **quantile level** and invert the empirical CDF, without linear interpolation:

```
epsilon(n, delta) = sqrt(ln(2 / delta) / (2 * n))
q_lower = q - epsilon(n, delta)
quantile_lower = empirical_CDF_inverse(q_lower)  if q_lower > 0
                 0                              otherwise  # bounded preservation in [0, 1]
```

At a nonpositive shifted level, the lower bound is vacuous. State q and delta separately: a 5th-percentile lower bound with 95% confidence is not a claim that every document preserves that much. Allocate the error probability across multiple cells or claims, and justify independence at the document level; treating iterations from one document as independent samples does not supply it. See [Massart's DKW result](https://projecteuclid.org/journals/annals-of-probability/volume-18/issue-3/The-Tight-Constant-in-the-Dvoretzky-Kiefer-Wolfowitz-Inequality/10.1214/aop/1176990746.short).

**Intervention.** Require committed production-loop measurements, a documented sampling design, fixed scorer/policy, per-cell sample counts, chosen q/delta and multiplicity allocation. Validate the empirical-CDF inverse on discrete and boundary cases before emitting a versioned quantile-bound field. Historical artifacts remain unchanged; amended guidance must be linked explicitly.

**Acceptance.** Documentation cites the precise population quantity, assumptions, source data, and valid confidence procedure. No numeric production bound or new receipt is claimed by this plan. Any change to render trust-scope language waits for that evidence; provenance alone does not establish quality.

**Cost.** Post-processing is inexpensive, but obtaining an appropriate production sample is separate work.

---

## T4 - Compositional metric audit - descriptive analysis corrected 2026-09-09

**Status:** The runner emits `sum.drift_metric_composition.v2`: descriptive median-curve fits, per-K observation counts and drift summaries, and paired K1-to-Kmax document changes. It hash-binds each input T1 receipt and explicitly reports that population inference and equivalence testing were not performed. Stable medians coexist with worsening documents in two measured corpora. There is no established multi-stage closure guarantee.

The historical `fixtures/bench_receipts/drift_composition_2026-05-22.json` (`sum.drift_metric_composition.v1`) remains byte-identical. Its composition-invariance interpretation is superseded by the dated erratum in [DRIFT_METRIC_COMPOSITION.md](DRIFT_METRIC_COMPOSITION.md#historical-erratum-2026-09-09). Neither the old comparison of drift differences to DKW epsilon nor later confidence-interval overlap establishes equivalence. The document-frequency coefficient is a count-share diagnostic on the same observations, not independent confirmation or a derived composition law.

Runner: `scripts/bench/runners/t4_drift_composition.py`. The corrected analysis is reproducible from the three existing T1 receipts with no new model calls. It does not rederive source-level scores or add independent human assessments.

**Acceptance.** The descriptive reporting correction is complete. Any future population or equivalence claim remains open pending a predeclared margin, document-level paired design, adequate independent sampling, multiplicity treatment, and independent source annotations covering omissions and additions. No release may infer that individual documents preserve meaning from unchanged corpus medians.

**Cost.** Pure analysis for this correction; new empirical validation is separate.

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
| `sum.drift_metric_composition.v2` | corrected descriptive analysis (T4); v1 retained | `docs/DRIFT_METRIC_COMPOSITION.md` |
| `sum.qid_resolution_accuracy.v1` | existing — no change | already shipped |

All new receipts follow the existing convention: NDJSON, pinned-model-snapshot field mandatory, `fixtures/bench_receipts/<schema_short>_<corpus?>_<YYYY-MM-DD>.json`, reproducible with one command.

## Truthfulness-contract compliance

Every claim added to SUM docs by this plan must carry its epistemic status per `docs/PROOF_BOUNDARY.md` §5:

- T1's iteration-stability result: `empirical-benchmark`.
- T2's capability regions: `empirical-benchmark`.
- T3's proposed population-quantile confidence bound: `empirical-benchmark` only after the stated sampling and inference gates are met.
- T4's current result: `empirical-benchmark` (descriptive only), with no population inference or composition-equivalence claim.
- T5's negative-control detection: `certified` (it is a property of the test suite itself, not the system under test).

A benchmark receipt authenticates its recorded evidence; its existence alone does not justify a guarantee. Any statistical claim additionally requires a valid procedure, supported assumptions, a precise population quantity, and its stated limits.
