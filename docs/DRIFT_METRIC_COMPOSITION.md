# Drift-metric composition audit

**T4 correction, 2026-09-09.** The observed median drift is unchanged over K=1..10 on the three committed T1 corpora. Individual documents do worsen. These data do not establish composition invariance, equivalence, or preservation of a source's complete meaning.

Runner: [`scripts/bench/runners/t4_drift_composition.py`](../scripts/bench/runners/t4_drift_composition.py). Tests: [`Tests/test_t4_drift_composition.py`](../Tests/test_t4_drift_composition.py). The [corrected analysis](../fixtures/bench_receipts/drift_composition_2026-09-09.json) uses `sum.drift_metric_composition.v2`, with exact source-receipt SHA-256 digests, descriptive per-iteration statistics, and paired endpoint changes. This is post-processing of existing measurements; it performs no new generation, scorer replay, or independent human assessment.

## 1. What is measured

T1 records per-document, per-iteration:

```
drift_pct = 100 * (1 - exact_match_recall(axioms_predicted, axioms_initial))
```

`axioms_initial` is the axiom set extracted from the original document. It is not an independently annotated inventory of source facts. Initial extraction omissions, missing qualifiers, and other source meaning absent from this representation are outside the denominator. Added axioms are recorded separately as `n_extra` and do not increase this recall-only drift metric.

T4 divides `drift_pct` by 100 to report fractions. Each T1 document is repeatedly generated and re-extracted under the recorded canonical-generation policy. Repeated iterations of one document are not independent documents.

## 2. Descriptive result on the committed T1 corpora

The source receipts were generated on 2026-05-21. The following values are recomputed from their per-document records, including their recorded rounding:

| Corpus | Complete K1/K10 pairs | Median at every K | Mean K1 | Mean K10 | Worsened / improved / unchanged | Maximum K1 | Maximum K10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed_v1 | 50 | 0% | 4% | 4% | 0 / 0 / 50 | 100% | 100% |
| seed_v2 | 20 | 0% | 15% | 15% | 1 / 1 / 18 | 100% | 100% |
| seed_long_paragraphs | 16 | 12.5% | 16.7584% | 21.0987% | 5 / 0 / 11 | 42.8571% | 50% |

The median is a description of the center of each recorded corpus. It can stay unchanged while individual documents deteriorate. In `seed_v2`, one document changes from 0% to 100% drift and another from 100% to 0%. In `seed_long_paragraphs`, the average paired increase is 4.3403 percentage points and the largest individual increase is 16.6666 points. These are observed changes, without population confidence intervals.

The runner pairs K=1 with the declared final iteration of the same document, regardless of row order. It discloses and excludes missing endpoint pairs rather than substituting zero or the last available iteration. Per-K observation counts remain visible. An input missing an entire declared iteration cannot support the curve fit and is rejected.

### Candidate median curves

The runner retains additive, multiplicative-survival, saturating, and constant (`fixed_point`) candidate curves. `best_law_by_ssr` names the smallest residual fit to observed **medians**, with a deterministic tie-break preferring the constant curve. It is an exploratory curve description, not a statistical test, validated mechanism, or evidence of a document-level fixed point. The saturating candidate has two fitted parameters, while the other candidates condition on the observed first median.

### Document-frequency coefficient

The secondary statistic is the squared Bhattacharyya coefficient, `F(p, q) = (sum sqrt(p_i * q_i))^2`, over each document's share of the total axiom count. The source receipts do not retain the full axiom identities needed for an axiom-key distribution. Only complete endpoint pairs enter this statistic; an empty total count yields an unavailable coefficient.

| Corpus | F at K1 | F at K10 | Absolute change |
|---|---:|---:|---:|
| seed_v1 | 1.000000 | 0.980000 | 0.020000 |
| seed_v2 | 0.941176 | 0.941176 | 0.000000 |
| seed_long_paragraphs | 0.996812 | 0.995057 | 0.001755 |

The former claim that all changes were at most 0.002 was incorrect for `seed_v1`. Count shares can agree while axiom identities change. This statistic uses the same observations as drift and is not independent confirmation. The retained `F1**K` comparison is only a candidate curve: these iterated transitions have no established tensor-product structure that would derive that law. No hypothesis is declared rejected by this calculation.

## Historical erratum: 2026-09-09

[`drift_composition_2026-05-22.json`](../fixtures/bench_receipts/drift_composition_2026-05-22.json), schema `sum.drift_metric_composition.v1`, remains byte-identical as historical evidence. Its SHA-256 is `cfc53e67fa5aa19f2068a7ccdb1a70fb00789f4d9502a6d9a072b542fa9acad1`. The interpretation of composition invariance in that artifact and the former version of this document is superseded, not silently amended or re-signed.

The historical procedure compared drift-value differences with DKW epsilon, which is measured in CDF-probability units. A later runner used overlap between separate DKW median intervals. Neither procedure establishes equivalence. Overlapping confidence intervals are not an equivalence test; separate 95% intervals also do not by themselves provide simultaneous 95% coverage over all iterations. Linear-interpolated percentiles were not the empirical-CDF inverse required by that argument. The [DKW-Massart result](https://projecteuclid.org/journals/annals-of-probability/volume-18/issue-3/The-Tight-Constant-in-the-Dvoretzky-Kiefer-Wolfowitz-Inequality/10.1214/aop/1176990746.short) concerns the empirical CDF of independent, identically distributed observations; it does not supply a composition law.

V2 retires `composition_invariance`, `dkw_per_K_95`, and `all_composition_invariant_dkw_95`. Their replacements are `median_stability`, `descriptive_per_K`, `paired_endpoint_changes`, and `all_observed_medians_unchanged`. `method.population_inference` and `method.equivalence_test` explicitly say `not_performed`. The `supersedes_interpretation` record links the historical artifact and this erratum. Consumers of the research analysis must inspect the schema instead of treating a changed field name as a new confidence claim.

Recompute corrected analysis from the original T1 receipts, with no model calls:

```bash
python -m scripts.bench.runners.t4_drift_composition \
  --out fixtures/bench_receipts/drift_composition_2026-09-09.json --pretty
```

The output binds the exact bytes of each input receipt. A hash verifies which evidence was analyzed; it does not establish that the original scorer was correct or its sample representative.

## 3. What a future composition claim needs

Predeclare the target population, complete transformation policy, practical equivalence or noninferiority margin, endpoint, and handling of missing data. Use document-level pairing and a justified sampling/inference design, accounting for multiple endpoints or selected comparisons. Measure additions and loss of critical claims against independent source annotations, including qualifiers and exceptions. Report the maximum **observed** loss separately from any population-tail statement.

Until that work exists, the supported finding is restricted to descriptive median stability and the paired changes above, on these recorded corpora and K=1..10. No claim follows for K>10, other corpora or policies, per-document safety, or complete meaning preservation.
