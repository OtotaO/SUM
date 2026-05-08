# Reviewer-anticipation Q&A

A list of questions a careful reviewer would ask about
`docs/arxiv/sheaf-detector-note-v0.md`, with prose answers and
pointers to the section / receipt that addresses each. Treat this as
a draft revision-cycle response: every Q has a §-pointer or a
receipt digest as the load-bearing artefact.

The verification chain is reproducible:

```bash
make reproduce-preprint
```

…runs every receipt-pinning test and verifies the prose-to-receipt
citation chain. 17 cited receipts, 6 named digest cites, all OK on
HEAD. Drift would surface as a failing test.

---

## Q1. Why should I believe the cross-runtime trust triangle holds, given that "byte-identical Ed25519 over JCS bytes across three runtimes" is a strong claim?

The K1–K4 valid-path matrix and A1–A6 adversarial matrix are **locked
in CI on every PR** (`make xruntime`, `make xruntime-adversarial`).
A new bundle that breaks byte-identity on any of the three runtimes
fails the gate. The harness is at `scripts/verify_cross_runtime.py`
and `scripts/verify_cross_runtime_adversarial.py`; the matrix's
exact rejection-class semantics are documented in PROOF_BOUNDARY
§1.2 / §1.3.1 and exercised on the worker / Node / browser
codepaths.

## Q2. The `bench_digest` claim of cross-machine reproducibility is unusual. What exactly is reproduced, and across what environments?

Four load-bearing benches reproduce **byte-for-byte**: `v3_2_validation`
(`b4d26c01…ed5a`), `complementary_hybrid` (`dc6e0260…343ce`),
`predicate_negatives` (`ddf41484…001f59c3`), `hybrid_comparison`
(`a7965803…6c2003` on Apple Accelerate / OpenBLAS-x86, *or*
`7fac833a…3bcb97` on OpenBLAS-arm — both stable per architecture;
see PROOF_BOUNDARY §2.9 for the cross-architecture two-digest pin).
Across three environments: Apple Accelerate / Apple Silicon
(operator); OpenBLAS Py 3.10 / numpy 1.25 / x86_64 (Modal); OpenBLAS
Py 3.12 / numpy 2.x / x86_64 (Modal). The harness is
`scripts/research/cross_machine_verify_modal.py`. PROOF_BOUNDARY §2.10
documents this as continuous-enforcement against drift; PR #160 is
the worked example of catching a real determinism bug (dict
iteration order between in-memory and JSON-cached snapshots).

## Q3. The synthetic-bench WIN (Δ=+0.043) is at AUC ~0.876. How is this not just narrow-corpus overfitting?

It mostly is, and **the preprint says so explicitly** in §4.7.x and
§7.2. The synthetic harness is one specific perturbation
distribution on one specific 16-doc corpus's specific source-triple
set; the hybrid was selected to compose well on that distribution.
The §4.7.2-§4.7.4.1 cross-family + cross-corpus extension *measured*
that the WIN does not generalise to real LLM perturbations on any
of three measured corpora at $n \geq 16$. We name this a Goodhart
artifact in regressional form (Manheim & Garrabrant, 2018) — the
substrate's truth-first discipline caught it; the hybrid retains
its real-on-the-synthetic-harness status, but is not advertised as
universal. Receipt:
`fixtures/bench_receipts/path2_cross_corpus_compare_2026-05-06.json`,
schema `sum.sheaf_path2_cross_corpus_compare.v1`.

## Q4. The §4.7.4 section title says "§4.7.3 is corpus-specific" but §4.7.4.1 says "actually it isn't, the corpus-specificity was small-n noise." Which is the load-bearing claim?

§4.7.4.1's resolution is load-bearing: at controlled sample sizes
($n \geq 16$), no LLM family produces a `HYBRID_BEATS` verdict on
real-LLM perturbations across three corpora. The §4.7.4 title now
reads "an apparent corpus-specificity (resolved in §4.7.4.1)" to
make the journey explicit. The lone BEATS cell on `seed_paragraphs`
× gpt-4o-mini at $n=8$ was extremal Goodhart on a sharp threshold;
extending to $n=16$ on the same encyclopedic style flips it to TIES.
Both per-doc walks are pinned-by-digest; reproducing the §4.7.4.1
extension is a one-line make target away.

## Q5. "Continuous-enforcement against drift" sounds vague. Does it actually catch real bugs, or is it just a slogan?

It caught at least one real bug *that the prior arc had
misdiagnosed*. PR #160 closed task #22 ("Phase 1 / Phase 2
same-process digest contamination"). The prior session had attributed
the digest drift to BLAS / asyncio state pollution — neither
hypothesis was correct. A probe-style investigation around the
digest pin found that in-memory snapshots from Phase 1 had docs in
*corpus order*, while cached snapshots from JSON round-trip had
them in *alphabetical order*; the trained sheaf consumed the
iteration order, producing different digests for the same snapshot
bytes. A one-line fix (`docs_with_src.sort(key=lambda x: x[0])`)
plus a regression test (`test_path2_v3_bench_invariant_to_snapshot_dict_order`)
closed the gap. PROOF_BOUNDARY §2.10 frames this generalisation
explicitly using Sachs et al. 2004 on biological mutualism
breakdown as the analogue.

## Q6. Cross-corpus is bounded: only three corpora at n≥16, and only one (`seed_news_briefs`) is stylistically distinct from encyclopedic. Doesn't that limit the structural claim?

Yes, and §4.7.4.1's "Honest scope" subsection names this directly:
*A robust corpus-independent claim would want 5–10 corpora spanning
genres (scientific abstracts, fiction, legal/policy, code commentary,
spoken transcripts).* Listed in §8 future work as "Deeper corpus
sampling." The current bound is sufficient to *falsify*
corpus-invariance (it didn't); it's a weak-positive claim, not a
strong-positive one. The honest reading we publish: at controlled
sample sizes on three measured corpora, the synthetic-bench WIN
does not generalise to any LLM family in the cross-organisational
sample.

## Q7. The merge bottleneck at N=10000 is 23.9 seconds. How does that scale to a real corpus?

Empirically measured scaling: $k_\text{merge}=1.497$ (sub-quadratic)
over six measured sizes (§4.9). Library-scale extrapolation:

| Target | $N$ | per-iter |
|---|---:|---:|
| small book | 1k | 0.62 s |
| medium book | 5k | 7.12 s |
| large book | 10k | 20.6 s |
| small library | 50k | 4.2 min (extrap) |
| modest library | 100k | 12.5 min (extrap) |

Phase 26 (property-graph backing store) is named as the gating
dependency for library-scale workloads above ~50k axioms. Receipt:
`fixtures/bench_receipts/performance_characterisation_2026-05-07.json`.

## Q8. The recursive-compression / SUM measurement (§4.10–§4.10.1) lists a "model-stable across LLM families" finding. How does it survive the same Goodhart concerns as §4.7.x?

It survives because the §4.10 metric is *continuous*, not
*thresholded*. §4.7.x classified verdict-classes (BEATS at $+0.030$,
LOSES at $-0.020$) — sharp boundaries that small-n noise pushed
cells across. §4.10 measures recall (a real number); the SUM at
threshold $\tau$ is the smallest-$n_\text{axioms}$ walk-step with
recall ≥ τ. Across 5 LLM families × 2 corpora × 16 docs = 160
cells, every cell reaches `SUM_AGREES_ALL_MODELS` at every threshold
tested. The variation across families is in the recall trajectory
(~0.2 spread), not in whether a SUM exists. There is no extremal-
Goodhart risk at the threshold because the SUM-existence claim is
monotonic in $\tau$.

## Q9. The deterministic-arm finding in §4.10 — "sieve(canonical_tome(A)) ≠ A" — sounds like a bug in the substrate. Is it?

It's an *architectural asymmetry the project did not previously
document*, not a bug. PROOF_BOUNDARY §1.1's "lossless round-trip"
claim holds at the *state-integer* level (parse(canonical_tome(S))
decodes to S's state); §4.10 measures that it does NOT hold at the
*triple* level (the canonical tome's bare-lemma rendering, "The
alice like cat", is not sieve-extractable because the verb is
uninflected). Both rendering paths have valid uses. Which is
appropriate depends on whether the downstream consumer needs
state-integer round-trip or triple-level round-trip. The §4.10.1
Arm 2 (LLM grammatical render) is the right operator for the
recursive-compression workload; the deterministic canonical tome
remains right for state-integer-anchored attestation.

## Q10. F3 STRUCTURAL FAIL is named as load-bearing. What does it actually prove?

That the v3.1 boundary-deviation signal is *mathematically blind*
to entity-set-preserving perturbations (A2-class predicate flips).
Not parametrically blind — *mathematically*. The cochain channel
discards predicate information by construction (§3.7). v3.2 closes
the *detector-layer* problem only by composition with
$v_\text{laplacian}^w$; standalone v3.1 deviation cannot be
rescued by retraining (the `aa34b6e8…` digest pin in
`Tests/research/test_recovery_experiment_digests.py` measures this:
predicate-perturbation training negatives do NOT lift A2). Future-
work item: *cochain redesign that propagates render content into
the interior* (§8). This is the real research direction the F3
finding opens.

## Q11. The §4.7.x audit found a misdiagnosed bug (Phase 1 contamination → actually dict-order). Are there other places in the preprint where the prose is ahead of the measurement?

The §7 four-tier audit was added precisely to surface this concern
(§7.1 holds-up; §7.2 corrected; §7.3 real-but-narrow; §7.4 limits).
§7.2 names two cases explicitly: §4.7.3 originally read as
universal-no-BEATS (corrected in §4.7.4.1 to "no BEATS at $n \geq
16$"); the synthetic-bench WIN was originally implicit-universal
(corrected to the Goodhart artifact reading). The preprint's
practice is to record findings as they were *and* their corrected
form. New corrections should land the same way: as a §7.2 entry
plus updated downstream references.

## Q12. How do I verify any of this from a fresh checkout?

```bash
git clone https://github.com/OtotaO/SUM
cd SUM
make install
make reproduce-preprint
```

The `reproduce-preprint` target runs every receipt-pinning test
and verifies the prose-to-receipt citation chain (`make
verify-preprint`). 15 receipt-pinning tests pass on HEAD; the
citation-chain verifier reports `17 ok, 0 drift, 0 missing`.

For the LLM-mediated arm of §4.10, the per-step rendered prose is
captured at `fixtures/bench_renders/recursive_walk_<corpus>_<model>.json`;
re-running Phase 2 against the cached snapshot is byte-deterministic
without API keys. Phase 1 (capture) requires an OpenAI / HF / Anthropic
key per family.

## Q13. What's not in the paper that the next version should address?

Named in §8 future work, in priority order:

1. **Real-LLM-aware per-triple V training** — the natural response
   to the Goodhart finding.
2. **Naturalistic perturbation synthesis** — decouple *what gets
   perturbed* from *how it propagates*.
3. **Deeper corpus sampling** — 5-10 stylistically distinct corpora
   to harden cross-corpus claims.
4. **Importance-weighted SUM** — replace min-by-$n_\text{axioms}$
   with information-content-weighted optimisation.
5. **Multi-modal compression dispatch** — different compressors
   matched to different content types (axioms / parables / poetry /
   quotes / emoji).
6. **Phase 26 property-graph backing store** — gates library-scale
   workloads above ~50k axioms.
7. **Cochain redesign** — root-cause F3 STRUCTURAL FAIL.
8. **Additional LAPACK environments** — Intel MKL, ARM Linux
   OpenBLAS, AMD AOCL.

## Q14. Why publish this now, if (1)-(3) above would tighten the load-bearing claims?

The publishable contribution is the *substrate plus the
methodology that produced the hybrid*, not the hybrid as a final
detector. The methodology (truth-first STOP-THE-LINE; named
falsifications; receipt-pinned digests; Goodhart-aware honest
scope) is what makes (1)-(3) tractable for the next version. Each
of those v0.4+ items is a clean follow-up arc; publishing the
methodology now lets others use it. The preprint also documents
several substantive negative-going findings (Goodhart explanation
of synthetic-vs-real; sieve↔canonical-tome asymmetry; F3 STRUCTURAL
FAIL diagnosis) that wouldn't survive being rolled into a longer
arc — they're cleaner as standalone results.

## Q15. The work cites a lot of biological / ecological / evolutionary literature (Sachs 2004, Manheim & Garrabrant, etc.). Is the metaphorical framing load-bearing or ornamental?

Both, named explicitly in scope. The Manheim & Garrabrant Goodhart
taxonomy is *load-bearing* in §4.7.4.1 — extremal Goodhart is the
specific mechanism we identify behind the lone BEATS cell, with
specific implications (sharp thresholds + small-n + adversarial
optimization → spurious crossings). Sachs et al. 2004 is *framing*
— the analogy of mutualism breakdown to substrate drift gives a
vocabulary for what continuous-enforcement is doing, but the
substrate's properties are specified independently and verified
mechanically (§4.8). PROOF_BOUNDARY §2.10's prose names this
distinction.
