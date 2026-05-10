# SMT-backed axiom-consistency spike findings

The substantive-math agent's #1 pick from the wide-net survey
(Z3 + Nelson-Oppen 1979 / De Moura-Bjørner CDCL(T) TACAS 2008),
implemented as a substrate-shaped check that catches axiom-set
contradictions before signing.

## What landed

  - `sum_engine_internal/research/smt_consistency/consistency.py`
    — `check_consistency(triples, predicate_properties=…)`
    returning a `ConsistencyResult` with `consistent`,
    `unsat_core`, and `z3_check_seconds`. Four predicate-property
    schemas via `PredicateProperty` enum:
    `ANTISYMMETRIC`, `IRREFLEXIVE`, `FUNCTIONAL`, `TRANSITIVE`.
  - `scripts/research/smt_consistency_substrate_spike.py` —
    two-experiment harness emitting
    `sum.smt_consistency_substrate_spike.v1`. Includes a starter
    `SUBSTRATE_PREDICATE_LIBRARY` (operator-vetted; conservative).
  - `Tests/test_smt_consistency.py` — 15 contract tests covering
    each property, UNSAT-core minimality, composition, edge cases.
  - `z3-solver` added as an optional dependency; tests
    `pytest.importorskip("z3")` so CI without it skips cleanly.

## Experiment 1 — synthetic verification matrix (PROVABLE kernel)

Six canonical cases × known expected verdict:

| case                         | n  | expected      | got           | UNSAT-core | time   |
|------------------------------|---:|---------------|---------------|-----------:|-------:|
| clean_no_props               |  1 | CONSISTENT    | CONSISTENT    |          0 |  23 ms |
| mutual_antisymmetric         |  2 | INCONSISTENT  | INCONSISTENT  |          2 |   2 ms |
| self_loop_irreflexive        |  1 | INCONSISTENT  | INCONSISTENT  |          1 |   1 ms |
| two_outputs_functional       |  2 | INCONSISTENT  | INCONSISTENT  |          2 |   1 ms |
| 3_cycle_transitive+irreflex  |  3 | INCONSISTENT  | INCONSISTENT  |          3 |   1 ms |
| needle_in_haystack (50+1)    | 51 | INCONSISTENT  | INCONSISTENT  | **1 (just the contradicting axiom)** | 4 ms |

**All six cases produce the correct verdict + minimal UNSAT
core.** The "needle in haystack" case is the load-bearing
substrate property: if 50 clean axioms surround 1 contradiction,
Z3's UNSAT core points exactly at the contradicting one.
Z3-CDCL(T) decides the substrate's QF_UF + EUF + small-quantified
fragment in milliseconds.

## Experiment 2 — substrate corpus check (HONEST gap surfaced)

Run on four labeled corpora with the starter
`SUBSTRATE_PREDICATE_LIBRARY` (parent_of, born_on, ancestor_of,
married_to, child_of, born_in):

| corpus               | triples | predicates_with_properties | verdict     | time |
|----------------------|--------:|----------------------------|-------------|-----:|
| seed_v1              |      50 | (none)                     | CONSISTENT  | 5 ms |
| seed_v2              |      16 | (none)                     | CONSISTENT  | 2 ms |
| seed_long_paragraphs |     120 | (none)                     | CONSISTENT  | 11 ms |
| seed_news_briefs     |      66 | (none)                     | CONSISTENT  | 6 ms |

**The verdict "CONSISTENT" is honest but uninformative.** The
sieve extracts verb-lemma predicates (`visit`, `be`, `have`,
`use`, etc.) that don't match the operator-vetted logical
predicates in the library. Z3 is asked to check zero
property-bearing predicates and trivially returns SAT.

This is the same shape as the RPCA spike's negative result and
the multiplier-bootstrap spectral-CI caveat: math kernel works
on its native domain (Experiment 1 is a pure kernel verification
in milliseconds with minimal cores); simplest substrate
application reveals a real gap that needs operator attention,
not a math fix.

## What the gap actually is

Two paths can close it:

1. **Curated logical-property declarations per substrate predicate.**
   The starter library covers FOAF-like family/biography
   predicates that don't appear in the seed corpora. For the
   substrate's actual vocabulary (action verbs from prose), an
   operator-curated mapping like
   ```
   visit:    {ANTISYMMETRIC?  no — symmetric in the abstract}
   be:       {TRANSITIVE — possibly; needs care for "X is Y" semantics}
   has:      {ANTISYMMETRIC, IRREFLEXIVE on physical possession}
   ```
   is a one-day operator task that immediately pays back with
   real contradiction detection.
2. **Predicate-classification preprocessing.** Map raw verb
   lemmas to a smaller "logical predicate" vocabulary
   (e.g. `visit / went_to / travelled_to` → `was_at`, with
   FUNCTIONAL on `(person, time)` arguments). This is
   adjacent-work to Phase B's multi-modal dispatch design.

## Compounds with the kernels just shipped

The Z3 verdict is a binary signal — it's an excellent companion
to the calibrated kernels:

  - **PR #183 conformal**: wrap the per-corpus rate of
    "axioms in UNSAT core" with `SplitConformal(alpha=0.1)` →
    distribution-free CI on contradiction prevalence
  - **PR #186 SPRT**: stop iterating on a corpus as soon as
    the consistency-check rate stabilises with operator-chosen
    (α, β) error
  - **PR #184 vN entropy**: low-entropy + high-contradiction is
    a different failure signature than high-entropy + low-
    contradiction; jointly they discriminate corruption classes

## Honest tier table

| component                                                  | tier        |
|------------------------------------------------------------|-------------|
| Z3 / Nelson-Oppen / CDCL(T)                                | [provable]  |
| All four predicate-property schemas decided correctly      | [certified] |
| Minimal UNSAT-core extraction                              | [certified] |
| Substrate detection at sub-second on labeled corpora       | [empirical, latency only] |
| Operator-vetted predicate library matched to corpus vocab  | [open question — operator task] |
| Wired into the substrate's pre-sign attestation gate       | [not yet]   |
