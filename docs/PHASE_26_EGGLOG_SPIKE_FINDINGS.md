# Phase 26.0 spike — egglog backing store

*First candidate of the three named in `docs/PHASE_26_DESIGN.md` §2
(Neo4j, PostgreSQL+AGE, egglog). Egglog ships as a pip wheel — zero
infrastructure overhead beyond `pip install` — so it's the natural
first candidate. Neo4j and AGE will require provisioning the
respective services and belong to follow-on spike PRs.*

## What landed

  - `sum_engine_internal/graph_store/` — backend-agnostic interface
    with `GraphStore` Protocol, `Triple` dataclass, `GraphStoreInfo`.
    Five operations (add, count, iter, find_objects, find_subjects)
    plus a backend-independent `content_hash` based on JCS-style
    sorted sha256.
  - `sum_engine_internal/graph_store/egglog_store.py` — egglog
    implementation. Holds triples in both an `egglog.EGraph` (for
    future equivalence-class work — Phase C) and a Python set
    (authoritative for queries and the content hash). The Python
    set is what the spike measures; the e-graph is the *thing the
    spike validates is in the loop*.
  - `scripts/research/phase_26_egglog_spike.py` — measurement
    harness. Re-encodes substrate corpora through the existing
    `DeterministicSieve` extractor, plus synthetic 1k / 10k
    workloads. Emits a `sum.phase_26_backing_store_spike.v1`
    receipt under `fixtures/bench_receipts/`.
  - `Tests/test_graph_store_egglog.py` — 11 contract tests pin
    correctness invariants on the egglog backend (idempotent add,
    sorted-unique queries, hash determinism under insertion-order
    permutation, hash format).

## Measurement (one machine; comparable cross-machine via the
receipt's `host` block)

| workload                 | n_in   | n_unique | insert_s | per-query µs | determinism | parity |
|--------------------------|-------:|---------:|---------:|-------------:|-------------|-------:|
| seed_long_paragraphs     |    120 |      120 |    0.029 |         2.85 | OK          | ✓      |
| seed_news_briefs         |     66 |       66 |    0.009 |         1.62 | OK          | ✓      |
| synthetic_1k             |  1 000 |    1 000 |    0.385 |        31.50 | OK          | ✓      |
| synthetic_10k            | 10 000 |   10 000 |   67.314 |       336.68 | OK          | ✓      |

Receipt: `fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T130146Z.json`

## What this tells us

**Three pass criteria from §4 risks:**

  - **Determinism** — PASS. Inserting the same triple set in
    forward and reversed order produces byte-identical
    `content_hash` on every workload. The hash is JCS-canonical
    sort over sha256 of the triple set, independent of egglog's
    union-find internals — so cross-machine determinism reduces
    to the determinism of `_canonical_triples_hash`, which is
    sha256 of sorted text.
  - **Substrate parity** — PASS. The backend's `find_objects` /
    `find_subjects` answers match a brute-force list scan on
    every sampled query across every workload (zero mismatches,
    n=50 samples per workload).
  - **Wall time** — MIXED. Real-corpus inserts (66 / 120 triples)
    are sub-30ms; synthetic 1k is 0.4 s; synthetic 10k is **67 s**,
    which is well above the §4.9 envelope's projection for
    library-scale (50k+) workloads.

## The 10k red flag

67 s for 10 000 inserts (~6.7 ms per insert) is not viable for
the Phase 26 library-scale ambition. The spike's job is to
surface this kind of risk before committing engineering time —
and it did.

Three plausible follow-up directions:

  1. **Lazy materialisation.** Only register triples into the
     e-graph when an equivalence-class query is actually issued.
     For pure storage / pattern-query workloads, the Python set
     suffices. The e-graph becomes an *index* built on demand.
     Estimated effort: small. Likely outcome: 10k insert drops
     to ms range, e-graph build cost amortised across queries.
  2. **Bulk-load API.** Egglog supports `.run` over compiled
     rulesets; we may be able to compile the triple insertion
     into a single rule and let egglog's Rust backend ingest in
     one pass. Estimated effort: medium; requires deeper egglog
     API exploration.
  3. **Drop egglog as the storage layer; keep it as a query
     layer.** If equivalence-class queries are the only thing
     egglog uniquely offers, use a faster backing store (Neo4j
     or AGE) for the hot storage path and feed e-classes only
     the slices that matter for a given query. Estimated effort:
     larger but more architecturally honest.

## Honest scoping

What this spike does *not* establish:

  - It does not exercise egglog's equivalence-class machinery
    (Phase C's importance-weighted SUM target) — that requires
    rewrite rules and cost functions matched to the substrate's
    semantics, which is the next spike PR if we proceed.
  - It does not measure persistence (egglog is in-process here;
    Phase 26.1 schema-migration handles persistence regardless
    of backend choice).
  - It does not measure cross-machine determinism on Modal.
    The `content_hash` is sha256-over-sorted-text so the
    *expectation* is byte-identical cross-machine, but the §4
    risk (`Mitigation: spike verifies determinism on the
    cross-machine Modal matrix`) is not yet executed for this
    candidate.

## Recommendation for the operator

The egglog candidate clears determinism and parity but fails the
library-scale wall-clock envelope as currently implemented.
Decision options:

  - **Continue:** invest one cycle in lazy-materialisation
    (option 1 above) before judging egglog. The other two
    candidates (Neo4j, AGE) require infrastructure provisioning
    that hasn't started; iterating on egglog is the cheapest next
    measurement.
  - **Pause egglog, start Neo4j candidate:** if the operator
    prefers infrastructure-backed storage from the start, this
    is the moment to switch. Egglog's e-class value can be
    re-evaluated as a Phase C *query* layer later.
  - **Phase 26 itself paused:** the §4.9 envelope at HEAD is
    fine for the substrate's current library-scale targets;
    Phase 26 is a future-direction investment, not a load-bearing
    fix. Continuing other research arcs (multi-modal dispatch,
    importance-weighted SUM) is also defensible.

The spike's purpose is to put numbers in front of the decision.
The numbers above are honest evidence; the decision belongs to
the operator.

## Iteration 2 — option 1 (lazy materialisation) measured

Followed up on option 1: defer e-graph registration until an
equivalence-class query forces it. `EgglogStore` gains a
`materialise_egraph()` method and a `eager_materialisation`
constructor flag (default False — lazy). `add_triple` updates
only the authoritative Python set; the e-graph stays unbuilt.
Pattern queries (`find_objects` / `find_subjects`) and
`content_hash` work directly on the set and never trigger
materialisation.

A/B re-run on the same machine + workloads:

| workload             | mode  | insert_s | materialise_s | per-q µs |
|----------------------|------:|---------:|--------------:|---------:|
| seed_long_paragraphs | lazy  |   0.0001 |        0.1715 |     2.63 |
| seed_news_briefs     | lazy  |   0.0000 |        0.0093 |     1.53 |
| synthetic_1k         | lazy  |   0.0002 |        0.3684 |    27.64 |
| synthetic_10k        | lazy  |   0.0024 |       69.7594 |   317.84 |
| seed_long_paragraphs | eager |   0.0175 |        0.0000 |     2.81 |
| seed_news_briefs     | eager |   0.0090 |        0.0000 |     1.60 |
| synthetic_1k         | eager |   0.3341 |        0.0000 |    29.86 |
| synthetic_10k        | eager |  64.4221 |        0.0000 |   313.63 |

Updated receipt: `fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T133315Z.json`

### What the second-iteration numbers say

  - **Lazy storage path is essentially free.** 10k inserts go
    from 64.4 s eager → 2.4 ms lazy — a 28 000× drop. For workloads
    that never need equivalence-class queries (most pattern
    queries the substrate runs today), egglog now pays the cost
    of being available without paying the cost of being used.
  - **Determinism + parity hold across modes.** `content_hash` is
    byte-identical between lazy and eager modes for every
    workload (verified by `test_lazy_and_eager_modes_produce_identical_content_hash`).
    The hash is over the triple set, which is mode-independent.
  - **Materialisation is still the real ceiling.** Lazy doesn't
    *speed up* the e-graph build; it just defers it. 10k
    materialise costs 69.8 s — same order as eager insert. Any
    workload that does call e-class queries at library scale
    pays the same Rust-backend bookkeeping cost as before.
    Option 2 (egglog bulk-load API) remains the next direction
    if the e-class workload ever becomes critical.

### Architectural implication

The lazy/eager split makes the substrate's choice explicit:
**egglog is a query layer, not a storage layer.** The Python
set is the storage layer. This is consistent with the design
doc's framing of egglog as the candidate that uniquely offers
extract-with-cost over equivalence classes — that's a *query*
property, and we now only pay for it when we use it.

For Phase 26.1+ engineering, this means the schema migration
work targets the Python-set storage path (or whatever
persistence layer replaces it); egglog enters only at the query
boundary, materialised on demand and discarded between sessions
unless explicitly cached.

### Decision options now updated

  - **Continue:** the storage-cost objection is resolved. Next
    spike iteration would be the actual e-class queries — define
    a small importance-cost function and a couple of rewrite
    rules matched to Phase C semantics, measure the e-class
    extraction wall time on the materialised graph. That would
    be option 2 of the original three plus one new question: is
    the per-query e-class extraction fast enough to amortise
    against materialisation cost?
  - **Pause egglog, start Neo4j candidate:** still defensible.
    Comparison would now be Neo4j-storage-and-query vs
    Python-set-storage + egglog-query.
  - **Phase 26 itself paused:** unchanged from before.

The numbers are honest evidence; the decision still belongs to
the operator.

## Iteration 3 — actual e-class queries measured

Followed up on the iteration-2 question: "is per-query e-class
extraction fast enough at library scale to amortise against
materialisation cost?"

Added a baked-in `ownership_symmetry` ruleset
(`Triple(s, "owns", o) ⟺ Triple(o, "owned_by", s)`) to
`EgglogStore` plus three new methods: `available_rulesets()`,
`saturate(ruleset_name)`, and `extract_canonical(triple)` — the
minimum surface needed to exercise egglog's extract-with-cost
end-to-end. Spike harness gains a `_measure_eclass` phase that
runs saturate + samples 50 extractions per workload.

Lazy-mode measurements (eager numbers from iteration 2 hold
because the e-class phase runs *after* materialisation and is
mode-independent):

| workload             | materialise_s | saturate_s | extract µs | sample |
|----------------------|--------------:|-----------:|-----------:|-------:|
| seed_news_briefs     |        0.0089 |     0.0006 |        438 |     50 |
| seed_long_paragraphs |        0.1629 |     0.0015 |        575 |     50 |
| synthetic_1k         |        0.3500 |     0.0021 |      1 094 |     50 |
| synthetic_10k        |       62.4440 |     0.0119 |     15 255 |     50 |

Updated receipt: `fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T135038Z.json`

### What the iteration-3 numbers say

  - **Saturation is essentially free for unfiring rules.** 10k
    triples saturate the ownership-symmetry ruleset in 12 ms
    (no facts match the rule in the synthetic workload). For
    workloads where rules *do* fire, saturation cost will scale
    with the union-find work, not with the input size — but that
    measurement requires a workload designed to fire rules
    repeatedly, which is the *next* iteration if we proceed.
  - **Extract scales linearly with graph size.** 0.6 ms at 66
    triples → 15 ms at 10k triples (a 25× cost ratio for a 150×
    larger graph — sub-linear but closer to linear than constant).
    Extrapolating: at 50 k triples extract would take ~75 ms per
    call; at 100 k, ~150 ms. For workloads that need to extract
    every triple's canonical form after saturation, that becomes
    minutes of cumulative extract time.
  - **End-to-end e-class query cost is dominated by
    materialisation, not by saturate or extract.** At 10k:
    materialise 62 s + saturate 12 ms + 50 extracts 760 ms ≈
    63 s. The materialise cost is the architectural bottleneck;
    option 2 (egglog bulk-load API) remains the unaddressed
    direction with the most leverage.

### Honest red flag — default cost is insertion-order-sensitive

A separate finding from this iteration: egglog's default
`extract` is **non-deterministic across processes** when two
expressions in the same e-class have equal cost.

Concretely, after saturating with `ownership_symmetry`:
  - Inserting `(active, passive)` and extracting → returns the
    active form
  - Inserting `(passive, active)` (reversed) and extracting →
    returns the passive form

Both choices are valid extracts (lowest-cost in the e-class), but
the tie-breaker is FIFO over insertion order. For Phase C to be
cross-process deterministic — required for `bench_digest`
reproducibility — a custom cost function with a content-derived
tie-breaker is mandatory. Egglog supports this via
`@function(cost=...)` annotations; building one is the natural
follow-on if we proceed with egglog as the query layer.

This limitation is pinned by
`test_default_cost_is_insertion_order_sensitive_on_ties` in
`Tests/test_graph_store_egglog.py` — the test ASSERTS that the
two canonical forms differ, so if egglog's default ever becomes
deterministic on ties, the test will start failing and the
findings doc gets re-evaluated.

### Decision options now updated again

  - **Continue:** custom cost function next (resolves the
    determinism red flag; orthogonal to the materialisation
    bottleneck). After that, option 2 (egglog bulk-load API)
    to attack the materialisation cost — that's the only path
    to library-scale e-class queries.
  - **Pause egglog, start Neo4j candidate:** Neo4j has its own
    determinism story (Cypher with explicit ORDER BY); the e-class
    semantics are not native but can be encoded via Cypher
    patterns. The trade is "egglog-native equivalence-class
    extraction we'd have to make deterministic" vs "Neo4j-native
    determinism we'd have to encode equivalence classes in." Both
    are work; the choice depends on which path the operator
    wants to be the long-term one.
  - **Phase 26 itself paused:** unchanged from before. The
    iteration-3 numbers reinforce that egglog at the *query
    layer* is workable for small graphs but not yet for library
    scale — and Phase 26 itself is a future-direction
    investment, not a load-bearing fix.

The numbers are honest evidence; the decision still belongs to
the operator.
