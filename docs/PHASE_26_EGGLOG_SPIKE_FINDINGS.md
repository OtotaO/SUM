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

## Iteration 4 — content-derived cost model resolves the
determinism red flag

The iteration-3 finding (egglog default extract is
non-deterministic across processes on ties) is now resolved by a
custom cost model passed via egglog's `extract(cost_model=…)`
hook. PR egglog-python#357 (merged 2025-10-02, present in v11.4.0
which we pin) added the API.

`_content_hash_cost_model(egraph, expr, children_costs) → int`
returns
`sum(children_costs) << 64 + sha256(str(expr))[:8]`. Two priors,
two bit-ranges, no overlap:

  - **High 64 bits (children_costs):** preserves egglog's "smaller
    subtree wins" structural intuition. A leaf still beats a
    multi-node tree for the same e-class.
  - **Low 64 bits (content hash):** sha256 over the expression's
    deterministic string repr; provides a content-derived total
    order on e-class members with equal structural cost. The
    hash depends only on the expression's identity, not on
    insertion order.

This is the layered-decomposition discipline named in the
deep-research article's §2 kernel: distinct priors handled in
distinct bit ranges, the most expensive-to-recover-from prior
occupying the high bits.

`extract_canonical(triple, deterministic=True)` (the default)
uses this cost model. `deterministic=False` falls back to
egglog's default extract; kept for spike comparisons and to
keep the iteration-3 limitation measurable as an A/B contrast.

### Tests pin the resolution

- `test_extract_canonical_is_deterministic_across_insertion_order`
  asserts the SAME canonical form across forward and reversed
  insertion under `deterministic=True`. Inverts the iteration-3
  known-limitation test.
- `test_extract_canonical_deterministic_false_preserves_default_behaviour`
  asserts the iteration-3 non-determinism still appears under
  `deterministic=False`. If this test starts failing, egglog
  upstream has solved the determinism issue and our wrapper
  can simplify.
- `test_content_hash_cost_model_is_pure` pins the cost model's
  purity (same args → same cost) and the bit-range layering
  (different children_costs differ by ≥ 2⁶⁴; different
  expressions differ in the low 64 bits).

### What this iteration did NOT address

- **Materialisation bottleneck.** Still ~70 s at 10k. egglog
  upstream issue #756 (efficient bulk-load) remains open; our
  option 2 has no upstream landing. This iteration trades
  determinism for nothing in the wall-clock dimension.
- **Custom cost in eager mode.** Eager-mode tests are not yet
  re-run with `deterministic=True`; the cost model applies at
  extract time only, so eager-mode behaviour should be
  identical, but a small follow-up could pin that explicitly.
- **Library-scale e-class queries.** The 50 k+ axiom regime
  remains untested for any spike candidate. The next blocking
  question, if Phase 26 proceeds, is the bulk-load workaround
  for option 2.

### Decision options now updated again

  - **Continue:** option 2 (bulk-load API) — egglog issue #756
    is open with no upstream landing. We'd be pioneering a
    workaround. Highest leverage if Phase 26 is still
    near-term-priority.
  - **Pause egglog, start Neo4j candidate:** the determinism
    objection is gone; the materialisation objection is the only
    remaining technical reason to switch. If the operator wants
    library-scale today, Neo4j's mature scale story may now
    out-weigh egglog's e-class-native story.
  - **Pause Phase 26 entirely, resume Phase A/B/C:** unchanged.
    The §4.9 envelope at HEAD remains fine for current
    library-scale targets.

External validation from the deep-research search (egglog
ecosystem):
  - egglog upstream issue #793 (math-microbenchmark
    nondeterminism) is OPEN as of April 2026 — our finding
    matches reality.
  - No public production users at >10k facts; we are the load
    test. UWPLSE's June 2025 paper "Parameterized Complexity of
    Running an E-Graph" makes clear that scale is pathological
    in adversarial cases.
  - egg (Rust) `CostFunction` trait is fully deterministic and
    has always supported content-derived costs cleanly. Bindings
    via `snake-egg` exist but are less maintained than
    egglog-python.

Two iterations of honest red flags + one iteration of resolution.
The numbers are honest evidence; the decision still belongs to
the operator.

## Iteration 4.1 — performance correction (the cost model is NOT free)

A pre-flight verification round before posting upstream egglog
issues caught an overclaim in the iteration-4 section above:
"the cost model only adds a sha256 per extract but that's
microseconds compared to the egglog overhead." That was wrong
in the dimension I had not measured.

The cost model is invoked **once per visited e-node per
extract call** (egglog's tree-walk into Python on every
visit). At our spike's workload sizes (lazy mode, on the same
machine, fresh receipt
`fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T161351Z.json`):

| workload             | default extract | deterministic extract | overhead ratio |
|----------------------|----------------:|----------------------:|---------------:|
| seed_news_briefs     |          471 µs |                89 ms  |          188 × |
| seed_long_paragraphs |          567 µs |               171 ms  |          302 × |
| synthetic_1k         |        1 266 µs |               1.1 s   |          860 × |
| synthetic_10k        |       17 511 µs |              11.5 s   |          657 × |

The overhead is real, large, and scales with graph size. A
single deterministic extract at 10k takes 11.5 seconds; at the
50k library-scale target this would be ~minutes per call.

### Why the overhead is so large

The Python cost model is a callback into Python from egglog's
Rust extractor for every e-node it touches during the walk.
Each call pays:
  - The Rust → Python boundary cost (~10 µs per call observed)
  - `str(expr)` construction (formats subject/predicate/object
    into a string)
  - `sha256(...)` of that string

`str(expr)` does contain the full content for our flat
`Triple(s, p, o)` shape (verified — the cost model sees
`'T("alice", "owns", "rex")'`, not an opaque ref). For deeply
nested expressions the cost model would only see the top-level
constructor and rely on the `children_costs` recursion for
content sensitivity; not relevant for our case but worth
knowing if Phase 26 grows nested expression types.

Caching the per-`str(expr)` sha256 results does not help —
each cost-model call sees a different sub-expression visited
during the walk, so cache hit rate is ~0 within a single
extract.

### What this changes about the iteration-4 recommendation

The iteration-4 section above frames `deterministic=True` as
the right default. With the perf data in hand, the more honest
framing is:

  - **`deterministic=True` is the workaround for cross-process
    determinism — but it has a 200-1000× per-extract overhead
    that scales with graph size.** At 10k axioms, single
    extracts are 11.5 s.
  - For workloads that need deterministic extract on small
    graphs (≤ 100 axioms, ≤ ~20 ms overhead), the workaround is
    viable today.
  - For library-scale workloads (1k+ axioms with frequent
    e-class queries), the workaround is not viable as
    implemented. The path forward is either:
    (a) a Rust-native deterministic-extract mode in upstream
        egglog (would eliminate the Rust↔Python callback cost),
    (b) batching extract calls so the per-call overhead amortises
        across many queries (not currently exposed),
    (c) avoiding e-class queries on the hot path entirely
        (back to "egglog as query layer for special-case rewrites,
        not as the substrate's primary extraction surface").

### Tests pin the new finding

`Tests/test_graph_store_egglog.py::test_deterministic_extract_has_measurable_overhead`
asserts that deterministic extract is at least 50× slower than
default on a small graph — small enough to pass quickly in CI
but large enough to catch a hypothetical future regression
where the cost model becomes free (which would be good news
worth detecting).

### What this means for the upstream egglog community

This iteration's findings — both the workaround (cross-process
determinism via content-hash cost model) and the cost (200-1000×
per-extract overhead) — were originally going to be reported as
upstream comments. **They aren't, because of iteration 5 below.**

## Iteration 5 — spike conclusion: union-find beats egglog for
SUM's actual need

After preparing draft upstream issue comments, the strategic
question came up: *can we just adapt what we need from egglog
and move on?* Answer: yes, and the resulting hand-rolled adapter
beats egglog by orders of magnitude on every dimension that
matters for the substrate.

**The substrate's actual need**, distilled across 4 spike
iterations:
  - "Given equivalent triples under a symmetry rule, return a
    deterministic canonical form."

That's a graph algorithm — union-find — not a special-purpose
e-graph requirement. The egglog spike taught us this; the
correct conclusion is to act on it.

`sum_engine_internal/graph_store/unionfind_store.py` implements
it directly:

  - O(α(N)) union via path-compressed union-find
  - O(class_size) extract by linear scan; 2 ms at 10k triples
  - **Deterministic by construction**: extract returns the
    lex-smallest member of the equivalence class. No cost model,
    no callback, no upstream issue to wait on.
  - Zero external dependency, zero version pin

### Three-way comparison (lazy egglog vs eager egglog vs union-find)

Same workloads, same machine, same harness. Receipt:
`fixtures/bench_receipts/phase_26_backing_store_spike_egglog_20260509T164443Z.json`.

| workload      | mode      | insert_s | materialise_s | extract µs (default) | extract µs (det) | overhead × |
|---------------|-----------|---------:|--------------:|---------------------:|-----------------:|-----------:|
| 66 triples    | unionfind |    0     |        0      |                  13  |               13 |        1   |
| 66 triples    | egglog-lz |    0     |       0.010   |                 470  |           88 220 |      188   |
| 120 triples   | unionfind |    0     |        0      |                  27  |               27 |        1   |
| 120 triples   | egglog-lz |    0     |       0.198   |                 606  |          165 686 |      273   |
| 1k triples    | unionfind |    0.0003|        0      |                 192  |              192 |        1   |
| 1k triples    | egglog-lz |    0.0002|       0.379   |               1 309  |        1 084 893 |      829   |
| 10k triples   | unionfind |    0.003 |        0      |               2 016  |            2 016 |        1   |
| 10k triples   | egglog-lz |    0.002 |      66.874   |              16 815  |       11 064 414 |      658   |

  - **Insert is competitive** (both essentially-free at lazy
    egglog; UnionFindStore takes the same set-add cost without
    the e-graph in the loop)
  - **Materialisation gone** — UnionFindStore has no separate
    materialisation step. The e-graph register cost (66 s at
    10k) is eliminated; we never paid it
  - **Extract: ~5 500× faster than deterministic egglog** at 10k.
    Default-mode egglog (FIFO, non-deterministic) is the closer
    comparison: UnionFindStore is ~8× faster on extract at 10k
    AND deterministic by construction
  - **Cross-process determinism**: same canonical form across
    forward/reversed insertion orders. Pinned by
    `test_extract_canonical_is_deterministic_across_insertion_order`
    in `Tests/test_graph_store_unionfind.py`
  - **Substrate-parity**: identical `content_hash` to the egglog
    backend on the same input (verified by
    `test_content_hash_matches_egglog_backend`); identical
    pattern-query answers (verified by
    `test_pattern_queries_match_egglog_backend`)

### Decision and recommendation

**`UnionFindStore` is the Phase 26 backing store.**

This isn't a knock on egglog. Egglog is the right tool for a
*different* problem — workloads with many rewrite rules requiring
real saturation under a unified theory. SUM doesn't have that
problem today. We have one symmetry rule and the substrate
needs a deterministic canonical extract. Union-find delivers
that with three orders of magnitude less infrastructure.

**Future direction.** If Phase C grows multiple symmetric
rewrite rules, register more equivalences into the union-find —
the data structure handles arbitrary equivalences natively. If
Phase C ever needs arbitrary saturation under conditional rules
(e.g., "Triple(s, P, o) where P matches some pattern unifies
with..."), egglog stays available as a one-shot tool: run
saturation once, dump the union-find structure, drop egglog.
Pay it only when it pays back.

**EgglogStore stays in the codebase** as a comparison candidate
for as long as Phase 26 spike data is load-bearing. Tests still
pass, the spike harness still measures it. If we ever need to
re-evaluate, the data lineage is intact.

### What this means for the upstream egglog community

The egglog upstream issues (#793 extract nondeterminism, #756
bulk-load) **stay open without SUM contributions**. We don't
need a Rust-native deterministic-extract mode anymore — our
substrate has one, by construction. Comments would be of the
form "we explored your tool, found it didn't fit our shape,
built our own; here's our finding for whoever else is
evaluating," which is downstream noise rather than upstream
signal. Filing them would burn maintainer attention without
pushing their roadmap forward.

If the maintainers ever solve cross-process determinism
natively, this spike's findings doc can be revisited — the
EgglogStore is still wired in.
