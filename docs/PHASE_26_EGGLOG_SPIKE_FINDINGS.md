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
