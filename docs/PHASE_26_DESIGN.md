# Phase 26 — Property-graph backing store: design before engineering

This document is the design pass for the substrate change PROOF_BOUNDARY §3
named as Phase 26: *property-graph backing store for corpora above ~10k
axioms (prime encoding demoted to signed witness)*.

The §4.9 measured envelope says we need this: at N=10000 axioms the
merge bottleneck is 23.9 s p50 (k=1.497 sub-quadratic empirical), and
recursive-compression iteration at library scale (≥50k axioms)
extrapolates to ≥4 minutes per step. The substrate is comfortable to
medium-book scale today and gates on Phase 26 for library scale.

This doc surfaces the design decisions that need to be made *before*
engineering, so the engineering work is informed. It does **not** pick a
backing store. That decision belongs to a spike-evaluation PR after this
design lands.

## 0. Problem statement

The current substrate uses a Gödel state integer as both:

  (a) **The canonical anchor** — `parse(canonical_tome(S)) == S` proves
      that S's axioms can be reconstructed from its canonical tome at
      the integer level. This is PROOF_BOUNDARY §1.1's lossless round-
      trip and is mechanically proven.

  (b) **The primary query substrate** — entailment is `state % prime ==
      0`; merge is `lcm(A, B)`; sync delta is `gcd(A, B)`. All operations
      run on big-integer arithmetic over primes minted per axiom.

Role (a) is **load-bearing for attestation**: external verifiers re-derive
the state-integer to prove the bundle's integrity. Role (b) is **load-
bearing for query throughput**: it's how `sum verify`, the sheaf bench,
and the recursive walk make their primary measurements.

The §4.9 measurement shows role (b) is the scaling bottleneck. Big-int
LCM at 3 million bits (≈10000 axioms × 64-bit primes) takes 24 seconds
per merge call. Library scale (50k+ axioms) is empirically infeasible
without architectural change.

Phase 26's proposal: **demote the prime encoding to attestation
witness; promote a property-graph store to primary query substrate.**
The state-integer remains derivable on demand for attestation; queries
that previously walked through prime arithmetic now walk through graph
indices.

## 1. Decision space

Five decisions must be made before engineering can start. Each is named
explicitly here so the design has a discrete, reviewable surface.

### 1.1 Data model

What graph schema does the property-graph store use?

**Option A — Triple-as-edge.** Nodes are entities (subjects, objects);
edges are predicates with the (s,p,o) triple represented as a directed
edge from s to o, labeled by p. Edge attributes carry: provenance
record id, signing key id, axiom prime (the witness), timestamp.

  - Pro: matches RDF / property-graph industry standard; tooling
    abundant (SPARQL, Cypher, Gremlin)
  - Pro: cheap to query "all triples with subject Alice" or "all triples
    with predicate `founded`"
  - Con: edges-with-attributes work less well in some graph DBs (Neo4j
    handles it via relationship properties; egglog needs e-class
    encoding; PostgreSQL needs join tables)
  - Con: hyper-edges (n-ary relations) need workarounds

**Option B — Triple-as-node, predicate-as-edge.** Nodes are *triples*;
edges between triples represent semantic relations (e.g. *uses-same-
predicate*, *shares-subject*, *temporally-after*). Standard graph queries
walk through these "meta" edges.

  - Pro: triples are first-class; per-triple attributes (provenance,
    prime, signing) live naturally on the node
  - Pro: hyper-edges (n-ary relations) trivially supported
  - Con: doubles the node count (triples *and* entities both need
    storage if we want subject-level queries to be cheap)
  - Con: less natural for SPARQL-style standard queries

**Option C — Hybrid.** Triples-as-nodes (Option B) for storage; project
to triple-as-edge (Option A) at query time via materialised views.

  - Pro: best query ergonomics
  - Con: more code; consistency between the two views needs maintenance
  - Con: cache invalidation problem

**Recommendation:** *Option A* for the spike (industry-standard, fast
to implement, matches existing prime-derived axiom semantics).
Re-evaluate after the spike if hyper-edges become a real need.

### 1.2 Migration semantics

How do existing receipts (which encode the prime→axiom map) survive the
transition?

The current `CanonicalBundle` schema has fields: `axioms`,
`canonical_tome`, `state_integer`, `prime_scheme`, `signatures`,
`branch`, etc. External verifiers re-derive the state_integer by
re-minting primes from the axioms and computing the LCM. This is the
core trust loop.

Phase 26's change: the `state_integer` becomes a *signed witness*, not
the primary query substrate. The bundle gains a property-graph layer
(call it `graph_attestation`) that records:

  - Per-triple node identifier (a content-addressed hash, not a prime)
  - The graph-store-root hash (Merkle root over the per-triple nodes)
  - The mapping (triple → prime) so external verifiers can still
    re-derive the state_integer if they want to

Two migration approaches:

**Approach A — Versioned schema.** New `bundle_version: 2.0.0`
introduces the graph layer. Verifiers MUST be 2.0.0+ to verify. Old
bundles continue to verify under 1.x semantics; new bundles use 2.x
exclusively.

  - Pro: clean break; no per-bundle conditional logic in the verifier
  - Con: existing bundles in the wild stay frozen at 1.x; tooling has
    to support both indefinitely

**Approach B — Backward-compatible extension.** `bundle_version: 1.2.0`
adds an optional `graph_attestation` field. Old bundles without it
still verify under unchanged semantics; new bundles include it. Verifier
checks both layers when present.

  - Pro: smooth migration; old verifiers ignore the new field
  - Con: bundle structure carries both representations forever
  - Con: tempting to let the dual-representation linger past its useful
    life

**Recommendation:** *Approach A*, with a clean cut at v2.0.0. The cost
of dual-maintenance forever (Approach B) is higher than the cost of
versioning. Versioning is what `bundle_version` is for.

### 1.3 API surface impact

What changes for the developer-visible surface?

**`sum attest`** — UNCHANGED at the user surface. Internally, after
prime-minting and state-integer derivation, also build the graph
attestation layer. The user doesn't see the graph; they see a bundle
with a new `graph_attestation` field they can ignore.

**`sum verify`** — Verifier checks both layers. State-integer derivation
still happens (witness check); graph-root verification is added (Merkle
proof against the graph attestation). Both must pass for `sum verify`
to succeed at v2.0.0+.

**`sum render`** — UNCHANGED. The render path doesn't query the graph;
it works from the in-bundle axioms directly.

**`sum inspect`** — Adds graph-shape statistics: node count, edge count,
predicate-set size, subject-set size.

**`sum ledger`** — UNCHANGED. The Akashic ledger already operates on
provenance records; the graph layer is per-bundle, not per-ledger.

**Research benches** — Most operate on triple sets directly, not on the
state integer. Should be largely unchanged. The §4.9 performance
characterisation is the exception: it MUST be re-run post-Phase 26 to
quantify the new scaling envelope.

**Sheaf detector** — The sheaf operates on the triple set, not the
state integer. UNCHANGED at the math level. The benchmark numbers may
change (the underlying merge cost is no longer a hot path) but the
detector surface is the same.

### 1.4 Cross-machine reproducibility under graph queries

This is the load-bearing one. PROOF_BOUNDARY §2.10 says `bench_digest`
is byte-stable across three LAPACK environments. Adding a property-
graph store could break this if:

  - Graph traversal order is non-deterministic (e.g. dict iteration
    order in Python; row-order under PostgreSQL without ORDER BY;
    e-graph node-id assignment varying across versions)
  - The graph store's serialisation format depends on insertion order
    (some property-graph DBs do this)
  - Hash collisions in content-addressing produce different indices on
    different platforms

**Required design constraint:** every graph query whose output feeds a
`bench_digest` MUST produce a deterministic ordering. This means:

  - Content-address node IDs by `sha256(canonical_jcs(triple))`. JCS
    canonicalisation gives byte-deterministic input across platforms.
  - All graph queries that return sets MUST use ORDER BY (or equivalent)
    on a deterministic key (the content-addressed node ID).
  - Any caching layer must use deterministic eviction (LRU is fine; size-
    based eviction tied to clock-time is not).

**Test strategy for cross-machine determinism:**
  - Add Phase 26 benches to the existing Modal cross-machine matrix
    (Apple Accelerate / OpenBLAS Py 3.10 / OpenBLAS Py 3.12)
  - Pin specific graph-query operations as new digest-pinned tests
  - Existing 4-bench cross-machine matrix becomes 5-bench (4 sheaf +
    1 graph-substrate)

If any candidate backing store can't provide deterministic ordering at
the API level, it's disqualified for Phase 26.

### 1.5 Verifier path semantics

The cross-runtime trust triangle (Python ↔ Node ↔ browser) byte-
verifies Ed25519 over JCS. Phase 26's question: do all three runtimes
need a property-graph implementation, or only Python?

**Two-tier proposal:**

  - **Python runtime**: full graph store; primary attestation path.
  - **Node runtime + browser**: state-integer-only verification (the
    current path, unchanged). The graph attestation field is *optional*
    in the verifier's view; verifiers MAY choose to verify it (full
    trust) or skip it (state-integer-only trust).

This means a verifier in JavaScript can still verify a v2.0.0 bundle by
checking the state-integer chain only. It loses the graph-attestation
guarantee but gains nothing at attestation cost — the state-integer
chain is still load-bearing.

Trade-off: v2.0.0 receipts have *two* attestation paths. The graph
attestation is stronger (fully captures axiom relationships) but only
the Python verifier can check it. The state-integer attestation is
weaker (the prime-witness layer) but works in three runtimes.

**Recommendation:** Two-tier verifier with explicit semantics in the
receipt format. Document which guarantees come from which path. The
K-matrix (xruntime tests) verifies the state-integer chain across all
three runtimes; a new optional gate verifies the graph chain in Python
only.

## 2. Backing-store candidates

Three real candidates plus one default-rejected option, each with a
trade-off matrix entry. The recommendation at the end of this section
is *spike, not commit*.

### 2.1 Neo4j

**What it is:** Industry-standard property-graph database; Cypher query
language; ACID transactions; on-disk storage; Java core with bindings
in many languages including Python (`neo4j` driver).

**Pros:**
  - Mature, well-documented, large community
  - Cypher is the most widely-known property-graph query language
  - Native deterministic ordering via `ORDER BY` clauses
  - Per-node and per-relationship attributes well-supported
  - Used in production at scale; not research-grade

**Cons:**
  - Heavy: requires running a Java server (or paying for AuraDB managed
    service)
  - Operational complexity (backup, monitoring, schema migration)
  - License: AGPL for core; commercial license for closed-source use
  - Embedded mode exists but is Java-only; no native Python embedded
  - Wire protocol (Bolt) adds latency for every query

**Cross-machine determinism:** Yes, with explicit `ORDER BY`. Verified
in the Neo4j docs.

**Best fit for SUM:** Production deployment at the operator's discretion
(self-hosted Neo4j or AuraDB). Requires an operational story for
keeping a graph DB available.

### 2.2 PostgreSQL with graph extensions (Apache AGE or pg-graph)

**What it is:** Extend PostgreSQL with property-graph capabilities;
Apache AGE (Cypher-on-Postgres), or other extensions. PostgreSQL is
relational at heart; the graph layer is built atop SQL tables.

**Pros:**
  - PostgreSQL is the most-deployed open-source database; tooling
    abundant
  - Existing PostgreSQL installations can adopt without new
    infrastructure
  - License: PostgreSQL license; permissive
  - Operational model is mature: backup, replication, HA all standard
  - SQL underneath means relational fallback for queries that don't
    need graph semantics

**Cons:**
  - Graph performance is bottlenecked by SQL execution; not as fast as
    native property-graph engines for deep traversals
  - Apache AGE Cypher implementation is incomplete relative to Neo4j
  - Two query languages (SQL + Cypher) in the same DB is cognitively
    expensive

**Cross-machine determinism:** Yes, with explicit `ORDER BY`. PostgreSQL
guarantees this.

**Best fit for SUM:** Deployments that already run PostgreSQL and want
graph semantics without a new operational dependency.

### 2.3 Egglog (egraph + Datalog)

**What it is:** Equality-saturation engine combining e-graphs (data
structures for compactly representing equivalence classes) with
Datalog (declarative bottom-up logic programming). Production-released
at v2.0.0; CodSpeed-tracked benchmarks; Python bindings.

**Pros:**
  - **Built-in equivalence-class semantics** — the substrate's
    canonicalisation layer (Layer 1 feature 4) maps directly onto
    e-graph union-find. Predicate canonicalisation becomes declarative
    rewrite rules.
  - **Built-in `(extract … :cost …)`** — directly implements
    importance-weighted SUM extraction (Phase C). The "smallest
    n_axioms" placeholder in `_identify_sums` becomes minimum-cost
    extraction over the saturated e-graph.
  - Datalog query semantics are deterministic; saturation is order-
    independent.
  - Active development (last commit yesterday at writing); v2.0.0
    release.

**Cons:**
  - Research-tool maturity; not deployed in production at the operator's
    scale (no library-scale benchmark public)
  - Rust core; Python bindings exist but interop overhead is non-trivial
  - Persistence story unclear (e-graphs are typically in-memory; on-
    disk serialisation is a research direction)
  - Smaller community than Neo4j or PostgreSQL
  - Cross-machine determinism is plausible but unverified at our specific
    workload — would need spike measurement

**Cross-machine determinism:** Plausible (Datalog is deterministic by
spec; egglog's saturation is order-independent). MUST be verified for
our specific workload via the cross-machine matrix during the spike.

**Best fit for SUM:** Phase C (importance-weighted SUM extraction)
*and* potentially Phase 26 backing store if persistence works out. The
strongest match in raw semantic alignment with our existing canonical
predicate normalisation layer.

### 2.4 In-process Python graph (NetworkX, custom)

**What it is:** No external DB; run an in-memory graph in the Python
process via NetworkX or a custom implementation.

**Pros:**
  - No new operational dependency
  - Trivial Python interop (it IS Python)
  - Easy to test

**Cons:**
  - Doesn't solve the scaling problem we're addressing — at 50k axioms
    the in-memory graph would be tens-of-MB, fine, but at 1M+ axioms
    we're back to the same scaling cliff
  - No persistence (without rolling our own)
  - No deterministic-ordering guarantees out of the box (NetworkX uses
    Python dicts internally)

**Best fit for SUM:** Prototyping during the spike; not a production
candidate.

### 2.5 Trade-off matrix

| Criterion | Neo4j | PostgreSQL+AGE | egglog | NetworkX |
|---|---|---|---|---|
| Production-readiness | ✓ | ✓ | ⚠ research-grade | ✗ |
| Python interop | mid (Bolt over wire) | mid (libpq) | mid (PyO3 bindings) | ✓ native |
| Equivalence-class semantics | manual (rewrite via app code) | manual | ✓ native | manual |
| Query language | Cypher | SQL+Cypher | Datalog | Python iteration |
| Deterministic-output for `bench_digest` | ✓ with ORDER BY | ✓ with ORDER BY | ⚠ verify in spike | ⚠ depends on impl |
| Library-scale (>50k axioms) | ✓ proven | ✓ proven | ⚠ unverified | ✗ |
| License | AGPL/commercial | PostgreSQL (permissive) | MIT | BSD |
| New operational dep | yes (Java server) | sometimes (Postgres often present) | yes (Rust binary) | no |
| Spike effort | medium (driver + schema) | medium (extension + schema) | medium (Python bindings + DSL) | low (prototype only) |

### 2.6 Recommendation

**Spike all three production candidates** (Neo4j, PostgreSQL+AGE,
egglog) on the same workload: re-encode the v3.x bench data as a
property graph; run the same queries the existing prime-arithmetic
substrate handles; measure (a) wall-clock vs the §4.9 envelope, (b)
cross-machine determinism via the Modal matrix, (c) integration cost
in Python LOC.

NetworkX is the prototyping substrate during the spike — useful for
"does the schema work?" but not a candidate for the production answer.

The spike should produce a comparison receipt
(`sum.phase_26_backing_store_spike.v1`) and a recommendation. After
that, engineering proceeds on the chosen backing store.

**Estimated spike duration:** 2-4 weeks of design-and-prototype work
(approximately 1 PR per candidate plus a comparison PR). Lower than
the full Phase 26 engineering estimate; surfaces all the integration
risks before committing to a backing store.

## 3. Open design questions

These are decisions deferred to the spike or to operator input:

### 3.1 Per-triple provenance vs per-bundle provenance

The current Akashic ledger tracks provenance per `prov_id` (a content-
address over the bundle). Phase 26 could push this to *per-triple*:
each triple in the graph store gets its own provenance record.

  - Pro: fine-grained attribution; "where did this fact come from?"
    answerable per fact, not per bundle
  - Con: cardinality explosion; provenance records grow N× with the
    triple set

Decision: defer to spike. Most likely answer: *both* (bundle-level for
attestation; triple-level optional, tagged when the source supports it).

### 3.2 Graph-attestation Merkle scheme

The graph layer needs a content-addressed root. Options:

  - **Sorted-leaf binary Merkle tree** (RFC 6962-style). Standard;
    deterministic given sorted input.
  - **Merkle Patricia trie** (Ethereum-style). Better for sparse
    updates; more complex.
  - **Zero-knowledge-friendly Merkle** (Poseidon-hash-based). Future-
    proof for ZK proofs of graph subset membership; heavier hash.

Decision: defer to spike. Initial recommendation: sorted-leaf with
sha256 (matches the existing JCS+SHA256 substrate).

### 3.3 Graph-aware sliders

The 5-axis slider (density / length / formality / audience /
perspective) currently operates on the axiom-set as a flat list.
Post-Phase 26, a graph-aware slider could:

  - **Density** could mean "trim leaf entities" rather than "drop the
    last lex N% of axioms"
  - **Audience** could promote/demote based on entity type (technical
    vs general)
  - **Perspective** could re-root the graph at a different node

This is a v0.6+ research direction; not on the Phase 26 critical path.
Mention here so the design doesn't preclude it.

### 3.4 Deletion semantics

Currently, a bundle is immutable: the state_integer captures the axioms
present. Phase 26 introduces the question of *deleting* a triple from a
graph that's been signed.

Options:

  - **Append-only with tombstones** (the conservative answer)
  - **Mutable with re-signing** (the aggressive answer; bundle version
    increments on every change)

Decision: append-only with tombstones, matching the Akashic ledger's
existing append-only semantics. Mutability is a v0.7+ concern.

## 4. Risks and honest scope

  - **Operational risk** — every backing store candidate adds an
    operational dependency the project doesn't currently have. The
    `sum` CLI's "pip install + python -m" simplicity loses ground.
    Mitigation: keep v1.x bundles working (Approach A migration); the
    operator chooses when to deploy v2.x. Single-user / small-corpus
    workloads can stay on v1.x indefinitely.
  - **Cross-runtime degradation risk** — the Python ↔ Node ↔ browser
    triangle might not extend to the graph layer. Mitigation: two-tier
    verifier (§1.5) with explicit semantics. Verifiers in non-Python
    runtimes can verify the state-integer chain (unchanged guarantee)
    and skip the graph layer.
  - **Determinism risk** — backing-store query non-determinism could
    break `bench_digest` cross-machine reproducibility. Mitigation: the
    spike verifies determinism on the cross-machine Modal matrix
    *before* committing to a backing store.
  - **Performance risk** — the new substrate might not actually meet
    the §4.9 envelope's projected library-scale ceiling. Mitigation: the
    spike includes the §4.9-style measurement at N=10k, 50k, 100k for
    each candidate; if no candidate hits library-scale comfortably, we
    learn that before sinking weeks of engineering.
  - **Research-grade-tool risk** (egglog) — egglog might not be
    production-stable enough for the operator's deployment story.
    Mitigation: spike measures stability and the operator decides
    based on data, not vibes.

## 5. Spike plan (Phase 26.0)

This design doc lands first. Then a separate spike PR (or three —
one per backing-store candidate) implements:

  1. A small graph-store interface module
     (`sum_engine_internal/graph_store/`) with a backend trait.
  2. Implementations: Neo4j (via `neo4j` driver), PostgreSQL+AGE
     (via `psycopg`), egglog (via Python bindings).
  3. The same micro-benchmark workload run against each: re-encode
     `seed_long_paragraphs` (16 docs, ~120 triples), `seed_news_briefs`
     (16 docs, ~66 triples), and a synthetic 1k / 10k / 50k axiom
     dataset. Measure: insert time, query time for the v3.x bench
     access patterns, deterministic-output stability.
  4. A receipt aggregator that loads the per-candidate receipts and
     produces a recommendation.

After the spike, the engineering arc (Phase 26.1+) implements the
chosen backing store on the `sum_engine_internal/graph_store/`
interface. The interface lets us swap candidates if a better one
emerges later.

## 6. Decision points (in order)

  1. **Now (this PR):** Land this design doc. No engineering changes.
  2. **Phase 26.0 spike:** ~2-4 weeks. Three candidates measured; one
     chosen. Receipt
     `fixtures/bench_receipts/phase_26_backing_store_spike_<date>.json`.
  3. **Phase 26.1 — schema migration:** Bundle v2.0.0 design + tests +
     migration utilities for existing receipts.
  4. **Phase 26.2 — graph-store integration:** Replace prime-arithmetic
     hot-path queries with graph-store queries. Re-run §4.9 to measure
     the new envelope.
  5. **Phase 26.3 — verifier extension:** Two-tier verifier (graph
     layer optional in Node/browser; required in Python).
  6. **Phase 26.4 — library-scale validation:** Re-run §4.10/§4.10.1
     at 50k+ axiom corpora. Receipt
     `fixtures/bench_receipts/library_scale_recursive_walk_<date>.json`.
     This is the original-vision *hand-it-a-library* claim, measured.

Each phase is a discrete PR series; engineering does not commit to
later phases until earlier phases land cleanly. Decision gates between
phases are explicit in the receipt — if the spike (26.0) doesn't find
a candidate that meets the determinism and library-scale criteria, we
*don't* proceed to 26.1 and the design returns to the drawing board.

## 7. What this design doc is not

  - Not a commitment to engineering Phase 26 *now*. The decision to
    proceed beyond this design is operator-gated.
  - Not a backing-store choice. That decision belongs to the spike.
  - Not a schema specification. The spike will produce one based on
    the chosen backing store.
  - Not a substitute for measurement. Every claim above (especially in
    §2 trade-off matrix) is reviewable; numbers replace claims as the
    spike measures them.

The artifact this PR produces: a design space mapped, decisions surfaced,
risks named, candidates compared on paper. The minimum prerequisite for
informed engineering.
