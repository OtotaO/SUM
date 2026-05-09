"""Egglog-backed graph store for the Phase 26.0 spike.

Egglog (https://github.com/egraphs-good/egglog) combines an
e-graph (union-find over expressions) with a Datalog-style query
engine. The substrate doesn't need e-classes for storage — it
needs them for *equivalence-class queries* (Phase C's importance-
weighted SUM target). So this backend defers e-graph registration
until an e-class query is actually issued.

**Lazy materialisation (the option-1 fix from the original spike's
findings doc):** `add_triple` updates only the authoritative Python
set; the e-graph stays unbuilt. `materialise_egraph()` registers
all unregistered triples into egglog in a single batched call.
The first e-class query auto-materialises; pure pattern queries
(`find_objects`, `find_subjects`) and the content hash never
trigger materialisation because they don't need the e-graph at
all. This collapses insert latency to set-insertion speed for
the storage-heavy hot path while keeping egglog available for
the queries that actually need it.

What we are NOT doing in this PR (kept scope-honest):

  - **No real equivalence-class queries yet.** Materialisation is
    exposed; rewrite rules and cost functions matching Phase C
    semantics are the *next* PR if these measurements still pass.
  - **No persistence.** Each `EgglogStore()` is process-local.

Determinism note: egglog itself is order-independent for
saturation, but our spike does not run saturation — we hold the
graph as plain facts. So determinism here reduces to the
determinism of `_canonical_triples_hash`, which is JCS-style sort
over sha256, independent of egglog's internals.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterator

from sum_engine_internal.graph_store.base import (
    GraphStore,
    GraphStoreInfo,
    Triple,
    _canonical_triples_hash,
)


def _content_hash_cost_model(egraph: Any, expr: Any, children_costs: list[int]) -> int:
    """Custom egglog cost model that breaks e-class ties via the
    sha256 of the expression's string repr.

    Iteration 3 of the spike pinned that egglog's default `extract`
    is non-deterministic across processes when two expressions in
    the same e-class have equal cost (FIFO tiebreaker over
    insertion order). For SUM's `bench_digest` cross-process
    reproducibility, this is unacceptable.

    This cost model layers three priors, in order:

      1. **Children cost**, accumulated multiplicatively into the
         high bits — preserves egglog's "smaller subtree wins"
         intuition. Without this, a tree-leaf would always beat a
         tree-root regardless of structure.
      2. **Content hash**, sha256 over the expression's
         deterministic string form, taking the leading 8 bytes as
         the low 64 bits — provides a content-derived total order
         on e-class members with equal structural cost. The hash
         is computed once and depends only on the expression's
         identity, not on insertion order.
      3. **Implicit insertion-order tiebreak**, falls through to
         egglog's default if two expressions hash to the exact
         same 64 bits (probability ≈ 2⁻⁶⁴).

    The (children_cost × 2⁶⁴ + content_hash) layering is the
    layered-decomposition discipline named in
    docs/PHASE_26_EGGLOG_SPIKE_FINDINGS.md iteration 4: distinct
    priors handled in distinct bit ranges, with the prior that's
    most expensive to recover from (structural correctness)
    occupying the high bits.

    Implementation note: egglog-python (PR #357, the API in v11.4.0
    that we pin) accepts any orderable Python value from a cost
    model, not just `int`. An equivalent implementation could
    return `(structural_cost, str(expr))` as a tuple and let
    Python's lexicographic tuple comparison do the tiebreak.
    The hash approach is preferred here because (a) it's a
    fixed-size key regardless of string length, (b) the comparison
    cost is constant rather than O(string length), and (c) the
    2⁻⁶⁴ collision probability is below the substrate's other
    failure rates by many orders of magnitude.
    """
    base = sum(children_costs) * (1 << 64)
    digest = hashlib.sha256(str(expr).encode("utf-8")).digest()
    tiebreak = int.from_bytes(digest[:8], "big")
    return base + tiebreak


class EgglogStore:
    """In-process graph store backed by an egglog e-graph.

    The store keeps an authoritative Python set of triples
    alongside the e-graph for now. Two reasons:

      1. Egglog's Python bindings do not yet expose a stable
         "iterate over all e-classes of type X" API across the
         versions we'd ship against; pinning the spike to that
         would couple us to a moving target.
      2. The spike's measurements are about *insert/query speed
         and determinism on substrate workloads*, not about
         egglog's e-graph internals. The Python set is our
         ground truth; the e-graph is what we'd lean on for the
         equivalence-class queries in the *next* PR.

    The spike receipt records both: (a) the externally-observable
    behaviour (count, queries, hash) which uses the Python set,
    and (b) the egglog presence (registered fact count) which
    records that the e-graph is in the loop.
    """

    def __init__(self, *, eager_materialisation: bool = False) -> None:
        """Build a lazy-by-default egglog store.

        Args:
            eager_materialisation: If True, every `add_triple` call
                synchronously registers into the e-graph (the
                original behaviour). Default is False — registration
                deferred until `materialise_egraph()` or an e-class
                query forces it. The eager path is preserved for
                A/B comparison in the bench harness.
        """
        # Defer the egglog import + EGraph construction until first
        # use. Constructing an EGraph spins up the Rust backend
        # (multi-hundred-ms cost) — pointless for a store that the
        # caller never queries against the e-graph.
        self._eager = eager_materialisation
        self._egraph = None  # built lazily on first materialise
        self._TripleNode = None
        self._String = None
        self._triples: set[tuple[str, str, str]] = set()
        self._registered = 0
        self._pending: list[tuple[str, str, str]] = []

    def _ensure_egraph(self) -> None:
        if self._egraph is not None:
            return
        from egglog import EGraph, Expr, String, ruleset, rewrite

        class _SUMTriple(Expr):  # type: ignore[misc]
            """E-graph node type for substrate triples."""
            def __init__(self, s: String, p: String, o: String) -> None: ...

        # Bake the spike's "ownership symmetry" rewrite rule
        # alongside the e-graph + Triple class. This is the minimum
        # equivalence-class semantics worth measuring: active vs
        # passive forms of an "owns" relation collapse to the same
        # e-class. Phase C's importance-weighted SUM will need
        # richer rules; the spike only needs to demonstrate the
        # extract-with-cost path is exercised.
        @ruleset
        def _ownership_symmetry_rules(s: String, p: String, o: String):
            yield rewrite(
                _SUMTriple(s, String("owns"), o)
            ).to(
                _SUMTriple(o, String("owned_by"), s)
            )
            yield rewrite(
                _SUMTriple(o, String("owned_by"), s)
            ).to(
                _SUMTriple(s, String("owns"), o)
            )

        self._egraph = EGraph()
        self._TripleNode = _SUMTriple
        self._String = String
        self._rulesets = {"ownership_symmetry": _ownership_symmetry_rules}

    def info(self) -> GraphStoreInfo:
        try:
            import egglog
            ver = getattr(egglog, "__version__", "unknown")
        except ImportError:  # pragma: no cover
            ver = "unavailable"
        mode = "eager" if self._eager else "lazy"
        return GraphStoreInfo(
            name="egglog",
            version=str(ver),
            notes=(
                f"Spike-stage backend (materialisation={mode}). "
                "Holds triples in a Python set (authoritative for "
                "queries and content-hash); the egglog EGraph is "
                "built on demand for equivalence-class queries via "
                "`materialise_egraph()`. Pattern queries "
                "(find_objects / find_subjects) and content_hash "
                "never trigger materialisation."
            ),
        )

    def add_triple(self, triple: Triple) -> None:
        key = triple.as_tuple()
        if key in self._triples:
            return
        self._triples.add(key)
        if self._eager:
            self._ensure_egraph()
            node = self._TripleNode(
                self._String(triple.subject),
                self._String(triple.predicate),
                self._String(triple.object),
            )
            self._egraph.register(node)
            self._registered += 1
        else:
            self._pending.append(key)

    def add_triples(self, triples: list[Triple]) -> None:
        new_keys: list[tuple[str, str, str]] = []
        for t in triples:
            key = t.as_tuple()
            if key in self._triples:
                continue
            self._triples.add(key)
            new_keys.append(key)
        if not new_keys:
            return
        if self._eager:
            self._ensure_egraph()
            new_nodes = [
                self._TripleNode(
                    self._String(s), self._String(p), self._String(o),
                )
                for (s, p, o) in new_keys
            ]
            self._egraph.register(*new_nodes)
            self._registered += len(new_nodes)
        else:
            self._pending.extend(new_keys)

    def materialise_egraph(self) -> int:
        """Flush all pending triples into the e-graph in one batched
        register call. Idempotent — repeated calls with no new
        pending triples are a no-op. Returns the number of triples
        registered by this call (zero if nothing was pending).

        E-class queries call this before consulting the e-graph.
        Callers can also invoke it explicitly to amortise the cost
        before a known query batch.
        """
        if not self._pending:
            return 0
        self._ensure_egraph()
        flushed = self._pending
        self._pending = []
        nodes = [
            self._TripleNode(
                self._String(s), self._String(p), self._String(o),
            )
            for (s, p, o) in flushed
        ]
        self._egraph.register(*nodes)
        self._registered += len(nodes)
        return len(nodes)

    def egraph_pending(self) -> int:
        """How many triples are in the set but not yet in the e-graph.
        Zero in eager mode; non-zero in lazy mode until
        `materialise_egraph()` is called."""
        return len(self._pending)

    def count_triples(self) -> int:
        return len(self._triples)

    def egraph_registered(self) -> int:
        """How many facts are currently in the e-graph. Distinct
        from `count_triples` in lazy mode until materialisation."""
        return self._registered

    def iter_triples(self) -> Iterator[Triple]:
        for s, p, o in self._triples:
            yield Triple(s, p, o)

    def find_objects(self, subject: str, predicate: str) -> list[str]:
        return sorted(
            o for (s, p, o) in self._triples
            if s == subject and p == predicate
        )

    def find_subjects(self, predicate: str, object: str) -> list[str]:
        return sorted(
            s for (s, p, o) in self._triples
            if p == predicate and o == object
        )

    def content_hash(self) -> str:
        return _canonical_triples_hash(self.iter_triples())

    # ── Equivalence-class machinery ──────────────────────────────────
    #
    # The methods below are the *Phase C-shaped* surface: saturate the
    # e-graph under a registered ruleset, then extract canonical
    # representatives by cost. The spike measures these to answer
    # whether per-query e-class extraction is fast enough at library
    # scale to justify egglog as the substrate's query layer.

    def available_rulesets(self) -> list[str]:
        """Names of rulesets baked into this backend. Returns
        an empty list before the e-graph is materialised."""
        if self._egraph is None:
            return []
        return list(self._rulesets.keys())

    def saturate(self, ruleset_name: str = "ownership_symmetry") -> None:
        """Run egglog saturation on a named ruleset. Materialises
        the e-graph if needed. Saturation is order-independent per
        the egglog spec; a fresh process running the same triples +
        ruleset reaches the same fixed point.

        Raises ``KeyError`` if the ruleset is unknown.
        """
        self.materialise_egraph()
        if ruleset_name not in self._rulesets:
            raise KeyError(
                f"unknown ruleset {ruleset_name!r}; available: "
                f"{list(self._rulesets.keys())}"
            )
        self._egraph.run(self._rulesets[ruleset_name].saturate())

    def extract_canonical(self, triple: Triple, *, deterministic: bool = True):
        """Return the lowest-cost representative of the e-class
        containing ``triple``. After saturation under a symmetry
        ruleset, two triples in the same e-class will both extract
        to the same canonical form.

        Args:
            triple: The triple whose canonical form to extract.
            deterministic: If True (default), use a content-derived
                cost model that breaks ties via sha256 over the
                expression's string repr — guarantees the same
                canonical form across processes regardless of
                insertion order. If False, use egglog's default
                cost (FIFO tiebreaker over insertion order); the
                default exists for spike comparisons and matches
                iteration-3's pinned non-determinism.

        Caller is responsible for materialising and saturating
        first; this method does NOT auto-saturate (saturation is
        cost-amortised across many extracts).
        """
        self.materialise_egraph()
        node = self._TripleNode(
            self._String(triple.subject),
            self._String(triple.predicate),
            self._String(triple.object),
        )
        if deterministic:
            return self._egraph.extract(node, cost_model=_content_hash_cost_model)
        return self._egraph.extract(node)


# Confirm at import time that the store satisfies the GraphStore
# Protocol. mypy / pyright will flag it; this also documents intent.
_proto_check: GraphStore = EgglogStore.__new__(EgglogStore)  # noqa: F841
