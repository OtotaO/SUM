"""Egglog-backed graph store for the Phase 26.0 spike.

Egglog (https://github.com/egraphs-good/egglog) combines an
e-graph (union-find over expressions) with a Datalog-style query
engine. For the spike's purposes we use the simplest mode: an
e-graph that holds `Triple(subject, predicate, object)` expressions,
inserted as plain facts, queried via the e-graph's enumeration
surface, and hashed externally with the substrate's standard
sha256-over-canonical-bytes scheme.

What we are NOT doing in the spike (deliberately, to keep scope
honest):

  - **No equivalence-class machinery.** The whole reason egglog
    is a candidate is its built-in extract-with-cost over
    e-classes — but exercising that requires defining cost
    functions and rewrite rules that match Phase C's
    importance-weighted SUM. That's the *next* PR if the spike
    measurements pass. This PR establishes only that egglog can
    serve as a graph backing store at all.
  - **No persistence.** Each `EgglogStore()` is process-local.
    Phase 26.1 schema migration handles persistence; spike
    measures access patterns.

Determinism note: egglog itself is order-independent for
saturation, but our spike does not run saturation — we hold the
graph as plain facts. So determinism here reduces to the
determinism of `_canonical_triples_hash`, which is JCS-style sort
over sha256, independent of egglog's internals.
"""
from __future__ import annotations

from typing import Iterator

from sum_engine_internal.graph_store.base import (
    GraphStore,
    GraphStoreInfo,
    Triple,
    _canonical_triples_hash,
)


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

    def __init__(self) -> None:
        # Lazy import — egglog has heavy startup cost (compiles
        # the Rust backend on first import) and SUM should not
        # pay it for callers that never touch this module.
        from egglog import EGraph, Expr, String

        class _SUMTriple(Expr):  # type: ignore[misc]
            """E-graph node type for substrate triples."""
            def __init__(self, s: String, p: String, o: String) -> None: ...

        self._egraph = EGraph()
        self._TripleNode = _SUMTriple
        self._String = String
        self._triples: set[tuple[str, str, str]] = set()
        self._registered = 0

    def info(self) -> GraphStoreInfo:
        try:
            import egglog
            ver = getattr(egglog, "__version__", "unknown")
        except ImportError:  # pragma: no cover
            ver = "unavailable"
        return GraphStoreInfo(
            name="egglog",
            version=str(ver),
            notes=(
                "Spike-stage backend. Holds triples in both an "
                "egglog EGraph (for future equivalence-class work) "
                "and a Python set (authoritative for queries and "
                "content-hash). Phase 26.0 measures access patterns "
                "only; equivalence-class queries are the next PR."
            ),
        )

    def add_triple(self, triple: Triple) -> None:
        key = triple.as_tuple()
        if key in self._triples:
            return
        self._triples.add(key)
        node = self._TripleNode(
            self._String(triple.subject),
            self._String(triple.predicate),
            self._String(triple.object),
        )
        self._egraph.register(node)
        self._registered += 1

    def add_triples(self, triples: list[Triple]) -> None:
        new_nodes = []
        for t in triples:
            key = t.as_tuple()
            if key in self._triples:
                continue
            self._triples.add(key)
            new_nodes.append(self._TripleNode(
                self._String(t.subject),
                self._String(t.predicate),
                self._String(t.object),
            ))
        if new_nodes:
            self._egraph.register(*new_nodes)
            self._registered += len(new_nodes)

    def count_triples(self) -> int:
        return len(self._triples)

    def egraph_registered(self) -> int:
        """How many facts have been registered into the e-graph.
        Distinct from `count_triples` only as a sanity check that
        the e-graph is in the loop."""
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


# Confirm at import time that the store satisfies the GraphStore
# Protocol. mypy / pyright will flag it; this also documents intent.
_proto_check: GraphStore = EgglogStore.__new__(EgglogStore)  # noqa: F841
