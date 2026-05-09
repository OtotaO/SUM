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
        from egglog import EGraph, Expr, String

        class _SUMTriple(Expr):  # type: ignore[misc]
            """E-graph node type for substrate triples."""
            def __init__(self, s: String, p: String, o: String) -> None: ...

        self._egraph = EGraph()
        self._TripleNode = _SUMTriple
        self._String = String

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


# Confirm at import time that the store satisfies the GraphStore
# Protocol. mypy / pyright will flag it; this also documents intent.
_proto_check: GraphStore = EgglogStore.__new__(EgglogStore)  # noqa: F841
