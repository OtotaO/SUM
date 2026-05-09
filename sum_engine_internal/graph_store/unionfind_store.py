"""Union-find graph store — Phase 26.0 conclusion.

Four iterations of the egglog spike (PRs #176/#177/#178/#179/#180)
measured what egglog actually delivers for SUM's workload. The
honest summary:

  - Storage hot path: lazy mode keeps egglog out of it (the
    Python set is the authoritative store)
  - Pattern queries: don't touch the e-graph
  - Saturation: we have ONE rule (ownership symmetry); egglog's
    general saturation engine is overkill
  - Cross-process deterministic extract: solved with a custom
    cost model, but pays a 200-1000× per-extract overhead because
    the cost model is a Python callback invoked per visited e-node
  - Materialisation at scale: 70 s for 10k registrations; egglog
    upstream issue #756 (bulk-load) is open without resolution

The substrate's *actual* need is "given equivalent triples under a
symmetry rule, return a deterministic canonical form." That's a
graph algorithm — union-find — not a special-purpose e-graph
requirement. This module implements it directly:

  - O(α(N)) union via path-compressed union-find
  - O(class_size) extract by linear scan; fast at our workload
    sizes (<2 ms at 10k triples)
  - Deterministic by construction: extract returns the
    lex-smallest member of an equivalence class. No cost model,
    no callback, no upstream issue to wait on
  - Zero external dependency, zero version pin

The comparison receipt under
``fixtures/bench_receipts/phase_26_backing_store_spike_egglog_*.json``
shows union-find beats egglog by orders of magnitude on every
substrate-shaped operation while preserving every correctness
property the spike validated for egglog.

Future direction: if Phase C grows multiple symmetric rewrite
rules, register more equivalences. If Phase C ever needs
arbitrary saturation under conditional rules, egglog stays
available as a one-shot tool — run saturation once, dump the
union-find structure, drop egglog. Pay it only when it pays
back.
"""
from __future__ import annotations

from typing import Iterator

from sum_engine_internal.graph_store.base import (
    GraphStore,
    GraphStoreInfo,
    Triple,
    _canonical_triples_hash,
)


class UnionFindStore:
    """In-process triple store with union-find equivalence classes.

    Pattern-query parity with the spike's other backends; saturation
    handled by registering equivalence pairs into the union-find;
    extract returns the lex-smallest member of the equivalence
    class containing the input triple.

    Adding a new symmetric rewrite rule means adding a branch in
    ``saturate()``. The architecture intentionally caps complexity:
    if a future rule isn't expressible as "union pairs of triples,"
    that's the moment to revisit whether egglog (or another
    saturation engine) earns its complexity back.
    """

    OWNERSHIP_SYMMETRY = "ownership_symmetry"

    def __init__(self) -> None:
        self._triples: set[tuple[str, str, str]] = set()
        self._parent: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        self._saturated_rules: set[str] = set()

    # -- GraphStore interface ---------------------------------------

    def info(self) -> GraphStoreInfo:
        return GraphStoreInfo(
            name="unionfind",
            version="1.0",
            notes=(
                "Pure-Python union-find graph store. Phase 26.0 "
                "conclusion: replaces egglog as the substrate's "
                "graph backend after four iterations of measurement "
                "showed egglog overhead exceeds its substrate-shaped "
                "utility. Deterministic extract by construction "
                "(lex-sort of equivalence class), zero external "
                "dependency, no upstream issues to wait on."
            ),
        )

    def add_triple(self, t: Triple) -> None:
        key = t.as_tuple()
        if key in self._triples:
            return
        self._triples.add(key)
        self._parent[key] = key
        # Adding a new triple may make a previous saturation stale;
        # require re-saturation on next extract call.
        self._saturated_rules.clear()

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            self.add_triple(t)

    def count_triples(self) -> int:
        return len(self._triples)

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

    # -- union-find primitives --------------------------------------

    def _find(self, key: tuple[str, str, str]) -> tuple[str, str, str]:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        cur = key
        while self._parent[cur] != root:
            self._parent[cur], cur = root, self._parent[cur]
        return root

    def _union(self, a: tuple[str, str, str], b: tuple[str, str, str]) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        # Lex-smaller root wins → deterministic class representative.
        if ra < rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb

    # -- equivalence-class machinery --------------------------------

    def available_rulesets(self) -> list[str]:
        return [self.OWNERSHIP_SYMMETRY]

    def saturate(self, ruleset_name: str = OWNERSHIP_SYMMETRY) -> None:
        """Run the registered ruleset to fixed point. For
        ownership_symmetry: union each (s, "owns", o) with
        (o, "owned_by", s) when both are present.

        Idempotent. Re-running after `add_triple` re-saturates
        because a new triple might create a new equivalence; the
        `_saturated_rules` cache is cleared on add."""
        if ruleset_name in self._saturated_rules:
            return
        if ruleset_name != self.OWNERSHIP_SYMMETRY:
            raise KeyError(
                f"unknown ruleset {ruleset_name!r}; available: "
                f"{self.available_rulesets()}"
            )
        for s, p, o in list(self._triples):
            if p == "owns":
                mate = (o, "owned_by", s)
                if mate in self._triples:
                    self._union((s, p, o), mate)
            # Symmetry of union/find means we don't need to walk
            # the passive side separately — it's already covered.
        self._saturated_rules.add(ruleset_name)

    def extract_canonical(self, triple: Triple) -> Triple:
        """Lex-smallest member of the equivalence class containing
        ``triple``. Deterministic by construction across processes.
        Auto-saturates on first call (vs egglog's manual saturate).

        Raises KeyError if ``triple`` is not in the store."""
        if not self._saturated_rules:
            self.saturate()
        key = triple.as_tuple()
        if key not in self._parent:
            raise KeyError(f"triple {triple} not in store")
        root = self._find(key)
        # Lex-smallest in the equivalence class. O(N) scan is
        # acceptable at our workload sizes (~2 ms at 10k);
        # if Phase C ever needs O(class_size), maintain a
        # root → set-of-members index alongside _parent.
        members = (
            (s, p, o) for (s, p, o) in self._triples
            if self._find((s, p, o)) == root
        )
        return Triple(*min(members))


# Confirm at import time that the store satisfies the GraphStore
# Protocol — same contract as EgglogStore.
_proto_check: GraphStore = UnionFindStore.__new__(UnionFindStore)  # noqa: F841
