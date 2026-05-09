"""Backend-agnostic graph store interface.

The interface is shaped to the substrate's actual access patterns,
not to a general property-graph DB. Five operations are enough to
exercise every Phase 26.0 spike candidate on the same workload:

  - `add_triple` — insert a fact
  - `count_triples` — total cardinality
  - `find_objects` / `find_subjects` — partial-pattern queries
    (the v3.x bench's hot path)
  - `content_hash` — sha256 over a canonical serialisation of the
    full graph state, used to verify cross-process determinism

`Triple` is a value-typed dataclass so callers do not need to know
the backend's internal representation. The backend is free to
canonicalise, intern, or e-classify internally; the interface
guarantees that two backends fed the same triples in the same
order produce the same `count_triples()` and the same answers to
the find queries.

Whether `content_hash` is identical *across* backends is an open
spike question: every backend uses sha256 over JCS-canonical bytes
of `sorted(triples)`, which makes the hash a property of the
*triple set*, not of any backend internal — so identical content
hashes across backends are achievable, and that's what the spike
asserts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True, slots=True)
class Triple:
    """A (subject, predicate, object) fact. Strings are normalised
    by the caller; the store does not lower-case or strip."""
    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


@dataclass(frozen=True, slots=True)
class GraphStoreInfo:
    """Self-description of a backend, captured into the spike
    receipt so two receipts from different machines can be
    compared meaningfully."""
    name: str
    version: str
    """Free-form version string of the underlying library."""
    notes: str = ""


class GraphStore(Protocol):
    """Minimum surface required for a Phase 26.0 spike candidate.

    Backends implement these methods on top of their native data
    structures. The `_canonical_triples` helper below standardises
    the content-hash so backends do not each invent their own
    hashing scheme.
    """

    def info(self) -> GraphStoreInfo: ...
    def add_triple(self, triple: Triple) -> None: ...
    def add_triples(self, triples: list[Triple]) -> None:
        """Default loop implementation; backends may override for
        bulk-insert speed."""
        for t in triples:
            self.add_triple(t)
    def count_triples(self) -> int: ...
    def iter_triples(self) -> Iterator[Triple]: ...
    def find_objects(self, subject: str, predicate: str) -> list[str]: ...
    def find_subjects(self, predicate: str, object: str) -> list[str]: ...
    def content_hash(self) -> str:
        """Default: sha256 over JCS-canonical bytes of sorted
        (subject, predicate, object) tuples. Any backend may
        override, but the default is sufficient because the
        backend-internal structure shouldn't leak into the hash —
        the *fact set* is what's being identified."""
        return _canonical_triples_hash(self.iter_triples())


def _canonical_triples_hash(triples: Iterator[Triple]) -> str:
    """JCS-flavoured canonical hash over a triple iterable.

    Sort by (subject, predicate, object), serialise as
    `subject\\x1Fpredicate\\x1Fobject\\n` lines, sha256 the bytes.
    The unit-separator `\\x1F` is unambiguous inside ascii prose;
    if a triple ever contained one literally we'd need to escape,
    but the SUM extractors don't produce such strings.
    """
    sorted_tuples = sorted(t.as_tuple() for t in triples)
    h = hashlib.sha256()
    for s, p, o in sorted_tuples:
        h.update(s.encode("utf-8"))
        h.update(b"\x1f")
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
        h.update(o.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()
