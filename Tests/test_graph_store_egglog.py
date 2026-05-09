"""Phase 26.0 spike tests — egglog graph-store backend.

Pin the substrate-shaped invariants on top of egglog. The spike's
*measurement* receipt lives in
`fixtures/bench_receipts/phase_26_backing_store_spike_egglog_*.json`;
this file is the contract.
"""
from __future__ import annotations

import pytest

# Skip the whole file if egglog isn't installed (it's an optional
# spike dependency, not a runtime SUM dependency).
egglog = pytest.importorskip("egglog")


@pytest.fixture
def store():
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    return EgglogStore()


def _t(s: str, p: str, o: str):
    from sum_engine_internal.graph_store import Triple
    return Triple(s, p, o)


def test_info_identifies_backend(store):
    info = store.info()
    assert info.name == "egglog"
    assert info.notes  # non-empty


def test_add_then_count(store):
    store.add_triple(_t("alice", "likes", "cats"))
    store.add_triple(_t("bob", "owns", "rex"))
    assert store.count_triples() == 2


def test_add_is_idempotent(store):
    store.add_triple(_t("alice", "likes", "cats"))
    store.add_triple(_t("alice", "likes", "cats"))
    store.add_triple(_t("alice", "likes", "cats"))
    assert store.count_triples() == 1
    # The backend dedups before pending — only one entry queued for
    # the e-graph (lazy mode) and only one would land in it after
    # materialisation.
    assert store.egraph_pending() == 1
    store.materialise_egraph()
    assert store.egraph_registered() == 1


def test_find_objects_returns_sorted_unique(store):
    store.add_triples([
        _t("alice", "likes", "cats"),
        _t("alice", "likes", "dogs"),
        _t("alice", "likes", "birds"),
        _t("bob", "likes", "rex"),
    ])
    assert store.find_objects("alice", "likes") == ["birds", "cats", "dogs"]


def test_find_subjects_returns_sorted_unique(store):
    store.add_triples([
        _t("alice", "owns", "rex"),
        _t("bob", "owns", "rex"),
        _t("carol", "owns", "rex"),
    ])
    assert store.find_subjects("owns", "rex") == ["alice", "bob", "carol"]


def test_find_returns_empty_for_unmatched_pattern(store):
    store.add_triple(_t("alice", "likes", "cats"))
    assert store.find_objects("nobody", "knows") == []
    assert store.find_subjects("knows", "nothing") == []


def test_iter_triples_returns_all(store):
    triples = [
        _t("alice", "likes", "cats"),
        _t("bob", "owns", "rex"),
    ]
    store.add_triples(triples)
    assert set(t.as_tuple() for t in store.iter_triples()) == set(
        t.as_tuple() for t in triples
    )


def test_content_hash_is_deterministic_under_insertion_order():
    """The defining spike invariant. Egglog's saturation is
    order-independent (per its spec); our content hash sorts
    canonically before sha256, so it doesn't depend on egglog
    internals. Both stores must produce the same hash."""
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    triples = [
        _t("alice", "likes", "cats"),
        _t("bob", "owns", "rex"),
        _t("carol", "writes", "code"),
        _t("dave", "drives", "tesla"),
    ]
    a = EgglogStore()
    a.add_triples(triples)
    b = EgglogStore()
    b.add_triples(list(reversed(triples)))
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_when_a_triple_is_added(store):
    store.add_triples([_t("alice", "likes", "cats")])
    h1 = store.content_hash()
    store.add_triple(_t("bob", "owns", "rex"))
    h2 = store.content_hash()
    assert h1 != h2


def test_content_hash_format_is_sha256_prefix(store):
    store.add_triple(_t("alice", "likes", "cats"))
    h = store.content_hash()
    assert h.startswith("sha256:")
    # 64 hex chars after the prefix
    assert len(h) == len("sha256:") + 64


def test_lazy_mode_defers_egraph_registration(store):
    """Default mode is lazy: add_triple does not register into the
    e-graph. The pending counter increases; the registered counter
    stays at zero until materialise_egraph() is called."""
    store.add_triple(_t("alice", "likes", "cats"))
    store.add_triple(_t("bob", "owns", "rex"))
    assert store.count_triples() == 2
    assert store.egraph_pending() == 2
    assert store.egraph_registered() == 0

    flushed = store.materialise_egraph()
    assert flushed == 2
    assert store.egraph_pending() == 0
    assert store.egraph_registered() == 2

    # idempotent: second call with no new pending is a no-op
    flushed2 = store.materialise_egraph()
    assert flushed2 == 0
    assert store.egraph_registered() == 2


def test_eager_mode_registers_immediately():
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    eager = EgglogStore(eager_materialisation=True)
    eager.add_triple(_t("alice", "likes", "cats"))
    assert eager.count_triples() == 1
    assert eager.egraph_pending() == 0
    assert eager.egraph_registered() == 1


def test_lazy_and_eager_modes_produce_identical_content_hash():
    """The defining cross-mode invariant: switching materialisation
    strategy MUST NOT change the content hash. The hash is over the
    triple set, which is mode-independent."""
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    triples = [
        _t("alice", "likes", "cats"),
        _t("bob", "owns", "rex"),
        _t("carol", "writes", "code"),
    ]
    lazy = EgglogStore()
    lazy.add_triples(triples)
    eager = EgglogStore(eager_materialisation=True)
    eager.add_triples(triples)
    assert lazy.content_hash() == eager.content_hash()


def test_pattern_queries_work_in_lazy_mode_without_materialisation(store):
    """find_objects / find_subjects must work on the Python set and
    NEVER trigger materialisation. If they did, lazy mode's whole
    storage-cost win would evaporate the moment a query hit."""
    store.add_triples([
        _t("alice", "likes", "cats"),
        _t("alice", "likes", "dogs"),
    ])
    assert store.egraph_pending() == 2
    assert store.find_objects("alice", "likes") == ["cats", "dogs"]
    # still pending — pattern queries do not force the e-graph
    assert store.egraph_pending() == 2
    assert store.egraph_registered() == 0


def test_canonical_hash_is_independent_of_backend():
    """The default content_hash on `GraphStore` is JCS-style sort
    over sha256 — by construction backend-independent. This test
    pins that the egglog backend uses that default (rather than
    leaking egglog internals into the hash) by hashing the same
    triple set via the helper directly and via the backend, and
    checking they agree."""
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    from sum_engine_internal.graph_store.base import _canonical_triples_hash
    triples = [
        _t("alice", "likes", "cats"),
        _t("bob", "owns", "rex"),
    ]
    store = EgglogStore()
    store.add_triples(triples)
    direct_hash = _canonical_triples_hash(iter(triples))
    assert store.content_hash() == direct_hash
