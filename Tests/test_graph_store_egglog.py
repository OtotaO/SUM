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
    # The e-graph should also see only one register call (the
    # spike backend dedups before registering)
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
