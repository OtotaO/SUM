"""Phase 26.0 conclusion tests — UnionFindStore.

Mirrors the structure of `Tests/test_graph_store_egglog.py` so the
two backends are directly comparable. Adds two cross-backend tests
that pin parity (content_hash + canonical extracts agree).

The egglog-specific test file kept its bench-canary
(`test_deterministic_extract_has_measurable_overhead`); the
UnionFindStore equivalent would be vacuous (deterministic by
construction; nothing to measure overhead against), so it's
not duplicated here.
"""
from __future__ import annotations

import pytest


def _t(s: str, p: str, o: str):
    from sum_engine_internal.graph_store import Triple
    return Triple(s, p, o)


@pytest.fixture
def store():
    from sum_engine_internal.graph_store.unionfind_store import UnionFindStore
    return UnionFindStore()


def test_info_identifies_backend(store):
    info = store.info()
    assert info.name == "unionfind"
    assert info.notes


def test_add_then_count(store):
    store.add_triple(_t("alice", "likes", "cats"))
    store.add_triple(_t("bob", "owns", "rex"))
    assert store.count_triples() == 2


def test_add_is_idempotent(store):
    store.add_triple(_t("alice", "likes", "cats"))
    store.add_triple(_t("alice", "likes", "cats"))
    store.add_triple(_t("alice", "likes", "cats"))
    assert store.count_triples() == 1


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
    from sum_engine_internal.graph_store.unionfind_store import UnionFindStore
    triples = [
        _t("alice", "likes", "cats"),
        _t("bob", "owns", "rex"),
        _t("carol", "writes", "code"),
    ]
    a = UnionFindStore()
    a.add_triples(triples)
    b = UnionFindStore()
    b.add_triples(list(reversed(triples)))
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_when_a_triple_is_added(store):
    store.add_triple(_t("alice", "likes", "cats"))
    h1 = store.content_hash()
    store.add_triple(_t("bob", "owns", "rex"))
    h2 = store.content_hash()
    assert h1 != h2


def test_content_hash_format_is_sha256_prefix(store):
    store.add_triple(_t("alice", "likes", "cats"))
    h = store.content_hash()
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_saturate_collapses_active_and_passive_ownership(store):
    """The substrate's actual need: active and passive forms must
    extract to the same canonical representative."""
    store.add_triples([
        _t("alice", "owns", "rex"),
        _t("rex", "owned_by", "alice"),
        _t("bob", "likes", "cats"),
    ])
    store.saturate("ownership_symmetry")
    can_active = str(store.extract_canonical(_t("alice", "owns", "rex")))
    can_passive = str(store.extract_canonical(_t("rex", "owned_by", "alice")))
    assert can_active == can_passive
    # Unrelated triple is unchanged
    can_other = str(store.extract_canonical(_t("bob", "likes", "cats")))
    assert "bob" in can_other and "likes" in can_other


def test_extract_canonical_auto_saturates(store):
    """Convenience over egglog's manual saturate path: extract
    auto-saturates if rules are pending."""
    store.add_triples([
        _t("alice", "owns", "rex"),
        _t("rex", "owned_by", "alice"),
    ])
    # NO explicit saturate() call
    can = str(store.extract_canonical(_t("rex", "owned_by", "alice")))
    # Auto-saturated → both forms collapse → lex-smallest wins
    can2 = str(store.extract_canonical(_t("alice", "owns", "rex")))
    assert can == can2


def test_extract_canonical_is_deterministic_across_insertion_order():
    """The headline correctness claim — same as the egglog
    iteration-4 test, but by construction (lex-sort), not by a
    custom cost model."""
    from sum_engine_internal.graph_store.unionfind_store import UnionFindStore
    triples = [
        _t("alice", "owns", "rex"),
        _t("rex", "owned_by", "alice"),
    ]
    a = UnionFindStore()
    a.add_triples(triples)
    can_a = str(a.extract_canonical(_t("alice", "owns", "rex")))

    b = UnionFindStore()
    b.add_triples(list(reversed(triples)))
    can_b = str(b.extract_canonical(_t("alice", "owns", "rex")))

    assert can_a == can_b


def test_saturate_unknown_ruleset_raises(store):
    store.add_triple(_t("alice", "owns", "rex"))
    with pytest.raises(KeyError, match="unknown ruleset"):
        store.saturate("does_not_exist")


def test_extract_unknown_triple_raises(store):
    store.add_triple(_t("alice", "owns", "rex"))
    with pytest.raises(KeyError, match="not in store"):
        store.extract_canonical(_t("not", "in", "store"))


def test_add_after_saturate_invalidates_saturation(store):
    """If a new triple arrives that creates a new equivalence,
    the cached saturation must be cleared so the next extract
    sees the new equivalence."""
    store.add_triple(_t("alice", "owns", "rex"))
    store.saturate()
    can1 = str(store.extract_canonical(_t("alice", "owns", "rex")))
    # Now add the symmetric form — should now collapse
    store.add_triple(_t("rex", "owned_by", "alice"))
    can2 = str(store.extract_canonical(_t("rex", "owned_by", "alice")))
    can3 = str(store.extract_canonical(_t("alice", "owns", "rex")))
    assert can2 == can3, (
        "symmetric form added post-saturation didn't collapse with active"
    )


# -- Cross-backend parity tests ----------------------------------------


def test_content_hash_matches_egglog_backend():
    """For the same triple input, both backends must produce
    identical content_hash (it's defined over the triple set,
    backend-independent — any divergence is a bug)."""
    eg_module = pytest.importorskip("egglog")  # skip if egglog absent
    from sum_engine_internal.graph_store.unionfind_store import UnionFindStore
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    triples = [
        _t("alice", "likes", "cats"),
        _t("bob", "owns", "rex"),
        _t("rex", "owned_by", "alice"),
    ]
    uf = UnionFindStore()
    uf.add_triples(triples)
    eg = EgglogStore()
    eg.add_triples(triples)
    assert uf.content_hash() == eg.content_hash()


def test_pattern_queries_match_egglog_backend():
    pytest.importorskip("egglog")
    from sum_engine_internal.graph_store.unionfind_store import UnionFindStore
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    triples = [
        _t("alice", "likes", "cats"),
        _t("alice", "likes", "dogs"),
        _t("bob", "owns", "rex"),
    ]
    uf = UnionFindStore(); uf.add_triples(triples)
    eg = EgglogStore(); eg.add_triples(triples)
    for s, p in [("alice", "likes"), ("bob", "owns"), ("nobody", "knows")]:
        assert uf.find_objects(s, p) == eg.find_objects(s, p)
    for p, o in [("owns", "rex"), ("likes", "cats")]:
        assert uf.find_subjects(p, o) == eg.find_subjects(p, o)
