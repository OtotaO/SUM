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


def test_saturate_collapses_active_and_passive_ownership(store):
    """The baked-in `ownership_symmetry` ruleset rewrites
    `Triple(s, "owns", o)` ⟺ `Triple(o, "owned_by", s)`. After
    saturation, both forms extract to the same canonical
    representative within a single store/process."""
    store.add_triples([
        _t("alice", "owns", "rex"),
        _t("rex", "owned_by", "alice"),  # equivalent
        _t("bob", "likes", "cats"),       # unrelated
    ])
    store.saturate("ownership_symmetry")
    ext_active = str(store.extract_canonical(_t("alice", "owns", "rex")))
    ext_passive = str(store.extract_canonical(_t("rex", "owned_by", "alice")))
    assert ext_active == ext_passive, (
        f"active and passive forms did not collapse: "
        f"{ext_active!r} vs {ext_passive!r}"
    )
    # The unrelated triple is unchanged
    ext_other = str(store.extract_canonical(_t("bob", "likes", "cats")))
    assert "bob" in ext_other and "likes" in ext_other


def test_saturate_unknown_ruleset_raises():
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    s = EgglogStore()
    s.add_triple(_t("alice", "owns", "rex"))
    s.materialise_egraph()
    with pytest.raises(KeyError, match="unknown ruleset"):
        s.saturate("does_not_exist")


def test_extract_canonical_matches_within_a_single_process(store):
    """Within one process/store, repeated extraction of equivalent
    triples returns the same canonical form. (Cross-process
    determinism is a separate question — see the next test.)"""
    store.add_triples([
        _t("alice", "owns", "rex"),
        _t("rex", "owned_by", "alice"),
    ])
    store.saturate("ownership_symmetry")
    ext1 = str(store.extract_canonical(_t("alice", "owns", "rex")))
    ext2 = str(store.extract_canonical(_t("alice", "owns", "rex")))
    assert ext1 == ext2


def test_default_cost_is_insertion_order_sensitive_on_ties():
    """KNOWN LIMITATION pinned as a test: egglog's default
    extract-with-cost is insertion-order-dependent when two
    expressions in the same e-class have equal cost. Forward
    insertion of (active, passive) extracts the active form;
    reversed insertion extracts the passive form.

    For Phase C cross-process determinism, a custom cost function
    with a content-derived tie-breaker is required. This test
    exists so the limitation cannot regress silently into a
    "looks deterministic" assumption."""
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    triples = [
        _t("alice", "owns", "rex"),
        _t("rex", "owned_by", "alice"),
    ]

    store_fwd = EgglogStore()
    store_fwd.add_triples(triples)
    store_fwd.saturate("ownership_symmetry")
    canonical_fwd = str(store_fwd.extract_canonical(_t("alice", "owns", "rex")))

    store_rev = EgglogStore()
    store_rev.add_triples(list(reversed(triples)))
    store_rev.saturate("ownership_symmetry")
    canonical_rev = str(store_rev.extract_canonical(_t("alice", "owns", "rex")))

    # The two extractions DIFFER under the default cost — this is
    # the limitation we're pinning. If/when egglog or our backend
    # makes default extract deterministic, this test will start
    # failing and we should celebrate by inverting the assertion.
    assert canonical_fwd != canonical_rev, (
        "If this test starts failing, egglog default extract has "
        "become deterministic on ties. Update the spike findings "
        "doc and invert this assertion."
    )


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
