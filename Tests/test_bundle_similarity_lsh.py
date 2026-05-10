"""BundleSimilarityIndex — MinHash-LSH over bundle axiom sets.

Pins the substrate-level bundle-vs-bundle similarity surface:

  - signature_for_axioms: empty input → empty signature; identical
    axiom sets → identical signature; permutation invariance
  - BundleSimilarityIndex: bands × rows == num_perm invariant;
    add / remove / replace; query_candidates returns identical
    bundles; query does NOT return disjoint bundles; query_top_k
    sorted desc; jaccard(a, b) round-trips
  - LSH S-curve behavior: high-similarity bundles surface as
    candidates; low-similarity bundles do not
"""
from __future__ import annotations

import pytest

from sum_engine_internal.research.lsh import (
    BundleSimilarityIndex, signature_for_axioms,
)


_AXIOMS_A = [
    "alice||likes||cats",
    "bob||owns||dog",
    "carol||builds||house",
    "dave||writes||book",
    "eve||teaches||class",
    "frank||paints||mural",
    "grace||plays||piano",
    "henry||reads||novel",
]


# -- signature_for_axioms --------------------------------------------


def test_signature_empty_axioms_is_empty():
    sig = signature_for_axioms([])
    assert sig.is_empty()


def test_signature_identical_axiom_sets_match():
    sig_a = signature_for_axioms(_AXIOMS_A)
    sig_b = signature_for_axioms(_AXIOMS_A)
    assert sig_a.jaccard(sig_b) == 1.0


def test_signature_permutation_invariant():
    """Axioms are a SET — permuting the input must not change the
    signature (MinHash takes the min over the hash family, which is
    permutation-invariant by construction)."""
    sig_a = signature_for_axioms(_AXIOMS_A)
    sig_b = signature_for_axioms(reversed(_AXIOMS_A))
    assert sig_a.hashes == sig_b.hashes


def test_signature_disjoint_axiom_sets_have_zero_jaccard():
    a = ["x||p||y" for _ in range(8)]
    a = [f"a{i}||p||v" for i in range(8)]
    b = [f"b{i}||q||w" for i in range(8)]
    sig_a = signature_for_axioms(a)
    sig_b = signature_for_axioms(b)
    # MinHash estimator: 0 with high probability for fully-disjoint
    # 8-element sets at num_perm=128. Allow a small floor for
    # accidental collisions.
    assert sig_a.jaccard(sig_b) <= 0.05


# -- BundleSimilarityIndex construction ------------------------------


def test_index_rejects_mismatched_bands_rows():
    with pytest.raises(ValueError):
        BundleSimilarityIndex(num_perm=128, bands=10, rows=10)


def test_index_rejects_zero_bands_or_rows():
    with pytest.raises(ValueError):
        BundleSimilarityIndex(num_perm=128, bands=0, rows=8)
    with pytest.raises(ValueError):
        BundleSimilarityIndex(num_perm=128, bands=16, rows=0)


def test_index_default_construction():
    idx = BundleSimilarityIndex()
    assert idx.num_perm == 128 and idx.bands == 16 and idx.rows == 8
    assert len(idx) == 0


# -- add / remove / replace ------------------------------------------


def test_index_add_then_contains():
    idx = BundleSimilarityIndex()
    idx.add("b1", _AXIOMS_A)
    assert "b1" in idx
    assert len(idx) == 1


def test_index_remove_clears_membership():
    idx = BundleSimilarityIndex()
    idx.add("b1", _AXIOMS_A)
    idx.remove("b1")
    assert "b1" not in idx
    assert len(idx) == 0
    # Removing a non-existent id must be a no-op
    idx.remove("nope")


def test_index_replace_does_not_double_count():
    idx = BundleSimilarityIndex()
    idx.add("b1", _AXIOMS_A)
    idx.add("b1", _AXIOMS_A[:4])  # replace with a smaller set
    assert len(idx) == 1
    # The replaced signature must be the new (smaller) one
    sig_now = idx._sigs["b1"]
    sig_new = signature_for_axioms(_AXIOMS_A[:4])
    assert sig_now.hashes == sig_new.hashes


# -- query_candidates ------------------------------------------------


def test_query_candidates_returns_identical_bundle():
    idx = BundleSimilarityIndex()
    idx.add("b1", _AXIOMS_A)
    candidates = idx.query_candidates(_AXIOMS_A)
    assert "b1" in candidates


def test_query_candidates_omits_fully_disjoint_bundle():
    """An LSH index of a fully-disjoint bundle should not surface
    as a candidate. Two disjoint sets share NO band exactly with
    overwhelming probability at num_perm=128, bands=16."""
    idx = BundleSimilarityIndex()
    idx.add("disjoint", [f"x{i}||p||v" for i in range(8)])
    candidates = idx.query_candidates(_AXIOMS_A)
    assert "disjoint" not in candidates


def test_query_candidates_surfaces_high_similarity_bundle():
    """A near-duplicate bundle (jaccard ~0.875: 7/8 axioms match)
    must surface above the LSH threshold (~0.71). Tests the
    sensitivity side of the S-curve."""
    idx = BundleSimilarityIndex()
    near_dup = list(_AXIOMS_A[:-1]) + ["zoe||cooks||pasta"]  # 7/8 overlap
    idx.add("near", near_dup)
    candidates = idx.query_candidates(_AXIOMS_A)
    assert "near" in candidates


# -- query_top_k -----------------------------------------------------


def test_query_top_k_sorted_descending():
    idx = BundleSimilarityIndex()
    idx.add("identical", _AXIOMS_A)
    idx.add("near", list(_AXIOMS_A[:-1]) + ["zoe||cooks||pasta"])
    top = idx.query_top_k(_AXIOMS_A, k=5)
    assert len(top) >= 1
    scores = [j for _, j in top]
    assert scores == sorted(scores, reverse=True)
    # The identical bundle must be the top hit at jaccard==1.0
    assert top[0][0] == "identical"
    assert top[0][1] == 1.0


def test_query_top_k_respects_min_jaccard_filter():
    idx = BundleSimilarityIndex()
    idx.add("identical", _AXIOMS_A)
    idx.add("near", list(_AXIOMS_A[:-1]) + ["zoe||cooks||pasta"])
    top = idx.query_top_k(_AXIOMS_A, k=5, min_jaccard=0.95)
    # Only the identical bundle (jaccard=1.0) should pass
    ids = {bid for bid, _ in top}
    assert ids == {"identical"}


def test_query_top_k_returns_empty_for_zero_k():
    idx = BundleSimilarityIndex()
    idx.add("b1", _AXIOMS_A)
    assert idx.query_top_k(_AXIOMS_A, k=0) == []


def test_query_top_k_empty_index():
    idx = BundleSimilarityIndex()
    assert idx.query_top_k(_AXIOMS_A, k=5) == []


# -- jaccard between indexed bundles ---------------------------------


def test_jaccard_between_indexed_bundles_round_trips():
    idx = BundleSimilarityIndex()
    idx.add("a", _AXIOMS_A)
    idx.add("b", _AXIOMS_A[:4])
    j = idx.jaccard("a", "b")
    assert j is not None
    assert 0.0 <= j <= 1.0


def test_jaccard_returns_none_for_missing_id():
    idx = BundleSimilarityIndex()
    idx.add("a", _AXIOMS_A)
    assert idx.jaccard("a", "missing") is None
    assert idx.jaccard("missing", "missing") is None
