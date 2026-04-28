"""Tests for the M1 Merkle set-commitment sidecar prototype.

Three layers of coverage:

  1. **Determinism + set semantics** — same fact set produces the
     same root regardless of input order; duplicates are silently
     deduplicated.
  2. **Round-trip** — every leaf in the tree produces an inclusion
     proof that verifies against the root.
  3. **Tamper detection** — a mutated key, mutated leaf hash,
     mutated sibling hash, or wrong root all reject.

Plus edge cases: empty set, single element, odd-numbered set
sizes (which exercise the promote-unchanged path).
"""
from __future__ import annotations

import hashlib

import pytest

from sum_engine_internal.merkle_sidecar import (
    EMPTY_TREE_ROOT,
    InclusionProof,
    LEAF_DOMAIN,
    MerkleTree,
    NODE_DOMAIN,
    build_tree,
    verify_inclusion,
)


_SAMPLE_KEYS = [
    "alice||graduated||2012",
    "alice||born||1990",
    "bob||owns||dog",
    "diamond||cuts||glass",
    "copper||carries||current",
]


# --------------------------------------------------------------------------
# Determinism + set semantics
# --------------------------------------------------------------------------


def test_same_set_produces_same_root_regardless_of_order():
    forward = build_tree(_SAMPLE_KEYS)
    reverse = build_tree(reversed(_SAMPLE_KEYS))
    shuffled = build_tree(["bob||owns||dog", "alice||graduated||2012",
                           "copper||carries||current", "diamond||cuts||glass",
                           "alice||born||1990"])
    assert forward.root == reverse.root == shuffled.root


def test_duplicates_deduplicated():
    """LCM-style set semantics: adding the same key twice is a
    no-op. Required for cross-substrate consistency with the LCM
    state integer."""
    once = build_tree(_SAMPLE_KEYS)
    twice = build_tree(_SAMPLE_KEYS + _SAMPLE_KEYS)
    assert once.root == twice.root
    assert once.size == twice.size == len(set(_SAMPLE_KEYS))


def test_different_sets_produce_different_roots():
    a = build_tree(_SAMPLE_KEYS)
    b = build_tree(_SAMPLE_KEYS + ["new||fact||appears"])
    assert a.root != b.root


# --------------------------------------------------------------------------
# Round-trip — every leaf verifies against the root
# --------------------------------------------------------------------------


def test_every_leaf_produces_a_verifying_proof():
    tree = build_tree(_SAMPLE_KEYS)
    for key in _SAMPLE_KEYS:
        proof = tree.inclusion_proof(key)
        assert verify_inclusion(key, proof, tree.root), (
            f"proof for key={key!r} did not verify against root"
        )


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7, 8, 15, 16, 100, 1000])
def test_round_trip_at_varied_tree_sizes(size: int):
    """Exercise both even-numbered (full binary tree) and
    odd-numbered (promote-unchanged) tree sizes."""
    keys = [f"k{i:04d}||predicate||value" for i in range(size)]
    tree = build_tree(keys)
    assert tree.size == size

    # Verify every leaf at this size produces a valid proof.
    for key in keys:
        proof = tree.inclusion_proof(key)
        assert verify_inclusion(key, proof, tree.root), (
            f"size={size}, key={key} failed round-trip"
        )


# --------------------------------------------------------------------------
# Tamper detection
# --------------------------------------------------------------------------


def test_wrong_key_rejected():
    """Verifier given a different key than the proof was built for
    rejects (the recomputed leaf hash differs)."""
    tree = build_tree(_SAMPLE_KEYS)
    proof = tree.inclusion_proof("alice||graduated||2012")
    assert not verify_inclusion("alice||born||1990", proof, tree.root)


def test_tampered_leaf_hash_rejected():
    """Mutating the proof's claimed leaf_hash makes the verifier's
    reproduction step diverge — even if the verifier also recomputes
    leaf_hash from the key (which it does, by spec), the proof's
    claimed leaf_hash field is checked against that reproduction
    in step 1 and the mismatch rejects."""
    tree = build_tree(_SAMPLE_KEYS)
    proof = tree.inclusion_proof("alice||graduated||2012")
    tampered = InclusionProof(
        leaf_index=proof.leaf_index,
        leaf_hash=b"\xff" * 32,
        siblings=proof.siblings,
    )
    assert not verify_inclusion("alice||graduated||2012", tampered, tree.root)


def test_tampered_sibling_rejected():
    """Mutating any sibling hash in the proof breaks the chain to
    the root; rejected at the final equality check."""
    tree = build_tree(_SAMPLE_KEYS)
    proof = tree.inclusion_proof("alice||graduated||2012")
    if not proof.siblings:
        pytest.skip("trivial 1-leaf tree has no siblings to tamper")
    # Flip the first sibling hash's first byte.
    pos, h = proof.siblings[0]
    tampered_h = bytes([h[0] ^ 0xFF]) + h[1:]
    tampered_siblings = ((pos, tampered_h),) + proof.siblings[1:]
    tampered = InclusionProof(
        leaf_index=proof.leaf_index,
        leaf_hash=proof.leaf_hash,
        siblings=tampered_siblings,
    )
    assert not verify_inclusion("alice||graduated||2012", tampered, tree.root)


def test_wrong_root_rejected():
    """A valid proof against the wrong root is invalid — the
    verifier reproduces the correct root from the proof, but it
    doesn't match the supplied (wrong) root."""
    tree = build_tree(_SAMPLE_KEYS)
    proof = tree.inclusion_proof("alice||graduated||2012")
    wrong_root = b"\xaa" * 32
    assert not verify_inclusion("alice||graduated||2012", proof, wrong_root)


def test_proof_with_invalid_position_rejected():
    """A proof entry with a malformed `position` value (not 'left'
    or 'right') rejects without raising."""
    tree = build_tree(_SAMPLE_KEYS)
    proof = tree.inclusion_proof("alice||graduated||2012")
    if not proof.siblings:
        pytest.skip("trivial 1-leaf tree has no siblings to corrupt")
    bad_siblings = (("middle", proof.siblings[0][1]),) + proof.siblings[1:]
    bad = InclusionProof(
        leaf_index=proof.leaf_index,
        leaf_hash=proof.leaf_hash,
        siblings=bad_siblings,
    )
    assert not verify_inclusion("alice||graduated||2012", bad, tree.root)


# --------------------------------------------------------------------------
# Empty + single-element edge cases
# --------------------------------------------------------------------------


def test_empty_tree_uses_sentinel_root():
    tree = build_tree([])
    assert tree.size == 0
    assert tree.root == EMPTY_TREE_ROOT
    assert tree.root == b"\x00" * 32


def test_empty_tree_inclusion_proof_raises():
    tree = build_tree([])
    with pytest.raises(KeyError):
        tree.inclusion_proof("any||key||here")


def test_inclusion_against_empty_root_always_fails():
    """No real fact set produces the all-zero root (because real
    leaf hashes always include the LEAF_DOMAIN prefix). Inclusion
    proofs against the empty-tree sentinel MUST reject."""
    tree = build_tree(_SAMPLE_KEYS)
    proof = tree.inclusion_proof("alice||graduated||2012")
    assert not verify_inclusion("alice||graduated||2012", proof, EMPTY_TREE_ROOT)


def test_single_element_tree():
    """N=1: root is just the leaf hash; inclusion proof has no
    siblings (empty list)."""
    tree = build_tree(["solo||fact||here"])
    assert tree.size == 1
    proof = tree.inclusion_proof("solo||fact||here")
    assert proof.siblings == ()
    assert verify_inclusion("solo||fact||here", proof, tree.root)


# --------------------------------------------------------------------------
# Domain separation locked at spec time
# --------------------------------------------------------------------------


def test_leaf_domain_prefix_is_load_bearing():
    """A reader who builds the same tree without the LEAF_DOMAIN
    prefix produces a different root. Pins the spec contract:
    bumping the leaf format requires a new domain prefix (v2 etc.)."""
    keys = ["alice||born||1990"]
    sum_root = build_tree(keys).root
    # Reproduce a "raw" tree without the domain prefix:
    raw_leaf = hashlib.sha256(keys[0].encode("utf-8")).digest()
    assert sum_root != raw_leaf, (
        "leaf domain prefix is not affecting the hash; spec violation"
    )


def test_node_domain_prefix_is_load_bearing():
    """Same as above for the internal-node domain. Without the
    NODE_DOMAIN prefix, an attacker could construct a tree whose
    internal nodes collide with leaf hashes from a different
    namespace."""
    keys = ["a||b||c", "d||e||f"]
    sum_root = build_tree(keys).root
    # Reproduce raw (no NODE_DOMAIN prefix):
    leaf_a = hashlib.sha256(LEAF_DOMAIN + keys[0].encode()).digest()
    leaf_b = hashlib.sha256(LEAF_DOMAIN + keys[1].encode()).digest()
    raw_node = hashlib.sha256(leaf_a + leaf_b).digest()
    assert sum_root != raw_node, (
        "node domain prefix is not affecting the hash; spec violation"
    )


# --------------------------------------------------------------------------
# Wire format (to_dict / from_dict)
# --------------------------------------------------------------------------


def test_inclusion_proof_dict_round_trip():
    tree = build_tree(_SAMPLE_KEYS)
    original = tree.inclusion_proof("diamond||cuts||glass")
    d = original.to_dict()
    assert d["schema"] == "sum.merkle_inclusion.v1"
    rebuilt = InclusionProof.from_dict(d)
    assert rebuilt == original
    assert verify_inclusion("diamond||cuts||glass", rebuilt, tree.root)


def test_inclusion_proof_unknown_schema_rejected():
    with pytest.raises(ValueError, match="schema"):
        InclusionProof.from_dict({
            "schema": "sum.merkle_inclusion.v99",
            "leaf_index": 0,
            "leaf_hash": "ab" * 32,
            "siblings": [],
        })
