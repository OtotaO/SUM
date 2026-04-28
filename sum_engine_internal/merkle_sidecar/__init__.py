"""Merkle set-commitment sidecar (M1).

Pure-Python prototype for the per-bundle Merkle tree over canonical
fact keys. Companion to the LCM substrate at low-thousands scale;
expected to dominate inclusion-proof verify time above the merge
ceiling. See ``docs/MERKLE_SIDECAR_FORMAT.md`` for the wire spec.

Public surface re-exported here:
    build_tree(canonical_fact_keys) -> MerkleTree
    MerkleTree                       — opaque handle; .root, .size
    MerkleTree.inclusion_proof(key)  -> InclusionProof
    verify_inclusion(proof, root)    -> bool
    InclusionProof                   — dataclass
    LEAF_DOMAIN, NODE_DOMAIN         — domain-separation byte strings
    EMPTY_TREE_ROOT                  — 32-byte all-zeros sentinel

No external dependencies — uses only ``hashlib`` and stdlib types.
"""
from sum_engine_internal.merkle_sidecar.tree import (
    EMPTY_TREE_ROOT,
    InclusionProof,
    LEAF_DOMAIN,
    MerkleTree,
    NODE_DOMAIN,
    build_tree,
    verify_inclusion,
)

__all__ = [
    "EMPTY_TREE_ROOT",
    "InclusionProof",
    "LEAF_DOMAIN",
    "MerkleTree",
    "NODE_DOMAIN",
    "build_tree",
    "verify_inclusion",
]
