"""Pure-Python Merkle set-commitment over canonical fact keys.

Implements the M1 prototype per ``docs/MERKLE_SIDECAR_FORMAT.md``:

  * Domain-separated leaf + node hashes (``SUM-MERKLE-FACT-LEAF-v1\0``,
    ``SUM-MERKLE-FACT-NODE-v1\0``) so the same SHA-256 namespace
    isn't shared with the Akashic Ledger hash-chain or any future
    hash-using surface.
  * Lex-sorted leaves so the same fact set always produces the same
    root regardless of insertion order.
  * Odd-level promote-unchanged construction (RFC 6962 / RFC 9162
    convention) so trailing-duplication bugs don't surface in
    inclusion-proof construction.
  * Empty-set sentinel (32-byte all-zeros root) distinct from any
    real-set root because real leaf hashes always include the
    domain prefix.

No external dependencies. Uses only ``hashlib`` from the stdlib.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence


LEAF_DOMAIN: bytes = b"SUM-MERKLE-FACT-LEAF-v1\0"
NODE_DOMAIN: bytes = b"SUM-MERKLE-FACT-NODE-v1\0"

EMPTY_TREE_ROOT: bytes = b"\x00" * 32


@dataclass(frozen=True)
class InclusionProof:
    """RFC 9162-style inclusion proof. Walked bottom-up by the
    verifier: at each step, the current hash is combined with the
    sibling using the indicated position to produce the parent.
    After ``len(siblings)`` steps, the result equals the root the
    proof is against."""

    leaf_index: int
    leaf_hash: bytes
    # Each sibling is (position, hash). position is "left" or
    # "right" indicating whether the sibling is on the LEFT of the
    # current node (so combine = sibling + current) or on the
    # RIGHT (so combine = current + sibling).
    siblings: tuple[tuple[str, bytes], ...]

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict matching the wire
        format in MERKLE_SIDECAR_FORMAT.md §"Inclusion proof shape"."""
        return {
            "schema": "sum.merkle_inclusion.v1",
            "leaf_index": self.leaf_index,
            "leaf_hash": self.leaf_hash.hex(),
            "siblings": [
                {"position": pos, "hash": h.hex()}
                for pos, h in self.siblings
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InclusionProof":
        if d.get("schema") != "sum.merkle_inclusion.v1":
            raise ValueError(
                f"unsupported inclusion-proof schema: {d.get('schema')!r}"
            )
        return cls(
            leaf_index=int(d["leaf_index"]),
            leaf_hash=bytes.fromhex(d["leaf_hash"]),
            siblings=tuple(
                (s["position"], bytes.fromhex(s["hash"]))
                for s in d["siblings"]
            ),
        )


def _hash_leaf(canonical_fact_key: str) -> bytes:
    """leaf_hash = sha256(LEAF_DOMAIN || utf8(key))."""
    h = hashlib.sha256()
    h.update(LEAF_DOMAIN)
    h.update(canonical_fact_key.encode("utf-8"))
    return h.digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    """node_hash = sha256(NODE_DOMAIN || left || right)."""
    if len(left) != 32 or len(right) != 32:
        raise ValueError(
            f"node hashes must be 32 bytes; got left={len(left)}, "
            f"right={len(right)}"
        )
    h = hashlib.sha256()
    h.update(NODE_DOMAIN)
    h.update(left)
    h.update(right)
    return h.digest()


@dataclass(frozen=True)
class MerkleTree:
    """Opaque handle around a fully-built Merkle tree.

    Construction sorts the leaves by canonical fact key (so the
    same fact set always produces the same root). Internal node
    layers are stored bottom-up so inclusion-proof construction
    is a single linear walk per leaf.
    """

    # Lex-sorted canonical fact keys, one per leaf, in tree-leaf
    # order (the index in this list is the leaf_index in any
    # inclusion proof).
    leaves: tuple[str, ...]
    # The leaf hashes (level 0). Same length and order as `leaves`.
    leaf_hashes: tuple[bytes, ...]
    # Internal node layers, bottom-up. layers[0] is leaf_hashes;
    # layers[1] is the first internal layer; layers[-1] is a
    # one-element list containing the root.
    layers: tuple[tuple[bytes, ...], ...] = field(repr=False)

    @property
    def root(self) -> bytes:
        if not self.leaves:
            return EMPTY_TREE_ROOT
        return self.layers[-1][0]

    @property
    def size(self) -> int:
        return len(self.leaves)

    def inclusion_proof(self, canonical_fact_key: str) -> InclusionProof:
        """Build an inclusion proof for ``canonical_fact_key``.

        Raises KeyError if the key is not in the tree.
        """
        if not self.leaves:
            raise KeyError(
                f"cannot build inclusion proof: tree is empty"
            )
        try:
            leaf_index = self.leaves.index(canonical_fact_key)
        except ValueError:
            raise KeyError(
                f"key {canonical_fact_key!r} not in tree (size={self.size})"
            ) from None

        leaf_hash = self.leaf_hashes[leaf_index]
        siblings: list[tuple[str, bytes]] = []
        idx = leaf_index
        for level in range(len(self.layers) - 1):
            layer = self.layers[level]
            # If idx is even, sibling is at idx+1 on the right.
            # If idx is odd, sibling is at idx-1 on the left.
            # If idx is even and at the end of an odd-length layer,
            # this leaf was promoted unchanged — no sibling at this
            # level.
            if idx % 2 == 0:
                if idx + 1 < len(layer):
                    siblings.append(("right", layer[idx + 1]))
                # else: promoted unchanged; no sibling.
            else:
                siblings.append(("left", layer[idx - 1]))
            idx //= 2

        return InclusionProof(
            leaf_index=leaf_index,
            leaf_hash=leaf_hash,
            siblings=tuple(siblings),
        )


def build_tree(canonical_fact_keys: Iterable[str]) -> MerkleTree:
    """Build a Merkle tree from a set of canonical fact keys.

    Lex-sorts the keys before hashing so the same fact set produces
    the same root regardless of input iteration order. Duplicates
    are silently deduplicated (set-shaped semantics, like LCM —
    `lcm(a, a) == a` ⟺ adding the same key twice is a no-op).
    """
    # Set semantics + lex-sort. Matches the LCM substrate's
    # idempotence + matches what TS canonicalize would produce on
    # the same set if/when a JS-side prototype lands.
    sorted_keys = tuple(sorted(set(canonical_fact_keys)))
    if not sorted_keys:
        # Empty tree: layers is a single empty layer; root is the
        # all-zero sentinel from MerkleTree.root.
        return MerkleTree(leaves=(), leaf_hashes=(), layers=((),))

    # Layer 0: leaf hashes.
    leaf_hashes = tuple(_hash_leaf(k) for k in sorted_keys)
    layers: list[tuple[bytes, ...]] = [leaf_hashes]

    # Build internal layers bottom-up until one node remains.
    current = list(leaf_hashes)
    while len(current) > 1:
        next_layer: list[bytes] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                next_layer.append(_hash_node(current[i], current[i + 1]))
            else:
                # Odd element at the end: promote unchanged.
                next_layer.append(current[i])
        layers.append(tuple(next_layer))
        current = next_layer

    return MerkleTree(
        leaves=sorted_keys,
        leaf_hashes=leaf_hashes,
        layers=tuple(layers),
    )


def verify_inclusion(
    canonical_fact_key: str,
    proof: InclusionProof,
    root: bytes,
) -> bool:
    """Verify that ``canonical_fact_key`` is a member of the set
    committed to by ``root``, using the given inclusion proof.

    Returns True on a valid proof, False on any mismatch (wrong
    key, tampered proof, wrong root). Does not raise on bad input
    — the boolean is the contract; raising would conflate
    "verifier said no" with "verifier crashed".
    """
    # An inclusion proof against the empty-tree sentinel is always
    # invalid; explicit reject.
    if root == EMPTY_TREE_ROOT:
        return False

    # Step 1: recompute the leaf hash from the canonical key.
    expected_leaf = _hash_leaf(canonical_fact_key)
    if expected_leaf != proof.leaf_hash:
        return False

    # Step 2: walk the siblings bottom-up, combining at each step.
    current = expected_leaf
    for position, sibling in proof.siblings:
        if position == "left":
            current = _hash_node(sibling, current)
        elif position == "right":
            current = _hash_node(current, sibling)
        else:
            return False  # malformed proof — bad position

    # Step 3: the final hash must equal the claimed root.
    return current == root
