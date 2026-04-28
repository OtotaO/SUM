# Merkle set-commitment sidecar (M1)

**Status:** prototype-first. Spec + Python implementation + benchmark land together; production wiring (sidecar carried alongside CanonicalBundle / render receipt) is a follow-on once the architectural decision to ship is made.

A Merkle set commitment over canonical fact keys gives **log-size membership proofs** alongside the existing LCM state integer. The two sit side by side: LCM stays the algebra for idempotent set union and divisibility-based entailment; Merkle is the external-membership surface for verifiers that need fast inclusion proofs at scales where the LCM substrate's `O(n²)` merge cost dominates (`PROOF_BOUNDARY.md` §2.2 documents merge `p50 ≈ 519 ms` at `N=1000`, extrapolating to ~50 s/op at `N=10,000`).

This is **not** the existing Akashic Ledger Merkle hash-chain ([`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.7). That hash-chain links *events* (`prev_hash` over operation/prime/axiom_key/branch). M1 is a different artifact: a static set commitment over canonical fact keys at bundle issuance, providing membership-witness compactness for a single bundle's fact set. The two surfaces share a hash function and nothing else.

## Wire format

A Merkle set commitment is two things:

1. **The root hash** — a single 32-byte (64-hex-char) value that commits to the entire set. Carried as an additional signed field in the bundle.
2. **Per-element inclusion proofs** — a list of sibling hashes that let a verifier reproduce the root from a single element. Computed on demand; not stored in the bundle.

### The root hash field

A bundle carrying a Merkle sidecar adds one optional payload field:

```json
{
  ...
  "merkle_root": "sha256-<hex>"
}
```

The `merkle_root` is the result of building a Merkle tree over the bundle's canonical fact keys (see §"Tree construction" below). When present, it's part of the signed-over payload like every other field — tampering with it invalidates the signature.

### Inclusion proof shape

```json
{
  "schema": "sum.merkle_inclusion.v1",
  "leaf_index": 7,
  "leaf_hash": "<hex>",
  "siblings": [
    {"position": "right", "hash": "<hex>"},
    {"position": "left",  "hash": "<hex>"},
    ...
  ]
}
```

| Field | Meaning |
|---|---|
| `leaf_index` | 0-based index of the element in the lex-sorted leaf array. |
| `leaf_hash` | SHA-256 of the canonical fact key with the leaf domain prefix (see §"Hash functions"). The verifier recomputes this from the canonical key and asserts it matches. |
| `siblings` | The chain of sibling hashes from the leaf up to the root. Each entry says whether to combine `(sibling, current)` or `(current, sibling)`. Length is approximately `log2(N)`. |

A verifier given a fact key + an inclusion proof + the root hash:

1. Recomputes `leaf_hash = sha256(LEAF_DOMAIN || canonical_fact_key_bytes)` and asserts it matches the proof's `leaf_hash`.
2. Walks the `siblings` array bottom-up: at each step, computes `current = sha256(NODE_DOMAIN || left_hash || right_hash)` where `(left, right)` is `(sibling, current)` if position is `"left"` else `(current, sibling)`.
3. After all siblings are processed, asserts `current == root_hash`.

If any step fails, the proof is invalid — the leaf is not a member of the set the root commits to.

## Hash functions

**Domain separation locked at spec time.** Without explicit prefixes, the same hash function namespace is shared with the Akashic Ledger hash-chain and any future hash-using surface, which invites collision-class confusion. The two byte strings:

```
LEAF_DOMAIN = b"SUM-MERKLE-FACT-LEAF-v1\0"
NODE_DOMAIN = b"SUM-MERKLE-FACT-NODE-v1\0"
```

The trailing `\0` byte is intentional — it ensures the hash input has an unambiguous boundary between the domain prefix and the rest, preventing length-extension-style ambiguity.

Concrete byte-level operations:

```
canonical_fact_key_bytes = utf8(canonical_fact_key)   # e.g. b"alice||graduated||2012"
leaf_hash = sha256(LEAF_DOMAIN || canonical_fact_key_bytes)
node_hash = sha256(NODE_DOMAIN || left_hash_32bytes || right_hash_32bytes)
```

**The `v1` in the domain prefix is load-bearing.** Bumping the leaf or node format requires a new domain prefix (`SUM-MERKLE-FACT-LEAF-v2\0`, etc.) so v1 verifiers can't confuse a v2 commitment for v1 — they fail to find a leaf in the v2 tree because their hashing produces different bytes.

## Tree construction

Given `N` canonical fact keys:

1. **Lex-sort** the keys. The sort ensures the same fact set always produces the same root regardless of insertion order; without lex-sort, two clients with different insertion sequences would produce different roots for the same set.
2. **Hash each key** to a leaf with `leaf_hash = sha256(LEAF_DOMAIN || utf8(key))`.
3. **Build the tree bottom-up.** At each level, pair adjacent hashes: `node_hash = sha256(NODE_DOMAIN || left || right)`. If a level has an odd number of hashes, the last one is **promoted unchanged** to the next level (RFC 6962 / RFC 9162 convention; preserves the "no trailing duplication" property that prevents subtle proof-construction bugs).
4. **Repeat** until one hash remains — that's the root.

### Empty-set root

The root for an empty set is the all-zero 32-byte hash. This is a sentinel — no real fact set produces a 32-byte zero hash because the leaf hashing always has the domain prefix as input. Verifiers presented with the all-zero root MUST treat the commitment as "empty set" and reject any inclusion proof against it.

### Single-element edge case

For `N=1`, the root is just the leaf hash (no internal nodes needed). The inclusion proof is empty (no siblings). This matches the bottom-up construction's natural termination.

## Why a sidecar, not a replacement

Replacing LCM with Merkle would give up the algebra LCM provides locally:

- **Idempotent set union** (LCM is commutative + associative + `lcm(a, a) == a`). Merkle has no analogous algebra — combining two sets requires re-sorting + re-hashing.
- **Divisibility-based entailment** (`state % prime == 0` ⟺ the fact is in the set). Merkle membership requires the explicit inclusion proof.
- **Compositional reasoning across bundles** (LCM of two state integers = the union set's state). Merkle requires re-building over the union set.

LCM is the right substrate for compositional reasoning at low-thousands scale. Merkle is the right substrate for **external membership verification** at scales where LCM's bit-length-quadratic merge cost dominates. The sidecar architecture lets each substrate do what it's best at.

A bundle carrying both:

```json
{
  ...
  "state_integer": "<decimal/hex>",   // LCM substrate (compositional)
  "merkle_root": "sha256-<hex>"       // Merkle substrate (external membership)
}
```

A verifier needing fast membership uses the Merkle path. A verifier doing LCM-style composition (delta bundles, bundle merges) uses the state-integer path. Both paths are signed; mutating either invalidates the signature.

## Implementation gating

**This PR lands the spec + Python prototype + benchmark.** Production wiring — actually emitting `merkle_root` in CanonicalBundle / render receipts — is a follow-on once:

1. Benchmark numbers ([`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §2.2) confirm Merkle inclusion-proof verify is **materially faster** than LCM divisibility check at `N=10,000`.
2. The leaf format is reviewed and locked (this spec is the proposal; review can refine before production).
3. The CanonicalBundle schema is extended to carry `merkle_root` as an optional signed field, with a new `bundle_version` minor bump (`1.0.0` → `1.1.0`) per [`docs/COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md).

Until those three are in place, the Python prototype is exercised only by its tests + the standalone benchmark — useful for measurement and review, not yet part of the live trust surface.

## Cross-references

- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.7 — the Akashic Ledger Merkle hash-chain (different surface, shared hash function only).
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §2.2 — the merge ceiling that motivates this sidecar.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) M1 — playbook entry that scoped this design.
- [`docs/COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) — the `bundle_version` bump policy that future production wiring follows.
- RFC 9162 (Certificate Transparency v2.0) — the "RFC 9162-inspired" framing in the playbook entry. M1 borrows the leaf/node hash structure but does NOT inherit RFC 9162's transparency-log lifecycle (signed tree heads, append-only consistency proofs); those are out of scope for the static-set-commitment shape.
