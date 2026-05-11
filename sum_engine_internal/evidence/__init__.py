"""Evidence-chain layer over CanonicalBundle.

Where the wires arc (#1-#6) exposes scalar metrics ABOUT a bundle
on its metadata, this layer exposes structural CLAIMS — the
explicit chain from source text → extracted axioms → signed
bundle. Combined with the bundle's HMAC/Ed25519 signature, gives
a reproducible end-to-end provenance: any third party can
re-fetch ``source_uri``, slice ``[byte_start:byte_end]``, confirm
the excerpt matches, and verify the bundle signature still
holds.

Aligns with mack/Radiant's evidence-chain pattern — every claim
in the chain has explicit support, and the chain is checkable
without trusting the producer.

The module is intentionally minimal at v0:
  - ``EvidenceLink`` — one (axiom_claim, provenance_record) pair.
  - ``verify_chain_well_formed`` — no malformed records, no
    duplicate (claim, byte_range) pairs.
  - ``verify_chain_covers_axioms`` — every axiom in the bundle
    has at least one link.
  - ``compose_bundle_with_evidence`` — convenience builder that
    runs the sieve, encodes state, exports the bundle, attaches
    the chain, and verifies coverage before returning.

Forward-compatible: future chain links can carry ``derived_from``
+ ``derivation_rule`` for non-leaf evidence (claim implied by
others under a rule). Lean-4 entailment certificates plug in at
that layer.
"""
from sum_engine_internal.evidence.chain import (
    EvidenceLink,
    EvidenceChainError,
    compose_bundle_with_evidence,
    verify_chain_well_formed,
    verify_chain_covers_axioms,
)
from sum_engine_internal.evidence.rules import (
    InferenceRule,
    TransitiveClosureRule,
    derive_non_leaf_links,
)

__all__ = [
    "EvidenceLink",
    "EvidenceChainError",
    "compose_bundle_with_evidence",
    "verify_chain_well_formed",
    "verify_chain_covers_axioms",
    "InferenceRule",
    "TransitiveClosureRule",
    "derive_non_leaf_links",
]
