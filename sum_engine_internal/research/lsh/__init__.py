"""MinHash-LSH bundle-similarity index.

Substrate-level near-duplicate detection at the bundle layer:
"have I seen a bundle close to this one?" answered in O(1)
via locality-sensitive-hash banding (Indyk & Motwani, STOC 1998;
Broder 1997 "On the resemblance and containment of documents").

Complements the existing per-bundle MMD wire (#4) — MMD answers
"how far is this bundle from the calibration baseline?", LSH
answers "which previous bundle does this one most resemble?".
Together they give the substrate both bundle-vs-baseline
distribution shift AND bundle-vs-bundle dedup.
"""
from sum_engine_internal.research.lsh.bundle_index import (
    BundleSimilarityIndex,
    signature_for_axioms,
)

__all__ = ["BundleSimilarityIndex", "signature_for_axioms"]
