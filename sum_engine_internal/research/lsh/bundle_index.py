"""BundleSimilarityIndex — MinHash-LSH for bundle axiom sets.

The bundle's active-axiom list (canonical "s||p||o" strings) is a
SET. Jaccard similarity ``|A ∩ B| / |A ∪ B|`` is a natural
bundle-vs-bundle similarity. MinHash gives an unbiased estimator
in O(num_perm) per comparison (Broder 1997, Compression and
Complexity of Sequences). LSH banding gives O(1) candidate lookup
across an N-bundle index by hashing each band's row-tuple into a
bucket — two signatures are LSH-candidates iff at least one band
matches exactly (Indyk-Motwani, STOC 1998).

The S-curve P(candidate | jaccard=j) = 1 - (1 - j^r)^b — for
``num_perm=128, bands=16, rows=8`` the threshold is ≈ ``(1/16)^(1/8)
≈ 0.71``, which matches the substrate's "near-duplicate" intuition
without flooding the candidate set with weak matches.

Reuses the keyed-blake2b MinHash from
``sum_engine_internal.algorithms.minhash``; this module only adds
the bundle-shaped tokeniser (one shingle per canonical axiom) and
the LSH banding bucket-table.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from sum_engine_internal.algorithms.minhash import (
    DEFAULT_NUM_PERM,
    MinHash,
)


def signature_for_axioms(
    axioms: Iterable[str],
    *,
    num_perm: int = DEFAULT_NUM_PERM,
) -> MinHash:
    """Build a MinHash signature over a bundle's canonical axiom set.

    Each axiom string ``"s||p||o"`` is treated as a single shingle
    (axioms are already canonical units; sub-tokenizing further
    would mean two bundles with identical axioms but different
    surface phrasing would no-longer match — and bundles are
    already canonicalised by the upstream sieve before reaching
    the codec).

    Empty inputs return an empty signature (``is_empty() == True``).
    """
    sig = MinHash(num_perm=num_perm)
    for axiom in axioms:
        sig.update(axiom.encode("utf-8"))
    return sig


class BundleSimilarityIndex:
    """In-memory MinHash-LSH index over bundle axiom sets.

    Build:
        idx = BundleSimilarityIndex(num_perm=128, bands=16, rows=8)
        idx.add(bundle_id="b1", axioms=["alice||likes||cats", ...])
        idx.add(bundle_id="b2", axioms=[...])

    Query:
        ids = idx.query_candidates(axioms=[...])
        # set[str] of bundle_ids whose signature shares ≥1 band.
        top = idx.query_top_k(axioms=[...], k=5, min_jaccard=0.5)
        # list[(bundle_id, jaccard_estimate)] sorted desc.

    The ``bands * rows == num_perm`` invariant is enforced at
    construction. Defaults: 128 / 16 / 8 → S-curve threshold ≈ 0.71.
    """

    def __init__(
        self,
        *,
        num_perm: int = DEFAULT_NUM_PERM,
        bands: int = 16,
        rows: int = 8,
    ) -> None:
        if bands <= 0 or rows <= 0:
            raise ValueError(
                f"bands and rows must be > 0, got bands={bands}, rows={rows}"
            )
        if bands * rows != num_perm:
            raise ValueError(
                f"bands * rows must equal num_perm "
                f"(got {bands} * {rows} = {bands * rows} != {num_perm})"
            )
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self._sigs: dict[str, MinHash] = {}
        self._buckets: list[dict[tuple[int, ...], set[str]]] = [
            defaultdict(set) for _ in range(bands)
        ]

    def __len__(self) -> int:
        return len(self._sigs)

    def __contains__(self, bundle_id: str) -> bool:
        return bundle_id in self._sigs

    def add(self, bundle_id: str, axioms: Iterable[str]) -> None:
        """Index a bundle. Re-adding the same id replaces the prior
        signature and rewires its bucket entries (so updates are
        consistent)."""
        if bundle_id in self._sigs:
            self._remove_from_buckets(bundle_id, self._sigs[bundle_id])
        sig = signature_for_axioms(axioms, num_perm=self.num_perm)
        self._sigs[bundle_id] = sig
        for band_idx, band_key in enumerate(self._band_keys(sig)):
            self._buckets[band_idx][band_key].add(bundle_id)

    def remove(self, bundle_id: str) -> None:
        """Remove a bundle from the index. No-op if not present."""
        sig = self._sigs.pop(bundle_id, None)
        if sig is not None:
            self._remove_from_buckets(bundle_id, sig)

    def query_candidates(self, axioms: Iterable[str]) -> set[str]:
        """Return the set of bundle_ids whose signature shares at
        least one band with the query. O(bands) hash lookups, then
        O(candidates) set union — independent of total index size
        in the typical case."""
        sig = signature_for_axioms(axioms, num_perm=self.num_perm)
        candidates: set[str] = set()
        for band_idx, band_key in enumerate(self._band_keys(sig)):
            candidates |= self._buckets[band_idx].get(band_key, set())
        return candidates

    def query_top_k(
        self,
        axioms: Iterable[str],
        *,
        k: int = 5,
        min_jaccard: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` (bundle_id, jaccard_estimate) tuples
        sorted by jaccard descending. Jaccard is computed exactly
        on the candidate set returned by LSH banding — sound for
        candidates, but bundles below the LSH threshold may be
        missed (that's the speed/recall tradeoff the substrate is
        accepting by using LSH at all)."""
        if k <= 0:
            return []
        sig = signature_for_axioms(axioms, num_perm=self.num_perm)
        candidates = self.query_candidates(axioms)
        scored = [
            (cid, sig.jaccard(self._sigs[cid]))
            for cid in candidates
        ]
        scored = [(c, j) for c, j in scored if j >= min_jaccard]
        scored.sort(key=lambda cj: cj[1], reverse=True)
        return scored[:k]

    def jaccard(self, bundle_id_a: str, bundle_id_b: str) -> Optional[float]:
        """Estimated Jaccard between two indexed bundles. Returns
        ``None`` if either id is missing."""
        sa = self._sigs.get(bundle_id_a)
        sb = self._sigs.get(bundle_id_b)
        if sa is None or sb is None:
            return None
        return sa.jaccard(sb)

    def _band_keys(self, sig: MinHash) -> list[tuple[int, ...]]:
        return [
            tuple(sig.hashes[b * self.rows:(b + 1) * self.rows])
            for b in range(self.bands)
        ]

    def _remove_from_buckets(self, bundle_id: str, sig: MinHash) -> None:
        for band_idx, band_key in enumerate(self._band_keys(sig)):
            bucket = self._buckets[band_idx]
            members = bucket.get(band_key)
            if members is None:
                continue
            members.discard(bundle_id)
            if not members:
                bucket.pop(band_key, None)
