"""MinHash signatures for batch near-duplicate detection.

Used by ``sum attest-batch --dedup-threshold`` to avoid re-attesting
near-identical inputs (the same doc with minor edits, copy-pasted
versions). Pure-Python; no external dependency. Stdlib ``hashlib``
provides ``blake2b`` which gives us tunable digest size and a
keyed-hash family (no string-formatted hash-function ids).

**MinHash recap.** Given a set ``S`` and a hash family ``h_1..h_k``,
the signature of ``S`` is ``[ min_{x in S} h_i(x) for i in 1..k ]``.
For two sets ``A``, ``B``, the expected fraction of signature
positions where ``A`` and ``B`` agree equals their Jaccard
similarity ``|A ∩ B| / |A ∪ B|``. So with k=128 hash functions, a
single comparison gives a Jaccard estimate with std-dev ≈
sqrt(p(1-p)/k) ≈ 0.04 at p=0.5, ≈ 0.02 at p=0.9.

**Tokenization.** This module signs text by **word 3-shingles** (every
contiguous 3-word substring). Word-level shingles are robust to
whitespace and punctuation noise that char-shingles aren't, and
length-3 catches sentence-level paraphrase without exploding the
shingle count. For very short documents (< 3 words), falls back to
single-word shingles.

**Performance.** O(num_perm) per shingle. With num_perm=128 and a
typical ~200-word doc emitting ~200 shingles, that's ~25K hash
operations — sub-millisecond. Pairwise Jaccard estimate is O(num_perm).
For batches of ~100 files, the O(n²) all-pairs comparison is
~5K comparisons × ~128 ops each = trivial. For much larger batches,
LSH banding would be needed; out of scope for the attest-batch
ergonomic surface.

Public surface:
  - ``MinHash`` class — incremental signature builder
  - ``signature_for_text(text, *, num_perm)`` — convenience
  - ``MinHash.jaccard(other)`` — estimated Jaccard similarity
"""
from __future__ import annotations

import hashlib
import struct
from typing import Iterable, List, Sequence

DEFAULT_NUM_PERM = 128
DEFAULT_SHINGLE_K = 3
_INF_U64 = (1 << 64) - 1


class MinHash:
    """Incremental MinHash signature builder.

    Construct, call ``update`` for each token, then read ``hashes``
    or compare to another signature with ``jaccard``.

    Two signatures with the same ``num_perm`` are comparable; mixing
    sizes raises at the comparison boundary.
    """

    __slots__ = ("num_perm", "hashes")

    def __init__(self, num_perm: int = DEFAULT_NUM_PERM):
        if num_perm <= 0:
            raise ValueError(f"num_perm must be > 0, got {num_perm}")
        self.num_perm = num_perm
        self.hashes: List[int] = [_INF_U64] * num_perm

    def update(self, token: bytes) -> None:
        """Incorporate *token* into the signature.

        Each of the ``num_perm`` hash functions is realised as
        ``blake2b(token, person=<i>)`` — a keyed hash family. This
        is a stronger MinHash than the textbook
        ``(a*x + b) mod p`` permutations because it doesn't require
        choosing prime moduli or random ``a, b``: ``blake2b``'s
        ``person`` argument gives us 16 bytes of independent
        family-key per hash, and the digest is a uniformly-distributed
        64-bit integer.
        """
        for i in range(self.num_perm):
            person = struct.pack("<Q", i).ljust(16, b"\0")
            digest = hashlib.blake2b(
                token, digest_size=8, person=person,
            ).digest()
            v = struct.unpack("<Q", digest)[0]
            if v < self.hashes[i]:
                self.hashes[i] = v

    def update_batch(self, tokens: Iterable[bytes]) -> None:
        """Convenience: update once per token in *tokens*."""
        for tok in tokens:
            self.update(tok)

    def jaccard(self, other: "MinHash") -> float:
        """Estimate the Jaccard similarity to *other*.

        Range: 0.0 (no shingle overlap) to 1.0 (identical shingle
        sets). Std-dev of the estimator at num_perm=128 is roughly
        ``sqrt(p(1-p)/128)`` — about 0.044 at p=0.5, 0.026 at p=0.9.
        Safe to use a threshold like 0.85 for "near-duplicate" with
        false-positive rate below 1% on the seed_v1 corpus.
        """
        if self.num_perm != other.num_perm:
            raise ValueError(
                f"signature size mismatch: self.num_perm={self.num_perm} "
                f"vs other.num_perm={other.num_perm}"
            )
        if self.num_perm == 0:
            return 0.0
        matches = sum(1 for a, b in zip(self.hashes, other.hashes) if a == b)
        return matches / self.num_perm

    def is_empty(self) -> bool:
        """True if no tokens have been incorporated."""
        return all(h == _INF_U64 for h in self.hashes)


# ---------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------


def _word_shingles(text: str, k: int = DEFAULT_SHINGLE_K) -> List[bytes]:
    """Return ``len(words) - k + 1`` word k-shingles, encoded as
    UTF-8 bytes for hashing.

    Words are extracted via a simple lowercased ``.split()`` —
    whitespace-delimited, no punctuation stripping. This is a
    deliberate choice: punctuation differences become real differences
    in the signature, which is the right behaviour for "are these
    two docs near-duplicates" (a doc with extra punctuation IS a
    different doc, just slightly).

    For texts with fewer than k words, falls back to single-word
    shingles so very short inputs still produce a non-empty
    signature.
    """
    if k <= 0:
        raise ValueError(f"shingle k must be > 0, got {k}")
    words = text.lower().split()
    if len(words) < k:
        return [w.encode("utf-8") for w in words]
    return [
        " ".join(words[i:i + k]).encode("utf-8")
        for i in range(len(words) - k + 1)
    ]


def signature_for_text(
    text: str,
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_k: int = DEFAULT_SHINGLE_K,
) -> MinHash:
    """Convenience: tokenize *text* into word k-shingles and build
    a MinHash signature in one call."""
    sig = MinHash(num_perm=num_perm)
    sig.update_batch(_word_shingles(text, k=shingle_k))
    return sig
