"""Unit tests for the MinHash near-duplicate primitive.

Pinned behaviours:

  1. **Identical inputs → Jaccard ≈ 1.0.** Two signatures built from
     the same text must agree at every position.

  2. **Disjoint inputs → Jaccard ≈ 0.0.** No shingle overlap should
     yield a near-zero estimate (within the estimator's std-dev).

  3. **Subset inputs → meaningful intermediate Jaccard.** A doc that
     contains another doc verbatim should land in (0, 1).

  4. **Estimator accuracy.** With num_perm=128, the estimate should
     match the true Jaccard within ~3× std-dev (~0.13 at p=0.5).

  5. **Determinism.** Same input → same signature byte-for-byte.

  6. **Tokenization.** Word 3-shingles, lowercased.

  7. **Edge cases.** Empty text, single-word text (falls back to
     1-shingles), unicode-bearing text.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import random

import pytest

from sum_engine_internal.algorithms.minhash import (
    DEFAULT_NUM_PERM,
    MinHash,
    _word_shingles,
    signature_for_text,
)


# --------------------------------------------------------------------------
# Class invariants
# --------------------------------------------------------------------------


def test_minhash_constructor_rejects_zero_perm():
    with pytest.raises(ValueError, match="must be > 0"):
        MinHash(num_perm=0)
    with pytest.raises(ValueError, match="must be > 0"):
        MinHash(num_perm=-1)


def test_minhash_empty_signature_is_empty():
    sig = MinHash(num_perm=64)
    assert sig.is_empty()


def test_minhash_update_breaks_emptiness():
    sig = MinHash(num_perm=64)
    sig.update(b"hello world")
    assert not sig.is_empty()


def test_minhash_jaccard_size_mismatch_raises():
    a = MinHash(num_perm=64)
    b = MinHash(num_perm=128)
    a.update(b"x")
    b.update(b"x")
    with pytest.raises(ValueError, match="signature size mismatch"):
        a.jaccard(b)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_signature_is_deterministic():
    """Same input → same signature byte-for-byte."""
    text = "Marie Curie won two Nobel Prizes for physics and chemistry."
    s1 = signature_for_text(text)
    s2 = signature_for_text(text)
    assert s1.hashes == s2.hashes


def test_signature_order_invariant_under_set_semantics():
    """MinHash is permutation-invariant: signature only depends on
    the SET of shingles, not their order. Two texts with the same
    shingles in different orders yield identical signatures."""
    sig_words = ["alpha bravo charlie", "bravo charlie delta", "charlie delta echo"]
    a = MinHash()
    b = MinHash()
    a.update_batch(s.encode("utf-8") for s in sig_words)
    b.update_batch(s.encode("utf-8") for s in reversed(sig_words))
    assert a.hashes == b.hashes


# --------------------------------------------------------------------------
# Jaccard correctness
# --------------------------------------------------------------------------


def test_identical_text_jaccard_is_one():
    text = (
        "Marie Curie won two Nobel Prizes. "
        "Albert Einstein proposed the theory of relativity."
    )
    a = signature_for_text(text)
    b = signature_for_text(text)
    assert a.jaccard(b) == 1.0


def test_disjoint_text_jaccard_is_near_zero():
    """Two completely disjoint texts should give Jaccard ≈ 0.
    Allow some slack for the estimator's std-dev at num_perm=128."""
    a = signature_for_text(
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    )
    b = signature_for_text(
        "kilo lima mike november oscar papa quebec romeo sierra tango"
    )
    assert a.jaccard(b) <= 0.05


def test_one_subset_of_other_yields_intermediate_jaccard():
    """One text that contains another verbatim should land in (0, 1).
    The smaller text's shingles all appear in the larger; the larger
    has additional shingles. Jaccard = |smaller| / |larger|."""
    small = signature_for_text(
        "alpha bravo charlie delta echo foxtrot golf hotel"
    )
    big = signature_for_text(
        "alpha bravo charlie delta echo foxtrot golf hotel "
        "india juliet kilo lima mike november oscar papa"
    )
    j = small.jaccard(big)
    # Truth: |small_shingles| = 6, |big_shingles| = 14, intersection = 6.
    # Jaccard = 6 / 14 ≈ 0.43. Allow 3-sigma slack.
    assert 0.30 <= j <= 0.55, f"Jaccard estimate {j:.3f} out of expected band"


def test_estimator_accuracy_at_known_jaccard():
    """Construct sets with known Jaccard and verify the estimator
    matches within 3× std-dev. ~99.7% confidence."""
    rng = random.Random(20260429)
    # Two sets with target Jaccard ≈ 0.5: 100 shared, 100 each unique.
    shared = [f"shared_{i}".encode("utf-8") for i in range(100)]
    only_a = [f"only_a_{i}".encode("utf-8") for i in range(100)]
    only_b = [f"only_b_{i}".encode("utf-8") for i in range(100)]
    a = MinHash()
    a.update_batch(shared + only_a)
    b = MinHash()
    b.update_batch(shared + only_b)
    j = a.jaccard(b)
    # True Jaccard = 100 / 300 = 0.333. std-dev = sqrt(p(1-p)/128) ≈ 0.042.
    # 3-sigma window: ~0.21–0.46.
    assert 0.20 <= j <= 0.47, f"Jaccard estimate {j:.3f} out of expected band"


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------


def test_word_shingles_basic():
    shingles = _word_shingles("the quick brown fox jumps", k=3)
    # Lowercased; 3-grams: 'the quick brown', 'quick brown fox', 'brown fox jumps'.
    assert shingles == [
        b"the quick brown",
        b"quick brown fox",
        b"brown fox jumps",
    ]


def test_word_shingles_short_text_falls_back_to_single_words():
    """If text has fewer than k words, fall back to k=1 so the
    signature is still non-empty."""
    shingles = _word_shingles("hi there", k=3)
    assert shingles == [b"hi", b"there"]


def test_word_shingles_empty_text():
    assert _word_shingles("", k=3) == []
    assert _word_shingles("   ", k=3) == []


def test_word_shingles_invalid_k():
    with pytest.raises(ValueError, match="must be > 0"):
        _word_shingles("anything", k=0)


def test_word_shingles_lowercases():
    """Lowercasing makes the signature case-insensitive — sensible
    default for near-duplicate detection."""
    a = _word_shingles("The Quick Brown Fox Jumps", k=3)
    b = _word_shingles("the quick brown fox jumps", k=3)
    assert a == b


# --------------------------------------------------------------------------
# End-to-end on real-shape inputs
# --------------------------------------------------------------------------


def test_signature_short_text_does_not_crash():
    sig = signature_for_text("hi")
    assert not sig.is_empty()


def test_signature_unicode_text():
    """Non-ASCII text works through the UTF-8 encode path."""
    sig = signature_for_text("résumé café 日本語 emoji 🚀")
    assert not sig.is_empty()


def test_signature_empty_text_is_empty_signature():
    sig = signature_for_text("")
    assert sig.is_empty()


def test_near_duplicate_with_minor_edits_has_high_jaccard():
    """A doc with one extra sentence appended should have Jaccard
    near 1.0 against the original — they share most shingles."""
    base = (
        "Marie Curie won two Nobel Prizes. "
        "Albert Einstein proposed the theory of relativity. "
        "Shakespeare wrote Hamlet. "
        "Water contains hydrogen. "
        "Dolphins are mammals. "
        "Beethoven composed symphonies. "
        "Newton discovered gravity. "
        "Galileo observed the moons of Jupiter."
    )
    edited = base + " The Mariana Trench is the deepest part of the ocean."
    a = signature_for_text(base)
    b = signature_for_text(edited)
    j = a.jaccard(b)
    # True Jaccard for this fixture is 0.76 (32 base shingles + 10 new
    # = 42 in edited, intersection = 32, union = 42; 32/42 ≈ 0.76).
    # Estimator at num_perm=128 has std-dev ~0.04 at p=0.76. Allow
    # 3-sigma slack on the lower end.
    assert j >= 0.64, f"near-duplicate Jaccard {j:.3f} unexpectedly low"
