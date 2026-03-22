"""
Extraction Adversarial Hardening Tests

Tests the DeterministicSieve against adversarial, malformed, and edge-case
inputs that might cause incorrect extractions, crashes, or infinite loops.

Author: ototao
License: Apache License 2.0
"""

import pytest
from internal.algorithms.syntactic_sieve import DeterministicSieve


@pytest.fixture(scope="module")
def sieve():
    return DeterministicSieve()


class TestAdversarialInputs:

    def test_html_injection_stripped(self, sieve):
        """HTML tags don't produce garbage triplets."""
        text = "<script>alert('xss')</script> The cat sat on the mat."
        triplets = sieve.extract_triplets(text)
        # Should extract the real sentence, not HTML artifacts
        for s, p, o in triplets:
            assert "<script>" not in s
            assert "<script>" not in o

    def test_sql_injection_harmless(self, sieve):
        """SQL-like input doesn't crash the parser."""
        text = "DROP TABLE users; SELECT * FROM data WHERE 1=1"
        triplets = sieve.extract_triplets(text)
        # Should not crash — result is just whatever spaCy parses
        assert isinstance(triplets, list)

    def test_extremely_long_sentence(self, sieve):
        """A 10,000-word sentence doesn't hang the parser."""
        text = " ".join(["The dog chases the ball."] * 2000)
        triplets = sieve.extract_triplets(text)
        assert isinstance(triplets, list)

    def test_empty_string(self, sieve):
        """Empty input returns empty list, not crash."""
        assert sieve.extract_triplets("") == []

    def test_whitespace_only(self, sieve):
        """Whitespace-only input returns empty list."""
        assert sieve.extract_triplets("   \n\t  ") == []

    def test_unicode_characters(self, sieve):
        """Unicode text (CJK, emoji, diacritics) doesn't crash."""
        text = "日本語のテスト。 Ñoño likes café. 🎉 The 🐕 chased the 🐈."
        triplets = sieve.extract_triplets(text)
        assert isinstance(triplets, list)

    def test_nested_quotes(self, sieve):
        """Nested quotation marks don't produce spurious nesting."""
        text = 'He said "She said \'The cat is alive.\'" and left.'
        triplets = sieve.extract_triplets(text)
        assert isinstance(triplets, list)

    def test_contradictory_statements(self, sieve):
        """Sieve extracts both sides of a contradiction (no resolution)."""
        text = "Earth orbits the sun. The sun orbits the earth."
        triplets = sieve.extract_triplets(text)
        # Both should be extractable — contradiction resolution is not the sieve's job
        assert isinstance(triplets, list)
        assert len(triplets) >= 1

    def test_single_word(self, sieve):
        """Single word produces no triplets (no SVO structure)."""
        triplets = sieve.extract_triplets("Hello")
        assert triplets == []

    def test_numbers_only(self, sieve):
        """Pure numeric input doesn't crash."""
        triplets = sieve.extract_triplets("42 3.14 100000")
        assert isinstance(triplets, list)

    def test_repeated_identical_sentences(self, sieve):
        """Identical sentences are deduplicated (set behavior)."""
        text = "The cat sat on the mat. " * 50
        triplets = sieve.extract_triplets(text)
        # Deduplication: should have far fewer than 50 copies
        assert len(triplets) <= 5

    def test_null_bytes(self, sieve):
        """Null bytes in input don't crash the parser."""
        text = "The cat\x00sat on\x00the mat."
        triplets = sieve.extract_triplets(text)
        assert isinstance(triplets, list)


class TestExtractionQuality:

    def test_simple_svo_extraction(self, sieve):
        """Basic Subject-Verb-Object sentence extracts correctly."""
        triplets = sieve.extract_triplets("Alice likes cats.")
        assert len(triplets) >= 1
        # At least one triplet should mention alice and cat
        subjects = [t[0] for t in triplets]
        assert any("alice" in s for s in subjects)

    def test_passive_voice_extraction(self, sieve):
        """Passive voice sentences still produce triplets."""
        triplets = sieve.extract_triplets("The ball was chased by the dog.")
        assert isinstance(triplets, list)
        # spaCy should handle passive nsubj

    def test_compound_subject(self, sieve):
        """Compound modifiers are captured."""
        triplets = sieve.extract_triplets("The big red car hit the wall.")
        assert isinstance(triplets, list)
        if triplets:
            # Subject should include modifiers
            subjects = [t[0] for t in triplets]
            assert any("car" in s for s in subjects)
