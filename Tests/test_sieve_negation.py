"""Tests for DeterministicSieve's negation handling.

A negated sentence like "Diamonds cannot cut through steel." asserts the
INVERSE of its bare (subject, predicate, object) form. Before the negation
guard, the sieve would silently emit ``(diamond, cut, steel)`` — a
positively-encoded asserted triple that contradicts the source sentence —
into the Gödel state with no surface marker that polarity was flipped.

This is a truth-layer bug, strictly worse than a miss. These tests pin the
fix: any sentence with a spaCy ``dep_ == "neg"`` child anywhere in the
sentence must produce NO triple under either extraction path.
"""
from __future__ import annotations

from internal.algorithms.syntactic_sieve import DeterministicSieve


def _sieve() -> DeterministicSieve:
    return DeterministicSieve()  # type: ignore[no-untyped-call]


class TestNegationSurfaceForms:
    """Each common negation surface form must produce an empty extraction."""

    def test_cannot_compound(self) -> None:
        assert _sieve().extract_triplets(
            "Diamonds cannot cut through steel."
        ) == []

    def test_does_not(self) -> None:
        assert _sieve().extract_triplets(
            "Water does not freeze above zero degrees."
        ) == []

    def test_is_not_copula(self) -> None:
        assert _sieve().extract_triplets("The cat is not a dog.") == []

    def test_contracted_dont(self) -> None:
        assert _sieve().extract_triplets("I don't like Mondays.") == []

    def test_contracted_hasnt(self) -> None:
        assert _sieve().extract_triplets("Bob hasn't read the book.") == []

    def test_never_adverbial(self) -> None:
        # spaCy tags "never" as dep_='neg' with pos_='ADV'.
        assert _sieve().extract_triplets("Cats never chase cars.") == []

    def test_will_not_future_negation(self) -> None:
        assert _sieve().extract_triplets("Alice will not eat cake.") == []


class TestBaselineUnaffected:
    """Non-negated SVO must still extract normally — regression guard."""

    def test_simple_svo(self) -> None:
        assert _sieve().extract_triplets("Alice likes cats.") == [
            ("alice", "like", "cat")
        ]

    def test_compound_subject_multi_word(self) -> None:
        triples = _sieve().extract_triplets(
            "Marie Curie, a physicist, won Nobel Prizes."
        )
        assert triples == [("marie_curie", "win", "nobel prizes")]

    def test_intensifier_is_not_negation(self) -> None:
        # "really" is advmod, not neg; should extract normally.
        assert _sieve().extract_triplets("Alice really likes cats.") == [
            ("alice", "like", "cat")
        ]

    def test_copula_positive_still_extracts(self) -> None:
        assert _sieve().extract_triplets("The cat is a mammal.") == [
            ("cat", "be", "mammal")
        ]


class TestAnnotatedPathAlsoSuppresses:
    """extract_annotated_triplets uses the same negation guard."""

    def test_annotated_empty_on_negation(self) -> None:
        out = _sieve().extract_annotated_triplets(
            "Diamonds cannot cut through steel."
        )
        assert out == []

    def test_annotated_nonempty_on_baseline(self) -> None:
        out = _sieve().extract_annotated_triplets("Alice likes cats.")
        assert len(out) == 1
        assert out[0]["subject"] == "alice"
        assert out[0]["predicate"] == "like"
        assert out[0]["object"] == "cat"


class TestMultiSentenceNegationIsolation:
    """In a mixed document, only the negated sentences are suppressed."""

    def test_mixed_doc_extracts_only_positive_sentences(self) -> None:
        text = (
            "Alice likes cats. "
            "Diamonds cannot cut through steel. "
            "Newton wrote Principia."
        )
        triples = set(_sieve().extract_triplets(text))
        assert triples == {
            ("alice", "like", "cat"),
            ("newton", "write", "principia"),
        }
