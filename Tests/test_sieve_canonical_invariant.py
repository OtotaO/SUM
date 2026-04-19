"""Tests the sieve output satisfies the canonical codec's single-token
subject/predicate invariant.

The canonical template is ``"The {s} {p} {o}."`` and OuroborosVerifier parses
it with ``^The (\\S+) (\\S+) (.+)\\.$``. Subject and predicate must match
``\\S+`` (no whitespace); object matches ``.+`` (greedy, whitespace allowed).

This test guards against regressions like the pre-commit bug where a
multi-word subject such as "Marie Curie" was emitted as ``"marie curie"``
(space-joined) and silently broke the canonical round-trip — Ouroboros
verification reported drift on any document with a multi-word subject,
contradicting §1.1's provable-lossless claim.
"""
from __future__ import annotations

import re

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.algorithms.syntactic_sieve import DeterministicSieve
from internal.ensemble.ouroboros import OuroborosVerifier
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator


_SUBJECT_PREDICATE_RE = re.compile(r"^\S+$")


def _sieve() -> DeterministicSieve:
    return DeterministicSieve()  # type: ignore[no-untyped-call]


class TestSieveCompliesWithCanonicalInvariant:
    """Subject and predicate from the sieve must not contain whitespace."""

    def test_single_token_subject_still_works(self) -> None:
        triples = _sieve().extract_triplets("Alice likes cats.")
        assert triples == [("alice", "like", "cat")]
        s, p, _ = triples[0]
        assert _SUBJECT_PREDICATE_RE.match(s)
        assert _SUBJECT_PREDICATE_RE.match(p)

    def test_multi_word_subject_is_underscore_joined(self) -> None:
        triples = _sieve().extract_triplets(
            "Marie Curie, a physicist, won Nobel Prizes."
        )
        assert triples, "sieve should extract at least one triple"
        s, p, _o = triples[0]
        assert s == "marie_curie"
        assert _SUBJECT_PREDICATE_RE.match(s), f"subject has whitespace: {s!r}"
        assert _SUBJECT_PREDICATE_RE.match(p), f"predicate has whitespace: {p!r}"

    def test_predicate_never_contains_whitespace(self) -> None:
        # Probe several constructions; predicate is always spaCy's lemma,
        # which is a single lexeme, but assert the invariant explicitly.
        for text in [
            "Alice likes cats.",
            "Newton wrote Principia.",
            "Marie Curie won Nobel Prizes.",
            "The moon orbits Earth.",
        ]:
            for s, p, _o in _sieve().extract_triplets(text):
                assert _SUBJECT_PREDICATE_RE.match(s), (
                    f"subject whitespace in {text!r}: {s!r}"
                )
                assert _SUBJECT_PREDICATE_RE.match(p), (
                    f"predicate whitespace in {text!r}: {p!r}"
                )


class TestCanonicalRoundtripPreservesMultiWordSubject:
    """End-to-end: sieve → canonical text → OuroborosVerifier must be lossless
    even when the subject is multi-word.
    """

    def test_marie_curie_canonical_round_trip_is_lossless(self) -> None:
        algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
        sieve = _sieve()
        generator = AutoregressiveTomeGenerator(algebra)
        verifier = OuroborosVerifier(algebra, sieve, generator)

        triples = sieve.extract_triplets(
            "Marie Curie, a physicist, won Nobel Prizes."
        )
        assert triples
        state = algebra.encode_chunk_state(list(triples))
        proof = verifier.verify_from_state(state)

        assert proof.is_conserved, (
            f"canonical round-trip diverged on multi-word subject: "
            f"missing={proof.missing_axioms!r} extra={proof.extra_axioms!r}"
        )
        assert proof.missing_axioms == []
        assert proof.extra_axioms == []
