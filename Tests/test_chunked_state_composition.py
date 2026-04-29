"""Property tests for chunked Gödel-state composition.

The algebra invariant under test:

    For any context-local extractor f (DeterministicSieve qualifies),
    splitting a corpus into chunks, encoding each chunk's state, and
    composing them with LCM yields the SAME state integer as encoding
    the entire corpus end-to-end.

This is the load-bearing property for arbitrary-size input handling
(Item 1 of the omni-format roadmap). If it ever regresses, the
chunked path stops being equivalent to the unchunked path — at which
point the public API silently produces different state integers
depending on chunk_chars, breaking every downstream verifier.

Coverage layers:

  1. Algebra-level:  ``compose_chunk_states`` is commutative,
                     associative, idempotent under duplicates,
                     and equal to ``math.lcm`` over the inputs.
  2. Corpus-level:   ``state_for_corpus`` produces the same state
                     for any chunk_chars and matches the unchunked
                     ``encode_chunk_state(extract_triplets(text))``.
  3. Edge-case:      single chunk, empty extraction, abbreviation-
                     heavy text, very small chunk_chars.
"""
from __future__ import annotations

import math
import random

import pytest


# --------------------------------------------------------------------------
# Algebra-level: compose_chunk_states properties
# --------------------------------------------------------------------------


def _algebra():
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    return GodelStateAlgebra()


def test_compose_chunk_states_empty_returns_one():
    """The identity of the LCM monoid is 1."""
    a = _algebra()
    assert a.compose_chunk_states([]) == 1


def test_compose_chunk_states_single_input_returns_input():
    a = _algebra()
    p = a.get_or_mint_prime("alice", "like", "cat")
    assert a.compose_chunk_states([p]) == p


def test_compose_chunk_states_equals_python_lcm():
    """The Zig fast path and Python fallback MUST agree with
    math.lcm for any reasonable input."""
    a = _algebra()
    triples = [
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
        ("paris", "be", "city"),
        ("einstein", "propose", "relativity"),
    ]
    primes = [a.get_or_mint_prime(*t) for t in triples]
    expected = math.lcm(*primes)
    assert a.compose_chunk_states(primes) == expected


def test_compose_chunk_states_is_commutative():
    """LCM is commutative — chunk ordering doesn't matter."""
    a = _algebra()
    triples = [
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
        ("water", "contain", "hydrogen"),
        ("dolphin", "be", "mammal"),
        ("shakespeare", "write", "hamlet"),
    ]
    primes = [a.get_or_mint_prime(*t) for t in triples]
    rng = random.Random(20260429)
    canonical = a.compose_chunk_states(primes)
    for _ in range(5):
        shuffled = primes[:]
        rng.shuffle(shuffled)
        assert a.compose_chunk_states(shuffled) == canonical


def test_compose_chunk_states_is_associative():
    """LCM is associative — chunk grouping doesn't matter.

        compose([a, b, c, d]) ==
        compose([compose([a, b]), compose([c, d])]) ==
        compose([compose([a]), compose([b, c, d])])
    """
    a = _algebra()
    triples = [
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
        ("water", "contain", "hydrogen"),
        ("dolphin", "be", "mammal"),
    ]
    p = [a.get_or_mint_prime(*t) for t in triples]
    flat = a.compose_chunk_states(p)
    pairs = a.compose_chunk_states(
        [a.compose_chunk_states(p[:2]), a.compose_chunk_states(p[2:])]
    )
    head_tail = a.compose_chunk_states(
        [a.compose_chunk_states(p[:1]), a.compose_chunk_states(p[1:])]
    )
    assert flat == pairs == head_tail


def test_compose_chunk_states_is_idempotent_under_duplicates():
    """LCM is idempotent: compose([s, s, ..., s]) == s."""
    a = _algebra()
    triples = [
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
    ]
    primes = [a.get_or_mint_prime(*t) for t in triples]
    state = a.compose_chunk_states(primes)
    duplicated = [state] * 7
    assert a.compose_chunk_states(duplicated) == state


# --------------------------------------------------------------------------
# Corpus-level: state_for_corpus equivalence to unchunked path
# --------------------------------------------------------------------------


# spaCy/sieve is heavy; mark these so they can be skipped on minimal envs
spacy = pytest.importorskip(
    "spacy",
    reason="state_for_corpus needs the syntactic_sieve / spaCy",
)


def _has_en_core_web_sm() -> bool:
    """en_core_web_sm may be absent in CI's minimal environment."""
    try:
        import spacy as _sp  # noqa: F401
        spacy.load("en_core_web_sm")
        return True
    except (OSError, ImportError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_en_core_web_sm(),
    reason="en_core_web_sm not installed; skipping corpus-level equivalence",
)


CORPUS_SHORT = (
    "Marie Curie won two Nobel Prizes. "
    "Einstein proposed relativity. "
    "Shakespeare wrote Hamlet. "
    "Water contains hydrogen. "
    "Dolphins are mammals."
)

# Abbreviation-heavy: tests that mid-sentence splits don't break
# state-equivalence (the regex-based splitter cannot tell "Dr." from
# end-of-sentence "."). The sieve is sentence-local; partial sentences
# at chunk boundaries fail to extract triples, so the bag is unchanged.
CORPUS_ABBREV = (
    "Dr. Smith wrote a paper. "
    "The U.S. has fifty states. "
    "For example, e.g. dolphins are mammals. "
    "Marie Curie won Nobel Prizes. "
    "Mt. Everest is tall. "
    "Einstein proposed relativity."
)


def _state_unchunked(text: str):
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    from sum_engine_internal.algorithms.syntactic_sieve import (
        DeterministicSieve,
    )
    a = GodelStateAlgebra()
    sieve = DeterministicSieve()
    triples = sieve.extract_triplets(text)
    return a.encode_chunk_state(list(triples)), sieve, a


def _state_chunked(text: str, chunk_chars: int):
    from sum_engine_internal.algorithms.chunked_corpus import (
        state_for_corpus,
    )
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    a = GodelStateAlgebra()
    state, _ = state_for_corpus(text, a, chunk_chars=chunk_chars)
    return state


@pytest.mark.parametrize(
    "corpus,name",
    [
        (CORPUS_SHORT, "short"),
        (CORPUS_ABBREV, "abbrev-heavy"),
    ],
)
@pytest.mark.parametrize("chunk_chars", [50, 80, 120, 1000, 200_000])
def test_state_for_corpus_matches_unchunked(corpus, name, chunk_chars):
    """The headline invariant: chunk_chars MUST not affect the output
    state. Tested across five chunk sizes (forces 1-, 2-, 3-, and
    n-chunk paths) and two corpora (clean + abbreviation-heavy)."""
    unchunked, _, _ = _state_unchunked(corpus)
    chunked = _state_chunked(corpus, chunk_chars=chunk_chars)
    assert chunked == unchunked, (
        f"chunked state differs from unchunked at chunk_chars={chunk_chars} "
        f"corpus={name!r}: chunked={chunked} unchunked={unchunked}"
    )


def test_state_for_corpus_is_chunk_size_invariant():
    """Pairwise: any two chunk_chars produce the same state. This
    is a stricter restatement of the parametrized test above."""
    sizes = [40, 70, 150, 500, 1_000_000]
    states = [_state_chunked(CORPUS_SHORT, chunk_chars=n) for n in sizes]
    assert len(set(states)) == 1, (
        f"chunk-size variance produced different states: "
        f"{dict(zip(sizes, states))}"
    )


def test_state_for_corpus_returns_dedup_triples():
    """The triples returned alongside the state MUST be the same
    deduplicated set the unchunked path produces."""
    from sum_engine_internal.algorithms.chunked_corpus import (
        state_for_corpus,
    )
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    from sum_engine_internal.algorithms.syntactic_sieve import (
        DeterministicSieve,
    )

    a = GodelStateAlgebra()
    sieve = DeterministicSieve()
    unchunked_triples = set(sieve.extract_triplets(CORPUS_SHORT))

    a2 = GodelStateAlgebra()
    _, chunked_triples = state_for_corpus(
        CORPUS_SHORT, a2, chunk_chars=60, sieve=DeterministicSieve()
    )
    assert set(chunked_triples) == unchunked_triples


def test_state_for_corpus_empty_text():
    from sum_engine_internal.algorithms.chunked_corpus import (
        state_for_corpus,
    )
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    a = GodelStateAlgebra()
    state, triples = state_for_corpus("", a)
    assert state == 1
    assert triples == []


# --------------------------------------------------------------------------
# Splitter properties (no spaCy needed)
# --------------------------------------------------------------------------


def test_chunk_text_on_sentences_round_trip():
    """Chunks concatenated MUST equal the original (modulo whitespace
    eaten by the boundary regex). Specifically: every character that
    is NOT a sentence-boundary whitespace is preserved exactly."""
    from sum_engine_internal.algorithms.chunked_corpus import (
        chunk_text_on_sentences,
    )
    text = (
        "First sentence. Second sentence! Third sentence? "
        "Fourth sentence. Fifth sentence."
    )
    chunks = list(chunk_text_on_sentences(text, chunk_chars=20))
    # Re-joining with a single space replicates the original whitespace
    # at the boundaries (the splitter consumes only \s+ between sentences).
    rejoined = " ".join(c.strip() for c in chunks)
    # Compare on whitespace-collapsed forms — chunk concatenation cannot
    # be expected to be byte-perfect when whitespace is consumed.
    assert _ws_collapse(rejoined) == _ws_collapse(text)


def _ws_collapse(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s).strip()


def test_chunk_text_on_sentences_respects_chunk_chars():
    """No chunk emitted exceeds chunk_chars + a small slack for
    boundary alignment. (Slack is bounded by the longest single
    sentence in the input — if a single sentence is longer than
    chunk_chars, we yield that whole sentence as one chunk.)"""
    from sum_engine_internal.algorithms.chunked_corpus import (
        chunk_text_on_sentences,
    )
    text = ". ".join(f"Sentence number {i}" for i in range(50)) + "."
    chunks = list(chunk_text_on_sentences(text, chunk_chars=80))
    for c in chunks:
        # Either ≤ chunk_chars, or contains no internal sentence boundary
        # (i.e. it's a single sentence longer than chunk_chars).
        assert len(c) <= 80 or ". " not in c.rstrip(".")
