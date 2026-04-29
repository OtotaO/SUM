"""End-to-end contract for the ``sum attest`` chunked sieve path.

Two claims this test pins:

  1. **Backwards-compat byte-identity.** For inputs that fit in a
     single chunk (the common case today), the new chunked code
     path emits the same ``state_integer`` it did before.

  2. **Arbitrary size unlocks.** For inputs >> spaCy's default
     ``nlp.max_length`` (1_000_000 chars), ``sum attest`` now
     succeeds end-to-end via the chunked path. Pre-chunking, this
     raised ``[E088]``.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from sum_cli.main import cmd_attest


# Skip cleanly if spaCy / en_core_web_sm aren't installed in the
# minimal test env (mirrors the chunked-state-composition gate).
spacy = pytest.importorskip("spacy")


def _has_en_core_web_sm() -> bool:
    try:
        spacy.load("en_core_web_sm")
        return True
    except (OSError, ImportError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_en_core_web_sm(),
    reason="en_core_web_sm not installed",
)


def _attest(text: str, tmp_path: Path) -> dict:
    in_path = tmp_path / "in.txt"
    in_path.write_text(text)
    args = argparse.Namespace(
        input=str(in_path),
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="Attested Tome",
        signing_key=None,
        ed25519_key=None,
        ledger=None,
        pretty=False,
        verbose=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_attest(args)
    finally:
        sys.stdout = old
    assert code == 0, f"attest failed: {buf.getvalue()}"
    return json.loads(buf.getvalue())


SHORT_CORPUS = (
    "Marie Curie won two Nobel Prizes. "
    "Einstein proposed relativity. "
    "Shakespeare wrote Hamlet. "
    "Water contains hydrogen. "
    "Dolphins are mammals."
)


def test_short_input_produces_state_byte_identical_to_unchunked(tmp_path):
    """Backwards-compat: a small input MUST mint a bundle whose
    state_integer matches the unchunked encoding. The single-chunk
    fast path in chunk_text_on_sentences (input ≤ chunk_chars yields
    the whole text) makes this exact, not approximate."""
    bundle = _attest(SHORT_CORPUS, tmp_path)

    # Independently compute the unchunked state.
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    from sum_engine_internal.algorithms.syntactic_sieve import (
        DeterministicSieve,
    )
    a = GodelStateAlgebra()
    sieve = DeterministicSieve()
    triples = sieve.extract_triplets(SHORT_CORPUS)
    expected_state = a.encode_chunk_state(list(triples))

    assert int(bundle["state_integer"]) == expected_state


@pytest.mark.slow
def test_megacorpus_above_spacy_max_length_does_not_raise(tmp_path):
    """Arbitrary-size: an input larger than spaCy's default
    ``nlp.max_length`` (1_000_000 chars) MUST attest successfully
    via the chunked path. Pre-chunking, this raised an [E088]
    ValueError ("Text of length N exceeds maximum of 1000000").

    We construct a 1.2 MB corpus by repeating the short corpus
    enough times. The state integer for a repeated set of axioms
    equals the state integer for one copy (LCM is idempotent), so
    we additionally assert the bundle's state matches the short-
    corpus state.
    """
    sentence = (
        "Einstein proposed relativity. "
        "Marie Curie won Nobel Prizes. "
        "Shakespeare wrote Hamlet. "
    )
    n_repeats = (1_200_000 // len(sentence)) + 1
    big_text = sentence * n_repeats
    assert len(big_text) > 1_000_000

    bundle = _attest(big_text, tmp_path)

    # State idempotence: the same set of axioms regardless of
    # how many times each appears. Compute reference state on a
    # one-copy version (cheap to extract) and compare.
    one_copy_bundle = _attest(sentence, tmp_path)
    assert int(bundle["state_integer"]) == int(one_copy_bundle["state_integer"])
