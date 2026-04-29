"""
Chunked corpus extraction — arbitrary-size handling for the sieve.

Item 1 of the omni-format roadmap: lift the implicit "fits in RAM /
fits in one spaCy doc" ceiling on the public state-of-corpus path
without weakening the cryptographic substrate.

The math is in ``sum_engine_internal/algorithms/semantic_arithmetic.py``
(``GodelStateAlgebra.compose_chunk_states``); this module is the
**pipeline** that consumes it: take a corpus → split on sentence
boundaries → extract per-chunk → merge states → return the same
state integer the unchunked path would have produced.

The equivalence is *only* exact for context-local extractors. The
``DeterministicSieve`` qualifies (it walks each ``doc.sents``
independently; no cross-sentence coreference), so this module is
restricted to the sieve. LLM extractors break the equivalence
because they resolve coreference across chunk boundaries; they
must use the unchunked path or a chunked-with-overlap variant
that is *not* claimed as state-equivalent.

**Public surface**:

  - ``chunk_text_on_sentences(text, *, chunk_chars)`` — splitter
  - ``state_for_corpus(text, algebra, *, chunk_chars=...)`` — full pipeline

**Tested invariant** (``Tests/test_chunked_state_composition.py``):

    state_for_corpus(text, algebra, chunk_chars=N1)
        == state_for_corpus(text, algebra, chunk_chars=N2)
        == algebra.encode_chunk_state(sieve.extract_triplets(text))

for any choice of N1, N2 large enough to admit at least two chunks.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Tuple

if TYPE_CHECKING:
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# Default chunk size: large enough that overhead is negligible
# (one spaCy parse per chunk), small enough that a 50 MB corpus
# fits in RAM (50 MB / 200 KB ≈ 250 chunks).
DEFAULT_CHUNK_CHARS = 200_000


def chunk_text_on_sentences(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> Iterator[str]:
    """Yield contiguous slices of *text* that respect sentence
    boundaries and are each at most ``chunk_chars`` characters.

    **Sentence detection**: uses spaCy's ``en_core_web_sm`` (with
    parser/NER/tagger disabled — only the sentencizer is needed)
    so the chunker and the ``DeterministicSieve`` extractor agree
    on what constitutes a sentence. Disagreement here would break
    the state-equivalence invariant: a regex chunker that splits
    "Dr." as a boundary would feed the sieve a chunk starting
    "Smith wrote a paper" — the sieve's nsubj-walker emits a
    different triple than it would on the unsplit "Dr. Smith
    wrote a paper", and the LCM ceases to equal the unchunked
    state.

    **Memory shape**: spaCy itself has a hard cap (default
    ~1M characters) on a single ``nlp(...)`` call. We respect it
    by processing the text in **windows** of ``2 * chunk_chars``
    (or up to ``SPACY_MAX_LENGTH``, whichever is smaller),
    splitting each window into spaCy-aligned sentences, then
    streaming sentences out as chunks of ≤ ``chunk_chars``. The
    last sentence of a window may be split mid-stream if it
    crosses the window boundary; the next window starts inside
    that split sentence, so spaCy re-parses it whole and the
    bag-of-sentences remains correct.

    For inputs ≤ ``chunk_chars``, yields the entire text as a
    single chunk (no overhead, no behavioural change).
    """
    if len(text) <= chunk_chars:
        yield text
        return

    # Lazy import: chunker is only used on inputs > chunk_chars
    import spacy
    try:
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "tagger", "ner", "lemmatizer", "attribute_ruler"],
        )
    except OSError:
        # Caller-friendly error: same path as DeterministicSieve.
        raise RuntimeError(
            "chunk_text_on_sentences: spaCy 'en_core_web_sm' missing. "
            "Run: python -m spacy download en_core_web_sm"
        )
    # Add a rule-based sentencizer if no senter is in the disabled
    # pipeline (with parser disabled, we need this for sent boundaries).
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    # spaCy default max_length is 1_000_000 — keep generous for big windows.
    spacy_max = min(nlp.max_length, 2_000_000)

    # Walk text in windows. Each window is parsed once for sentence
    # boundaries; sentences are aggregated into chunks ≤ chunk_chars.
    cursor = 0
    n = len(text)
    window_size = min(max(chunk_chars * 2, 50_000), spacy_max - 1)

    pending = ""
    while cursor < n:
        window_end = min(cursor + window_size, n)
        window = text[cursor:window_end]
        doc = nlp(window)
        sents = list(doc.sents)

        # If we did not consume the full text, the LAST sentence in
        # the window may be truncated (it could continue past
        # window_end). Drop it from this pass; the next window will
        # start at its beginning so spaCy re-segments it whole.
        if window_end < n and len(sents) > 1:
            consumed_sents = sents[:-1]
            advance = sents[-1].start_char
        else:
            consumed_sents = sents
            advance = window_end - cursor

        for sent in consumed_sents:
            piece = sent.text_with_ws
            if not piece:
                continue
            if len(pending) + len(piece) > chunk_chars and pending:
                yield pending
                pending = piece
            else:
                pending += piece

        cursor += advance

    if pending:
        yield pending


def state_for_corpus(
    text: str,
    algebra: "GodelStateAlgebra",
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    sieve=None,
) -> Tuple[int, List[Tuple[str, str, str]]]:
    """Compute the corpus-level Gödel state of *text* by chunking on
    sentence boundaries, extracting per-chunk via the sieve, encoding
    each chunk's state, and composing them with LCM.

    Returns ``(state_integer, all_triples)`` where ``all_triples`` is
    the deduplicated bag of triples across every chunk. Provides the
    same surface contract as the unchunked path so callers can swap
    in this function when they need arbitrary-size input handling.

    **State-equivalence guarantee** (sieve only):

        state_for_corpus(text, algebra, chunk_chars=N)[0]
            == algebra.encode_chunk_state(
                   DeterministicSieve().extract_triplets(text)
               )

    for any ``N >= 1``. Tested in
    ``Tests/test_chunked_state_composition.py``.

    Args:
        text:        The raw corpus.
        algebra:     A ``GodelStateAlgebra`` instance (caller-managed
                     so prime tables stay coherent across calls).
        chunk_chars: Maximum characters per chunk. Default 200K.
        sieve:       Optional pre-built ``DeterministicSieve``. If
                     omitted, one is constructed (incurs spaCy load).

    Returns:
        ``(state_integer, deduplicated_triples)``.
    """
    if sieve is None:
        from sum_engine_internal.algorithms.syntactic_sieve import (
            DeterministicSieve,
        )
        sieve = DeterministicSieve()

    chunk_states: list[int] = []
    triple_bag: set[Tuple[str, str, str]] = set()
    for chunk in chunk_text_on_sentences(text, chunk_chars=chunk_chars):
        triples = sieve.extract_triplets(chunk)
        # Drop triples that the algebra would reject (empty / '||' in
        # component) BEFORE encoding, so the returned bag matches the
        # encoded state. Otherwise len(triples) overstates axiom_count
        # and breaks the verifier's round-trip count check. The filter
        # mirrors get_or_mint_prime's defensive validation; keeping
        # the two in lockstep is a deliberate dependency.
        triples = [t for t in triples if _is_valid_triple(t)]
        if not triples:
            continue
        triple_bag.update(triples)
        chunk_states.append(algebra.encode_chunk_state(list(triples)))

    if not chunk_states:
        return 1, []

    state = algebra.compose_chunk_states(chunk_states)
    return state, sorted(triple_bag)


def _is_valid_triple(triple: Tuple[str, str, str]) -> bool:
    """Mirror of ``GodelStateAlgebra.get_or_mint_prime``'s validation
    contract. Reject triples whose components are empty/whitespace-only
    or contain a pipe character (which would round-trip-collide with
    the ``||`` axiom-key separator). Both cases would break canonical-
    tome round-trip verification."""
    s, p, o = triple
    if not s.strip() or not p.strip() or not o.strip():
        return False
    if "|" in s or "|" in p or "|" in o:
        return False
    return True
