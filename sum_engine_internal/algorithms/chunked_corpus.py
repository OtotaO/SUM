"""Bounded sentence-window extraction and exact LCM composition.

``chunk_chars`` is a target, not a hard cut through a sentence. The splitter
retains an unfinished sentence across windows and grows that window up to
``max_sentence_chars``. If no safe boundary is found within that bound it
raises ``SentenceTooLongError``. It never silently splits an overlong sentence.

Every returned chunk is a contiguous source slice, so concatenating chunks
recovers the exact input. The maximum parsing window is bounded; retaining
whole sentences can produce a chunk larger than the target.

LCM composition preserves the union of per-chunk triples exactly. Equality
to whole-document extraction is conditional on matching segmentation and
sentence-local parsing, not universal: spaCy's rule-based sentencizer and its
dependency parser can choose different sentence boundaries. Tests cover named
corpora, abbreviations, and an overlong-sentence regression. LLM extractors
with cross-sentence context are outside this contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Tuple

if TYPE_CHECKING:
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

DEFAULT_CHUNK_CHARS = 200_000
DEFAULT_MAX_SENTENCE_CHARS = 900_000


class SentenceTooLongError(ValueError):
    """A complete sentence boundary could not be found in a bounded window.

    ``start_char`` and ``limit`` locate the unprocessed source continuation.
    Callers may retry with a larger supported limit or request preprocessing;
    they must not present any partially accumulated state as complete.
    """

    def __init__(self, start_char: int, limit: int):
        self.start_char = start_char
        self.limit = limit
        super().__init__(
            f"No complete sentence boundary within {limit} characters at "
            f"source offset {start_char}; increase max_sentence_chars within "
            "the model limit or segment this source explicitly."
        )


def chunk_text_on_sentences(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
) -> Iterator[str]:
    """Yield exact source slices, retaining whole detected sentences.

    Chunk size is a target; a single sentence can exceed it. Parsing windows
    grow only as needed, up to ``max_sentence_chars`` (and spaCy max_length).
    A non-final window's last sentence is always retained for the next pass.
    This avoids losing a subject when a long sentence crosses a window.
    SentenceTooLongError includes the source offset and effective limit.
    """
    for name, value in (("chunk_chars", chunk_chars), ("max_sentence_chars", max_sentence_chars)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not text:
        yield text
        return

    import spacy
    try:
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "tagger", "ner", "lemmatizer", "attribute_ruler"],
        )
    except OSError as exc:
        raise RuntimeError(
            "chunk_text_on_sentences: spaCy en_core_web_sm missing. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    limit = min(max_sentence_chars, nlp.max_length)
    # Apply the actual model cap before the short-input shortcut. A caller
    # can raise both requested limits beyond spaCy max_length.
    if len(text) <= min(chunk_chars, limit):
        yield text
        return
    target = min(chunk_chars, limit)
    initial_window = min(max(target * 2, 50_000), limit)
    window_size = initial_window
    cursor = 0
    n = len(text)
    pending = ""
    while cursor < n:
        window_end = min(cursor + window_size, n)
        window = text[cursor:window_end]
        sents = list(nlp(window).sents)
        if window_end < n:
            if len(sents) < 2:
                if window_size >= limit:
                    raise SentenceTooLongError(cursor, limit)
                window_size = min(window_size * 2, limit)
                continue
            # The final sentence may continue in the next window. Carry its
            # original source offset, not a reconstructed or trimmed string.
            advance = sents[-1].start_char
            if advance <= 0:
                raise SentenceTooLongError(cursor, limit)
            starts = [0] + [sent.start_char for sent in sents[1:-1]]
            ends = starts[1:] + [advance]
        else:
            advance = len(window)
            starts = [0] + [sent.start_char for sent in sents[1:]]
            ends = starts[1:] + [advance]

        for start, end in zip(starts, ends):
            piece = window[start:end]
            if not piece:
                continue
            if pending and len(pending) + len(piece) > target:
                yield pending
                pending = ""
            pending += piece
        cursor += advance
        window_size = initial_window

    if pending:
        yield pending


def state_for_corpus(
    text: str,
    algebra: "GodelStateAlgebra",
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
    sieve=None,
) -> Tuple[int, List[Tuple[str, str, str]]]:
    """Compute the corpus-level Gödel state of *text* by chunking on
    sentence boundaries, extracting per-chunk via the sieve, encoding
    each chunk's state, and composing them with LCM.

    Returns ``(state_integer, all_triples)`` where ``all_triples`` is
    the deduplicated bag of triples across every chunk. Provides the
    same surface contract as the unchunked path so callers can swap
    in this function when they need arbitrary-size input handling.

    LCM composition exactly preserves the union of the extracted triples.
    Equality to whole-document extraction additionally requires identical
    sentence boundaries and sentence-local parsing. The rule-based splitter
    can disagree with the sieve's dependency parser; that equality is tested
    on the named fixtures, not guaranteed for arbitrary prose or chunk sizes.
    Unfinished window sentences are retained intact. A sentence boundary that
    cannot be resolved within ``max_sentence_chars`` raises ``SentenceTooLongError``
    rather than emitting a partial-sentence state.

    Args:
        text:        The raw corpus.
        algebra:     A ``GodelStateAlgebra`` instance (caller-managed
                     so prime tables stay coherent across calls).
        chunk_chars: Target characters per chunk. A complete sentence may exceed it.
        max_sentence_chars: Hard parsing-window limit. Default 900K. The effective
                     cap is also bounded by the loaded spaCy model max_length.
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
    for chunk in chunk_text_on_sentences(
        text, chunk_chars=chunk_chars, max_sentence_chars=max_sentence_chars,
    ):
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
