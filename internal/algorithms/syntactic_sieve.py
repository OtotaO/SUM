"""
Deterministic Syntactic Sieve — High-Fidelity Edge NLP

Extracts topological (Subject, Predicate, Object) triplets using strict
grammatical dependency parsing via spaCy.  Replaces the LLM for bulk
ingestion, parsing text at bare-metal CPU speeds.

Cost: $0.  Speed: 10,000+ words per second.  Deterministic: always.

Phase 13: Zenith of Process Intensification.
Stage 4 — Hedging detection for linguistic confidence signals.

Author: ototao
License: Apache License 2.0
"""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from internal.infrastructure.provenance import (
    EXCERPT_MAX_CHARS,
    ProvenanceRecord,
    sha256_uri_for_text,
)

SIEVE_EXTRACTOR_ID = "sum.sieve:deterministic_v1"


# ─── Hedging / Epistemic Markers ──────────────────────────────────────
# Words and phrases that indicate uncertainty in the source text.
# Presence reduces confidence at the linguistic level.

HEDGING_MARKERS = [
    # Modal verbs of uncertainty
    re.compile(r"\b(may|might|could|would)\b", re.IGNORECASE),
    # Epistemic adverbs
    re.compile(r"\b(possibly|probably|perhaps|likely|unlikely|apparently|"
               r"allegedly|purportedly|supposedly|seemingly|arguably|"
               r"conceivably|presumably|ostensibly)\b", re.IGNORECASE),
    # Hedging verbs
    re.compile(r"\b(suggest|imply|indicate|appear|seem|tend|believe|"
               r"estimate|speculate|hypothesize|propose|conjecture)\b",
               re.IGNORECASE),
    # Hedging phrases
    re.compile(r"\b(it is (thought|believed|estimated|assumed)|"
               r"according to( some)?|some (researchers|scientists|experts)|"
               r"there is (some )?evidence|in (some|certain) cases|"
               r"not (entirely |fully )?clear)\b", re.IGNORECASE),
]

# Each matched marker reduces certainty by this factor
HEDGE_PENALTY_PER_MARKER = 0.15
HEDGE_FLOOR = 0.20  # minimum confidence from hedging alone


_FALLBACK_CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ"})


def _is_negated(sent) -> bool:
    """Return True iff the sentence contains a negation particle scoping the
    main predication.

    spaCy tags ``not``, ``n't``, ``never`` (and similar) as ``dep_ == "neg"``
    attached to the ROOT verb or copular AUX. When a negation is present,
    the SVO structure still parses — but its semantic polarity is inverted
    relative to what the bare triple would assert. Emitting a positive
    (s, p, o) from a negated source sentence is worse than emitting nothing:
    it silently ships a false assertion into the Gödel state with no
    surface marker that the original sentence denied it.

    The hedging detector (``detect_hedging``) handles the weaker modal
    class (``may``, ``might``, ``possibly``) by lowering a certainty score.
    Negation is not uncertainty — it is an inversion — so the correct
    response is to refuse extraction, not to annotate it.

    Scope: any ``dep_ == "neg"`` anywhere in the sentence triggers suppression.
    This is intentionally aggressive: a doubly-negated sentence is ambiguous
    under SUM's SVO frame, and false negatives (missing a triple) are
    strictly preferable to false positives (asserting an inverted fact).
    """
    for token in sent:
        if token.dep_ == "neg":
            return True
    return False


def _extract_from_sent(sent) -> Optional[Tuple[str, str, str]]:
    """Extract at most one (subject, predicate, object) triple from a sentence.

    Returns None if the sentence is negated, produces no valid ROOT verb, or
    yields a parse whose subject/object exceed the size filters. The POS
    fallback is consulted only when dependency-based extraction fails.

    This helper is the single source of truth for per-sentence extraction.
    ``extract_triplets`` and ``extract_with_provenance`` both call it, so
    their outputs remain triple-for-triple identical — the provenance path
    just adds metadata around the same extraction decisions.
    """
    if _is_negated(sent):
        return None

    subject = None
    predicate = None
    object_ = None

    for token in sent:
        if token.dep_ == "ROOT" or token.pos_ == "VERB":
            predicate = token.lemma_
            # Compound modifiers are joined with '_' for subject (not space)
            # so multi-word subjects satisfy the canonical template's "\S+"
            # parser in OuroborosVerifier. Object keeps space-joining because
            # the canonical regex for object is ".+" and accommodates spaces.
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "csubj", "npadvmod"):
                    modifiers = [
                        c.text for c in child.children
                        if c.dep_ in ("amod", "compound")
                    ]
                    subject = "_".join(modifiers + [child.lemma_]).strip()
            for child in token.children:
                if child.dep_ in ("dobj", "pobj", "attr", "acomp"):
                    modifiers = [
                        c.text for c in child.children
                        if c.dep_ in ("amod", "compound")
                    ]
                    object_ = " ".join(modifiers + [child.lemma_]).strip()

    if subject and predicate and object_:
        if len(subject.split()) <= 5 and len(object_.split()) <= 8:
            return (subject.lower(), predicate.lower(), object_.lower())

    return _pos_fallback_triplet(sent)


def _pos_fallback_triplet(sent):
    """POS-based fallback extraction for sentences the dep parser misparses.

    Activates only when dep-based extraction yielded nothing for the sentence.
    Strategy: if the sentence contains EXACTLY three content tokens
    (NOUN / PROPN / VERB / ADJ — excluding DET / AUX / ADV / ADP / PUNCT / PART),
    emit them in order as (subject, predicate, object).

    This targets the known spaCy en_core_web_sm failure mode on sentences
    like "Dogs chase cats" where the verb is mis-tagged as NOUN and the
    ROOT is shifted to the object noun. Conservative: the exact three-content
    rule refuses to fire on sentences with adverbial modifiers, adjectives
    stacking on the object, passive-voice auxiliaries, or prepositional
    phrases — all of which the dep-based path handles correctly.

    Returns (subject_lemma, predicate_lemma, object_lemma) all lowercased,
    or None if the pattern does not match.
    """
    content = [t for t in sent if t.pos_ in _FALLBACK_CONTENT_POS]
    if len(content) != 3:
        return None
    s, p, o = content
    if not (p.lemma_.isalpha() and 1 < len(p.lemma_) <= 20):
        return None

    # When spaCy mis-tags a plural noun as ADJ (e.g. "Dogs" in "Dogs chase
    # cats"), the token lemma preserves the plural form. Reverse the
    # common -s plural so the canonical key matches the expected singular.
    s_lemma = s.lemma_.lower()
    if (
        s.tag_.startswith("JJ")
        and s_lemma.endswith("s")
        and len(s_lemma) > 2
        and s_lemma[:-1].isalpha()
    ):
        s_lemma = s_lemma[:-1]

    return (s_lemma, p.lemma_.lower(), o.lemma_.lower())


def detect_hedging(text: str) -> float:
    """Score the linguistic certainty of a text.

    Returns a value in [HEDGE_FLOOR, 1.0] where 1.0 means no hedging
    detected and lower values indicate increasing uncertainty.

    This is a metadata-only signal — it does NOT affect the algebra.
    """
    if not text:
        return 1.0

    hit_count = 0
    for pattern in HEDGING_MARKERS:
        hits = pattern.findall(text)
        hit_count += len(hits)

    if hit_count == 0:
        return 1.0

    certainty = 1.0 - (hit_count * HEDGE_PENALTY_PER_MARKER)
    return max(HEDGE_FLOOR, certainty)


class DeterministicSieve:
    """
    High-Fidelity Edge NLP.

    Extracts topological (Subject, Predicate, Object) triplets using
    strict grammatical dependency parsing.

    Cost: $0. Speed: 10,000+ words per second.
    """

    def __init__(self):
        import spacy  # Lazy import: only required when sieve is instantiated

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
            )
            self.nlp = spacy.load("en_core_web_sm")

    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Parse text into semantic triplets using dependency grammar.

        Walks each sentence's dependency tree to find the ROOT verb,
        then extracts its nominal subject and direct/prepositional
        object, including adjectival and compound modifiers.

        Args:
            text: Raw text to parse.

        Returns:
            Deduplicated list of (subject, predicate, object) tuples.
        """
        doc = self.nlp(text)
        triplets = []
        for sent in doc.sents:
            triple = _extract_from_sent(sent)
            if triple is not None:
                triplets.append(triple)
        return list(set(triplets))  # Deduplicate

    def extract_with_provenance(
        self,
        text: str,
        source_uri: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> List[Tuple[Tuple[str, str, str], ProvenanceRecord]]:
        """Extract (s, p, o) triples paired with per-sentence ProvenanceRecords.

        Each returned record locates the originating sentence's byte range in
        ``source_uri``'s bytes, names the extractor version, and carries a
        literal text excerpt (up to EXCERPT_MAX_CHARS) so third-party auditors
        can validate the claim without refetching the source.

        Args:
            text:        Input text. Also becomes the content-addressable
                         source if ``source_uri`` is omitted.
            source_uri:  Optional override. Defaults to ``sha256:<hex>`` of
                         ``text``'s UTF-8 bytes, which makes the byte
                         ranges self-consistent and third-party-verifiable
                         without any network dependency.
            timestamp:   Optional ISO-8601 UTC timestamp. Defaults to
                         ``datetime.now(timezone.utc).isoformat()``.

        Returns:
            List of ``((s, p, o), ProvenanceRecord)`` pairs — NOT deduplicated
            at the triple level. Two sentences producing the same triple yield
            two records with different byte ranges and different prov_ids.
            The AkashicLedger is the dedup boundary, not this method.
        """
        src = source_uri or sha256_uri_for_text(text)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        doc = self.nlp(text)
        out: List[Tuple[Tuple[str, str, str], ProvenanceRecord]] = []
        for sent in doc.sents:
            triple = _extract_from_sent(sent)
            if triple is None:
                continue
            # spaCy's sent.start_char / end_char are character offsets in
            # the original text; convert to byte offsets in the UTF-8
            # representation so the byte_range is correct for any consumer
            # that stores bytes, not Python strings.
            byte_start = len(text[: sent.start_char].encode("utf-8"))
            byte_end = len(text[: sent.end_char].encode("utf-8"))
            excerpt = sent.text[:EXCERPT_MAX_CHARS]
            record = ProvenanceRecord(
                source_uri=src,
                byte_start=byte_start,
                byte_end=byte_end,
                extractor_id=SIEVE_EXTRACTOR_ID,
                timestamp=ts,
                text_excerpt=excerpt,
            )
            out.append((triple, record))
        return out

    def extract_annotated_triplets(
        self, text: str
    ) -> List[Dict[str, object]]:
        """Extract triplets with per-sentence hedging annotation.

        Returns a list of dicts:
            {
                "subject": str,
                "predicate": str,
                "object": str,
                "linguistic_certainty": float,  # 1.0 = definite, <1.0 = hedged
            }

        The linguistic_certainty score is a metadata-only signal
        that does NOT affect the Gödel algebra.
        """
        doc = self.nlp(text)
        results = []

        for sent in doc.sents:
            # Negated sentences produce no triple — see _is_negated.
            if _is_negated(sent):
                continue

            subject = None
            predicate = None
            object_ = None

            for token in sent:
                if token.dep_ == "ROOT" or token.pos_ == "VERB":
                    predicate = token.lemma_
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass", "csubj", "npadvmod"):
                            modifiers = [
                                c.text for c in child.children
                                if c.dep_ in ("amod", "compound")
                            ]
                            # '_'-joined to satisfy canonical "\S+" subject invariant.
                            subject = "_".join(modifiers + [child.lemma_]).strip()
                    for child in token.children:
                        if child.dep_ in ("dobj", "pobj", "attr", "acomp"):
                            modifiers = [
                                c.text for c in child.children
                                if c.dep_ in ("amod", "compound")
                            ]
                            object_ = " ".join(modifiers + [child.lemma_]).strip()

            if subject and predicate and object_:
                if len(subject.split()) <= 5 and len(object_.split()) <= 8:
                    certainty = detect_hedging(sent.text)
                    results.append({
                        "subject": subject.lower(),
                        "predicate": predicate.lower(),
                        "object": object_.lower(),
                        "linguistic_certainty": certainty,
                    })

        return results
