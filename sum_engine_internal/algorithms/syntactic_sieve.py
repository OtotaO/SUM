"""
Deterministic Syntactic Sieve — High-Fidelity Edge NLP

Extracts topological (Subject, Predicate, Object) triplets using strict
grammatical dependency parsing via spaCy.  Replaces the LLM for bulk
ingestion, parsing text at bare-metal CPU speeds.

Cost: $0.  Speed: 10,000+ words per second.  Deterministic: always.

Phase 13: Zenith of Process Intensification.
Stage 4 — Hedging detection for linguistic confidence signals.
Extractor v2: clause guard (one triple per sentence, from one main-clause
predicate; negated, question, conditional, cross-clause and attributed
sentences are suppressed and counted).

Author: ototao
License: Apache License 2.0
"""

import bisect
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple, TypedDict

from sum_engine_internal.infrastructure.provenance import (
    EXCERPT_MAX_CHARS,
    ProvenanceRecord,
    sha256_uri_for_text,
)

# v2 (clause guard): a triple comes from one main-clause predicate, and
# conditional, question and cross-clause sentences are suppressed. See
# the "Clause guard" section below. v1 records stay valid as history;
# the id names the extraction behaviour that produced a record, and
# ``DeterministicSieve(extractor_id=SIEVE_EXTRACTOR_ID_V1)`` replays v1.
SIEVE_EXTRACTOR_ID_V1 = "sum.sieve:deterministic_v1"
SIEVE_EXTRACTOR_ID = "sum.sieve:deterministic_v2"


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


# ─── Noise filter ─────────────────────────────────────────────────────
#
# The dependency parser was trained on prose. When fed markdown, code
# fences, table cells, or YAML/TOML blocks, it cheerfully extracts
# triples like ('|', 'close', 'this') (table-cell pipe), ('#', 'license',
# 'apache') (heading hash), or ('proof_boundary.md`](docs', 'be',
# 'arbiter') (link-residue) — all real captures from running on
# README.md. These triples encode no semantic content and break the
# canonical-tome round-trip in the algebra layer (the '||' axiom-key
# separator gets corrupted by stray '|' characters in components,
# yielding empty fields after split).
#
# The right filter point is here, at the extraction boundary: noise
# never enters the pipeline. The algebra layer keeps a defensive
# backstop (sum_engine_internal.algorithms.semantic_arithmetic) for
# inputs from non-sieve extractors, but for sieve outputs this filter
# fires first.
#
# Heuristics chosen to be conservative — drop only what's clearly
# syntactic noise. Single ASCII letters, short numerics, alphanum-
# only punctuation-free strings are kept; legitimate facts like
# ('alice', 'has', '3 cats') survive even though '3' alone wouldn't.

_NOISE_PUNCT_CHARS = frozenset("|\\`*<>[]{}()=/")  # markdown / code / table noise

# Round-2 (post-#91): quantity / measurement / range markers. Components
# carrying these are numeric values, not semantic entities. Surfaced by
# survey across README + CHANGELOG + PROOF_BOUNDARY: `~$0.12`, `21.1×`,
# `0.94–0.96`, `1.0→0.9`, `80%+`, `%_drift`. Adding `+` is conservative
# (rules out "C++" but typical SUM corpora don't carry that).
_QUANTITY_CHARS = frozenset("$%×÷±≈→←↔–—~+°≤≥≠∞")

# Tokens that signal the component is a path/URL/file reference rather
# than a semantic entity. Substring match — generous on purpose.
_PATH_LIKE_NEEDLES = ("://", ".md", ".py", ".js", ".json", "](", "**(", "**[")

# Round-2: components longer than this are almost always hashes, IDs,
# or path strings — never legitimate noun phrases. spaCy parses such
# tokens as a single noun and uses them whole as a subject/object.
_MAX_COMPONENT_LEN = 80

# Round-2: pure-numeric matches like '0.5750', '10.2', '1.0', '21.1'.
# Years like '2026' are bare integers, not multi-decimal — they pass.
_DECIMAL_NUMBER_RE = re.compile(r"^\d+(\.\d+)+$")

# Round-2: long hex-only strings (sha256, sha1, ed25519 hex, …). 16+
# chars with no other characters confirms a hash, not a word.
_HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{16,}$")


def _is_noise_component(s: str) -> bool:
    """Return True if *s* is syntactic noise that should not enter
    the extracted-triple bag.

    Round-1 (PR #91) reject conditions:
      - Empty / whitespace-only.
      - Single character that isn't a multi-letter abbreviation
        (catches stray '#', '*', '\\', '|', '§', '✓', and bare digits).
      - Contains any of ``_NOISE_PUNCT_CHARS`` (table pipe, code
        backtick, footnote asterisk, link bracket, etc.).
      - Contains a path/URL needle from ``_PATH_LIKE_NEEDLES``.
      - No alphanumeric character at all (pure punctuation).

    Round-2 (this PR) additional reject conditions:
      - Contains any of ``_QUANTITY_CHARS`` (currency/math/range
        symbols indicate measurements, not entities).
      - Pure decimal-number form like ``0.5750`` or ``21.1``
        (matches ``_DECIMAL_NUMBER_RE``); years like ``2026`` pass
        because they're bare integers without the decimal pattern.
      - Length exceeds ``_MAX_COMPONENT_LEN`` (hashes, IDs, paths).
      - Hex-only string of 16+ chars (matches ``_HEX_HASH_RE``).
    """
    s = s.strip()
    if not s:
        return True
    if len(s) <= 1:
        return True
    # Round-2: length cap (catches long hash / nanoid / path strings).
    if len(s) > _MAX_COMPONENT_LEN:
        return True
    if any(c in _NOISE_PUNCT_CHARS for c in s):
        return True
    # Round-2: quantity / measurement markers.
    if any(c in _QUANTITY_CHARS for c in s):
        return True
    s_lower = s.lower()
    if any(needle in s_lower for needle in _PATH_LIKE_NEEDLES):
        return True
    if not any(c.isalnum() for c in s):
        return True
    # Round-2: pure decimal-number form (`0.5750`, `21.1`).
    if _DECIMAL_NUMBER_RE.match(s):
        return True
    # Round-2: hex-only hash form (16+ hex chars with no other content).
    if _HEX_HASH_RE.match(s):
        return True
    return False


def _is_clean_triple(triple: Tuple[str, str, str]) -> bool:
    """Triple-level filter: every component must pass
    ``_is_noise_component`` (negated)."""
    s, p, o = triple
    return not (
        _is_noise_component(s)
        or _is_noise_component(p)
        or _is_noise_component(o)
    )


def _is_negated(sent: Any) -> bool:
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


def _is_passive(sent: Any) -> bool:
    """Return True iff the sentence's ROOT verb carries a passive-voice
    grammatical subject (``dep_ == "nsubjpass"``).

    A passive construction inverts the surface order: the grammatical
    subject is the semantic OBJECT, and the semantic subject (if
    recoverable) lives inside the agent prepositional phrase — spaCy
    tags ``by`` with ``dep_ == "agent"`` and the agent noun as a
    ``pobj`` child of the ``by`` token. Emitting a triple in surface
    (s,p,o) order from such a sentence produces the inverted fact —
    "Hamlet was written by Shakespeare" → (hamlet, write, shakespeare)
    which asserts the opposite of the source. The POS fallback is
    especially dangerous here because for three-content-token passives
    (e.g. "Hamlet/written/Shakespeare") it produces the inverted
    triple even when the dep-based path bails out. Callers that detect
    passive should either run the swap-and-emit path below
    (``_extract_passive``) or refuse to extract at all.
    """
    for child in sent.root.children:
        if child.dep_ == "nsubjpass":
            return True
    return False


def _extract_passive(sent: Any) -> Optional[Tuple[str, str, str]]:
    """Extract an active-form triple from a passive-voice sentence.

    Strategy (works for both "Hamlet was written by Shakespeare" and
    any other ``nsubjpass + agent-by-pobj`` surface):

        real subject = the pobj under the agent ``by`` (semantic agent)
        real object  = the nsubjpass noun (semantic patient)
        predicate    = ROOT verb's lemma

    If the passive is agentless ("The paper was submitted."), the
    agent is grammatically absent and the semantic subject cannot be
    recovered — return None. This is the same discipline as negation:
    refusing to extract is strictly preferable to asserting an
    inverted fact.
    """
    root = sent.root
    subj_token = None
    obj_token = None
    for child in root.children:
        if child.dep_ == "nsubjpass" and obj_token is None:
            obj_token = child
        elif child.dep_ == "agent":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    subj_token = grandchild
                    break
    if subj_token is None or obj_token is None:
        return None

    subj_modifiers = [
        c.text for c in subj_token.children
        if c.dep_ in ("amod", "compound")
    ]
    subject = "_".join(subj_modifiers + [subj_token.lemma_]).strip()
    obj_modifiers = [
        c.text for c in obj_token.children
        if c.dep_ in ("amod", "compound")
    ]
    object_ = " ".join(obj_modifiers + [obj_token.lemma_]).strip()
    predicate = root.lemma_

    if not (subject and predicate and object_):
        return None
    if len(subject.split("_")) > 5 or len(object_.split()) > 8:
        return None
    return (subject.lower(), predicate.lower(), object_.lower())


# ─── Clause guard (extractor v2) ──────────────────────────────────────
#
# v1 walked every ROOT-or-VERB token of a sentence and let the last
# subject, the last predicate and the last object win independently. On
# a multi-clause sentence that assembles triples the source never
# asserts: "If the tenant pays rent late, the landlord charges a fee."
# gave (landlord, charge, fee), a consequent stated as fact; "Bob said
# that Alice stole the car." gave (alice, steal, car); "Alice owns the
# house where Bob grew up." gave (bob, grow, house), a subject and
# predicate from the relative clause stitched to the main clause's
# object.
#
# v2 checks each sentence in this order and stops at the first hit:
#
#   negation      v1's ``_is_negated``, plus "neither" / "nor" and
#                 determiner or pronoun negation ("no", "nobody",
#                 "nothing", "none", "noone", "nowhere";
#                 ``_has_negative_quantifier``), which spaCy does not tag
#                 as negation ("Neither Alice nor Bob owns the car." gave
#                 (alice, own, car), "No student passed the exam." gave
#                 (student, pass, exam)).
#   question      the sentence ends with "?" (``_is_question``).
#   conditional   a conditional or hypothetical clause is present
#                 (``_is_conditional``), or two clauses are joined by
#                 "or" (``_is_clause_disjunction``); neither clause is
#                 asserted.
#   attribution   the main clause is reported content: "according to"
#                 anywhere, or a reporting verb attached to a main-clause
#                 predicate as a parenthetical or an "as" clause
#                 (``_is_attributed``: "The suspect, police said, stole
#                 the car."). Counted as ``cross_clause``, like "Bob said
#                 that Alice stole the car.".
#   passive       unchanged from v1 (``_is_passive``/``_extract_passive``).
#   single clause subject, predicate and object come from ONE main-clause
#                 predicate (``_single_clause_triple``). When none has
#                 all three but v1 would have stitched a triple from
#                 other clauses, the sentence is suppressed as
#                 ``cross_clause``; otherwise the POS fallback runs, and
#                 its triple is also suppressed as ``cross_clause`` when
#                 the sentence has more than one clause
#                 (``_has_several_clauses``).
#
# The same reasoning as negation applies throughout: a suppressed
# sentence is a recall miss, a stitched or conditional triple is a false
# fact in the bundle. ``DeterministicSieve.extract_triplets_with_report``
# counts suppressions per reason.
#
# v2 also stops taking a noun-phrase adverbial (npadvmod: "last year",
# "the next day") as the subject when the predicate has a real subject
# or can inherit one: v1 gave (last_year, buy, car) for "Alice bought a
# car last year.".

SUPPRESSION_REASONS = ("negation", "conditional", "question", "cross_clause")


class SuppressionReport(TypedDict):
    """Per-call count of what the sieve did with each sentence."""

    sentences: int
    extracted: int
    suppressed: Dict[str, int]


_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj"})
_OBJECT_DEPS = frozenset({"dobj", "pobj", "attr", "acomp"})
_MODIFIER_DEPS = frozenset({"amod", "compound"})
_NOMINAL_POS = frozenset({"NOUN", "PROPN", "PRON"})

# Subjects that make a token head a finite clause (used to tell a
# conditional "Provided the buyer pays ..." from a participle phrase
# "Assuming control of the board, ...").
_CLAUSE_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"})

# Single-word subordinators that make their clause conditional when
# attached as mark/advmod. "when" is handled separately: it counts only
# when it introduces an adverbial clause ("When you cancel the plan, we
# charge a fee."), not as a relative adverb ("the day when ...").
_CONDITIONAL_SUBORDINATORS = frozenset({"if", "unless", "whether", "lest", "whenever"})

# Participles that head a conditional clause anywhere in the sentence
# ("Provided (that) the buyer pays ...", "... provided the buyer pays").
_CONDITIONAL_PARTICIPLES = frozenset({"provided", "providing"})

# Hypothesis openers. They count as the sentence's first word, after a
# comma, or as an adverbial clause ("Alice will buy the car, assuming
# Bob agrees."), and only when they head a finite clause.
_HYPOTHESIS_OPENERS = frozenset({"assuming", "supposing", "suppose"})

# Auxiliaries that open an inverted conditional ("Had the tenant paid
# rent, ...", "Were the court to approve ...", "Should the buyer
# default, ...").
_INVERSION_AUX = frozenset({"had", "were", "should"})
_NP_START_POS = frozenset({"DET", "PRON", "PROPN", "NOUN", "ADJ", "NUM"})

# "or" between clauses, and "either" before a coordination, make a
# disjunction (``_is_clause_disjunction``). "neither" / "nor" negate
# every member of their coordination, so v2 counts them as negation.
_DISJUNCTIVE_CC = frozenset({"or"})
_DISJUNCTIVE_PRECONJ = frozenset({"either"})
_NEGATIVE_COORDINATORS = frozenset({"neither", "nor"})

# Negative pronouns ("no one" is the determiner "no" on "one").
_NEGATIVE_PRONOUNS = frozenset({"nobody", "nothing", "none", "noone", "nowhere"})

# Reporting verbs (lemmas) whose parenthetical or "as" clause marks the
# main clause as someone's report (``_is_attributed``).
_REPORTING_VERBS = frozenset({
    "say", "claim", "report", "allege", "state", "tell", "add", "note",
    "insist", "admit", "deny", "argue", "suggest", "believe", "think",
    "estimate", "warn", "write",
})

# Signals that a sentence has more than one clause, for the POS
# fallback (``_has_several_clauses``).
_CLAUSAL_DEPS = frozenset({
    "ccomp", "xcomp", "advcl", "relcl", "acl", "csubj", "csubjpass",
    "parataxis", "pcomp",
})
_WH_TAGS = frozenset({"WDT", "WP", "WP$", "WRB"})

_CLOSING_PUNCT = "\"')]}\u201d\u2019\u00bb"


def _modified_lemma(token: Any, sep: str) -> str:
    modifiers = [c.text for c in token.children if c.dep_ in _MODIFIER_DEPS]
    return sep.join(modifiers + [token.lemma_]).strip()


def _predicate_slots(token: Any) -> Tuple[Any, Any, Any]:
    """Return the (subject, adverbial, object) tokens of one predicate.

    The per-token rules are v1's: subject deps nsubj / nsubjpass /
    csubj, object deps dobj / pobj / attr / acomp, and when a token has
    several children of one kind the last one wins. v1 also counted an
    npadvmod child as a subject; v2 returns it separately as
    ``adverbial`` so that a real or inherited subject takes precedence.
    Strings are built by ``_modified_lemma``: amod / compound modifiers
    '_'-joined for the subject (so multi-word subjects satisfy the
    canonical template's ``\\S+`` subject parser in OuroborosVerifier)
    and space-joined for the object (the canonical object regex is
    ``.+``).
    """
    subject = None
    adverbial = None
    object_ = None
    for child in token.children:
        if child.dep_ in _SUBJECT_DEPS:
            subject = child
        elif child.dep_ == "npadvmod":
            adverbial = child
        elif child.dep_ in _OBJECT_DEPS:
            object_ = child
    return subject, adverbial, object_


def _is_main_clause(token: Any) -> bool:
    """True iff *token* is the sentence ROOT or reaches it through a chain
    of ``conj`` links (coordinated main clauses). Predicates of ccomp,
    xcomp, advcl, relcl, acl and csubj clauses are never main-clause."""
    while token.dep_ == "conj":
        token = token.head
    return bool(token.dep_ == "ROOT")


def _is_passive_predicate(token: Any) -> bool:
    """True iff *token* is a passive predicate. Its nsubjpass is the
    semantic object, so a passive conj never supplies a single-clause
    triple (a ROOT passive goes through ``_extract_passive``)."""
    return any(c.dep_ in ("nsubjpass", "auxpass") for c in token.children)


def _inherited_subject(token: Any) -> Any:
    """Subject token a subjectless conj predicate shares with the
    predicate it is coordinated with ("Alice wrote and published the
    report"), or None.

    Inherited only when a coordinator (CCONJ) precedes the predicate and
    every noun, proper noun or pronoun between them belongs to the
    predicate itself (its own dependents, such as "the next day" in
    "..., and the next day bought a car"). spaCy sometimes attaches the
    second clause's real subject to the first clause's object and leaves
    the predicate subjectless: "Alice founded the company and Bob runs
    the company." or, with an appositive, "Alice founded the company and
    Bob, her brother, bought it." Inheriting there would assert (alice,
    run, company) or (alice, buy, it).
    """
    doc = token.doc
    start = token.sent.start
    i = token.i - 1
    while i >= start and doc[i].pos_ != "CCONJ":
        i -= 1
    if i < start:
        return None
    own = {t.i for t in token.subtree}
    if any(
        doc[j].pos_ in _NOMINAL_POS and j not in own
        for j in range(i + 1, token.i)
    ):
        return None
    while token.dep_ == "conj":
        token = token.head
        subject, _, _ = _predicate_slots(token)
        if subject is not None:
            return subject
    return None


def _within_size(subject: Optional[str], object_: Optional[str]) -> bool:
    return bool(
        subject and object_
        and len(subject.split()) <= 5 and len(object_.split()) <= 8
    )


def _single_clause_triple(sent: Any) -> Tuple[Optional[Tuple[str, str, str]], bool]:
    """Apply the single-clause rule to *sent*.

    Candidate predicates are v1's (the ROOT and every VERB). Only active
    main-clause candidates (``_is_main_clause``, not
    ``_is_passive_predicate``) may supply a triple, and its subject,
    predicate and object all come from that one token. A predicate
    without a subject child may inherit one when it is a conj
    (``_inherited_subject``); otherwise a ROOT predicate falls back to
    its npadvmod as v1 did. When several main-clause predicates are
    complete, the last in token order wins, which is what v1 returned
    for coordinated main clauses.

    Returns ``(triple, stitched)``. ``triple`` is the raw (unlowered,
    unfiltered) main-clause triple or None. ``stitched`` is True only
    when ``triple`` is None but v1's any-verb assembly (last subject and
    last object over all candidates) would have emitted a triple, i.e.
    the sentence would have yielded content taken from outside a single
    main-clause predicate.
    """
    triple = None
    last_subject = None
    last_object = None
    for token in sent:
        if not (token.dep_ == "ROOT" or token.pos_ == "VERB"):
            continue
        subject, adverbial, object_ = _predicate_slots(token)
        if subject is not None or adverbial is not None:
            last_subject = _modified_lemma(subject or adverbial, "_")
        if object_ is not None:
            last_object = _modified_lemma(object_, " ")
        if not _is_main_clause(token) or _is_passive_predicate(token):
            continue
        if subject is None:
            if token.dep_ == "conj":
                subject = _inherited_subject(token)
            else:
                subject = adverbial
        if subject is not None and object_ is not None:
            triple = (
                _modified_lemma(subject, "_"),
                token.lemma_,
                _modified_lemma(object_, " "),
            )
    stitched = triple is None and _within_size(last_subject, last_object)
    return triple, stitched


def _is_question(sent: Any) -> bool:
    """A sentence whose text ends with "?" asks rather than asserts."""
    return bool(sent.text.rstrip().rstrip(_CLOSING_PUNCT).rstrip().endswith("?"))


def _heads_clause(tokens: Any, k: int) -> bool:
    """True iff the marker at ``tokens[k]`` introduces a subordinate
    finite clause: walking up the heads from the next word (skipping an
    optional "that" and the marker itself), the first token with a
    clause subject is not the sentence ROOT."""
    marker = tokens[k]
    j = k + 1
    if j < len(tokens) and tokens[j].lower_ == "that":
        j += 1
    if j >= len(tokens) or tokens[j].pos_ in ("ADP", "PART", "PUNCT"):
        return False
    tok = tokens[j]
    while True:
        if tok.i != marker.i and any(
            c.dep_ in _CLAUSE_SUBJECT_DEPS for c in tok.children
        ):
            return bool(tok.dep_ != "ROOT")
        if tok.head.i == tok.i:
            return False
        tok = tok.head


def _is_inverted_conditional(tokens: Any, k: int) -> bool:
    """True iff had/were/should at ``tokens[k]`` opens a subordinate
    clause with subject-auxiliary inversion. The auxiliary must be the
    first word of a clause that is neither the main clause nor a
    coordinated one, and a noun phrase must follow it."""
    tok = tokens[k]
    if tok.pos_ not in ("AUX", "VERB") or tok.dep_ in ("ROOT", "conj"):
        return False
    clause = tok.head if tok.dep_ in ("aux", "auxpass") else tok
    if clause.dep_ in ("ROOT", "conj") or clause.left_edge.i != tok.i:
        return False
    return k + 1 < len(tokens) and tokens[k + 1].pos_ in _NP_START_POS


def _conditional_phrase_at(tokens: Any, k: int) -> bool:
    """Multiword conditional markers starting at ``tokens[k]``: "as long
    as" / "so long as" (second "as" a clause marker, so comparatives like
    "as long as the table" pass), "in case" (not "in case studies"), "in
    the event that/of", "in the event" directly followed by a clause
    ("In the event the buyer defaults, ..."; spaCy often tags such a
    clause's verb as a noun, so a following determiner, pronoun or
    proper noun also counts), "on condition that"."""
    words = [t.lower_ for t in tokens[k:k + 4]]
    if words[:3] in (["as", "long", "as"], ["so", "long", "as"]):
        return bool(tokens[k + 2].dep_ == "mark")
    if words[:2] == ["in", "case"]:
        return len(words) < 3 or tokens[k + 2].pos_ not in ("NOUN", "PROPN")
    if words[:3] == ["in", "the", "event"]:
        if len(words) < 4:
            return False
        return (
            words[3] in ("that", "of")
            or tokens[k + 3].pos_ in ("DET", "PRON", "PROPN")
            or _heads_clause(tokens, k + 2)
        )
    return words[:3] == ["on", "condition", "that"]


def _opens_clause_here(tokens: Any, k: int, first: Optional[int]) -> bool:
    """True iff ``tokens[k]`` is the sentence's first word, follows a
    comma, or is attached as an adverbial clause (where a hypothesis
    opener such as "assuming" can stand)."""
    tok = tokens[k]
    return bool(
        tok.i == first
        or (k > 0 and tokens[k - 1].text == ",")
        or tok.dep_ == "advcl"
    )


def _is_conditional(sent: Any) -> bool:
    """Return True iff *sent* contains a conditional or hypothetical
    clause, so that neither of its clauses is asserted as fact.

    Detection is matched to en_core_web_sm parses, which are not
    uniform across these constructions (for example "Provided the buyer
    pays ..." parses "Provided" as prep, "Providing the tenant pays ..."
    as csubj, "Were the court to approve ..." makes "Were" the advcl):

      - if / unless / whether / lest / whenever as mark or advmod;
      - "when" or "where" as mark or advmod of an advcl ("Where the
        tenant fails to pay rent, ..."; a relative "where" is not one),
        and "once" as mark ("Once the buyer pays ..."; not "Once a
        farmer, ...");
      - "subject to" as the first word, after a comma or as an
        adverbial clause;
      - "as long as", "so long as", "in case", "in the event (that /
        of)", "on condition that";
      - provided / providing (optionally + "that") heading a clause,
        anywhere; assuming / supposing / suppose and "given that"
        heading a clause as the first word, after a comma or as an
        adverbial clause ("given that" is often causal, "since"; the
        sieve cannot tell, so it does not assert either clause);
      - inverted had / were / should (``_is_inverted_conditional``).
    """
    tokens = list(sent)
    first = next((t.i for t in tokens if not t.is_punct), None)
    for k, tok in enumerate(tokens):
        low = tok.lower_
        if tok.dep_ in ("mark", "advmod"):
            if low in _CONDITIONAL_SUBORDINATORS:
                return True
            if low in ("when", "where") and tok.head.dep_ == "advcl":
                return True
            if low == "once" and tok.dep_ == "mark":
                return True
        if (
            low == "subject"
            and k + 1 < len(tokens)
            and tokens[k + 1].lower_ == "to"
            and _opens_clause_here(tokens, k, first)
        ):
            return True
        if _conditional_phrase_at(tokens, k):
            return True
        if low in _CONDITIONAL_PARTICIPLES and _heads_clause(tokens, k):
            return True
        if (
            (
                low in _HYPOTHESIS_OPENERS
                or (
                    low == "given"
                    and k + 1 < len(tokens)
                    and tokens[k + 1].lower_ == "that"
                )
            )
            and _opens_clause_here(tokens, k, first)
            and (
                _heads_clause(tokens, k)
                # spaCy often tags the hypothesis's verb as a noun object
                # ("Assuming the market recovers" -> dobj "recovers"), so
                # an opener parsed as an adverbial clause counts even
                # without a clause subject. This also withholds the
                # participle "Assuming control of the company, Bob fired
                # the board.", a recall cost taken for precision.
                or (low in _HYPOTHESIS_OPENERS and tok.dep_ == "advcl")
            )
        ):
            return True
        if low in _INVERSION_AUX and _is_inverted_conditional(tokens, k):
            return True
    return False


def _is_predicate(token: Any) -> bool:
    return token.pos_ in ("VERB", "AUX") and token.dep_ not in ("aux", "auxpass")


def _is_clause_disjunction(sent: Any) -> bool:
    """Return True iff *sent* joins clauses with "or", or opens a
    coordination with "either".

    A disjunction asserts neither disjunct ("P or Q" says: if not P,
    then Q), so the clause guard counts it with the conditionals. v2's
    single-clause rule alone would assert the first disjunct: "Alice
    owns the house or Bob owns the car." gave (alice, own, house), and
    "The tenant pays rent or the landlord evicts the tenant." gave
    (tenant, pay, rent).

    "or" counts when a predicate precedes it and a predicate after it is
    a coordinated main clause (a conj that reaches the ROOT) or a ccomp
    of a main clause, which is how en_core_web_sm attaches the second
    clause; spaCy often hangs the "or" itself on the first clause's
    object, so where the "or" attaches is not checked. As a result an
    "or" between noun phrases or inside a relative clause also counts
    when a coordinated main clause follows it: "The firm, which sells
    tea or coffee, opened a shop and hired staff." and "Alice reads
    books or magazines and Bob writes poems." are withheld. That recall
    cost is kept on purpose: requiring the "or" to be the coordinator of
    the two clauses (measured 2026-10-04 on the BillSum sources and the
    repository's markdown) extracted 32 more sentences, 22 of them with
    the "or" inside a subordinate clause, mostly long enumerations the
    parser misreads, with stitched triples such as (woman, contain, such
    other information) and two consequents of conditionals whose marker
    spaCy had split off. Without a later coordinated main clause, an
    "or" inside a relative clause ("The tenant who pays late or skips
    rent loses the deposit.") or between noun phrases ("Alice reads
    books or magazines.") does not count, and the main clause or the
    first member is extracted.
    """
    tokens = list(sent)
    for k, tok in enumerate(tokens):
        low = tok.lower_
        if tok.dep_ == "preconj" and low in _DISJUNCTIVE_PRECONJ:
            return True
        if low not in _DISJUNCTIVE_CC or tok.pos_ != "CCONJ":
            continue
        if not any(_is_predicate(t) for t in tokens[:k]):
            continue
        for t in tokens[k + 1:]:
            if not _is_predicate(t):
                continue
            if t.dep_ == "conj" and _is_main_clause(t):
                return True
            if t.dep_ == "ccomp" and _is_main_clause(t.head):
                return True
    return False


def _has_negative_quantifier(sent: Any) -> bool:
    """True iff *sent* negates an argument: the determiner "no" ("No
    student passed the exam.", "The company paid no dividends.", "No one
    signed ...") or a negative pronoun (``_NEGATIVE_PRONOUNS``: "Nobody
    stole the car.") in any role. The interjection "No, ..." (dep intj)
    does not count. "no longer" and "no doubt" are tagged dep neg by
    en_core_web_sm and were already withheld by ``_is_negated``.

    As with ``_is_negated``, any occurrence counts, also inside a
    subordinate clause; the sieve does not decide the scope of the
    negation, so a sentence it cannot read safely yields nothing.
    """
    doc = sent.doc
    for t in sent:
        low = t.lower_
        if low == "no" and t.dep_ == "det":
            return True
        if low in _NEGATIVE_PRONOUNS or low in ("no-one", "no-body"):
            return True
        # spaCy splits "No-one" into "No" "-" "one", none of them a det
        if (
            low == "no"
            and t.i + 2 < sent.end
            and doc[t.i + 1].text == "-"
            and doc[t.i + 2].lower_ in ("one", "body")
        ):
            return True
    return False


def _is_attributed(sent: Any) -> bool:
    """True iff the main clause of *sent* is reported content.

    Two shapes, both of which v1 and the first v2 read as plain fact:

      - "according to" anywhere ("According to the indictment, the CEO
        stole the pension fund.", "..., according to the indictment.").
        This also withholds the "in accordance with" sense ("The tool
        sorts files according to size."), a recall cost taken for
        precision: the parse does not separate the two.
      - a reporting verb (``_REPORTING_VERBS``) with a reporter (a
        subject, or a "by" agent: "as stated by the ministry"),
        attached to a main-clause predicate as a parenthetical (dep
        parataxis: "The suspect, police said, stole the car.") or as an
        "as" clause ("As the company claims, the drug cures
        migraines.", "..., as police said."). Without a reporter the
        clause is a cross-reference ("as noted above", "as stated in
        section 5"), and "add" needs a subject because "as added by
        section 2" is legal drafting, not a report. A reporting verb
        that is itself the main predicate ("Alice said goodbye.", "Bob
        said that Alice stole the car.") is not matched here; the
        single-clause rule handles it.
    """
    tokens = list(sent)
    for k, t in enumerate(tokens):
        if (
            t.lower_ == "according"
            and k + 1 < len(tokens)
            and tokens[k + 1].lower_ == "to"
        ):
            return True
        lemma = t.lemma_.lower()
        if lemma not in _REPORTING_VERBS or not _is_main_clause(t.head):
            continue
        reporter = ("nsubj", "nsubjpass") if lemma == "add" else (
            "nsubj", "nsubjpass", "agent")
        if not any(c.dep_ in reporter for c in t.children):
            continue
        if t.dep_ == "parataxis":
            return True
        if t.dep_ == "advcl" and any(
            c.dep_ == "mark" and c.lower_ == "as" for c in t.children
        ):
            return True
    return False


def _has_several_clauses(sent: Any) -> bool:
    """True iff *sent* shows more than one clause: a clausal dependency
    (ccomp, xcomp, advcl, relcl, acl, csubj, parataxis, pcomp), a
    wh-word or subordinating conjunction, a non-initial "that", or more
    than one verb. The POS fallback reads three content words left to
    right, so on such a sentence its triple mixes clauses: "I think Bob
    lies." gave (think, bob, lie) and "Dogs that bark bite." gave
    (dogs, bark, bite)."""
    verbs = 0
    first = sent.start
    for t in sent:
        if t.dep_ in _CLAUSAL_DEPS or t.tag_ in _WH_TAGS or t.pos_ == "SCONJ":
            return True
        if t.lower_ == "that" and t.i != first:
            return True
        if t.pos_ == "VERB":
            verbs += 1
    return verbs > 1


def _guard_reason(sent: Any) -> Optional[str]:
    """The suppression reason a whole-sentence guard assigns, or None."""
    if (
        _is_negated(sent)
        or any(t.lower_ in _NEGATIVE_COORDINATORS for t in sent)
        or _has_negative_quantifier(sent)
    ):
        return "negation"
    if _is_question(sent):
        return "question"
    if _is_conditional(sent) or _is_clause_disjunction(sent):
        return "conditional"
    if _is_attributed(sent):
        return "cross_clause"
    return None


def _extract_sentence(sent: Any) -> Tuple[Optional[Tuple[str, str, str]], Optional[str]]:
    """Extract at most one triple from *sent* and say why if suppressed.

    Returns ``(triple, reason)``. ``reason`` is one of
    ``SUPPRESSION_REASONS`` when a guard withheld the sentence, else
    None; ``triple`` is None when nothing was extracted. Order: the
    whole-sentence guards, then passive handling, then the single-clause
    rule, then (only when no clause was stitched) the POS fallback,
    which is withheld on a sentence with several clauses. The noise
    filter (``_is_clean_triple``) is applied by the callers.
    """
    reason = _guard_reason(sent)
    if reason is not None:
        return None, reason

    # Passive voice inverts surface (s,p,o) order. Handle it with a
    # dedicated extractor that swaps the agent phrase's pobj into the
    # subject position and the nsubjpass into the object position. An
    # agentless passive ("The paper was submitted.") cannot recover
    # its semantic subject, so _extract_passive returns None and the
    # sentence is suppressed — the POS fallback is skipped because its
    # left-to-right heuristic would re-emit the inverted triple for
    # three-content-token passives.
    if _is_passive(sent):
        return _extract_passive(sent), None

    triple, stitched = _single_clause_triple(sent)
    if triple is not None:
        subject, predicate, object_ = triple
        if _within_size(subject, object_):
            return (subject.lower(), predicate.lower(), object_.lower()), None
    elif stitched:
        return None, "cross_clause"
    fallback = _pos_fallback_triplet(sent)
    if fallback is not None and _has_several_clauses(sent):
        return None, "cross_clause"
    return fallback, None


def _extract_from_sent(sent: Any) -> Optional[Tuple[str, str, str]]:
    """Extract at most one (subject, predicate, object) triple from a sentence.

    Thin wrapper over ``_extract_sentence`` that drops the suppression
    reason. Every public extraction method goes through
    ``_extract_sentence``, so their outputs stay triple-for-triple
    identical; the provenance and annotated paths only add metadata.
    """
    return _extract_sentence(sent)[0]


# ─── Extractor v1 (frozen) ────────────────────────────────────────────
#
# ``DeterministicSieve(extractor_id=SIEVE_EXTRACTOR_ID_V1)`` runs this
# path. It is the v1 per-sentence extractor as of 0.11.1, kept unchanged
# so that results recorded under ``sum.sieve:deterministic_v1`` (the
# research bench receipts and their pinned digests) can be replayed.
# It has the clause-stitching defects described above; do not use it
# for new attestations.


def _slot_triple_v1(sent: Any) -> Optional[Tuple[str, str, str]]:
    """v1's slot loop: the last subject, predicate and object over every
    ROOT-or-VERB token, lowercased, or None when one is missing or the
    size filters reject it."""
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
    return None


def _extract_from_sent_v1(sent: Any) -> Optional[Tuple[str, str, str]]:
    """Extractor v1: at most one triple per sentence, no clause guard.

    Returns None if the sentence is negated, produces no valid ROOT verb, or
    yields a parse whose subject/object exceed the size filters. The POS
    fallback is consulted only when dependency-based extraction fails.
    """
    if _is_negated(sent):
        return None

    if _is_passive(sent):
        return _extract_passive(sent)

    triple = _slot_triple_v1(sent)
    if triple is not None:
        return triple
    return _pos_fallback_triplet(sent)


def _extract_sentence_v1(sent: Any) -> Tuple[Optional[Tuple[str, str, str]], Optional[str]]:
    """``_extract_sentence`` for extractor v1: only negation is counted."""
    if _is_negated(sent):
        return None, "negation"
    return _extract_from_sent_v1(sent), None


_SENTENCE_EXTRACTORS = {
    SIEVE_EXTRACTOR_ID_V1: _extract_sentence_v1,
    SIEVE_EXTRACTOR_ID: _extract_sentence,
}


# ─── Suppression report ───────────────────────────────────────────────


def _new_report() -> SuppressionReport:
    return {
        "sentences": 0,
        "extracted": 0,
        "suppressed": {reason: 0 for reason in SUPPRESSION_REASONS},
    }


def merge_suppression_reports(
    reports: Iterable[SuppressionReport],
) -> SuppressionReport:
    """Sum suppression reports (e.g. one per chunk or per document) into
    a fresh report of the same shape."""
    total = _new_report()
    for report in reports:
        total["sentences"] += report["sentences"]
        total["extracted"] += report["extracted"]
        for reason in SUPPRESSION_REASONS:
            total["suppressed"][reason] += report["suppressed"].get(reason, 0)
    return total


_REASON_LABELS = {"cross_clause": "cross-clause"}


def format_suppression_notice(
    report: SuppressionReport, label: Optional[str] = None,
) -> Optional[str]:
    """One-line stderr notice for a suppression report, or None when no
    sentence was suppressed.

    ``label`` (e.g. ``"file=<path>"``) is put after ``sum:`` so a caller
    handling several inputs can say which one the line is about. The
    line does not mention a bundle: on a zero-triple exit none is
    written.
    """
    counts = report["suppressed"]
    total = sum(counts.values())
    if not total:
        return None
    n = report["sentences"]
    parts = ", ".join(
        f"{_REASON_LABELS.get(r, r)} {counts[r]}"
        for r in SUPPRESSION_REASONS if counts.get(r)
    )
    prefix = f"sum: {label} " if label else "sum: "
    return (
        f"{prefix}{total} of {n} sentence{'' if n == 1 else 's'} "
        f"{'was' if total == 1 else 'were'} not extracted ({parts}); "
        "see docs/PROOF_BOUNDARY.md section 2.1."
    )


def _write_notice(report: SuppressionReport, stream: Optional[TextIO]) -> None:
    if stream is None:
        return
    line = format_suppression_notice(report)
    if line is not None:
        stream.write(line + "\n")


# ─── Markdown headings (extractor v2) ─────────────────────────────────
#
# spaCy does not end a sentence at a markdown heading, so "## Company\n\n
# The company hire bob." (the canonical tome's layout, see
# AutoregressiveTomeGenerator.generate_canonical) parses as one sentence
# whose ROOT is the heading noun, and the clause guard then withholds
# the axiom as cross-clause. v2 makes each line that begins with "#" one
# sentence of its own (``_mark_heading_breaks``, run before the parser)
# and extracts nothing from it (``_is_heading``): a heading is a title,
# not an assertion, and on its own a three-word heading such as "##
# Bench harness substrate" would feed the POS fallback. Only heading
# lines are split; a general blank-line split was measured to admit
# junk triples from markdown tables and lists. Only the heading
# sentences the component marked are skipped, so a "#" inside a line
# ("Alice owns the car. # Bob ...") is ordinary text, and a "#" line
# indented four or more columns (a CommonMark code block line) is its
# own sentence but is not skipped.

_HEADING_COMPONENT = "sum_markdown_heading_breaks"

# Doc.user_data key: ascending token indices of the heading sentences
# the component marked. A list, not a set: spaCy serialises user_data
# with msgpack (Doc.to_bytes, DocBin, nlp.pipe with n_process > 1),
# which rejects sets. Absent when the text has no heading.
_HEADING_STARTS = "sum_markdown_heading_starts"

# An ATX heading: one to six "#" followed by a space, a tab or the end of
# the line (CommonMark), after at most three spaces of indentation.
# "#1 priority ..." and "#hashtag" are not headings.
_HEADING_RE = re.compile(r"#{1,6}(?:[ \t]|$)")
_MAX_HEADING_INDENT = 3


def _mark_heading_breaks(doc: Any) -> Any:
    """spaCy component: a markdown heading line is exactly one sentence.

    Runs in time linear in the document. ``Doc.text`` rebuilds the whole
    string on every access, so it is read once and sliced; the
    ``Token.is_sent_start`` setter scans the whole document on every
    call, so the boundaries are written once with ``Doc.from_array``.
    """
    text = doc.text
    n = len(doc)
    # (first, end): tokens [first, end) are one heading line, end is the
    # token after its newline (or n)
    lines: List[Tuple[int, int]] = []

    def hash_line(first: int, end_char: int, indent: str) -> Optional[bool]:
        # indent: the line's leading whitespace carried by the preceding
        # newline token; a leading space token at the start of the text
        # is part of the line itself. Returns None for a line that does
        # not start with "#", True for a heading, False for a "#" line
        # indented four or more columns (a code block line).
        line = text[doc[first].idx:end_char]
        body = line.lstrip(" \t")
        indent += line[:len(line) - len(body)]
        if indent.strip(" \t") or not _HEADING_RE.match(body):
            return None
        return "\t" not in indent and len(indent) <= _MAX_HEADING_INDENT

    # (first, end, heading): every "#" line is its own sentence, so a
    # code block line cannot be glued to the prose around it and stitch
    # a triple; only headings are skipped
    marked: List[Tuple[int, int, bool]] = []
    line_start = 0
    indent = ""
    for token in doc:
        if token.is_space and "\n" in token.text:
            if line_start < token.i:
                kind = hash_line(line_start, token.idx, indent)
                if kind is not None:
                    marked.append((line_start, token.i + 1, kind))
            line_start = token.i + 1
            indent = token.text[token.text.rfind("\n") + 1:]
    if line_start < n:
        kind = hash_line(line_start, len(text), indent)
        if kind is not None:
            marked.append((line_start, n, kind))
    if marked:
        # SENT_START: 1 starts a sentence, -1 continues one, 0 unset
        values = doc.to_array("SENT_START").astype("int64")
        for first, end, _ in marked:
            if first > 0:
                values[first] = 1
            values[first + 1:min(end, n)] = -1
            if end < n:
                values[end] = 1
        doc.from_array(["SENT_START"], values.astype("uint64"))
    lines = [(first, end) for first, end, heading in marked if heading]
    if lines:
        doc.user_data[_HEADING_STARTS] = sorted(first for first, _ in lines)
    else:
        doc.user_data.pop(_HEADING_STARTS, None)
    return doc


def _is_heading(sent: Any) -> bool:
    """True iff *sent* is a markdown heading line that
    ``_mark_heading_breaks`` marked as its own sentence."""
    starts = sent.doc.user_data.get(_HEADING_STARTS, ())
    i = bisect.bisect_left(starts, sent.start)
    return i < len(starts) and starts[i] == sent.start


def _add_heading_breaks(nlp: Any) -> None:
    from spacy.language import Language

    if not Language.has_factory(_HEADING_COMPONENT):
        Language.component(_HEADING_COMPONENT, func=_mark_heading_breaks)
    if _HEADING_COMPONENT not in nlp.pipe_names:
        nlp.add_pipe(_HEADING_COMPONENT, before="parser")


def _pos_fallback_triplet(sent: Any) -> Optional[Tuple[str, str, str]]:
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


class SieveUnavailableError(RuntimeError):
    """The local extractor or its language model is not installed."""


class DeterministicSieve:
    """
    High-Fidelity Edge NLP.

    Extracts topological (Subject, Predicate, Object) triplets using
    strict grammatical dependency parsing.

    Cost: $0. Speed: 10,000+ words per second.

    ``extractor_id`` selects the extraction behaviour and is recorded in
    every ProvenanceRecord. The default is the current extractor
    (``SIEVE_EXTRACTOR_ID``, v2 with the clause guard).
    ``SIEVE_EXTRACTOR_ID_V1`` replays the frozen v1 extractor, so that
    results recorded under it (the research bench receipts) reproduce;
    it has v1's clause-stitching defects and is not for new attestations.
    """

    def __init__(
        self,
        *,
        allow_download: bool = True,
        extractor_id: str = SIEVE_EXTRACTOR_ID,
    ):
        if extractor_id not in _SENTENCE_EXTRACTORS:
            raise ValueError(
                f"unknown sieve extractor_id {extractor_id!r}; expected one "
                f"of {sorted(_SENTENCE_EXTRACTORS)}"
            )
        self.extractor_id = extractor_id
        self._sentence_extractor = _SENTENCE_EXTRACTORS[extractor_id]
        try:
            import spacy  # Lazy import: only required when sieve is instantiated
        except ImportError as exc:
            raise SieveUnavailableError(
                "Install sum-engine[sieve] and run: python -m spacy download en_core_web_sm"
            ) from exc

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            if not allow_download:
                raise SieveUnavailableError(
                    "spaCy model en_core_web_sm is missing; automatic downloads are disabled. "
                    "Install it before starting the server: python -m spacy download en_core_web_sm"
                ) from exc
            import subprocess
            import sys

            # CRITICAL: route spaCy's download progress to stderr so it does
            # not contaminate the CLI's stdout. `sum attest > bundle.json`
            # must emit nothing but the CanonicalBundle JSON; the CI's
            # pip-install smoke test catches this regression. Announcing the
            # fallback on stderr is also more honest than silent auto-install.
            print(
                "sum: spaCy model 'en_core_web_sm' missing; downloading "
                "(~50 MB, one-time)…",
                file=sys.stderr,
            )
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=sys.stderr,
            )
            self.nlp = spacy.load("en_core_web_sm")
        self._skip_headings = extractor_id != SIEVE_EXTRACTOR_ID_V1
        if self._skip_headings:
            _add_heading_breaks(self.nlp)

    def _extract_sentences(
        self, text: str,
    ) -> Tuple[List[Tuple[Any, Tuple[str, str, str]]], SuppressionReport]:
        """Run the shared per-sentence extraction over *text*.

        Returns ``(kept, report)``: ``kept`` lists ``(sent, triple)`` for
        every sentence that yielded a clean triple, in document order;
        ``report`` is a fresh suppression report (see
        ``extract_triplets_with_report``). Nothing is cached on the
        instance, so concurrent calls do not share counts.
        """
        doc = self.nlp(text)
        report = _new_report()
        kept: List[Tuple[Any, Tuple[str, str, str]]] = []
        for sent in doc.sents:
            if self._skip_headings and _is_heading(sent):
                continue
            report["sentences"] += 1
            triple, reason = self._sentence_extractor(sent)
            if reason is not None:
                report["suppressed"][reason] += 1
                continue
            if triple is None or not _is_clean_triple(triple):
                continue
            report["extracted"] += 1
            kept.append((sent, triple))
        return kept, report

    def extract_triplets(
        self, text: str, *, suppressed_notice: Optional[TextIO] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Parse text into semantic triplets using dependency grammar.

        Each sentence yields at most one triple, taken from a single
        main-clause predicate; negated, question, conditional and
        cross-clause sentences are suppressed (see the "Clause guard"
        section of this module).

        Triples whose components contain markdown/code/table syntactic
        noise (pipe characters, single-character punctuation, link
        residue, path-like substrings) are dropped at this boundary.
        See ``_is_noise_component`` for the filter rules.

        Args:
            text: Raw text to parse.
            suppressed_notice: Optional text stream. When given and at
                least one sentence was suppressed, one summary line
                (``format_suppression_notice``) is written to it.

        Returns:
            Deduplicated list of clean (subject, predicate, object) tuples.
        """
        triplets, report = self.extract_triplets_with_report(text)
        _write_notice(report, suppressed_notice)
        return triplets

    def extract_triplets_with_report(
        self, text: str,
    ) -> Tuple[List[Tuple[str, str, str]], SuppressionReport]:
        """``extract_triplets`` plus a per-call suppression report.

        Returns ``(triples, report)`` where ``triples`` equals
        ``extract_triplets(text)`` and::

            report = {
                "sentences": N,   # sentences spaCy segmented, without
                                  # markdown heading lines (v2)
                "extracted": K,   # sentences that yielded a clean triple
                "suppressed": {"negation": a, "conditional": b,
                               "question": c, "cross_clause": d},
            }

        A sentence is counted under at most one reason. Sentences that
        are neither extracted nor suppressed had no extractable triple
        (no subject-verb-object, an agentless passive, or noise).
        """
        kept, report = self._extract_sentences(text)
        # Deduplicate AND sort lexicographically. The sort is load-bearing
        # for cross-invocation reproducibility: bare `set(triplets)` returns
        # a set whose iteration order depends on Python's hash randomization
        # (PYTHONHASHSEED-dependent), which then propagates through
        # KnowledgeSheafV2.from_triples → trained vertex order → bench AUCs.
        # Sorting on the (subject, predicate, object) tuple gives stable
        # cross-process ordering and lets bench_digest values reproduce
        # without environment-variable manipulation.
        return sorted(set(triple for _, triple in kept)), report

    def extract_with_provenance(
        self,
        text: str,
        source_uri: Optional[str] = None,
        timestamp: Optional[str] = None,
        *,
        suppressed_notice: Optional[TextIO] = None,
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
            suppressed_notice: Optional text stream; as in
                         ``extract_triplets``.

        Returns:
            List of ``((s, p, o), ProvenanceRecord)`` pairs — NOT deduplicated
            at the triple level. Two sentences producing the same triple yield
            two records with different byte ranges and different prov_ids.
            The AkashicLedger is the dedup boundary, not this method.
        """
        src = source_uri or sha256_uri_for_text(text)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        kept, report = self._extract_sentences(text)
        out: List[Tuple[Tuple[str, str, str], ProvenanceRecord]] = []
        for sent, triple in kept:
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
                extractor_id=self.extractor_id,
                timestamp=ts,
                text_excerpt=excerpt,
            )
            out.append((triple, record))
        _write_notice(report, suppressed_notice)
        return out

    def extract_annotated_triplets(
        self, text: str
    ) -> List[Dict[str, object]]:
        """Extract triplets with per-sentence hedging annotation.

        Returns a list of dicts, one per sentence that yielded a triple
        (not deduplicated):
            {
                "subject": str,
                "predicate": str,
                "object": str,
                "linguistic_certainty": float,  # 1.0 = definite, <1.0 = hedged
            }

        Uses the same per-sentence extraction as ``extract_triplets``
        (clause guard, passive handling, POS fallback, noise filter), so
        the triples are the same ones in document order. Up to 0.11.1
        this method ran its own copy of the v1 slot loop without passive
        handling, the POS fallback or the noise filter; under
        ``SIEVE_EXTRACTOR_ID_V1`` it still does, so v1 annotated output
        replays 0.11.1 exactly.

        The linguistic_certainty score is a metadata-only signal
        that does NOT affect the Gödel algebra.
        """
        if self.extractor_id == SIEVE_EXTRACTOR_ID_V1:
            return [
                {
                    "subject": triple[0],
                    "predicate": triple[1],
                    "object": triple[2],
                    "linguistic_certainty": detect_hedging(sent.text),
                }
                for sent in self.nlp(text).sents
                if not _is_negated(sent)
                for triple in [_slot_triple_v1(sent)]
                if triple is not None
            ]
        kept, _ = self._extract_sentences(text)
        return [
            {
                "subject": triple[0],
                "predicate": triple[1],
                "object": triple[2],
                "linguistic_certainty": detect_hedging(sent.text),
            }
            for sent, triple in kept
        ]
