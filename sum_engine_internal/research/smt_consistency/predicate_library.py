"""Operator-curated predicate library for Z3 consistency checking.

Tuned to the predicates the deterministic sieve actually produces
from the substrate's labeled corpora (verb-lemma vocabulary).
Curation discipline (operator-vetted, conservative):

  - **IRREFLEXIVE** declared whenever ``X p X`` is plainly
    malformed for the predicate (most action verbs)
  - **ANTISYMMETRIC** declared only when mutual-application
    would constitute a clear contradiction in plain prose
    (avoiding metaphorical edge cases like ``know``)
  - **TRANSITIVE** declared only for spatial / containment-style
    relations where the implication holds rigorously
  - **FUNCTIONAL** declared for predicates with a single-valued
    semantic (one birth date, one origin)

Predicates with context-dependent semantics (``be``, ``know``,
``confirm``, ``describe``) are intentionally omitted — leaving
them unconstrained is the conservative choice.

Method history: surveyed 7 labeled corpora (189 distinct
predicates); curated declarations for top-frequency verb lemmas.
See ``docs/SMT_CONSISTENCY_SPIKE_FINDINGS.md`` iteration 2 for
the full curation rationale + injection-test evidence.
"""
from __future__ import annotations

from sum_engine_internal.research.smt_consistency.consistency import (
    PredicateProperty as P,
)


SUBSTRATE_PREDICATE_LIBRARY: dict[str, set[P]] = {
    # Containment / spatial: contain is the textbook anti-sym + irref + trans
    "contain": {P.ANTISYMMETRIC, P.IRREFLEXIVE, P.TRANSITIVE},
    "cover":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},

    # Production / authorship — antisymmetric (no mutual production
    # / authoring in normal prose) and irreflexive (X doesn't
    # build / write / produce X)
    "build":     {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "write":     {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "produce":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "develop":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "establish": {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "create":    {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "compose":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},

    # Action / discovery — same logic
    "discover":  {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "release":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "execute":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "open":      {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "introduce": {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "propose":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "capture":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},

    # Receipt / transfer — antisymmetric (mutual receipt is
    # ungrammatical) + irreflexive
    "receive": {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "take":    {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "emit":    {P.ANTISYMMETRIC, P.IRREFLEXIVE},

    # Achievement
    "win":   {P.ANTISYMMETRIC, P.IRREFLEXIVE},
    "reach": {P.IRREFLEXIVE},  # antisym too aggressive for metaphorical use

    # State change — irreflexive only (became / began are not strictly antisymmetric)
    "become": {P.IRREFLEXIVE},
    "begin":  {P.IRREFLEXIVE},

    # Functional single-valued attributes (illustrative; sieve
    # rarely produces these from prose, but kept for the
    # predicate-library shape so future operators can extend)
    "born_on": {P.FUNCTIONAL},
    "born_in": {P.FUNCTIONAL},
}
