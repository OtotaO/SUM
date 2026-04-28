"""§2.5 generator-side interventions — second receipt cycle.

The L0–L3 canonicalisation-replay receipt
(`fixtures/bench_receipts/s25_canonicalization_replay_2026-04-28.json`)
established that **canonicalisation alone cannot close the §2.5 gap**.
The dominant failure mode is generator elaboration — the LLM produces
~12 reconstructed axioms per source axiom and elaborates *around* the
source claim rather than paraphrasing it. There is nothing for
predicate normalisation to canonicalise *to*.

This module ships the two generator-side interventions named in the
prior receipt's "operational read":

    (A) Canonical-first generator prompt — prompts the generator to
        surface each source claim verbatim before elaborating. Pure
        prompt change; same API surface; addresses the elaboration
        failure mode at the source rather than the symptom.

    (B) Constrained-decoding extractor — per-doc Pydantic schema with
        ``Literal`` enums pinned to the source-axiom vocabulary.
        Forces the second-pass extractor to emit triples whose
        subject/predicate/object are drawn only from the source
        vocabulary. Surfaces a different failure mode: if the
        constrained extractor returns empty for a doc whose prose
        does not reference the source claim, that is the receipt
        — the prose elaborated past the source claim and the
        constraint cannot manufacture a match that does not exist.

Both interventions ship under sibling receipt schemas:
    sum.s25_canonical_first_generator.v1
    sum.s25_constrained_extractor.v1
    sum.s25_combined.v1            (A + B)

Pure functions where possible; LLM call surfaces gated behind
``LiveLLMAdapter`` so the offline scaffold is testable without API
spend.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, List, Literal, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


Triple = Tuple[str, str, str]


# ---------------------------------------------------------------------
# Intervention A — canonical-first generator prompt
# ---------------------------------------------------------------------


CANONICAL_FIRST_SYS_PROMPT = (
    "You are a precise technical writer. Your job is to extrapolate "
    "facts into a cohesive narrative WITHOUT losing the verbatim "
    "surface form of the source facts.\n\n"
    "Process:\n"
    "1. The user will give you a list of facts as 'subject predicate "
    "   object' triples.\n"
    "2. In your output, surface each source fact in its canonical "
    "   form FIRST (a sentence shaped 'The {subject} {predicate} "
    "   {object}.' for each one, using the EXACT subject and predicate "
    "   tokens from the input).\n"
    "3. Then, elaborate around those canonical sentences with "
    "   contextual prose. The elaboration MUST NOT replace the "
    "   canonical sentences.\n\n"
    "An output that elaborates beautifully but loses the canonical "
    "form of even one source fact is a FAILED render. Reproduce the "
    "tokens; reproduce the order; then elaborate. Do not invent facts."
)


def build_canonical_first_user_prompt(
    target_axioms: List[str],
    negative_constraints: Optional[List[str]] = None,
) -> str:
    """Construct the user-side prompt for intervention (A).

    Pure function — no API call. Testable offline.
    """
    parts = [f"FACTS TO INCLUDE:\n{chr(10).join(target_axioms)}\n"]
    parts.append(
        "REQUIRED OUTPUT SHAPE:\n"
        "First, write one canonical sentence per fact in the form "
        "'The {subject} {predicate} {object}.' using the EXACT tokens "
        "above. Then, optionally, elaborate. The canonical sentences "
        "MUST appear verbatim in the output."
    )
    if negative_constraints:
        parts.append(
            "CRITICAL NEGATIVE CONSTRAINTS "
            "(DO NOT INCLUDE THESE HALLUCINATIONS):\n"
            f"{chr(10).join(negative_constraints)}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------
# Intervention B — constrained-decoding extractor
# ---------------------------------------------------------------------


# Closed predicate vocabulary the extractor may emit when no source
# predicate fits. Kept small + canonical so future receipts can pin to
# the same set. Matches predicates likely to appear in canonical
# `seed_v1`-style axioms.
DEFAULT_CANONICAL_PREDICATES: tuple[str, ...] = (
    "be", "have", "do", "say", "make", "go", "see", "know", "take",
    "get", "give", "find", "think", "tell", "become", "show", "leave",
    "feel", "put", "bring", "begin", "keep", "hold", "write", "stand",
    "hear", "let", "mean", "set", "meet", "pay", "sit", "speak", "lie",
    "lead", "read", "grow", "lose", "fall", "send", "build", "understand",
    "draw", "break", "spend", "cut", "rise", "drive", "buy", "wear",
    "choose", "include", "discover", "invent", "describe", "explain",
    "propose", "develop", "create", "design", "study", "research",
    "analyse", "measure", "observe", "predict", "verify", "prove",
    "contain", "produce", "consist", "comprise", "involve", "enable",
    "cause", "affect", "influence", "depend", "result", "lead_to",
    "consist_of", "is_a", "part_of", "type_of", "owns", "uses",
)


def _candidate_lemmas(predicate: str) -> set[str]:
    """Cheap inflection→lemma candidates for English predicates.

    Returns every plausible lemma of ``predicate`` so the
    constrained-schema builder can refuse to put both forms in the
    same enum. Empirically derived from the §2.5 residual receipt:
    every failing doc had a source predicate whose lemma sat in
    DEFAULT_CANONICAL_PREDICATES, so the LLM extractor (offered
    both forms in the enum) chose the lemma every time.

    Conservative — only fires on standard English suffixes. Will
    miss irregulars (`taught` → `teach`); those are not present in
    the corpus's failure set, so this is the right scope for the
    fix that closes the observed residual.
    """
    p = predicate.lower()
    out: set[str] = set()
    if p.endswith("ed") and len(p) > 3:
        out.add(p[:-2])             # "proposed" → "propose"
        out.add(p[:-1])             # "proposed" → "proposed" (no-op for past)
        # Doubled-consonant: "stopped" → "stop"
        if len(p) >= 4 and p[-3] == p[-4] and p[-3] in "bdfgklmnprt":
            out.add(p[:-3])
    if p.endswith("ies") and len(p) > 4:
        out.add(p[:-3] + "y")       # "carries" → "carry"
    elif p.endswith("es") and len(p) > 3:
        out.add(p[:-2])             # "proposes" → "propose"
    if p.endswith("s") and len(p) > 2 and not p.endswith("ss"):
        out.add(p[:-1])             # "contains" → "contain"
    if p.endswith("ing") and len(p) > 4:
        out.add(p[:-3])             # "running" → "runn" (then dedupe handles)
        out.add(p[:-3] + "e")       # "writing" → "write"
        if len(p) >= 5 and p[-4] == p[-5] and p[-4] in "bdfgklmnprt":
            out.add(p[:-4])         # "stopping" → "stop"
    # Compound predicates ("build_nests"): also forbid the head verb
    # in isolation so the LLM cannot drop the modifier.
    if "_" in p:
        out.add(p.split("_", 1)[0])
    return out


def build_constrained_extraction_schema(
    source_axioms: Sequence[Triple],
    extra_predicates: Sequence[str] = DEFAULT_CANONICAL_PREDICATES,
    schema_name: str = "ConstrainedExtractionResponse",
) -> Type[BaseModel]:
    """Build a Pydantic schema for OpenAI structured-output that pins
    the extractor's vocabulary to the source-axiom vocabulary.

    The schema constrains:
        subject   ∈ {s for (s, _, _) in source_axioms}
        predicate ∈ {p for (_, p, _) in source_axioms}
                    ∪ (extra_predicates − lemmas-of-source-predicates)
        object    ∈ {o for (_, _, o) in source_axioms}

    The lemma-exclusion (added 2026-04-28 after the §2.5 closure
    receipt's residual analysis) prevents the failure mode where
    ``proposed`` (source) and ``propose`` (canonical padding) both
    sat in the enum and the LLM extractor chose the lemma every
    time. With the lemma removed from the canonical padding set,
    the LLM has only the source surface form to choose from.

    Empty source axioms produce a schema that allows no triples (the
    only valid response is ``triplets: []``).
    """
    subjects: list[str] = sorted({s for (s, _, _) in source_axioms})
    objects: list[str] = sorted({o for (_, _, o) in source_axioms})

    src_predicates: set[str] = {p for (_, p, _) in source_axioms}

    # Exclude lemmas of source predicates from the canonical-padding
    # set so the LLM extractor cannot collapse a source predicate to
    # its lemma when both forms are otherwise available.
    forbidden_lemmas: set[str] = set()
    for sp in src_predicates:
        forbidden_lemmas |= _candidate_lemmas(sp)
    extra_filtered: set[str] = set(extra_predicates) - forbidden_lemmas

    predicates: list[str] = sorted(src_predicates | extra_filtered)

    if not subjects or not objects:
        # Build a fully-constrained schema where any non-empty triple
        # fails. The model can still emit an empty `triplets` list.
        # We use a sentinel literal that no real LLM will emit.
        subjects = ["__NO_SUBJECT_PERMITTED__"]
        objects = ["__NO_OBJECT_PERMITTED__"]

    SubjectLiteral: Any = Literal[tuple(subjects)] if len(subjects) > 1 else Literal[subjects[0]]
    PredicateLiteral: Any = Literal[tuple(predicates)] if len(predicates) > 1 else Literal[predicates[0]]
    ObjectLiteral: Any = Literal[tuple(objects)] if len(objects) > 1 else Literal[objects[0]]

    ConstrainedTriplet = create_model(
        f"{schema_name}__Triplet",
        subject=(SubjectLiteral, Field(description="Subject pinned to source vocabulary")),
        predicate=(PredicateLiteral, Field(description="Predicate pinned to source ∪ canonical vocabulary")),
        object_=(ObjectLiteral, Field(alias="object", description="Object pinned to source vocabulary")),
    )

    Response = create_model(
        schema_name,
        triplets=(List[ConstrainedTriplet], Field(default_factory=list)),  # type: ignore[arg-type]
    )

    return Response


def vocabulary_summary(source_axioms: Sequence[Triple]) -> dict[str, int]:
    """Quick stats over a per-doc constrained vocabulary.

    Used by the runner to log how aggressive the constraint is per doc
    (e.g. a doc with 1 source axiom pins subjects to 1 token). Pure
    function.
    """
    subjects = {s for (s, _, _) in source_axioms}
    predicates = {p for (_, p, _) in source_axioms}
    objects = {o for (_, _, o) in source_axioms}
    return {
        "n_source_axioms": len(list(source_axioms)),
        "n_unique_subjects": len(subjects),
        "n_unique_predicates": len(predicates),
        "n_unique_objects": len(objects),
        "n_predicates_with_canonical_padding": len(predicates) + len(DEFAULT_CANONICAL_PREDICATES),
    }


# ---------------------------------------------------------------------
# Receipt schema names — pinned at module scope so the runner and
# downstream readers reference the same identifiers.
# ---------------------------------------------------------------------

RECEIPT_SCHEMA_CANONICAL_FIRST = "sum.s25_canonical_first_generator.v1"
RECEIPT_SCHEMA_CONSTRAINED_EXTRACTOR = "sum.s25_constrained_extractor.v1"
RECEIPT_SCHEMA_COMBINED = "sum.s25_combined.v1"
