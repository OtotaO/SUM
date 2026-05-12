"""extract — tags-from-tome transform.

The dream's bi-directional fulfillment: the slider compresses tomes via
the density axis; ``extract`` is the inverse direction — take a piece
of text (or a bundle) and yield the canonical tag set as a *named*
output, with a signed receipt.

T2 v0 scope:
  - Single extractor: ``sieve`` (DeterministicSieve from
    ``sum_engine_internal.algorithms.syntactic_sieve``).
  - Input: text string OR ``CanonicalBundle``-shaped dict (with
    ``triples`` already extracted, in which case the transform is the
    identity over the triple set).
  - Output: sorted unique tag set, JCS-canonicalised.

T6 (this revision) — multi-school mode:
  - ``multi_school=True`` runs N extractors in tandem, unions their
    outputs, and tags each output triple with the set of extractors
    that produced it. Closes the dream's "multiple schools of
    categorization in tandem" element.
  - Two extractors registered for v0: ``sieve`` (DeterministicSieve)
    and ``naive`` (the naive token-pair extractor implemented in
    this module — no external deps, deterministic, demonstrably
    different from the sieve's dependency-grammar approach).
  - Output shape in multi_school mode:
        [{"triple": ["s","p","o"], "extractors": ["sieve", "naive"]}, ...]
    sorted by triple lex order so the output_hash is byte-stable.

Deferred still:
  - LLM extractor (requires ``[openai]`` or ``[anthropic]`` extra).
  - Wikidata QID resolver.
  - Side-by-side UI in single_file_demo (T6b).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.transforms._base import (
    DigitalSourceType,
    TransformEnv,
    TransformResult,
)


ExtractorChoice = Literal["sieve", "llm", "auto"]


_VALID_EXTRACTORS = ("sieve", "llm", "auto")


def _validate_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Validate + fill defaults. Returns a normalised parameters dict
    used both by canonicalize_parameters() (for the receipt hash) and
    by apply() (for runtime behaviour). Same normalisation in both
    so the hash matches what actually ran."""
    extractor = params.get("extractor", "sieve")
    if extractor not in _VALID_EXTRACTORS:
        raise ValueError(
            f"extract: parameter 'extractor' must be one of "
            f"{_VALID_EXTRACTORS!r}; got {extractor!r}"
        )
    max_tags = params.get("max_tags", 64)
    if not isinstance(max_tags, int) or max_tags < 1 or max_tags > 1024:
        raise ValueError(
            f"extract: parameter 'max_tags' must be an int in "
            f"[1, 1024]; got {max_tags!r}"
        )
    multi_school = bool(params.get("multi_school", False))
    return {
        "extractor": extractor,
        "max_tags": max_tags,
        "multi_school": multi_school,
    }


def _normalize_triple(triple: tuple[str, str, str]) -> tuple[str, str, str]:
    """Tag-set normalisation: lowercase + strip per component.
    Mirrors the rule in LiveLLMAdapter.extract_triplets so sieve-
    and LLM-extracted triples produce comparable tag sets."""
    return tuple(c.strip().lower() for c in triple)  # type: ignore[return-value]


def _unique_sorted(triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """De-dup + componentwise-sort. The output_hash binds to THIS
    canonical shape; two callers with different insertion order
    produce byte-identical tag sets and therefore byte-identical
    output_hash."""
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for t in triples:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return sorted(out)


@dataclass
class ExtractTransform:
    """Tags-from-tome transform.

    Closes the dream's bi-directional gap: the slider compresses
    tomes; this transforms in the reverse direction (text → tags).
    """
    name: str = "extract"
    requires_llm: bool = False  # sieve path is no-LLM; llm path flips at apply()
    digital_source_type: DigitalSourceType = "algorithmicMedia"

    def canonicalize_parameters(self, params: dict[str, Any]) -> bytes:
        return canonicalize(_validate_parameters(params))

    def canonicalize_input(self, raw_input: Any) -> bytes:
        """Two input shapes accepted:

        1. ``{"text": "..."}`` — raw prose; the transform will extract
           triples from it.
        2. ``{"triples": [...]}`` — already-extracted triples; the
           transform is the identity over the triple set
           (useful for chaining: slider output → extract input).

        Canonical bytes: for text input, UTF-8 of the text; for
        triples input, JCS of the componentwise-sorted normalised
        triple list. Two inputs that produce the same OUTPUT canonical
        bytes must produce the same input_hash, so we canonicalise on
        the post-normalisation shape.
        """
        if not isinstance(raw_input, dict):
            raise ValueError(
                f"extract input: expected dict, got {type(raw_input).__name__}"
            )
        if "text" in raw_input:
            text = raw_input["text"]
            if not isinstance(text, str):
                raise ValueError("extract input: 'text' must be a string")
            return text.encode("utf-8")
        if "triples" in raw_input:
            triples = [
                _normalize_triple((str(t[0]), str(t[1]), str(t[2])))
                for t in raw_input["triples"]
            ]
            return canonicalize(_unique_sorted(triples))
        raise ValueError(
            "extract input: expected dict with 'text' or 'triples' key"
        )

    def canonicalize_output(self, output: Any) -> bytes:
        """Output is one of two shapes:

        1. Single-school (default): list of normalised triples.
           Canonical bytes = JCS of componentwise-sorted unique
           tuple list.

        2. Multi-school: list of ``{"triple": [s,p,o], "extractors":
           [...]}`` dicts. Canonical bytes = JCS of the list sorted
           by triple lex order; each dict's `extractors` field is
           also sorted alphabetically.

        The two shapes produce DIFFERENT output_hash values for the
        same underlying triple set — `multi_school` is a distinct
        output, not just metadata.
        """
        if not isinstance(output, list):
            raise ValueError(
                f"extract output: expected list, got "
                f"{type(output).__name__}"
            )
        if not output:
            return canonicalize([])
        # Detect shape from the first element.
        first = output[0]
        if isinstance(first, dict) and "triple" in first and "extractors" in first:
            # Multi-school shape: list of {triple, extractors}.
            entries = []
            for entry in output:
                t = entry["triple"]
                triple = _normalize_triple(
                    (str(t[0]), str(t[1]), str(t[2]))
                )
                extractors = sorted({str(e) for e in entry["extractors"]})
                entries.append({
                    "triple": list(triple),
                    "extractors": extractors,
                })
            entries.sort(key=lambda e: (e["triple"], e["extractors"]))
            return canonicalize(entries)
        # Single-school shape: plain list of triples.
        triples = [
            _normalize_triple((str(t[0]), str(t[1]), str(t[2])))
            for t in output
        ]
        return canonicalize(_unique_sorted(triples))

    async def apply(
        self,
        input: Any,
        parameters: dict[str, Any],
        env: TransformEnv,
    ) -> TransformResult:
        norm = _validate_parameters(parameters)
        extractor = norm["extractor"]
        max_tags = norm["max_tags"]
        multi_school = norm["multi_school"]

        if multi_school:
            # T6: run all registered extractors in tandem; union the
            # outputs; tag each triple with the set of extractors
            # that produced it.
            return await self._apply_multi_school(input, max_tags)

        if not isinstance(input, dict):
            raise ValueError(
                f"extract input: expected dict, got {type(input).__name__}"
            )

        # Identity path: input already has triples, just normalise + return.
        if "triples" in input:
            triples = [
                _normalize_triple((str(t[0]), str(t[1]), str(t[2])))
                for t in input["triples"]
            ]
            tag_set = _unique_sorted(triples)[:max_tags]
            return TransformResult(
                output=[list(t) for t in tag_set],
                model="canonical-deterministic-v0",
                provider="canonical-path",
                digital_source_type="algorithmicMedia",
                llm_calls_made=0,
                extra={"extractor_used": "identity", "tag_count": len(tag_set)},
            )

        # Text path: extract via the chosen extractor.
        if "text" not in input:
            raise ValueError(
                "extract input: expected 'text' or 'triples' key"
            )
        text = input["text"]
        if not isinstance(text, str):
            raise ValueError("extract input: 'text' must be a string")

        if extractor == "llm" or (extractor == "auto" and (
            env.anthropic_api_key or env.openai_api_key
        )):
            # LLM extraction path — deferred; the legacy
            # LiveLLMAdapter.extract_triplets is the supported surface.
            # When this lands, it goes here.
            raise NotImplementedError(
                "extract transform v0 (T2) supports the sieve extractor "
                "only. extractor='llm' (and extractor='auto' with an LLM "
                "key configured) is deferred — use "
                "sum_engine_internal.ensemble.live_llm_adapter."
                "LiveLLMAdapter.extract_triplets in the meantime, or pass "
                "extractor='sieve' explicitly to use the deterministic "
                "spaCy-based path."
            )

        # Default: sieve (no-LLM, deterministic).
        try:
            from sum_engine_internal.algorithms.syntactic_sieve import (
                DeterministicSieve,
            )
        except ImportError as e:
            raise ImportError(
                "extract transform's sieve extractor requires the [sieve] "
                "extra. Install: pip install 'sum-engine[sieve]'"
            ) from e

        sieve = DeterministicSieve()
        raw_triples = sieve.extract_triplets(text)
        normalised = [_normalize_triple(t) for t in raw_triples]
        tag_set = _unique_sorted(normalised)[:max_tags]

        return TransformResult(
            output=[list(t) for t in tag_set],
            model="canonical-deterministic-v0",
            provider="canonical-path",
            digital_source_type="algorithmicMedia",
            llm_calls_made=0,
            extra={
                "extractor_used": "sieve",
                "tag_count": len(tag_set),
                "raw_count": len(raw_triples),
            },
        )


    async def _apply_multi_school(
        self,
        input: Any,
        max_tags: int,
    ) -> TransformResult:
        """Run all registered extractors in tandem; union the
        outputs; tag each triple with the set of extractors that
        produced it.

        Two extractors registered for v0:
          - ``sieve`` (DeterministicSieve; requires [sieve] extra)
          - ``naive`` (naive_token_pair_extract; always available)

        Sieve is skipped (with a marker entry) if spaCy / the
        en_core_web_sm model isn't available. Naive always runs.
        """
        if not isinstance(input, dict) or "text" not in input:
            raise ValueError(
                "extract input in multi_school mode: expected dict with "
                "'text' key (multi-school operates on raw text)"
            )
        text = input["text"]
        if not isinstance(text, str):
            raise ValueError("extract input: 'text' must be a string")

        by_extractor: dict[str, list[tuple[str, str, str]]] = {}

        # Naive — always available.
        naive_triples = naive_token_pair_extract(text)
        by_extractor["naive"] = [_normalize_triple(t) for t in naive_triples]

        # Sieve — skipped if spaCy/model unavailable. The receipt
        # records which extractors actually ran via the extra field.
        sieve_ran = False
        try:
            from sum_engine_internal.algorithms.syntactic_sieve import (
                DeterministicSieve,
            )
            sieve = DeterministicSieve()
            raw = sieve.extract_triplets(text)
            by_extractor["sieve"] = [_normalize_triple(t) for t in raw]
            sieve_ran = True
        except (ImportError, OSError):
            by_extractor["sieve"] = []

        # Union: each triple tagged with which extractors saw it.
        provenance: dict[tuple[str, str, str], set[str]] = {}
        for name, triples in by_extractor.items():
            for t in triples:
                provenance.setdefault(t, set()).add(name)

        entries = [
            {"triple": list(t), "extractors": sorted(provenance[t])}
            for t in sorted(provenance.keys())
        ][:max_tags]

        return TransformResult(
            output=entries,
            model="canonical-deterministic-v0",
            provider="canonical-path",
            digital_source_type="algorithmicMedia",
            llm_calls_made=0,
            extra={
                "extractor_used": "multi_school",
                "extractors_ran": sorted(by_extractor.keys()),
                "sieve_ran": sieve_ran,
                "tag_count": len(entries),
                "by_extractor_counts": {
                    name: len(triples)
                    for name, triples in by_extractor.items()
                },
            },
        )


# ─── Naive token-pair extractor — second school for T6 ──────────────
#
# Designed to be different in kind from the sieve, not just different
# in tuning. Sieve: dependency-grammar; emits semantically structured
# (subject, predicate, object). Naive: adjacent-token pairs; emits
# (token_n, "next_to", token_n+1). Two genuinely different schools
# of thought. The overlap (or lack thereof) between their outputs is
# the diagnostic surface multi_school mode exposes.

_TOKENPAIR_PREDICATE = "next_to"

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "to", "in", "on", "at", "for", "with",
    "this", "that", "these", "those", "it", "its",
})


def naive_token_pair_extract(text: str) -> list[tuple[str, str, str]]:
    """Naive second school: adjacent-token bigrams as triples.

    Tokenises on whitespace + punctuation; drops stopwords + tokens
    shorter than 3 chars; emits (token_n, 'next_to', token_n+1)
    for each remaining adjacent pair.

    Deterministic, dependency-free, demonstrably different from the
    sieve's dependency-grammar approach. The point isn't to be
    *better* than the sieve — it's to be *different* so the union /
    intersection of their outputs is a meaningful diagnostic surface.
    """
    import re

    tokens = [
        t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text)
        if t and len(t) >= 3 and t.lower() not in _STOPWORDS
    ]
    return [
        (tokens[i], _TOKENPAIR_PREDICATE, tokens[i + 1])
        for i in range(len(tokens) - 1)
    ]


# Module-level instance — auto-registered in transforms/__init__.py.
EXTRACT_TRANSFORM = ExtractTransform()
