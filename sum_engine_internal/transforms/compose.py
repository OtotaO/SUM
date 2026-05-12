"""compose — multi-bundle LCM-merge into a SUM-of-SUMs.

The dream's libraries-of-books fulfillment: take N bundles (extracted
from N source documents — articles, book chapters, a whole library)
and produce one merged bundle whose state integer is the LCM of the
inputs and whose axiom set is the union, deduped + sorted.

Pure Gödel-state algebra: no LLM call. The merge is commutative and
associative — composing (A, B, C) gives the same result as composing
((A, B), C) — and idempotent on duplicates (math.lcm(p, p) == p).

v0 scope:
  - Input: ``{"bundles": [B1, B2, …]}`` where each Bi is a dict with
    a ``triples`` key (list of [s,p,o] triples). The ``state_integer``
    field is RE-COMPUTED from the merged triple set so the output's
    state matches its triples regardless of what the inputs had.
  - Output: ``{"axioms": [...], "state_integer": <int>}`` — the
    canonical SUM-of-SUMs.
  - Parameters: ``{"merge_strategy": "lcm"}`` is the only supported
    strategy today. ``"intersect"`` and ``"diff"`` are reserved in
    the spec but raise NotImplementedError until a downstream
    consumer needs them.

Same trust-loop machinery as every other transform: produces a
``sum.transform_receipt.v1`` envelope with ``provider="canonical-path"``
and ``digital_source_type="algorithmicMedia"``.
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


MergeStrategy = Literal["lcm", "intersect", "diff"]
_VALID_STRATEGIES = ("lcm", "intersect", "diff")


def _validate_parameters(params: dict[str, Any]) -> dict[str, Any]:
    strategy = params.get("merge_strategy", "lcm")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"compose: parameter 'merge_strategy' must be one of "
            f"{_VALID_STRATEGIES!r}; got {strategy!r}"
        )
    return {"merge_strategy": strategy}


def _normalize_triple(triple: tuple[str, str, str]) -> tuple[str, str, str]:
    """Same normalisation as the extract transform: lowercase + strip
    per component. Two extractors that produced different casing for
    the same fact compose to a single deduped triple."""
    return tuple(c.strip().lower() for c in triple)  # type: ignore[return-value]


def _bundle_triples(bundle: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Extract the triple list from a bundle dict. Accepts both the
    minimal ``{"triples": [...]}`` shape and the full CanonicalBundle
    shape with ``axioms`` key (existing v1.1.0 bundle format from
    ``sum attest``)."""
    if "triples" in bundle:
        return [
            (str(t[0]), str(t[1]), str(t[2])) for t in bundle["triples"]
        ]
    if "axioms" in bundle:
        # CanonicalBundle axioms shape: list of dicts with subject/
        # predicate/object keys OR list of tuples. Handle both.
        out: list[tuple[str, str, str]] = []
        for a in bundle["axioms"]:
            if isinstance(a, dict):
                out.append((
                    str(a.get("subject", "")),
                    str(a.get("predicate", "")),
                    str(a.get("object", "")),
                ))
            else:
                out.append((str(a[0]), str(a[1]), str(a[2])))
        return out
    raise ValueError(
        "compose: bundle dict must have 'triples' or 'axioms' key"
    )


def _union_triples(
    bundles: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Componentwise-normalised union of triples across all bundles.
    Sorted + unique so output_hash is byte-stable regardless of input
    ordering."""
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for bundle in bundles:
        for t in _bundle_triples(bundle):
            norm = _normalize_triple(t)
            if norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
    return sorted(out)


def _intersect_triples(
    bundles: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Componentwise-normalised intersection — triples present in
    EVERY bundle. Reserved for future use."""
    if not bundles:
        return []
    sets = [
        {_normalize_triple(t) for t in _bundle_triples(b)}
        for b in bundles
    ]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def _diff_triples(
    bundles: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Componentwise-normalised difference — triples in the FIRST
    bundle that don't appear in any other. Reserved for future use."""
    if not bundles:
        return []
    first = {_normalize_triple(t) for t in _bundle_triples(bundles[0])}
    rest_union: set[tuple[str, str, str]] = set()
    for b in bundles[1:]:
        rest_union.update(_normalize_triple(t) for t in _bundle_triples(b))
    diff = first - rest_union
    return sorted(diff)


@dataclass
class ComposeTransform:
    """Multi-bundle merge — LCM-by-default, intersection / difference
    reserved for follow-ups."""
    name: str = "compose"
    requires_llm: bool = False
    digital_source_type: DigitalSourceType = "algorithmicMedia"

    def canonicalize_parameters(self, params: dict[str, Any]) -> bytes:
        return canonicalize(_validate_parameters(params))

    def canonicalize_input(self, raw_input: Any) -> bytes:
        """Input shape: ``{"bundles": [bundle1, bundle2, …]}``. Each
        bundle contributes its triple set; canonicalisation is over
        the array of per-bundle SORTED triple lists, so the merge is
        order-invariant: composing (A, B) gives the same input_hash
        as composing (B, A) iff their triple sets are identical.

        Note: the order of the OUTER array IS significant for hashing
        (so re-ordering input bundles produces a different
        input_hash). This matches the user's intent — explicitly
        composing the same N bundles in a different order is a
        different request, even though the merged output is the
        same.
        """
        if not isinstance(raw_input, dict) or "bundles" not in raw_input:
            raise ValueError(
                "compose input: expected dict with 'bundles' key"
            )
        bundles = raw_input["bundles"]
        if not isinstance(bundles, list):
            raise ValueError(
                f"compose input: 'bundles' must be a list, got "
                f"{type(bundles).__name__}"
            )
        # Each bundle's triples are sorted to make per-bundle ordering
        # invisible to the hash; the OUTER array order is preserved
        # to preserve "I composed A then B" semantics.
        normalised = [
            sorted([_normalize_triple(t) for t in _bundle_triples(b)])
            for b in bundles
        ]
        return canonicalize(normalised)

    def canonicalize_output(self, output: Any) -> bytes:
        """Output is a merged bundle dict. Canonical bytes: JCS over
        the dict, where ``axioms`` is sorted and de-duplicated. The
        ``state_integer`` is included in the canonicalisation so a
        verifier re-computing state from the axioms can cross-check
        the receipt."""
        if not isinstance(output, dict):
            raise ValueError(
                f"compose output: expected dict, got {type(output).__name__}"
            )
        if "axioms" not in output or "state_integer" not in output:
            raise ValueError(
                "compose output: dict must have 'axioms' and "
                "'state_integer' keys"
            )
        # Re-sort + re-dedup for canonical bytes regardless of how
        # the caller laid out the dict.
        axioms = [
            list(_normalize_triple((str(a[0]), str(a[1]), str(a[2]))))
            for a in output["axioms"]
        ]
        seen: set[tuple[str, str, str]] = set()
        canonical_axioms: list[list[str]] = []
        for a in axioms:
            t = tuple(a)
            if t in seen:
                continue
            seen.add(t)
            canonical_axioms.append(a)
        canonical_axioms.sort()
        canonical = {
            "axioms": canonical_axioms,
            "state_integer": int(output["state_integer"]),
        }
        return canonicalize(canonical)

    async def apply(
        self,
        input: Any,
        parameters: dict[str, Any],
        env: TransformEnv,
    ) -> TransformResult:
        norm = _validate_parameters(parameters)
        strategy = norm["merge_strategy"]

        if not isinstance(input, dict) or "bundles" not in input:
            raise ValueError(
                "compose input: expected dict with 'bundles' key"
            )
        bundles = input["bundles"]
        if not isinstance(bundles, list):
            raise ValueError("compose input: 'bundles' must be a list")
        if len(bundles) == 0:
            # Empty compose is the identity: merge nothing → empty bundle.
            return TransformResult(
                output={"axioms": [], "state_integer": 1},
                model="canonical-deterministic-v0",
                provider="canonical-path",
                digital_source_type="algorithmicMedia",
                llm_calls_made=0,
                extra={"merge_strategy": strategy, "bundle_count": 0,
                       "axiom_count": 0},
            )

        if strategy == "lcm":
            merged = _union_triples(bundles)
        elif strategy == "intersect":
            raise NotImplementedError(
                "compose transform v0 (T3) supports merge_strategy='lcm' "
                "only. 'intersect' is reserved in the spec and will land "
                "when a downstream consumer (e.g. \"what facts do all "
                "these bundles agree on?\") needs it."
            )
        elif strategy == "diff":
            raise NotImplementedError(
                "compose transform v0 (T3) supports merge_strategy='lcm' "
                "only. 'diff' is reserved in the spec and will land "
                "when a downstream consumer (e.g. \"what's unique to the "
                "first bundle?\") needs it."
            )
        else:
            raise ValueError(f"unsupported merge_strategy: {strategy!r}")

        # Recompute the state integer from the merged triple set.
        # Pure-Python path; ``GodelStateAlgebra.encode_chunk_state``
        # is the canonical implementation. Lazy import to keep the
        # transform protocol module import-light.
        try:
            from sum_engine_internal.algorithms.semantic_arithmetic import (
                GodelStateAlgebra,
            )
        except ImportError as e:
            raise ImportError(
                "compose transform requires sum_engine_internal.algorithms; "
                "this package ships with the base sum-engine install — "
                "an ImportError here indicates a corrupted install."
            ) from e

        algebra = GodelStateAlgebra()
        state_integer = algebra.encode_chunk_state(merged)

        merged_bundle = {
            "axioms": [list(t) for t in merged],
            "state_integer": state_integer,
        }
        return TransformResult(
            output=merged_bundle,
            model="canonical-deterministic-v0",
            provider="canonical-path",
            digital_source_type="algorithmicMedia",
            llm_calls_made=0,
            extra={
                "merge_strategy": strategy,
                "bundle_count": len(bundles),
                "axiom_count": len(merged),
                "input_axiom_counts": [
                    len(_bundle_triples(b)) for b in bundles
                ],
            },
        )


COMPOSE_TRANSFORM = ComposeTransform()
