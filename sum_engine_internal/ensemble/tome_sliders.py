"""
Tome Sliders — Control Parameters for Bidirectional Knowledge Rendering

The founder's core dream made concrete: tunable knobs for the tag→tome
direction. Every SUM rendering takes a TomeSliders record in [0.0, 1.0]^5
and produces output conformant to those controls.

Implementation status (as of module authoring):
    density    — implemented on the deterministic canonical path
                 (axiom subsetting via lexicographic ordering)
    length     — LLM-gated; no-op without an extrapolator
    formality  — LLM-gated
    audience   — LLM-gated
    perspective — LLM-gated

Slider values are captured in the output artefact's header so the same
narrative can be regenerated with adjusted parameters and the difference
audited.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TomeSliders:
    """Slider configuration for controlled tome rendering.

    All values in [0.0, 1.0]. Defaults preserve lossless / balanced behavior.

    - density:     0.0 = empty, 1.0 = full axiom coverage
    - length:      0.0 = telegraphic, 1.0 = expansive
    - formality:   0.0 = casual, 1.0 = academic
    - audience:    0.0 = novice, 1.0 = expert / jargon-dense
    - perspective: 0.0 = first-person, 1.0 = omniscient / third-person
    """

    density: float = 1.0
    length: float = 0.5
    formality: float = 0.5
    audience: float = 0.5
    perspective: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "density",
            "length",
            "formality",
            "audience",
            "perspective",
        ):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} out of [0, 1]: {v}")

    def requires_extrapolator(self) -> bool:
        """True if any slider besides density deviates from its balanced
        default (0.5), meaning an LLM extrapolator is needed to honour it.
        The canonical deterministic path can only action the density slider.
        """
        return not (
            self.length == 0.5
            and self.formality == 0.5
            and self.audience == 0.5
            and self.perspective == 0.5
        )

    def header_line(self) -> str:
        """Single-line serialization for canonical tome headers."""
        return (
            f"@sliders: density={self.density:.3f} "
            f"length={self.length:.3f} "
            f"formality={self.formality:.3f} "
            f"audience={self.audience:.3f} "
            f"perspective={self.perspective:.3f}"
        )


def apply_density(
    axiom_keys: Sequence[str], density: float
) -> list[str]:
    """Deterministic axiom subsetting by density.

    Sorts axiom keys lexicographically and keeps the first floor(N * density)
    entries. Deterministic across runs and machines. Empty list when density
    rounds to zero; full sorted list when density >= 1.0.

    The lexicographic ordering ensures stability: running the same state
    through the same density on two hosts produces identical subsets.
    """
    if not axiom_keys:
        return []
    if density >= 1.0:
        return sorted(axiom_keys)
    if density <= 0.0:
        return []
    sorted_keys = sorted(axiom_keys)
    n = len(sorted_keys)
    k = int(n * density)
    return sorted_keys[:k]


# ─── Phase E additions: discretization + axis prompt-conditioning ─────
#
# The slider renderer (sum_engine_internal/ensemble/slider_renderer.py)
# uses these to (a) snap continuous slider positions to a finite cache
# bin grid, and (b) build prompt fragments per axis without leaking
# axis semantics into the renderer module.

# Cache-bin granularity per axis. 5 bins ⇒ 3125 cache cells per
# triples-set. Empirically: anything finer is wasted (LLM output is
# perceptually identical between adjacent bins); anything coarser
# loses control resolution users actually exercise. Adjust when E.6
# trial telemetry shows the actual axis-position distribution.
SLIDER_BINS_PER_AXIS: int = 5


def snap_to_bin(value: float, bins: int = SLIDER_BINS_PER_AXIS) -> float:
    """Quantize a continuous [0, 1] slider value to one of `bins` discrete
    positions, returning the bin's centre as a float in [0, 1].

    Pure function. Same input → same output forever. Used by the cache
    layer to collapse near-identical slider positions to a single key.
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"slider value out of [0, 1]: {value}")
    if bins < 2:
        raise ValueError(f"bins must be >= 2: {bins}")
    idx = min(int(value * bins), bins - 1)
    # Bin centre. For bins=5, returns one of {0.1, 0.3, 0.5, 0.7, 0.9}.
    return (idx + 0.5) / bins


def quantize(sliders: TomeSliders, bins: int = SLIDER_BINS_PER_AXIS) -> TomeSliders:
    """Snap the four LLM axes to their bin centres. Density passes through
    unchanged. Returns a new frozen TomeSliders.

    Why density is exempt: density is deterministic — the canonical path
    selects exactly floor(N * density) triples per the lexicographic
    prefix. Binning would cost UX (1.0 silently maps to 0.9 ⇒ user can
    never request 'all triples') without saving cache space (no LLM call
    to dedupe; the cache cell is a tiny dict lookup either way). Only
    the four LLM axes need binning to collapse near-identical drag
    positions onto the same expensive LLM call."""
    return TomeSliders(
        density=sliders.density,
        length=snap_to_bin(sliders.length, bins),
        formality=snap_to_bin(sliders.formality, bins),
        audience=snap_to_bin(sliders.audience, bins),
        perspective=snap_to_bin(sliders.perspective, bins),
    )


# Per-axis fragments are keyed by bin centre (5-bin grid). The renderer
# only ever calls these with quantized values — i.e. one of {0.1, 0.3,
# 0.5, 0.7, 0.9}. Direct dict lookup is faster than range checks and
# guarantees identical output for identical input. Mid-band (0.5) is
# always the empty string so we don't bloat the system prompt at default.

_LENGTH_FRAGMENTS: dict[float, str] = {
    0.1: "Use the most concise prose possible: each fact in one short sentence, no elaboration.",
    0.3: "Be brief. Prefer short sentences. Minimal connective tissue between facts.",
    0.5: "",
    0.7: "Expand each fact with relevant elaboration and context.",
    0.9: "Write expansively: develop each fact into a detailed paragraph with rich context and examples.",
}

_FORMALITY_FRAGMENTS: dict[float, str] = {
    0.1: "Use casual, conversational tone. Contractions are encouraged. Address the reader directly.",
    0.3: "Use a friendly, approachable tone. Light contractions are fine.",
    0.5: "",
    0.7: "Use formal academic register. Prefer precise vocabulary. Avoid contractions.",
    0.9: "Write in strict academic register: passive voice where appropriate, no contractions, measured hedging language.",
}

_AUDIENCE_FRAGMENTS: dict[float, str] = {
    0.1: "Write for a curious general reader. Avoid all domain-specific jargon. Use everyday words.",
    0.3: "Write for an interested non-specialist. Define any technical terms inline on first use.",
    0.5: "",
    0.7: "Write for a domain practitioner. Use field-specific terminology freely.",
    0.9: "Write for a domain expert. Use precise technical jargon without explanation.",
}

_PERSPECTIVE_FRAGMENTS: dict[float, str] = {
    0.1: "Write in first person throughout ('I observed', 'we found', 'our data shows').",
    0.3: "Write primarily in first person, with occasional third-person framing.",
    0.5: "",
    0.7: "Write in third-person omniscient. Avoid first-person pronouns.",
    0.9: "Write in pure third-person omniscient narration. Use no first-person pronouns at all.",
}


def _lookup_fragment(table: dict[float, str], value: float, axis_name: str) -> str:
    """Quantize-then-lookup. Tolerates floating-point noise around bin
    centres so callers can pass either snapped or near-snapped values."""
    snapped = snap_to_bin(value)
    for centre, fragment in table.items():
        if abs(snapped - centre) < 1e-6:
            return fragment
    raise ValueError(f"{axis_name}_fragment: value {value} (snapped {snapped}) not in 5-bin grid")


def length_fragment(value: float) -> str:
    """Prompt fragment for the length axis. 0.0=telegraphic, 1.0=expansive.
    Returns the empty string at the neutral midpoint. See SLIDER_CONTRACT.md §Length."""
    return _lookup_fragment(_LENGTH_FRAGMENTS, value, "length")


def formality_fragment(value: float) -> str:
    """Prompt fragment for the formality axis. 0.0=casual, 1.0=academic.
    See SLIDER_CONTRACT.md §Formality."""
    return _lookup_fragment(_FORMALITY_FRAGMENTS, value, "formality")


def audience_fragment(value: float) -> str:
    """Prompt fragment for the audience axis. 0.0=lay reader, 1.0=domain expert.
    See SLIDER_CONTRACT.md §Audience."""
    return _lookup_fragment(_AUDIENCE_FRAGMENTS, value, "audience")


def perspective_fragment(value: float) -> str:
    """Prompt fragment for the perspective axis. 0.0=first-person, 1.0=omniscient third-person.
    See SLIDER_CONTRACT.md §Perspective."""
    return _lookup_fragment(_PERSPECTIVE_FRAGMENTS, value, "perspective")


def build_system_prompt(sliders: TomeSliders) -> str:
    """Assemble the LLM system prompt for a styled render. Always starts
    with the truth-preservation contract; appends per-axis fragments only
    when the axis deviates from neutral (avoids prompt bloat at default).

    Pure function (no LLM call). Deterministic per input.

    STATE 4 will fill the per-axis fragments. STATE 2 ships the
    skeleton so the renderer module can import + type-check.
    """
    base = (
        "You are a precise technical writer. Render the following "
        "facts as a cohesive narrative. Do not invent facts. Do not "
        "omit facts. Preserve every (subject, predicate, object) "
        "relationship in the input."
    )
    fragments: list[str] = []
    if abs(sliders.length - 0.5) >= 1e-6:
        fragments.append(length_fragment(sliders.length))
    if abs(sliders.formality - 0.5) >= 1e-6:
        fragments.append(formality_fragment(sliders.formality))
    if abs(sliders.audience - 0.5) >= 1e-6:
        fragments.append(audience_fragment(sliders.audience))
    if abs(sliders.perspective - 0.5) >= 1e-6:
        fragments.append(perspective_fragment(sliders.perspective))
    return base if not fragments else base + " " + " ".join(fragments)
