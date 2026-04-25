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
    """Snap every axis of a TomeSliders to its bin centre. Returns a new
    frozen TomeSliders. Used as the cache-key derivation for renderer
    requests so two near-identical drag positions hit the same cache."""
    return TomeSliders(
        density=snap_to_bin(sliders.density, bins),
        length=snap_to_bin(sliders.length, bins),
        formality=snap_to_bin(sliders.formality, bins),
        audience=snap_to_bin(sliders.audience, bins),
        perspective=snap_to_bin(sliders.perspective, bins),
    )


def length_fragment(value: float) -> str:
    """Prompt fragment for the length axis. 0.0 = telegraphic, 1.0 =
    expansive. Returned string is appended to the system prompt of the
    LLM extrapolator. Empirical contract: at value=0.5, the fragment
    should be neutral (no length pressure either way).

    NOTE: implemented in STATE 4. Stub returns the deterministic empty
    string at value=0.5 only; raises NotImplementedError otherwise so
    a half-finished slider call fails loudly rather than silently
    producing length-uncontrolled output.
    """
    if abs(value - 0.5) < 1e-6:
        return ""
    raise NotImplementedError(
        f"length_fragment(value={value}) — STATE 4 deliverable. "
        "See docs/SLIDER_CONTRACT.md §Length."
    )


def formality_fragment(value: float) -> str:
    """Prompt fragment for the formality axis. 0.0 = casual / colloquial,
    1.0 = academic / passive-voice / hedge-laden. STATE 4 deliverable;
    same fail-loud semantics as length_fragment."""
    if abs(value - 0.5) < 1e-6:
        return ""
    raise NotImplementedError(
        f"formality_fragment(value={value}) — STATE 4 deliverable. "
        "See docs/SLIDER_CONTRACT.md §Formality."
    )


def audience_fragment(value: float) -> str:
    """Prompt fragment for the audience axis. 0.0 = lay reader / no
    jargon, 1.0 = domain expert / jargon-dense. STATE 4 deliverable."""
    if abs(value - 0.5) < 1e-6:
        return ""
    raise NotImplementedError(
        f"audience_fragment(value={value}) — STATE 4 deliverable. "
        "See docs/SLIDER_CONTRACT.md §Audience."
    )


def perspective_fragment(value: float) -> str:
    """Prompt fragment for the perspective axis. 0.0 = first-person /
    subjective, 1.0 = omniscient / third-person. STATE 4 deliverable."""
    if abs(value - 0.5) < 1e-6:
        return ""
    raise NotImplementedError(
        f"perspective_fragment(value={value}) — STATE 4 deliverable. "
        "See docs/SLIDER_CONTRACT.md §Perspective."
    )


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
