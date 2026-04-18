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
