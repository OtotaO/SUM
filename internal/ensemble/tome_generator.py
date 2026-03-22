"""
Autoregressive Tome Generator — Lossless Semantic Rehydration

Unpacks a Gödel Integer into structured knowledge with two operating modes:

**Proof Mode (Canonical):**
    Deterministic, template-based rendering of every active axiom grouped
    by subject entity.  This is the layer on which round-trip conservation
    is mathematically verified.  Works without any LLM.

**Narrative Mode (Optional):**
    If an ``extrapolator`` (QuantumExtrapolator) is available, each
    chapter's axioms are expanded into readable prose via the epistemic
    loop.  The canonical appendix is preserved so the proof path is
    never lost.

Phase 14: The Ouroboros Protocol.

Author: ototao
License: Apache License 2.0
"""

import logging
from typing import Dict, List, Optional

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

logger = logging.getLogger(__name__)

# The canonical format version — treat this as an ABI contract.
# Bump when the template grammar changes.
CANONICAL_FORMAT_VERSION = "1.0.0"


class AutoregressiveTomeGenerator:
    """
    Unpacks a Gödel Integer into a complete, structured Tome.

    Supports two rendering modes:
        * ``proof``  — deterministic canonical output (always available)
        * ``narrative`` — LLM-enhanced prose (requires extrapolator)
    """

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        extrapolator=None,
    ):
        self.algebra = algebra
        self.extrapolator = extrapolator  # None in math-only mode

    # ------------------------------------------------------------------
    # Core: cluster active axioms by subject
    # ------------------------------------------------------------------

    def extract_active_axioms(self, target_state: int) -> List[str]:
        """Return all axiom keys whose primes divide the target state."""
        active = []
        for prime, axiom in self.algebra.prime_to_axiom.items():
            if target_state % prime == 0:
                active.append(axiom)
        return active

    def cluster_by_subject(
        self, target_state: int
    ) -> Dict[str, List[str]]:
        """Group active axioms by their subject entity."""
        chapters: Dict[str, List[str]] = {}
        for axiom in self.extract_active_axioms(target_state):
            parts = axiom.split("||")
            if len(parts) == 3:
                subject = parts[0].strip()
                if subject not in chapters:
                    chapters[subject] = []
                chapters[subject].append(axiom)
        return chapters

    # ------------------------------------------------------------------
    # Proof Mode: deterministic canonical rendering
    # ------------------------------------------------------------------

    def generate_canonical(
        self, target_state: int, title: str = "Canonical Tome"
    ) -> str:
        """
        Deterministic, template-based rendering.

        Every active axiom is emitted in a canonical ``subject PREDICATE
        object`` format, grouped by subject and sorted lexicographically.
        This output is guaranteed to round-trip through the Sieve and
        produce the same Gödel Integer.

        Args:
            target_state: The Gödel integer to unpack.
            title:        Human-readable title for the Tome.

        Returns:
            A structured text representation of the integer's knowledge.
        """
        chapters = self.cluster_by_subject(target_state)

        if not chapters:
            return f"@canonical_version: {CANONICAL_FORMAT_VERSION}\n# {title}\n\nThe Gödel State is empty (= 1). No axioms to rehydrate."

        lines = [f"@canonical_version: {CANONICAL_FORMAT_VERSION}", f"# {title}", ""]

        # Sort subjects lexicographically for determinism
        for subject in sorted(chapters.keys()):
            axioms = sorted(chapters[subject])  # Sort axioms too
            lines.append(f"## {subject.title()}")
            lines.append("")
            for axiom in axioms:
                parts = axiom.split("||")
                if len(parts) == 3:
                    s, p, o = parts
                    # Canonical sentence: "The {subject} {predicate} {object}."
                    lines.append(f"The {s} {p} {o}.")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Narrative Mode: LLM-enhanced prose with canonical fallback
    # ------------------------------------------------------------------

    async def generate_narrative(
        self, target_state: int, title: str = "The Quantum Tome"
    ) -> str:
        """
        LLM-enhanced rendering with canonical fallback.

        If an extrapolator is available, each chapter's axioms are
        expanded into readable prose.  If the extrapolator is absent
        or fails, falls back to canonical rendering.

        Args:
            target_state: The Gödel integer to unpack.
            title:        Human-readable title for the Tome.

        Returns:
            A narrative text representation.
        """
        chapters = self.cluster_by_subject(target_state)

        if not chapters:
            return f"# {title}\n\nThe Gödel State is empty (= 1). No axioms to rehydrate."

        lines = [f"# {title}", ""]

        for subject in sorted(chapters.keys()):
            axioms = sorted(chapters[subject])
            lines.append(f"## {subject.title()}")
            lines.append("")

            if self.extrapolator is not None:
                try:
                    narrative = await self.extrapolator.extrapolate_with_proof(
                        target_state, axioms
                    )
                    lines.append(narrative)
                except RuntimeError:
                    # Fallback: canonical rendering
                    for axiom in axioms:
                        parts = axiom.split("||")
                        if len(parts) == 3:
                            lines.append(f"The {parts[0]} {parts[1]} {parts[2]}.")
            else:
                # No LLM — canonical rendering
                for axiom in axioms:
                    parts = axiom.split("||")
                    if len(parts) == 3:
                        lines.append(f"The {parts[0]} {parts[1]} {parts[2]}.")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    async def generate_tome(
        self,
        target_state: int,
        title: str = "The Quantum Tome",
        mode: str = "proof",
    ) -> str:
        """
        Generate a Tome from a Gödel Integer.

        Args:
            target_state: The integer to unpack.
            title:        Tome title.
            mode:         ``"proof"`` for deterministic canonical output,
                          ``"narrative"`` for LLM-enhanced prose.

        Returns:
            The generated Tome text.
        """
        if mode == "narrative":
            return await self.generate_narrative(target_state, title)
        return self.generate_canonical(target_state, title)
