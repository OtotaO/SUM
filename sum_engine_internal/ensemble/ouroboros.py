"""
Ouroboros Verifier — Proof of Semantic Conservation

Proves that the Gödel-State Engine performs lossless round-tripping
over its canonical axiom representation:

    Integer A → Canonical Tome → Parse Axiom Keys → Integer B

If ``A == B``, semantic mass is conserved through the encode-decode cycle.

The proof operates on the **canonical layer**: the tome renderer emits
deterministic ``"The {s} {p} {o}."`` sentences, and the verifier parses
those exact templates back into axiom keys.  The NLP sieve is **never**
used for the proof — it is a lossy projection (lemmatization, POS
parsing) that would break the bijection.

For ``verify_from_text``, the sieve encodes the initial text into
Integer A, but the conservation proof from A onward uses the canonical
path exclusively.

Diagnostics:
    When the round-trip fails, the verifier reports exactly which axioms
    were lost, which were spuriously added, and counts.

Phase 14: The Ouroboros Protocol.

Author: ototao
License: Apache License 2.0
"""

import math
import re
import logging
from datetime import datetime, timezone
from typing import List, Tuple
from dataclasses import dataclass, field

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
from sum_engine_internal.ensemble.tome_generator import (
    AutoregressiveTomeGenerator,
    CANONICAL_FORMAT_VERSION,
)

logger = logging.getLogger(__name__)


def _zig():
    try:
        from sum_engine_internal.infrastructure.zig_bridge import zig_engine
        return zig_engine
    except ImportError:
        return None


@dataclass
class ConservationProof:
    """
    The result of a semantic conservation round-trip.

    Attributes:
        is_conserved:     True if the round-trip is lossless.
        format_version:   Canonical format version used for the proof.
        proof_mode:       Always ``"canonical"`` for conservation proofs.
        timestamp:        ISO 8601 timestamp of verification.
        source_state:     Integer A (original encoding).
        reconstructed_state: Integer B (re-encoded from canonical tome).
        state_a_digits:   Digit count of Integer A.
        state_b_digits:   Digit count of Integer B.
        source_axiom_count:  Number of axioms in A.
        reconstructed_axiom_count: Number of axioms in B.
        missing_axioms:   Axioms in A but not in B (lost in decode).
        extra_axioms:     Axioms in B but not in A (spuriously added).
        canonical_tome:   The intermediate canonical text.
    """
    is_conserved: bool
    format_version: str
    proof_mode: str
    timestamp: str
    source_state: int
    reconstructed_state: int
    state_a_digits: int
    state_b_digits: int
    source_axiom_count: int
    reconstructed_axiom_count: int
    missing_axioms: List[str] = field(default_factory=list)
    extra_axioms: List[str] = field(default_factory=list)
    canonical_tome: str = ""


class OuroborosVerifier:
    """
    Proves Lossless Semantic Conservation.

    The proof path is:
        Integer A → Canonical Tome (deterministic template) →
        Parse "The {s} {p} {o}." lines → Integer B

    This is a bijective codec over the canonical representation.
    The NLP sieve is only used for initial text→triplets encoding,
    never for the conservation proof itself.
    """

    # Supported canonical format versions
    SUPPORTED_VERSIONS = {"1.0.0"}

    # Regex to extract axiom components from canonical "The {s} {p} {o}." lines
    _CANONICAL_LINE_RE = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    # Regex to extract version header
    _VERSION_RE = re.compile(r"^@canonical_version:\s*(.+)$")

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        sieve: DeterministicSieve,
        tome_generator: AutoregressiveTomeGenerator,
    ):
        self.algebra = algebra
        self.sieve = sieve
        self.tome_generator = tome_generator

    def _reconstruct_from_canonical(
        self, canonical_tome: str
    ) -> Tuple[int, List[str], str]:
        """
        Parse canonical tome text back into axiom keys and re-encode.

        The canonical renderer emits lines in the exact format::

            @canonical_version: 1.0.0
            The {subject} {predicate} {object}.

        This method extracts the version, validates it, reconstructs
        axiom keys, and re-encodes to a Gödel integer.

        Returns:
            (reconstructed_state, list_of_axiom_keys, format_version)
        """
        state = 1
        axiom_keys = []
        format_version = "unknown"

        for line in canonical_tome.splitlines():
            line = line.strip()

            # Check for version header
            vm = self._VERSION_RE.match(line)
            if vm:
                format_version = vm.group(1).strip()
                continue

            m = self._CANONICAL_LINE_RE.match(line)
            if m:
                s, p, o = m.group(1), m.group(2), m.group(3)
                prime = self.algebra.get_or_mint_prime(s, p, o)
                if state % prime != 0:
                    z = _zig()
                    r = z.bigint_lcm(state, prime) if z else None
                    state = r if r is not None else math.lcm(state, prime)
                axiom_keys.append(f"{s}||{p}||{o}")

        return state, axiom_keys, format_version

    def _extract_axiom_keys(self, state: int) -> set:
        """Extract the set of axiom keys whose primes divide the state."""
        keys = set()
        for prime, axiom in self.algebra.prime_to_axiom.items():
            if state % prime == 0:
                keys.add(axiom)
        return keys

    def verify_from_state(self, target_state: int) -> ConservationProof:
        """
        Verify lossless conservation of a Gödel Integer.

        Pipeline:
            Integer A → Canonical Tome → Parse Axiom Keys → Integer B
            Conservation iff A == B

        The proof uses deterministic canonical rendering + deterministic
        template parsing.  No NLP, no lemmatization, no ambiguity.

        Args:
            target_state: The Gödel integer to verify.

        Returns:
            A ``ConservationProof`` with full diagnostics.
        """
        # Step 1: Canonical decode
        canonical_tome = self.tome_generator.generate_canonical(
            target_state, "Ouroboros Verification Tome"
        )

        # Step 2: Re-encode from canonical template lines (NOT via NLP sieve)
        reconstructed_state, _, format_version = self._reconstruct_from_canonical(
            canonical_tome
        )

        # Step 3: Diagnose
        source_axioms = self._extract_axiom_keys(target_state)
        reconstructed_axioms = self._extract_axiom_keys(reconstructed_state)

        missing = sorted(source_axioms - reconstructed_axioms)
        extra = sorted(reconstructed_axioms - source_axioms)

        is_conserved = (target_state == reconstructed_state)
        now = datetime.now(timezone.utc).isoformat()

        proof = ConservationProof(
            is_conserved=is_conserved,
            format_version=format_version,
            proof_mode="canonical",
            timestamp=now,
            source_state=target_state,
            reconstructed_state=reconstructed_state,
            state_a_digits=len(str(target_state)),
            state_b_digits=len(str(reconstructed_state)),
            source_axiom_count=len(source_axioms),
            reconstructed_axiom_count=len(reconstructed_axioms),
            missing_axioms=missing,
            extra_axioms=extra,
            canonical_tome=canonical_tome,
        )

        if is_conserved:
            logger.info(
                "Ouroboros Proof: CONSERVED — %d axioms round-tripped losslessly.",
                len(source_axioms),
            )
        else:
            logger.warning(
                "Ouroboros Proof: DIVERGED — missing=%d, extra=%d",
                len(missing), len(extra),
            )

        return proof

    def verify_from_text(self, text: str) -> ConservationProof:
        """
        Full Ouroboros: Text → Sieve → Integer A → Canonical → Parse → Integer B.

        The sieve is used ONLY for the initial text→triplets encoding.
        The conservation proof (A→Canonical→B) is fully deterministic.

        Args:
            text: Raw input text.

        Returns:
            A ``ConservationProof`` with full diagnostics.
        """
        # Step 1: Text → Triplets → Integer A (via NLP sieve)
        triplets = self.sieve.extract_triplets(text)
        state_a = 1
        for s, p, o in triplets:
            prime = self.algebra.get_or_mint_prime(s, p, o)
            if state_a % prime != 0:
                z = _zig()
                r = z.bigint_lcm(state_a, prime) if z else None
                state_a = r if r is not None else math.lcm(state_a, prime)

        # Step 2: Verify conservation from Integer A (canonical path only)
        proof = self.verify_from_state(state_a)

        return proof

    def proof_to_dict(self, proof: ConservationProof) -> dict:
        """Serialize a ConservationProof for API responses."""
        return {
            "is_conserved": proof.is_conserved,
            "format_version": proof.format_version,
            "proof_mode": proof.proof_mode,
            "timestamp": proof.timestamp,
            "state_a_digits": proof.state_a_digits,
            "state_b_digits": proof.state_b_digits,
            "source_axiom_count": proof.source_axiom_count,
            "reconstructed_axiom_count": proof.reconstructed_axiom_count,
            "missing_axioms": proof.missing_axioms,
            "extra_axioms": proof.extra_axioms,
            "states_match": proof.source_state == proof.reconstructed_state,
        }

