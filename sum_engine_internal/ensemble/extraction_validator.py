"""
Extraction Validator — Structural Gate for LLM→Algebra Boundary

Phase 19A: Validates, canonicalizes, and deduplicates extracted triplets
BEFORE they enter the Gödel State Algebra. Malformed or underspecified
outputs are rejected with audit reasons, not silently ingested.

This is the system's immune system at the NLP boundary.

Pipeline:
    1. Structural validation (non-empty, length bounds, illegal chars)
    2. Predicate canonicalization (synonym collapse)
    3. Batch deduplication (identical triplets within one extraction)
    4. Return accepted + rejected with audit trail

Author: ototao
License: Apache License 2.0
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from sum_engine_internal.algorithms.predicate_canon import canonicalize

logger = logging.getLogger(__name__)

# ── Constraints ───────────────────────────────────────────────────────

MIN_FIELD_LENGTH = 2          # Single-char subjects/objects are garbage
MAX_FIELD_LENGTH = 200        # Absurdly long strings indicate extraction failure
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
JSON_FRAGMENT_PATTERN = re.compile(r'[{}\[\]]')


@dataclass
class RejectedTriplet:
    """A triplet that failed validation, with the reason why."""
    subject: str
    predicate: str
    object_: str
    reason: str


@dataclass
class ValidatedExtraction:
    """Result of running extraction through the validation gate."""
    accepted: List[Tuple[str, str, str]] = field(default_factory=list)
    rejected: List[RejectedTriplet] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def valid_schema_rate(self) -> float:
        total = self.accepted_count + self.rejected_count
        return self.accepted_count / total if total > 0 else 0.0


class ExtractionValidator:
    """
    Structural gate between LLM extraction and Gödel algebra.

    Validates each triplet, canonicalizes predicates, deduplicates
    within a batch, and returns an auditable result.
    """

    def validate_field(self, value: str, field_name: str) -> Optional[str]:
        """
        Validate a single triplet field. Returns rejection reason or None.
        """
        if not value or not value.strip():
            return f"{field_name} is empty"

        stripped = value.strip()

        if len(stripped) < MIN_FIELD_LENGTH:
            return f"{field_name} too short ({len(stripped)} chars, min {MIN_FIELD_LENGTH})"

        if len(stripped) > MAX_FIELD_LENGTH:
            return f"{field_name} too long ({len(stripped)} chars, max {MAX_FIELD_LENGTH})"

        if CONTROL_CHAR_PATTERN.search(stripped):
            return f"{field_name} contains control characters"

        if JSON_FRAGMENT_PATTERN.search(stripped) and len(stripped) < 10:
            return f"{field_name} appears to be a JSON fragment"

        return None

    def validate_triplet(
        self,
        subject: str,
        predicate: str,
        object_: str,
    ) -> Optional[str]:
        """
        Validate a full triplet. Returns rejection reason or None if valid.
        """
        for val, name in [
            (subject, "subject"),
            (predicate, "predicate"),
            (object_, "object"),
        ]:
            reason = self.validate_field(val, name)
            if reason:
                return reason

        return None

    def validate_batch(
        self,
        triplets: List[Tuple[str, str, str]],
        canonicalize_predicates: bool = True,
    ) -> ValidatedExtraction:
        """
        Validate, canonicalize, and deduplicate a batch of extracted triplets.

        Args:
            triplets: Raw (subject, predicate, object) tuples from LLM.
            canonicalize_predicates: If True, run predicate canonicalization.

        Returns:
            ValidatedExtraction with accepted and rejected lists.
        """
        result = ValidatedExtraction()
        seen: set = set()

        for s, p, o in triplets:
            # Normalize
            s_clean = s.strip().lower()
            p_clean = p.strip().lower().replace(" ", "_")
            o_clean = o.strip().lower()

            # Structural validation
            reason = self.validate_triplet(s_clean, p_clean, o_clean)
            if reason:
                result.rejected.append(
                    RejectedTriplet(s_clean, p_clean, o_clean, reason)
                )
                continue

            # Predicate canonicalization
            if canonicalize_predicates:
                p_clean = canonicalize(p_clean)

            # Batch deduplication
            key = (s_clean, p_clean, o_clean)
            if key in seen:
                result.rejected.append(
                    RejectedTriplet(s_clean, p_clean, o_clean, "duplicate in batch")
                )
                continue

            seen.add(key)
            result.accepted.append((s_clean, p_clean, o_clean))

        if result.rejected_count > 0:
            logger.info(
                "Extraction gate: %d accepted, %d rejected (%.0f%% valid)",
                result.accepted_count,
                result.rejected_count,
                result.valid_schema_rate * 100,
            )

        return result
