"""
Phase 19A: Extraction Validator Tests

Tests the structural gate between LLM extraction and Gödel algebra.
Covers: empty/whitespace rejection, length bounds, illegal chars,
predicate canonicalization, batch deduplication, pass-through, and
audit trail completeness.

Author: ototao
License: Apache License 2.0
"""

import pytest
from internal.ensemble.extraction_validator import (
    ExtractionValidator,
    ValidatedExtraction,
    RejectedTriplet,
    MIN_FIELD_LENGTH,
    MAX_FIELD_LENGTH,
)


@pytest.fixture
def validator():
    return ExtractionValidator()


class TestFieldRejection:

    def test_empty_subject_rejected(self, validator):
        result = validator.validate_batch([("", "causes", "warming")])
        assert result.rejected_count == 1
        assert result.accepted_count == 0
        assert "empty" in result.rejected[0].reason

    def test_whitespace_only_rejected(self, validator):
        result = validator.validate_batch([("  ", "causes", "warming")])
        assert result.rejected_count == 1
        assert "empty" in result.rejected[0].reason

    def test_empty_predicate_rejected(self, validator):
        result = validator.validate_batch([("sun", "", "warming")])
        assert result.rejected_count == 1
        assert "predicate" in result.rejected[0].reason

    def test_empty_object_rejected(self, validator):
        result = validator.validate_batch([("sun", "causes", "")])
        assert result.rejected_count == 1
        assert "object" in result.rejected[0].reason

    def test_single_char_subject_rejected(self, validator):
        result = validator.validate_batch([("x", "causes", "warming")])
        assert result.rejected_count == 1
        assert "too short" in result.rejected[0].reason

    def test_single_char_predicate_rejected(self, validator):
        result = validator.validate_batch([("sun", "x", "warming")])
        assert result.rejected_count == 1
        assert "too short" in result.rejected[0].reason

    def test_overlength_subject_rejected(self, validator):
        long_str = "a" * (MAX_FIELD_LENGTH + 1)
        result = validator.validate_batch([(long_str, "causes", "warming")])
        assert result.rejected_count == 1
        assert "too long" in result.rejected[0].reason

    def test_control_chars_rejected(self, validator):
        result = validator.validate_batch([("sun\x00data", "causes", "warming")])
        assert result.rejected_count == 1
        assert "control" in result.rejected[0].reason.lower()


class TestPredicateCanonicalization:

    def test_synonym_collapsed(self, validator):
        """'leads_to' should be canonicalized to 'causes'."""
        result = validator.validate_batch(
            [("sun", "leads_to", "warming")],
            canonicalize_predicates=True,
        )
        assert result.accepted_count == 1
        s, p, o = result.accepted[0]
        assert p == "causes"

    def test_triggers_becomes_causes(self, validator):
        result = validator.validate_batch(
            [("heat", "triggers", "expansion")],
            canonicalize_predicates=True,
        )
        assert result.accepted[0][1] == "causes"

    def test_prevents_becomes_inhibits(self, validator):
        result = validator.validate_batch(
            [("vaccine", "prevents", "infection")],
            canonicalize_predicates=True,
        )
        assert result.accepted[0][1] == "inhibits"

    def test_canonical_predicate_unchanged(self, validator):
        """Already-canonical predicates should pass through unchanged."""
        result = validator.validate_batch(
            [("sun", "causes", "warming")],
            canonicalize_predicates=True,
        )
        assert result.accepted[0][1] == "causes"

    def test_unknown_predicate_passthrough(self, validator):
        """Predicates not in the canonical map should pass through."""
        result = validator.validate_batch(
            [("mars", "orbits", "sun")],
            canonicalize_predicates=True,
        )
        assert result.accepted[0][1] == "orbits"

    def test_canonicalization_disabled(self, validator):
        """With canonicalization off, synonyms should not be collapsed."""
        result = validator.validate_batch(
            [("sun", "leads_to", "warming")],
            canonicalize_predicates=False,
        )
        assert result.accepted[0][1] == "leads_to"


class TestBatchDeduplication:

    def test_exact_duplicates_collapsed(self, validator):
        result = validator.validate_batch([
            ("earth", "orbits", "sun"),
            ("earth", "orbits", "sun"),
        ])
        assert result.accepted_count == 1
        assert result.rejected_count == 1
        assert "duplicate" in result.rejected[0].reason

    def test_synonym_duplicates_collapsed(self, validator):
        """After canonicalization, synonym predicates become the same."""
        result = validator.validate_batch([
            ("heat", "triggers", "expansion"),
            ("heat", "causes", "expansion"),
        ])
        # Both map to "causes" — second should be rejected as duplicate
        assert result.accepted_count == 1
        assert result.rejected_count == 1

    def test_distinct_triplets_kept(self, validator):
        result = validator.validate_batch([
            ("earth", "orbits", "sun"),
            ("moon", "orbits", "earth"),
        ])
        assert result.accepted_count == 2
        assert result.rejected_count == 0


class TestValidTripletPassthrough:

    def test_normal_triplet_accepted(self, validator):
        result = validator.validate_batch([
            ("earth", "orbits", "sun"),
        ])
        assert result.accepted_count == 1
        assert result.rejected_count == 0
        assert result.accepted[0] == ("earth", "orbits", "sun")

    def test_case_normalization(self, validator):
        result = validator.validate_batch([
            ("  Earth  ", "ORBITS", "  Sun  "),
        ])
        assert result.accepted[0] == ("earth", "orbits", "sun")

    def test_multi_word_predicate_underscored(self, validator):
        result = validator.validate_batch([
            ("cell", "is part of", "tissue"),
        ])
        # Space → underscore, then canonicalized
        assert result.accepted[0][1] == "has_part"

    def test_mixed_batch(self, validator):
        """A batch with good, bad, and duplicate triplets."""
        result = validator.validate_batch([
            ("earth", "orbits", "sun"),         # good
            ("", "causes", "warming"),            # rejected: empty subject
            ("earth", "orbits", "sun"),          # rejected: duplicate
            ("x", "causes", "warming"),           # rejected: too short
            ("moon", "orbits", "earth"),          # good
        ])
        assert result.accepted_count == 2
        assert result.rejected_count == 3

    def test_valid_schema_rate_calculation(self, validator):
        result = validator.validate_batch([
            ("earth", "orbits", "sun"),
            ("", "causes", "warming"),
        ])
        assert result.valid_schema_rate == 0.5

    def test_empty_batch(self, validator):
        result = validator.validate_batch([])
        assert result.accepted_count == 0
        assert result.rejected_count == 0
        assert result.valid_schema_rate == 0.0


class TestAuditTrail:

    def test_rejection_includes_reason(self, validator):
        result = validator.validate_batch([("", "causes", "warming")])
        rej = result.rejected[0]
        assert isinstance(rej, RejectedTriplet)
        assert rej.reason
        assert rej.subject == ""
        assert rej.predicate == "causes"
        assert rej.object_ == "warming"

    def test_multiple_rejections_tracked(self, validator):
        result = validator.validate_batch([
            ("", "causes", "warming"),
            ("x", "y", "warming"),
        ])
        assert result.rejected_count == 2
        reasons = [r.reason for r in result.rejected]
        assert any("empty" in r for r in reasons)
        assert any("too short" in r for r in reasons)
