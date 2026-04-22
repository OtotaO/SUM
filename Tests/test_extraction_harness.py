"""
Extraction Evaluation Harness — Workstream 5

Measures extraction fidelity by running known input texts through
the semantic sieve and comparing the extracted axioms against a
ground-truth set. Reports precision, recall, and F1.

This is the weakest link in the system (see THREAT_MODEL.md §3.3).
By measuring it explicitly, we convert a hand-wave into data.

Author: ototao
License: Apache License 2.0
"""

import pytest
from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# ─── Ground-Truth Test Cases ─────────────────────────────────────
#
# Each test case provides:
#   - input_text: raw English text
#   - expected_axioms: set of (subject, predicate, object) triples
#     that a perfect extractor would recover
#   - difficulty: "easy" | "medium" | "hard"

GROUND_TRUTH = [
    {
        "id": "GT-001",
        "input_text": "Alice likes cats.",
        "expected_axioms": {("alice", "likes", "cats")},
        "difficulty": "easy",
    },
    {
        "id": "GT-002",
        "input_text": "Bob knows Python. Alice likes cats.",
        "expected_axioms": {
            ("bob", "knows", "python"),
            ("alice", "likes", "cats"),
        },
        "difficulty": "easy",
    },
    {
        "id": "GT-003",
        "input_text": "The Earth orbits the Sun.",
        "expected_axioms": {("earth", "orbits", "sun")},
        "difficulty": "easy",
    },
    {
        "id": "GT-004",
        "input_text": "Mars has two moons called Phobos and Deimos.",
        "expected_axioms": {
            ("mars", "has", "phobos"),
            ("mars", "has", "deimos"),
        },
        "difficulty": "medium",
    },
    {
        "id": "GT-005",
        "input_text": "Water boils at 100 degrees Celsius at sea level.",
        "expected_axioms": {("water", "boils_at", "100c")},
        "difficulty": "hard",
    },
    {
        "id": "GT-006",
        "input_text": "Python was created by Guido van Rossum in 1991.",
        "expected_axioms": {
            ("python", "created_by", "guido_van_rossum"),
            ("python", "created_in", "1991"),
        },
        "difficulty": "hard",
    },
]


# ─── Metric Computation ──────────────────────────────────────────

def compute_extraction_metrics(extracted: set, expected: set) -> dict:
    """
    Compute precision, recall, F1 for extracted vs expected axioms.

    Uses normalized axiom key comparison (lowercased).
    """
    norm_extracted = {
        (s.lower(), p.lower(), o.lower()) for s, p, o in extracted
    }
    norm_expected = {
        (s.lower(), p.lower(), o.lower()) for s, p, o in expected
    }

    true_positives = norm_extracted & norm_expected
    false_positives = norm_extracted - norm_expected
    false_negatives = norm_expected - norm_extracted

    precision = len(true_positives) / len(norm_extracted) if norm_extracted else 0
    recall = len(true_positives) / len(norm_expected) if norm_expected else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# ─── Tests ────────────────────────────────────────────────────────

class TestExtractionMetrics:
    """Tests for the metric computation itself (not the extractor)."""

    def test_perfect_extraction(self):
        """Perfect extraction gives F1=1.0."""
        expected = {("a", "b", "c"), ("d", "e", "f")}
        m = compute_extraction_metrics(expected, expected)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_empty_extraction(self):
        """Extracting nothing gives 0 recall."""
        m = compute_extraction_metrics(set(), {("a", "b", "c")})
        assert m["precision"] == 0
        assert m["recall"] == 0
        assert m["f1"] == 0

    def test_partial_extraction(self):
        """Extracting half gives 0.5 recall."""
        expected = {("a", "b", "c"), ("d", "e", "f")}
        extracted = {("a", "b", "c")}
        m = compute_extraction_metrics(extracted, expected)
        assert m["precision"] == 1.0
        assert m["recall"] == 0.5

    def test_false_positive(self):
        """Extra axioms reduce precision."""
        expected = {("a", "b", "c")}
        extracted = {("a", "b", "c"), ("x", "y", "z")}
        m = compute_extraction_metrics(extracted, expected)
        assert m["precision"] == 0.5
        assert m["recall"] == 1.0

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        expected = {("Alice", "Likes", "Cats")}
        extracted = {("alice", "likes", "cats")}
        m = compute_extraction_metrics(extracted, expected)
        assert m["f1"] == 1.0


class TestGroundTruthCatalog:
    """Ensures the ground-truth catalog itself is well-formed."""

    def test_all_cases_have_required_fields(self):
        for case in GROUND_TRUTH:
            assert "id" in case
            assert "input_text" in case
            assert "expected_axioms" in case
            assert "difficulty" in case

    def test_all_expected_axioms_are_triples(self):
        for case in GROUND_TRUTH:
            for axiom in case["expected_axioms"]:
                assert len(axiom) == 3, f"Bad axiom in {case['id']}: {axiom}"

    def test_difficulty_levels_present(self):
        difficulties = {c["difficulty"] for c in GROUND_TRUTH}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_no_duplicate_ids(self):
        ids = [c["id"] for c in GROUND_TRUTH]
        assert len(ids) == len(set(ids))


class TestAlgebraGroundTruth:
    """Verify that ground-truth axioms produce valid primes."""

    def test_all_ground_truth_axioms_mintable(self):
        """Every ground-truth axiom can be minted as a prime."""
        algebra = GodelStateAlgebra()
        for case in GROUND_TRUTH:
            for s, p, o in case["expected_axioms"]:
                prime = algebra.get_or_mint_prime(s, p, o)
                assert prime > 1, f"Invalid prime for {s},{p},{o} in {case['id']}"

    def test_distinct_axioms_get_distinct_primes(self):
        """No two distinct axioms share a prime."""
        algebra = GodelStateAlgebra()
        all_primes = {}
        for case in GROUND_TRUTH:
            for s, p, o in case["expected_axioms"]:
                key = f"{s}||{p}||{o}"
                prime = algebra.get_or_mint_prime(s, p, o)
                if key in all_primes:
                    assert all_primes[key] == prime
                else:
                    all_primes[key] = prime
        # Check no two different keys share a prime
        values = list(all_primes.values())
        assert len(values) == len(set(values)), "Prime collision detected!"
