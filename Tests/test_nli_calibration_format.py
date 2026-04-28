"""Format-validation tests for fixtures/nli_audit_calibration_v1.jsonl.

Pins the calibration fixture's schema. The implementation cycle for
v0.9.E (local ONNX NLI swap target) consumes this fixture verbatim;
a row that doesn't match the spec would silently fail the
calibration step and produce a wrong threshold. These tests catch
shape regressions at PR time.

The spec lives at docs/NLI_MODEL_REGISTRY.md §"Calibration fixture
format". This test file is the executable form of that section's
contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "nli_audit_calibration_v1.jsonl"
)

_REQUIRED_KEYS = {
    "schema",
    "fact",
    "fact_key",
    "passage",
    "embedding_score",
    "openai_verdict",
    "local_model_verdict",
    "adjudicated_label",
    "rationale",
    "source",
}

_VALID_VERDICTS = {"entails", "neutral", "contradicts"}


def _rows() -> list[dict]:
    """Read all rows from the calibration fixture; one row per
    non-empty line."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"calibration fixture not found at {_FIXTURE_PATH}")
    rows = []
    with _FIXTURE_PATH.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                pytest.fail(f"line {lineno} is not valid JSON: {e}")
    return rows


def test_fixture_exists_and_nonempty():
    """The starter set must have at least 3 rows so the format
    contract is exercised on multiple cases (entails, neutral,
    contradicts at minimum)."""
    rows = _rows()
    assert len(rows) >= 3, (
        f"calibration fixture has {len(rows)} rows; the starter set "
        f"is sized for format demonstration. Actual implementation-"
        f"phase calibration requires ≥30 rows per "
        f"docs/NLI_MODEL_REGISTRY.md §'Calibration set sizing'."
    )


def test_every_row_has_correct_schema():
    """Every row's schema field MUST be sum.nli_audit_calibration.v1.
    Mixed-version sets must be rejected."""
    for i, row in enumerate(_rows()):
        assert row.get("schema") == "sum.nli_audit_calibration.v1", (
            f"row {i}: schema={row.get('schema')!r}, "
            f"expected 'sum.nli_audit_calibration.v1'"
        )


def test_every_row_has_required_keys():
    """Every row carries the 10 required keys per the spec table."""
    for i, row in enumerate(_rows()):
        actual = set(row.keys())
        missing = _REQUIRED_KEYS - actual
        assert not missing, f"row {i} missing keys: {sorted(missing)}"


def test_fact_is_well_formed_triple():
    """fact field is a 3-tuple of non-empty strings; fact_key is the
    canonical s||p||o form."""
    for i, row in enumerate(_rows()):
        fact = row["fact"]
        assert isinstance(fact, list), f"row {i}: fact must be a list, got {type(fact).__name__}"
        assert len(fact) == 3, f"row {i}: fact must have 3 elements, got {len(fact)}"
        assert all(isinstance(x, str) and x for x in fact), (
            f"row {i}: fact components must be non-empty strings: {fact!r}"
        )
        expected_key = "||".join(fact)
        assert row["fact_key"] == expected_key, (
            f"row {i}: fact_key={row['fact_key']!r}, "
            f"expected {expected_key!r} (derived from fact)"
        )


def test_passage_is_nonempty_string():
    for i, row in enumerate(_rows()):
        passage = row["passage"]
        assert isinstance(passage, str) and passage, (
            f"row {i}: passage must be a non-empty string"
        )


def test_embedding_score_is_unit_float():
    for i, row in enumerate(_rows()):
        score = row["embedding_score"]
        assert isinstance(score, (int, float)), (
            f"row {i}: embedding_score must be numeric, got {type(score).__name__}"
        )
        assert 0.0 <= score <= 1.0, (
            f"row {i}: embedding_score={score} out of [0,1] range"
        )


def test_verdicts_are_from_taxonomy():
    """openai_verdict, local_model_verdict, and adjudicated_label
    each MUST be one of {entails, neutral, contradicts}, OR null
    for local_model_verdict (which is populated by the
    implementation cycle)."""
    for i, row in enumerate(_rows()):
        oa = row["openai_verdict"]
        local = row["local_model_verdict"]
        adj = row["adjudicated_label"]

        assert oa in _VALID_VERDICTS, (
            f"row {i}: openai_verdict={oa!r}, "
            f"must be one of {sorted(_VALID_VERDICTS)}"
        )
        assert adj in _VALID_VERDICTS, (
            f"row {i}: adjudicated_label={adj!r}, "
            f"must be one of {sorted(_VALID_VERDICTS)}"
        )
        # local_model_verdict is null until the implementation cycle
        # populates it. Once populated, must be in the taxonomy.
        assert local is None or local in _VALID_VERDICTS, (
            f"row {i}: local_model_verdict={local!r}, "
            f"must be null or one of {sorted(_VALID_VERDICTS)}"
        )


def test_rationale_is_meaningful_string():
    """Rationale must be present and non-trivial (≥10 chars). Helps
    audit tracing when verdicts disagree; an empty rationale defeats
    the purpose."""
    for i, row in enumerate(_rows()):
        rationale = row["rationale"]
        assert isinstance(rationale, str), (
            f"row {i}: rationale must be a string, got {type(rationale).__name__}"
        )
        assert len(rationale) >= 10, (
            f"row {i}: rationale too short ({len(rationale)} chars); "
            f"must be a meaningful one-sentence justification"
        )


def test_source_documents_origin():
    """source field documents where the row came from. Lets a future
    review fetch the original render context."""
    for i, row in enumerate(_rows()):
        source = row["source"]
        assert isinstance(source, str) and source, (
            f"row {i}: source must be a non-empty string documenting origin"
        )


def test_taxonomy_diversity_in_starter_set():
    """The starter set should exercise multiple adjudicated labels
    (entails, neutral, contradicts) so consumers see how each
    verdict shape is represented. This is a soft check — not all
    labels must appear, but the format-demo set should at least
    show the variety."""
    labels_seen = {row["adjudicated_label"] for row in _rows()}
    assert len(labels_seen) >= 2, (
        f"starter set only exercises one adjudicated_label "
        f"({labels_seen}); format demo should show at least two of "
        f"the three taxonomy values"
    )
