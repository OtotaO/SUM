"""Contract tests for the GDPR Article 30 validator.

Mirrors the EU AI Act Article 12 test layout (per-rule negative +
clean-pass + e2e through the real CLI). The substrate's value
compounds when there are 2+ regimes consuming the same shape; this
file is the second instance proving the
``sum.compliance_report.v1`` shape held without modification.

Every rule R1..R5 has at least one negative test pinning the
violation message + rule_id, plus a clean-pass test confirming the
rule doesn't false-positive on well-formed rows.
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from sum_engine_internal.compliance import gdpr_article_30 as gv
from sum_engine_internal.compliance.report import ValidationReport


# ─── Helpers ──────────────────────────────────────────────────────────


def _good_attest_row(**overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T04:00:00.123Z",
        "operation": "attest",
        "cli_version": "0.5.0",
        "source_uri": "sha256:abc123",
        "axiom_count": 3,
        "state_integer_digits": 57,
        "extractor": "sieve",
        "branch": "main",
        "signed": False,
        "hmac": False,
        "input_format": "plaintext",
    }
    row.update(overrides)
    return row


def _good_verify_row(**overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T04:00:01.234Z",
        "operation": "verify",
        "cli_version": "0.5.0",
        "ok": True,
        "axiom_count": 3,
        "state_integer_digits": 57,
        "branch": "main",
        "signatures": {"hmac": "absent", "ed25519": "absent"},
    }
    row.update(overrides)
    return row


def _good_render_row(mode: str = "local-deterministic", **overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T04:00:02.345Z",
        "operation": "render",
        "cli_version": "0.5.0",
        "mode": mode,
        "axiom_count_input": 3,
        "tome_chars": 200,
    }
    row.update(overrides)
    return row


# ─── Clean-pass cases ─────────────────────────────────────────────────


def test_clean_audit_log_passes_cleanly():
    """A well-formed audit log with one of each operation passes
    every Art 30 per-row floor rule. Note that Art 30 is *thinner*
    than Art 12 at the per-row level — Art 12 needs operation-
    specific anchors (source_uri, axiom_count, mode); Art 30's
    per-row floor only needs the four shared traceability fields
    (timestamp, operation, cli_version, schema). So a row that
    passes Art 12 trivially passes Art 30."""
    rows = [_good_attest_row(), _good_verify_row(), _good_render_row()]
    report = gv.validate(rows)
    assert isinstance(report, ValidationReport)
    assert report.ok, f"clean log should pass; got {report.violations}"
    assert report.violation_count == 0
    assert report.regime == "gdpr-article-30"
    assert report.rows_examined == 3


def test_empty_audit_log_passes_cleanly():
    """An empty stream has no rows to violate. Art 30 *does* require
    the controller to maintain records — but an empty stream is a
    deployment-level concern (no processing happened in this period),
    not a per-row violation. Mirrors Art 12's empty-log behaviour."""
    report = gv.validate([])
    assert report.ok
    assert report.rows_examined == 0


def test_clean_log_with_minimum_floor_passes():
    """A row with ONLY the Art 30 per-row floor fields (no Art 12
    operation-specific anchors) still passes Art 30. This is the
    "shape generalizes" assertion: Art 30's floor is a strict
    subset of Art 12's required fields, so the validator must
    accept rows that lack Art 12-specific anchors."""
    minimal = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T04:00:00Z",
        "operation": "custom_processing",  # not an Art 12 operation
        "cli_version": "0.5.0",
    }
    report = gv.validate([minimal])
    assert report.ok, (
        f"Art 30 per-row floor should accept minimal rows that don't "
        f"meet Art 12 operation-specific anchors; got "
        f"{report.violations}"
    )


# ─── R1: schema-pinned ────────────────────────────────────────────────


def test_r1_wrong_schema_flagged():
    bad = _good_attest_row(schema="some.other.schema")
    report = gv.validate([bad])
    assert not report.ok
    rule_ids = {v.rule_id for v in report.violations}
    assert "gdpr-art-30.schema-pinned" in rule_ids


def test_r1_missing_schema_flagged():
    bad = _good_attest_row()
    del bad["schema"]
    report = gv.validate([bad])
    assert "gdpr-art-30.schema-pinned" in {v.rule_id for v in report.violations}


def test_non_dict_row_flagged_under_r1():
    """Fail-open behaviour — a non-dict row surfaces as a violation,
    not a Python exception."""
    report = gv.validate(["not a dict", 42, None])
    assert report.violation_count == 3
    for v in report.violations:
        assert v.rule_id == "gdpr-art-30.schema-pinned"
        assert "not a dict" in v.message


# ─── R2: timestamp-present ────────────────────────────────────────────


def test_r2_missing_timestamp_flagged():
    bad = _good_attest_row()
    del bad["timestamp"]
    report = gv.validate([bad])
    assert "gdpr-art-30.timestamp-present" in {v.rule_id for v in report.violations}


def test_r2_empty_timestamp_flagged():
    bad = _good_attest_row(timestamp="")
    report = gv.validate([bad])
    assert "gdpr-art-30.timestamp-present" in {v.rule_id for v in report.violations}


def test_r2_null_timestamp_flagged():
    bad = _good_attest_row(timestamp=None)
    report = gv.validate([bad])
    assert "gdpr-art-30.timestamp-present" in {v.rule_id for v in report.violations}


# ─── R3: timestamp-iso8601-utc ────────────────────────────────────────


def test_r3_non_iso_timestamp_flagged():
    bad = _good_attest_row(timestamp="May 3, 2026")
    report = gv.validate([bad])
    assert "gdpr-art-30.timestamp-iso8601-utc" in {v.rule_id for v in report.violations}


def test_r3_iso_without_z_suffix_flagged():
    bad = _good_attest_row(timestamp="2026-05-03T04:00:00+02:00")
    report = gv.validate([bad])
    rule_ids = {v.rule_id for v in report.violations}
    assert "gdpr-art-30.timestamp-iso8601-utc" in rule_ids


def test_r3_z_suffix_but_unparseable_body_flagged():
    bad = _good_attest_row(timestamp="not-a-timestamp-Z")
    report = gv.validate([bad])
    assert "gdpr-art-30.timestamp-iso8601-utc" in {v.rule_id for v in report.violations}


def test_r3_well_formed_timestamp_passes_r3():
    """Sanity: the canonical ISO-UTC timestamps used by SUM's CLI
    don't trip R3."""
    good = _good_attest_row(timestamp="2026-05-03T04:00:00.123Z")
    report = gv.validate([good])
    assert "gdpr-art-30.timestamp-iso8601-utc" not in {
        v.rule_id for v in report.violations
    }


def test_r3_does_not_double_count_when_r2_already_fires():
    """If timestamp is missing entirely, R2 fires but R3 should NOT
    (no value to validate). Avoids piling violations on a single
    defect — same audit-tightening pattern as Art 12's R3."""
    bad = _good_attest_row()
    del bad["timestamp"]
    report = gv.validate([bad])
    rule_ids = {v.rule_id for v in report.violations}
    assert "gdpr-art-30.timestamp-present" in rule_ids
    assert "gdpr-art-30.timestamp-iso8601-utc" not in rule_ids, (
        "R3 should not fire when R2 already caught a missing timestamp"
    )


# ─── R4: processing-category-present ──────────────────────────────────


def test_r4_missing_operation_flagged():
    bad = _good_attest_row()
    del bad["operation"]
    report = gv.validate([bad])
    assert "gdpr-art-30.processing-category-present" in {
        v.rule_id for v in report.violations
    }


def test_r4_empty_operation_flagged():
    bad = _good_attest_row(operation="")
    report = gv.validate([bad])
    assert "gdpr-art-30.processing-category-present" in {
        v.rule_id for v in report.violations
    }


def test_r4_null_operation_flagged():
    bad = _good_attest_row(operation=None)
    report = gv.validate([bad])
    assert "gdpr-art-30.processing-category-present" in {
        v.rule_id for v in report.violations
    }


# ─── R5: processor-identity-present ───────────────────────────────────


def test_r5_missing_cli_version_flagged():
    bad = _good_attest_row()
    del bad["cli_version"]
    report = gv.validate([bad])
    assert "gdpr-art-30.processor-identity-present" in {
        v.rule_id for v in report.violations
    }


def test_r5_empty_cli_version_flagged():
    bad = _good_attest_row(cli_version="")
    report = gv.validate([bad])
    assert "gdpr-art-30.processor-identity-present" in {
        v.rule_id for v in report.violations
    }


def test_r5_null_cli_version_flagged():
    bad = _good_attest_row(cli_version=None)
    report = gv.validate([bad])
    assert "gdpr-art-30.processor-identity-present" in {
        v.rule_id for v in report.violations
    }


# ─── Report shape sanity ──────────────────────────────────────────────


def test_report_to_dict_carries_schema_v1():
    report = gv.validate([_good_attest_row()])
    d = report.to_dict()
    assert d["schema"] == "sum.compliance_report.v1"
    assert d["regime"] == "gdpr-article-30"
    assert "violations_by_rule" in d
    assert "violations" in d


def test_report_violations_by_rule_aggregates_correctly():
    """Multiple rows with overlapping defects aggregate per rule."""
    rows = [
        _good_attest_row(schema="bad"),                    # R1
        _good_attest_row(timestamp=None),                  # R2
        _good_attest_row(timestamp="bad"),                 # R3
        _good_attest_row(operation=None, schema="bad"),    # R1 + R4
    ]
    report = gv.validate(rows)
    by_rule = report.violations_by_rule()
    assert by_rule["gdpr-art-30.schema-pinned"] == 2
    assert by_rule["gdpr-art-30.timestamp-present"] == 1
    assert by_rule["gdpr-art-30.timestamp-iso8601-utc"] == 1
    assert by_rule["gdpr-art-30.processing-category-present"] == 1


def test_violation_carries_row_index():
    rows = [
        _good_attest_row(),                    # row 0 — clean
        _good_attest_row(schema="oops"),       # row 1 — R1 violation
        _good_attest_row(),                    # row 2 — clean
    ]
    report = gv.validate(rows)
    assert report.violation_count == 1
    assert report.violations[0].row_index == 1
    assert report.violations[0].operation == "attest"


# ─── Cross-regime contract: ValidationReport shape is shared ──────────


def test_validation_report_shape_matches_eu_ai_act():
    """The substrate-tightening proof: GDPR Art 30 returns a
    ``ValidationReport`` byte-shape-identical to EU AI Act Art 12's.
    Both regimes consume the same audit log; both return the same
    report shape; only ``regime`` and ``rule_id`` strings differ.
    Downstream consumers can ingest reports across regimes without
    per-regime adapters — proven here, not just documented."""
    from sum_engine_internal.compliance import eu_ai_act_article_12 as ev

    rows = [_good_attest_row(schema="bad")]
    art_12_report = ev.validate(rows)
    art_30_report = gv.validate(rows)

    art_12_dict = art_12_report.to_dict()
    art_30_dict = art_30_report.to_dict()

    # Same top-level keys
    assert set(art_12_dict.keys()) == set(art_30_dict.keys()), (
        f"ValidationReport shape diverged across regimes: "
        f"art_12 keys={sorted(art_12_dict)}, art_30 keys={sorted(art_30_dict)}"
    )
    # Same schema string (regime-agnostic substrate)
    assert art_12_dict["schema"] == art_30_dict["schema"] == "sum.compliance_report.v1"
    # Different regime strings
    assert art_12_dict["regime"] != art_30_dict["regime"]
    # Each violation has the same structural shape
    if art_12_report.violations and art_30_report.violations:
        v12 = art_12_report.violations[0]
        v30 = art_30_report.violations[0]
        from dataclasses import fields
        assert {f.name for f in fields(v12)} == {f.name for f in fields(v30)}


# ─── End-to-end: real audit log produced by sum CLI passes ────────────


def test_real_sum_cli_audit_log_passes_validation(tmp_path, monkeypatch):
    """Drive the SUM CLI with SUM_AUDIT_LOG set, then validate the
    resulting JSONL against Article 30. A real audit log produced
    by the shipping CLI must pass Art 30 cleanly. Mirrors the Art 12
    end-to-end test pattern."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.main import cmd_attest, cmd_verify, cmd_render
    monkeypatch.setattr("sys.stdin", io.StringIO(
        "Alice likes cats. Bob owns a dog. Carol writes code."
    ))
    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="art-30-test", title="Article 30 Test",
        signing_key=None, ed25519_key=None, ledger=None,
        format="auto", pretty=False, verbose=False,
    )
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        assert cmd_attest(args) == 0
    finally:
        sys.stdout = old
    bundle = json.loads(out.getvalue())
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    sys.stdout = io.StringIO()
    try:
        cmd_verify(argparse.Namespace(
            input=str(bundle_path), signing_key=None, strict=False, pretty=False,
        ))
    finally:
        sys.stdout = old

    sys.stdout = io.StringIO()
    try:
        cmd_render(argparse.Namespace(
            input=str(bundle_path),
            density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5,
            title="art-30-test",
            output=None, use_worker=None,
            json=False, pretty=False, verbose=False,
        ))
    finally:
        sys.stdout = old

    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 3

    report = gv.validate(rows)
    assert report.ok, (
        f"shipping CLI's audit log must pass Art 30 cleanly; "
        f"violations: {[v.message for v in report.violations]}"
    )
