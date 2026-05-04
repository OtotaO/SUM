"""Contract tests for the ISO/IEC 27001:2022 A.8.15 Logging validator.

Same shape as GDPR Art 30 (R1–R5 record-keeping floor); pins
ISO-specific rule_id strings + statutory message anchors.
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from sum_engine_internal.compliance import iso_27001_8_15 as iv
from sum_engine_internal.compliance.report import ValidationReport


def _good_attest_row(**overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T06:00:00.123Z",
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
        "timestamp": "2026-05-03T06:00:01.234Z",
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
        "timestamp": "2026-05-03T06:00:02.345Z",
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
    rows = [_good_attest_row(), _good_verify_row(), _good_render_row()]
    report = iv.validate(rows)
    assert isinstance(report, ValidationReport)
    assert report.ok, f"clean log should pass; got {report.violations}"
    assert report.regime == "iso-27001-8-15"
    assert report.rows_examined == 3


def test_empty_audit_log_passes_cleanly():
    report = iv.validate([])
    assert report.ok and report.rows_examined == 0


def test_minimum_floor_row_passes():
    """A row with only the A.8.15 per-row floor (no operation-
    specific anchors) passes."""
    minimal = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T06:00:00Z",
        "operation": "any_activity",
        "cli_version": "0.5.0",
    }
    assert iv.validate([minimal]).ok


# ─── R1: schema-pinned ────────────────────────────────────────────────


def test_r1_wrong_schema_flagged():
    bad = _good_attest_row(schema="x")
    report = iv.validate([bad])
    assert "iso-27001-8-15.schema-pinned" in {v.rule_id for v in report.violations}


def test_r1_missing_schema_flagged():
    bad = _good_attest_row(); del bad["schema"]
    assert "iso-27001-8-15.schema-pinned" in {v.rule_id for v in iv.validate([bad]).violations}


def test_non_dict_row_flagged_under_r1():
    report = iv.validate(["nope", 0, None])
    assert report.violation_count == 3
    for v in report.violations:
        assert v.rule_id == "iso-27001-8-15.schema-pinned"


# ─── R2: timestamp-present ────────────────────────────────────────────


def test_r2_missing_timestamp_flagged():
    bad = _good_attest_row(); del bad["timestamp"]
    assert "iso-27001-8-15.timestamp-present" in {
        v.rule_id for v in iv.validate([bad]).violations
    }


def test_r2_empty_timestamp_flagged():
    assert "iso-27001-8-15.timestamp-present" in {
        v.rule_id for v in iv.validate([_good_attest_row(timestamp="")]).violations
    }


# ─── R3: timestamp-iso8601-utc ────────────────────────────────────────


def test_r3_non_iso_timestamp_flagged():
    assert "iso-27001-8-15.timestamp-iso8601-utc" in {
        v.rule_id for v in iv.validate([_good_attest_row(timestamp="May 3")]).violations
    }


def test_r3_iso_without_z_suffix_flagged():
    assert "iso-27001-8-15.timestamp-iso8601-utc" in {
        v.rule_id
        for v in iv.validate([_good_attest_row(timestamp="2026-05-03T06:00:00+02:00")]).violations
    }


def test_r3_does_not_double_count_when_r2_already_fires():
    bad = _good_attest_row(); del bad["timestamp"]
    rule_ids = {v.rule_id for v in iv.validate([bad]).violations}
    assert "iso-27001-8-15.timestamp-present" in rule_ids
    assert "iso-27001-8-15.timestamp-iso8601-utc" not in rule_ids


# ─── R4: activity-recorded ────────────────────────────────────────────


def test_r4_missing_operation_flagged():
    bad = _good_attest_row(); del bad["operation"]
    assert "iso-27001-8-15.activity-recorded" in {
        v.rule_id for v in iv.validate([bad]).violations
    }


def test_r4_empty_operation_flagged():
    assert "iso-27001-8-15.activity-recorded" in {
        v.rule_id for v in iv.validate([_good_attest_row(operation="")]).violations
    }


# ─── R5: system-component-identified ──────────────────────────────────


def test_r5_missing_cli_version_flagged():
    bad = _good_attest_row(); del bad["cli_version"]
    assert "iso-27001-8-15.system-component-identified" in {
        v.rule_id for v in iv.validate([bad]).violations
    }


def test_r5_empty_cli_version_flagged():
    assert "iso-27001-8-15.system-component-identified" in {
        v.rule_id for v in iv.validate([_good_attest_row(cli_version="")]).violations
    }


# ─── Report shape ─────────────────────────────────────────────────────


def test_report_to_dict_carries_schema_v1():
    d = iv.validate([_good_attest_row()]).to_dict()
    assert d["schema"] == "sum.compliance_report.v1"
    assert d["regime"] == "iso-27001-8-15"


def test_violation_carries_row_index():
    rows = [_good_attest_row(), _good_attest_row(schema="oops"), _good_attest_row()]
    report = iv.validate(rows)
    assert report.violation_count == 1
    assert report.violations[0].row_index == 1


# ─── Cross-regime substrate proof (fourth instance) ───────────────────


def test_validation_report_shape_matches_other_regimes():
    """Fourth-regime instance proof. ValidationReport shape +
    Violation field set are identical across all four regimes.
    """
    from sum_engine_internal.compliance import (
        eu_ai_act_article_12 as ev,
        gdpr_article_30 as gv,
        hipaa_164_312_b as hv,
    )
    rows = [_good_attest_row(schema="bad")]
    dicts = [
        ev.validate(rows).to_dict(),
        gv.validate(rows).to_dict(),
        hv.validate(rows).to_dict(),
        iv.validate(rows).to_dict(),
    ]
    keys = {frozenset(d.keys()) for d in dicts}
    assert len(keys) == 1, "ValidationReport shape diverged across 4 regimes"
    schemas = {d["schema"] for d in dicts}
    assert schemas == {"sum.compliance_report.v1"}
    regimes = {d["regime"] for d in dicts}
    assert len(regimes) == 4


# ─── End-to-end: real audit log ───────────────────────────────────────


def test_real_sum_cli_audit_log_passes_validation(tmp_path, monkeypatch):
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.main import cmd_attest, cmd_verify, cmd_render
    monkeypatch.setattr("sys.stdin", io.StringIO(
        "Alice likes cats. Bob owns a dog. Carol writes code."
    ))
    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="iso-test", title="ISO Test",
        signing_key=None, ed25519_key=None, ledger=None,
        format="auto", pretty=False, verbose=False,
    )
    out = io.StringIO(); old = sys.stdout; sys.stdout = out
    try: assert cmd_attest(args) == 0
    finally: sys.stdout = old
    bundle = json.loads(out.getvalue())
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    sys.stdout = io.StringIO()
    try:
        cmd_verify(argparse.Namespace(input=str(bundle_path), signing_key=None, strict=False, pretty=False))
    finally: sys.stdout = old

    sys.stdout = io.StringIO()
    try:
        cmd_render(argparse.Namespace(
            input=str(bundle_path), density=1.0, length=0.5, formality=0.5,
            audience=0.5, perspective=0.5, title="iso-test",
            output=None, use_worker=None, json=False, pretty=False, verbose=False,
        ))
    finally: sys.stdout = old

    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    report = iv.validate(rows)
    assert report.ok, f"violations: {[v.message for v in report.violations]}"
