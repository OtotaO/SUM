"""Contract tests for the SOC 2 CC7.2 System Operations validator.

Same shape as ISO 27001 A.8.15 (R1–R5 record-keeping floor); pins
SOC-2-specific rule_id strings + statutory message anchors.
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from sum_engine_internal.compliance import soc_2_cc_7_2 as sv
from sum_engine_internal.compliance.report import ValidationReport


def _good_attest_row(**overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T07:00:00.123Z",
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
        "timestamp": "2026-05-03T07:00:01.234Z",
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
        "timestamp": "2026-05-03T07:00:02.345Z",
        "operation": "render",
        "cli_version": "0.5.0",
        "mode": mode,
        "axiom_count_input": 3,
        "tome_chars": 200,
    }
    row.update(overrides)
    return row


def test_clean_audit_log_passes_cleanly():
    rows = [_good_attest_row(), _good_verify_row(), _good_render_row()]
    report = sv.validate(rows)
    assert report.ok and report.regime == "soc-2-cc-7-2" and report.rows_examined == 3


def test_empty_audit_log_passes_cleanly():
    report = sv.validate([])
    assert report.ok and report.rows_examined == 0


def test_minimum_floor_row_passes():
    minimal = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T07:00:00Z",
        "operation": "monitored_activity",
        "cli_version": "0.5.0",
    }
    assert sv.validate([minimal]).ok


# ─── R1 ───────────────────────────────────────────────────────────────


def test_r1_wrong_schema_flagged():
    assert "soc-2-cc-7-2.schema-pinned" in {
        v.rule_id for v in sv.validate([_good_attest_row(schema="x")]).violations
    }


def test_r1_missing_schema_flagged():
    bad = _good_attest_row(); del bad["schema"]
    assert "soc-2-cc-7-2.schema-pinned" in {v.rule_id for v in sv.validate([bad]).violations}


def test_non_dict_row_flagged_under_r1():
    report = sv.validate(["x", 0, None])
    assert report.violation_count == 3
    for v in report.violations:
        assert v.rule_id == "soc-2-cc-7-2.schema-pinned"


# ─── R2 ───────────────────────────────────────────────────────────────


def test_r2_missing_timestamp_flagged():
    bad = _good_attest_row(); del bad["timestamp"]
    assert "soc-2-cc-7-2.timestamp-present" in {
        v.rule_id for v in sv.validate([bad]).violations
    }


def test_r2_empty_timestamp_flagged():
    assert "soc-2-cc-7-2.timestamp-present" in {
        v.rule_id for v in sv.validate([_good_attest_row(timestamp="")]).violations
    }


# ─── R3 ───────────────────────────────────────────────────────────────


def test_r3_non_iso_timestamp_flagged():
    assert "soc-2-cc-7-2.timestamp-iso8601-utc" in {
        v.rule_id for v in sv.validate([_good_attest_row(timestamp="x")]).violations
    }


def test_r3_iso_without_z_suffix_flagged():
    assert "soc-2-cc-7-2.timestamp-iso8601-utc" in {
        v.rule_id for v in sv.validate([_good_attest_row(timestamp="2026-05-03T07:00:00+02:00")]).violations
    }


def test_r3_does_not_double_count_when_r2_already_fires():
    bad = _good_attest_row(); del bad["timestamp"]
    rule_ids = {v.rule_id for v in sv.validate([bad]).violations}
    assert "soc-2-cc-7-2.timestamp-present" in rule_ids
    assert "soc-2-cc-7-2.timestamp-iso8601-utc" not in rule_ids


# ─── R4 ───────────────────────────────────────────────────────────────


def test_r4_missing_operation_flagged():
    bad = _good_attest_row(); del bad["operation"]
    assert "soc-2-cc-7-2.activity-classified" in {
        v.rule_id for v in sv.validate([bad]).violations
    }


def test_r4_empty_operation_flagged():
    assert "soc-2-cc-7-2.activity-classified" in {
        v.rule_id for v in sv.validate([_good_attest_row(operation="")]).violations
    }


# ─── R5 ───────────────────────────────────────────────────────────────


def test_r5_missing_cli_version_flagged():
    bad = _good_attest_row(); del bad["cli_version"]
    assert "soc-2-cc-7-2.system-component-identified" in {
        v.rule_id for v in sv.validate([bad]).violations
    }


def test_r5_empty_cli_version_flagged():
    assert "soc-2-cc-7-2.system-component-identified" in {
        v.rule_id for v in sv.validate([_good_attest_row(cli_version="")]).violations
    }


# ─── Report shape ─────────────────────────────────────────────────────


def test_report_to_dict_carries_schema_v1():
    d = sv.validate([_good_attest_row()]).to_dict()
    assert d["schema"] == "sum.compliance_report.v1"
    assert d["regime"] == "soc-2-cc-7-2"


def test_violation_carries_row_index():
    rows = [_good_attest_row(), _good_attest_row(schema="oops"), _good_attest_row()]
    report = sv.validate(rows)
    assert report.violation_count == 1
    assert report.violations[0].row_index == 1


# ─── Cross-regime substrate proof (fifth instance) ────────────────────


def test_validation_report_shape_matches_other_regimes():
    """Fifth-regime instance proof. Five regimes return byte-shape-
    identical sum.compliance_report.v1; only `regime` and `rule_id`
    strings differ."""
    from sum_engine_internal.compliance import (
        eu_ai_act_article_12 as ev,
        gdpr_article_30 as gv,
        hipaa_164_312_b as hv,
        iso_27001_8_15 as iv,
    )
    rows = [_good_attest_row(schema="bad")]
    dicts = [
        ev.validate(rows).to_dict(),
        gv.validate(rows).to_dict(),
        hv.validate(rows).to_dict(),
        iv.validate(rows).to_dict(),
        sv.validate(rows).to_dict(),
    ]
    keys = {frozenset(d.keys()) for d in dicts}
    assert len(keys) == 1, "ValidationReport shape diverged across 5 regimes"
    schemas = {d["schema"] for d in dicts}
    assert schemas == {"sum.compliance_report.v1"}
    regimes = {d["regime"] for d in dicts}
    assert len(regimes) == 5


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
        branch="soc-test", title="SOC Test",
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
            audience=0.5, perspective=0.5, title="soc-test",
            output=None, use_worker=None, json=False, pretty=False, verbose=False,
        ))
    finally: sys.stdout = old

    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    report = sv.validate(rows)
    assert report.ok, f"violations: {[v.message for v in report.violations]}"
