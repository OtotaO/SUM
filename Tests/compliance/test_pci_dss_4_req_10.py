"""Contract tests for the PCI DSS v4.0 Requirement 10 validator.

Same shape as HIPAA § 164.312(b) (R1–R6 with operation-specific
anchors at R6); pins PCI-DSS-specific rule_id strings + statutory
message anchors. Sixth and final regime in the record-keeping
shape slate from Priority 11.
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from sum_engine_internal.compliance import pci_dss_4_req_10 as pv
from sum_engine_internal.compliance.report import ValidationReport


def _good_attest_row(**overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T08:00:00.123Z",
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
        "timestamp": "2026-05-03T08:00:01.234Z",
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
        "timestamp": "2026-05-03T08:00:02.345Z",
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
    report = pv.validate(rows)
    assert isinstance(report, ValidationReport)
    assert report.ok, f"clean log should pass; got {report.violations}"
    assert report.regime == "pci-dss-4-req-10"
    assert report.rows_examined == 3


def test_empty_audit_log_passes_cleanly():
    report = pv.validate([])
    assert report.ok and report.rows_examined == 0


def test_verify_with_ok_false_is_compliant():
    """Failed outcomes are still RECORDED outcomes — Req 10.2.2
    'success/failure indication' requires presence, not truthy."""
    rows = [_good_verify_row(ok=False)]
    report = pv.validate(rows)
    assert "pci-dss-4-req-10.event-content-completeness" not in {
        v.rule_id for v in report.violations
    }


# ─── R1: schema-pinned ────────────────────────────────────────────────


def test_r1_wrong_schema_flagged():
    bad = _good_attest_row(schema="x")
    assert "pci-dss-4-req-10.schema-pinned" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r1_missing_schema_flagged():
    bad = _good_attest_row(); del bad["schema"]
    assert "pci-dss-4-req-10.schema-pinned" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_non_dict_row_flagged_under_r1():
    report = pv.validate(["x", 0, None])
    assert report.violation_count == 3
    for v in report.violations:
        assert v.rule_id == "pci-dss-4-req-10.schema-pinned"


# ─── R2: timestamp-present ────────────────────────────────────────────


def test_r2_missing_timestamp_flagged():
    bad = _good_attest_row(); del bad["timestamp"]
    assert "pci-dss-4-req-10.timestamp-present" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r2_empty_timestamp_flagged():
    assert "pci-dss-4-req-10.timestamp-present" in {
        v.rule_id for v in pv.validate([_good_attest_row(timestamp="")]).violations
    }


# ─── R3: timestamp-iso8601-utc ────────────────────────────────────────


def test_r3_non_iso_timestamp_flagged():
    assert "pci-dss-4-req-10.timestamp-iso8601-utc" in {
        v.rule_id for v in pv.validate([_good_attest_row(timestamp="x")]).violations
    }


def test_r3_iso_without_z_suffix_flagged():
    assert "pci-dss-4-req-10.timestamp-iso8601-utc" in {
        v.rule_id
        for v in pv.validate([_good_attest_row(timestamp="2026-05-03T08:00:00+02:00")]).violations
    }


def test_r3_does_not_double_count_when_r2_already_fires():
    bad = _good_attest_row(); del bad["timestamp"]
    rule_ids = {v.rule_id for v in pv.validate([bad]).violations}
    assert "pci-dss-4-req-10.timestamp-present" in rule_ids
    assert "pci-dss-4-req-10.timestamp-iso8601-utc" not in rule_ids


# ─── R4: event-type-recorded ──────────────────────────────────────────


def test_r4_missing_operation_flagged():
    bad = _good_attest_row(); del bad["operation"]
    assert "pci-dss-4-req-10.event-type-recorded" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r4_empty_operation_flagged():
    assert "pci-dss-4-req-10.event-type-recorded" in {
        v.rule_id for v in pv.validate([_good_attest_row(operation="")]).violations
    }


# ─── R5: origination-identified ───────────────────────────────────────


def test_r5_missing_cli_version_flagged():
    bad = _good_attest_row(); del bad["cli_version"]
    assert "pci-dss-4-req-10.origination-identified" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r5_empty_cli_version_flagged():
    assert "pci-dss-4-req-10.origination-identified" in {
        v.rule_id for v in pv.validate([_good_attest_row(cli_version="")]).violations
    }


# ─── R6: event-content-completeness ───────────────────────────────────


def test_r6_attest_missing_source_uri_flagged():
    bad = _good_attest_row(); del bad["source_uri"]
    assert "pci-dss-4-req-10.event-content-completeness" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r6_attest_empty_source_uri_flagged():
    assert "pci-dss-4-req-10.event-content-completeness" in {
        v.rule_id for v in pv.validate([_good_attest_row(source_uri="")]).violations
    }


def test_r6_verify_missing_ok_flagged():
    bad = _good_verify_row(); del bad["ok"]
    assert "pci-dss-4-req-10.event-content-completeness" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r6_render_invalid_mode_flagged():
    bad = _good_render_row(); bad["mode"] = "unknown"
    assert "pci-dss-4-req-10.event-content-completeness" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r6_render_missing_mode_flagged():
    bad = _good_render_row(); del bad["mode"]
    assert "pci-dss-4-req-10.event-content-completeness" in {
        v.rule_id for v in pv.validate([bad]).violations
    }


def test_r6_does_not_apply_to_unknown_operation():
    """An unknown operation type passes R6 silently; R4 enforces
    operation presence so unknown-but-present operations satisfy
    the per-row floor."""
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T08:00:00Z",
        "operation": "future_op",
        "cli_version": "0.5.0",
    }
    report = pv.validate([row])
    assert "pci-dss-4-req-10.event-content-completeness" not in {
        v.rule_id for v in report.violations
    }


# ─── Report shape ─────────────────────────────────────────────────────


def test_report_to_dict_carries_schema_v1():
    d = pv.validate([_good_attest_row()]).to_dict()
    assert d["schema"] == "sum.compliance_report.v1"
    assert d["regime"] == "pci-dss-4-req-10"


def test_violation_carries_row_index():
    rows = [_good_attest_row(), _good_attest_row(schema="oops"), _good_attest_row()]
    report = pv.validate(rows)
    assert report.violation_count == 1
    assert report.violations[0].row_index == 1


# ─── Cross-regime substrate proof (sixth instance) ────────────────────


def test_validation_report_shape_matches_other_regimes():
    """Sixth-regime instance proof. Six regimes return byte-shape-
    identical sum.compliance_report.v1; only `regime` and `rule_id`
    strings differ. The substrate's regime-agnosticism is settled
    empirical fact at this point."""
    from sum_engine_internal.compliance import (
        eu_ai_act_article_12 as ev,
        gdpr_article_30 as gv,
        hipaa_164_312_b as hv,
        iso_27001_8_15 as iv,
        soc_2_cc_7_2 as sv,
    )
    rows = [_good_attest_row(schema="bad")]
    dicts = [
        ev.validate(rows).to_dict(),
        gv.validate(rows).to_dict(),
        hv.validate(rows).to_dict(),
        iv.validate(rows).to_dict(),
        sv.validate(rows).to_dict(),
        pv.validate(rows).to_dict(),
    ]
    keys = {frozenset(d.keys()) for d in dicts}
    assert len(keys) == 1, "ValidationReport shape diverged across 6 regimes"
    schemas = {d["schema"] for d in dicts}
    assert schemas == {"sum.compliance_report.v1"}
    regimes = {d["regime"] for d in dicts}
    assert len(regimes) == 6


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
        branch="pci-test", title="PCI Test",
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
            audience=0.5, perspective=0.5, title="pci-test",
            output=None, use_worker=None, json=False, pretty=False, verbose=False,
        ))
    finally: sys.stdout = old

    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    report = pv.validate(rows)
    assert report.ok, f"violations: {[v.message for v in report.violations]}"
