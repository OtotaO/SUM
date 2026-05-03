"""Contract tests for the HIPAA § 164.312(b) Audit Controls validator.

Mirrors the EU AI Act Art 12 + GDPR Art 30 test layouts (per-rule
negative + clean-pass + e2e). HIPAA is the third regime to consume
``sum.compliance_report.v1``; the C1/C2/C3 cross-regime contracts
in ``test_cli_dispatch.py`` extend automatically once HIPAA is
registered.
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from sum_engine_internal.compliance import hipaa_164_312_b as hv
from sum_engine_internal.compliance.report import ValidationReport


# ─── Helpers ──────────────────────────────────────────────────────────


def _good_attest_row(**overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T05:00:00.123Z",
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
        "timestamp": "2026-05-03T05:00:01.234Z",
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
        "timestamp": "2026-05-03T05:00:02.345Z",
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
    report = hv.validate(rows)
    assert isinstance(report, ValidationReport)
    assert report.ok, f"clean log should pass; got {report.violations}"
    assert report.violation_count == 0
    assert report.regime == "hipaa-164-312-b"
    assert report.rows_examined == 3


def test_empty_audit_log_passes_cleanly():
    """Empty stream — nothing to violate. Whether HIPAA *requires*
    any logs is a separate (deployment-level) concern."""
    report = hv.validate([])
    assert report.ok
    assert report.rows_examined == 0


def test_clean_log_with_worker_mode_render_passes():
    rows = [_good_render_row(mode="worker", render_receipt_kid="kid-2026-05-03")]
    report = hv.validate(rows)
    assert report.ok


def test_verify_with_ok_false_is_examinable():
    """A failed verification is still EXAMINABLE — the auditor can
    see it failed. R6 requires `ok` to be PRESENT, not necessarily
    True. A verification that returns False is recorded compliantly."""
    rows = [_good_verify_row(ok=False)]
    report = hv.validate(rows)
    assert report.ok, (
        f"verify with ok=False should pass R6 (presence not truthy); "
        f"violations: {report.violations}"
    )


# ─── R1: schema-pinned ────────────────────────────────────────────────


def test_r1_wrong_schema_flagged():
    bad = _good_attest_row(schema="some.other.schema")
    report = hv.validate([bad])
    assert "hipaa-164-312-b.schema-pinned" in {v.rule_id for v in report.violations}


def test_r1_missing_schema_flagged():
    bad = _good_attest_row()
    del bad["schema"]
    report = hv.validate([bad])
    assert "hipaa-164-312-b.schema-pinned" in {v.rule_id for v in report.violations}


def test_non_dict_row_flagged_under_r1():
    report = hv.validate(["not a dict", 42, None])
    assert report.violation_count == 3
    for v in report.violations:
        assert v.rule_id == "hipaa-164-312-b.schema-pinned"


# ─── R2: timestamp-present ────────────────────────────────────────────


def test_r2_missing_timestamp_flagged():
    bad = _good_attest_row()
    del bad["timestamp"]
    report = hv.validate([bad])
    assert "hipaa-164-312-b.timestamp-present" in {v.rule_id for v in report.violations}


def test_r2_empty_timestamp_flagged():
    bad = _good_attest_row(timestamp="")
    report = hv.validate([bad])
    assert "hipaa-164-312-b.timestamp-present" in {v.rule_id for v in report.violations}


# ─── R3: timestamp-iso8601-utc ────────────────────────────────────────


def test_r3_non_iso_timestamp_flagged():
    bad = _good_attest_row(timestamp="May 3, 2026")
    report = hv.validate([bad])
    assert "hipaa-164-312-b.timestamp-iso8601-utc" in {v.rule_id for v in report.violations}


def test_r3_iso_without_z_suffix_flagged():
    bad = _good_attest_row(timestamp="2026-05-03T05:00:00+02:00")
    report = hv.validate([bad])
    assert "hipaa-164-312-b.timestamp-iso8601-utc" in {v.rule_id for v in report.violations}


def test_r3_does_not_double_count_when_r2_already_fires():
    """If timestamp is missing, R2 fires; R3 should NOT also fire
    (no value to validate). Pattern shared with Art 12 / GDPR Art 30."""
    bad = _good_attest_row()
    del bad["timestamp"]
    report = hv.validate([bad])
    rule_ids = {v.rule_id for v in report.violations}
    assert "hipaa-164-312-b.timestamp-present" in rule_ids
    assert "hipaa-164-312-b.timestamp-iso8601-utc" not in rule_ids


# ─── R4: activity-type-recorded ───────────────────────────────────────


def test_r4_missing_operation_flagged():
    bad = _good_attest_row()
    del bad["operation"]
    report = hv.validate([bad])
    assert "hipaa-164-312-b.activity-type-recorded" in {
        v.rule_id for v in report.violations
    }


def test_r4_empty_operation_flagged():
    bad = _good_attest_row(operation="")
    report = hv.validate([bad])
    assert "hipaa-164-312-b.activity-type-recorded" in {
        v.rule_id for v in report.violations
    }


# ─── R5: system-component-identified ──────────────────────────────────


def test_r5_missing_cli_version_flagged():
    bad = _good_attest_row()
    del bad["cli_version"]
    report = hv.validate([bad])
    assert "hipaa-164-312-b.system-component-identified" in {
        v.rule_id for v in report.violations
    }


def test_r5_empty_cli_version_flagged():
    bad = _good_attest_row(cli_version="")
    report = hv.validate([bad])
    assert "hipaa-164-312-b.system-component-identified" in {
        v.rule_id for v in report.violations
    }


# ─── R6: examination-completeness (per-operation anchors) ─────────────


def test_r6_attest_missing_source_uri_flagged():
    bad = _good_attest_row()
    del bad["source_uri"]
    report = hv.validate([bad])
    assert "hipaa-164-312-b.examination-completeness" in {
        v.rule_id for v in report.violations
    }


def test_r6_attest_empty_source_uri_flagged():
    bad = _good_attest_row(source_uri="")
    report = hv.validate([bad])
    assert "hipaa-164-312-b.examination-completeness" in {
        v.rule_id for v in report.violations
    }


def test_r6_verify_missing_ok_flagged():
    bad = _good_verify_row()
    del bad["ok"]
    report = hv.validate([bad])
    assert "hipaa-164-312-b.examination-completeness" in {
        v.rule_id for v in report.violations
    }


def test_r6_verify_ok_false_is_compliant():
    """Failure outcomes are still examinable — R6 requires presence,
    not truthy value. Pinned twice (here and in the clean-pass section)
    because this is the most likely place for a future bug to land."""
    rows = [_good_verify_row(ok=False)]
    report = hv.validate(rows)
    assert "hipaa-164-312-b.examination-completeness" not in {
        v.rule_id for v in report.violations
    }


def test_r6_render_invalid_mode_flagged():
    bad = _good_render_row()
    bad["mode"] = "unknown-mode"
    report = hv.validate([bad])
    assert "hipaa-164-312-b.examination-completeness" in {
        v.rule_id for v in report.violations
    }


def test_r6_render_missing_mode_flagged():
    bad = _good_render_row()
    del bad["mode"]
    report = hv.validate([bad])
    assert "hipaa-164-312-b.examination-completeness" in {
        v.rule_id for v in report.violations
    }


def test_r6_does_not_apply_to_unknown_operation():
    """An operation HIPAA's R6 doesn't classify (e.g. a future
    operation type) is silently passed at R6 — the rule covers the
    three known operations only. Unknown operations still fail R4
    if `operation` is empty, but a non-empty unknown operation
    name passes both R4 and R6."""
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-03T05:00:00Z",
        "operation": "future_operation_type",
        "cli_version": "0.5.0",
    }
    report = hv.validate([row])
    assert "hipaa-164-312-b.examination-completeness" not in {
        v.rule_id for v in report.violations
    }


# ─── Report shape ─────────────────────────────────────────────────────


def test_report_to_dict_carries_schema_v1():
    report = hv.validate([_good_attest_row()])
    d = report.to_dict()
    assert d["schema"] == "sum.compliance_report.v1"
    assert d["regime"] == "hipaa-164-312-b"


def test_violation_carries_row_index():
    rows = [
        _good_attest_row(),                    # row 0 — clean
        _good_attest_row(schema="oops"),       # row 1 — R1 violation
        _good_attest_row(),                    # row 2 — clean
    ]
    report = hv.validate(rows)
    assert report.violation_count == 1
    assert report.violations[0].row_index == 1


# ─── Cross-regime substrate proof (third instance) ────────────────────


def test_validation_report_shape_matches_other_regimes():
    """Third-regime instance proof of the substrate's regime-
    agnosticism. Same to_dict() keys + Violation fields as Art 12
    and GDPR Art 30; only `regime` and `rule_id` strings differ.
    """
    from sum_engine_internal.compliance import (
        eu_ai_act_article_12 as ev,
        gdpr_article_30 as gv,
    )
    rows = [_good_attest_row(schema="bad")]
    art_12_dict = ev.validate(rows).to_dict()
    art_30_dict = gv.validate(rows).to_dict()
    hipaa_dict = hv.validate(rows).to_dict()

    assert (
        set(art_12_dict.keys())
        == set(art_30_dict.keys())
        == set(hipaa_dict.keys())
    ), "ValidationReport shape diverged across the three regimes"
    assert (
        art_12_dict["schema"]
        == art_30_dict["schema"]
        == hipaa_dict["schema"]
        == "sum.compliance_report.v1"
    )
    assert len({art_12_dict["regime"], art_30_dict["regime"], hipaa_dict["regime"]}) == 3


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
        branch="hipaa-test", title="HIPAA Test",
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
            title="hipaa-test",
            output=None, use_worker=None,
            json=False, pretty=False, verbose=False,
        ))
    finally:
        sys.stdout = old

    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 3

    report = hv.validate(rows)
    assert report.ok, (
        f"shipping CLI's audit log must pass HIPAA § 164.312(b) cleanly; "
        f"violations: {[v.message for v in report.violations]}"
    )
