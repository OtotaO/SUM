"""Contract tests for the EU AI Act Article 12 validator.

Every rule R1..R6 has at least one negative test pinning the
violation message + rule_id, and the integration tests pin that
clean audit-log streams pass cleanly.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path

import pytest

from sum_engine_internal.compliance import eu_ai_act_article_12 as ev
from sum_engine_internal.compliance.report import ValidationReport


# ─── Helpers ──────────────────────────────────────────────────────────


def _good_attest_row(**overrides) -> dict:
    row = {
        "schema": "sum.audit_log.v1",
        "timestamp": "2026-05-02T04:00:00.123Z",
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
        "timestamp": "2026-05-02T04:00:01.234Z",
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
        "timestamp": "2026-05-02T04:00:02.345Z",
        "operation": "render",
        "cli_version": "0.5.0",
        "mode": mode,
        "axiom_count_input": 3,
        "tome_chars": 200,
        "sliders": {
            "density": 1.0, "length": 0.5, "formality": 0.5,
            "audience": 0.5, "perspective": 0.5,
        },
    }
    row.update(overrides)
    return row


# ─── Clean-pass cases ─────────────────────────────────────────────────


def test_clean_audit_log_passes_cleanly():
    """A well-formed audit log with one of each operation passes
    every rule. This is the "happy path" sanity check — if this
    fails, a rule is too aggressive."""
    rows = [_good_attest_row(), _good_verify_row(), _good_render_row()]
    report = ev.validate(rows)
    assert isinstance(report, ValidationReport)
    assert report.ok, f"clean log should pass; got {report.violations}"
    assert report.violation_count == 0
    assert report.regime == "eu-ai-act-article-12"
    assert report.rows_examined == 3


def test_clean_log_with_worker_mode_render_passes():
    """Worker-mode render rows must also pass R6 cleanly."""
    rows = [_good_render_row(mode="worker", render_receipt_kid="kid-2026-05-02")]
    report = ev.validate(rows)
    assert report.ok


def test_empty_audit_log_passes_cleanly():
    """An empty audit log has nothing to violate; the report should
    be ok with rows_examined=0 (not "fail-closed on no events").

    Article 12(1) requires logs to be generated; absence of events
    is a separate concern (a SUM deployment that didn't log anything
    in a reporting period would surface in OUT-OF-BAND audit, not
    in this validator's per-row checks)."""
    report = ev.validate([])
    assert report.ok
    assert report.rows_examined == 0


# ─── R1: schema pinned ────────────────────────────────────────────────


def test_r1_wrong_schema_flagged():
    bad = _good_attest_row(schema="some.other.schema")
    report = ev.validate([bad])
    assert not report.ok
    rule_ids = {v.rule_id for v in report.violations}
    assert "eu-ai-act-art-12.schema-pinned" in rule_ids


def test_r1_missing_schema_flagged():
    bad = _good_attest_row()
    del bad["schema"]
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.schema-pinned" in {v.rule_id for v in report.violations}


def test_non_dict_row_flagged_under_r1():
    """A row that is not even a dict (e.g. a malformed JSONL line
    that decoded to a list or a string) must surface as a
    violation rather than crash the validator."""
    report = ev.validate(["not a dict", 42, None])
    assert report.violation_count == 3
    assert all(v.rule_id == "eu-ai-act-art-12.schema-pinned" for v in report.violations)


# ─── R2: required traceability fields ─────────────────────────────────


def test_r2_missing_timestamp_flagged():
    bad = _good_attest_row()
    del bad["timestamp"]
    report = ev.validate([bad])
    msgs = [v.message for v in report.violations
            if v.rule_id == "eu-ai-act-art-12.required-traceability-fields"]
    assert any("timestamp" in m for m in msgs)


def test_r2_missing_cli_version_flagged():
    bad = _good_attest_row()
    del bad["cli_version"]
    report = ev.validate([bad])
    rule_msgs = [v.message for v in report.violations
                 if v.rule_id == "eu-ai-act-art-12.required-traceability-fields"]
    assert any("cli_version" in m for m in rule_msgs)


def test_r2_empty_string_treated_as_missing():
    """Empty-string traceability fields are as bad as missing —
    a downstream auditor cannot trace from an empty string."""
    bad = _good_attest_row(cli_version="")
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.required-traceability-fields" in {
        v.rule_id for v in report.violations
    }


# ─── R3: timestamp must be ISO 8601 UTC ───────────────────────────────


def test_r3_non_iso_timestamp_flagged():
    bad = _good_attest_row(timestamp="2026/05/02 04:00:00")
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.timestamp-iso8601-utc" in {
        v.rule_id for v in report.violations
    }


def test_r3_iso_without_z_suffix_flagged():
    """A timestamp without the trailing 'Z' (i.e. no UTC marker)
    is rejected; mixed-tz logs silently mis-sort in time-series
    stores."""
    bad = _good_attest_row(timestamp="2026-05-02T04:00:00.123+02:00")
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.timestamp-iso8601-utc" in {
        v.rule_id for v in report.violations
    }


def test_r3_well_formed_timestamp_passes_r3():
    rows = [_good_attest_row()]  # 2026-05-02T04:00:00.123Z
    report = ev.validate(rows)
    assert "eu-ai-act-art-12.timestamp-iso8601-utc" not in {
        v.rule_id for v in report.violations
    }


# ─── R4: attest must have source_uri ──────────────────────────────────


def test_r4_attest_missing_source_uri_flagged():
    bad = _good_attest_row()
    del bad["source_uri"]
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.attest-source-uri-present" in {
        v.rule_id for v in report.violations
    }


def test_r4_attest_empty_source_uri_flagged():
    bad = _good_attest_row(source_uri="")
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.attest-source-uri-present" in {
        v.rule_id for v in report.violations
    }


def test_r4_does_not_apply_to_verify_or_render_rows():
    """R4 is attest-specific — verify/render rows without
    source_uri must NOT trigger R4 (they trigger R5/R6 instead)."""
    rows = [_good_verify_row(), _good_render_row()]
    report = ev.validate(rows)
    rule_ids = {v.rule_id for v in report.violations}
    assert "eu-ai-act-art-12.attest-source-uri-present" not in rule_ids


# ─── R5: verify must have axiom_count + state_integer_digits ──────────


def test_r5_verify_missing_axiom_count_flagged():
    bad = _good_verify_row()
    del bad["axiom_count"]
    report = ev.validate([bad])
    msgs = [v.message for v in report.violations
            if v.rule_id == "eu-ai-act-art-12.verify-bundle-anchor-present"]
    assert any("axiom_count" in m for m in msgs)


def test_r5_verify_missing_state_integer_digits_flagged():
    bad = _good_verify_row()
    del bad["state_integer_digits"]
    report = ev.validate([bad])
    msgs = [v.message for v in report.violations
            if v.rule_id == "eu-ai-act-art-12.verify-bundle-anchor-present"]
    assert any("state_integer_digits" in m for m in msgs)


def test_r5_does_not_apply_to_attest_or_render_rows():
    rows = [_good_attest_row(), _good_render_row()]
    report = ev.validate(rows)
    assert "eu-ai-act-art-12.verify-bundle-anchor-present" not in {
        v.rule_id for v in report.violations
    }


# ─── R6: render must have mode ────────────────────────────────────────


def test_r6_render_missing_mode_flagged():
    bad = _good_render_row()
    del bad["mode"]
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.render-mode-present" in {
        v.rule_id for v in report.violations
    }


def test_r6_render_invalid_mode_flagged():
    bad = _good_render_row(mode="some-future-mode")
    report = ev.validate([bad])
    assert "eu-ai-act-art-12.render-mode-present" in {
        v.rule_id for v in report.violations
    }


def test_r6_render_local_deterministic_passes():
    rows = [_good_render_row(mode="local-deterministic")]
    report = ev.validate(rows)
    assert "eu-ai-act-art-12.render-mode-present" not in {
        v.rule_id for v in report.violations
    }


def test_r6_render_worker_passes():
    rows = [_good_render_row(mode="worker")]
    report = ev.validate(rows)
    assert "eu-ai-act-art-12.render-mode-present" not in {
        v.rule_id for v in report.violations
    }


# ─── ValidationReport shape ───────────────────────────────────────────


def test_report_to_dict_carries_schema_v1():
    rows = [_good_attest_row()]
    report = ev.validate(rows)
    d = report.to_dict()
    assert d["schema"] == "sum.compliance_report.v1"
    assert d["regime"] == "eu-ai-act-article-12"
    assert d["rows_examined"] == 1
    assert d["ok"] is True
    assert d["violation_count"] == 0


def test_report_violations_by_rule_aggregates_correctly():
    rows = [
        _good_attest_row(schema="bad"),
        _good_attest_row(schema="bad", source_uri=""),
        _good_render_row(mode=None),
    ]
    report = ev.validate(rows)
    by_rule = report.violations_by_rule()
    assert by_rule["eu-ai-act-art-12.schema-pinned"] == 2
    assert by_rule["eu-ai-act-art-12.attest-source-uri-present"] == 1
    assert by_rule["eu-ai-act-art-12.render-mode-present"] == 1


def test_violation_carries_row_index():
    """Per-violation row_index lets a human jump to the offending
    line in the JSONL audit log."""
    rows = [
        _good_attest_row(),                    # row 0 — clean
        _good_attest_row(schema="oops"),       # row 1 — R1 violation
        _good_attest_row(),                    # row 2 — clean
    ]
    report = ev.validate(rows)
    assert report.violation_count == 1
    assert report.violations[0].row_index == 1
    assert report.violations[0].operation == "attest"


# ─── End-to-end: real audit log produced by sum CLI passes ────────────


def test_real_sum_cli_audit_log_passes_validation(tmp_path, monkeypatch):
    """Drive the SUM CLI with SUM_AUDIT_LOG set, then validate the
    resulting JSONL against Article 12. A real audit log produced
    by the shipping CLI must pass cleanly — if it does not, either
    the CLI is emitting non-compliant rows or the validator is too
    strict against the actual emit shape."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    # 1. attest
    from sum_cli.main import cmd_attest, cmd_verify, cmd_render
    monkeypatch.setattr("sys.stdin", io.StringIO(
        "Alice likes cats. Bob owns a dog. Carol writes code."
    ))
    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="art-12-test", title="Article 12 Test",
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

    # 2. verify
    sys.stdout = io.StringIO()
    try:
        cmd_verify(argparse.Namespace(
            input=str(bundle_path), signing_key=None, strict=False, pretty=False,
        ))
    finally:
        sys.stdout = old

    # 3. render
    sys.stdout = io.StringIO()
    try:
        cmd_render(argparse.Namespace(
            input=str(bundle_path),
            density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5,
            title="art-12-test",
            output=None, use_worker=None,
            json=False, pretty=False, verbose=False,
        ))
    finally:
        sys.stdout = old

    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 3

    report = ev.validate(rows)
    assert report.ok, (
        f"shipping CLI's audit log must pass Article 12 cleanly; "
        f"violations: {[v.message for v in report.violations]}"
    )
