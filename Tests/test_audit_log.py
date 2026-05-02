"""Contract tests for the SUM audit-log primitive (Path 3 / compliance).

Pin the ``sum.audit_log.v1`` schema + the per-operation row shape +
the fail-open semantics (an unwritable audit destination must NOT
break the trust loop).
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


# ─── Helpers ──────────────────────────────────────────────────────────


def _read_audit_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mint_unsigned_bundle() -> dict:
    """Build a real CanonicalBundle in-memory via the codec."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    algebra = GodelStateAlgebra()
    state = 1
    for s, p, o in [("alice", "likes", "cats"), ("bob", "owns", "dog")]:
        prime = algebra.get_or_mint_prime(s, p, o)
        state = math.lcm(state, prime)
    codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
    return codec.export_bundle(state, branch="audit-log-test", title="Audit Log Test")


# ─── Schema ───────────────────────────────────────────────────────────


def test_audit_log_unset_no_writes(tmp_path, monkeypatch):
    """When SUM_AUDIT_LOG is unset, emit_audit_event is a no-op.
    The trust loop's existing semantics are preserved."""
    monkeypatch.delenv("SUM_AUDIT_LOG", raising=False)
    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("verify", {"axiom_count": 5})
    # Nothing should appear anywhere — the function must return
    # without writing to stdout/stderr/anywhere.
    audit_path = tmp_path / "audit.jsonl"
    assert not audit_path.exists()


def test_audit_log_writes_one_jsonl_row_per_event(tmp_path, monkeypatch):
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("attest", {"axiom_count": 5, "extractor": "sieve"})
    emit_audit_event("verify", {"ok": True, "axiom_count": 5})

    rows = _read_audit_lines(audit)
    assert len(rows) == 2
    assert rows[0]["operation"] == "attest"
    assert rows[1]["operation"] == "verify"


def test_audit_log_schema_v1_required_fields(tmp_path, monkeypatch):
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("attest", {"axiom_count": 3})

    row = _read_audit_lines(audit)[0]
    assert row["schema"] == "sum.audit_log.v1"
    assert "timestamp" in row
    assert row["operation"] == "attest"
    assert "cli_version" in row
    # Operation-specific payload is merged into the row
    assert row["axiom_count"] == 3


def test_audit_log_timestamp_is_iso8601_utc(tmp_path, monkeypatch):
    """Timestamps must be ISO 8601 UTC ending in 'Z'. Compliance
    consumers ingest these into time-series stores; non-ISO formats
    would silently mis-sort."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("verify", {})
    row = _read_audit_lines(audit)[0]
    ts = row["timestamp"]
    # Format: 2026-05-01T18:35:14.123Z
    assert ts.endswith("Z")
    assert "T" in ts
    # Parses as datetime
    from datetime import datetime
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_audit_log_concurrent_appends_serialize_cleanly(tmp_path, monkeypatch):
    """Multiple emissions to the same file must produce parseable
    JSONL — one valid JSON object per line, no inter-leaving."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.audit_log import emit_audit_event
    for i in range(20):
        emit_audit_event("verify", {"ok": True, "iteration": i})

    rows = _read_audit_lines(audit)
    assert len(rows) == 20
    iterations = [r["iteration"] for r in rows]
    assert iterations == list(range(20))


# ─── Fail-open semantics ──────────────────────────────────────────────


def test_audit_log_fails_open_on_unwritable_path(tmp_path, monkeypatch):
    """If SUM_AUDIT_LOG points at an unwritable destination, the
    operation MUST proceed without raising — audit logging is
    advisory, the canonical bundle / receipt remains the
    load-bearing trust artifact. Compliance consumers should
    monitor disk space themselves; a full disk should not break
    `sum verify`.
    """
    # Path with a non-existent intermediate directory — the open()
    # call will raise FileNotFoundError. The audit-log helper must
    # swallow it.
    bad_path = tmp_path / "does-not-exist-dir" / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(bad_path))

    from sum_cli.audit_log import emit_audit_event
    # Must not raise:
    emit_audit_event("attest", {"axiom_count": 5})

    # No row was actually written (the parent dir doesn't exist):
    assert not bad_path.exists()


def test_audit_log_dash_writes_to_stdout(tmp_path, monkeypatch, capsys):
    """SUM_AUDIT_LOG=- routes audit rows to stdout. Useful for
    piping into another tool that consumes the audit stream.
    """
    monkeypatch.setenv("SUM_AUDIT_LOG", "-")

    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("verify", {"ok": True, "axiom_count": 7})

    captured = capsys.readouterr()
    line = captured.out.strip()
    parsed = json.loads(line)
    assert parsed["schema"] == "sum.audit_log.v1"
    assert parsed["operation"] == "verify"


# ─── Integration with CLI commands ────────────────────────────────────


def test_attest_emits_audit_row(tmp_path, monkeypatch):
    """Running `sum attest` with SUM_AUDIT_LOG set must produce
    exactly one audit row with operation='attest' and the
    expected operation-specific fields."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.main import cmd_attest

    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="audit-test", title="Audit Test",
        signing_key=None, ed25519_key=None, ledger=None,
        format="auto", pretty=False, verbose=False,
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("Alice likes cats. Bob owns a dog."))
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_attest(args)
    finally:
        sys.stdout = old
    assert code == 0

    rows = _read_audit_lines(audit)
    assert len(rows) == 1
    row = rows[0]
    assert row["operation"] == "attest"
    assert row["extractor"] == "sieve"
    assert row["branch"] == "audit-test"
    assert row["axiom_count"] == 2
    assert row["state_integer_digits"] > 0
    # Unsigned bundle in this test
    assert row["signed"] is False
    assert row["hmac"] is False


def test_verify_emits_audit_row(tmp_path, monkeypatch):
    """Running `sum verify` with SUM_AUDIT_LOG set produces one
    audit row with operation='verify' and the verification result."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    bundle = _mint_unsigned_bundle()
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    from sum_cli.main import cmd_verify
    args = argparse.Namespace(
        input=str(bundle_path), signing_key=None, strict=False, pretty=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_verify(args)
    finally:
        sys.stdout = old
    assert code == 0

    rows = _read_audit_lines(audit)
    assert len(rows) == 1
    row = rows[0]
    assert row["operation"] == "verify"
    assert row["ok"] is True
    assert row["axiom_count"] == 2
    assert row["branch"] == "audit-log-test"
    assert row["signatures"] == {"hmac": "absent", "ed25519": "absent"}


def test_render_emits_audit_row(tmp_path, monkeypatch):
    """Running `sum render` with SUM_AUDIT_LOG set produces one
    audit row with operation='render' and slider/mode metadata."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    bundle = _mint_unsigned_bundle()
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    from sum_cli.main import cmd_render
    args = argparse.Namespace(
        input=str(bundle_path),
        density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5,
        title="Audit Render Test",
        output=None, use_worker=None,
        json=False, pretty=False, verbose=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_render(args)
    finally:
        sys.stdout = old
    assert code == 0

    rows = _read_audit_lines(audit)
    assert len(rows) == 1
    row = rows[0]
    assert row["operation"] == "render"
    assert row["mode"] == "local-deterministic"
    assert row["axiom_count_input"] == 2
    assert row["sliders"]["density"] == 1.0
    # No render_receipt in local mode
    assert "render_receipt_kid" not in row


def test_full_attest_verify_render_sequence_produces_three_rows(tmp_path, monkeypatch):
    """End-to-end audit trail: attest a doc, verify the bundle,
    render it back. Exactly three rows, in that order, with
    matching state-integer / axiom-count metadata for compliance
    cross-referencing."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.main import cmd_attest, cmd_verify, cmd_render

    # 1. attest
    monkeypatch.setattr("sys.stdin", io.StringIO("Alice likes cats. Bob owns a dog. Carol writes code."))
    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="main", title="audit-trail",
        signing_key=None, ed25519_key=None, ledger=None,
        format="auto", pretty=False, verbose=False,
    )
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        cmd_attest(args)
    finally:
        sys.stdout = old
    bundle = json.loads(out.getvalue())
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    # 2. verify
    cmd_verify(argparse.Namespace(
        input=str(bundle_path), signing_key=None, strict=False, pretty=False,
    ))

    # 3. render
    sys.stdout = io.StringIO()
    try:
        cmd_render(argparse.Namespace(
            input=str(bundle_path),
            density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5,
            title="audit-trail",
            output=None, use_worker=None,
            json=False, pretty=False, verbose=False,
        ))
    finally:
        sys.stdout = old

    rows = _read_audit_lines(audit)
    assert len(rows) == 3
    assert [r["operation"] for r in rows] == ["attest", "verify", "render"]
    # Cross-reference: attest's axiom_count == verify's axiom_count == render's axiom_count_input
    assert rows[0]["axiom_count"] == rows[1]["axiom_count"]
    assert rows[1]["axiom_count"] == rows[2]["axiom_count_input"]
