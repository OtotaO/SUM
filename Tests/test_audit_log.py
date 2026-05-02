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


def _mp_worker_emit(args: tuple[str, int, int]) -> int:
    """Top-level worker for the multi-process O_APPEND atomicity test.

    multiprocessing requires picklable callables, so this lives at
    module scope. Each call sets SUM_AUDIT_LOG in the worker process'
    environment and emits ``n_emits`` audit rows tagged with the
    worker_id so the parent can assert no rows were lost or duplicated.
    Returns the worker_id for sanity.
    """
    audit_path, worker_id, n_emits = args
    os.environ["SUM_AUDIT_LOG"] = audit_path
    from sum_cli.audit_log import emit_audit_event
    for i in range(n_emits):
        emit_audit_event("verify", {"worker_id": worker_id, "iteration": i})
    return worker_id


def _write_ed25519_pem(path: Path) -> None:
    """Generate an Ed25519 PEM private key for signed-attest tests."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.generate()
    path.write_bytes(sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))


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


# ─── Gap closures (2026-05-02 self-audit) ─────────────────────────────


def test_audit_log_empty_string_treated_as_unset(tmp_path, monkeypatch):
    """SUM_AUDIT_LOG="" is explicitly handled as unset
    (sum_cli/audit_log.py treats both ``None`` and ``""`` as no-op).
    Pin the empty-string branch so a future change can't silently
    start writing to the empty path (which on POSIX would raise
    OSError and be swallowed by fail-open — silent loss)."""
    monkeypatch.setenv("SUM_AUDIT_LOG", "")
    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("verify", {"axiom_count": 1})
    # No file should appear in tmp_path; nothing on stdout either.
    assert list(tmp_path.iterdir()) == []


def test_attest_with_ed25519_key_emits_signed_true(tmp_path, monkeypatch):
    """The attest audit row must carry ``signed: true`` when an
    Ed25519 PEM key is supplied. Pins that the signed-attestation
    branch of cmd_attest still feeds the audit emit — a future
    refactor that drops the bundle-key check (``"public_signature"
    in bundle``) would silently lose Ed25519 signing visibility
    for compliance consumers."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))
    pem = tmp_path / "sk.pem"
    _write_ed25519_pem(pem)

    from sum_cli.main import cmd_attest
    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="signed-test", title="Signed Test",
        signing_key=None, ed25519_key=str(pem), ledger=None,
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

    row = _read_audit_lines(audit)[0]
    assert row["operation"] == "attest"
    assert row["signed"] is True, (
        f"Ed25519 attest must emit signed=True; row={row}"
    )
    assert row["hmac"] is False


def test_attest_with_signing_key_emits_hmac_true(tmp_path, monkeypatch):
    """The attest audit row must carry ``hmac: true`` when an HMAC
    signing key is supplied. Pins the HMAC branch of cmd_attest's
    audit emit (``"signature" in bundle``)."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    from sum_cli.main import cmd_attest
    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="hmac-test", title="HMAC Test",
        signing_key="audit-test-hmac-key-32bytes!!!!!", ed25519_key=None, ledger=None,
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

    row = _read_audit_lines(audit)[0]
    assert row["operation"] == "attest"
    assert row["signed"] is False
    assert row["hmac"] is True, (
        f"HMAC attest must emit hmac=True; row={row}"
    )


def test_attest_with_both_keys_emits_signed_and_hmac_true(tmp_path, monkeypatch):
    """Dual-signing path: both Ed25519 and HMAC. Both flags must
    be true in the audit row. Pins that the audit emit reads the
    bundle's signature fields independently — neither branch
    suppresses the other."""
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))
    pem = tmp_path / "sk.pem"
    _write_ed25519_pem(pem)

    from sum_cli.main import cmd_attest
    args = argparse.Namespace(
        input=None, extractor="sieve", model=None, source=None,
        branch="dual-test", title="Dual Test",
        signing_key="dual-hmac-key-32bytes!!!!!!!!!!!", ed25519_key=str(pem),
        ledger=None,
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

    row = _read_audit_lines(audit)[0]
    assert row["signed"] is True
    assert row["hmac"] is True


def test_audit_log_multiprocess_appends_no_torn_writes(tmp_path):
    """Pin sum_cli/audit_log.py's docstring claim that O_APPEND on
    POSIX produces a serialised total ordering of single-line JSONL
    records under racing processes.

    Eight worker processes each emit twenty rows. Total = 160 lines.
    Asserts: every line parses as JSON, no row is missing, no row
    is duplicated, every (worker_id, iteration) pair appears exactly
    once. This is the actual claim — the in-process serial test
    upstream covers ordering but NOT atomicity under racing writes.
    """
    import multiprocessing as mp

    audit = tmp_path / "audit.jsonl"
    n_workers = 8
    n_emits = 20

    # Use spawn so the test behaves identically on macOS (default
    # spawn since 3.8) and Linux CI. Workers read SUM_AUDIT_LOG from
    # the path passed in; setting os.environ in the worker is local
    # to that worker process.
    ctx = mp.get_context("spawn")
    work = [(str(audit), wid, n_emits) for wid in range(n_workers)]
    with ctx.Pool(n_workers) as pool:
        returned = pool.map(_mp_worker_emit, work)
    assert sorted(returned) == list(range(n_workers))

    rows = _read_audit_lines(audit)
    assert len(rows) == n_workers * n_emits, (
        f"expected {n_workers * n_emits} rows, got {len(rows)} "
        f"(possible torn writes or lost emits)"
    )
    # Every (worker_id, iteration) pair appears exactly once.
    seen = {(r["worker_id"], r["iteration"]) for r in rows}
    assert len(seen) == n_workers * n_emits
    expected = {(w, i) for w in range(n_workers) for i in range(n_emits)}
    assert seen == expected, (
        f"missing or duplicate (worker_id, iteration) pairs; "
        f"missing={expected - seen}, extra={seen - expected}"
    )
    # Sanity: every row carries the required schema fields.
    for r in rows:
        assert r["schema"] == "sum.audit_log.v1"
        assert r["operation"] == "verify"


def test_render_worker_mode_emits_receipt_fields(tmp_path, monkeypatch):
    """Worker-mode render audit row must carry mode='worker',
    render_receipt_kid, render_receipt_schema, and worker_url. The
    upstream test (test_render_emits_audit_row) only pins the
    LOCAL-mode shape and asserts render_receipt_kid is *absent* —
    the positive-shape branch of _emit_render_output (lines
    1171-1175 of sum_cli/main.py) was untested before this PR.

    Approach: drive _emit_render_output directly with a synthetic
    worker envelope. The audit-emit branch is what we're pinning,
    not the HTTP round-trip — exercising it with a real Worker
    request would couple this test to network state. The synthetic
    envelope is byte-shaped exactly like the worker-path output of
    _post_render_to_worker.
    """
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SUM_AUDIT_LOG", str(audit))

    envelope = {
        "tome": "# Synthetic worker tome\n\nSome rendered prose.\n",
        "sliders": {
            "density": 1.0, "length": 0.5, "formality": 0.5,
            "audience": 0.5, "perspective": 0.5,
        },
        "mode": "worker",
        "axiom_count_input": 3,
        "title": "worker-mode-test",
        "render_receipt": {
            "kid": "test-render-key-2026-05-02",
            "schema": "sum.render_receipt.v1",
        },
        "worker_url": "https://sum-demo.ototao.workers.dev/render",
    }

    from sum_cli.main import _emit_render_output
    args = argparse.Namespace(
        output=None, json=False, pretty=False, verbose=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = _emit_render_output(envelope, args)
    finally:
        sys.stdout = old
    assert code == 0

    row = _read_audit_lines(audit)[0]
    assert row["operation"] == "render"
    assert row["mode"] == "worker"
    assert row["axiom_count_input"] == 3
    assert row["tome_chars"] == len(envelope["tome"])
    assert row["sliders"]["density"] == 1.0
    # Worker-mode-specific fields:
    assert row["render_receipt_kid"] == "test-render-key-2026-05-02"
    assert row["render_receipt_schema"] == "sum.render_receipt.v1"
    assert row["worker_url"] == "https://sum-demo.ototao.workers.dev/render"
