"""sum_cli.audit_log — universal audit-log streaming for compliance.

When the ``SUM_AUDIT_LOG`` environment variable is set to a file path,
every ``sum attest`` / ``sum verify`` / ``sum render`` operation
appends a single JSONL row to that file describing what was done.

This is the regime-agnostic foundation of the compliance primitives
direction (Path 3). A specific regime (GDPR-Art-30, HIPAA-164.514,
EU AI Act Annex IV, etc.) can be implemented as a downstream
consumer of this audit log: tail the file, validate per-regime
required fields, raise on policy violation. The audit log itself
makes no regime-specific assumptions — it records *what happened*
verbatim so any auditor can reconstruct the operation.

Schema: ``sum.audit_log.v1`` — additive; new optional fields may
appear in future minor versions; consumers should ignore unknown
keys.

Required fields per row:
  - ``schema`` = ``"sum.audit_log.v1"``
  - ``timestamp`` (ISO 8601 UTC, e.g. ``"2026-05-01T18:35:14.123Z"``)
  - ``operation`` ∈ ``{"attest", "verify", "render"}``
  - ``cli_version`` (the ``sum-cli`` version that produced the row)

Operation-specific optional fields:
  - attest:  ``source_uri``, ``state_integer_digits``, ``axiom_count``,
             ``extractor``, ``signed`` (bool — Ed25519 attached?),
             ``hmac`` (bool), ``branch``
  - verify:  ``state_integer_digits``, ``axiom_count``, ``signatures``
             ``{ed25519, hmac}``, ``branch``, ``ok`` (bool)
  - render:  ``axiom_count_input``, ``mode`` (``"local-deterministic"``
             or ``"worker"``), ``sliders``, ``render_receipt_kid`` (if
             worker), ``worker_url`` (if worker)

Failures (exit code != 0) emit a row with ``error`` field describing
the failure class. Audit-log writes themselves never fail loudly —
if ``SUM_AUDIT_LOG`` points at a non-writable path, the operation
proceeds and the failure is reported on stderr only when ``--verbose``
is set; the audit semantics fail-open by design (a non-functional
audit destination should not break the trust loop).

Env var precedence:
  1. ``SUM_AUDIT_LOG`` env var (path to JSONL file; appended to)
  2. unset → no audit logging

The path may be ``-`` for stdout (rare; mostly useful for piping
into another tool); otherwise treated as a file path with append-mode
open.

Concurrency: writes are O_APPEND on POSIX, so multiple sum
processes writing to the same audit log produce a serialised
ordering (atomic per write() up to PIPE_BUF on most systems —
single-line JSONL records well under that bound). We write each
row in one ``f.write()`` call to maximise atomicity.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any


_AUDIT_LOG_SCHEMA = "sum.audit_log.v1"


def _audit_log_path() -> str | None:
    """Return the configured audit-log destination, or None if unset."""
    p = os.environ.get("SUM_AUDIT_LOG")
    if p is None or p == "":
        return None
    return p


def emit_audit_event(operation: str, payload: dict[str, Any]) -> None:
    """Append a single JSONL row to the audit log if configured.

    Fail-open: if the audit destination is unwritable, the operation
    proceeds normally; the failure is silent unless verbose stderr
    is enabled by the caller. Audit logging is *advisory* — the
    canonical bundle / receipt still carries the load-bearing trust
    properties; the audit log is for downstream compliance tools
    to ingest.
    """
    path = _audit_log_path()
    if path is None:
        return

    from sum_cli import __version__

    row = {
        "schema": _AUDIT_LOG_SCHEMA,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
                     + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z",
        "operation": operation,
        "cli_version": __version__,
        **payload,
    }
    line = json.dumps(row, separators=(",", ":")) + "\n"

    try:
        if path == "-":
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        # Fail-open: audit destination unavailable; do not block
        # the operation. The canonical bundle / receipt remains the
        # load-bearing trust artifact.
        pass
