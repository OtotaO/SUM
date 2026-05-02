"""EU AI Act Article 12 — record-keeping for high-risk AI systems.

**Regulation (EU) 2024/1689 of the European Parliament and of the
Council of 13 June 2024 laying down harmonised rules on artificial
intelligence (Artificial Intelligence Act).**

Article 12 ("Record-keeping") obliges providers of high-risk AI
systems to enable automatic logging of events ("logs") over the
lifetime of the system. Article 12(1)–(2) together require that
logs:

  1. are generated automatically over the lifetime of the system;
  2. ensure traceability appropriate to the intended purpose;
  3. enable monitoring of operation with respect to risk situations
     and substantial modifications.

Article 12(3) further specifies, for high-risk AI systems whose
intended purpose is identification and biometric categorisation,
minimum information to be recorded — but the *general* Article
12(1)–(2) traceability obligation applies to every high-risk AI
system regardless of subdomain.

This validator pins the per-row Article 12(1)–(2) requirements
against ``sum.audit_log.v1`` rows. It does not implement Article
12(3) biometric-specific fields; SUM is not a biometric system, so
those fields would not apply, and a downstream validator could
extend this one if a future SUM use-case crosses into that
regime.

Rule set (this validator's contract):

  R1. ``sum.audit_log.v1.schema-pinned``
      Every row must be tagged ``schema = "sum.audit_log.v1"``. A
      mixed-schema log breaks downstream traceability tools.
  R2. ``required-traceability-fields``
      Every row must carry non-null ``timestamp``, ``operation``,
      ``cli_version``. These are the minimum traceability fields
      Article 12(1) requires for "events" — what, when, by which
      version of the system.
  R3. ``timestamp-iso8601-utc``
      Every ``timestamp`` must parse as ISO 8601 UTC (ending in
      ``Z``). Article 12(1) "automatically generated logs" requires
      a parseable timestamp; mixed timezone formats silently
      mis-sort in time-series stores.
  R4. ``attest-source-uri-present``
      Every ``operation: "attest"`` row must carry a non-empty
      ``source_uri``. Article 12(2) traceability of the AI system's
      *operation* requires identifying the input artifact that was
      processed; ``source_uri`` (typically ``sha256:<hex>``) is
      that identifier.
  R5. ``verify-bundle-anchor-present``
      Every ``operation: "verify"`` row must carry a non-null
      ``axiom_count`` and ``state_integer_digits``. These two
      together anchor the verified bundle to a specific verifiable
      artifact; without them, the verify event is opaque to a
      downstream auditor.
  R6. ``render-mode-present``
      Every ``operation: "render"`` row must carry a non-null
      ``mode`` field (``"local-deterministic"`` or ``"worker"``).
      Without ``mode``, an auditor cannot tell whether the rendered
      output passed through a signed Worker pipeline (worker mode)
      or was generated locally without a render receipt — material
      to "monitoring of operation" under Article 12(2).

This validator returns a regime-agnostic
:class:`~sum_engine_internal.compliance.report.ValidationReport`
that downstream consumers can ingest without per-regime adapters.

The validator is intentionally NOT a CLI; it is a pure function
over a list of audit-log rows. The ``sum compliance check`` CLI
verb wraps this function with file/stdin IO. Other consumers
(retention pipelines, dashboards, batch auditors) should call
:func:`validate` directly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sum_engine_internal.compliance.report import (
    ValidationReport,
    Violation,
    make_report,
)


REGIME = "eu-ai-act-article-12"

# Rule identifiers — stable across versions. New rules append; never
# rename or repurpose an existing rule_id, since downstream
# dashboards may filter on these strings.
_RULE_SCHEMA_PINNED = "eu-ai-act-art-12.schema-pinned"
_RULE_REQUIRED_TRACEABILITY_FIELDS = "eu-ai-act-art-12.required-traceability-fields"
_RULE_TIMESTAMP_ISO8601_UTC = "eu-ai-act-art-12.timestamp-iso8601-utc"
_RULE_ATTEST_SOURCE_URI = "eu-ai-act-art-12.attest-source-uri-present"
_RULE_VERIFY_BUNDLE_ANCHOR = "eu-ai-act-art-12.verify-bundle-anchor-present"
_RULE_RENDER_MODE = "eu-ai-act-art-12.render-mode-present"


def _is_iso8601_utc(s: Any) -> bool:
    """ISO 8601 UTC string ending in ``Z`` and parseable."""
    if not isinstance(s, str):
        return False
    if not s.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


def _violation(rule_id: str, row_index: int, row: dict[str, Any], msg: str) -> Violation:
    return Violation(
        rule_id=rule_id,
        row_index=row_index,
        operation=row.get("operation"),
        message=msg,
        row=dict(row),
    )


def validate(rows: Iterable[dict[str, Any]]) -> ValidationReport:
    """Apply Article 12 rules R1–R6 to a stream of audit-log rows.

    The input may be any iterable — typically a list parsed from a
    JSONL file. Returns a :class:`ValidationReport` aggregating all
    violations across all rules; the report is "ok" iff zero
    violations were found.

    Per the audit-log fail-open philosophy, this validator does NOT
    raise on malformed input — a row that is not a dict, or is
    missing the schema entirely, surfaces as a violation rather
    than a Python exception. Compliance pipelines should not crash
    on a single malformed row.
    """
    rows_list = list(rows)
    violations: list[Violation] = []

    for i, row in enumerate(rows_list):
        if not isinstance(row, dict):
            violations.append(Violation(
                rule_id=_RULE_SCHEMA_PINNED,
                row_index=i,
                operation=None,
                message=f"row is not a dict: {type(row).__name__}",
                row={},
            ))
            continue

        # R1 — schema pinned
        if row.get("schema") != "sum.audit_log.v1":
            violations.append(_violation(
                _RULE_SCHEMA_PINNED, i, row,
                f"schema must be 'sum.audit_log.v1'; got {row.get('schema')!r}",
            ))

        # R2 — required traceability fields (timestamp, operation, cli_version)
        for field_name in ("timestamp", "operation", "cli_version"):
            if not row.get(field_name):
                violations.append(_violation(
                    _RULE_REQUIRED_TRACEABILITY_FIELDS, i, row,
                    f"required traceability field {field_name!r} missing or empty",
                ))

        # R3 — timestamp must be ISO 8601 UTC
        ts = row.get("timestamp")
        if ts is not None and not _is_iso8601_utc(ts):
            violations.append(_violation(
                _RULE_TIMESTAMP_ISO8601_UTC, i, row,
                f"timestamp {ts!r} is not parseable ISO 8601 UTC ending in 'Z'",
            ))

        op = row.get("operation")

        # R4 — attest must have source_uri
        if op == "attest":
            su = row.get("source_uri")
            if not su or not isinstance(su, str):
                violations.append(_violation(
                    _RULE_ATTEST_SOURCE_URI, i, row,
                    "attest row missing non-empty 'source_uri' "
                    "(required for input-artifact traceability)",
                ))

        # R5 — verify must have axiom_count and state_integer_digits
        if op == "verify":
            for anchor_field in ("axiom_count", "state_integer_digits"):
                v = row.get(anchor_field)
                if v is None:
                    violations.append(_violation(
                        _RULE_VERIFY_BUNDLE_ANCHOR, i, row,
                        f"verify row missing bundle anchor field {anchor_field!r}",
                    ))

        # R6 — render must have mode
        if op == "render":
            mode = row.get("mode")
            if mode not in ("local-deterministic", "worker"):
                violations.append(_violation(
                    _RULE_RENDER_MODE, i, row,
                    f"render row 'mode' must be 'local-deterministic' or "
                    f"'worker'; got {mode!r}",
                ))

    return make_report(
        regime=REGIME,
        rows_examined=len(rows_list),
        violations=violations,
    )
