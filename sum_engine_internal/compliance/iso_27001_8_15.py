"""ISO/IEC 27001:2022 Annex A.8.15 — Logging.

**ISO/IEC 27001:2022 (Information security, cybersecurity and
privacy protection — Information security management systems —
Requirements), Annex A control A.8.15 (Logging).**

The ISO/IEC 27001:2022 Annex A control statement reads:

    Logs that record activities, exceptions, faults and other
    relevant events shall be produced, stored, protected and
    analysed.

The detailed implementation guidance lives in ISO/IEC 27002:2022
§8.15, which lists what such logs should contain (user IDs, system
activities, dates and times of key events, device identity,
successful/rejected attempts, etc.).

This validator pins the **per-row form floor** an audit recording
must satisfy *for the recording to count as a "produced" log under
A.8.15*. The "stored", "protected", and "analysed" verbs map to
deployment-scope obligations (file-system policy, access control,
SIEM integration) that live outside this validator and are named
explicitly in ``docs/COMPLIANCE_ISO_27001_8_15.md``.

Fourth regime to consume ``sum.compliance_report.v1``. The
substrate's regime-agnosticism is now a regularity. ISO 27001
A.8.15's per-row shape is identical to GDPR Article 30's (five
rules covering schema, timestamp presence + parseability, activity
classification, system identification) — empirical confirmation
that there is a *minimum record-keeping floor* common to most
record-keeping regimes. Each regime keeps its own ``rule_id``
strings (downstream dashboards filter on them) but the shape is
shared.

Rule set:

  R1. ``iso-27001-8-15.schema-pinned``
      Every row tagged ``schema = "sum.audit_log.v1"``. A.8.15
      requires logs be machine-readable as a single record-set
      (the "produced" verb implies a coherent collection).
  R2. ``iso-27001-8-15.timestamp-present``
      Non-empty ``timestamp`` on every row. ISO 27002:2022 §8.15
      explicitly lists "dates and times of key events" as
      required content.
  R3. ``iso-27001-8-15.timestamp-iso8601-utc``
      ``timestamp`` parses as ISO 8601 UTC ending in ``Z``.
      Mixed timezones break analysis (the "analysed" verb in
      A.8.15) by silently mis-sorting time-series.
  R4. ``iso-27001-8-15.activity-recorded``
      Non-empty ``operation``. The control statement requires
      logs to "record activities" — ``operation`` (e.g.
      ``"attest"``, ``"verify"``, ``"render"``) is the per-row
      activity indicator.
  R5. ``iso-27001-8-15.system-component-identified``
      Non-empty ``cli_version``. ISO 27002:2022 §8.15 lists
      "device identity" / "system component name" as required;
      ``cli_version`` is the per-row component-version
      attribution. (Multi-component deployments may need a
      separate ``system_id`` field via schema extension.)

Behaviour matches the other regime validators: pure function,
fail-open on malformed input, returns the regime-agnostic
:class:`~sum_engine_internal.compliance.report.ValidationReport`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sum_engine_internal.compliance.report import (
    ValidationReport,
    Violation,
    make_report,
)


REGIME = "iso-27001-8-15"

_RULE_SCHEMA_PINNED = "iso-27001-8-15.schema-pinned"
_RULE_TIMESTAMP_PRESENT = "iso-27001-8-15.timestamp-present"
_RULE_TIMESTAMP_ISO8601_UTC = "iso-27001-8-15.timestamp-iso8601-utc"
_RULE_ACTIVITY_RECORDED = "iso-27001-8-15.activity-recorded"
_RULE_SYSTEM_COMPONENT_IDENTIFIED = "iso-27001-8-15.system-component-identified"


def _is_iso8601_utc(s: Any) -> bool:
    """ISO 8601 UTC string ending in ``Z`` and parseable.

    Same predicate as the other record-keeping validators; lifted
    to ``compliance/_predicates.py`` is a future refactor if 5+
    regimes confirm the duplication is load-bearing.
    """
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
    """Apply ISO 27001 A.8.15 rules R1–R5 to an audit-log stream.

    Pure function over rows; fail-open on malformed input; returns
    regime-agnostic ValidationReport.
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

        # R2 — timestamp present
        ts = row.get("timestamp")
        if not ts:
            violations.append(_violation(
                _RULE_TIMESTAMP_PRESENT, i, row,
                "ISO 27001 A.8.15 / 27002:2022 §8.15 requires "
                "non-empty 'timestamp' (dates and times of key events)",
            ))

        # R3 — timestamp ISO 8601 UTC (only when present)
        if ts is not None and not _is_iso8601_utc(ts):
            violations.append(_violation(
                _RULE_TIMESTAMP_ISO8601_UTC, i, row,
                f"timestamp {ts!r} is not parseable ISO 8601 UTC ending in 'Z'",
            ))

        # R4 — activity recorded (operation present)
        if not row.get("operation"):
            violations.append(_violation(
                _RULE_ACTIVITY_RECORDED, i, row,
                "ISO 27001 A.8.15 'record activities' requires non-empty "
                "'operation' on every row",
            ))

        # R5 — system component identified (cli_version present)
        if not row.get("cli_version"):
            violations.append(_violation(
                _RULE_SYSTEM_COMPONENT_IDENTIFIED, i, row,
                "ISO 27002:2022 §8.15 requires system-component "
                "identification; non-empty 'cli_version' is the per-row "
                "component-version attribution",
            ))

    return make_report(
        regime=REGIME,
        rows_examined=len(rows_list),
        violations=violations,
    )
