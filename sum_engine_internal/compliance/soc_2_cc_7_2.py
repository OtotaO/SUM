"""SOC 2 Trust Services Criteria CC7.2 — System Operations.

**AICPA TSP Section 100A "Trust Services Criteria for Security,
Availability, Processing Integrity, Confidentiality, and Privacy"
(2017, with 2022 Points of Focus revisions), Common Criteria
CC7.2 — System Operations.**

The criterion statement reads:

    The entity monitors system components and the operation of
    those components for anomalies that are indicative of
    malicious acts, natural disasters, and errors affecting the
    entity's ability to meet its objectives; anomalies are
    analyzed to determine whether they represent security events.

CC7.2 is a *monitoring* criterion — the audit log is the input
that *enables* monitoring (no log → nothing to monitor → CC7.2
fails). The illustrative controls in TSP §100A include
"Implementation of Detection Tools" and "Procedures for Monitoring
of Anomalies." The detection / monitoring / analysis activities
themselves live at deployment scope (SIEM rules, alert routing,
oncall rotations) and are named in
``docs/COMPLIANCE_SOC_2_CC_7_2.md`` as out of this validator's
reach.

Fifth regime to consume ``sum.compliance_report.v1``. Same shape
as ISO 27001 A.8.15 — five rules covering schema, timestamp
presence + parseability, activity classification, system
identification. Each regime's ``rule_id`` strings stay regime-
specific; the substrate is now a clear regularity rather than a
claim under test.

Rule set:

  R1. ``soc-2-cc-7-2.schema-pinned``
      Every row tagged ``schema = "sum.audit_log.v1"``. CC7.2's
      monitoring detection-tools point of focus assumes a
      uniform record-set as input; mixed-schema logs break
      detection.
  R2. ``soc-2-cc-7-2.timestamp-present``
      Non-empty ``timestamp`` — required to distinguish recent
      anomalies (the criterion's "monitors... for anomalies"
      verb) from historical activity.
  R3. ``soc-2-cc-7-2.timestamp-iso8601-utc``
      Timestamp parses as ISO 8601 UTC ending in ``Z``. Mixed
      timezones break the "are analyzed to determine whether
      they represent security events" verb by silently mis-
      sorting in time-series anomaly-detection windows.
  R4. ``soc-2-cc-7-2.activity-classified``
      Non-empty ``operation``. Anomaly detection requires
      classifying baseline activity by type; ``operation`` is the
      per-row activity-type label.
  R5. ``soc-2-cc-7-2.system-component-identified``
      Non-empty ``cli_version``. The criterion explicitly cites
      "monitors system components" — per-row component
      attribution is the minimum to associate an anomaly with
      a specific component.

Behaviour matches the other regime validators: pure function,
fail-open on malformed input, returns the regime-agnostic
:class:`~sum_engine_internal.compliance.report.ValidationReport`.
"""
from __future__ import annotations

from typing import Any, Iterable

from sum_engine_internal.compliance._predicates import is_iso8601_utc
from sum_engine_internal.compliance.report import (
    ValidationReport,
    Violation,
    make_report,
)


REGIME = "soc-2-cc-7-2"

_RULE_SCHEMA_PINNED = "soc-2-cc-7-2.schema-pinned"
_RULE_TIMESTAMP_PRESENT = "soc-2-cc-7-2.timestamp-present"
_RULE_TIMESTAMP_ISO8601_UTC = "soc-2-cc-7-2.timestamp-iso8601-utc"
_RULE_ACTIVITY_CLASSIFIED = "soc-2-cc-7-2.activity-classified"
_RULE_SYSTEM_COMPONENT_IDENTIFIED = "soc-2-cc-7-2.system-component-identified"


def _violation(rule_id: str, row_index: int, row: dict[str, Any], msg: str) -> Violation:
    return Violation(
        rule_id=rule_id,
        row_index=row_index,
        operation=row.get("operation"),
        message=msg,
        row=dict(row),
    )


def validate(rows: Iterable[dict[str, Any]]) -> ValidationReport:
    """Apply SOC 2 CC7.2 rules R1–R5 to an audit-log stream."""
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
                "SOC 2 CC7.2 monitoring requires non-empty 'timestamp' "
                "to distinguish recent anomalies from historical activity",
            ))

        # R3 — timestamp ISO 8601 UTC (only when present)
        if ts is not None and not is_iso8601_utc(ts):
            violations.append(_violation(
                _RULE_TIMESTAMP_ISO8601_UTC, i, row,
                f"timestamp {ts!r} is not parseable ISO 8601 UTC ending in 'Z'",
            ))

        # R4 — activity classified (operation present)
        if not row.get("operation"):
            violations.append(_violation(
                _RULE_ACTIVITY_CLASSIFIED, i, row,
                "SOC 2 CC7.2 anomaly detection requires non-empty "
                "'operation' (per-row activity-type label)",
            ))

        # R5 — system component identified (cli_version present)
        if not row.get("cli_version"):
            violations.append(_violation(
                _RULE_SYSTEM_COMPONENT_IDENTIFIED, i, row,
                "CC7.2 'monitors system components' requires non-empty "
                "'cli_version' (per-row component attribution)",
            ))

    return make_report(
        regime=REGIME,
        rows_examined=len(rows_list),
        violations=violations,
    )
