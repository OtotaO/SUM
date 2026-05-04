"""PCI DSS v4.0 Requirement 10 — Log and Monitor All Access to System Components and Cardholder Data.

**Payment Card Industry Data Security Standard (PCI DSS) v4.0
(March 2022, with subsequent errata; PCI Security Standards
Council), Requirement 10: "Log and Monitor All Access to System
Components and Cardholder Data."**

Requirement 10 is the most structurally complex requirement in
the slate of record-keeping regimes shipped under Priority 11.
It comprises seven sub-requirements:

  10.1 Processes and mechanisms for restricting and monitoring
       all access to system components and cardholder data are
       defined and understood.
  10.2 Audit logs are implemented to support the detection of
       anomalies and suspicious activity, and the forensic
       analysis of events.
       10.2.1 Audit logs are enabled and active.
       10.2.1.{1..7} Specific event types must be captured
                     (cardholder data access, admin access, log
                     access, invalid attempts, credential
                     changes, log start/stop, system-level
                     objects).
       10.2.2 Audit logs record specific information for each
              event: user identification, type of event, date
              and time, success/failure indication, origination
              of event, identity or name of affected data,
              system component, resource, or service.
  10.3 Audit logs are protected from destruction and unauthorized
       modifications.
  10.4 Audit logs are reviewed to identify anomalies or
       suspicious activity.
  10.5 Audit log history is retained and available for analysis
       (12 months minimum, 3 months immediately available).
  10.6 Time-synchronization mechanisms support consistent time
       settings across all systems.
  10.7 Failures of critical security control systems are
       detected, alerted, and addressed promptly.

This validator pins the **per-row content visible in
``sum.audit_log.v1``** that maps to PCI DSS 10.2.2 (event content
specifics) plus 10.6 (consistent time). All other sub-requirements
— 10.1 organisational, 10.2.1.* event-type coverage, 10.3 log
protection, 10.4 log review, 10.5 retention, 10.7 failure
alerting — live above the per-row layer and are named explicitly
in ``docs/COMPLIANCE_PCI_DSS_4_REQ_10.md`` §"What this validator
does NOT pin."

**Critical structural gap (truth-first).** PCI DSS 10.2.2 lists
"user identification" as the FIRST required field for each
audit-log event. ``sum.audit_log.v1`` does not currently carry a
``user_id`` field — SUM is a single-process CLI tool without a
multi-user model. PCI DSS deployments using SUM as a payment-
adjacent component therefore need *either* a schema extension
(adding ``user_id`` to every row), *or* an authenticating proxy
whose own logs carry the user identity at the aggregation layer.
The wire-spec doc names this gap explicitly. **A green report
from this validator does NOT mean SUM is PCI-compliant — it means
SUM's per-row form satisfies the parts of 10.2.2 visible in the
current schema.**

This is the sixth regime to consume ``sum.compliance_report.v1``
and the last in the record-keeping shape slate. PCI DSS Req 10
genuinely *did* expose the substrate's per-row scope limit (the
``user_id`` gap) — the substrate held without modification, but
the wire-spec doc is meaningfully longer than the others because
PCI DSS Req 10 has more obligations that don't fit the per-row
shape.

Rule set:

  R1. ``pci-dss-4-req-10.schema-pinned``
      Every row tagged ``schema = "sum.audit_log.v1"``.
  R2. ``pci-dss-4-req-10.timestamp-present``
      Non-empty ``timestamp`` (10.2.2: "date and time" of each
      event).
  R3. ``pci-dss-4-req-10.timestamp-iso8601-utc``
      ``timestamp`` parses as ISO 8601 UTC ending in ``Z``. The
      Req 10.6 "consistent time settings" obligation requires
      logs across systems be timestamp-comparable; mixed timezone
      formats break this.
  R4. ``pci-dss-4-req-10.event-type-recorded``
      Non-empty ``operation`` (10.2.2: "type of event").
  R5. ``pci-dss-4-req-10.origination-identified``
      Non-empty ``cli_version`` (10.2.2: "origination of event"
      — for a code-as-origin, the version is the per-row
      origination identifier; multi-host deployments would need
      additional ``host_id`` / ``ip_address`` fields via schema
      extension).
  R6. ``pci-dss-4-req-10.event-content-completeness``
      Per-operation anchors mapping 10.2.2's "identity or name
      of affected data, system component, resource, or service"
      and "success/failure indication":
        - ``operation: "attest"`` — non-empty ``source_uri``
          (identity of affected data: the input artifact)
        - ``operation: "verify"`` — ``ok`` field PRESENT
          (success/failure indication; presence not truthy)
        - ``operation: "render"`` — non-null ``mode`` field
          (the rendering pipeline used, mapping to "affected
          system component, resource, or service")

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


REGIME = "pci-dss-4-req-10"

_RULE_SCHEMA_PINNED = "pci-dss-4-req-10.schema-pinned"
_RULE_TIMESTAMP_PRESENT = "pci-dss-4-req-10.timestamp-present"
_RULE_TIMESTAMP_ISO8601_UTC = "pci-dss-4-req-10.timestamp-iso8601-utc"
_RULE_EVENT_TYPE_RECORDED = "pci-dss-4-req-10.event-type-recorded"
_RULE_ORIGINATION_IDENTIFIED = "pci-dss-4-req-10.origination-identified"
_RULE_EVENT_CONTENT_COMPLETENESS = "pci-dss-4-req-10.event-content-completeness"


def _violation(rule_id: str, row_index: int, row: dict[str, Any], msg: str) -> Violation:
    return Violation(
        rule_id=rule_id,
        row_index=row_index,
        operation=row.get("operation"),
        message=msg,
        row=dict(row),
    )


def validate(rows: Iterable[dict[str, Any]]) -> ValidationReport:
    """Apply PCI DSS v4.0 Requirement 10 rules R1–R6 to an audit-log stream.

    Pins the per-row content visible in ``sum.audit_log.v1`` against
    Req 10.2.2 (event content) plus 10.6 (consistent time). Other
    sub-requirements (10.1, 10.2.1.*, 10.3, 10.4, 10.5, 10.7) live
    above the per-row layer and are out of scope — see the wire-
    spec doc for the explicit naming.
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

        # R2 — timestamp present (10.2.2: date and time)
        ts = row.get("timestamp")
        if not ts:
            violations.append(_violation(
                _RULE_TIMESTAMP_PRESENT, i, row,
                "PCI DSS Req 10.2.2 requires 'date and time' of each "
                "event; non-empty 'timestamp' is the per-row anchor",
            ))

        # R3 — timestamp ISO 8601 UTC (only when present, to avoid
        # double-counting with R2)
        if ts is not None and not is_iso8601_utc(ts):
            violations.append(_violation(
                _RULE_TIMESTAMP_ISO8601_UTC, i, row,
                f"timestamp {ts!r} is not parseable ISO 8601 UTC ending "
                f"in 'Z' (Req 10.6 consistent time settings)",
            ))

        # R4 — event type recorded (10.2.2: type of event)
        op = row.get("operation")
        if not op:
            violations.append(_violation(
                _RULE_EVENT_TYPE_RECORDED, i, row,
                "PCI DSS Req 10.2.2 'type of event' requires non-empty "
                "'operation' on every row",
            ))

        # R5 — origination identified (10.2.2: origination of event)
        if not row.get("cli_version"):
            violations.append(_violation(
                _RULE_ORIGINATION_IDENTIFIED, i, row,
                "PCI DSS Req 10.2.2 'origination of event' requires "
                "non-empty 'cli_version' on every row (per-row "
                "origination identifier; multi-host deployments need "
                "additional fields via schema extension)",
            ))

        # R6 — event content completeness (10.2.2: success/failure +
        # identity of affected data/component/resource)
        if op == "attest":
            su = row.get("source_uri")
            if not su or not isinstance(su, str):
                violations.append(_violation(
                    _RULE_EVENT_CONTENT_COMPLETENESS, i, row,
                    "Req 10.2.2 'identity of affected data' requires the "
                    "input artifact: 'attest' row missing non-empty "
                    "'source_uri'",
                ))
        elif op == "verify":
            if "ok" not in row:
                violations.append(_violation(
                    _RULE_EVENT_CONTENT_COMPLETENESS, i, row,
                    "Req 10.2.2 'success/failure indication' requires "
                    "'verify' row to carry the 'ok' field (presence, "
                    "not truthy — a False outcome is still a recorded "
                    "outcome)",
                ))
        elif op == "render":
            mode = row.get("mode")
            if mode not in ("local-deterministic", "worker"):
                violations.append(_violation(
                    _RULE_EVENT_CONTENT_COMPLETENESS, i, row,
                    f"Req 10.2.2 'identity of affected service' requires "
                    f"'render' row 'mode' to be 'local-deterministic' "
                    f"or 'worker'; got {mode!r}",
                ))

    return make_report(
        regime=REGIME,
        rows_examined=len(rows_list),
        violations=violations,
    )
