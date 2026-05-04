"""HIPAA Security Rule 45 CFR § 164.312(b) — Audit Controls.

**Health Insurance Portability and Accountability Act of 1996,
Security Rule, 45 CFR § 164.312(b) (Technical Safeguards — Audit
Controls).**

The full statutory text is one sentence:

    Audit controls. Implement hardware, software, and/or
    procedural mechanisms that record and examine activity in
    information systems that contain or use electronic protected
    health information.

The standard is intentionally flexible — § 164.312(b) is a
"Standard" without numbered specifications (unlike § 164.312(a)
Access Control, which has Required + Addressable specifications).
HHS guidance (NIST SP 800-66r2 §4.4.4 implementation guidance)
describes the mechanisms as "the activities of a system." The
record-and-examine pair implies recordings sufficient to *examine*
activity later — chronology, classification, system-component
attribution, and per-operation outcome details.

This validator pins the audit-log row stream against the form
floor a HIPAA-compliant audit recording must satisfy *for the
recording to support examination at all*. As with the GDPR Art 30
validator, this is a per-row scope check: the auditing function
itself (people examining the logs, retention policies, system
isolation, ePHI inventory) lives at deployment scope and is named
explicitly in ``docs/COMPLIANCE_HIPAA_164_312_B.md`` §"What this
validator does NOT pin."

This is the third regime to consume ``sum.compliance_report.v1``.
The shape held without modification across (Art 12, GDPR Art 30,
HIPAA § 164.312(b)) — three is no longer "second-instance proof"
of regime-agnosticism; it's a regularity. The CLI dispatch table,
exit-code contract, report schema, and Violation dataclass are
fully reused.

**Rule overlap with EU AI Act Article 12.** HIPAA's "examine
activity" obligation overlaps in shape with Article 12(2)
"traceability of operation" — both require operation-specific
anchors so an auditor can reconstruct what each event did. R6
("examination-completeness") is the HIPAA analogue of Art 12 R4
+ R5 + R6 unified into one rule. The rules are NOT lifted into
a shared module; the statutory anchors differ (HIPAA points at
ePHI activity, Art 12 at AI traceability), so the rule_ids stay
regime-specific even though the per-row check shape is similar.
A future PR may extract a shared predicate library if 4+ regimes
end up needing the same per-operation anchors.

Rule set (this validator's contract):

  R1. ``hipaa-164-312-b.schema-pinned``
      Every row must be tagged ``schema = "sum.audit_log.v1"``.
      § 164.312(b)'s "examine activity" verb requires the records
      be machine-readable as a single, schema-pinned stream;
      mixed-schema logs break examination tooling.
  R2. ``hipaa-164-312-b.timestamp-present``
      Every row carries a non-null ``timestamp``. Examination
      under § 164.312(b) requires chronology — at minimum, when
      the activity occurred — to reconstruct an incident timeline.
  R3. ``hipaa-164-312-b.timestamp-iso8601-utc``
      ``timestamp`` parses as ISO 8601 UTC ending in ``Z``. Mixed
      timezones silently mis-sort; an ambiguous chronology fails
      examination integrity.
  R4. ``hipaa-164-312-b.activity-type-recorded``
      Every row carries a non-null ``operation``. § 164.312(b)
      "record... activity" requires classifying each event by the
      type of activity — ``operation`` (e.g. ``"attest"``,
      ``"verify"``, ``"render"``) is the per-row activity-type
      indicator.
  R5. ``hipaa-164-312-b.system-component-identified``
      Every row carries a non-null ``cli_version``. § 164.312(b)
      audits "information systems"; identifying which version of
      the system generated the row is the minimum component
      attribution. (For multi-component deployments a future
      extension may add ``system_id`` field; the current SUM
      surface has one component.)
  R6. ``hipaa-164-312-b.examination-completeness``
      Operation-specific anchors that enable per-event
      examination:
        - ``operation: "attest"`` — non-empty ``source_uri``
          (what artifact was processed)
        - ``operation: "verify"`` — non-null ``ok`` field
          (success / failure outcome of the verification)
        - ``operation: "render"`` — non-null ``mode`` field
          (rendering pipeline used; auditor reproducibility)
      Without these anchors, the examination function under
      § 164.312(b) cannot reconstruct *what each event did*.

Behaviour matches the EU AI Act Article 12 and GDPR Article 30
validators: pure function, fail-open on malformed input, returns
the regime-agnostic
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


REGIME = "hipaa-164-312-b"

_RULE_SCHEMA_PINNED = "hipaa-164-312-b.schema-pinned"
_RULE_TIMESTAMP_PRESENT = "hipaa-164-312-b.timestamp-present"
_RULE_TIMESTAMP_ISO8601_UTC = "hipaa-164-312-b.timestamp-iso8601-utc"
_RULE_ACTIVITY_TYPE_RECORDED = "hipaa-164-312-b.activity-type-recorded"
_RULE_SYSTEM_COMPONENT_IDENTIFIED = "hipaa-164-312-b.system-component-identified"
_RULE_EXAMINATION_COMPLETENESS = "hipaa-164-312-b.examination-completeness"


def _is_iso8601_utc(s: Any) -> bool:
    """ISO 8601 UTC string ending in ``Z`` and parseable.

    Same predicate as the EU AI Act Art 12 and GDPR Art 30
    validators. Three regimes share this contract; if a fourth
    regime needs a different timestamp format, this lifts to
    ``compliance/_predicates.py`` — for now, the predicate is
    duplicated cheaply across modules.
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
    """Apply HIPAA § 164.312(b) rules R1–R6 to an audit-log stream.

    The input may be any iterable — typically a list parsed from a
    JSONL file. Returns a :class:`ValidationReport` aggregating all
    violations across all rules; the report is "ok" iff zero
    violations were found.

    Per the audit-log fail-open philosophy, this validator does
    NOT raise on malformed input — a row that is not a dict, or
    is missing the schema entirely, surfaces as a violation rather
    than a Python exception.
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
                "§ 164.312(b) examination requires non-empty 'timestamp' "
                "on every row (chronology required to reconstruct activity)",
            ))

        # R3 — timestamp ISO 8601 UTC (only when present, to avoid
        # double-counting with R2)
        if ts is not None and not _is_iso8601_utc(ts):
            violations.append(_violation(
                _RULE_TIMESTAMP_ISO8601_UTC, i, row,
                f"timestamp {ts!r} is not parseable ISO 8601 UTC ending in 'Z'",
            ))

        # R4 — activity type recorded (operation present)
        op = row.get("operation")
        if not op:
            violations.append(_violation(
                _RULE_ACTIVITY_TYPE_RECORDED, i, row,
                "§ 164.312(b) 'record... activity' requires non-empty "
                "'operation' on every row (the activity-type indicator)",
            ))

        # R5 — system component identified (cli_version present)
        if not row.get("cli_version"):
            violations.append(_violation(
                _RULE_SYSTEM_COMPONENT_IDENTIFIED, i, row,
                "§ 164.312(b) audits 'information systems'; non-empty "
                "'cli_version' is the per-row system-component attribution",
            ))

        # R6 — examination completeness (per-operation anchors)
        if op == "attest":
            su = row.get("source_uri")
            if not su or not isinstance(su, str):
                violations.append(_violation(
                    _RULE_EXAMINATION_COMPLETENESS, i, row,
                    "§ 164.312(b) examination requires the artifact "
                    "processed: 'attest' row missing non-empty 'source_uri'",
                ))
        elif op == "verify":
            # `ok` is a bool; we want to ensure the field is *present*,
            # not just truthy (a False value still represents an
            # outcome — the verification failed, which is examinable).
            if "ok" not in row:
                violations.append(_violation(
                    _RULE_EXAMINATION_COMPLETENESS, i, row,
                    "§ 164.312(b) examination requires the verification "
                    "outcome: 'verify' row missing 'ok' field (success/"
                    "failure indicator)",
                ))
        elif op == "render":
            mode = row.get("mode")
            if mode not in ("local-deterministic", "worker"):
                violations.append(_violation(
                    _RULE_EXAMINATION_COMPLETENESS, i, row,
                    f"§ 164.312(b) examination requires the rendering "
                    f"pipeline used: 'render' row 'mode' must be "
                    f"'local-deterministic' or 'worker'; got {mode!r}",
                ))

    return make_report(
        regime=REGIME,
        rows_examined=len(rows_list),
        violations=violations,
    )
