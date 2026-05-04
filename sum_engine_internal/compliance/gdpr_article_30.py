"""GDPR Article 30 — Records of Processing Activities (RoPA).

**Regulation (EU) 2016/679 of the European Parliament and of the
Council of 27 April 2016 on the protection of natural persons with
regard to the processing of personal data and on the free movement
of such data ("General Data Protection Regulation").**

Article 30 ("Records of processing activities") obliges controllers
and processors to maintain a record of processing activities under
their responsibility. Article 30(1) lists the controller-side fields
((a) controller name + contact, (b) purposes of processing, (c)
categories of data subjects + personal data, (d) categories of
recipients, (e) third-country transfers, (f) erasure time limits,
(g) general security measures); Article 30(2) lists the processor-
side fields. Article 30(3) requires the records to be in writing,
including electronic form. Article 30(4) requires the record be
made available to the supervisory authority on request.

**Critical scope distinction (truth-first).** Article 30 splits
naturally into two categories of obligation:

  - **Record-set scope.** Most of Art 30(1)(a)–(g) and (2)(a)–(d)
    are *meta-level* fields describing the processing activity
    overall — controller name, purposes, categories, recipients,
    transfers, erasure timing, security measures. These fields
    live above any single audit-log row and require organisational
    metadata maintained outside the row stream.
  - **Per-row scope.** A subset of Art 30 obligations IS visible
    per-row in ``sum.audit_log.v1``: the *form* requirements that
    enable record-keeping at all (Art 30(3) electronic form, plus
    the floor of fields needed to derive categories of processing,
    processor identity, and erasure timing).

This validator pins the **per-row scope** floor. The record-set
scope is named explicitly in
``docs/COMPLIANCE_GDPR_ARTICLE_30.md`` §"What this validator does
NOT pin" — a controller still needs to maintain Art 30(1)(a)–(g)
metadata separately. A green report from this validator says "the
audit log row stream satisfies the per-row form requirements
enabling Art 30 reporting"; it does not say "the controller is in
full Art 30 compliance."

This is the substrate-tightening discipline at work: the
``sum.compliance_report.v1`` shape was designed regime-agnostic
when EU AI Act Article 12 was the only consumer. GDPR Art 30 is the
second consumer; the shape held without modification, *and* the
truth-first scope-naming pattern carries cleanly across regimes.

Rule set (per-row floor for Art 30 reporting):

  R1. ``gdpr-art-30.schema-pinned``
      Every row tagged ``schema = "sum.audit_log.v1"``. A mixed-
      schema log breaks downstream traceability; Art 30(3)'s
      "in writing, including in electronic form" requires the
      record to be machine-readable as a single record-set, which
      a mixed-schema stream isn't.
  R2. ``gdpr-art-30.timestamp-present``
      Every row carries a non-null ``timestamp``. Without
      timestamps the controller cannot apply Art 30(1)(f) erasure
      time limits to specific records, nor demonstrate Art 30(4)
      retrievability (which records pre-date / post-date a
      supervisory request).
  R3. ``gdpr-art-30.timestamp-iso8601-utc``
      Every ``timestamp`` parses as ISO 8601 UTC ending in ``Z``.
      Mixed timezone formats silently mis-sort in time-series
      stores; a record-set whose chronology is ambiguous fails
      Art 30(1)(f) retention assessment.
  R4. ``gdpr-art-30.processing-category-present``
      Every row carries a non-null ``operation``. Article 30(1)(b)
      "purposes of the processing" and 30(2)(b) "categories of
      processing" both require classifying each event by *what
      kind of processing occurred*. ``operation`` (e.g.
      ``"attest"``, ``"verify"``, ``"render"``) is the per-row
      processing-category indicator.
  R5. ``gdpr-art-30.processor-identity-present``
      Every row carries a non-null ``cli_version``. Article 30(2)(a)
      requires "the name and contact details of the processor or
      processors". For an automated processor (a CLI tool), the
      processor's *version* is the minimum identification: it tells
      the supervisory authority which version of the system
      generated the row, which is the per-row analogue of "name
      and contact details" for code-as-processor.

Behavior matches the EU AI Act Article 12 validator:

  - Pure function: no IO, no state. The CLI wraps it.
  - Fail-open: malformed rows surface as violations rather than
    Python exceptions, so compliance pipelines don't crash on a
    single bad row.
  - Returns the regime-agnostic
    :class:`~sum_engine_internal.compliance.report.ValidationReport`
    so downstream consumers don't need per-regime adapters.

The validator is intentionally NOT a CLI; ``sum compliance check
--regime gdpr-article-30`` wraps :func:`validate` with file/stdin
IO. Other consumers (retention pipelines, dashboards, batch
auditors) call :func:`validate` directly.
"""
from __future__ import annotations

from typing import Any, Iterable

from sum_engine_internal.compliance._predicates import is_iso8601_utc
from sum_engine_internal.compliance.report import (
    ValidationReport,
    Violation,
    make_report,
)


REGIME = "gdpr-article-30"

# Rule identifiers — stable across versions. New rules append; never
# rename or repurpose an existing rule_id, since downstream
# dashboards may filter on these strings.
_RULE_SCHEMA_PINNED = "gdpr-art-30.schema-pinned"
_RULE_TIMESTAMP_PRESENT = "gdpr-art-30.timestamp-present"
_RULE_TIMESTAMP_ISO8601_UTC = "gdpr-art-30.timestamp-iso8601-utc"
_RULE_PROCESSING_CATEGORY_PRESENT = "gdpr-art-30.processing-category-present"
_RULE_PROCESSOR_IDENTITY_PRESENT = "gdpr-art-30.processor-identity-present"


def _violation(rule_id: str, row_index: int, row: dict[str, Any], msg: str) -> Violation:
    return Violation(
        rule_id=rule_id,
        row_index=row_index,
        operation=row.get("operation"),
        message=msg,
        row=dict(row),
    )


def validate(rows: Iterable[dict[str, Any]]) -> ValidationReport:
    """Apply Article 30 per-row floor rules R1–R5 to an audit-log stream.

    The input may be any iterable — typically a list parsed from a
    JSONL file. Returns a :class:`ValidationReport` aggregating all
    violations across all rules; the report is "ok" iff zero
    violations were found.

    Per the audit-log fail-open philosophy, this validator does NOT
    raise on malformed input — a row that is not a dict, or is
    missing the schema entirely, surfaces as a violation rather
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
                "Art 30(1)(f) retention assessment requires non-empty "
                "'timestamp' on every row",
            ))

        # R3 — timestamp ISO 8601 UTC (only checked when present, since
        # absence already surfaces under R2 — avoids double-counting)
        if ts is not None and not is_iso8601_utc(ts):
            violations.append(_violation(
                _RULE_TIMESTAMP_ISO8601_UTC, i, row,
                f"timestamp {ts!r} is not parseable ISO 8601 UTC ending in 'Z'",
            ))

        # R4 — processing category (operation) present
        op = row.get("operation")
        if not op:
            violations.append(_violation(
                _RULE_PROCESSING_CATEGORY_PRESENT, i, row,
                "Art 30(1)(b) / 30(2)(b) categorisation requires non-empty "
                "'operation' on every row (the processing-category indicator)",
            ))

        # R5 — processor identity (cli_version) present
        if not row.get("cli_version"):
            violations.append(_violation(
                _RULE_PROCESSOR_IDENTITY_PRESENT, i, row,
                "Art 30(2)(a) processor identity requires non-empty "
                "'cli_version' on every row (per-row processor-version "
                "identifier; the controller separately maintains the "
                "human-readable processor name + contact details out-of-band)",
            ))

    return make_report(
        regime=REGIME,
        rows_examined=len(rows_list),
        violations=violations,
    )
