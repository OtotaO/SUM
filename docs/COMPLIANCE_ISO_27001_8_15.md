# ISO/IEC 27001:2022 Annex A.8.15 validator — `iso-27001-8-15`

ISO/IEC 27001:2022 Annex A.8.15 ("**Logging**") obliges
information-security management systems to:

> Logs that record activities, exceptions, faults and other
> relevant events shall be produced, stored, protected and
> analysed.

This validator turns SUM's `sum.audit_log.v1` substrate into an
actionable per-row A.8.15 floor verdict. It is the **fourth
regime** to consume `sum.compliance_report.v1`.

## What this validator pins

A.8.15 has four verbs: produced, stored, protected, analysed.
This validator pins the per-row form floor that makes the
"produced" verb meaningful. Five rules:

| Rule ID | What it pins |
|---|---|
| `iso-27001-8-15.schema-pinned` | Every row tagged `schema = "sum.audit_log.v1"`. |
| `iso-27001-8-15.timestamp-present` | Non-empty `timestamp` (ISO 27002:2022 §8.15 lists "dates and times of key events" as required content). |
| `iso-27001-8-15.timestamp-iso8601-utc` | ISO 8601 UTC parseability (mixed timezones break analysis). |
| `iso-27001-8-15.activity-recorded` | Non-empty `operation` ("record activities" → activity indicator). |
| `iso-27001-8-15.system-component-identified` | Non-empty `cli_version` (ISO 27002:2022 §8.15: "device identity / system component name"). |

The shape is identical to GDPR Article 30's per-row floor — empirical confirmation that there is a *minimum record-keeping floor* common to most record-keeping regimes. Rule IDs differ; statutory anchors differ; the per-row check shape is shared.

## What this validator does NOT pin

- **The "stored" verb (A.8.15).** Storage durability, redundancy,
  retention period are deployment concerns (file-system policy,
  log-rotation, off-site backup); not visible per-row.
- **The "protected" verb (A.8.15).** Tamper-evidence, access
  control over the log itself, integrity protection mechanisms
  are deployment concerns. SUM's audit log can be Ed25519-signed
  for tamper-evidence (the existing render-receipt machinery),
  but the validator audits *content*, not whether the content is
  cryptographically protected at rest.
- **The "analysed" verb (A.8.15).** Whether anyone (or any
  automated tool) actually examines the logs is an organisational
  concern. The validator audits the records, not whether they're
  analysed.
- **ISO 27002:2022 §8.15 detail content** beyond what's visible
  per-row: user IDs, network addresses, privilege escalation
  events. SUM's `audit_log.v1` schema does not currently carry
  user-level or network-level fields; multi-user / network-aware
  deployments need a schema extension.
- **Other Annex A controls.** A.8.16 monitoring activities, A.5.7
  threat intelligence, A.8.34 protection of information systems
  during audit testing — separate controls, not in scope here.

A green report says: **"the audit log row stream satisfies the
per-row form requirements for A.8.15 'produced' logs."** It does
NOT say the ISMS is in full A.8.15 compliance.

## How to use

```bash
sum compliance check \
    --regime iso-27001-8-15 \
    --audit-log /var/log/sum/audit.jsonl \
    --pretty
```

Exit codes 0 / 1 / 2 (ok / violations / usage error). Pipe-friendly.

## Cross-regime substrate

Fourth-regime instance proof of `sum.compliance_report.v1`'s
regime-agnosticism. `Tests/compliance/test_iso_27001_8_15.py
::test_validation_report_shape_matches_other_regimes` pins shape
parity across all four regimes (Art 12, GDPR Art 30, HIPAA
§ 164.312(b), ISO 27001 A.8.15).
