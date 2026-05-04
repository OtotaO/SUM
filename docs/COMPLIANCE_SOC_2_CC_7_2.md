# SOC 2 CC7.2 System Operations validator — `soc-2-cc-7-2`

AICPA Trust Services Criteria CC7.2 (TSP §100A:2017, with 2022
Points of Focus revisions) obliges SOC 2 audited entities to:

> The entity monitors system components and the operation of
> those components for anomalies that are indicative of malicious
> acts, natural disasters, and errors affecting the entity's
> ability to meet its objectives; anomalies are analyzed to
> determine whether they represent security events.

CC7.2 is a **monitoring** criterion — the audit log is the input
that *enables* monitoring (no log → nothing to monitor → CC7.2
fails). This validator pins the per-row form floor required for
the audit log to function as that input.

Fifth regime to consume `sum.compliance_report.v1`. Same shape
as ISO 27001 A.8.15 / GDPR Article 30 (the minimum record-keeping
floor); only `regime` and `rule_id` strings differ.

## What this validator pins

| Rule ID | What it pins |
|---|---|
| `soc-2-cc-7-2.schema-pinned` | Every row tagged `schema = "sum.audit_log.v1"`. CC7.2's monitoring detection-tools point of focus assumes a uniform record-set as input. |
| `soc-2-cc-7-2.timestamp-present` | Non-empty `timestamp` — required to distinguish recent anomalies from historical activity. |
| `soc-2-cc-7-2.timestamp-iso8601-utc` | ISO 8601 UTC parseability (mixed timezones break time-series anomaly-detection windows). |
| `soc-2-cc-7-2.activity-classified` | Non-empty `operation` (anomaly detection requires baseline classification by activity type). |
| `soc-2-cc-7-2.system-component-identified` | Non-empty `cli_version` (the criterion explicitly cites "monitors system components"). |

## What this validator does NOT pin

- **The detection / monitoring activities themselves.** SIEM rules,
  alert routing, oncall rotations, triage runbooks — all
  organisational concerns that operate *on top of* the audit log.
  The validator audits whether the log is suitable as input;
  whether anyone actually monitors it is a separate concern.
- **TSP §100A illustrative controls beyond CC7.2.** CC7.1 (system
  configuration changes), CC7.3 (security incident management),
  CC7.4 (incident response) all sit alongside CC7.2 in the System
  Operations group. Each has its own audit-log relationship; this
  validator covers CC7.2 only.
- **CC6 (Logical and Physical Access Controls) and CC8 (Change
  Management).** These are separate criteria with their own
  control activities; not in scope for a CC7.2 validator.
- **Anomaly-detection analytics quality.** Even with a clean
  audit log, the question of whether the entity's anomaly-
  detection tooling is *good enough* to satisfy the "anomalies
  are analyzed to determine whether they represent security
  events" half of CC7.2 is an audit-judgment matter outside any
  per-row check.
- **Type 1 vs Type 2 distinction.** SOC 2 Type 1 audits design;
  Type 2 audits operating effectiveness over a period. This
  validator is content-only; the operating-effectiveness question
  is the auditor's, not the validator's.

A green report says: **"the audit log row stream is suitable as
input to a CC7.2 monitoring function."** It does NOT say the
entity is in full CC7.2 compliance — that requires the monitoring
function to actually exist and operate over the period under audit.

## How to use

```bash
sum compliance check \
    --regime soc-2-cc-7-2 \
    --audit-log /var/log/sum/audit.jsonl \
    --pretty
```

Exit codes 0 / 1 / 2 (ok / violations / usage error). Pipe-friendly.

## Cross-regime substrate

Fifth-regime instance proof. Five regimes now consume
`sum.compliance_report.v1` without shape modification:
`eu-ai-act-article-12`, `gdpr-article-30`, `hipaa-164-312-b`,
`iso-27001-8-15`, `soc-2-cc-7-2`. The C1/C2/C3 cross-regime
contracts in `Tests/compliance/test_cli_dispatch.py` extend
automatically when a regime is registered in both
`_COMPLIANCE_REGIMES` and `_compliance_validators()`.
