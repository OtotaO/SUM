# HIPAA § 164.312(b) Audit Controls validator — `hipaa-164-312-b`

HIPAA Security Rule 45 CFR § 164.312(b) ("**Audit Controls**")
obliges covered entities and business associates to:

> Implement hardware, software, and/or procedural mechanisms
> that record and examine activity in information systems that
> contain or use electronic protected health information.

This validator turns SUM's `sum.audit_log.v1` substrate into an
actionable per-row § 164.312(b) form-floor verdict. It is the
**third regime** to consume `sum.compliance_report.v1`, after
EU AI Act Article 12 and GDPR Article 30.

## What this validator pins

§ 164.312(b) is a "Standard" without numbered specifications — the
flexibility lets covered entities choose mechanisms suited to
their environment. NIST SP 800-66r2 §4.4.4 implementation guidance
describes the records as "the activities of a system." HHS
guidance discusses logs containing user identifier, type of action,
date/time, identifier of object accessed, and success/failure.

This validator pins the **per-row form floor** an audit recording
must satisfy *for the recording to support examination at all*:

| Rule ID | What it pins |
|---|---|
| `hipaa-164-312-b.schema-pinned` | Every row tagged `schema = "sum.audit_log.v1"`. The "examine activity" verb requires records be machine-readable as a single, schema-pinned stream. |
| `hipaa-164-312-b.timestamp-present` | Non-empty `timestamp` (chronology required to reconstruct an incident timeline). |
| `hipaa-164-312-b.timestamp-iso8601-utc` | ISO 8601 UTC parseability — mixed timezones silently mis-sort and break examination integrity. |
| `hipaa-164-312-b.activity-type-recorded` | Non-empty `operation` ("record... activity" → activity-type indicator). |
| `hipaa-164-312-b.system-component-identified` | Non-empty `cli_version` ("information systems" → per-row component attribution). |
| `hipaa-164-312-b.examination-completeness` | Per-operation anchors so each event is independently examinable: `attest` → `source_uri` (artifact processed); `verify` → `ok` field present (success/failure outcome); `render` → `mode ∈ {local-deterministic, worker}` (rendering pipeline used). |

## What this validator does NOT pin

- **Auditor function.** § 164.312(b) requires "examine," not just
  "record" — the covered entity must have humans (or automated
  processes) who actually examine the logs. This validator audits
  the *records*, not whether anyone looks at them.
- **Retention.** § 164.530(j)(2) requires HIPAA records be kept
  for six years. Retention is a deployment concern (file-system
  policy, log-aggregation lifecycle); this validator audits
  *content*, not *retention duration*.
- **System isolation.** § 164.312(a)(1) Access Control and
  § 164.308(a)(4) Information Access Management govern *who*
  has access to systems containing ePHI. This validator is a
  technical safeguard companion under § 164.312(b) only; the
  surrounding access controls live in separate Standards.
- **ePHI inventory.** Whether a given SUM deployment processes
  ePHI at all is a deployment-level fact (the covered entity
  identifies "information systems that contain or use ePHI" per
  the introductory clause of § 164.312). If SUM doesn't process
  ePHI, § 164.312(b) doesn't apply at all and this validator
  isn't a relevant compliance control.
- **User identification.** HHS guidance lists "user identifier"
  as desirable in audit records. SUM's `sum.audit_log.v1` does
  not currently carry a user_id field — multi-user deployments
  using SUM as a HIPAA-relevant component would need to extend
  the schema (or run SUM behind an authenticating proxy whose
  logs carry the user identity at the aggregation layer).

A green report from `sum compliance check --regime hipaa-164-312-b`
says: **"the audit log row stream satisfies the per-row form
requirements for § 164.312(b) audit recording."** It does NOT say
the covered entity is in full HIPAA Security Rule compliance.

## How to use

```bash
# 1. Generate audit-log rows.
export SUM_AUDIT_LOG=/var/log/sum/audit.jsonl
echo 'Patient activity narrative...' | sum attest > bundle.json
sum verify --input bundle.json
sum render --input bundle.json > tome.md

# 2. Validate against § 164.312(b).
sum compliance check \
    --regime hipaa-164-312-b \
    --audit-log /var/log/sum/audit.jsonl \
    --pretty
```

Exit code: `0` ok, `1` violations, `2` usage error — pipe-friendly:

```bash
sum compliance check --regime hipaa-164-312-b --audit-log audit.jsonl \
    || { echo "§ 164.312(b) violation"; exit 1; }
```

## Cross-regime substrate

`hipaa-164-312-b` is the **third regime** to ship; the substrate's
regime-agnosticism is no longer "second-instance proof" but a
regularity. `Tests/compliance/test_hipaa_164_312_b.py
::test_validation_report_shape_matches_other_regimes` empirically
pins that all three regimes (Art 12, GDPR Art 30, HIPAA
§ 164.312(b)) return the same `to_dict()` shape and the same
`Violation` dataclass fields — only `regime` and `rule_id`
strings differ. Downstream consumers (compliance dashboards,
retention pipelines, batch auditors) ingest reports across all
three regimes without per-regime adapters.

The cross-regime CLI dispatch contracts in
`Tests/compliance/test_cli_dispatch.py` (C1 registry consistency,
C2 schema, C3 exit codes) extend automatically when a regime is
registered in both `_COMPLIANCE_REGIMES` and
`_compliance_validators()` in `sum_cli/main.py`.

## Programmatic use

```python
from sum_engine_internal.compliance import hipaa_164_312_b as hv

with open("audit.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]
report = hv.validate(rows)
if not report.ok:
    for v in report.violations:
        print(v.rule_id, v.row_index, v.message)
```
