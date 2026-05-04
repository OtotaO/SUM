# PCI DSS v4.0 Requirement 10 validator — `pci-dss-4-req-10`

PCI DSS v4.0 Requirement 10 ("**Log and Monitor All Access to
System Components and Cardholder Data**") is the most structurally
complex requirement in the slate of record-keeping regimes shipped
under Priority 11. It comprises seven sub-requirements (10.1
through 10.7), with 10.2 itself further subdivided into 10.2.1,
10.2.1.1–10.2.1.7, and 10.2.2.

This validator is the **sixth and final** regime in the record-
keeping shape slate. It pins the per-row content visible in
`sum.audit_log.v1` against PCI DSS Req 10.2.2 (event content
specifics) plus 10.6 (consistent time). All other sub-
requirements live above the per-row layer and are named
explicitly in the §"What this validator does NOT pin" section
below.

## What this validator pins

Seven per-row rules mapping to Req 10.2.2 + 10.6:

| Rule ID | What it pins |
|---|---|
| `pci-dss-4-req-10.schema-pinned` | Every row tagged `schema = "sum.audit_log.v1"`. |
| `pci-dss-4-req-10.timestamp-present` | Non-empty `timestamp` (Req 10.2.2: "date and time" of each event). |
| `pci-dss-4-req-10.timestamp-iso8601-utc` | ISO 8601 UTC parseability (Req 10.6 consistent time settings). |
| `pci-dss-4-req-10.event-type-recorded` | Non-empty `operation` (Req 10.2.2: "type of event"). |
| `pci-dss-4-req-10.origination-identified` | Non-empty `cli_version` (Req 10.2.2: "origination of event" — for code-as-origin, the version is the per-row identifier). |
| `pci-dss-4-req-10.event-content-completeness` | Per-operation anchors mapping 10.2.2's "identity or name of affected data, system component, resource, or service" + "success/failure indication": `attest` → `source_uri`; `verify` → `ok` PRESENT; `render` → `mode ∈ {local-deterministic, worker}`. |
| `pci-dss-4-req-10.user-identification` (R7, **Sprint 4 / PR #140**) | Non-empty `user_id` on every row. Closes the load-bearing gap previously named in this doc. Operators populate `user_id` via the `SUM_AUDIT_USER_ID` env var; the audit-log emit path reads it at process start. PCI-relevant deployments source the env var from the authenticating proxy's session identity. |

## What this validator does NOT pin

This list is meaningfully longer than the other regimes' because
PCI DSS Req 10 has more obligations that don't fit the per-row
shape. **Truth-first naming** is essential here — a green report
from this validator covers a *minority* of Req 10's surface.

### 10.1 Organisational obligations (out of scope)
- 10.1.1 Documented and up-to-date security policies and
  operational procedures.
- 10.1.2 Documented and assigned roles and responsibilities for
  Req 10 activities.

These are organisational artifacts, not row content.

### 10.2.1.* Specific event-type coverage (out of scope)
- 10.2.1 Audit logs enabled and active for all system components
  and cardholder data.
- 10.2.1.1 All individual user access to cardholder data.
- 10.2.1.2 All actions by individuals with administrative access.
- 10.2.1.3 All access to audit logs.
- 10.2.1.4 All invalid logical access attempts.
- 10.2.1.5 All changes to identification and authentication
  credentials.
- 10.2.1.6 All initialization, stopping, or pausing of audit logs.
- 10.2.1.7 All creation and deletion of system-level objects.

These are *event-coverage* obligations — the validator audits the
content of rows that exist; whether all required event types
*get logged at all* is an integration question outside the
validator's row-content scope. SUM does not currently log all
these event categories (it has `attest` / `verify` / `render`
operations, not "user authentication failure" or "admin action").

### 10.2.2 User identification (CLOSED 2026-05-03 / PR #140)

**Status: closed at the substrate layer.** Sprint 4 of the
intensification path to arXiv (PR #140) extended the audit-log
emit path to read three optional identity env vars at process
start — `SUM_AUDIT_USER_ID`, `SUM_AUDIT_HOST_ID`,
`SUM_AUDIT_IP_ADDRESS` — populating optional `user_id` /
`host_id` / `ip_address` fields on every row. The PCI validator's
new R7 rule (`pci-dss-4-req-10.user-identification`) fires on
rows lacking `user_id`, converting the previously-named gap
into a validatable contract.

**How operators close the gap in production:** source
`SUM_AUDIT_USER_ID` from your authenticating proxy's session
identity at process start. The proxy already has the user
identity (it authenticated the user); pass it through to SUM as
an env var. The audit-log row then carries the identity per
Req 10.2.2's first-required-field requirement.

**Backward compatibility:** the new fields are optional. Audit
logs predating Sprint 4 still pass every existing validator;
they fail R7 specifically (which is the truthful signal that
they don't meet Req 10.2.2's user-identification requirement).
SUM's per-row schema is still `sum.audit_log.v1` — additive
optional fields, not a breaking schema bump.

**What this still doesn't close:** Req 10.2.2 also names
"identity or name of affected data, system component, resource,
or service" — the per-operation anchors handled by R6 (attest
source_uri, verify ok, render mode). For multi-host / multi-
container deployments that need finer attribution than
`host_id` provides, additional schema fields would be needed
(e.g. `container_id`, `request_id`, `correlation_id`); these
remain a future schema extension if a deployment context calls
for them.

### 10.3 Log file protection (out of scope)
- 10.3.1 Read access limited to job-related need.
- 10.3.2 Files protected against modifications by individuals.
- 10.3.3 Logs promptly backed up to secure central log server.
- 10.3.4 File integrity monitoring / change detection on logs.

These are deployment-layer obligations — file-system permissions,
backup pipelines, FIM tooling. SUM does support Ed25519 signing of
audit-log rows (PR #119 added the attestation row machinery for
tamper-evidence), which is a *content* contribution to 10.3.4
specifically; but the validator audits content shape, not whether
the content is cryptographically protected at rest.

### 10.4 Log review (out of scope)
- 10.4.1 Daily review of specified logs.
- 10.4.1.1 Automated review mechanisms.
- 10.4.2 Periodic review of other system component logs.
- 10.4.3 Anomalies addressed during review.

The "Audit logs are reviewed" verb requires humans (or automated
tooling) that actually read the logs and triage findings. The
validator audits the records, not whether anyone reviews them.

### 10.5 Retention (out of scope)
- 10.5.1 Audit log history retained for at least 12 months, with
  the most recent 3 months immediately available for analysis.

Retention is a deployment concern (file-system policy, log-
aggregation lifecycle, archive tier). The validator audits
content, not duration.

### 10.6 Time synchronization (partially in scope)
- 10.6.1 System clocks and time synchronized using time-
  synchronization technology.
- 10.6.2 Systems configured to correct and consistent time.
- 10.6.3 Time synchronization settings and data protected.

The validator covers 10.6 only insofar as it pins ISO 8601 UTC
parseability per row (R3). The actual NTP infrastructure, clock-
drift detection, and time-source-protection mechanisms are
deployment-layer concerns.

### 10.7 Failure detection (out of scope)
- 10.7.1 Service-provider-only: failures of critical security
  control systems detected, alerted, and addressed promptly.
- 10.7.2 Failures of critical security control systems detected,
  alerted, and addressed promptly.
- 10.7.3 Failures responded to promptly.

This is an alerting / incident-response layer — when a critical
security control fails (firewall, IDS, anti-malware, etc.), the
entity must detect, alert, and respond. SUM is not itself a
critical security control system in the PCI DSS sense; this
sub-requirement applies to the surrounding deployment, not the
audit log content.

### Cardholder data inventory (out of scope)

Whether a given SUM deployment processes cardholder data at all
is a deployment-level fact. If SUM doesn't process cardholder
data, Req 10 doesn't apply at all — and this validator isn't a
relevant compliance control. Conversely, if SUM *does* process
cardholder data, the deployment owes the full Req 10 surface,
not just the per-row content.

## A green report says

> "The audit log row stream satisfies the per-row form
> requirements visible in `sum.audit_log.v1` for PCI DSS Req
> 10.2.2 event content + 10.6 timestamp parseability."

It does NOT say:
- The deployment is PCI DSS Req 10 compliant.
- The deployment is PCI DSS compliant overall.
- All required event types are being logged.
- Logs are protected, retained, reviewed, or alerting on
  failures.

(User identification IS captured per row when
`SUM_AUDIT_USER_ID` is set — see §10.2.2 above.)

A QSA (Qualified Security Assessor) auditing the deployment will
need additional evidence covering the §"NOT pin" sections above.

## How to use

```bash
sum compliance check \
    --regime pci-dss-4-req-10 \
    --audit-log /var/log/sum/audit.jsonl \
    --pretty
```

Exit codes 0 / 1 / 2 (ok / violations / usage error). Pipe-friendly.

## Cross-regime substrate

Sixth and final regime in the record-keeping shape slate. Six
regimes consume `sum.compliance_report.v1` without shape
modification:

| Regime | Statute | Origin |
|---|---|---|
| `eu-ai-act-article-12` | Reg (EU) 2024/1689 Art 12 | EU AI law |
| `gdpr-article-30` | Reg (EU) 2016/679 Art 30 | EU privacy law |
| `hipaa-164-312-b` | 45 CFR § 164.312(b) | US health law |
| `iso-27001-8-15` | ISO/IEC 27001:2022 A.8.15 | International standard |
| `soc-2-cc-7-2` | AICPA TSP §100A CC7.2 | US audit-attestation |
| `pci-dss-4-req-10` | PCI DSS v4.0 Req 10 | Payment-card industry standard |

`Tests/compliance/test_pci_dss_4_req_10.py
::test_validation_report_shape_matches_other_regimes` pins
six-way byte-shape parity. The substrate's regime-agnosticism is
empirical fact: across statutes spanning AI law, privacy law,
health-information law, an international ISMS standard, an audit-
attestation framework, and a payment-card industry standard, the
same `ValidationReport` shape carries every regime's verdict.
