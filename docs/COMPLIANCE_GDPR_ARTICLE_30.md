# GDPR Article 30 validator — `gdpr-article-30`

Regulation (EU) 2016/679 ("**GDPR**") Article 30 obliges
controllers and processors to maintain **Records of Processing
Activities** (RoPA). This validator turns SUM's `sum.audit_log.v1`
substrate into an actionable per-row Article 30 floor verdict.

It is the second regime to consume `sum.compliance_report.v1`
(EU AI Act Article 12 was the first), proving the substrate's
regime-agnosticism claim by demonstration. The rule shape, exit-
code contract, and report JSON shape are byte-shape-identical to
Article 12's; only the `regime` and `rule_id` strings differ.

## What this validator pins

Article 30 splits naturally into two scopes of obligation:

- **Per-row scope** (this validator's domain) — the form
  requirements visible in each audit-log row that enable Art 30
  reporting at all (electronic-form record per Art 30(3); the
  floor of fields needed to derive categories of processing,
  processor identity, and erasure timing).
- **Record-set scope** (out of this validator's domain) —
  Art 30(1)(a)–(g) and Art 30(2)(a)–(d) meta-level fields
  describing the processing activity overall: controller name +
  contact, purposes, categories of data subjects + personal data,
  categories of recipients, third-country transfers, erasure time
  limits, security measures. These live above any single row and
  require organisational metadata maintained outside the row
  stream.

Five rules pin the per-row scope:

| Rule ID | What it pins |
|---|---|
| `gdpr-art-30.schema-pinned` | Every row tagged `schema = "sum.audit_log.v1"`. Art 30(3)'s electronic-form requirement implies the record-set is machine-readable as a single stream. |
| `gdpr-art-30.timestamp-present` | Non-empty `timestamp` on every row (required to apply Art 30(1)(f) erasure time limits and demonstrate Art 30(4) chronological retrievability). |
| `gdpr-art-30.timestamp-iso8601-utc` | `timestamp` parses as ISO 8601 UTC ending in `Z`. Mixed timezone formats silently mis-sort; an ambiguous chronology fails Art 30(1)(f) retention assessment. |
| `gdpr-art-30.processing-category-present` | Non-empty `operation` on every row. Art 30(1)(b) "purposes of the processing" and 30(2)(b) "categories of processing" both require classifying each event by the kind of processing that occurred. |
| `gdpr-art-30.processor-identity-present` | Non-empty `cli_version` on every row. Art 30(2)(a) requires processor identification; for a code-as-processor, the version is the per-row analogue of "name and contact details." |

## What this validator does NOT pin

- **Article 30(1)(a)–(g) controller-level metadata.** Controller
  name, purposes, categories of data subjects + personal data,
  categories of recipients, third-country transfers, erasure
  time limits, and the general description of security measures
  must be maintained by the controller out-of-band. A future
  regime extension could pin a `sum.audit_log.v1.metadata`
  preamble row carrying these fields, but the current
  `sum.audit_log.v1` schema does not require one.
- **Article 30(2)(a)–(d) processor-level metadata.** Likewise,
  the human-readable processor name + contact details, the
  categorisation of processing on behalf of each controller, and
  the general security-measures description live at record-set
  scope and are not visible per-row.
- **Article 30(4) availability to the supervisory authority.**
  This is an operational obligation (the controller must make
  the record available on request); not a per-row content check.
- **Article 30(5) small-organisation exemption.** Whether the
  exemption applies is a controller-level fact (employees < 250,
  occasional processing, no special-category data, no risk to
  rights and freedoms). This validator does not assess the
  exemption — it audits the *form* of records that are kept,
  not whether records were required at all.
- **Lawful-basis verification.** Art 6 lawful-basis assessment
  is *upstream* of Art 30 record-keeping; this validator audits
  the record, not the legal basis for the processing it records.

A green report from `sum compliance check --regime gdpr-article-30`
says: **"the audit log row stream satisfies the per-row form
requirements enabling Art 30 reporting."** It does NOT say the
controller is in full Art 30 compliance. The controller still
needs the record-set scope metadata, operational availability,
and lawful-basis verification to be in compliance overall.

## How to use

```bash
# 1. Generate audit-log rows by running the CLI with SUM_AUDIT_LOG set.
export SUM_AUDIT_LOG=/var/log/sum/audit.jsonl
echo 'Alice likes cats.' | sum attest > bundle.json
sum verify --input bundle.json
sum render --input bundle.json > tome.md

# 2. Validate the resulting log against Article 30.
sum compliance check \
    --regime gdpr-article-30 \
    --audit-log /var/log/sum/audit.jsonl \
    --pretty
```

Exit code is `0` when `ok: true` and there are no parse errors,
`1` otherwise — pipe-friendly for CI gates:

```bash
sum compliance check --regime gdpr-article-30 --audit-log audit.jsonl \
    || { echo "Article 30 violation"; exit 1; }
```

## Listing available regimes

```bash
sum compliance regimes
```

Lists every registered regime; both `eu-ai-act-article-12` and
`gdpr-article-30` appear.

## Report shape

A successful `check` invocation emits a `sum.compliance_report.v1`
JSON object on stdout. Schema is byte-shape-identical to the EU
AI Act Article 12 report; only `regime` and `rule_id` strings
differ:

```json
{
  "schema": "sum.compliance_report.v1",
  "regime": "gdpr-article-30",
  "rows_examined": 142,
  "ok": false,
  "violation_count": 3,
  "violations_by_rule": {
    "gdpr-art-30.timestamp-iso8601-utc": 2,
    "gdpr-art-30.processor-identity-present": 1
  },
  "violations": [
    {
      "rule_id": "gdpr-art-30.timestamp-iso8601-utc",
      "row_index": 17,
      "operation": "attest",
      "message": "timestamp 'May 3, 2026' is not parseable ISO 8601 UTC ending in 'Z'",
      "row": { /* the offending row, verbatim */ }
    }
  ]
}
```

## Programmatic use

```python
from sum_engine_internal.compliance import gdpr_article_30 as gv

with open("audit.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]
report = gv.validate(rows)
if not report.ok:
    for v in report.violations:
        print(v.rule_id, v.row_index, v.message)
```

## Cross-regime substrate (regime-agnosticism, demonstrated)

GDPR Art 30 is the second regime to ship; EU AI Act Art 12 was
the first. Both consume the same `sum.audit_log.v1` row stream
and both return the same `ValidationReport` shape. `Tests/
compliance/test_cli_dispatch.py` pins three substrate-level
contracts that span every regime:

- **C1.** `_COMPLIANCE_REGIMES` keys exactly equal
  `_compliance_validators()` keys (no orphaned descriptions, no
  un-described validators).
- **C2.** Every regime returns a `sum.compliance_report.v1`
  schema from `cmd_compliance_check`.
- **C3.** Exit codes: 0 = ok, 1 = violations, 2 = usage error.

Adding a further regime (HIPAA § 164.312(b), SOC 2 CC7.2,
ISO 27001 A.8.15, PCI DSS 4.0 Req 10 — all four have since **shipped**,
so all six regimes are live)
inherits all three contracts automatically by registering in both
the description registry and the dispatch table.
