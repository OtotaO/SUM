# EU AI Act Article 12 validator — `eu-ai-act-article-12`

Regulation (EU) 2024/1689 ("**AI Act**") Article 12 obliges
providers of high-risk AI systems to enable automatic logging of
events ("logs") over the lifetime of the system. This validator
turns SUM's existing `sum.audit_log.v1` substrate (PR #117) into
an actionable Article 12 pass/fail verdict.

## What this validator pins

Article 12(1)–(2) requires logs that:
1. are generated automatically over the lifetime of the AI system,
2. ensure traceability appropriate to the intended purpose,
3. enable monitoring of operation with respect to risk situations
   and substantial modifications.

Six rules pin the per-row traceability fields the audit log must
carry to satisfy (1)–(2):

| Rule ID | What it pins |
|---|---|
| `eu-ai-act-art-12.schema-pinned` | Every row tagged `schema = "sum.audit_log.v1"`. |
| `eu-ai-act-art-12.required-traceability-fields` | Non-empty `timestamp`, `operation`, `cli_version` on every row. |
| `eu-ai-act-art-12.timestamp-iso8601-utc` | `timestamp` parses as ISO 8601 UTC ending in `Z`. |
| `eu-ai-act-art-12.attest-source-uri-present` | Every `attest` row carries non-empty `source_uri` (input-artifact traceability). |
| `eu-ai-act-art-12.verify-bundle-anchor-present` | Every `verify` row carries `axiom_count` and `state_integer_digits` (verified-bundle anchor). |
| `eu-ai-act-art-12.render-mode-present` | Every `render` row carries `mode ∈ {local-deterministic, worker}`. |

## What this validator does NOT pin

- **Article 12(3) biometric-specific fields.** SUM is not a
  biometric / categorisation system, so these fields don't apply.
  A downstream validator could extend this one if a future SUM
  use-case crosses into that regime.
- **Article 11 / Annex IV technical documentation.** Annex IV is a
  static *document* obligation, not a per-event log obligation —
  this validator covers the runtime-log subset of Article 12, not
  the documentation subset of Article 11.
- **Out-of-band controller / data-subject metadata.** GDPR
  Article 30(1)(a)–(g) require controller-level metadata that
  doesn't live in the audit log. A future GDPR validator will
  layer on top of this one once the controller-metadata channel
  exists.
- **Retention.** Article 12(4) requires logs to be kept for at
  least six months. Retention is a deployment concern (file-system
  policy, log-aggregation lifecycle); this validator audits
  *content*, not *retention duration*.

## How to use

```bash
# 1. Generate audit-log rows by running the CLI with SUM_AUDIT_LOG set.
export SUM_AUDIT_LOG=/var/log/sum/audit.jsonl
echo 'Alice likes cats.' | sum attest > bundle.json
sum verify --input bundle.json
sum render --input bundle.json > tome.md

# 2. Validate the resulting log against Article 12.
sum compliance check \
    --regime eu-ai-act-article-12 \
    --audit-log /var/log/sum/audit.jsonl \
    --pretty
```

Exit code is `0` when `ok: true`, `1` otherwise — pipe-friendly
for CI gates:

```bash
sum compliance check --regime eu-ai-act-article-12 --audit-log audit.jsonl \
    || { echo "Article 12 violation"; exit 1; }
```

## Listing available regimes

```bash
sum compliance regimes
```

Emits a `sum.compliance_regimes.v1` JSON object listing every
regime this CLI can validate against.

## Report shape

A successful `check` invocation emits a `sum.compliance_report.v1`
JSON object on stdout:

```json
{
  "schema": "sum.compliance_report.v1",
  "regime": "eu-ai-act-article-12",
  "rows_examined": 142,
  "ok": false,
  "violation_count": 3,
  "violations_by_rule": {
    "eu-ai-act-art-12.attest-source-uri-present": 2,
    "eu-ai-act-art-12.render-mode-present": 1
  },
  "violations": [
    {
      "rule_id": "eu-ai-act-art-12.attest-source-uri-present",
      "row_index": 17,
      "operation": "attest",
      "message": "attest row missing non-empty 'source_uri' (required for input-artifact traceability)",
      "row": { /* the offending row, verbatim */ }
    }
  ]
}
```

`row_index` is 0-based against the input JSONL, so a human can
jump directly to the offending line.

If parse errors are encountered (a malformed JSONL line that
doesn't decode), they appear under `parse_errors`. A malformed
line is itself a traceability defect (a logged event the auditor
cannot inspect).

## Programmatic use

Bypass the CLI for in-process validation:

```python
from sum_engine_internal.compliance import eu_ai_act_article_12 as ev

with open("audit.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]
report = ev.validate(rows)
if not report.ok:
    for v in report.violations:
        print(v.rule_id, v.row_index, v.message)
```

The `ValidationReport` shape is regime-agnostic
(`sum.compliance_report.v1`) — future regime validators in
`sum_engine_internal.compliance.*` return the same dataclass so
downstream consumers don't need per-regime adapters.

## Adding a new regime

1. Add a module under `sum_engine_internal/compliance/<regime>.py`
   exposing `REGIME` (string id) and `validate(rows) -> ValidationReport`.
2. Append the regime id to `_COMPLIANCE_REGIMES` in `sum_cli/main.py`
   and route it from `cmd_compliance_check`.
3. Add `Tests/compliance/test_<regime>.py` with one negative test
   per rule + an end-to-end test driving the real CLI's output
   through the validator (the `test_real_sum_cli_audit_log_passes_validation`
   pattern).

Existing regime ids are stable; downstream dashboards may filter
on `rule_id` strings, so don't rename existing rules — only append.
