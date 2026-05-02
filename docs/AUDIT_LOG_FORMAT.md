# Audit log format — `sum.audit_log.v1`

The CLI emits a single JSONL row per `sum attest` / `sum verify` /
`sum render` operation when the `SUM_AUDIT_LOG` environment variable
is set to a writable destination. This file is the canonical
description of the row schema for compliance consumers.

The audit log is **regime-agnostic foundation infrastructure**. A
specific compliance regime (GDPR-Art-30, HIPAA-164.514, EU AI Act
Annex IV, NIST AI RMF, internal-audit) implements its own validator
on top by tailing the JSONL file and checking per-regime required
fields.

## Enabling

```bash
# Append every CLI operation's audit row to a file
export SUM_AUDIT_LOG=/var/log/sum/audit.jsonl

# Or stream to stdout (rare; useful for piping into another tool)
export SUM_AUDIT_LOG=-

# Disable (default)
unset SUM_AUDIT_LOG
```

The trust loop's existing semantics are preserved when the audit log
is unset; the canonical bundle / receipt remains the load-bearing
trust artifact regardless of audit-log configuration.

## Row format (additive; consumers should ignore unknown keys)

Every row is a single line of compact JSON terminated by `\n`:

```json
{
  "schema": "sum.audit_log.v1",
  "timestamp": "2026-05-01T18:35:14.123Z",
  "operation": "attest" | "verify" | "render",
  "cli_version": "0.5.0",
  ...operation-specific fields...
}
```

### Required fields

| Field         | Type                                    | Notes                                                          |
|---------------|-----------------------------------------|----------------------------------------------------------------|
| `schema`      | string `"sum.audit_log.v1"`             | Pinned. Bumps with breaking schema changes only.               |
| `timestamp`   | ISO 8601 UTC string ending in `Z`       | Millisecond precision. Compliance time-series stores ingest these. |
| `operation`   | one of `"attest"`, `"verify"`, `"render"` | Operation kind.                                                |
| `cli_version` | string                                  | Version of `sum-cli` that produced the row.                    |

### Operation-specific fields

#### `operation: "attest"`

| Field                  | Type                                   | Notes                                                        |
|------------------------|----------------------------------------|--------------------------------------------------------------|
| `source_uri`           | string                                 | `sha256:<hex>` of the input bytes, or caller-supplied URI.   |
| `axiom_count`          | int                                    | Number of triples extracted.                                 |
| `state_integer_digits` | int                                    | Decimal-digit count of the Gödel state integer.              |
| `extractor`            | string `"sieve"` \| `"llm"`             | Which extractor produced the triples.                        |
| `branch`               | string                                 | Bundle branch metadata.                                      |
| `signed`               | bool                                   | True if Ed25519 signature is attached.                       |
| `hmac`                 | bool                                   | True if HMAC-SHA256 is attached.                             |
| `input_format`         | string                                 | e.g. `"plaintext"`, `"pdf"`, `"html"`.                       |

#### `operation: "verify"`

| Field                  | Type                                                          | Notes                                                        |
|------------------------|---------------------------------------------------------------|--------------------------------------------------------------|
| `ok`                   | bool                                                          | Verification verdict.                                        |
| `axiom_count`          | int                                                           | Reconstructed axiom count.                                   |
| `state_integer_digits` | int                                                           | Decimal-digit count.                                         |
| `branch`               | string                                                        | Bundle branch.                                               |
| `signatures`           | object `{hmac: status, ed25519: status}`                      | Each `status` ∈ `"absent"`, `"verified"`, `"skipped"`, `"invalid"`. |
| `extraction`           | object `{extractor, verifiable, source}`                      | Whether re-extraction is reproducible (sieve only).          |

#### `operation: "render"`

| Field                  | Type                                       | Notes                                                            |
|------------------------|--------------------------------------------|------------------------------------------------------------------|
| `mode`                 | `"local-deterministic"` \| `"worker"`      | Render path used.                                                |
| `axiom_count_input`    | int                                        | Triples in the input bundle.                                     |
| `tome_chars`           | int                                        | Character length of the rendered tome.                           |
| `sliders`              | object with the 5 axes                     | The slider position used for rendering.                          |
| `render_receipt_kid`   | string (worker mode only)                  | The signing key ID of the worker's render receipt.               |
| `render_receipt_schema`| string (worker mode only)                  | e.g. `"sum.render_receipt.v1"`.                                  |
| `worker_url`           | string (worker mode only)                  | The worker URL POSTed to.                                        |

## Fail-open semantics

If `SUM_AUDIT_LOG` points at an unwritable destination (path with
non-existent intermediate directory, full disk, permission denied,
etc.), the CLI operation **proceeds without raising**. The audit
log is *advisory*: the canonical bundle / receipt remains the
load-bearing trust artifact regardless of audit-log status.
Compliance tooling is responsible for monitoring the audit
destination's health independently.

This decision is deliberate: a non-functional audit destination must
not break the trust loop. A degraded audit pipeline is a
compliance-tooling problem; a broken `sum verify` is a substrate
problem.

## Concurrency

Writes use `O_APPEND` on POSIX. Single-line JSONL records well under
the `PIPE_BUF` atomic-write threshold on every platform we support.
Multiple `sum` processes writing to the same audit log produce a
serialised total ordering of rows.

## Cross-referencing across operations

The three operations (`attest`, `verify`, `render`) share several
fields suitable for joining across rows:

- `axiom_count` (attest) ≡ `axiom_count` (verify) ≡ `axiom_count_input` (render)
- `state_integer_digits` cross-references attest ↔ verify
- `source_uri` (attest) provides a stable identifier auditors can use
  to trace re-attestation events

## Use as a compliance primitive

A regime-specific validator is just `tail -f audit.jsonl |
your-validator`. SUM ships one first-party validator today:

- **EU AI Act Article 12** (`sum compliance check --regime
  eu-ai-act-article-12 --audit-log audit.jsonl`). Pins per-row
  traceability fields + operation-specific anchors. Wire spec at
  [`COMPLIANCE_EU_AI_ACT_ARTICLE_12.md`](COMPLIANCE_EU_AI_ACT_ARTICLE_12.md).

Examples of what other downstream validators might enforce:

- **GDPR Article 30 (Records of Processing Activities):** require
  `source_uri` + `branch` + `cli_version` on every row; cross-
  reference with separately-tracked data-subject identifiers.
- **HIPAA 164.514 (de-identification verification):** require
  `extractor: "sieve"` on every PHI-relevant attest (no LLM
  extractor on PHI by policy) plus a separately-maintained
  PHI-bundle allowlist.
- **EU AI Act Annex IV (high-risk system technical documentation):**
  require `signed: true` (Ed25519-attested) on every render
  operation in the high-risk classification, plus retention of
  the render receipts referenced by `render_receipt_kid`.
- **Internal forensics:** retain audit log + the corresponding
  bundle / receipt files indefinitely; cross-reference timestamps
  with HR / sales-process events.

These are *examples*. The audit log makes no regime-specific
assumptions; the validator is the regime-specific layer.
