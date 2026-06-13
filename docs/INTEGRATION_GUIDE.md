# Integration Guide

**Audience:** SOC integrators, AI safety teams, compliance officers.
**Purpose:** the shortest path from "I have SUM installed" to "SUM is producing audit-grade evidence in my pipeline."
**Status:** stable; the surfaces this guide describes are pinned by [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md), [`docs/AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md), [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md), and the per-regime validators under [`docs/COMPLIANCE_*.md`](.).

This document is a triage layer. It does NOT restate the wire specs, regime validators, or proofs that live in other docs — it points each audience to the right per-domain doc and gives the *minimum recipe* to start producing usable evidence. If you read more than one section you should also read [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) so you know exactly what a signed SUM artefact does and does NOT prove.

---

## 0. At a glance

| Audience | What SUM gives you | Minimum integration | Per-domain doc |
|---|---|---|---|
| SOC integrator | JSONL audit-log rows (`sum.audit_log.v1`) per attest / verify / render call, plus Ed25519-signed render receipts (`sum.render_receipt.v1`) | Set `SUM_AUDIT_LOG=/var/log/sum/audit.jsonl`; tail the file into Splunk / Elastic / Sentinel; verify receipts on incident review | [`AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md), [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md), [`INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) |
| AI safety team | Cross-runtime byte-identical canonical form for arbitrary AI text output; signed receipts whose `model` + `provider` fields reflect what *actually* served; multi-vendor adapter (Anthropic, OpenAI, HF Inference Providers); MCP server exposing attest/verify as tools | `pip install 'sum-engine[openai,anthropic]'`; `sum attest` the LLM's output; verify with `sum verify` or the Node / browser verifiers | [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md), [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md), [`MCP_INTEGRATION.md`](MCP_INTEGRATION.md), [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) |
| Compliance officer | Per-regime validators that turn a single audit log into evidence for **EU AI Act**, **GDPR**, **HIPAA**, **ISO 27001**, **SOC 2**, **PCI DSS** — all from one `sum.audit_log.v1` shape | `sum compliance check --audit-log audit.jsonl --regime eu-ai-act-article-12` (run `sum compliance regimes` for the full id list) | [`COMPLIANCE_EU_AI_ACT_ARTICLE_12.md`](COMPLIANCE_EU_AI_ACT_ARTICLE_12.md), [`COMPLIANCE_GDPR_ARTICLE_30.md`](COMPLIANCE_GDPR_ARTICLE_30.md), [`COMPLIANCE_HIPAA_164_312_B.md`](COMPLIANCE_HIPAA_164_312_B.md), [`COMPLIANCE_ISO_27001_8_15.md`](COMPLIANCE_ISO_27001_8_15.md), [`COMPLIANCE_SOC_2_CC_7_2.md`](COMPLIANCE_SOC_2_CC_7_2.md), [`COMPLIANCE_PCI_DSS_4_REQ_10.md`](COMPLIANCE_PCI_DSS_4_REQ_10.md) |

---

## 1. SOC integrators

### What you get

- **Structured audit-log rows** — one JSONL row per CLI operation (`attest`, `verify`, `render`). Schema `sum.audit_log.v1`. Forward-compatible: consumers MUST ignore unknown keys.
- **Signed render receipts** — every `/api/render` call emits a `sum.render_receipt.v1` receipt: detached JWS over a JCS-canonical payload, Ed25519 (RFC 8032) keys distributed via JWKS at `/.well-known/jwks.json`. Verifiable in Python, Node, or any browser — no issuer infrastructure required for verification.
- **Tampering detection** — the AkashicLedger event log is a SHA-256 Merkle hash chain; `verify_chain()` reports the first broken link. Concurrency-hardened (`BEGIN IMMEDIATE` write transactions) so the invariant holds under concurrent writers.

### Minimum recipe

```bash
# 1. Enable structured audit logging
export SUM_AUDIT_LOG=/var/log/sum/audit.jsonl

# 2. Run the operations as normal
echo "Alice likes cats." | sum attest > bundle.json
sum verify --strict < bundle.json

# 3. Tail the audit log into your SIEM
#    Splunk:   forwarder monitors /var/log/sum/audit.jsonl
#    Elastic:  Filebeat input type: log, paths: [/var/log/sum/audit.jsonl]
#    Sentinel: Azure Monitor Agent custom log collection rule
```

Every operation appends one line; rotate with `logrotate` as you would any structured-log source.

### Verifying a receipt during incident review

If a downstream consumer flagged a render receipt as suspect, the verifier surface is the source of truth:

```bash
# Python verifier (requires sum-engine[receipt-verify])
python <<'PY'
import json, urllib.request
from sum_engine_internal.render_receipt import verify_receipt

with open('receipt.json') as f:
    receipt = json.load(f)
req = urllib.request.Request(
    'https://sum-demo.ototao.workers.dev/.well-known/jwks.json',
    headers={'user-agent': 'curl/8'},
)
with urllib.request.urlopen(req, timeout=10) as r:
    jwks = json.loads(r.read())

result = verify_receipt(receipt, jwks)
print('verified:', result.verified, 'kid:', result.kid)
PY

# Node verifier (no install required if you have node + the standalone bundle)
node standalone_verifier/verify.js bundle.json

# Browser verifier (offline, paste-and-verify)
open single_file_demo/index.html
```

All three produce byte-identical accept/reject outcomes per the K1–K4 cross-runtime gate matrix (see [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.3.1).

### Operator runbooks

Eight incident-response runbooks are documented in [`INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md), covering: render-key compromise (case 1), JWKS drift (case 2), PyPI release compromise (case 3), GHA workflow compromise (case 4), Worker deploy compromise (case 5), benchmark claim later wrong (case 6), LLM provider model drift (case 7), canonicalisation bug (case 8). Each runbook names the detection surface, the on-call decision, and the rollback steps.

---

## 2. AI safety teams

### What you get

- **Cross-vendor portability.** SUM signs the semantic provenance of an LLM render at production time, with the receipt's `provider` field reflecting what actually served — `anthropic`, `openai`, `cf-ai-gateway-anthropic`, `cf-ai-gateway-openai`, or `canonical-path` for the deterministic (no-LLM) tome path. The model field carries the API's echoed `model`, never the configured-default, so silent snapshot drift is visible.
- **MCP surface.** Any MCP-aware client (Claude Desktop, Claude Code, Cursor, Continue, custom agents) can call `extract`, `attest`, `verify`, `inspect`, `schema`, `render` as tools — see [`MCP_INTEGRATION.md`](MCP_INTEGRATION.md). Lets an AI-safety review pipeline collect attested outputs from any tool-using agent without per-agent integration.
- **Slider product contract.** Five-axis adjustable rendering (density, length, formality, audience, perspective) with per-axis fact-preservation thresholds and NLI-audited median 1.000 / p10 0.818 (short-doc, n=8) — see [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md). Every render carries its slider position in the signed receipt, so the transform is reproducible from the receipt alone.

### Minimum recipe

```bash
# 1. Install with the LLM adapters you need
pip install 'sum-engine[openai,anthropic,receipt-verify]'

# 2. Set API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Attest an LLM output and emit a signed bundle
echo "$(curl -s api.openai.com/.../chat/completions | jq -r .choices[0].message.content)" \
    | sum attest --ed25519-key keys/render.priv > bundle.json

# 4. Render with the slider product (Worker endpoint)
curl -X POST https://sum-demo.ototao.workers.dev/api/render \
     -H 'content-type: application/json' \
     -d '{
       "triples": [["alice","likes","cats"]],
       "slider_position": {"density":1.0,"length":0.5,
                           "formality":0.5,"audience":0.5,"perspective":0.5},
       "provider": "openai"
     }' | jq .render_receipt
```

### What the receipt does and does NOT prove

**Proves:** the issuer signed this exact tuple (`render_id`, `sliders_quantized`, `triples_hash`, `tome_hash`, `model`, `provider`, `signed_at`, `digital_source_type`) at the given time, using the key identified by `kid`. Bound to a JCS-canonical payload byte-identical across Python / Node / browser verifiers.

**Does NOT prove:** factual truth of the tome's claims; freshness on cache HIT (the receipt carries the original issuance time); the issuer's beliefs about defaults; that the slider's fact-preservation invariant held for *this specific render* (that's a separate bench measurement). See [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §2.

### MCP wiring for an evaluation pipeline

```bash
# Install the MCP server
pip install 'sum-engine[mcp,sieve]'

# Add to your MCP client config (Claude Desktop / Code / Cursor / Continue)
# See docs/MCP_INTEGRATION.md for the per-client config blocks.
# The server exposes: extract / attest / verify / inspect / schema / render
```

A tool-using agent that calls `sum_mcp.attest` after every LLM-generated tool output produces a stream of signed bundles your evaluation pipeline can replay deterministically.

### Multi-vendor benchmarking

The Python `sum_engine_internal.ensemble.llm_dispatch.get_adapter(model)` routes by model-id prefix:

- `claude-*` → AnthropicAdapter
- `gpt-* / o1-* / o3-* / o4-*` → OpenAIAdapter
- `org/model` (HF-namespaced) → HuggingFace Inference Providers router
- anything else → `ValueError` (explicit refusal; no silent provider misrouting)

This is the same dispatcher the §2.5 frontier-model benchmarks use. Cross-family corroboration on §4.7.4 confirms `STRUCTURAL_GAP_NO_MODEL_BEATS` in 3/3 corpora at controlled sample sizes — the dispatcher is the substrate that makes such claims auditable.

---

## 3. Compliance officers

### What you get

One audit log shape (`sum.audit_log.v1`), six per-regime validators. The validator reads the JSONL, walks every row, and emits a `sum.compliance_report.v1` document listing in-scope events, out-of-scope events, and any violations.

### Minimum recipe

```bash
# 1. Enable structured audit logging at the application level
export SUM_AUDIT_LOG=/var/log/sum/audit.jsonl

# 2. Run the operations as normal — the application produces evidence
sum attest --source-uri https://example.com/article-42 < article.txt > bundle.json
sum verify --strict < bundle.json

# 3. Run the regime-specific compliance check.
#    Flag is `--audit-log`; regime ids are canonical and must match what
#    `sum compliance regimes` emits. All six are wired into the CLI.
sum compliance check --audit-log /var/log/sum/audit.jsonl --regime eu-ai-act-article-12
sum compliance check --audit-log /var/log/sum/audit.jsonl --regime gdpr-article-30
sum compliance check --audit-log /var/log/sum/audit.jsonl --regime hipaa-164-312-b
sum compliance check --audit-log /var/log/sum/audit.jsonl --regime iso-27001-8-15
sum compliance check --audit-log /var/log/sum/audit.jsonl --regime soc-2-cc-7-2
sum compliance check --audit-log /var/log/sum/audit.jsonl --regime pci-dss-4-req-10

# Each emits a sum.compliance_report.v1 JSON document.
# Available regimes:
sum compliance regimes
```

### Regime-to-doc map

| Regime | Pin doc | What the validator checks |
|---|---|---|
| EU AI Act Article 12 (logging) | [`COMPLIANCE_EU_AI_ACT_ARTICLE_12.md`](COMPLIANCE_EU_AI_ACT_ARTICLE_12.md) | Article 12 logging-completeness predicates over `sum.audit_log.v1` |
| GDPR Article 30 (records of processing) | [`COMPLIANCE_GDPR_ARTICLE_30.md`](COMPLIANCE_GDPR_ARTICLE_30.md) | Processing-record fields and retention markers |
| HIPAA 45 CFR §164.312(b) (audit controls) | [`COMPLIANCE_HIPAA_164_312_B.md`](COMPLIANCE_HIPAA_164_312_B.md) | Audit-control completeness on PHI-touching operations |
| ISO/IEC 27001:2022 Control 8.15 | [`COMPLIANCE_ISO_27001_8_15.md`](COMPLIANCE_ISO_27001_8_15.md) | Logging control, retention, and tamper-evidence |
| SOC 2 Trust Services CC 7.2 | [`COMPLIANCE_SOC_2_CC_7_2.md`](COMPLIANCE_SOC_2_CC_7_2.md) | System monitoring + event-log preservation |
| PCI DSS 4.0 Requirement 10 | [`COMPLIANCE_PCI_DSS_4_REQ_10.md`](COMPLIANCE_PCI_DSS_4_REQ_10.md) | Logging requirements over user-id-tracked operations |

### EU AI Act Article 50 / GPAI Code of Practice context

The Code of Practice on Transparency of AI-Generated Content describes a *multilayered* approach combining visible disclosure with machine-readable metadata/watermark. SUM's signed render receipts (`sum.render_receipt.v1`) sit cleanly in the machine-readable-metadata layer — the receipt's `digital_source_type` field uses the C2PA v2.2 vocabulary (`trainedAlgorithmicMedia` for LLM-served renders, `algorithmicMedia` for deterministic ones), so a downstream verifier can distinguish AI-generated from algorithm-generated output using a standards-aligned predicate.

The Article 50 enforcement date is **2026-08-02** (a grace window to 2026-12-02 for the machine-readable marking obligation on pre-existing systems is provisionally agreed under the Digital Omnibus); the Code of Practice on Transparency of AI-Generated Content was **finalized 2026-06-10**. Compliance teams targeting EU-facing AI features should treat `sum.render_receipt.v1` evidence as in-scope material for transparency-obligation documentation.

### Threat model and proof boundary

Two documents bound what a signed SUM artefact does and does not prove. Read both before signing off on SUM as compliance-evidence infrastructure:

- [`THREAT_MODEL.md`](THREAT_MODEL.md) — 18-row attack-surface table covering bundle tampering, key compromise, render-receipt forgery, JWKS drift, LLM provider silent model drift, supply-chain compromise, etc. Each row names the defence and the residual risk.
- [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — the proved-vs-measured discipline applied to every claim in the repo. §1.3.1 covers the cross-runtime Ed25519 trust triangle; §2 covers continuous-enforcement against mutualism breakdown.

---

## 4. Cross-cutting: verifying a SUM bundle from scratch

The verification recipe is identical across all three audiences when you're inspecting a specific artefact:

```bash
# Python — verify_receipt(receipt, jwks) accepts the receipt dict
# and a parsed JWKS; raises VerifyError on any tamper. The tome bytes
# are committed via payload.tome_hash inside the signed envelope — to
# verify they match the served tome, recompute sha256 separately and
# compare to receipt['payload']['tome_hash'].
pip install 'sum-engine[receipt-verify]'
python <<'PY'
import json, hashlib
from sum_engine_internal.render_receipt import verify_receipt

with open('receipt.json') as f:
    receipt = json.load(f)
with open('jwks.json') as f:
    jwks = json.load(f)
result = verify_receipt(receipt, jwks)
print('signature ok:', result.verified, 'kid:', result.kid)

# Optional: confirm the served tome matches payload.tome_hash
served_tome = open('tome.txt').read()
expected = 'sha256-' + hashlib.sha256(served_tome.encode('utf-8')).hexdigest()
print('tome bytes match:', expected == receipt['payload']['tome_hash'])
PY

# Node — single file, no install
node standalone_verifier/verify.js bundle.json

# Browser — open the single-file demo, paste the bundle
open single_file_demo/index.html
```

All three are kept byte-equivalent by the K1 / K1-multiword / K2 / K3 / K4 cross-runtime gate, which runs on every PR. Gate failures block merges. See [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.3.1 for the formal statement.

---

## 5. Related specifications

For when you need to implement against a SUM artefact rather than just consume it:

- [`API_REFERENCE.md`](API_REFERENCE.md) — CLI and Python API
- [`AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md) — `sum.audit_log.v1` row schema
- [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — `sum.render_receipt.v1` wire spec
- [`TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) — algorithm registry + key history
- [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) — slider product semantics + per-axis thresholds
- [`MCP_INTEGRATION.md`](MCP_INTEGRATION.md) — MCP server tools + client wiring
- [`CANONICAL_ABI_SPEC.md`](CANONICAL_ABI_SPEC.md) — Gödel-state canonical-form ABI
- [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — proved vs measured discipline
- [`THREAT_MODEL.md`](THREAT_MODEL.md) — attack surface + residual risks
- [`INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) — eight operator runbooks
- [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) — semver + forward-compat rules
