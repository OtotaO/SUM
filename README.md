# SUM — verifiable bidirectional knowledge distillation

[![SUM Knowledge OS CI](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml/badge.svg)](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml)
[![PyPI — sum-engine](https://img.shields.io/pypi/v/sum-engine.svg?label=PyPI%20sum-engine)](https://pypi.org/project/sum-engine/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **NLI audit + 4-layer fact-preservation substrate + cryptographically-signed render receipts = verifiable fact preservation across slider axis changes, with median 1.000 / p10 0.769 measured at scale, catastrophic-failure mode eliminated, and every render carrying its own Ed25519-signed attestation. Live at https://sum-demo.ototao.workers.dev.**

Every term in that claim is empirically true at the current `main` head (PR #41 v0.9.A + PR #42 v0.9.A.2) and traces to a source-of-truth document a reader can follow:

| Claim term | Source of truth |
|---|---|
| NLI audit | [`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md) §"Headline result (NLI-verified, scale-checked)" |
| 4-layer fact-preservation substrate | strict / normalized / semantic / NLI; defined in [`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md), v0.4 entry in [`CHANGELOG.md`](CHANGELOG.md) |
| signed render receipts | [`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md) (`sum.render_receipt.v1`, Ed25519/JCS detached JWS, JWKS) |
| median 1.000 / p10 0.769 | long-doc bench (n=16, 9–24 triples/doc) — [`scripts/bench/corpora/seed_long_paragraphs.json`](scripts/bench/corpora/seed_long_paragraphs.json); short-doc bench p10 = 0.818 (n=8) |
| catastrophic-failure mode eliminated | v0.7 prompt hardening: `≥5 facts lost` cells went 2 → 0 on the same long-doc bench; min preservation lifted from 0.111 → 0.700 ([`CHANGELOG.md`](CHANGELOG.md) v0.7 entry) |
| Live at sum-demo.ototao.workers.dev | curl `https://sum-demo.ototao.workers.dev/api/render` |

A render receipt verifies the *render attestation* (issuer signed this tome, these triples, this slider position, this model, at this time). Fact preservation across axes is verified separately by the bench. See [`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md) §5 (Trust Scope) for what the receipt does and does not prove.

---

## Verify it yourself in 60 seconds

The trust loop: hit the live Worker, get back a tome plus a detached Ed25519 JWS over the JCS-canonicalised receipt payload, fetch the issuer JWKS, verify.

```bash
# 1. JWKS — single Ed25519 OKP JWK, application/jwk-set+json
curl -sS https://sum-demo.ototao.workers.dev/.well-known/jwks.json | jq .
# → {"keys":[{"crv":"Ed25519","kty":"OKP","x":"...","alg":"EdDSA","use":"sig","kid":"sum-render-2026-04-27-1"}]}

# 2. Render — tome + render_receipt (signed JWS over JCS payload)
curl -sS -X POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -d '{"triples":[["alice","graduated","2012"],["alice","born","1990"]],
       "slider_position":{"density":1.0,"length":0.5,"formality":0.7,"audience":0.5,"perspective":0.5}}' \
  | jq '.render_receipt | {schema, kid, payload, jws_segments: (.jws | split(".") | length)}'
```

A minimal Node verifier using `jose` + `canonicalize` is in [`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md) §A.5; the same format is reachable from Python (`joserfc` + `jcs`), Go, and Rust per §3.

---

## What ships today

| Surface | Status | Verifies |
|---|---|---|
| `pip install 'sum-engine[sieve]'` — `sum attest` / `sum verify` / `sum resolve` / `sum ledger` / `sum inspect` / `sum schema` | shipped on PyPI (v0.3.0) | structural reconstruction; HMAC-SHA256 + Ed25519 signatures (W3C VC 2.0 `eddsa-jcs-2022`) |
| Cloudflare Worker at `sum-demo.ototao.workers.dev` | shipped (Phase E.1 v0.9.A.2) | `/api/render` → tome + `render_receipt`; `/.well-known/jwks.json` → JWKS; `/api/qid` → Wikidata resolver |
| Single-file browser demo (`single_file_demo/index.html`) | shipped | paste prose → in-browser attest → CanonicalBundle JSON; same bytes verify under `node standalone_verifier/verify.js` (Chrome 113+, Firefox 129+, Safari 17+) |
| Cross-runtime trust triangle | locked by CI (`make xruntime`) | K1 / K1-mw / K2 / K3 / K4 — Python ↔ Node ↔ Browser agree byte-for-byte on valid bundles. `make xruntime-adversarial` adds A1–A6 rejection-class equivalence. |
| 5-axis slider rendering surface | density actioned deterministically; length / formality / audience / perspective LLM-conditioned via the Worker (Anthropic, Cloudflare AI Gateway optional) | bench: median LLM-axis fact preservation 1.000, p10 0.769 (long, n=16) / 0.818 (short, n=8), order preservation 1.000 wherever measurable |

The slider's product claim — *axis changes do not lose facts* — is the load-bearing empirical result. It is verified by NLI audit on every embedding-flagged "loss" cell; the v0.4 → v0.9 arc traces in [`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]`.

---

## CLI quick start

```bash
pip install 'sum-engine[sieve]'

echo "Alice likes cats. Bob owns a dog." \
  | sum attest --extractor=sieve > bundle.json

sum verify --input bundle.json
# → sum: ✓ verified 2 axiom(s), state integer matches (hmac=absent, ed25519=absent)
```

Add cryptographic attestation with one flag:

```bash
# Ed25519 / W3C VC 2.0 (eddsa-jcs-2022)
python -m scripts.generate_did_web --domain your.example --private-key-out keys/issuer.pem
sum attest --ed25519-key keys/issuer.pem < prose.txt | sum verify --strict
# → hmac=absent, ed25519=verified
```

The same bundle bytes verify under `sum verify` (Python), `node standalone_verifier/verify.js` (WebCrypto), and the in-browser demo (SubtleCrypto). [`docs/DID_SETUP.md`](docs/DID_SETUP.md) walks the did:key / did:web issuer setup. [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §1.3.1 documents what the cross-runtime Ed25519 contract proves.

### Calling SUM from MCP-aware LLM clients

```bash
pip install 'sum-engine[mcp,sieve]'
# Claude Desktop / Claude Code / Cursor / Continue: add to MCP config:
#   { "mcpServers": { "sum": { "command": "sum-mcp" } } }
```

`sum-mcp` exposes `extract`, `attest`, `verify`, `inspect`, `schema` as MCP tools. Bundles attested via MCP verify byte-identically through the CLI / Node / browser verifiers — same canonical codec. See [`docs/MCP_INTEGRATION.md`](docs/MCP_INTEGRATION.md) for the full client wiring.

### Calling SUM over HTTP

The hosted Worker at `https://sum.ototao.com` exposes `/api/render`, `/api/complete`, `/api/qid`, and the `/.well-known/{jwks,revoked-kids}.json` verification surfaces. [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) is the wire spec — request/response shapes, error codes, the six-step receipt-verification flow, working Node + Python examples. Use this when the caller is a web app, mobile app, or server-side service; use the MCP server when the caller is a local LLM client.

---

## How the trust loop fits together

```
prose ─► /api/render ─►  tome
                         + render_receipt {kid, payload, jws}
                                              │
                                              ▼
                                  /.well-known/jwks.json
                                  (Ed25519 OKP JWK by kid)
                                              │
                                              ▼
                              jose.flattenedVerify(JCS(payload))
                                              │
                                              ▼
                          render attested ✓ — issuer signed
                          (this tome, these triples, this slider
                          position, this model, at this time)
```

The receipt is a *render attestation*, not a truth oracle. Fact preservation is verified by the bench (NLI audit on weak cells). The receipt is what a downstream system keeps as durable proof; the tome is what a reader consumes. See [`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md) §5.

---

## Underlying substrate

Below the slider sits the substrate that earlier phases shipped and verified. Pointers, not paraphrase — every claim links to its source-of-truth doc.

- **Canonical round-trip conservation (provable).** `reconstruct(parse(canonical_tome(S))) == S` for every Gödel state `S`. 0.00% drift on `seed_tiny_v1` / `seed_v1` / `seed_v2`. [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §1.1.
- **Cross-runtime state equivalence (provable).** Python (`sympy`), Node (BigInt + Miller-Rabin), in-browser JS produce byte-identical state integers. Locked by 4 harnesses (`make xruntime` + `make xruntime-adversarial`). [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §1.2.
- **Bundle public-key attestation (provable).** Ed25519-signed CanonicalBundles are tamper-detectable by any third party in any of the three runtimes. [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §1.3.1.
- **Merkle hash-chain integrity (provable, including under concurrent writers).** [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §1.7.
- **Extraction F1 (empirical-benchmark).** 1.000 on `seed_v1` (50 simple-SVO docs); 0.762 with precision 1.000 on `seed_v2` (20-doc difficulty corpus). Every remaining `seed_v2` failure is a recall miss, not a truth inversion. [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §2.1.
- **103 numbered features**, each with a reproducible verification command, in [`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md).

---

## Reproduce the bench

```bash
# Short corpus (n=8, 4–12 triples/doc, ~$0.30, ~2 min with NLI)
bash scripts/bench/run_paragraphs.sh

# Long corpus (n=16, 9–24 triples/doc, ~$1.50, ~10 min with NLI)
bash scripts/bench/run_long_paragraphs.sh
```

Both runners require `OPENAI_API_KEY` (NLI audit + extraction). Pinned model snapshots are mandatory; the harness raises `SystemExit` on unpinned identifiers (see [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §2.6). Output is NDJSON `sum.slider_drift_bench.v1`, with per-cell strict / normalized / semantic / NLI fact-preservation columns.

---

## Future developments

This roadmap names only unshipped work. Items already landed live in [`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]`.

**Hardening playbook (Priorities 3–8 in [`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md))**

- `sha256_128_v2` activation — Node side exists, Python side not yet `CURRENT_SCHEME`. Pre-empts the 2³² collision frontier.
- `/api/qid` SPARQL disambiguation — moves entity resolution from the current `wbsearchentities`-only path to a measured >95% accuracy floor.
- Threat-model validation — every documented defence in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) gets an executable test.
- Delta-bundle composition semantics — specifies what `bundle.is_delta` means cross-runtime.
- Sigstore / cosign signing of release artifacts.
- LLM-extraction honesty guardrails — `extraction.verifiable: true | false` so signed ≠ true is visible at the consumer interface.

**Phase E follow-ups**

- v0.9.B browser receipt verifier (in-page detached-JWS verify against the same JWKS).
- v0.9.C Python receipt verifier (`joserfc` + `jcs` against the bench).
- Calibration-set authoring for Venn-Abers conformal intervals.
- Remaining sieve recall work on `seed_v2` (apposition / relative-clause / compound-conjunct).

**Phase B platform surface (post-hardening)**

Source anchoring in the bundle schema (B1), bundle explorer / viewer (B2), `sum verify --explain` (B3), `sum tutorial` onboarding (B4), shareable bundle URLs `/b/{hash}` (B5), PWA-installable demo (B6), `sum attest <url>` fetch mode (B7). Each item names its dependency in [`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md).

---

## Verification surface

`make help` lists every dev command. Common targets:

```bash
make install              # editable install with sieve + dev extras
make test                 # full pytest run (1000+ tests)
make xruntime             # cross-runtime K1/K1-mw/K2/K3/K4 (Python ↔ Node)
make xruntime-adversarial # rejection-matrix A1–A6
make fortress             # 21-check pure-math invariants
make smoke                # fresh-venv install + attest|verify round-trip
make demo                 # open the single-file browser demo
```

CI runs the full suite on every push (`.github/workflows/quantum-ci.yml`); the `cross-runtime-harness` job runs K1–K4 + A1–A6 on Node 22; `pypi-install-smoke` builds the wheel and runs `echo prose | sum attest | sum verify` in a throwaway venv.

---

## Truthfulness contract

Every claim in this repo carries an explicit epistemic status — `provable`, `certified`, `empirical-benchmark`, or `expert-opinion`. The arbiter is [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md). A summary surface that quotes an empirical-benchmark number alongside language like "mathematically guaranteed" is a policy violation per §5 and must be corrected.

Performance language (`fast`, `efficient`, `low-latency`, `scalable`) requires a benchmark in the same commit. Adversarial input agreement (the A-matrix) is a separate proof from valid-input agreement (the K-matrix); both run in CI.

If a number in this README disagrees with [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) or [`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md), the docs are canonical and this README is wrong.

---

## Contributing

1. Fork and branch.
2. `make install && make test && make xruntime`.
3. Read [`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md) for principles, stop-the-line triggers, and the Phase A → B → C → D ordering rule.
4. Open a PR. Every claim added to docs or commit messages must trace to a test, a measurement, or an explicit `designed, not proved` label.

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the test-gate matrix and the verification-gate runbook.

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).
