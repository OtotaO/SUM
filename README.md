# SUM — verifiable bidirectional knowledge distillation

[![CI](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml/badge.svg)](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml)
[![PyPI — sum-engine](https://img.shields.io/pypi/v/sum-engine.svg?label=PyPI%20sum-engine)](https://pypi.org/project/sum-engine/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **SUM lets people and agents transform knowledge without losing the ability to verify what changed, what stayed the same, who signed it, and what remains unproven.**

Every transformation — extract triples from prose, render a tome at a controlled slider position, compose bundles across documents, share a render — emits a cryptographically-signed receipt that any third party can verify offline. The receipt attests *that the transformation happened and what its inputs were*. Separate per-axis benchmarks attest *how much the transformation preserved meaning*. Both are kept honest by separate proof discipline — and the project never blurs the line between them.

*Live trust loop:* https://sum-demo.ototao.workers.dev — three runtimes (Python, Node, modern browsers) produce byte-identical Ed25519 signatures over the same JCS-canonical bytes; verify offline against `/.well-known/jwks.json`. Mechanically proven; locked in CI on every PR.

**Built for:** journalists working under deepfake-era citation requirements, academic survey writers who need provenance back to source PDFs, agentic-AI builders who need their agents to pass verifiable evidence and not just messages, and regulated-domain content (EU AI Act Article 12, FTC AI disclosure, HIPAA, SOC 2, PCI DSS) where "we say it's true" isn't enough.

The cryptographic side is **mechanically proven** — three independent verifier implementations agreeing byte-for-byte on every signed bundle, locked in CI on every PR. The semantic side (extraction quality, slider fact preservation) is **empirically measured** with explicit per-corpus numbers and explicit per-corpus boundaries. [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) is the arbiter.

Headline supporting numbers (each links to its source of truth):

| Claim | Status | Source |
|---|---|---|
| Three-runtime byte-symmetric Ed25519 over JCS bytes | provable; locked by `make xruntime` (K1–K4) + `make xruntime-adversarial` (A1–A6) | [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §1.2, §1.3.1 |
| Canonical round-trip `reconstruct(parse(canonical_tome(S))) == S` | provable; 0.00% drift on every CI run | [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §1.1 |
| Render receipt — `sum.render_receipt.v1`, Ed25519 / JCS / detached JWS | shipped; verifier in three runtimes | [`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md) |
| Slider fact preservation: median 1.000, p10 0.769 (long n=16) / 0.818 (short n=8) | empirical-benchmark — measured; same-commit replay receipt still pending (bench-hardening T2/T3) | [`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md) |
| Extraction F1 = 1.000 (`seed_v1`), 0.762 with precision 1.000 (`seed_v2`) | empirical-benchmark | [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §2.1 |

A render receipt verifies the *render attestation* (issuer signed this tome, these triples, this slider position, this model, at this time). It does not verify the truth of the tome's content — that is what the slider bench measures separately. See [`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md) §5 for the explicit trust scope.

---

## Why it matters

More of what people read is now produced or reshaped by AI — summarised, translated, distilled, rewritten. As that grows, the ability to check *what changed, what was preserved, and what was lost* stops being a nicety and becomes shared infrastructure for a trustworthy information commons.

SUM is built to be that layer **in the open**: Apache-2.0, offline-verifiable by anyone, and aligned with open standards (C2PA `digital_source_type`, W3C VC 2.0, JOSE / JWS / JWKS) rather than a proprietary trust silo. It does not ask you to trust *SUM* — any third party verifies the receipt themselves, in three independent runtimes, and the project states plainly where proof ends and measurement begins. The aim is a checkable **chain of custody for knowledge in motion**, not another walled garden.

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
| `pip install 'sum-engine[sieve]'` — `sum attest` / `sum verify` / `sum render` / `sum resolve` / `sum ledger` / `sum inspect` / `sum schema` | shipped on PyPI ≥ 0.4.1 | structural reconstruction; HMAC-SHA256 + Ed25519 signatures (W3C VC 2.0 `eddsa-jcs-2022`); bidirectional `sum attest` ↔ `sum render` symmetry from the shell |
| Cloudflare Worker at `sum-demo.ototao.workers.dev` | shipped | `/api/render` → tome + `render_receipt`; `/api/transform` → generic transform-registry dispatch + `sum.transform_receipt.v1`; `/api/complete` → LLM proxy; `/api/qid` → Wikidata resolver; `/.well-known/jwks.json` + `/.well-known/revoked-kids.json` → trust-loop endpoints. Public LLM-axis routes are rate-limited per IP — see [`docs/PUBLIC_API_RATE_LIMITS.md`](docs/PUBLIC_API_RATE_LIMITS.md) (5/day operator-keyed demo; 100/hr with BYO key via `X-Render-LLM-Key-Anthropic` / `-OpenAI`). |
| Single-file browser demo (`single_file_demo/index.html`) | shipped | paste prose → in-browser attest → CanonicalBundle JSON; same bytes verify under `node standalone_verifier/verify.js` (Chrome / Firefox / Safari with WebCrypto Ed25519 support) |
| Cross-runtime trust triangle | locked by CI (`make xruntime`) | K1 / K1-mw / K2 / K3 / K4 — Python ↔ Node ↔ Browser agree byte-for-byte on valid bundles. `make xruntime-adversarial` adds A1–A6 rejection-class equivalence. |
| 5-axis slider rendering surface | density actioned deterministically; length / formality / audience / perspective LLM-conditioned. Two dispatch paths: Worker `/api/render` (Anthropic + Cloudflare AI Gateway optional) producing `sum.render_receipt.v1`, OR Python `sum transform apply slider` (OpenAI via `OPENAI_API_KEY`) producing `sum.transform_receipt.v1` | bench: median LLM-axis fact preservation 1.000, p10 0.769 (long, n=16) / 0.818 (short, n=8), order preservation 1.000 wherever measurable. Tightening worktrail at [`docs/BENCH_HARDENING_FROM_QCVV.md`](docs/BENCH_HARDENING_FROM_QCVV.md) adds iteration-stability + DKW worst-case bounds + capability-region headlines |
| MCP server (`sum-mcp` console script) | shipped | five tools (`extract` / `attest` / `verify` / `inspect` / `schema`) exposed over stdio; bundles attested via MCP verify byte-identically through the CLI / Node / browser verifiers |
| Transform substrate (`sum.transform_receipt.v1` + registry) | shipped on PyPI ≥ 0.7.0 | `sum transform list` / `sum transform apply <name>` — three registered transforms (`slider` / `extract` / `compose`); receipts via Ed25519 / JCS / detached JWS just like render-receipts; 20-fixture cross-runtime K-matrix locks accept + reject across Python ↔ Node ↔ browser; T4 `source_chain_hash` binds receipts to source byte ranges; T5 `ShareableRender` round-trips signed renders for offline verification; T6 multi-school extract runs two extractors in tandem for adversarial-divergence detection. Wire spec at [`docs/TRANSFORM_RECEIPT_FORMAT.md`](docs/TRANSFORM_RECEIPT_FORMAT.md); design at [`docs/TRANSFORM_REGISTRY.md`](docs/TRANSFORM_REGISTRY.md). |
| Replay-defense window (`signed_at_out_of_window`) | shipped | opt-in `max_age_seconds` parameter across all four verifier surfaces (Python render / Python transform / JS render / JS transform). Default-off preserves archival use; receivers opt in per use-case (agent-swarm 60s, real-time 600s, newsletter 1d, legal-discovery no window). |
| `sum verify --explain` layered output | shipped | Per-dimension report (`sum.verify_explained.v1`): cryptographic integrity / canonical reconstruction / axiom consistency / extraction provenance / source evidence coverage / semantic preservation / truth of content. Each carries `epistemic_status` (`provable` / `certified` / `empirical-benchmark` / `not-asserted`). Truth of content is ALWAYS `not_asserted` — locked by test. |
| Negative-control corpus (T5 of bench-hardening) | shipped | 20 hand-authored documents across 5 failure modes (ambiguous coref / predicate-alias / contradictions / entity-resolution-adversarial / non-extractable). Runner exits 1 if observed failures don't match annotations. Baseline at [`fixtures/bench_receipts/negative_control_2026-05-17.json`](fixtures/bench_receipts/negative_control_2026-05-17.json). |
| Compliance validators (six regimes) | shipped | `sum compliance check --regime <id> --audit-log <path>` — EU AI Act Article 12, GDPR Article 30, HIPAA § 164.312(b), ISO/IEC 27001 A.8.15, SOC 2 CC 7.2, PCI DSS v4.0 Req 10. All six produce the same `sum.compliance_report.v1` schema; per-regime docs at `docs/COMPLIANCE_*.md`. |

The slider's product claim — *axis changes do not lose facts* — is the load-bearing empirical result. It is verified by NLI audit on every embedding-flagged "loss" cell; full attribution in [`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md). In keeping with the "what remains unproven" half of the promise above: these headline numbers are **measured observations**, not yet same-commit-replayable — the bench harness (`Tests/benchmarks/slider_drift_bench.py`) is scaffold-state and no `sum.slider_drift_bench.v1` receipt is committed. Closing that to a replayable receipt is bench-hardening tasks T2 / T3 ([`docs/BENCH_HARDENING_FROM_QCVV.md`](docs/BENCH_HARDENING_FROM_QCVV.md)); see the reproducibility-status note in [`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md).

## Strategic context

The operational compass — read in this order if you want the project's intent + how it operates + where it's going:

- [`docs/CHARTER_2026-05-17.md`](docs/CHARTER_2026-05-17.md) — intent, the Why, strategy, objectives, success criteria, constraints, and the operational loop. The compass every other doc resolves to.
- [`docs/PRODUCT_DELIBERATION_2026-05-14.md`](docs/PRODUCT_DELIBERATION_2026-05-14.md) — three-option strategic analysis + grant-outcome decision tree.
- [`docs/ZENITH_FRAMING_2026-05-16.md`](docs/ZENITH_FRAMING_2026-05-16.md) — destination framing (SUM as chain-of-custody for AI-transformed knowledge) plus three new concepts (Perspective Receipts, Trust Profiles, Epistemic Nutrition Label) on the design queue.
- [`docs/BENCH_HARDENING_FROM_QCVV.md`](docs/BENCH_HARDENING_FROM_QCVV.md) — five-task empirical-benchmark hardening plan (T1–T5; T5 shipped, T1–T4 queued).
- [`docs/DOGFOOD_QUICKSTART.md`](docs/DOGFOOD_QUICKSTART.md) — five-minute guide to running SUM on your own writing.

### LLM narrative round-trip — closed across measured corpora (2026-04-28)

The hardest measurement in `PROOF_BOUNDARY.md` is the full LLM narrative round-trip (`text → LLM-extract → axioms → LLM-generate → prose' → LLM-extract → axioms'`). The unprompted-pipeline baseline on `seed_v1` was **drift = 107.75% / exact-match recall = 0.12** — facts preserved, keys not.

A two-layer generator-side intervention (canonical-first generator prompt + constrained-decoding extractor with vocab-pinned `Literal` enums + lemma-exclusion of source-predicate lemmas from the canonical-padding set) now closes this across every measured corpus shape:

| Corpus | n_docs | axioms / doc | combined recall | drift_pct | full recall |
|---|---:|---:|---:|---:|---:|
| seed_v1 (single-fact SVO) | 50 | 1 | **1.0000** | 0.00 | 50 / 50 |
| seed_v2 (7 difficulty parse patterns + multi-fact) | 20 | 1–2 | **0.9750** | 5.00 | 19 / 20 |
| seed_long_paragraphs (16-topic multi-paragraph) | 16 | 11–28 | **0.9972** | 0.57 | 15 / 16 |

The combined intervention lands **≥ 0.97 recall and ≤ 5 % drift on every measured corpus shape** — single-fact short-form, multi-fact difficulty-pattern, and multi-paragraph dense-prose. The §2.5 closure is corpus-independent. The remaining gap on each corpus traces to upstream LLM source-extraction artifacts (corrupted axioms on seed_v2 doc_015, semantically-duplicate predicates on seed_long solar_system), not to the intervention pattern.

Receipt artifacts:
- [`fixtures/bench_receipts/s25_generator_side_2026-04-28.json`](fixtures/bench_receipts/s25_generator_side_2026-04-28.json) — full ablation matrix on seed_v1.
- [`fixtures/bench_receipts/s25_residual_closure_2026-04-28.json`](fixtures/bench_receipts/s25_residual_closure_2026-04-28.json) — combined + lemma-exclusion on seed_v1.
- [`fixtures/bench_receipts/s25_generator_side_seed_v2_2026-04-28.json`](fixtures/bench_receipts/s25_generator_side_seed_v2_2026-04-28.json) — all three ablations on seed_v2.
- [`fixtures/bench_receipts/s25_generator_side_seed_long_combined_2026-04-28.json`](fixtures/bench_receipts/s25_generator_side_seed_long_combined_2026-04-28.json) — combined ablation on seed_long_paragraphs.

Reproducible: `python -m scripts.bench.runners.s25_generator_side --ablation combined --corpus <path> --out <path>` (~$0.07–$0.20 OpenAI per corpus, ~3–8 min wall clock). Full attribution + per-ablation breakdowns + per-doc failure analysis in [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §2.5.

The deterministic canonical round-trip (the one `sum attest | sum verify` exercises) is **mechanically proven** (§1.1, 0.00% drift). The LLM round-trip is **not**, and this section is here to keep that distinction above the fold.

---

## CLI quick start

```bash
pip install 'sum-engine[sieve]'

echo "Alice likes cats. Bob owns a dog." \
  | sum attest --extractor=sieve > bundle.json

sum verify --input bundle.json
# → sum: ✓ verified 2 axiom(s), state integer matches (hmac=absent, ed25519=absent)

sum render < bundle.json > tome.md
# → bundle's axioms re-emitted as canonical prose; round-trips to the same state integer
```

The reverse direction also runs under explicit slider control. The local path actions only the density slider; non-neutral length / formality / audience / perspective require the LLM extrapolator and route through the hosted Worker:

```bash
sum render --density 0.5 < bundle.json
# → keeps the lex-prefix half of the axioms; @sliders header records what was requested

sum render --length 0.9 --use-worker https://sum-demo.ototao.workers.dev --json < bundle.json
# → LLM-conditioned tome + signed render_receipt (sum.render_receipt.v1) on stdout
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

The hosted Worker at `https://sum-demo.ototao.workers.dev` exposes `/api/render`, `/api/complete`, `/api/qid`, and the `/.well-known/{jwks,revoked-kids}.json` verification surfaces. [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) is the wire spec — request/response shapes, error codes, the six-step receipt-verification flow, working Node + Python examples. Use this when the caller is a web app, mobile app, or server-side service; use the MCP server when the caller is a local LLM client.

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
- **170 numbered features**, each with a reproducible verification command, in [`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md).

### Research substrate (under `sum_engine_internal/research/`)

Less-surfaced but shipped:

- **MinHash-LSH bundle similarity index** (`research/lsh/`) — near-duplicate bundle detection at scale.
- **Robust PCA corruption score** (`research/robust_pca/`) — `corruption_score` field in bundle metadata; flags adversarially-perturbed bundles.
- **Sequential & conformal-prediction** (`research/sequential/`, `research/conformal/`) — bench-side confidence bounds with documented coverage guarantees.
- **MMD distribution distance** (`research/mmd/`) — `axiom_distribution_mmd` field on bundles; surfaces when an attested bundle is structurally unlike its baseline corpus.
- **Spectral entropy** (`research/spectral_entropy/`) — axiom-graph entropy on every bundle, with confidence interval.
- **Bootstrap multiplier spike detection** (`research/bootstrap/`) — see [`docs/MULTIPLIER_BOOTSTRAP_SPIKE_FINDINGS.md`](docs/MULTIPLIER_BOOTSTRAP_SPIKE_FINDINGS.md).
- **SMT consistency checking** (`research/smt_consistency/`) — z3-backed `axiom_consistency_check` on every bundle.
- **Sheaf-Laplacian hallucination detection** — see [`docs/SHEAF_HALLUCINATION_DETECTOR.md`](docs/SHEAF_HALLUCINATION_DETECTOR.md) (research direction).

### Other substrate-adjacent surfaces

- **Trust-root manifest** (`sum_engine_internal/trust_root/`) — operator-issued signed manifest binding kid lifecycle, revocation policy, and verifier expectations.
- **Merkle sidecar format** (`sum_engine_internal/merkle_sidecar/`) — see [`docs/MERKLE_SIDECAR_FORMAT.md`](docs/MERKLE_SIDECAR_FORMAT.md).
- **Evidence-chain layer** (`sum_engine_internal/evidence/`) — substrate behind `source_chain_hash` (T4).
- **Algorithm registry** — see [`docs/ALGORITHM_REGISTRY.md`](docs/ALGORITHM_REGISTRY.md) (the in-tree list of permitted signing algs; crypto-agility gate).
- **Audit log format** — every CLI operation can emit `sum.audit_log.v1` events; see [`docs/AUDIT_LOG_FORMAT.md`](docs/AUDIT_LOG_FORMAT.md).
- **Agent surface** (`sum_engine_internal/agent_surface/`) — see [`docs/AGENT_SURFACE_FINDINGS.md`](docs/AGENT_SURFACE_FINDINGS.md).

### Internal research surfaces (NOT shipped, present in repo)

- **`api/quantum_router.py` + `quantum_main.py`** — FastAPI surface with 26+ endpoints (branchable knowledge graph, ZK semantic proofs, federated KG sync, JWT-tenant knowledge OS). 1,684 LOC; 58/58 tests pass; runs locally via `uvicorn quantum_main:app`. **NOT in the PyPI wheel** (`pyproject.toml` excludes `api*`), **NOT in the live Worker**, **NOT in the dogfood quickstart**. The substrate it composes is load-bearing for the shipping surfaces above; only the FastAPI HTTP layer is internal-research. Promote to a shipping `[api]` extra only if a named buyer or grant deliverable explicitly references one of the endpoint clusters. See top-of-file banner in `api/quantum_router.py` for the full triage rationale.

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

This roadmap names only unshipped work. Items already landed live in [`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]`. Detailed sequencing lives in [`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md).

**Closing the LLM round-trip drift.** This is the headline open problem. The full LLM round-trip (`text → LLM-extract → axioms → LLM-generate → prose' → LLM-extract → axioms'`) currently produces 107.75 % drift and 0.12 exact-match recall on `seed_v1` — facts preserved, keys not. Closing this gap is a canonicalisation problem (entity resolution, predicate normalisation, pinned-vocabulary extraction); none of those passes are shipped yet. See [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §2.5 for the full attribution and per-document failure modes.

**Hardening backlog**

- `sha256_128_v2` default-activation — Python ↔ Node byte-identity now locked (12-key K1-v2 + 6-state K2-v2 gate runs on every PR; `scripts/verify_godel_v2_cross_runtime.py`). The default scheme stays `sha256_64_v1`; flipping the default is a separate operator decision that requires a `bundle_version` minor bump per `docs/COMPATIBILITY_POLICY.md`. The migration path is now empirically open.
- `/api/qid` accuracy floor — **measured 2026-04-28** on a 30-term hand-curated corpus across people, places, concepts, and common nouns: **hit-rate 100% (30/30), label-substring-match 100% (24/24, excluding 6 common-noun rows)**. Receipt at [`fixtures/bench_receipts/qid_accuracy_2026-04-28.json`](fixtures/bench_receipts/qid_accuracy_2026-04-28.json) under schema `sum.qid_resolution_accuracy.v1`. **Boundary:** label-substring match accepted `relativity` → `Q201607 (Relativity Records)` — a music-label entity, not the physics theory. The two-tier metric is robust to wbsearchentities's quirks but does not measure semantic-accuracy against canonical Q-IDs; that's a follow-on with hand-verified ground-truth pairs. The current resolver is a thin layer over `wbsearchentities`; SPARQL-driven disambiguation that prefers the most-linked-to entity for ambiguous terms remains an unshipped enhancement.
- Threat-model validation — every documented defence in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) gets an executable test.
- Delta-bundle composition semantics — specifies what `bundle.is_delta` means cross-runtime.
- Sigstore / cosign signing of release artifacts.
- LLM-extraction honesty guardrails — `extraction.verifiable: true | false` so signed ≠ true is visible at the consumer interface.
- Calibration-set authoring for the Venn-Abers conformal-interval implementation that already ships.
- Remaining sieve recall work on `seed_v2` (apposition / relative-clause / compound-conjunct) — gated on the §2.5 work, see [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §6.

**Platform surface (post-hardening)**

Source anchoring in the bundle schema, bundle explorer / viewer, `sum verify --explain`, `sum tutorial` onboarding, shareable bundle URLs `/b/{hash}`, PWA-installable demo, `sum attest <url>` fetch mode. Each item names its dependency in [`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md).

---

## Verification surface

`make help` lists every dev command. Common targets:

```bash
make install              # editable install with sieve + dev extras
make test                 # full pytest run (2000+ tests)
make xruntime             # cross-runtime K1/K1-mw/K2/K3/K4 (Python ↔ Node)
make xruntime-adversarial # rejection-matrix A1–A6
make fortress             # 21-check pure-math invariants
make smoke                # fresh-venv install + attest|verify round-trip
make demo                 # open the single-file browser demo
```

CI runs the full suite on every push (`.github/workflows/quantum-ci.yml`); the `cross-runtime-harness` job runs K1–K4 + A1–A6 on Node 22; `pypi-install-smoke` builds the wheel and runs `echo prose | sum attest | sum verify` in a throwaway venv.

---

## Epistemic contract

Every claim in this repo carries an explicit epistemic status — `provable`, `certified`, `empirical-benchmark`, or `expert-opinion`. The arbiter is [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md). A summary surface that quotes an empirical-benchmark number alongside language like "mathematically guaranteed" is a policy violation per §5 and must be corrected.

Performance language (`fast`, `efficient`, `low-latency`, `scalable`) requires a benchmark in the same commit. Adversarial input agreement (the A-matrix) is a separate proof from valid-input agreement (the K-matrix); both run in CI.

If a number in this README disagrees with [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) or [`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md), the docs are canonical and this README is wrong.

---

## Contributing

1. Fork and branch.
2. `make install && make test && make xruntime`.
3. Read [`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md) for principles, stop-the-line triggers, and the work-ordering rule.
4. Open a PR. Every claim added to docs or commit messages must trace to a test, a measurement, or an explicit `designed, not proved` label.

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the test-gate matrix and the verification-gate runbook.

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).
