# SUM docs index

The 17 active docs in this directory, grouped by reader. Pick by what you're trying to do.

## I want to verify SUM's claims

- **[`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md)** — the arbiter. Every claim in the repo traces to one of four epistemic categories here: `provable` / `certified` / `empirical-benchmark` / `expert-opinion`. Read this first.
- **[`THREAT_MODEL.md`](THREAT_MODEL.md)** — what the trust surface defends against and what it does not.
- **[`FEATURE_CATALOG.md`](FEATURE_CATALOG.md)** — every shipped feature with a one-command verification recipe. 117 entries.

## I want to integrate SUM into my system

- **[`API_REFERENCE.md`](API_REFERENCE.md)** — the hosted Cloudflare Worker HTTP API. `/api/render`, `/api/qid`, `/api/complete`, `/.well-known/jwks.json`, `/.well-known/revoked-kids.json`. Use when the caller is a web app, mobile app, or server-side service.
- **[`MCP_INTEGRATION.md`](MCP_INTEGRATION.md)** — the `sum-mcp` Model Context Protocol server. Use when the caller is a local LLM client (Claude Desktop, Claude Code, Cursor, Continue).
- **[`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md)** — `sum.render_receipt.v1` wire spec. Ed25519 over JCS-canonical bytes, detached JWS, JWKS-distributed keys. Read this if you need to verify a receipt yourself.

## I want to understand the bundle / ledger / cryptographic primitives

- **[`CANONICAL_ABI_SPEC.md`](CANONICAL_ABI_SPEC.md)** — `CanonicalBundle` wire spec and the `sum_engine` ABI surface.
- **[`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md)** — schema-version policy for the `sum.*.v1` artifacts. What changes are additive, what require a major bump.
- **[`ALGORITHM_REGISTRY.md`](ALGORITHM_REGISTRY.md)** — supported cryptographic algorithms and their fail-closed posture. NLI model registry and `sha256_128_v2` design history are linked from here (full design docs in `archive/`).
- **[`MERKLE_SIDECAR_FORMAT.md`](MERKLE_SIDECAR_FORMAT.md)** — set-commitment sidecar (M1 prototype). Wire spec for `sum.merkle_inclusion.v1`.
- **[`TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md)** — TUF-inspired trust-root manifest format. Supersedes the standalone Rekor/CT design notes (which are linked from inside this doc).
- **[`TRUST_ROOT_LOG.md`](TRUST_ROOT_LOG.md)** — append-only log of trust-root manifests issued, by date and `kid`. Audit data, not spec.

## I want to operate or extend SUM

- **[`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md)** — the slider's product contract. Five axes, per-axis drift formulas, fact-preservation thresholds. Includes the v0.2-research-pass history that informs current decisions.
- **[`INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md)** — operator runbook. Kid revocation, JWKS rollback, bundle-format incidents.
- **[`MODULE_AUDIT.md`](MODULE_AUDIT.md)** — activation checklists for the 13 scaffolded (🔧) features in `FEATURE_CATALOG.md`.
- **[`DID_SETUP.md`](DID_SETUP.md)** — one-time issuer-keypair generation for `did:web` / `did:key` deployments.

## Project process

- **[`NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md)** — work queue and stop-the-line triggers. Session-shaped rather than user-facing; included for transparency.

## Archive

`docs/archive/` holds session-shaped or superseded documents preserved for git history and external-link continuity:
- `WASM_PERFORMANCE.md` — older WASM benchmark notes
- `MODEL_CALL_EVIDENCE_FORMAT.md` — design for an unshipped surface
- `DEMO_RECORDING.md` — screen-recording instructions
- `STAGE3_128BIT_DESIGN.md` — `sha256_128_v2` design rationale
- `SLIDER_V02_RESEARCH.md` — v0.2 slider-substrate research; load-bearing decisions folded into `SLIDER_CONTRACT.md`
- `NLI_MODEL_REGISTRY.md` — supported NLI models; folded into `ALGORITHM_REGISTRY.md`
- `FORMAL_MODELS.md` — formal-verification roadmap; folded into `PROOF_BOUNDARY.md` §3
- `TRANSPARENCY_ANCHOR.md` — Rekor/CT anchoring design; folded into `TRUST_ROOT_FORMAT.md`
- Older archive entries (`ULTIMATE_VISION.md`, etc.) preserve pre-v0.4 vision documents

These are reachable in the public repo via the `archive/` path so external links survive. Folded content has a forwarding pointer at the original path.
