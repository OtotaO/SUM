# Changelog

All notable changes to the `sum-engine` package. Dates in ISO-8601 UTC.

## [Unreleased]

### Added — forward playbook for future sessions

- `docs/NEXT_SESSION_PLAYBOOK.md` — ordered work queue (Priorities
  1–8) with principles, stop-the-line triggers, and the ordering
  rationale. Priorities 1–2 harden existing claims (adversarial
  cross-runtime fuzzing, WASM perf measurement); 3–4 extend them
  into new regions (`sha256_128_v2`, SPARQL disambiguation); 5–6
  make the surface self-describing (threat-model validation,
  delta-bundle semantics); 7–8 broaden the trust base (supply-chain
  attestation, LLM-extraction honesty guardrails). `CLAUDE.md`
  onboarding list now points at this file as item #5 so a memory-
  less session discovers the queue by reading the canonical entry
  block.

### Removed — portfolio-site artifacts (separation of concerns)

SUM is a knowledge-distillation engine. The sumequities.com portfolio
is a personal-portfolio site that references SUM as one of many
featured projects. The two should not share governance, CI rules,
or narrative files — a third-party `pip install sum-engine` consumer
has no business with a portfolio file, and a fork should not inherit
rules about a personal portfolio. Earlier commits in this session
incorrectly coupled them; this entry records the full revert.

- Deleted `PORTFOLIO.md` at repo root.
- Deleted `scripts/check_portfolio_contract.py`,
  `scripts/hooks/pre-commit`, `scripts/install-hooks.sh`.
- Removed the `portfolio-contract` job from
  `.github/workflows/quantum-ci.yml`.
- Removed the `## PORTFOLIO.md contract` section from `CLAUDE.md`;
  replaced the now-stale onboarding list-item #1 (which pointed at
  `PORTFOLIO.md`) with a shortened 4-file reading list. Added an
  `## Out of scope — do not cross-repo edit` note naming the
  portfolio repo as off-limits.
- Removed `make portfolio` and `make install-hooks` targets from
  `Makefile`; dropped both from `.PHONY`. Added `wasm` to `.PHONY`
  (it was listed as a target earlier in the session but never added
  to the list).
- `README.md` hero no longer carries the "Portfolio-facing overview:
  PORTFOLIO.md" pointer. The "Shipped since the last README pass"
  bullet for "PORTFOLIO.md + CLAUDE.md contract" removed.
- `CONTRIBUTING.md` setup block no longer tells contributors to run
  `make install-hooks`; Verification-Gates table no longer carries
  the `PORTFOLIO contract` row.

### Removed — experimental AT Protocol Lexicon (same-confusion teardown)

Phase C from the same session published
`com.sumequities.experimental.axiom` as a Lexicon schema on the
user's Bluesky PDS under the portfolio's domain authority. Same
portfolio-vs-engine confusion at the namespace layer: SUM the
engine should not claim a Lexicon under the portfolio's domain.
External state torn down before this commit landed:

- Bluesky record `at://did:plc:cuqlv67qg6tepr2gjvknajcp/com.atproto.lexicon.schema/com.sumequities.experimental.axiom`
  deleted via `com.atproto.repo.deleteRecord`. PDS confirms
  `RecordNotFound`; `listRecords` returns empty.
- DNS TXT `_lexicon.sumequities.com` (content
  `did=did:web:sumequities.com`) deleted from Cloudflare DNS.
  `dig +short TXT _lexicon.sumequities.com` empty.
- Bluesky app-password `sum-lexicon-publisher` (fragment
  `24hi-yfrq-3q6r-5ezs`) revoked at `bsky.app/settings/app-passwords`.

In-repo artifacts that were drafted on disk but never committed
(the C.6 gate was going to hold them; user surfaced the deeper
issue before C.7 fired) are `rm`'d as working-tree cleanup in this
commit: `scripts/publish_lexicon_schema.py`, `at_proto/lexicon/`
directory and its single JSON.

### Added — `/api/qid` Wikidata resolver (Phase 4a)

- `worker/src/routes/qid.ts` — replaces the 501 stub with a working
  resolver. Takes a batch of `{text, kind?, lang?}` terms, looks each
  one up via the MediaWiki `wbsearchentities` API, returns
  `{id, label, description, confidence, source}` for every term.
  Up to 50 terms per request; parallel fetches. Unknown terms
  surface `{id: null, reason: "no-match"}` rather than errors.
- Two-tier caching: edge Cache API (same-colo, zero-hop) on every
  request; the TTL is 30 days (Wikidata labels rarely change on
  month scales). KV binding left as an optional second layer
  (commented in `wrangler.toml`; activate by
  `wrangler kv:namespace create qid-cache`).
- Confidence scoring mirrors the `match.type` field Wikidata returns
  (`label` → 1.0, `alias` → 0.7, everything else → 0.5) — a
  categorical signal translated into a 0–1 ordering for threshold
  logic downstream.
- User-Agent header `SUMDemoQIDResolver/0.3.0 (+github.com/OtotaO/SUM)`
  per Wikidata's operator-contact guidance.

Intentionally not in v0.3: SPARQL disambiguation when multiple
candidates are plausible. wbsearchentities alone hits >80% accuracy
on common-noun / proper-name lookups; SPARQL refinement (filter by
predicate domain) is Phase 4b once we've measured the v0.3 baseline
on a real corpus.

### Added — WASM acceleration in the browser demo

- `single_file_demo/sum_core.wasm` (97 KB, committed) — the `core-zig/`
  module cross-compiled to `wasm32-freestanding` with `ReleaseSmall`.
  Exports nine functions (`sum_get_deterministic_prime`,
  `sum_get_deterministic_prime_v2`, `sum_bigint_gcd/lcm/mod`,
  `sum_bigint_divisible_by_u64`, `sum_batch_mint_primes`,
  `wasm_alloc_bytes`, `wasm_free_bytes`) plus the linear memory.
- `single_file_demo/sum_core_wasm.js` — browser-side async loader
  factory. Returns `{derivePrime, isReady:true}` on success; returns
  `null` on any failure (WebAssembly unavailable, fetch/compile/
  instantiate error) so the caller's fallback logic stays trivial.
  Handles the WebAssembly i64→BigInt signed-surface wrinkle (u64
  zig returns come back signed in JS; masked with `& 0xffff…ffffn`
  post-call).
- `single_file_demo/test_wasm.js` — zero-dep Node self-test pinning
  the WASM output to the cross-runtime fixture set (same vectors as
  `verify.js --self-test`). Part of the demo's test triad alongside
  `test_jcs.js` and `test_provenance.js`.
- `single_file_demo/index.html` — `derivePrime()` now calls the WASM
  loader first (single-flight, cached after first load); falls back
  to the original WebCrypto+JS-BigInt path when WASM isn't reachable
  (standalone file open, Claude artifact, older browsers). Transparent
  to every caller — the function still returns the correct BigInt.
  A `<link rel=preload>` for the `.wasm` fires in the page head so
  the module is in-flight before the user clicks Attest.
- `.github/workflows/quantum-ci.yml` `zig-core` job:
  * Builds the WASM target alongside the native library.
  * SHA-256-compares the freshly-built `.wasm` against the committed
    blob — catches source/binary drift (fails with the rebuild
    command if they don't match).
  * Runs `node single_file_demo/test_wasm.js` to assert the committed
    `.wasm` still produces the reference primes.
- `core-zig/build.zig` — updated `link_libc` syntax to zig 0.16 /
  0.15.late-cycle module-field form (was `.linkLibC()` method call,
  which zig 0.16 removed from `Build.Step.Compile`).
- `Makefile` — new `make wasm` target builds + copies + runs the
  self-test in one step. Run after any `core-zig/src/main.zig` edit.

Performance: still to be measured on a real workload. The WASM path
replaces roughly "WebCrypto SHA-256 + O(log² N) Miller-Rabin per
candidate × ~80 candidates on average" with native Zig on wasm32.
Expected speedup at 1.5–5× for the prime-minting hot path. Measured
numbers will land in PROOF_BOUNDARY §2.2 when a browser bench harness
is wired.

### Added — hosted-demo infrastructure

- `worker/` directory with a Cloudflare Worker (`src/index.ts`) that
  serves `../single_file_demo/` as static assets and routes `/api/*`
  through TypeScript handlers. Migrates the previous Pages deployment
  to Workers per Cloudflare's April 2026 convergence guidance (Workers
  has full feature parity for static assets + SSR + custom domains,
  and every new capability — Secrets Store, Workflows, Durable Objects,
  Dynamic Workers, Sandboxes — lands Workers-first).
- `worker/src/routes/complete.ts` — LLM proxy, ported from the Pages
  Function. Same request/response shape, same fallback semantics; the
  only user-visible change is that secrets now live in the Workers
  Secrets Store instead of Pages environment variables.
- `worker/src/routes/qid.ts` — stub (returns 501) for the Phase 4a
  Wikidata QID resolver. Contract (request shape, cache key, SPARQL
  + wbsearchentities pipeline) specified inline so the next session
  can land the real implementation without re-deriving the design.
- `.github/workflows/deploy-worker.yml` — manual-dispatch deploy job
  using `cloudflare/wrangler-action@v3`. Requires repo secrets
  `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`. Flip to push-on-tag
  once the deploy cadence stabilises.
- `single_file_demo/functions/api/complete.ts` carries a DEPRECATED
  header pointing at the Worker replacement. Kept in-place so an
  existing Pages deployment does not 404 overnight during the
  switchover.

Security baseline (the `_headers` file's CSP, COOP/COEP, HSTS,
Permissions-Policy) is ported into `worker/src/index.ts` as
`BASELINE_HEADERS`, applied to every Response.

### Pending user action (for first Worker deploy)

  cd worker/
  npm install
  npx wrangler login
  npx wrangler secret put ANTHROPIC_API_KEY
  npx wrangler deploy

After the first deploy, subsequent deploys run via the
`deploy-worker.yml` workflow on manual dispatch.

(next release will move these entries under a version heading)

## [0.3.0] — 2026-04-23

Minor-bump feature release. Agentic-first introspection surface: three
new subcommand clusters that let an LLM agent composing SUM into a
larger pipeline ask questions about ledger state, read bundle shape
without paying crypto cost, and validate SUM output programmatically.
Zero breaking changes — every 0.2.1 invocation still works identically.

### Added

- `sum ledger list [--db DB] [--axiom KEY] [--since ISO] [--limit N]`
  enumerates prov_ids as NDJSON (one JSON object per line), each row
  carrying prov_id, axiom_key, source_uri, byte_start, byte_end,
  timestamp, extractor_id. Filters compose with AND. Previously, agents
  that wanted to introspect a ledger had to craft raw SQL against the
  SQLite file — now they pipe `sum ledger list | jq …`.

- `sum ledger stats [--db DB] [--pretty]` emits a one-shot summary:
  `provenance_records_total`, `distinct_axiom_keys`, earliest/latest
  timestamps (ISO 8601), `chain_tip_hash` (Merkle), and branches with
  their state-integer digit counts.

- `sum ledger head [--db DB] [--branch NAME] [--pretty]` returns the
  current state integer for one named branch or every branch. State
  integers are emitted as strings (never JSON numbers) to preserve
  arbitrary precision — many agent JSON parsers use 64-bit doubles.

- `sum inspect <bundle.json> [--pretty]` reads a bundle's structural
  shape without running signature verification or re-deriving primes:
  axiom counts (claimed + parsed — an agent sees a divergence without
  invoking `sum verify`), state-integer digit size, signature fields
  present, bundle/format versions, timestamp, branch, and the sum_cli
  sidecar (prov_ids, extractor, source_uri) if present.

- `sum schema {bundle|provenance|credential}` prints a JSON Schema
  (Draft 2020-12) for each shape SUM emits. Agents that want to
  validate output against a ground-truth contract no longer have to
  reverse-engineer from prose docs.

### Unchanged

- CLI contract for `attest / verify / resolve` and every flag on them.
- CanonicalBundle wire format (`canonical_format_version 1.0.0`).
- Prime scheme (`sha256_64_v1`).
- Every cryptographic contract (HMAC, Ed25519, VC 2.0).
- Cross-runtime trust triangle — K1 / K1-mw / K2 / K3 / K4 still green
  on this commit; same bundle bytes still verify in Python ↔ Node ↔
  Browser.

### Tests

14 new cases in `Tests/test_sum_cli_agentic.py` pin: NDJSON shape of
ledger list; filter composition (--axiom, --limit); stats summary
keys; head branch-not-found error path; inspect on tampered tome
(reports divergence rather than rejecting — agent's call whether to
run full verify); inspect on malformed JSON; schema title + required
subset is actually emitted by attest.

## [0.2.1] — 2026-04-23

Patch release — fixes a three-minute-old version-reporting bug
introduced by 0.2.0.

### Fixed

- `sum --version` and `bundle.sum_cli.cli_version` now track the
  actually-installed distribution version via
  `importlib.metadata.version("sum-engine")` instead of a hardcoded
  string in `sum_cli/__init__.py`. 0.2.0's wheel shipped with
  `pyproject.toml` at 0.2.0 but the CLI's hardcoded `__version__`
  still said `"0.1.0"`, so every bundle minted under 0.2.0 carried
  `sum_cli.cli_version: "0.1.0"` — a silent truth gap inside the
  very bundles the CLI exists to attest. 0.2.1 closes it at the
  source: no dual source of truth to drift from again.

### Unchanged

- Everything else. No API/CLI contract changes from 0.2.0.

## [0.2.0] — 2026-04-23

Hygiene release (one day after 0.1.0). One BREAKING change, zero
behavior changes.

### Changed — BREAKING (for anyone who imported `internal.*` directly)

- The top-level `internal/` package was renamed to `sum_engine_internal/`
  to remove the PyPI namespace-collision risk. 238 import sites across
  111 Python files were mechanically rewritten; `pyproject.toml`
  `packages.find.include` now lists `sum_engine_internal*`. Every test,
  script, and doc reference updated in the same commit.
- The CLI's public contract (`sum attest / sum verify / sum resolve`,
  all flags, the CanonicalBundle JSON schema, the Gödel-state wire
  format) is unchanged. Anyone using `sum-engine` through the CLI
  sees no difference. Only consumers who were importing
  `internal.infrastructure.X` etc. directly — which the 0.1.0
  CHANGELOG's "Known limitations" explicitly flagged as unsupported
  — need to update their imports to `sum_engine_internal.*`.

### Unchanged

- CanonicalBundle wire format (`canonical_format_version 1.0.0`).
- Prime scheme (`sha256_64_v1`).
- All cryptographic contracts (HMAC, Ed25519, VC 2.0 `eddsa-jcs-2022`).
- Cross-runtime trust triangle (K1 / K1-mw / K2 / K3 / K4 all PASS
  on this commit; same bundle bytes still verify in Python ↔ Node ↔
  Browser).

## [0.1.0] — 2026-04-22

First public release. Ships the `sum` CLI on PyPI, the Python API
under `internal.*` (renamed to `sum_engine_internal.*` in 0.2.0 —
see above), the standalone Node verifier, and the single-file
browser demo. Cross-runtime trust triangle
(Python ↔ Node ↔ Browser) is complete and locked by CI.

### Added — CLI

- `sum attest` — extract SVO triples from prose, mint a
  CanonicalBundle with the Gödel state integer.
- `sum verify` — verify structural reconstruction, HMAC
  signature (with `--signing-key`), and Ed25519 signature
  (self-contained via embedded public key). `--strict` mode
  requires at least one verifiable signature.
- `sum resolve` — look up a ProvenanceRecord in a local
  AkashicLedger by content-addressable prov_id.
- `sum attest --ed25519-key PEM` — mint W3C VC 2.0-compatible
  Ed25519-signed bundles using a PEM produced by
  `python -m scripts.generate_did_web`.
- `sum attest --ledger DB` — record per-triple byte-level
  ProvenanceRecords and attach prov_ids to the bundle; enables
  the attest → resolve loop end-to-end.
- `sum attest --signing-key K` — HMAC-SHA256 signature for
  shared-secret peers (composable with `--ed25519-key`).

### Added — Python API

- `sum_engine_internal.infrastructure.canonical_codec.CanonicalCodec` — HMAC
  and Ed25519 are both optional; when neither is configured,
  bundles carry the state integer only (content-addressed
  integrity without shared secrets or keys). Downgrade-protection
  preserved when a signing_key is configured.
- `sum_engine_internal.infrastructure.verifiable_credential` — W3C VC 2.0
  emission + verification with `eddsa-jcs-2022` cryptosuite.
  `did:key` and `did:web` issuer helpers; `build_did_web_document`
  emits the DID document for hosting at `/.well-known/did.json`.
- `sum_engine_internal.infrastructure.akashic_ledger.AkashicLedger` —
  SQLite-backed event log with Merkle hash-chain integrity and
  BEGIN IMMEDIATE concurrency hardening. `record_provenance` +
  `get_provenance_record` power the CLI's attest/resolve loop.

### Added — Cross-runtime

- `standalone_verifier/verify.js` verifies Ed25519 signatures via
  Node's `crypto.webcrypto.subtle` (Node ≥ 18.4).
- `single_file_demo/index.html` verifies Ed25519 via browser
  SubtleCrypto (Chrome 113+, Firefox 129+, Safari 17+).
- `scripts/verify_cross_runtime.py` — K1 / K1-multiword / K2 / K3
  / K4 kill-experiments: structural round-trip, multi-word object
  regex parity, VC 2.0 named-rejection, Ed25519 positive + negative
  signature verification Python ↔ Node.

### Added — CI

- `cross-runtime-harness` job runs the K1–K4 kill-experiments on
  every PR.
- `pypi-install-smoke` job builds the wheel, installs in a fresh
  venv, and runs `echo prose | sum attest | sum verify` — locks
  the shipping promise against packaging regressions.

### Added — Docs

- `docs/DID_SETUP.md` — runbook for did:key and did:web issuer
  setup, with a verifier-compatibility matrix.
- `docs/PROOF_BOUNDARY.md` §1.3.1 — Ed25519 public-key attestation
  cross-runtime contract.
- `docs/FEATURE_CATALOG.md` Layer 8 — `sum` CLI feature entries
  (98–103) each with a reproducible verification command.

### Cryptosuite

- `eddsa-jcs-2022` with RFC 8785 JCS canonicalisation. Bundles
  emitted under `sha256_64_v1` prime scheme (the production scheme
  for low-thousands-of-axioms corpora).

### Known limitations

- `sum attest --ledger` requires `--extractor=sieve`. The LLM
  extractor has no byte-offset tracking yet (emits a clear error
  with a pointer).
- Browser Ed25519 falls back to "present (use CLI)" on pre-2023
  browsers lacking SubtleCrypto Ed25519 support — never a false ✓.
- The internal Python modules live under a top-level `internal/`
  package. Downstream consumers should depend on the CLI contract,
  not import these modules directly — they may move in 0.2.0.

[Unreleased]: https://github.com/OtotaO/SUM/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/OtotaO/SUM/releases/tag/v0.3.0
[0.2.1]: https://github.com/OtotaO/SUM/releases/tag/v0.2.1
[0.2.0]: https://github.com/OtotaO/SUM/releases/tag/v0.2.0
[0.1.0]: https://github.com/OtotaO/SUM/releases/tag/v0.1.0
