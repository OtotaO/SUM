# Changelog

All notable changes to the `sum-engine` package. Dates in ISO-8601 UTC.

## [Unreleased]

(next release will move these entries under a version heading)

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

[Unreleased]: https://github.com/OtotaO/SUM/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/OtotaO/SUM/releases/tag/v0.2.1
[0.2.0]: https://github.com/OtotaO/SUM/releases/tag/v0.2.0
[0.1.0]: https://github.com/OtotaO/SUM/releases/tag/v0.1.0
