Knowledge distilled into prime-factored integers, signed, and verified cross-runtime — `pip install sum-engine`, mint a CanonicalBundle from prose, verify anywhere.

## Current State

SUM is live on PyPI as `sum-engine` — `0.1.0` first release (2026-04-22), `0.2.0`/`0.2.1` namespace + version-reporting hygiene (2026-04-23), `0.3.0` agentic introspection (2026-04-23: `sum ledger list|stats|head`, `sum inspect`, `sum schema`). What's locked in code today:

| Metric | Value | Epistemic status |
|---|---|---|
| Canonical round-trip drift (seed_v1) | 0.00 % | **proved** — Ouroboros protocol, [`scripts/bench/runners/roundtrip.py`](scripts/bench/runners/roundtrip.py) |
| Extraction F1 on `seed_v1` (50 docs) | 1.000 (precision 1.000, recall 1.000) | **empirical-benchmark** — [`scripts/bench/run_bench.py`](scripts/bench/run_bench.py), corpus `scripts/bench/corpora/seed_v1.json` |
| Extraction F1 on `seed_v2` (20 docs, difficulty corpus) | 0.762 (precision 1.000, recall 0.615) | **empirical-benchmark** — every remaining miss is a recall gap, not a truth inversion. See [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) §2.1 |
| Regeneration FActScore on `seed_v1` | 0.940 (`gpt-4o-mini-2024-07-18` as generator + entailment checker) | **empirical-benchmark** — [`scripts/bench/runners/regeneration.py`](scripts/bench/runners/regeneration.py) |
| LLM narrative round-trip drift on `seed_v1` | 107.75 % (facts preserved, surface keys not — see §2.5) | **empirical-benchmark** — [`scripts/bench/runners/llm_roundtrip.py`](scripts/bench/runners/llm_roundtrip.py) |
| Fortress gate | 21 / 21 passing | **proved** — [`scripts/verify_fortress.py`](scripts/verify_fortress.py) |
| Cross-runtime kill-experiments (Python ↔ Node ↔ Browser) | 5 / 5 green (K1 / K1-multiword / K2 / K3 / K4) | **proved** — [`scripts/verify_cross_runtime.py`](scripts/verify_cross_runtime.py), CI-gated |
| Merkle-chain integrity under concurrent writers | holds (50–200-event bursts) | **proved** — [`Tests/test_akashic_ledger.py`](Tests/test_akashic_ledger.py), post commit `9c4139d` |
| Test suite size | 1000+ collected | **proved** — `python -m pytest Tests/ --collect-only -q` (1035 at v0.3.0, rounded per contract) |
| Feature catalog entries | 82 Production, 14 Scaffolded, 2 Designed | **proved** — [`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md), every row has a named verification command |

**Agentic CLI surface (v0.3.0, on PyPI):**

```bash
pip install 'sum-engine[sieve]'
echo "Alice likes cats. Bob owns a dog." | sum attest --extractor=sieve | sum verify
# → sum: ✓ verified 2 axiom(s), state integer matches (hmac=absent, ed25519=absent)
```

**Full cryptographic attestation, opt-in and composable:**

- `--signing-key KEY` — HMAC-SHA256 for shared-secret peers. **proved** tamper detection via [`Tests/test_adversarial_bundles.py`](Tests/test_adversarial_bundles.py).
- `--ed25519-key PEM` — Ed25519 signatures under the W3C VC 2.0 `eddsa-jcs-2022` cryptosuite. Bundles verifiable by any DIF-conformant verifier (Universal Resolver, Digital Bazaar, Spruce ssi, Veramo). **proved** round-trip via [`Tests/test_sum_cli_attest_ed25519.py`](Tests/test_sum_cli_attest_ed25519.py) and cross-runtime K3/K4.
- `--ledger DB` — per-triple byte-level ProvenanceRecords into a SQLite AkashicLedger. `sum resolve <prov_id>` walks axiom → source byte range. **proved** round-trip via [`Tests/test_sum_cli_ledger.py`](Tests/test_sum_cli_ledger.py).

**Agentic introspection (v0.3.0):**

- `sum ledger list --db DB [--axiom KEY] [--since ISO] [--limit N]` — enumerate prov_ids as NDJSON. **proved** via [`Tests/test_sum_cli_agentic.py`](Tests/test_sum_cli_agentic.py) `TestLedgerList`.
- `sum ledger stats --db DB` — one-shot summary: record counts, timestamp range, Merkle chain tip, branch heads. **proved** via `TestLedgerStats`.
- `sum ledger head --db DB [--branch NAME]` — current state integer per branch (string-encoded for arbitrary precision). **proved** via `TestLedgerHead`.
- `sum inspect bundle.json` — structural read of a bundle's shape without running crypto verification. Surfaces axiom-count divergence for agents that want to route before paying the full verify cost. **proved** via `TestInspect`.
- `sum schema {bundle|provenance|credential}` — JSON Schema (Draft 2020-12) for each output shape, so agents can validate programmatically. **proved** via `TestSchema`, with coverage asserting the bundle schema's required fields are a subset of what `sum attest` actually emits.

**Cross-runtime trust triangle:** the same bundle bytes verify in Python (`sum verify`), Node (`standalone_verifier/verify.js` via WebCrypto), and modern browsers (`single_file_demo/index.html` via SubtleCrypto — Chrome 113+, Firefox 129+, Safari 17+). Tampered-bundle rejection and positive-path acceptance are both locked into CI via kill-experiments K3 (positive) and K4 (negative).

## Future Directions

Ordered by leverage. Each item names scaffolded code that exists today and what it becomes when wired.

1. **Wikidata QID annotations at extraction.** Today triples are opaque strings like `alice||likes||cats`; annotating with `wd:Q25169||wd:P1082||wd:Q17` unlocks semantic-web interop and collision-free identity. Scaffolding: [`sum_engine_internal/algorithms/predicate_canon.py`](sum_engine_internal/algorithms/predicate_canon.py) already normalises predicates; the QID-resolver service slot is reserved in [`sum_engine_internal/ensemble/live_llm_adapter.py`](sum_engine_internal/ensemble/live_llm_adapter.py) `extract_triplets`. When wired, every emitted ProvenanceRecord carries entity URIs and joins the Linked-Open-Data graph.

2. **`sha256_128_v2` prime scheme with hard-fail collision semantics.** 64-bit seeds are astronomically safe for low-thousands-of-axioms corpora; 128-bit seeds remove the collision floor entirely. Scaffolding: [`sum_engine_internal/algorithms/semantic_arithmetic.py`](sum_engine_internal/algorithms/semantic_arithmetic.py) and [`standalone_verifier/math.js`](standalone_verifier/math.js) already export `derivePrimeV2` with BPSW primality; [`core-zig/src/main.zig:365`](core-zig/src/main.zig) ships `sum_get_deterministic_prime_v2`; 20-fixture v2 cross-runtime parity test passes. Activation flips the default scheme and emits a new `CANONICAL_FORMAT_VERSION` for transport.

3. **P2P gossip mesh — lock-free mathematical consensus.** Two nodes that independently process the same prose produce identical primes, so merge is `math.lcm` — zero coordination. Scaffolding: [`sum_engine_internal/infrastructure/p2p_mesh.py`](sum_engine_internal/infrastructure/p2p_mesh.py) (168 LOC, gossip protocol + peer registry) plus the Gödel Sync Protocol shape in [`api/quantum_router.py`](api/quantum_router.py). Production activation needs peer discovery + persistent peer list; mathematics is already proved by feature 2 (cross-runtime state equivalence).

4. **WASM core for browser-native computation.** [`core-zig/src/main.zig:19`](core-zig/src/main.zig) exports `wasm_alloc_bytes` / `wasm_free_bytes` today; the eight BigInt operations (gcd, lcm, mod, batch-mint-primes, etc.) are already `export fn` so compiling to wasm32 is a build-flag flip. Replaces the demo's JS BigInt fallback with microsecond-latency prime derivation.

5. **AT Protocol Lexicon (`io.sum.axiom`).** Bluesky-compatible record schema for publishing signed bundles to a user-controlled PDS, making SUM the first attested-knowledge layer over AT Proto. Scaffolding: [`sum_engine_internal/infrastructure/verifiable_credential.py`](sum_engine_internal/infrastructure/verifiable_credential.py) already ships `did:key` + `did:web` helpers that satisfy AT Proto's DID requirements; the Lexicon schema is the one additional file.

6. **ZK entailment proofs — "I know this fact without revealing the fact".** Scaffolding: [`sum_engine_internal/algorithms/zk_semantics.py:69`](sum_engine_internal/algorithms/zk_semantics.py) ships `verify_proof`. Current form is a structural proof stub; wiring a real Schnorr or Bulletproof adapter on the Ed25519 keypair already required by the issuer closes the loop.

## Technical Stack

Ships today, nothing aspirational:

- **Core arithmetic** — Pure Python integer math (sympy.nextprime, `math.lcm`, `math.gcd`). Zig fast path via C-ABI when `core-zig/zig-out/lib/libsum_core.{dylib,so}` is present; otherwise identical pure-Python fallback (`sum_engine_internal/infrastructure/zig_bridge.py`).
- **Prime scheme (production)** — `sha256_64_v1`. First 8 bytes of SHA-256 over `subject||predicate||object` → `nextprime(seed)`. Deterministic, cross-runtime byte-identical.
- **Canonical codec** — `sum_engine_internal/infrastructure/canonical_codec.py`. Optional HMAC-SHA256, optional Ed25519; downgrade-protection when a signing key is configured; downgradeable-to-structural when not.
- **Canonical semantic ABI** — `The {subject} {predicate} {object}.` one-axiom-per-line, parsed by a single regex shared between Python, Node, and browser (`standalone_verifier/math.js::CANONICAL_LINE_REGEX`).
- **Verifiable Credentials** — RFC 8785 JCS-canonicalised VC 2.0 with `eddsa-jcs-2022` cryptosuite (`sum_engine_internal/infrastructure/verifiable_credential.py`). `did:key` + `did:web` issuer bootstrap at `scripts/generate_did_web.py`.
- **Provenance ledger** — SQLite + Merkle hash-chain + `BEGIN IMMEDIATE` concurrency hardening (`sum_engine_internal/infrastructure/akashic_ledger.py`). Content-addressable `prov_id = sha256(canonical(ProvenanceRecord))`.
- **Extractor** — spaCy `en_core_web_sm` with a deterministic SVO sieve (`sum_engine_internal/algorithms/syntactic_sieve.py`): negation suppression, passive-voice inversion, multi-word-subject underscore-joining. Paired LLM path via OpenAI structured outputs (`sum_engine_internal/ensemble/live_llm_adapter.py`).
- **Single-file browser demo** — `single_file_demo/index.html` (779 lines, zero external deps). SubtleCrypto Ed25519 verification, same bundle schema as the CLI.
- **Cloudflare Worker** — `worker/src/index.ts` serves `single_file_demo/` as static assets plus `/api/complete` (Anthropic / OpenAI proxy) and `/api/qid` (Wikidata resolver stub for Phase 4a). Migrated from the prior Pages deployment per Cloudflare's April 2026 convergence guidance; Secrets Store replaces the previous env-var key handling.
- **Python package** — `sum-engine` on PyPI. `sum` CLI binary (`attest` / `verify` / `resolve`). Python ≥ 3.10.
- **Standalone Node verifier** — `standalone_verifier/verify.js`, zero npm dependencies, self-test + v2-parity-test built in.
- **CI** — `.github/workflows/quantum-ci.yml` runs Zig tests, the 1000+ pytest suite, the 21-check fortress gate, the cross-runtime harness (K1/K1-mw/K2/K3/K4), the PORTFOLIO.md epistemic-label contract check, and a pip-install smoke that builds the wheel and runs `echo prose | sum attest | sum verify` in a fresh venv on every push. `.github/workflows/publish-pypi.yml` handles OIDC-authenticated PyPI releases on version tags.
- **Docs** — `docs/PROOF_BOUNDARY.md` (proved vs. measured discipline), `docs/FEATURE_CATALOG.md` (103 numbered features, each with a reproduce-verification command), `docs/DID_SETUP.md` (did:web / did:key runbook), `CHANGELOG.md`.
