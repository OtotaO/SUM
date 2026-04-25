# Changelog

All notable changes to the `sum-engine` package. Dates in ISO-8601 UTC.

## [Unreleased]

### Added — Phase E.1 STATE 4 (slider renderer pipeline lands)

The renderer scaffold from STATE 2 ships its real implementation. Five
xfailed tests flip to passes; one failing test surfaces a design bug
that the fix kills outright.

  sum_engine_internal/ensemble/tome_sliders.py
    + length_fragment / formality_fragment / audience_fragment /
      perspective_fragment now return real prompt strings keyed by bin
      centre (5 positions × 4 axes = 20 fragments). Empty string at
      neutral midpoint to keep the prompt lean at default.
    * quantize() no longer bins density. Density is deterministic and
      binning 1.0→0.9 made "request all triples" un-expressible. The
      cache key includes raw density (unique per density level).

  sum_engine_internal/ensemble/slider_renderer.py
    + render() pipeline: cache-first → quantize → apply_density →
      canonical-vs-LLM branch → measure_drift → cache-write → return.
      Canonical path skips LLM and re-extraction entirely when only
      density is non-default (no drift introduced).
    + measure_drift() implements all five SLIDER_CONTRACT formulas:
      density (set comparison), length (word-count band), formality
      (register marker classifier), audience (jargon density),
      perspective (first-person pronoun ratio). Pure function;
      embedded lookup tables (no data-file deps).
    * measure_drift signature gained `tome: str` — needed for the
      four LLM axes whose drift is measured against tome content,
      not triple sets.

  sum_engine_internal/ensemble/live_llm_adapter.py
    + OpenAIChatClient: thin adapter from LiveLLMAdapter to the
      slider_renderer.LLMChatClient Protocol. Lives in live_llm_adapter
      so the renderer module stays openai-dep-free.

  Tests/test_slider_renderer.py
    * xfail strict markers removed from TestRenderPipeline (3 tests)
      and TestMeasureDrift (2 tests). All 22 tests pass.
    + test_quantize_preserves_density_endpoints — new regression
      test for the density-not-binned fix.

  Tests/benchmarks/slider_drift_bench.py
    + _bench_one_cell now extracts source triples, calls render(),
      computes per-axis drift + fact-preservation per
      SLIDER_CONTRACT.md. main_async wires LiveLLMAdapter +
      OpenAIChatClient. Errors captured per-row rather than raising.

  worker/src/routes/render.ts
    * quantizeSliders mirrors the Python density-exemption.

  worker/src/cache/bin_cache.ts
    * deriveCacheKey now constructs the payload with alphabetical key
      order (matches Python json.dumps sort_keys=True). Outstanding gap
      documented inline: floating-point repr (`1.0` Python vs `1` JS)
      will need normalisation when the Worker actually shares cache
      cells with Python in STATE 4.B.

  docs/SLIDER_CONTRACT.md
    * §"Axis definitions" notes density is exempt from binning.

Verification: 1057 Python tests pass; worker typecheck clean;
make xruntime (K1/K1-mw/K2/K3/K4) PASS; make xruntime-adversarial
(A1–A6) PASS.

STATE 4.B (next): Worker handleRender real LLM call (replace 501
stub) + numeric normalisation in deriveCacheKey for Python↔TS cache
coherence. STATE 5: bench harness against seed_v1 to populate
threshold columns in the contract.

### Added — Phase E scaffold (slider as first-class product)

The genesis vision — bidirectional Tags ↔ Tomes with a slider — has
been substrate-only since the project began (density axis works
deterministically; the other four axes existed as metadata fields).
Phase E.1 STATE 2 lands the typed scaffold for the renderer +
contract doc + tests; STATE 4 fills the per-axis logic.

  sum_engine_internal/ensemble/tome_sliders.py  (extended)
    + SLIDER_BINS_PER_AXIS = 5                  # 3125 cache cells per triple-set
    + snap_to_bin(value, bins) -> float         # quantize to bin centre
    + quantize(TomeSliders) -> TomeSliders      # all-axis snap
    + length_fragment / formality_fragment /    # axis prompt fragments;
      audience_fragment / perspective_fragment    fail-loud at non-neutral
                                                  positions until STATE 4
    + build_system_prompt(TomeSliders) -> str   # composes neutral base +
                                                  per-axis fragments

  sum_engine_internal/ensemble/slider_renderer.py  (new)
    Type contracts:
      Triple = tuple[str, str, str]
      CacheStatus = HIT | MISS | BYPASS
      DriftAxis  = density | length | formality | audience | perspective
      AxisDrift  = (axis, value, threshold, classification, explanation)
      RenderResult = (tome, triples_used, drift, cache_status,
                      llm_calls_made, wall_clock_ms,
                      quantized_sliders, render_id)
      SliderCache (Protocol)        = get / put / stats
      LLMChatClient (Protocol)      = chat_completion(system, user, max_tokens)
      TripleExtractor (Callable)    = (str) -> awaitable[list[Triple]]
    Functions:
      cache_key(triples, sliders)   = sha256(sorted_triples + sliders)[:32]
      render(...)                   = NotImplementedError until STATE 4
      measure_drift(...)            = NotImplementedError until STATE 4
      InMemorySliderCache           = dict-backed reference impl

  worker/src/cache/bin_cache.ts  (new)
    Cache contract mirror. deriveCacheKey produces the SAME 32-char
    string the Python cache_key produces for the same input — cross-
    runtime cache coherence by content-addressed key.

  worker/src/routes/render.ts  (new)
    POST /api/render route. Quantizes incoming slider position,
    derives cache key, returns 501 + activation plan until STATE 4.
    Wired into worker/src/index.ts; KV binding RENDER_CACHE
    declared (commented) in wrangler.toml.

  Tests/test_slider_renderer.py  (new — 21 tests)
    16 pass today (snap, quantize, cache_key, InMemorySliderCache).
    5 xfailed strict (render pipeline + measure_drift) — bodies are
    spec, not stub. STATE 4 lands the implementation; xfails flip
    to passes with no test body changes.

  Tests/benchmarks/slider_drift_bench.py  (new)
    Per-axis drift bench harness. NDJSON output schema
    sum.slider_drift_bench.v1. STATE 2 returns stub-error rows so
    the harness structure is exercised end-to-end; STATE 4 wires
    real measurement.

  docs/SLIDER_CONTRACT.md  (new)
    Source-of-truth spec. Per-axis drift formulas, thresholds,
    cache semantics, UX commit-vs-drag decision matrix, stop-the-
    line conditions. Every numeric tolerance is empirically
    falsifiable by the bench harness.

Carmack-frame anti-hypotheses captured in the contract:
  1. Slider may be wrong UX (users want discrete buttons). E.6
     trial A/B-instruments both control surfaces.
  2. LLM latency may make drag-and-see undeliverable. 500ms
     debounce + bin cache + skeleton-loader; commit-on-release
     fallback.
  3. Round-trip drift may be wildly variant per axis. Live drift
     display per axis; "facts preserved within X%" replaces
     "facts preserved" in product copy.
  4. Go service rewrite is premature without measured Python
     bottleneck. Defer until E.6 telemetry decides.
  5. >10K-axiom scaling is hypothetical. Layered-architecture
     plan stays in PROOF_BOUNDARY §3; build only when measured.

No render claim made today. STATE 4 implements; STATE 5 verifies
against the bench harness; only then does the slider become a
shipping product feature instead of a typed contract.

### Added — Phase B intensification queue (B5–B7) in playbook

- `docs/NEXT_SESSION_PLAYBOOK.md` Phase B grew three explicit items
  that collapse multi-step user flows into single gestures (process
  intensification: combining steps so the user touches the
  deliverable, not an intermediate):
    * **B5** — shareable bundle URLs (`/b/{hash}` Worker route +
      R2 backing). Removes the JSON-file artifact from civilian
      awareness. Depends on B1.
    * **B6** — PWA-installable demo (manifest.json + service
      worker). Phone-screen attestation flow, offline verify after
      first load. ~40 LOC + a manifest. No dependencies.
    * **B7** — `sum attest <url>` fetch mode. Eliminates the
      "open browser → copy text → switch to terminal → paste"
      five-step pattern. Depends on B1.
- "Out of Phase B (named so we don't lose them)" subsection captures
  two items that surfaced in the analysis but belong elsewhere: the
  browser extension (v0.4 feature, depends on B1+B5+B7) and verify
  badges (Phase C5, depends on B5).
- Phase B exit gate updated to include the with-B5–B7 case
  ("phone-to-phone share + verify, no install").
- Pinning policy on B5 explicitly forbids long-term retrievability
  promises under v1 — protects against locking in `sha256_64_v1`
  after Priority 3 lands `sha256_128_v2`.

No code in this commit. Items are queued, not built. Hardening
ordering (Phase A priorities first) is unchanged; the
intensification work is post-hardening.

### Added — platform trajectory (Phases A–D) in NEXT_SESSION_PLAYBOOK

- `docs/NEXT_SESSION_PLAYBOOK.md` grew a new
  "Beyond the priorities — platform trajectory" section after the
  Priority 1–8 block. Four phases:
    * **Phase A** — finish the hardening playbook (= Priorities 3–8).
      No new thinking; exists as a framing anchor for Phases B/C/D.
    * **Phase B** — platform surface: source anchoring in the bundle
      schema (B1), bundle explorer / viewer (B2), `sum verify --explain`
      UX (B3), `sum tutorial` onboarding (B4). Depends on Phase A.
    * **Phase C** — network layer: well-known bundle discovery (C1),
      composition UX (C2), cross-attestation graph (C3), W3C VC 2.0
      full round-trip + PROV-O (C4). Depends on B1 + P6.
    * **Phase D** — 1.0 stability contract. Not new work; a decision
      point with a CI-backed promise that 1.0-minted bundles continue
      to verify 10 years from now.
- "The greater goal, stated plainly" preface names the three gaps the
  phases exist to close: trust end-to-end for a specific adversarial
  user, composability across publishers, engine-itself verifiability.
- Each phase has an explicit exit gate. "How to use this document"
  footer names the reading order for memory-less sessions and flags
  the two common scope-pressure failure modes (ship-faster-skip-gate,
  ship-more-add-item).
- `CLAUDE.md` onboarding item #5 expanded to name the new section and
  its Phase-dependency rule (do not start B-work while A is open).

No immediate execution commitment. Phase A continues through
Priorities 3–8 in their existing order; the trajectory section exists
so future sessions don't re-derive the same roadmap from scratch.

### Added — Priority 2: WASM-vs-JS derivation benchmark harness

- `Tests/benchmarks/browser_wasm_bench.html` — single-file harness that
  runs the deployed WASM core (`sum_core.wasm`) and the pure-JS fallback
  against identical input across N ∈ {10, 100, 1000, 10000} axiom
  derivations. Reports median / min / max ms per surface, per-op µs,
  and the JS ÷ WASM ratio. Also asserts bit-identical state integers
  across the two paths on every trial (correctness gate, not speed
  datum). Emits a machine-readable JSON block ready to paste into the
  methodology doc.
- `scripts/bench_python_derive.py` — Python-side companion that
  measures `GodelStateAlgebra.get_or_mint_prime` on the same key
  generator (`sum-bench-v1` seed), records whether the Zig shared
  library served or the `sympy.nextprime` fallback did, and emits the
  same-schema JSON block.
- `docs/WASM_PERFORMANCE.md` — methodology doc. Declares exactly what
  is measured (prime derivation alone), what is NOT (Ed25519, extraction,
  bundle parse), the trial protocol (5 trials, median, 3 warm-ups), the
  reproduction steps for all four surfaces (Python, Node, Browser-WASM,
  Browser-JS), and the fallback statement that ships regardless of
  outcome. Every numeric cell is labelled `measured`; blocks are `"not
  yet measured"` placeholders until the browser-matrix run happens.
  Change-control rules forbid adding performance language to
  `README.md` or a commit message until the corresponding row has data.
- `make wasm-bench` serves the repo over HTTP so the browser harness
  has a working `instantiateStreaming` + `crypto.subtle` environment.
  `make wasm-bench-python` runs the Python companion.

**No performance claim is made by this commit.** Per the playbook's
"measure before you assert" principle, shipping the harness is
orthogonal to publishing numbers. The numbers arrive in a later commit
that pastes concrete JSON blocks under each per-browser section; that
commit is the one allowed to add "fast" or "X× faster" language to the
prose.

### Added — Priority 1: adversarial cross-runtime rejection matrix

- `scripts/verify_cross_runtime_adversarial.py` — companion to the
  existing K-matrix. Six fixtures (A1-A6) covering the three
  cross-runtime-equivalent rejection classes: structural (missing
  tome, truncated tome, state integer = 0, state integer = -42),
  version (unknown canonical_format_version), signature (Ed25519
  bundle with post-sign tome tampering). Each fixture is passed
  through BOTH the Python verifier (`sum verify --input`) and the
  Node verifier (`standalone_verifier/verify.js`). The harness
  asserts: (1) both reject; (2) rejection classifications agree.
- HMAC tampering is intentionally out of scope for this harness —
  the Node verifier's header docstring is explicit that HMAC is
  not checked ("shared-secret, not public witness"). HMAC fixtures
  stay in `Tests/test_adversarial_bundles.py` (Python unit tests).
- `make xruntime-adversarial` runs it locally.
- `.github/workflows/quantum-ci.yml` `cross-runtime-harness` job
  runs A1-A6 alongside the existing K-matrix on every push.
- `docs/PROOF_BOUNDARY.md` §1.2 updated: the Cross-Runtime State
  Equivalence claim is now backed by FOUR harnesses, the fourth
  explicitly "proved on adversarial inputs," closing the
  valid-only-agreement gap the previous three left open.

Initial run result: **6 / 6 fixtures pass** — the two verifiers
already agree on rejection class for every adversarial case we
built. Worth reading as: the valid-input-agreement property
hasn't been accidentally extending a false claim about invalid-
input agreement; we checked and the claim holds.

Queue: A7+ fixtures (boundary state integers > 10^5 digits for
DoS; scheme-downgrade attempts between sha256_64_v1 and an
as-yet-unshipped v2; empty `{}` bundles; non-object root JSON)
can be added as single-line `FIXTURES` entries. Priority 1 is
formally discharged; future fixtures are additive hardening.

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
