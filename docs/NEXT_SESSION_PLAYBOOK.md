# Next-Session Playbook

*Re-authored 2026-04-27 at the close of Phase E doc-pass PR A (`README.md` + `CLAUDE.md` reflect v0.9.A.2). The previous draft was authored 2026-04-24, before the Phase E.1 v0.4 → v0.9.A.2 arc landed; that arc shipped in parallel with the priority queue below and unblocked new work that this revision now names. Read this before touching code in a new Claude Code session.*

## Non-negotiable principles

Before any work: read [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) cover-to-cover and internalise the proved / measured / designed distinction. Every claim emitted from this codebase — in docstrings, README, commit messages, blog posts, tweets — carries one of those three labels implicitly. Making the label explicit is discipline, not paranoia. A "proved" claim survives hostile review; a "measured" claim survives replication; a "designed" claim survives only in the author's head until it earns a label upgrade.

**Measure before you assert performance.** Never write "fast," "efficient," "low-latency," or "scalable" in a commit message or README without a benchmark in the same commit. If you don't have time to benchmark, you don't have time to make the claim.

**Adversarial testing is first-class work, not a nice-to-have.** A verifier that passes K1–K4 on well-formed bundles has been proved to agree on validity; it has NOT been proved to agree on invalidity. Those are two different proofs and only one is automatic. The same separation applies to render receipts: a verifier that accepts a valid signed receipt has not by that fact rejected every tampered receipt — write the negative path explicitly.

**Truthfulness over velocity.** If the honest answer to "does this work?" is "on the happy path, yes; under adversarial input, unknown" — write it that way. A repo that overclaims once loses the trust it took eight months to build.

**The receipt is a render attestation, not a truth oracle.** [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §5 is canonical: the signed payload binds tome / triples / sliders / model / provider / timestamp; it does not assert content accuracy, freshness, or issuer honesty. Any prose that conflates the two is a policy violation per [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §5.

---

## State at the head of this session

`main` head: PR #43 (PR A doc-pass) merged on top of v0.9.A.2 (PR #42).

| Surface | Status |
|---|---|
| `pip install 'sum-engine[sieve]'` — CLI v0.3.0 | shipped on PyPI |
| Cloudflare Worker at `sum-demo.ototao.workers.dev` | live; serves `/api/render`, `/.well-known/jwks.json`, `/api/qid`, `/api/complete` |
| Render receipts (`sum.render_receipt.v1`) | shipped v0.9.A; signing in `worker/src/receipt/sign.ts`; spec in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) |
| 5-axis slider renderer | density actioned deterministically; length / formality / audience / perspective LLM-conditioned via Worker |
| Slider product claim | **verified at scale** — median LLM-axis fact preservation 1.000, p10 0.769 (n=16) / 0.818 (n=8); v0.7 prompt hardening eliminated catastrophic outliers (2 → 0); see [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) |
| Cross-runtime trust triangle | locked by CI (K1–K4 valid + A1–A6 adversarial; `make xruntime` + `make xruntime-adversarial`) |

The doc-pass cycle completes the truthfulness re-anchor: README + CLAUDE.md (PR A), and now PROOF_BOUNDARY.md + FEATURE_CATALOG.md + this playbook (PR B). PR C (cross-link sweep across SLIDER_CONTRACT.md / MODULE_AUDIT.md / DID_SETUP.md / DEMO_RECORDING.md / THREAT_MODEL.md / CONTRIBUTING.md) is queued for the session that follows.

---

## Closed since the previous playbook revision

### ✅ Priority 1 — Cross-runtime adversarial rejection matrix

Landed as `scripts/verify_cross_runtime_adversarial.py` + `make xruntime-adversarial`. Six fixtures (A1–A6) cover the structural / version / signature rejection classes; both Python and Node verifiers must reject AND classify equivalently. **6/6 pass at HEAD.** CI runs A1–A6 alongside K1–K4 on every push (`.github/workflows/quantum-ci.yml::cross-runtime-harness`). [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.2 now cites four harnesses, the fourth explicitly "proved on adversarial inputs," closing the valid-only-agreement gap.

A7+ fixtures (boundary state integers, scheme-downgrade attempts, empty `{}` bundles, non-object root JSON) are additive — file under the existing `FIXTURES` array as discovered.

### ✅ Phase E.1 v0.4 → v0.9.A.2 — slider product + render-receipt loop

Out-of-band relative to the priority queue, but closes the central gap the previous playbook left implicit ("the slider exists as substrate; nobody has rendered through it"):

- **v0.4** — NLI audit verifies the slider's load-bearing fact-preservation claim. 0/45 LLM-axis cells showed real loss after audit; semantic median 1.000 / p10 0.818 / min 0.727 across 8 short docs.
- **v0.5** — Worker render path (`worker/src/routes/render.ts`) replaces the 501 stub. TS axis-prompt mirror (`worker/src/render/axis_prompts.ts`) byte-equivalent to the Python `tome_sliders.build_system_prompt`. Slider UI in `single_file_demo/index.html`.
- **v0.6** — Long-doc bench (n=16, 9–24 triples/doc, median 17). Median 1.000 held; p10 dropped 0.818 → 0.769. Two catastrophic outliers surfaced.
- **v0.7** — Prompt hardening (`FACT_PRESERVATION_REINFORCEMENT` clause when any non-density axis ≤ 0.3). Real losses on LLM axes 36 → 1; catastrophic outliers 2 → 0; min lifted 0.111 → 0.700.
- **v0.8** — Four-layer defence against `LengthFinishReasonError` (prompt cap, salvage, retry, terminal raise). 0/400 cells errored on the same long-doc bench.
- **v0.9.A** — Render receipts shipped. `worker/src/receipt/sign.ts` produces `sum.render_receipt.v1`: Ed25519 (RFC 8032) over JCS-canonical bytes (RFC 8785), wrapped as detached JWS (RFC 7515 §A.5). JWKS at `/.well-known/jwks.json`.
- **v0.9.A.1 / .A.2** — Triple-sort fix, route `/.well-known/*` through Worker, keygen helper polish.

Live at `https://sum-demo.ototao.workers.dev`. Verifiable end-to-end: `curl /.well-known/jwks.json` + `POST /api/render` + `jose.flattenedVerify(JCS(payload))`.

### 🟡 Priority 2 — WASM-vs-JS derivation benchmark — harness shipped, numbers pending

`Tests/benchmarks/browser_wasm_bench.html` + `scripts/bench_python_derive.py` + `docs/WASM_PERFORMANCE.md` are in. The methodology doc has placeholder result blocks for each surface (Python CLI, Node verifier, Chrome 113+, Firefox 129+, Safari 17+). Concrete JSON blocks under each section are still to be pasted from a real run. Until they are, no performance language ships in `README.md` — the change-control rule in the methodology doc enforces this.

`make wasm-bench` (HTTP server) + `make wasm-bench-python` are the entry points.

---

## Open priorities

The numbering keeps continuity with the previous playbook so `git log --grep '\[P3\]'` still works. P1 is closed; P2 is partially closed; P3–P8 are unchanged in scope but updated for current state.

### Priority 3 — Activate `sha256_128_v2`

**Problem.** `sha256_64_v1` is the default prime scheme. 64-bit primes have a birthday-paradox collision probability that becomes non-trivial at around 2³² axioms per branch. Below that, it's fine. A large-scale attestation user — say, someone attesting every fact in a Wikipedia dump — is above that. The v2 scheme is "designed" per [`docs/STAGE3_128BIT_DESIGN.md`](STAGE3_128BIT_DESIGN.md); the Node side exists in `standalone_verifier/math.js` (`derivePrimeV2`, `nextPrimeBPSW`); the Python side is not yet `CURRENT_SCHEME`.

**Work.**
- Land 128-bit prime derivation as a Python-side first-class scheme. Cross-runtime fixtures must prove Python / Node / Browser agree on the same 128-bit primes for the same axiom keys.
- Extend the cross-runtime harness: K1 currently tests `sha256_64_v1`; add K1-v2 for `sha256_128_v2`. All bundles in the corpus get a v2 counterpart.
- Keep `sha256_64_v1` as the default for 0.x (backward compat). Add `--prime-scheme=sha256_128_v2` to `sum attest`. Document the migration path in [`docs/COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md).
- CHANGELOG entry labels the transition: v1 stays "proved"; v2 becomes "proved" when the K-matrix passes.

**Success criterion.** K1-v2, K2-v2, K3-v2, K4-v2 all PASS. Two users can attest the same corpus under v2 independently and get the same state integer. Documentation warns that mixing schemes within a single branch is undefined.

**Proof-boundary outcome.** v2 moves from "designed" to "proved." Pre-empts a collision issue that grows worse with SUM's success.

### Priority 4 — SPARQL disambiguation for `/api/qid`

**Problem.** `/api/qid` hits `wbsearchentities` alone today (`worker/src/routes/qid.ts`). That gives ~80 % accuracy on common-noun / proper-name lookups — a 1-in-5 lie rate on a surface framed as "this QID is this entity." Not acceptable for an attestation surface.

**Work.**
- When `wbsearchentities` returns multiple candidates with close confidence, fire a SPARQL query against `query.wikidata.org` filtering by predicate domain. For a subject with predicate "was born in" → restrict candidates to those with `P19` (place of birth).
- If only one candidate passes the SPARQL filter, return it with `confidence: "sparql-disambiguated"`. If multiple pass, return all with `confidence: "ambiguous"`. Never pick one silently.
- Keep the `User-Agent: SUMDemoQIDResolver/X.Y (+github.com/OtotaO/SUM)` header per Wikidata operator-contact guidance. Rate-limit SPARQL.

**Success criterion.** A held-out benchmark set (50+ known-disambiguation cases curated from Wikipedia disambiguation-page lists) where the pre-change resolver scores ~80 % and the post-change resolver scores >95 %.

**Proof-boundary outcome.** `/api/qid` moves from "best-effort entity resolution" to "entity resolution with measured accuracy floor." `confidence` becomes meaningful rather than heuristic.

### Priority 5 — Threat-model validation

**Problem.** [`docs/THREAT_MODEL.md`](THREAT_MODEL.md) exists. Every documented threat is a claim of the form "SUM defends against X." A documented-but-untested defence is designed, not proved.

**Work.**
- Read `docs/THREAT_MODEL.md` end-to-end. For each documented threat, write a test that embodies the threat and confirms the defence. File under `Tests/adversarial/test_threat_<n>.py`.
- Threats that turn out to be undefended become either (a) new work items or (b) [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) entries labelled "known limitation."
- Threats not yet listed but should be — prompt injection in extracted axioms, replay across DIDs, partial-bundle truncation, clock-skew signature validation, **render-receipt freshness replay**, **JWKS rotation grace-window violations** — add, test, document.

**Success criterion.** Every threat in the document has a corresponding test. Every test under `Tests/adversarial/` maps to a threat in the document. A CI check keeps the two lists in sync.

**Proof-boundary outcome.** The threat model moves from prose to prose + executable assertions.

### Priority 6 — Delta-bundle semantics

**Problem.** `bundle.is_delta` exists in the schema (`sum_cli/main.py`) and the verifier checks for it (`standalone_verifier/verify.js`). What a verifier is supposed to DO with a delta bundle is under-specified. Under-specified means different implementations will diverge.

**Work.**
- Write `docs/DELTA_SEMANTICS.md`: what a delta references (prior state? prior CID? prior signature?), how composition works mathematically (LCM with an ordering constraint?), when a delta is rejected (missing prior, scheme mismatch, non-monotone timestamp).
- Add K5 to the cross-runtime harness: delta-bundle composition. All three verifiers compose the same (full, delta) pair to the same `combined_state_integer`.
- Update the schema's `isDelta` description to point at the spec doc.

**Success criterion.** A composition test produces `combined_state_integer`, and all three verifiers compute the same value from the same inputs. Doc names every failure case.

**Proof-boundary outcome.** Delta bundles move from "supported in schema, semantics undefined" to "proved composable cross-runtime." Unblocks Phase C network-layer work.

### Priority 7 — Supply-chain attestation

**Problem.** The PyPI wheel is OIDC-published — that's good. The Cloudflare Worker is deployed via `wrangler deploy`; the deployed JS is not independently attested. The single-file demo is `single_file_demo/index.html` committed to the repo and served via raw GitHub URLs with no signature on the served bytes. The Worker also serves render receipts that bind the *render* but not the *Worker code* that produced them.

**Work.**
- Sign release artifacts with Sigstore / cosign. Wheel + single-file demo + standalone verifier's Node entrypoint + the deployed Worker bundle all get signed. Public key published in the repo and on the project site.
- Consider a SLSA provenance attestation on every release tag in GitHub Actions.
- Document the trust chain in `docs/SUPPLY_CHAIN.md`: "here's what we sign, here's how to verify, here's what we DON'T sign and why."

**Success criterion.** `pip install sum-engine` → verify signature → run. `node standalone_verifier/verify.js` → verify signature on `verify.js` itself. Worker bundle hash recoverable from a signed release manifest. All three runtimes have a verifiable bytes-to-identity path.

**Proof-boundary outcome.** Trust model moves from "trust the GitHub repo" to "verify the bytes."

### Priority 8 — LLM-extraction honesty guardrails

**Problem.** `sum attest --extractor=llm` lets a user mint axioms via an LLM. LLMs hallucinate. The resulting bundle says "here are the axioms, attested and signed" — but "attested and signed" applies to the computation (the state integer is correctly derived from these keys), not to the axioms' truth. A reader might reasonably infer that an Ed25519-signed bundle means "these facts are true." That's not what it means.

**Work.**
- Add `extraction.verifiable: true | false` to the bundle schema. `true` for sieve, `false` for LLM. Visible at the consumer's interface.
- In `sum verify`, when verifying an LLM-extracted bundle, emit a prominent warning: "axioms were extracted by an LLM and are not independently verified."
- In the schema emitted by `sum schema bundle`, add doc text that signed ≠ true.
- Consider provenance spans (byte-level source-text references) as light-touch verification: if the LLM said "Alice likes cats" and the source span literally says "Alice likes cats," the extraction is corroborated; if not, flag it.

**Success criterion.** A consumer reading a signed bundle cannot confuse "signed by whoever held the key" with "facts are true." Bundle epistemic status is a first-class, machine-readable field.

**Proof-boundary outcome.** The truth claim SUM has always made — "we bind the set of axioms, not their truth" — becomes visible to downstream consumers rather than buried in docs.

---

## Phase E.1 follow-ups (unblocked by v0.9.A)

These are concrete next-cycle items the render-receipt landing made possible. They are *post-hardening* in the sense that they extend the v0.9.A surface; they are *pre-Phase B* in the sense that they close the trust loop the README's load-bearing claim depends on.

### Phase E.1 v0.9.B — browser receipt verifier

**What.** In-page detached-JWS verification against the same JWKS the Worker publishes. The single-file demo gains a "Verify receipt" pane that:
1. Fetches `/.well-known/jwks.json` (cached per `Cache-Control` header).
2. Takes a `render_receipt` (paste, drag-drop, or auto-loaded after a `/api/render` call).
3. Runs the six-step verifier algorithm from [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §2.1 client-side.
4. Renders pass/fail with the protected header inspected.

**Library.** `jose` (panva, MIT) ships ESM browser builds. `canonicalize` (Erdtman, Apache 2.0) is single-file dependency-free.

**Success criterion.** Same receipt that verifies under Node `jose.flattenedVerify` verifies in Chrome 113+ / Firefox 129+ / Safari 17+ with byte-identical canonical bytes. Tampered-receipt fails closed with the expected error code.

### Phase E.1 v0.9.C — Python receipt verifier

**What.** `sum_engine_internal/`-side verifier composing `joserfc` (or `authlib`) + the existing pure-Python JCS at `sum_engine_internal/infrastructure/jcs.py`. Wired into the bench harness so a bench run that hits the live Worker can independently verify each receipt rather than trusting the Worker's word.

**Success criterion.** 10/10 fixtures byte-identical between TS `canonicalize@3` and Python `jcs` on the receipt's payload shape (this was Probe 3 of v0.9.A research; reconfirm with the production payload). Tampered-payload fails verify; unknown-kid surfaces a clean named error.

### Phase E.1 v0.9.D — receipt-aware bench

Compose v0.9.C into the long-doc bench. Each cell that hits `/api/render` records the receipt and verifies it; a verified receipt with `model: canonical-deterministic-v0` is the canonical-path attestation, with `model: claude-haiku-…` is the LLM-path attestation. Aggregate column: % of receipts verified, % of cells where canonical-path served (sliders all neutral except density).

Unblocks: a bench run becomes its own audit trail. Replay a historical bench run's receipts to confirm the engine produced the claimed renders without re-paying the LLM cost.

---

## Ordering rationale

P1 closed the cross-runtime invalidity-agreement gap. P2 stays open on numbers; the harness landing was the larger half. P3 prevents a collision frontier that worsens with success. P4 fixes a 1-in-5 lie rate on a live attestation surface. P5 turns prose threat claims into executable assertions. P6 specifies an under-specified field. P7 raises trust from "trust the repo" to "verify the bytes." P8 makes the signed-vs-true distinction visible at the consumer interface.

The Phase E.1 v0.9.B / v0.9.C / v0.9.D items are sized for one short cycle each and are independently unblocked — pick any order.

Phase A → B → C → D below applies on top of these. **Phase B and C work depends on Phase A priorities being closed first** — a platform built on an unstable foundation fails in a way that blames the platform, not the foundation.

---

## Stop-the-line triggers

If you discover any of these during the work above, pause and surface it before continuing:

- **A cross-runtime disagreement on a valid bundle.** Breaks K1–K4 and invalidates the central public claim. Fix before anything else.
- **A verifier that accepts a bundle the spec says it must reject.** Same severity.
- **A render receipt that verifies despite a tampered signed field.** Breaks the canonicalisation contract; almost always a JCS divergence (the integer-vs-float-zero edge case from [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §4). Bundle a regression fixture.
- **A render receipt with `kid` not present in JWKS while cached responses can still surface that kid.** Rotation grace-window violation per [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6.
- **A claim in `README.md` / `CHANGELOG.md` / docs that you cannot defend** with a test, a measurement, or an explicit "designed, not proved" label. Most common lie to discover; corrodes trust.

---

## Beyond the priorities — platform trajectory

Priorities 1–8 above are the **hardening playbook**: they close known truth gaps in the current surface. Everything below is **post-hardening** and depends on the surface being solid. Don't run Phase B or C work while Phase A priorities are still open.

### The greater goal, stated plainly

SUM today is a cryptographically-attested knowledge-bundle engine **plus** a slider-driven verifiable-render surface, with a proved cross-runtime verifier triangle and per-render attestation. "Extremely useful platform" means three things it is *not yet*:

1. **Trustworthy end-to-end for a specific user in a specific adversarial context** (journalist verifying a claim, LLM operator auditing extraction, researcher replicating a cited result).
2. **Composable across publishers**, so bundles + receipts flow and aggregate rather than sit in isolation.
3. **Self-describing enough that a motivated skeptic can verify the engine itself**, not just the bundles or receipts it produces.

Every phase below is an answer to one of those three gaps.

### Phase A — finish the hardening playbook

| Priority | Effect |
|---|---|
| P2 numbers | Replaces an implicit performance expectation with measured browser-matrix data. |
| P3 `sha256_128_v2` | Moves prime scheme from "safe for <2³² axioms" to "safe at knowledge-graph scale." |
| P4 SPARQL disambig | Moves `/api/qid` from 80 % correct to a measured >95 % floor. |
| P5 threat-model validation | Every documented defence gets an executable test. |
| P6 delta semantics | Specifies what composition means. **Precondition for every Phase C item.** |
| P7 supply-chain attestation | Consumers verify the bytes, not just trust the repo. |
| P8 LLM honesty guardrails | Signed ≠ true becomes visible at the interface the consumer reads. |
| Phase E.1 v0.9.B / C / D | Closes the receipt verification loop in browser, Python, and bench. |

**Gate to exit Phase A.** Every priority's success criterion passes. CI matrix has K1–K5 + A1–A6 + threat-suite + supply-chain verify + LLM-guardrail assertions + receipt-verifier fixtures all green on every push. WASM perf numbers are pasted under each per-browser section in `docs/WASM_PERFORMANCE.md`.

### Phase B — platform surface

These items only make sense *after* A. Shipping earlier adds surface to a foundation still settling.

**B1 — source anchoring in the bundle schema.** Every axiom optionally carries `{source_uri, content_hash, byte_range, retrieved_at}`. Backward-compat extension: `provenance` becomes required when `extractor == "llm"`, optional when `"sieve"`. Builds on the existing `AkashicLedger` byte-range machinery.

**B2 — bundle explorer / viewer.** `single_file_demo/index.html` extended from "paste → attest" to "open → read → verify → trace provenance." Drop-in file, paste-in URL, or QR-scan entry points.

**B3 — `sum verify --explain`.** When verification fails, the CLI tells the user exactly why in one sentence they can act on. Same input, new output format; underlying logic unchanged.

**B4 — `sum tutorial`.** `pip install sum-engine[sieve]` → `echo "…" | sum attest | sum verify` must produce a visibly interesting result in under 60 seconds with zero ambiguous steps. Five numbered prompts walk a new user through attest → sign → verify → view-in-browser.

**B5 — shareable bundle URLs (`/b/{hash}`).** Worker route `PUT /b/` accepts a bundle, content-addresses it (sha256 over JCS-canonical bytes), stores in R2, returns a short URL. `GET /b/{hash}` renders a verified HTML view. Idempotent on identical bundles. Pinning policy explicit: do NOT promise indefinite retrievability under v1 prime scheme — that locks v1 in even after `sha256_128_v2` lands. Spec a soft-expiry of "best-effort 12 months" until 1.0 contract (Phase D) freezes the pinning rules. **Foundation.** B1.

**B6 — PWA-installable demo.** `manifest.json` + service worker caching the single-file demo's static assets. ~40 LOC + a manifest. **Foundation.** None.

**B7 — `sum attest <url>`.** Positional URL argument fetches with `httpx`, sets `source_uri` + `retrieved_at`, records HTTP status + final URL after redirects. **Foundation.** B1.

#### Out of Phase B (named so we don't lose them)

- **Browser extension: "Attest this selection."** Right-click → mints + signs a bundle with current URL + byte range as provenance → uploads to `/b/` → copies the shareable URL. **Depends on B1 + B5 + B7.** Ship as v0.4 in its own session.
- **Verify badges (Shields.io-shaped SVG).** `GET /badge/{hash}.svg` returns a live-verifying badge (✓ verified / ✗ invalidated / ⚠ unreachable). Publishers embed in READMEs, blog posts, papers. **Depends on B5.** Add to Phase C as `C5` once B5 lands.

**Gate to exit Phase B.** A fresh user, given only the README, can produce a verified signed bundle with source-anchored axioms and open it in the explorer without asking a question. With B5–B7 landed: that user can also produce + share a bundle by URL from their phone, and a recipient can verify the URL from their phone without installing anything.

### Phase C — network layer

SUM stops being a tool and starts being a platform. Conditional on **B1** and **P6** both landing.

**C1 — bundle registry / discovery.** Open index any publisher can push to, any consumer can pull from. Simplest shape: a well-known URL pattern (`https://{domain}/.well-known/sum/bundles.json`) enumerating recent bundles with CIDs + signer DIDs. No centralised server, no network-effects mandate.

**C2 — bundle composition UX.** B2's explorer learns to open a second bundle as a delta over the first and show the composed state. Fact-checker flow: open journalist's bundle → add own attestations over a subset → export delta → publish at own well-known URL.

**C3 — cross-attestation graph.** Given a root bundle and its deltas, produce a traversable graph showing who attested what and when. Primary artefact for fact-checking at scale.

**C4 — standards interop.** Full W3C VC 2.0 emit/verify path (some scaffolding exists). PROV-O export. Optional JSON-LD context. A bundle survives as a W3C credential in W3C-credential tooling and round-trips back unchanged.

**C5 — verify badges.** Promoted from Phase B's deferred list once B5 lands.

**Gate to exit Phase C.** Two independent publishers can produce bundles that compose into a third-party aggregate verifier's state, and the aggregate verifier agrees with both publishers' local verifiers. Disagreement surfaces as a named diagnostic, not silent drift.

### Phase D — 1.0 stability contract

Not a phase of new work so much as a decision point. Before tagging `v1.0.0`: an explicit compatibility contract. What bundles + receipts shipping 1.0 will continue to verify 10 years from now. What schema fields are frozen. What CLI flags are stable. What wire-format guarantees hold across minor versions. What requires a major bump.

Backed by a CI gate that tests the promise: a corpus of 1.0-minted bundles + receipts that every later release must continue to verify. The hand-off point between "interesting prototype" and "dependency anyone can build on."

---

## How to use this document

- **Reading order** for a memory-less session: closed-priorities + Phase E.1 retrospective first (so you know what's already done), then open priorities, then platform trajectory. Don't skip ahead to Phase B work if Phase A priorities are still open.
- **Phase tags in commits** are optional but useful: prefix `[P3] …` or `[B1] …` so `git log --grep` finds all work on a given item.
- **When you discover a dependency** between two items not captured here, add the edge as an explicit sentence in the item text. Future learning refines the ordering.
- **Scope pressure** comes from two directions — "ship faster, skip the gate" and "ship more, add an item." Both are wrong. The priorities close debts that already exist; the phases add surface that depends on those debts being closed. Shipping anything new before its gates pass moves the debt, not the value.

## On the discipline itself

The single best habit to carry into the next session is: when you're about to write a claim, ask "what would adversarial review do with this sentence?" If the answer is "catch me," rewrite it before you commit.

SUM's whole thesis is that content-addressed, cryptographically-attested, cross-runtime-verified bundles + signed render receipts are a truth-telling surface because they cannot lie without being caught. The repo itself should hold to the same standard the bundles and receipts do.
