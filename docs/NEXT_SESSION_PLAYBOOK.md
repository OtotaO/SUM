# Next-Session Playbook

*Re-authored 2026-04-27 at the close of Phase E doc-pass PR A. **State-at-head section refreshed 2026-04-30** to reflect PRs #83–#95 (chunked algebra, omni-format pivot, self-attestation, frontier-LLM cross-vendor closure, sieve-quality rounds 1+2, MinHash dedup, cold-install fix). For a focused handover to the immediately-next session see [`docs/SESSION_HANDOVER_2026-04-30.md`](SESSION_HANDOVER_2026-04-30.md) — read that first if you are picking the thread up cold from this session. This playbook below is long-term priority context.*

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

## External-awareness checkpoint (2026-04-29)

A focused audit of relevant external developments since SUM's
current substrate decisions. Each item is sized for one cycle
and orthogonal to the others.

### 🔎 §2.5 LLM-refresh measurement (high leverage, ~$1–3 budget)

The §2.5 round-trip closure is locked at recall ≥ 0.97 across
three corpora using `gpt-4o-mini-2024-07-18` (Jul 2024). Two
frontier LLMs have shipped since then — **Anthropic Claude Opus
4.7** (16 Apr 2026) and **OpenAI GPT-5.5** (23 Apr 2026). The
intervention pattern (canonical-first generator + constrained-
decoding extractor + lemma-exclusion) is *probably* model-
independent; the hypothesis remains unmeasured on frontier models.

**Action:** add Anthropic SDK support to
`scripts/bench/runners/s25_generator_side.py` (it currently calls
OpenAI only); re-run combined ablation on `seed_v1` + `seed_long`
under both new models; ship a `sum.s25_frontier_models_2026.v1`
receipt. Budget ~$1–3 of API spend (frontier model token rates are
~3–5× gpt-4o-mini).

### 🔎 Sigstore-signed PyPI uploads (medium leverage, no budget)

The `sigstore` PyPI package is now Production/Stable; PyPI accepts
Sigstore in-toto attestations; cosign v3 shipped; no 2026
incidents. The "Sigstore signing of release artifacts"
hardening-backlog item moves from "wait for maturity" to
"ship it."

**Action:** add a `sigstore sign` step to `.github/workflows/
publish-pypi.yml` gated on the GitHub OIDC token; emit a Sigstore
in-toto attestation alongside the PyPI provenance. Closes
threat-model row "supply-chain compromise" with a stronger
guarantee than "trust the GitHub Trusted Publisher."

### 🔎 MCP discovery shim (low leverage, no budget)

MCP next spec drop is **June 2026**; SEP-1649
(`.well-known/mcp/server-card.json`) is broadly adopted but not
yet merged. SUM's `sum-mcp` is stdio-only today; an HTTP-MCP
variant remains deliberately deferred (auth design hasn't
landed). Worth adding the `.well-known/mcp/server-card.json`
shim to the Worker so when an HTTP-MCP variant ships the
discovery URL is conventional.

**Action:** small Worker route under `worker/src/routes/`
returning the canonical server-card JSON. Track
SEP-1649 / SEP-1960 progress through June 2026.

### Other external-audit findings (no action needed today)

* **C2PA `digital_source_type`** — taxonomy values unchanged
  through C2PA 2.4. SUM's `trainedAlgorithmicMedia` and
  `algorithmicMedia` mappings remain authoritative; documented as
  deliberate text-on-image-taxonomy mapping in
  `docs/RENDER_RECEIPT_FORMAT.md` §7.
* **PQC / Ed25519 / SHA-256 framing** — NIST SP 800-131A r3
  keeps SHA-256 fully approved through 2030; the 2030 deprecation
  target is RSA/ECDSA, not Ed25519 explicitly. No code change
  today; tracking note: Ed25519 will eventually need a hybrid PQC
  partner (likely ML-DSA) ahead of 2030.
* **W3C VC 2.0 / Data Integrity 1.1** — `eddsa-jcs-2022`
  interop tests re-ran 22 Feb 2026 and pass. Data Integrity 1.1's
  Render Method targets REC in Sept 2026; once it lands, evaluate
  emitting it alongside the existing detached-JWS receipt.

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

### v0.3.1 — PyPI long-description sync + CI gate (micro-cycle)

**Problem.** The `sum-engine` v0.3.0 wheel was published before PR A's README rewrite, so `pypi.org/project/sum-engine` shows a long-description that says "PyPI: `pip install sum-engine[sieve]` — shipping soon" — a tautology against itself, and a drift the truthfulness contract specifically prohibits. PyPA's metadata model freezes the long-description at publish time; the surface rots independently of the GitHub README.

**Work.**
- Cut a v0.3.1 hygiene release that picks up the post-PR-A README. No code changes; `pyproject.toml` version bump only.
- Add a CI gate that fails if the next wheel's `Description` metadata diverges from `README.md` head: `python -m build` → extract `Description` from the built wheel's `*.dist-info/METADATA` → normalise line endings → diff against `README.md`. Reuse `twine check` to validate the long-description renders on Warehouse, and `check-wheel-contents` to catch wheel-content mistakes. The custom diff gate answers a question those two don't ("is this actually the README we intended to ship") and complements them.
- **PEP 740 attestation (staged publish + verify).** Publish via Trusted Publishing with `pypa/gh-action-pypi-publish` — the action generates and uploads PEP 740 attestations automatically by default, binding each wheel/sdist to a cryptographic digest and a GitHub Actions workflow identity. The verification flow has to be **pre-promotion**, not post-production-publish: a verifier check that runs after production PyPI publish can only *detect* a misconfigured provenance, it cannot *prevent* the artifact going public. The staging pattern that gives a real fail-closed gate is: build → publish to TestPyPI via Trusted Publishing → fetch the staged attestation, verify it points at the expected `OtotaO/SUM` repo + workflow identity using `pypi-attestations` (or the PyPI Integrity API) → only on success, promote the same artifact to production PyPI. Add a post-publish verify step on production too — that one is post-publish *detection*, documented as such; it catches the case where TestPyPI and production diverge unexpectedly, but the load-bearing gate is the staging step.
- Run sequence in CI:
  ```
  python -m build --sdist --wheel                                  # local build
  twine check dist/*                                               # pre-publish gate
  check-wheel-contents dist/*.whl                                  # pre-publish gate
  python scripts/check_long_description_sync.py                    # pre-publish gate
  publish to TestPyPI via pypa/gh-action-pypi-publish              # staging
  python scripts/verify_pypi_attestation.py --index test \
                                            sum-engine==0.3.1      # pre-promotion gate (FAIL CLOSED)
  promote same dist/* to production PyPI via pypa/gh-action-pypi-publish
  python scripts/verify_pypi_attestation.py --index prod \
                                            sum-engine==0.3.1      # post-publish detection
  ```

**Success criterion.** v0.3.1 ships with (a) a long-description that matches `README.md` head verbatim, AND (b) a verifiable PyPI provenance attestation on the staged TestPyPI artifact pointing at the expected repo+workflow identity, gating production promotion. The pre-publish gates (twine check / wheel contents / README diff) and the pre-promotion gate (TestPyPI provenance verify) all fail closed before any production-PyPI bytes ship. Post-publish detection on production catches TestPyPI/production divergence as an alarm, not as a gate. A deliberate test break in a branch confirms each pre-publish/pre-promotion gate fires before its respective publish step.

**Why before v0.9.A.3.** It's a 1–3 hour cycle, prevents an immediate truthfulness-contract violation, and is independently scoped from any other work. The PEP 740 piece is partial-credit on Priority 7 (supply-chain attestation) for the wheel surface specifically — full P7 still covers the standalone verifier bundle and the Worker bundle, which are out of v0.3.1 scope.

### Pre-A3 truthfulness hygiene — Demo UI provenance/preservation label (micro-cycle)

**Naming note.** This sits *before* v0.9.A.3 (the CORS fix) in the work queue, so it is **not** numbered v0.9.A.4 — that label would imply it comes after A.3, which inverts the ordering. Treat as a pre-A3 hygiene cycle that lands while v0.9.A.3 is being scoped.

**Problem.** The Worker render route doesn't recompute fact preservation per call — preservation is the bench's contract, not a live property of any single render ([`worker/src/routes/render.ts`](../worker/src/routes/render.ts), [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md)). A casual user reading the README's "median 1.000 / p10 0.769 measured at scale" claim and then watching a live render could conflate "this receipt verified" with "this render's fact preservation was independently checked." The receipt format spec covers the trust scope in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §5; the live demo HTML doesn't currently surface that distinction. Independent observation: the live demo as of 2026-04-27 displays `p10 0.818` (short-corpus number) without the more conservative `p10 0.769` (long-corpus) the README foregrounds — the demo number is itself drifting from the README's load-bearing claim.

**Work.**
- Add a label in `single_file_demo/index.html` next to the rendered tome + receipt. Three lines, jargon kept to what the spec uses, every claim backed by a doc anchor:
  > **Provenance verified:** the receipt proves the issuer signed this render tuple.
  > **Preservation benchmarked:** median 1.000; p10 0.769 long / 0.818 short. Not recomputed for this render.
  > **Signed does not mean true:** the receipt is not a truth oracle.
- Normalize the demo's preservation copy to the README's long/short distinction so the surface reads consistently across `/api/render` consumers and the README's headline claim.
- Confirmation: any user reading the live demo can answer "what does this receipt prove?" without consulting docs, and the answer matches §5 of the receipt spec.

**Why before v0.9.B.** v0.9.B's verifier UI will inherit this labelling pattern; doing it now means v0.9.B doesn't have to retrofit. Lightweight (~1–2 hours) Worker-side change.

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

### Phase E.1 v0.9.E — local ONNX NLI rescue

**Problem.** The NLI audit is load-bearing: long-doc bench reports 99.8 % rescue rate (653 / 654 audited cells), and the slider's product claim runs through this layer. The current path calls OpenAI per audit, which makes NLI both the largest LLM-cost line in the bench and a coupling point that ties the slider's headline number to a third-party API's availability + pricing. Decoupling is high-leverage.

**Work.**
- Export a DeBERTa-family MNLI/FEVER/ANLI model to ONNX (candidate class; pin the exact model slug only after benchmark validation against adjudicated labels — see calibration artifact below). ONNX Runtime is cross-platform and gives a Python path now and a Node / browser path later if v0.9.B's verifier wants in-page audit; ONNX Runtime is the unified API behind Node.js, Web, and React Native bindings.
- Wire as a swap target behind `LiveLLMAdapter.check_entailment` — same `EntailmentResponse` Pydantic shape, new local backend. Keep the OpenAI path as a fallback; the substrate stays plural until the local path proves out.
- Calibrate the audit threshold against the existing benchmark corpus: current `--audit-threshold` is 0.7 with `text-embedding-3-small` at τ=0.85. The local model's entailment confidence distribution is different; threshold needs re-tuning against the same corpus to preserve rescue precision/recall.

**Calibration artifact (`fixtures/nli_audit_calibration_v1.jsonl`).** Author once; both the OpenAI path and the local path verify against it. Each row:

```json
{
  "premise": "...rendered tome...",
  "hypothesis": "...source fact as sentence...",
  "source_fact_key": "subject||predicate||object",
  "embedding_score": 0.82,
  "openai_verdict": "entails",
  "local_model_verdict": "entails",
  "adjudicated_label": "entails",
  "notes": "paraphrase; fact preserved"
}
```

The **adjudicated label is ground truth** — human review on the cases where embedding flagged loss, with a documented rubric for "entails" vs "neutral" vs "contradicts." OpenAI is a comparison backend, *not* the oracle. If OpenAI and the local model both ship their verdicts faithfully but diverge from the adjudicated label, that's a finding (one or both backends are biased on this case class), not a model defect.

**Success criterion.** Local rescue path matches the **adjudicated labels** within tolerance (e.g., ≥ 99 % agreement on the calibration set), AND OpenAI/local disagreements are triaged (re-adjudicated, root-caused, fixture amended) rather than silently inherited from either backend. Headline claim in [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) gains a "verified locally + via OpenAI, both against adjudicated labels" footnote rather than swapping the canonical path immediately.

**Tradeoff.** Operational complexity goes up: model-file packaging, ONNX runtime version pinning, cold-start behaviour, threshold drift across model upgrades. Worth pinning a model-version contract (`docs/NLI_MODEL_REGISTRY.md` or similar — small, just a kid-shaped pin so rescues across runs are comparable).

---

## Governance & continuity tracks (parallel with Phase A)

These three tracks address institutional-durability gaps the Phase A → D trajectory undercounts. They are sized to run **in parallel with Phase A** — none are blocked by P3–P8 — and each costs roughly one short cycle. Run them sooner rather than later: cheap institutional hedges compound, and the bus-factor failure modes they defend against are not the kind that surface in CI.

The technical trust contract (cryptographic attestation + cross-runtime equivalence + bench-verified slider claim) is strong. The institutional trust contract is currently one signing key held by one author plus one repo on one platform. A long-horizon claim ("verifies in 10 years") is undefended against the actor who isn't an attacker — the actor who is just *time*. These tracks are the institutional counterpart to the cryptographic core.

### G1 — Continuity & archival

**Problem.** The repo, the spec mirrors, and the test vectors all live on GitHub today. GitHub is durable but not eternal. A future where this org is dormant, abandoned, or compromised is not adversarial — it is the default outcome for most repos at decade scale.

**Work.**
- Mirror [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md), [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md), [`docs/CANONICAL_ABI_SPEC.md`](CANONICAL_ABI_SPEC.md), `CHANGELOG.md`, and the cross-runtime fixtures (`scripts/verify_*.py` reference vectors + the v0.9.B/C fixture set landed by Phase E.1 follow-ups) to at least one third-party archival surface: Software Heritage (free, designed for source-code archival, citation-stable), Zenodo (DOI, decade-plus commitment from CERN), IPFS via a pinning service, or all three.
- Document the archival policy in `docs/ARCHIVAL.md`: what is mirrored, where, by whom, on what cadence, what cryptographic integrity is preserved on the mirror side, and what a consumer does when GitHub disappears.
- Document a succession playbook in `docs/SUCCESSION.md`: how the project's signing keys, control of the `sum-engine` PyPI package, control of the `sum-demo.ototao.workers.dev` deployment, and editorial control of this repo transition if the current author is unavailable. "Fork-and-key-rotate-forward" is an acceptable answer; *no* documented answer is not.

**Success criterion.** A third-party reader can recover the spec + reference fixtures + a working verifier from non-GitHub-hosted sources in under 10 minutes. Succession plan exists, has an explicit successor (named or named-by-condition), and lists the cryptographic + infrastructure handoffs required.

**Blast radius if skipped.** When the bus factor hits, every signed receipt + every signed bundle becomes an unverifiable artifact within ~6 months as JWKS endpoints rot, package indexes garbage-collect abandoned releases, and the spec drifts from cited memory.

### G2 — Independent verifier requirement

**Problem.** Python `sum verify`, Node `standalone_verifier/verify.js`, and the in-browser inlined JS all share authorship. Cross-runtime byte-identity is locked by CI, but a bug in the author's understanding of the spec replicates across all three implementations identically — the K-matrix would not catch it. The cross-runtime trust triangle is byte-equivalent but not author-independent; that is a weaker durability claim than the README implies.

**Work.**
- Source one verifier implementation maintained by an author/team distinct from the engine. Either a third-party-contributed verifier (preferred) or a clean-room re-implementation by an outside author working from [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) + [`docs/CANONICAL_ABI_SPEC.md`](CANONICAL_ABI_SPEC.md) alone, with no read access to the in-tree source.
- Add the independent verifier to the cross-runtime CI matrix as **K6** for CanonicalBundle and as the receipt-verifier counterpart against the v0.9.B/C fixture set. Drift between the independent verifier and the in-tree verifiers is the failure case the K-matrix is currently blind to.
- Document the spec-only path in `docs/IMPLEMENTERS_GUIDE.md`: how an outside author implements a verifier from the docs without consulting the in-tree source. This doc is the deliverable that proves the spec is implementable from text alone — the strongest test of the spec's completeness.

**Success criterion.** A receipt minted by the Worker verifies in (a) Python `sum verify`, (b) Node `verify.js`, (c) browser SubtleCrypto, AND (d) an independently-authored verifier — same bytes, same outcome on positive AND negative paths. K6 green in CI on every push.

**Blast radius if skipped.** A spec ambiguity that the engine author handled identically in all three in-tree implementations passes CI and ships forever. The "cross-runtime trust triangle" claim becomes one engineer's interpretation of the spec, replicated three times, not three independent witnesses.

### G3 — Revocation & crypto-agility policy

**Problem.** [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6 specifies key *rotation* but not key *revocation*. A compromised kid issued legitimate-looking receipts during the compromise window; without a revocation surface, those receipts continue to verify forever even after the compromise is detected. Separately, Ed25519 is not quantum-safe; "1.0 bundles verify forever" (Phase D's load-bearing promise) has a 20–30-year asterisk that has not yet been written down.

**Work.**

*Revocation:*
- Specify a revocation surface: `/.well-known/revoked-kids.json` listing kids that MUST NOT be trusted past their `effective_revocation_at` timestamp, with a `reason` field for audit-trail honesty (`compromise`, `superseded`, `policy`).
- Define verifier behaviour: a verifier that fetches a revocation list and observes a receipt's kid in it MUST reject the receipt with a `revoked-kid` classification distinct from `signature-invalid`. Verifiers SHOULD fetch the revocation list at the same cadence they fetch JWKS (default 1 h `Cache-Control`).
- Specify scope explicitly: revocation invalidates *future* trust. A receipt that VERIFIED before revocation can be archived as "verified-as-of timestamp X"; revocation does not retroactively unverify the past, it tells future verifiers not to trust this kid going forward. This distinction matters for archival use cases that pin verification timestamps alongside receipts.

*Crypto-agility:*
- Specify a dual-sign migration pattern: when migrating from Ed25519 to a successor algorithm (post-quantum signature, larger curve, etc.), receipts emit BOTH signatures during a deprecation window. Verifiers MUST verify at least one and SHOULD verify both; receipts whose only signature is in an unsupported algorithm fail closed with a `unsupported-alg` classification.
- Define the deprecation-window length policy: ≥ render cache TTL, ≥ documented archival validity window, ≥ ecosystem migration time (give consumers a year, not a month).
- Maintain an in-tree algorithm registry (`docs/ALGORITHM_REGISTRY.md`): the algorithms a verifier is allowed to accept. An algorithm not in the registry is rejected closed — no implicit trust in unknown algorithms even if the JWS protected header advertises one.

*Spec updates:*
- Update [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6 to reference the revocation surface and dual-sign pattern. Update Phase D's 1.0 stability contract to enumerate the crypto-agility migration story explicitly — the "verifies in 10 years" claim must be backed by a written migration path, not by hope.

**Success criterion.** A consumer reading the spec can answer: (a) "what happens to receipts when their kid is revoked?", (b) "how does this engine survive a quantum break of Ed25519?", (c) "what is the verifier's fail-closed behaviour on an unknown algorithm?" — without ambiguity, without consulting the author. **K7 in CI**: tampered = `signature-invalid`; revoked-kid = `revoked-kid`; receipt with only-unsupported-alg = `unsupported-alg`. All three failure classes distinct, all three locked by fixture.

**Blast radius if skipped.** First serious key compromise lacks a revocation mechanism; receipts issued during the compromise window are indistinguishable from honest receipts to every downstream verifier. First quantum-break public demonstration triggers a rush migration with no migration playbook, leaving downstream consumers to choose between fail-closed-on-everything (breaking archives) and trust-anyway (breaking the security claim).

---

## R0 — Resilience preflight (parallel with Phase A, between Pre-A3 and v0.9.A.3 in the queue)

The Phase A priorities + the Phase E.1 follow-ups + the Governance tracks together define the *verifiable happy path* — what's true when the engine works as designed. R0 is the *verifiable recovery path* — what stays true when something is hostile, broken, or compromised while the algebra remains correct.

The framing question for R0:

> What if the repo, CI, PyPI release, Worker deploy, JWKS, signing key, model provider, or benchmark process is compromised while the algebra remains correct?

G3 (revocation + crypto-agility) covers the receipt-key compromise case cryptographically. G1 (continuity + archival) covers the bus-factor case institutionally. **R0 covers the operational layer** — vulnerability intake, incident response, trust-root manifest, CI blast-radius control, fuzz harnesses for the trust boundaries — that sits between the cryptographic and institutional layers. None of these duplicate G1/G2/G3; they are orthogonal.

R0 lands as a single short cycle whose goal is "establish the operational primitives." The contents are mostly docs and CI configuration; the cost is one cycle, the value is that black-swan recovery has a documented playbook before the black swan arrives. **Insert in the queue between Pre-A3 truthfulness hygiene and v0.9.A.3 (CORS).**

### R0.1 — `SECURITY.md` + `docs/INCIDENT_RESPONSE.md`

**Problem.** No `SECURITY.md` exists at the repo root today. GitHub's own guidance recommends one as the canonical surface for vulnerability reporting; without it, a researcher who finds a real bug has no documented intake path. Separately, the codebase has no incident-response playbook — when (not if) one of the trust surfaces breaks, the response will be improvised under pressure rather than executed from a written reflex.

**Work.**
- Add `SECURITY.md` at repo root (or `.github/SECURITY.md`). Cover: the supported version range, the vulnerability disclosure email / GitHub private advisory channel, the expected response window, and the public disclosure policy (coordinated disclosure with a documented embargo).
- Add `docs/INCIDENT_RESPONSE.md` with one short subsection per failure case the existing trust surface admits is possible. Each case carries exactly four fields:
  ```
  Detection:
  Containment:
  Revocation / correction:
  Public disclosure wording:
  ```
  Cases to cover at minimum:
  - Render signing key suspected compromised.
  - JWKS served wrong / stale / revoked key.
  - PyPI release suspected compromised.
  - GitHub Actions workflow compromised.
  - Worker deploy compromised.
  - Benchmark claim later found wrong.
  - LLM provider silently changes served-model behavior.
  - Canonicalization bug discovered in Python / Node / browser verifier.
- Cross-reference G3's revocation surface (`/.well-known/revoked-kids.json`) from the receipt-key compromise case.

**Success criterion.** A researcher with a real finding has a documented intake path. The author (or a successor under G1) facing any of the listed crisis cases has a written reflex covering the four operational fields, not an improvisation.

### R0.2 — `sum.trust_root.v1.json` release manifest

**Problem.** A skeptic verifying SUM today has to compose the trust chain by hand from multiple surfaces: the GitHub release artifacts, the PyPI wheel, the published JWKS, the Worker bundle, the standalone verifier. There's no single artifact that names "as of release v0.X.Y, here are the bytes I claim to be shipping, here are their hashes, here are the public keys / kids that are current, here is the algorithm registry." The TUF (The Update Framework) project has the right model — root + targets + snapshot + timestamp metadata roles, with the root specifying trusted keys + signature thresholds — but adopting full TUF is overweight for SUM's current scale.

**Work.**
- Specify `sum.trust_root.v1.json` (TUF-inspired, not full TUF). Minimal shape:
  ```json
  {
    "schema": "sum.trust_root.v1",
    "issued_at": "2026-04-27T00:00:00Z",
    "repo": "OtotaO/SUM",
    "commit": "<release_commit_sha>",
    "release": "v0.3.1",
    "artifacts": [
      {
        "name": "sum_engine-0.3.1-py3-none-any.whl",
        "sha256": "...",
        "pypi_provenance": "present|absent",
        "github_attestation": "present|absent",
        "cosign_bundle": "present|absent"
      },
      {"name": "standalone_verifier/verify.js", "sha256": "..."},
      {"name": "worker-bundle.js", "sha256": "..."}
    ],
    "render_receipt_jwks": {
      "current_kids": ["sum-render-2026-04-27-1"],
      "revoked_kids": [],
      "jwks_sha256": "..."
    },
    "algorithm_registry": {
      "prime_scheme_current": "sha256_64_v1",
      "prime_scheme_next": "sha256_128_v2"
    }
  }
  ```
- Add `scripts/build_trust_manifest.py` to assemble the manifest at release time from CI-known facts (commit SHA, built artifact hashes, current JWKS state, algorithm-registry contents).
- Sign the manifest with the same Ed25519 substrate the receipts use (or a separate release-signing key — the choice is a documented spec decision; default to a separate release key so render-key compromise doesn't invalidate trust manifests).
- Publish as a GitHub release **asset** (not a path under `/releases/v<version>/` — that's not a fetchable URL). Canonical asset-download URLs:
  - Versioned: `https://github.com/OtotaO/SUM/releases/download/v<version>/sum.trust_root.v1.json`
  - Latest: `https://github.com/OtotaO/SUM/releases/latest/download/sum.trust_root.v1.json` (302-redirects to the most recent release's asset)
  - API form (machine-readable enumeration of all release assets, including hash + size): `https://api.github.com/repos/OtotaO/SUM/releases/tags/v<version>`
- Also publish the *current* release's manifest at `https://sum-demo.ototao.workers.dev/.well-known/sum-trust-root.json` for consumers who want to fetch it from the same origin as the JWKS without dereferencing the GitHub redirect.

**Success criterion.** A skeptic verifying SUM v0.3.1 fetches `sum.trust_root.v1.json` once, checks the signature, and from that one artifact knows: which commit shipped, what artifact hashes to expect, which kids are current, which are revoked, and which prime scheme is in force. The manifest is the one thing they verify *first*; everything else verifies against it.

### R0.3 — CI/runtime hardening baseline

**Problem.** The release pipeline has the same blast radius as any other GitHub Actions workflow: a compromised third-party action or an over-permissioned `GITHUB_TOKEN` is the default failure mode for supply-chain attacks on small projects. GitHub's own security-hardening guidance is to grant minimum permissions and pin third-party actions; OpenSSF Scorecard and StepSecurity Harden-Runner add advisory + observability layers cheaply.

**Work.**
- Default every workflow to `permissions: contents: read`. Grant `id-token: write` only in publish jobs; grant other write scopes only where strictly required.
- Pin every third-party GitHub Action by full commit SHA (not by mutable tag). Add a CI lint that fails on tag-pinned actions.
- Add OpenSSF Scorecard as an advisory job (non-blocking initially; promote to blocking once known issues are addressed).
- Add StepSecurity Harden-Runner in audit mode on the release / publish workflows so anomalous network egress, file changes, or process activity inside CI runners produces an audit trail. Compromised CI often looks like "normal build, slightly different network/file behaviour" — Harden-Runner records the baseline so deviations become visible.

**Success criterion.** Every workflow grants the minimum scope it needs. Every third-party action is SHA-pinned. Scorecard runs on every push. Harden-Runner records release-job baselines and surfaces deviations.

### R0.4 — Differential fuzz / property-test harness

**Problem.** SUM's adversarial fixture set covers known failure modes (A1–A6 + the receipt-fixture matrix). Property-based testing is the orthogonal axis — generated inputs at the trust boundaries, with shrinking on counterexamples. This catches the bug nobody predicted.

**Work.**
- Author a property-based harness using **Hypothesis** (Python) and **fast-check** (TypeScript / JavaScript) over the boundaries where bugs hide:
  - JCS canonicalization payloads (the integer-vs-float-zero edge case from RENDER_RECEIPT_FORMAT.md §4 is a known foot-gun; property-test for byte equality across Python `jcs` and TS `canonicalize@3`).
  - Detached JWS receipt verification (one-bit mutations on every signed field → all verifiers MUST reject).
  - Triple sorting / componentwise tuple-lex ordering (cross-runtime byte-stable, including triples that contain `|` separators in components).
  - `sha256_64_v1` (current) and `sha256_128_v2` (Priority 3) prime derivation byte-equivalence.
  - Malformed bundle roots and partial truncations.
- The properties to assert are *invariants*, not example-equality:
  - "For all generated valid receipts, Python / Node / browser verifiers produce the same outcome."
  - "For all one-bit signed-field mutations on a valid receipt, all verifiers reject."
  - "For all generated valid triple sets, componentwise tuple-lex sort is byte-identical across runtimes."
- Integrate into CI as a non-flaky job (Hypothesis with a seeded random, fast-check with a fixed `numRuns` budget). Failures are real bugs; pin the failing seed in a regression fixture and resolve.

**Success criterion.** Property-based harness runs in CI on every push. Counterexamples that surface get pinned as regression fixtures and fixed (or documented as known limitations in `docs/PROOF_BOUNDARY.md`). The test suite becomes generative, not just example-based.

**Deferred (post-R0):** stateful REST API fuzzing via RESTler against an OpenAPI spec for the Worker endpoints. Heavier-weight; pin as future work once `/api/render`, `/api/qid`, `/api/complete`, JWKS, and any future bundle routes are stable enough to spec.

### R0.5 — Design-now / prototype-later anchors

These three are scoped as *design notes within R0 v1*, not deliverables of the R0 cycle. Each gets one paragraph in `docs/INCIDENT_RESPONSE.md` or an adjacent doc so the design exists before it's needed; implementation lands as its own cycle.

#### Receipt / JWKS transparency anchoring (Rekor or OpenTimestamps)

**Design.** Daily, collect `{jwks_sha256, trust_root_sha256, release_manifest_sha256}` (and optionally a Merkle root over receipt hashes from that day) and write one external transparency anchor to Sigstore Rekor (immutable transparency log with inclusion proofs) or OpenTimestamps (Bitcoin-anchored timestamping). Publish the pointer in `docs/TRUST_ROOT_LOG.md`. **Do not log raw receipt content** — only hashes / Merkle roots. The anchor gives external attestation of "as of date X, these were the trust-root bytes" — independent of GitHub, Cloudflare, or the repo author.

**When to implement.** Once R0.1–R0.4 are in place and the trust-root manifest (R0.2) has a stable shape. Not before — anchoring an unstable shape just locks the instability.

#### `sum.model_call_evidence.v1` sidecar

**Design.** Optional sidecar that hashes (does not store raw) the model-call provenance: prompt template, axis prompt, request body, provider response ID, model reported, temperature, max_tokens, output tome:
```json
{
  "schema": "sum.model_call_evidence.v1",
  "render_id": "...",
  "prompt_template_hash": "sha256-...",
  "axis_prompt_hash": "sha256-...",
  "request_body_hash": "sha256-...",
  "provider_response_id_hash": "sha256-...",
  "model_reported": "claude-haiku-4-5-20251001",
  "temperature": 0,
  "max_tokens": 1234,
  "output_tome_hash": "sha256-..."
}
```
Hashes are forensic binding without leaking prompts/responses. Protects against future disputes of the form "the receipt says model X produced this — what prompt actually went out, what response came back?" Does not prove semantic truth; proves *process evidence*.

**When to implement.** As an opt-in flag on `/api/render` once v0.9.B/C/D verifiers exist. Don't ship a side-channel before the primary verification surface is fully closed.

#### Bounded formal modeling (TLA+ or Alloy) of three small protocols

**Design.** One-page model, bounded state, CI optional. Three candidates worth modelling:
1. JWKS rotation / cache TTL / revocation interaction (G3 + receipt cache durability).
2. Delta-bundle composition semantics (Priority 6).
3. Trust-root manifest update / rollback prevention (R0.2 lifecycle).

**Strict guardrail.** "If the model becomes a research project, stop." TLA+ and Alloy are tools for finding subtle protocol bugs in small bounded state machines, not for verifying the prime algebra (which is already pinned by tests). The value is exclusively in the protocol surfaces, where the wire-format / lifecycle decisions are easier to reason about formally than to debug after deploy.

**When to implement.** Optional. Pin as a one-week experiment if a future cycle has slack; defer indefinitely if not. Not a Phase A gate.

---

## External tooling tracks (parallel / post-Phase A)

The Phase A priorities define *what work* needs doing; this section names *which third-party tools* are the recommended substrates for that work where the choice is now informed enough to commit to. Pinning the substrates here means future cycles don't re-litigate them under scope pressure. Items in this section are categorized by maturity: *recommended now* (production-ready substrates), *prototype first* (architectural research that should be benchmarked before commitment), *research branches* (perf/ergonomics experiments, not migration targets), and *negative recommendations* (paths that look attractive but should not be taken at current maturity).

### Recommended substrates (production-ready, fit existing priorities)

These tools fit cleanly into existing priority work; cycles that touch the corresponding priority should default to these substrates unless a specific reason emerges to choose otherwise.

| Priority | Recommended substrate | One-line why |
|---|---|---|
| P5 — threat-model validation (rate-limit wired, abuse controls) | **Cloudflare Workers Rate Limiting + AI Gateway + Turnstile** (Turnstile only on anonymous-burst endpoints) | Demo lives in Cloudflare; native primitives beat rolled-our-own middleware. |
| P7 — supply-chain attestation | **GitHub Artifact Attestations + Sigstore Cosign + SLSA build provenance** (see subsection below) | Three substrates, three scopes. |
| P8 — LLM-extraction honesty | **Instructor (or Outlines)** for `LiveLLMAdapter.extract_triplets` | Strict triple schema before canonicalization; complements (not replaces) the deterministic sieve + NLI rescue. |
| Packaging hygiene (v0.3.1 above) | **`build` + `twine check` + `check-wheel-contents` + custom `README ↔ wheel.METADATA` diff** | PyPA-canonical stack; each tool answers a different question. |

#### P7 substrates in detail

PR #46 review flagged this row as too long for a table. Three substrates with non-overlapping scopes — keeping them as bullets makes the trade-offs visible at a glance:

- **GitHub Artifact Attestations.** Native GitHub primitive. Captures workflow / repo / org / environment / commit SHA / triggering event / OIDC-derived provenance for GitHub-release artifacts. The canonical "where was this built" answer for the GitHub-hosted release path. GitHub's own framing — *"attestations are not a guarantee that an artifact is secure, only a verifiable link to where/how it was built"* — fits SUM's "signed ≠ true" culture exactly.
- **Sigstore Cosign.** Signs arbitrary blobs and emits a portable bundle (signature + certificate + timestamp + transparency-log inclusion proof). The right substrate for **offline verification**, the standalone verifier JS bundle, the Worker bundle, and any artifact published outside a GitHub release.
- **SLSA build provenance.** Standardises the build-process claim itself — *"this artifact traces back through this build to its source"* — separate from the signing layer. Cosign signs the artifact; SLSA describes how it was made. The two compose.

Verify all three from a clean machine before attaching to the release.

### Prototype first — architectural research before commitment

#### M1 — Merkle set-commitment sidecar, RFC 9162-inspired

**Scope note (load-bearing — read first).** This is **not** the existing Akashic Ledger Merkle hash-chain ([`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.7). That hash-chain links *events* (`prev_hash` over operation/prime/axiom_key/branch) and proves no event was modified after the fact. M1 is a different artifact: a **static set commitment** over canonical fact keys at bundle issuance, providing membership-witness compactness for a single bundle's fact set. The two surfaces share a hash function and nothing else; conflating them in code or docs would be a regression.

**RFC 9162 scope.** RFC 9162 Certificate Transparency v2.0 specifies *both* (a) inclusion proofs against a tree root and (b) consistency proofs that prove append-only evolution between older and newer signed tree heads. A static per-bundle fact set gives the **inclusion** half cleanly. Consistency proofs only apply if SUM later maintains a transparency log of signed tree heads across bundle/log versions — that is a separate decision and explicit infrastructure, not a free byproduct of the leaf commitment. M1 ships *inclusion proofs first*; the consistency-proof path is a follow-on track contingent on whether SUM's deployment shape ever needs cross-bundle append-only continuity.

**Problem.** PROOF_BOUNDARY §2.2 documents the merge ceiling: `p50 ≈ 519 ms` at `N=1000`, extrapolating to `~50 s/op` at `N=10,000` and `>1 hr/op` at `N=100,000`. Above that ceiling, prime encoding is a viable signed witness for compositional reasoning but not the right substrate for efficient third-party membership verification at scale. A Merkle set commitment over canonical fact keys gives log-size inclusion proofs that scale where LCM does not — *without* giving up the algebra LCM provides locally.

**Architecture (sidecar, NOT replacement).** Keep LCM as the algebra for idempotent set union and divisibility-based entailment; add a Merkle set commitment alongside, not in place of. A signed bundle carries both the LCM state integer (for divisibility checks + merge composition) and the Merkle root (for log-size inclusion proofs). Verifiers needing fast third-party membership use the Merkle path; verifiers needing compositional reasoning use the LCM path. Both surfaces live in the same bundle, both get signed.

**Work.**
- Define the leaf + node format with explicit domain separation locked at spec time:
  ```
  leaf_hash = sha256("SUM-MERKLE-FACT-LEAF-v1\0" || canonical_fact_key)
  node_hash = sha256("SUM-MERKLE-FACT-NODE-v1\0" || left_hash || right_hash)
  ```
  Domain prefixes are cheap now and expensive later — without them, the same hash function namespace is shared with the Akashic Ledger hash-chain and any future hash-using surface, which invites collision-class confusion.
- Canonicalise insertion order (lex-sort on canonical fact key) so the same fact set always produces the same root regardless of insertion sequence. Locking insertion canonicality is the second load-bearing spec decision — once shipped, it's load-bearing forever.
- Use a known-good library: `pymerkle` or `ct-merkle` are the candidate Python implementations; both expose inclusion proofs against a static root. Cross-runtime equivalence demands a TS implementation too — `@transparency-dev/merkle` or equivalent.
- Benchmark proof size + proof-verify time + state-update cost at `N=100`, `N=1k`, `N=10k` against the LCM-only substrate. Numbers go in `docs/PROOF_BOUNDARY.md` §2.2 alongside the existing N-table.
- The success criterion is *empirical*: log-size inclusion proofs that verify materially faster than the LCM divisibility path at `N=10k`, with a verifier that's straightforward to implement from spec alone (G2 independence concern).

**Consistency-proof follow-on (out of scope for M1 v1).** If SUM later maintains a signed-tree-head log across bundle versions, RFC 9162-style consistency proofs become applicable. That is its own track — requires a tree-head log surface, head-signing key, retention policy, and verifier semantics for "this newer root is an append-only extension of that older one." Pin as future work; do not bundle into M1.

**Tradeoff.** A Merkle root does not preserve LCM's algebraic composability — adding a fact requires re-inserting a leaf and recomputing inclusion paths, not just multiplying primes. That's why this is a *sidecar*: LCM remains the local algebra; the Merkle path is an external-membership surface.

**Not before:** P3 (`sha256_128_v2`) — the two share the spec-stability surface, and changing the leaf hashing scheme in Merkle land while changing the prime scheme on the LCM side is a multi-axis migration that's better done sequentially.

### Research branches (perf / ergonomics experiments, not migration targets)

These are scoped as **branch-level benchmarks**, not as migration commitments. Run them, post the numbers, decide afterwards. Do not preemptively re-architect around any of them.

#### R1 — GMP / gmpy2 big-int benchmark

Swap the Python hot path to `gmpy2.mpz` wherever the Zig core is not already authoritative; benchmark encode + merge + entail + memory at `N=100 / N=1k / N=10k` against CPython `int` and the current Zig-assisted path. If GMP doesn't materially reduce merge cost, drop it. The Zig path is the existing escape hatch (the README "GMP via Zig" line is honest); GMP-via-Python is a fallback for environments without the Zig core.

**Tradeoff.** GMP-backed paths add system dependencies and complicate packaging. Don't ship GMP as a hard dependency; gate it behind an opt-in extra (`pip install sum-engine[gmp]`).

#### R2 — WIT-based cross-runtime ABI for the Zig/WASM core

Define a minimal WIT world for prime lookup, encode, merge, divisibility checks; generate bindings via `wit-bindgen` for native + `jco` for browser/JS; replay existing cross-runtime fixtures through the typed interface. The current `standalone_verifier/math.js`-shared-source pattern works fine for now; WIT is the upgrade for when polyglot bindings cost more to maintain than re-stating math twice.

**Tradeoff.** `jco` is explicitly experimental, and the component model adds tooling complexity. WIT also pulls SUM toward "one core everywhere" — keep G2 (independent verifier) explicit alongside any WIT migration to avoid monoculture bugs.

### Negative recommendations — paths that look attractive but should not be taken at current maturity

#### Do NOT replace LCM with raw multiplication

LCM gives **set semantics**: adding the same canonical fact twice does not change state. Multiplication encodes **multiset semantics**: duplicates accumulate. SUM's truthfulness contract assumes set semantics throughout (Ouroboros round-trip relies on it; `state % prime == 0` returns the same boolean regardless of how many times the prime entered the state). Switching to multiplication would be a semantic regression sold as a simplification, and would break the algebra invariants in PROOF_BOUNDARY §1.4 that the bench harness pins. **This is a stop-the-line trigger.**

#### Do NOT migrate to RSA accumulators or KZG / vector commitments yet

Both are real research directions; neither has a production-ready, browser-friendly, independently-audited library that fits SUM's "human-auditable spec" posture today. Cambrian-style RSA accumulator implementations explicitly self-label as research-grade. KZG libraries (`c-kzg-4844`, `arkworks-rs/poly-commit`) are Ethereum/zkSNARK-shaped and carry trusted setup + pairing complexity that doesn't pay for itself at SUM's current corpus size. **Run shadow prototypes if curious; do not migrate.** The Merkle sidecar (M1 above) is the path that does pay for itself at SUM's actual scale.

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
- **A render receipt that verifies despite its `kid` being on the published revocation list** (once G3 lands). The revocation surface exists to be honoured; a verifier that ignores it is a verifier that cannot be trusted with future revocations.
- **A render receipt that verifies under an algorithm not declared in the in-tree algorithm registry** (once G3 lands). Fail-closed-on-unknown-alg is load-bearing — implicit trust in unannounced algorithms is the crypto-agility analogue of a TLS downgrade attack.
- **A proposal that replaces LCM with raw multiplication** (or otherwise removes set semantics from the algebra). LCM is load-bearing for the truthfulness contract: idempotent fact union, no multiset accumulation, `state % prime == 0` invariant on the cardinality of additions. Any "simplification" that breaks this is a semantic regression dressed as cleanup. See External Tooling Tracks §"Negative recommendations."
- **A release that ships without a signed `sum.trust_root.v1.json` manifest after R0 has landed.** Once the manifest exists, every release MUST publish one — the manifest is the single artifact downstream consumers verify first. A release without one breaks that contract silently and makes future trust composition impossible.
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

**Governance tracks (G1 / G2 / G3) run in parallel with Phase A** — see the section above. They are not Phase A exit gates (so as not to block hardening cycles on third-party-archival logistics), but they SHOULD be substantially landed before Phase B begins. Phase B adds platform surface that consumers depend on; that dependency is much stronger when the institutional hedges (archival, independent verifier, revocation policy) are in place.

**Gate to exit Phase A.** Every priority's success criterion passes. CI matrix has K1–K5 + A1–A6 + threat-suite + supply-chain verify + LLM-guardrail assertions + receipt-verifier fixtures + property-based fuzz harness (R0.4) all green on every push. WASM perf numbers are pasted under each per-browser section in `docs/WASM_PERFORMANCE.md`. R0 deliverables (`SECURITY.md`, `INCIDENT_RESPONSE.md`, signed trust-root manifest, CI permissions baseline) are in place — the recovery path is documented before Phase B platform surface lands on top.

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

**Crypto-agility is part of the 1.0 contract, not an afterthought.** The "verifies in 10 years" promise is undefended unless G3's revocation surface and dual-sign migration pattern are in the spec by 1.0. Ed25519 is not quantum-safe; the contract must explicitly say what happens when the algorithm needs to be retired (dual-sign window, deprecation policy, fail-closed-on-unknown-alg). Without this, 1.0 is a promise the engine cannot keep at decade scale.

[`docs/ALGORITHM_REGISTRY.md`](ALGORITHM_REGISTRY.md) is the canonical surface for this — the in-tree allowlist of accepted signature algorithms + prime schemes, the deprecation lifecycle, and the dual-sign migration pattern (envelope shape spec'd; implementation lands when a successor algorithm is chosen). Verifiers fail closed with `unsupported_alg` on any algorithm not under `current` in the registry; the registry is the spec contract that 1.0's "verifies in 10 years" promise rides on.

Backed by a CI gate that tests the promise: a corpus of 1.0-minted bundles + receipts that every later release must continue to verify, plus the K6 + K7 invariants from G2 + G3. The hand-off point between "interesting prototype" and "dependency anyone can build on."

---

## How to use this document

- **Reading order** for a memory-less session: closed-priorities + Phase E.1 retrospective first (so you know what's already done), then open priorities, then platform trajectory. Don't skip ahead to Phase B work if Phase A priorities are still open.
- **Phase tags in commits** are optional but useful: prefix `[P3] …` or `[B1] …` so `git log --grep` finds all work on a given item.
- **When you discover a dependency** between two items not captured here, add the edge as an explicit sentence in the item text. Future learning refines the ordering.
- **Scope pressure** comes from two directions — "ship faster, skip the gate" and "ship more, add an item." Both are wrong. The priorities close debts that already exist; the phases add surface that depends on those debts being closed. Shipping anything new before its gates pass moves the debt, not the value.

## On the discipline itself

The single best habit to carry into the next session is: when you're about to write a claim, ask "what would adversarial review do with this sentence?" If the answer is "catch me," rewrite it before you commit.

SUM's whole thesis is that content-addressed, cryptographically-attested, cross-runtime-verified bundles + signed render receipts are a truth-telling surface because they cannot lie without being caught. The repo itself should hold to the same standard the bundles and receipts do.
