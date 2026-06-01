# SUM — Claude Code repo notes

Instructions that Claude Code inherits every session in this repo. Keep this
short. Repo-wide engineering conventions live in `docs/`; this file is for
rules a Claude Code session needs to know that are not otherwise obvious from
the code.

## Onboarding a memory-less session

If this is your first turn in this repo, read these files in order and you
will have the full picture.

### Operational compass — READ FIRST

These four planning artifacts are how the project actually operates.
The charter is the compass; the deliberation has the decision tree;
the zenith framing has the destination; the bench-hardening plan has
the empirical-discipline worktrail. **A session that skips these will
auto-pivot to substrate work and miss the standing direction.**

00a. **[`docs/CHARTER_2026-05-17.md`](docs/CHARTER_2026-05-17.md)** —
   operational charter. Intent, the Why (three layers), strategy
   (substrate first → adoption through writers → standards via
   real-customer pull → product-company only if a funder pays for
   it), objectives (short / medium / long term), success criteria
   (with explicit false-signals / auto-pivot-trap warnings),
   constraints (10 "won't do" rules), the operational loop
   (per-session / weekly / on-grant-signal / on-dogfood-finding /
   monthly / on-external-pull cadences). Supersedes scattered
   framing across this document and other memory entries for
   strategic-direction questions.

00b. **[`docs/PRODUCT_DELIBERATION_2026-05-14.md`](docs/PRODUCT_DELIBERATION_2026-05-14.md)**
   — tactical deferral artifact. Three-option analysis (writer's
   tool / standardization / omni-modal) + the grant-outcome
   decision tree (per-branch next-action so signal arrival doesn't
   trigger re-deliberation from scratch). Reference for the
   "scope-before-signal" rule that gates substrate work during
   grant-decision windows.

00c. **[`docs/ZENITH_FRAMING_2026-05-16.md`](docs/ZENITH_FRAMING_2026-05-16.md)**
   — destination framing. SUM as the chain-of-custody layer for
   AI-transformed knowledge. Three new concepts persisted from a
   cross-session deliberation: **Perspective Receipts** (rename
   the 5-axis sliders as named perspectives — novice / expert /
   regulator / etc.), **Trust Profiles** (`sum verify --profile
   <use-case>` bundling compliance regimes as product features),
   **Epistemic Nutrition Label** (per-artifact user-visible
   summary of the proof-boundary discipline). Plus the canonical
   one-sentence opener and the layered `sum verify --explain` v1
   output design.

00d. **[`docs/BENCH_HARDENING_FROM_QCVV.md`](docs/BENCH_HARDENING_FROM_QCVV.md)**
   — empirical-benchmark hardening plan distilled from Hashim et
   al., PRX Quantum 6, 030202 (2025). Five tasks (T5 negative-
   control corpus → T1 iterated round-trip → T4 drift-composition
   audit → T2 capability regions → T3 DKW worst-case bounds);
   recommended order T5 → T1 → T4 → T2 → T3. Hard rules: no use
   of "guarantee" downstream of this plan without a same-commit
   benchmark receipt; no quantum-vocabulary imports (Pauli, Choi,
   fidelity-as-alias). Three new schemas
   (`sum.iterated_round_trip_drift.v1`,
   `sum.slider_capability_region.v1`,
   `sum.slider_drift_bench.v2`).

### Historical research-arc handovers

The session-handover documents below are research-arc-specific
context (Path 2 / Sprint 7 / intensification / v3-diagnostic /
v0.5.0 work). Read them when a specific narrative is in question.
The compass above takes precedence for strategic-direction
questions.

0. **[`docs/SESSION_HANDOVER_2026-06-01_post_v0.7.0_arc.md`](docs/SESSION_HANDOVER_2026-06-01_post_v0.7.0_arc.md)**
   — **most recent session-block handover** (PRs #246–#260 over
   2026-05-21 → 2026-06-01). The 13-PR arc that closed bench-hardening
   T1 + T4 (§2.5 closure is now empirically multi-stage load-bearing
   under DKW 95% on every measured corpus), fixed F4 (`sum attest`
   missing axioms field, Scenario A unblocker), fixed F13 + F14
   (spacy/typer/click dep rot in `[sieve]` install path + spacy floor
   broken at runtime; both shipped with a new 2-axis floor-venv-smoke
   CI gate), aligned 5+3+2 doc surfaces with reality (PRs #253 / #255
   / #259 — outcome-coherence pass + live-state-probe corrections +
   self-inflicted orphan cross-links), shipped the categorical-
   foundations DisCoCat-vocabulary recasting for §2.5.1 (PR #254),
   and demoted `api/quantum_router.py` (1684 LOC, 26+ endpoints, 58
   passing tests) to internal-research per operator decision (PR #260)
   with banner-only approach preserving substrate option value. Two
   load-bearing process disciplines established + memorialised:
   **live-state empirical probing > source-presence checks**
   (`feedback_live_state_probing.md`), and **`git pull --ff-only`
   before `git checkout -b`** (`feedback_pull_before_branch.md`).
   **READ THIS HANDOVER FIRST** — §6 names the operator queue that
   is the actual rate-limit on outcome delivery; §7 names anti-
   patterns the next session must avoid. The prior handover at
   [`docs/SESSION_HANDOVER_2026-05-05_path2_arc.md`](docs/SESSION_HANDOVER_2026-05-05_path2_arc.md)
   — covers PRs #156 / #157 / #158 /
   #159 / #160 / #161 / #163 / #164 — Path 2 real-LLM-rendered
   adversarial bench closes the §7 asterisk; multi-LLM cross-family
   corroboration extends to six LLM lineages from six organisations
   across closed + open weights; cross-corpus extension across three
   corpora initially appears corpus-specific, then resolved by
   §4.7.4.1 as extremal-Goodhart at small n — at controlled sample
   sizes (n ≥ 16) across 3 corpora × 4-6 LLM lineages, joint finding
   is `STRUCTURAL_GAP_NO_MODEL_BEATS` in 3/3 corpora). The
   synthetic-bench WIN (+0.043) is now read as a Goodhart artifact:
   hybrid selected to compose well on a measure (the synthetic
   harness), measure stops being a good measure once it is the
   target. The "Phase 1 same-process contamination" was
   misdiagnosed and root-caused as a dict-iteration-order bug fixed
   in PR #160; the open-weights extension routes through HF
   Inference Providers via the HF route in
   `llm_dispatch.get_adapter`. §7 restructured into a four-tier
   audit; PROOF_BOUNDARY §2.10 reframed as continuous-enforcement
   against mutualism breakdown. Open v0.4+ candidates:
   real-LLM-aware per-triple V training, naturalistic perturbation
   synthesis, deeper corpus sampling (5-10 corpora at n ≥ 16).
   **Read this first if the §4.7.x narrative is in question.** The
   prior handover at
   [`docs/SESSION_HANDOVER_2026-05-05_sprint_7_5_arc.md`](docs/SESSION_HANDOVER_2026-05-05_sprint_7_5_arc.md)
   — Sprint 7 v0.1 prose fold-in + Sprint 7.5 hardening arc (PRs
   #142 / #146 / #144 / #145 / #147 — complementary-hybrid
   detector recovery and cross-machine bench_digest verification;
   five-PR arc summary; preprint v0 → v0.1; baseline-comparison
   gate fires STOP-THE-LINE; four recovery experiments produce
   HYBRID_BEATS_BASELINE WIN at trusted-mean AUC 0.876, Δ=+0.043
   vs B2; Modal cross-machine confirms both digest and verdict
   reproduce; substrate-first reframe lands; in-repo doc sync).
   Names operator-only items remaining before arXiv submit:
   optional v0.6.0 release, pre-circulation packet, arXiv submit
   — those are still open at the §4.7.3 close. The prior handover at
   [`docs/SESSION_HANDOVER_2026-05-04_intensification_arc.md`](docs/SESSION_HANDOVER_2026-05-04_intensification_arc.md)
   carries the intensification-arc context (PRs #134–#140 — catalog
   refresh, bench-digest determinism, fastapi pin, shared compliance
   predicates, sheaf detector library API doc, PROOF_BOUNDARY v1.6.0
   refresh, PCI user-id gap closure). The handover before that at
   [`docs/SESSION_HANDOVER_2026-05-02_v3_diagnostic_arc.md`](docs/SESSION_HANDOVER_2026-05-02_v3_diagnostic_arc.md)
   carries the v3-diagnostic-arc context (PRs #119–#125: audit-log
   gap closure, EU AI Act Article 12 validator, v3 receipt-
   weighted detector, v3.1 harmonic extension, F3 STRUCTURAL FAIL
   diagnostic, `bench_digest` substrate). The earlier handover at
   [`docs/SESSION_HANDOVER_2026-05-01_research_arc.md`](docs/SESSION_HANDOVER_2026-05-01_research_arc.md)
   (PRs #103–#117 + v0.5.0 + initial sheaf-Laplacian work) carries
   the previous arc's context;
   [`docs/SESSION_HANDOVER_2026-05-01.md`](docs/SESSION_HANDOVER_2026-05-01.md)
   (PRs #97–#102; v0.4.0 → v0.4.1 publish-path arc) and
   [`docs/SESSION_HANDOVER_2026-04-30.md`](docs/SESSION_HANDOVER_2026-04-30.md)
   (PRs #83–#95; v0.3 → v0.4 substrate arc) carry the earlier
   thread-pickup context.
1. **[`CHANGELOG.md`](CHANGELOG.md)** — release history. `[0.1.0]`
   was the first PyPI release (2026-04-22); `[0.2.0]` /`[0.2.1]`
   are hygiene fixes; `[0.3.0]` (2026-04-23) added the agentic
   introspection surface (`sum ledger`, `sum inspect`, `sum schema`).
   Anything after that lives under `[Unreleased]`.
2. **[`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md)** — proved-vs-
   measured discipline for every claim in the repo. Section 1.3.1 covers
   the cross-runtime Ed25519 trust triangle (Python ↔ Node ↔ Browser).
3. **[`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md)** — **170** numbered
   features (current at 2026-05-25; the post-0.7.0 catalog refresh added
   entries 169–170 — evidence-chain layer + T1 iterated round-trip runner;
   v3 / v3.1 / F3 diagnostic ship as research under the `[research]` extras
   flag, not as catalog features), each with a reproducible verification
   command. Summary at the bottom gives the Production / Scaffolded /
   Designed counts (currently 151 / 18 / 1). Counts are mechanically
   refreshed; treat them as authoritative over any prose in this file.
4. **[`Makefile`](Makefile)** — every dev command canonicalised. `make help`
   renders the full list. Common ones: `make install`, `make test`,
   `make xruntime`, `make smoke`.
5. **[`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md)** —
   ordered work queue (Priorities 1–8) plus post-hardening platform
   trajectory (Phases A–D), principles you must internalise before
   editing claims, stop-the-line triggers. Read first if you are
   picking the thread up cold. The ordering is precedence, not
   preference: earlier priorities harden existing claims; later ones
   extend the surface. Phases B and C depend on Phase A priorities
   being closed first — do not start Phase B work while Phase A
   priorities are still open.
6. **[`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md)** — slider
   product contract. Five axes, per-axis drift formulas, fact-
   preservation thresholds, the v0.4 → v0.7 NLI-audit / scale-bench /
   prompt-hardening arc. Canonical source for the headline numbers
   (median LLM-axis fact preservation = 1.000; p10 = 0.818 short-doc
   n=8, 0.769 long-doc n=16; min lifted from 0.111 → 0.700 by v0.7
   prompt hardening; catastrophic outliers eliminated 2 → 0).
7. **[`docs/SLIDER_V02_RESEARCH.md`](docs/SLIDER_V02_RESEARCH.md)** —
   research/methodology behind the slider's v0.2+ substrate. Itself
   stale relative to the current head (pre-v0.7) but useful as
   context: which choices the survey validated as load-bearing
   (verifiable rewards, cycle-consistency, content-addressed
   provenance, IB Pareto frontier), MontageLie threat model,
   constrained-decoding rationale, NLI audit positioning.
8. **[`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md)** —
   wire spec for the trust loop. `sum.render_receipt.v1`: Ed25519
   (RFC 8032) over JCS-canonical bytes (RFC 8785), wrapped as
   detached JWS (RFC 7515 §A.5) with public keys distributed via
   JWKS (RFC 7517). Defines payload field semantics, six-step
   verifier algorithm, cross-runtime canonicalisation rule (the
   integer-vs-float-zero gotcha), trust scope (what a verified
   receipt does and does NOT prove), key rotation cadence, C2PA
   `digital_source_type` alignment. Source-of-truth for every
   receipt-related claim PR A or any future doc-pass writes.
9. **[`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]`** — the full
   v0.4 → v0.9.A.2 arc since the last tagged release. v0.4 NLI audit
   verified the slider product claim; v0.5 Worker render path + slider
   UI; v0.6 long-doc scale verification (n=16); v0.7 prompt hardening
   eliminated catastrophic outliers; v0.8 four-layer defence against
   `LengthFinishReasonError`; v0.9.A render receipts (signed JWS +
   JWKS); v0.9.A.1 review-pass triple-sort + doc-bytes regen; v0.9.A.2
   route `/.well-known/*` through Worker + keygen polish.
10. **[`docs/SHEAF_HALLUCINATION_DETECTOR.md`](docs/SHEAF_HALLUCINATION_DETECTOR.md)** —
    research direction (2026-05-01). Specifies a sheaf-Laplacian
    consistency score over signed render-receipt manifolds, grounded
    in Gebhart, Hansen & Schrater (2023, AISTATS, arXiv:2110.03789)
    and the sheaf-Laplacian theory of Hansen & Ghrist (2019). The
    SUM-to-Knowledge-Sheaves mapping is charted concretely (§2.3 of
    that doc); v1/v2/v3 procedures specified; falsifiable predictions
    named; bounded claims set. **No code shipped yet** — the present
    document is the specification; `sum_engine_internal/research/`
    will scaffold in a separate PR. This is the first artifact that
    grounds SUM's primitives inside the peer-reviewed categorical-AI
    conversation; read it before adding new research-flavoured
    directions to the project.

Shipping surface at the current HEAD: the `sum` binary (currently
`v0.7.0` on `pyproject.toml` and on PyPI; v0.7.0 closed the transform-
substrate arc — `sum.transform_receipt.v1` wire format, transform
registry (slider / extract / compose), `POST /api/transform` Worker
route, `sum transform` CLI subcommand, T4 source-chain binding, T5
ShareableRender, T6 multi-school extract, 20-fixture cross-runtime
K-matrix, Python LLM-axis slider dispatch, opt-in replay-defense
window check across all four verifier surfaces, multi-provider
cascade in `LiveLLMAdapter.from_model` (OpenAI / Anthropic-via-Worker
/ HF / NIM / Groq / Cerebras / Ollama / llama.cpp / `local:`), per-IP
rate limiter + BYO-key gate on public Worker LLM-axis routes, `sum
verify --explain` layered output (`sum.verify_explained.v1`), F1 / F7
fixes, T5 negative-control bench corpus. **Post-0.7.0 on main:** T1
iterated-round-trip CLOSED (PRs #248 + #250 — three K=10 receipts,
all corpora STABLE under composition); F4 attest-axioms-field fix
(#251 — Scenario A's `attest → compose` step unblocked); F12
v1/v2/v3 NIM rate-limit retry hardening (#246 / #247 / #249).
**In-flight:** T4 drift-composition audit (PR #252 — drift_pct fits
a fixed-point composition law on every measured corpus within DKW
95% bound). The Node verifier in `standalone_verifier/`, and the
browser demo in `single_file_demo/`, both verify Ed25519 on the same
bundle bytes — single_file_demo extended in PR #243 with cascade
BYO-keys + CLI-recipe builder. The cross-runtime harness (`make
xruntime` → K1 / K1-mw / K2 / K3 / K4) proves this and runs on
every PR.

**Bench-hardening worktrail status** (per `docs/BENCH_HARDENING_FROM_QCVV.md`,
recommended order T5 → T1 → T4 → T2 → T3):
- T5 — negative-control corpus: **DONE** (`fixtures/bench_receipts/negative_control_2026-05-17.json`)
- T1 — iterated round-trip: **CLOSED 2026-05-21** (PRs #248 + #250)
- T4 — drift-composition audit: **CLOSED 2026-05-22** (PR #252 — pending merge)
- T2 — volumetric capability regions for slider bench: **OPEN** (needs `sum.slider_drift_bench.v1` receipts)
- T3 — DKW worst-case bounds for render receipt trust scope: **OPEN** (needs `sum.slider_drift_bench.v1` receipts)

**Internal research surfaces (NOT shipping, but present in repo):**
- `api/quantum_router.py` + `quantum_main.py` — FastAPI surface with 26+ endpoints (`/state`, `/sync`, `/branch`, `/merge`, `/zk/prove`, `/zk/verify`, `/peers`, `/time-travel`, `/auth/token`, etc.). 1,684 LOC of working code, 58/58 tests pass via pytest default discovery (`Tests/test_phase13_zenith.py`, `test_phase14_ouroboros.py`, `test_phase15_abi.py`, `test_browser_extension.py`). NOT in PyPI wheel (`pyproject.toml:167` excludes `api*`); NOT in live Worker; NOT in dogfood quickstart. Banners at top of both files explain. **Promote to a shipping `[api]` extra only if a named buyer or grant deliverable explicitly references one of the endpoint clusters.** Demoted 2026-05-30 per operator decision following deeper-audit triage; substrate it composes (`GodelStateAlgebra`, `AkashicLedger`, `OuroborosVerifier`, `ZKSemanticProver`, `EpistemicMeshNetwork`) remains load-bearing for the shipping surfaces under `sum_engine_internal/`.

If you're about to make a change and want to know what's already deferred,
check the task list for items marked "deferred" (Wikidata QIDs SPARQL
disambiguation, `sha256_128_v2` activation, browser-bench perf numbers).

## Out of scope — do not cross-repo edit

- Anything under `~/SUMequities` or `github.com/OtotaO/SUMequities`. That
  is a personal-portfolio repo with its own Claude Code session; SUM the
  engine ships independently and does not maintain portfolio-narrative
  artifacts. If the portfolio needs a description of SUM, it authors one
  in its own repo or pulls from `README.md` via its own loader.
