# Session handover — post-v0.7.0 substrate arc (2026-05-21 → 2026-06-01)

This handover covers the 13-PR arc that closed bench-hardening T1 + T4, fixed F4 / F13 / F14 in the install + dogfood path, demoted `api/quantum_router.py` to internal-research, and established two load-bearing process disciplines (live-state probing; pull-before-branch). At the close, **§2.5 closure is empirically multi-stage load-bearing under DKW 95%, the install path has 2-axis CI protection (latest + floor), and the substrate is empirically clean by every probe we can run.**

**If you take only one thing from this handover:** the rate-limit on SUM's outcome delivery has shifted from substrate to operator-side. Read §6 (operator queue) before producing any new substrate work.

---

## 0. Quick state at HEAD (`fad1cae`, 2026-06-01)

| Surface | State |
|---|---|
| Open PRs | 0 |
| PyPI | sum-engine 0.7.0 (released 2026-05-18) |
| Repo version | 0.7.0 (matches PyPI) |
| Live Worker | byte-identical to `single_file_demo/index.html` (last deploy 2026-05-24, version `1c829014-...`) |
| Cross-runtime trust triangle | locked — `make xruntime` + `make xruntime-adversarial` both green |
| Full pytest suite | **2310 passed, 1 skipped, 2 xfailed, 0 failed** (last full run 2026-05-28) |
| CI on main | green (8/8 jobs including new floor-venv-smoke matrix from PR #258) |
| Self-attestation | clean (`python -m scripts.attest_repo_docs --check` passes) |
| Repo manifest | current (`python -m scripts.repo_manifest --check meta/repo_manifest.json` passes) |
| Bench-hardening worktrail | T5 DONE, T1 CLOSED, T4 CLOSED, T2/T3 OPEN ($-blocked) |

---

## 1. The arc in one paragraph

T1 (iterated round-trip K=10) closed 2026-05-21 with all three measured corpora (seed_v1 / seed_v2 / seed_long_paragraphs) returning composition verdict STABLE — receipts on main, §2.5 closure now a multi-stage claim. T4 (drift composition law audit) closed 2026-05-22 with fixed-point composition law identified within DKW 95% bound on every corpus. F4 (`sum attest` missing `axioms` field) was Scenario A's step-4 blocker — fixed in PR #251. F13 + F14 caught upstream dep-rot in the `[sieve]` install path (spacy/typer/click chain; spacy 3.7 floor empirically broken) — fixed in PRs #256 + #258, with a new 2-axis CI gate (latest-venv-smoke + floor-venv-smoke) preventing recurrence. Doc surfaces were aligned with reality post-0.7.0 across 5 surfaces (PR #253) + 3 follow-up gaps caught by live-state probe (PR #255) + 2 self-inflicted orphans cross-linked (PR #259). `api/quantum_router.py` (1684 LOC, 26+ endpoints, 58 passing tests, no shipping surface) was demoted to internal-research per operator decision (PR #260) with the substrate preserved for option-value. Two memory entries written: **live-state probing > source-presence** and **pull-before-branch**.

---

## 2. PRs landed this arc (chronological)

| PR | Date | Subject | Why it mattered |
|---|---|---|---|
| #246 | 2026-05-21 | F12 — throttle + 5 retries on T1 NIM rate-limit | T1 runner kept dying mid-batch under NIM's 40 req/min |
| #247 | 2026-05-21 | F12 v2 — 8 retries, 65s+ 429 backoff | NIM's sticky 60s window reset; v1 retries exhausted |
| #248 | 2026-05-21 | T1 seed_long_paragraphs K=10 STABLE receipt | first T1 receipt; §2.5.1 added |
| #249 | 2026-05-22 | F12 v3 — 502 + broader 5xx detection | seed_v1 K=10 crashed on 502; class-name + string matches |
| #250 | 2026-05-22 | T1 CLOSED — §2.5.1.a/b/c all three corpora STABLE | retires §2.5 "open characterization" caveat empirically |
| #251 | 2026-05-22 | F4 — `sum attest` emits `axioms` field | unblocks Scenario A `attest → compose` end-to-end |
| #252 | 2026-05-24 | T4 — drift_pct composes as fixed-point on every corpus | §2.5 closure becomes multi-stage load-bearing under DKW 95% |
| #253 | 2026-05-24 | post-0.7.0 outcome-coherence pass | CLAUDE.md / worker README / README / CHANGELOG / FEATURE_CATALOG realigned |
| #254 | 2026-05-24 | categorical-foundations doc — DisCoCat vocabulary recasting | standards-track positioning for §2.5.1 fixed-point |
| #255 | 2026-05-25 | doc-coherence corrections (3 gaps caught by live-state probe) | DOGFOOD version 0.6→0.7, wrote EVIDENCE_CHAIN.md, README count 168→170 |
| #256 | 2026-05-28 | F13 — `click>=8.0` in `[sieve]` extra | `pip install sum-engine[sieve]` was broken on fresh venv (typer 0.13 stopped pulling click) |
| #257 | 2026-05-28 | DOGFOOD F13 capture | F13 added to findings ledger |
| #258 | 2026-05-29 | F14 floor bump + floor-venv-smoke CI gate | spacy 3.7 floor empirically broken; bumped to 3.8; 2-axis CI prevents recurrence |
| #259 | 2026-05-31 | orphan-doc close (CATEGORICAL_FOUNDATIONS + DOGFOOD_FINDINGS_2026-05-29) | self-inflicted orphans cross-linked from natural sibling docs |
| #260 | 2026-05-31 | `api/quantum_router.py` demoted to internal research | 1684 LOC unreachable from shipping surface; banner + README/CLAUDE.md notes |

---

## 3. New artifacts added this arc

### Code

- `scripts/bench/runners/s25_iterated_round_trip.py` — T1 runner (shipped 2026-05-18, used this arc)
- `scripts/bench/runners/t4_drift_composition.py` — T4 runner (new)
- `Tests/test_sum_cli_attest_axioms_field.py` — F4 regression guard
- `Tests/test_t4_drift_composition.py` — T4 runner test
- `pyproject.toml` `[sieve]` floor: `spacy>=3.8.0` + `click>=8.0` (F13 + F14)
- `.github/workflows/quantum-ci.yml` — new `pypi-install-smoke-floor` job

### Receipts

- `fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json` (T1)
- `fixtures/bench_receipts/s25_iterated_K10_seed_v2_2026-05-21.json` (T1)
- `fixtures/bench_receipts/s25_iterated_K10_seed_long_paragraphs_2026-05-21.json` (T1)
- `fixtures/bench_receipts/drift_composition_2026-05-22.json` (T4)

### Schemas

- `sum.iterated_round_trip_drift.v1` — T1 receipt schema
- `sum.drift_metric_composition.v1` — T4 receipt schema

### Docs

- `docs/CATEGORICAL_FOUNDATIONS.md` (PR #254) — DisCoCat-vocabulary recasting of §2.5.1 fixed-point. **Cited from `docs/PROOF_BOUNDARY.md` §2.5.1.d.** Explicit non-Frobenius framing per §5.
- `docs/DRIFT_METRIC_COMPOSITION.md` (PR #252) — T4's prose distillation
- `docs/EVIDENCE_CHAIN.md` (PR #255) — code-derived from `sum_engine_internal/evidence/chain.py`; closes a long-standing broken cross-ref in `docs/TRANSFORM_RECEIPT_FORMAT.md`. NOT fabricated: every symbol citation validated by importing the module before commit.
- `docs/DOGFOOD_FINDINGS_2026-05-28.md` — F13 (spacy/click dep rot)
- `docs/DOGFOOD_FINDINGS_2026-05-29.md` — F14 (spacy floor broken)
- Multiple updates to `docs/PROOF_BOUNDARY.md` (§2.5.1.a/b/c/d), `docs/BENCH_HARDENING_FROM_QCVV.md` (T1 + T4 closed), `CLAUDE.md` (shipping surface block refreshed; bench-hardening status block added; internal research surfaces block added), `README.md` (5 surface refreshes), `CHANGELOG.md` `[Unreleased]` (populated).

### Memory (new entries, indexed in MEMORY.md)

- `project_substrate_2026-05-28.md` — T1+T4 closure state, §2.5 multi-stage, arXiv gate open
- `feedback_live_state_probing.md` — discipline rule: live-state empirical > source-presence
- `feedback_pull_before_branch.md` — discipline rule: `git pull --ff-only` before `git checkout -b`

---

## 4. Disciplines established this arc (load-bearing patterns)

These are NOT just rules — they came from real corrections. Each was paid for in a specific recurrence.

1. **Live-state empirical probing > source-presence checks.** Frontend audit declared "5 sliders wired" by source grep; user opened the live page; PR #243's cascade BYO panel was missing from production (deployment drift). Then a follow-up audit said "no silent gaps"; live-state probe of the doc layer caught 3 gaps (DOGFOOD 0.6.0, missing EVIDENCE_CHAIN.md, README count 168 vs catalog 170). Memory: `feedback_live_state_probing.md`.

2. **`git pull --ff-only` before `git checkout -b`.** Two recurrences this arc (#258 vs #257; #260 vs #259) of branching from stale main → meta/* rebase conflicts at GitHub mergeability evaluation. Memory: `feedback_pull_before_branch.md`.

3. **Buyer-or-dream filter applied to substrate decisions.** Quantum_router triage (Option A/B/C); semhash defer-vs-build. Neither has buyer pull; both deferred. The filter prevents speculative substrate accretion in the wait+dogfood window.

4. **Decision-brief produce-then-defer.** semhash design + quantum_router triage both produced as full implementation briefs (empirical analysis, costs, options, risks) → operator decided → demoted/deferred per decision. Briefs live in conversation history; no doc accretion required.

5. **Code-derived (not fabricated) docs.** EVIDENCE_CHAIN.md wrote claims from real module imports; one wrong-file-path claim caught mid-drafting via `grep -rn "^class ProvenanceRecord"` and corrected before commit. Pattern: introspect the code; never invent the spec.

---

## 5. Engine state at HEAD (load-bearing claims, with epistemic status)

### Provable (mechanically proven)
- Three-runtime byte-symmetric Ed25519 over JCS bytes — locked by `make xruntime` (K1-K4)
- Three-runtime adversarial rejection-class equivalence — locked by `make xruntime-adversarial` (A1-A6)
- Canonical round-trip `reconstruct(parse(canonical_tome(S))) = S` — 0.00% drift on every CI run

### Empirical-benchmark
- **§2.5 closure is composition-stable under K=10 iteration on all three measured corpora** within DKW 95% bound (T1+T4 receipts on main). **This is the most load-bearing addition of this arc.** See PROOF_BOUNDARY §2.5.1.a/b/c/d.
- Slider fact preservation: median 1.000, p10 0.769 (long, n=16) / 0.818 (short, n=8) — see SLIDER_CONTRACT.md
- Extraction F1: 1.000 on seed_v1, 0.762 with precision 1.000 on seed_v2

### Certified
- F4 fix verified end-to-end (Scenario A `attest → compose` runs)
- F13 + F14 fixes verified via fresh-venv + floor-venv CI smokes
- 2310 passing tests at HEAD

### Not-asserted
- Truth of bundle content (verify never asserts factual truth; locked by test)
- Semantic preservation under arbitrary slider position (not measured at verify-time)

---

## 6. Operator queue at handover (the rate-limit)

**The single most important fact for the next session:** the rate-limit on SUM's outcome delivery has shifted from substrate (mine to ship) to operator-side (Umar's to advance). The next session should READ THIS SECTION before producing any new substrate work.

### Tier 1 — Money + signal (highest leverage)

1. **LinkedIn URLs × 4 grant applications** — per memory `project_grant_funnel`. ~$225K aggregate at stake. Pure administrative; hours to do.
2. **Dogfood pass on real writing** — Scenario A from `docs/DOGFOOD_QUICKSTART.md` is end-to-end runnable post-F4. Charter §3 names dogfood as load-bearing signal source. **0 user-side dogfood passes in this entire arc.**
3. **Funding-tracker check** at `~/SUMequities/.private-notes/` — would inform whether any pending grant references quantum_router-aligned / semhash-shaped work; flips multiple downstream decisions.

### Tier 2 — Administrative / one-time

4. SFF EIN registration
5. Foresight Bay-vs-Berlin node decision

### Tier 3 — Trigger-watch (calendar)

6. 6 grant decisions, May-Aug 2026 window — now in June. Some may have returned. Decision tree pre-loaded at `docs/PRODUCT_DELIBERATION_2026-05-14.md`.

### Tier 4 — Operator-triage decisions still open

7. `requirements-prod.txt` ↔ `pyproject.toml` divergence for remaining ~12 packages (6 of 18 resolved by #260)
8. 4 pre-existing orphan docs (`CI_WIRING_TO_PASTE_2026-05-17`, `INTEGRATION_GUIDE`, `PERFORMANCE_CHARACTERISATION`, `RECURSIVE_COMPRESSION`)
9. `bench_history.jsonl` disposition (70KB stale archival at repo root; last entry 2026-04-19)
10. Frontend↔backend parity expansion (~30% covered; filter-gated)
11. Whether to surface known xfails to funders (`Tests/test_threat_model.py` has 2)

### Tier 5 — Trigger-gated work

12. **T2 / T3 slider bench-hardening** — needs $ + OPENAI_API_KEY + auth
13. **arXiv preprint submission** — substrate gate now open (T1+T4 closed); operator decision per CHARTER strategy
14. **v0.7.1 PyPI cut** — `[Unreleased]` accumulated since 2026-05-18 (T1 + T4 + F4 + F13 + F14 + categorical-foundations doc); tag when users should get post-0.7.0 fixes
15. **Worker TS-port of `compose` + `extract`** — buyer-pull-gated (currently slider-only Worker-side, documented in PR #260's worker README clarification)
16. **`sum compare` / semhash surface** — deferred 2026-05-31. Design captured in conversation history (PR #260 conversation). Triggers to revisit:
    - A dogfood pass surfaces F-finding shaped like "I cannot verify meaning preservation between my source(s) and my distilled brief"
    - A grant deliverable references semantic verification / content-integrity-beyond-attestation / meaning-preservation
    - A standards-track collaborator asks for vector-similarity-of-bundles
    - Operator explicit override

---

## 7. What the next session should NOT do

These are filter-discipline rules, not personal preferences:

1. **Don't accrete new docs without operator pull.** This arc added 5 new docs (CATEGORICAL_FOUNDATIONS, DRIFT_METRIC_COMPOSITION, EVIDENCE_CHAIN, DOGFOOD_FINDINGS_2026-05-28, 2026-05-29). Each adds maintenance surface. Next session: extend existing series only, no new doc files unless operator explicitly asks.
2. **Don't build substrate without buyer pull.** semhash was deferred for exactly this reason; quantum_router was demoted not deleted for exactly this reason. The buyer-or-dream filter is the gate.
3. **Don't treat "go ahead" as authorization for multiple items.** Pattern this arc: user says "go ahead"; default = the ONE recommended item, not the menu. Reconfirm before expanding scope.
4. **Don't trust source-presence checks for "is X up to date" questions.** Empirical probe required. See `feedback_live_state_probing.md`.
5. **Don't branch from main without `git pull --ff-only` first.** See `feedback_pull_before_branch.md`. Two recurrences in 10 days; rule is paid for.
6. **Don't generate substrate work to fill the wait.** The rate-limit is operator-side. Substrate-work-as-fidget-toy is the anti-pattern.
7. **Don't fabricate doc claims.** EVIDENCE_CHAIN.md was code-derived via real `python -c "from ... import ..."` introspection; one wrong-path claim was caught mid-drafting. Pattern holds: introspect, don't invent.
8. **Don't presume the funder/journalist outcome without specific signal.** Each speculative substrate proposal can be justified by "if I build it they will come." The base rate disfavours it at SUM's stage.

---

## 8. What the next session SHOULD do (if anything)

In rough order of leverage, given the operator queue above:

1. **Verify the clean-state claim is still true** before believing it (live-state-probing discipline). Quick probes: open PRs, main CI status, live worker drift, version coherence.
2. **Acknowledge the operator queue as the actual rate-limit.** Don't generate new substrate work to fill operator-side waiting.
3. **If operator returns from a dogfood pass with an F-finding:** capture as `DOGFOOD_FINDINGS_2026-06-XX.md` F15+. Triage by severity. Buyer-or-dream filter applied to any proposed fix.
4. **If operator returns with a grant signal:** consult `docs/PRODUCT_DELIBERATION_2026-05-14.md` decision tree; act per branch without re-deliberation.
5. **If operator returns from funding-tracker check with grant relevance to quantum_router or semhash:** flip the corresponding deferred decision (Option B → A on quantum_router; defer → build on semhash).
6. **If operator authorizes a small unilateral item:** the diminishing-returns list is in conversation history (requirements-mapping, count-drift CI lint, orphan-doc triage brief, F14 spacy auto-download replacement, Makefile help docs). All small; each accretes maintenance surface; recommend only if explicitly asked.

---

## 9. Key pointers for the next session

### Read first
- `CLAUDE.md` — operational compass; READ FIRST every session
- `docs/CHARTER_2026-05-17.md` — strategic direction (intent, why, strategy, constraints)
- `docs/PROOF_BOUNDARY.md` — epistemic-status discipline. **§2.5 + §2.5.1 are this arc's load-bearing addition; multi-stage closure under DKW 95%.**
- This handover

### Memory (in `~/.claude/projects/-Users-ototao-Github-Projects-SUM-SUM/memory/`)
- `MEMORY.md` — index (always loaded)
- `feedback_live_state_probing.md` — pattern from this arc
- `feedback_pull_before_branch.md` — pattern from this arc
- `project_substrate_2026-05-28.md` — substrate state snapshot

### Bench receipts (load-bearing for §2.5)
- `fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json` + `_seed_v2_` + `_seed_long_paragraphs_` — T1 receipts
- `fixtures/bench_receipts/drift_composition_2026-05-22.json` — T4 receipt

### Dogfood signal capture (the load-bearing signal source)
- `docs/DOGFOOD_QUICKSTART.md` — the journalist Scenario A path; end-to-end runnable post-F4
- `docs/DOGFOOD_FINDINGS_2026-05-17.md` (F1–F7), `_2026-05-18.md` (F8–F11), `_2026-05-28.md` (F13), `_2026-05-29.md` (F14)

### Status surfaces
- `CHANGELOG.md` `[Unreleased]` — what accumulated since v0.7.0 PyPI
- `docs/BENCH_HARDENING_FROM_QCVV.md` — T5/T1/T4 CLOSED, T2/T3 OPEN

### Crypto trust loop (don't break)
- `make xruntime` (K1-K4 valid-path)
- `make xruntime-adversarial` (A1-A6 rejection-class)
- `make smoke` (fresh venv attest→verify)
- Floor-venv-smoke (CI only; new in PR #258)
- `make verify-release-bytes` (pre-tag check; never run this arc — operator-side decision)

### Internal research surfaces (NOT shipping; documented as such)
- `api/quantum_router.py` + `quantum_main.py` — see banner at top of router file; promote to `[api]` extra only if a named buyer or grant deliverable pulls
- `sum_engine_internal/research/` modules — load-bearing for substrate; specific research directions (MinHash-LSH, Robust PCA, MMD, Sheaf-Laplacian, etc.) catalogued in FEATURE_CATALOG

---

## 10. The single sentence to start the next session with

**Substrate is empirically clean; the rate-limit is operator-side (LinkedIn × 4, dogfood pass, funding-tracker check); do not generate substrate work to fill the wait.**

If that sentence stops being true — either because operator-side work moved and surfaced a signal, OR because a buyer/funder pulled something — the queue in §6 has the next moves with explicit triggers.

---

## 11. Closing arc-level metric

This arc opened 2026-05-21 with §2.5 closure carrying an "open characterization" caveat ("the §2.5 measurement is single-step; whether closure holds under K-step iteration is not yet measured"). It closed 2026-06-01 with the caveat retired by 3 STABLE receipts under T1 plus a fixed-point composition law fit under T4 within DKW 95% bound on every measured corpus. **§2.5 closure is now a multi-stage load-bearing claim.** That, plus the F13+F14 install-path fix that made `pip install sum-engine[sieve]` actually work on a fresh venv, are the two things this arc delivered that no prior arc had.

Everything else this arc shipped was hygiene around those two facts. Important hygiene — outcome-coherence at the doc layer; deployment drift closed; categorical positioning recorded; orphans cross-linked; quantum_router triaged — but hygiene.

The next arc's load-bearing change is unknown until operator-side work surfaces the signal.
