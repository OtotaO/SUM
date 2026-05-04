# SUM session handover — intensification arc closing (2026-05-04)

## Mission elevation (read first; this is the lens for every decision)

This is an open-source project the operator has invested significant time in. The goal has been elevated from "good" to "**among the greats**" — but only via methods that are **verifiable, provable, and testable**. The aspirational standard:

> Each artifact in the substrate should be so good that no one ever questions the reason for its existence.

In practice this means: every claim resolves to either (a) a mechanically pinned property in `Tests/`, (b) a measured value in a `fixtures/bench_receipts/*.json` receipt with a reproducible `bench_digest`, or (c) an explicit "what this does NOT pin" boundary. **No aspirational claims, no unverified assertions, no load-bearing prose.** The truth-first discipline that runs through `docs/PROOF_BOUNDARY.md`, `docs/SHEAF_HALLUCINATION_DETECTOR.md` (F3 STRUCTURAL FAIL named honestly), every `docs/COMPLIANCE_*.md` (each with "what this validator does NOT pin"), and the CHANGELOG (every test count mechanically refreshed) is the load-bearing property.

When deciding what to ship, keep asking: *would an external reader, looking at this artifact, find any reason to doubt it?* If yes, fix the reason, then ship. If no, the artifact justifies its own existence.

---

## State at HEAD `b02a129`

**Substrate is in a clean, fully-consistent state.** Seven PRs landed on 2026-05-03 closing the intensification path's load-bearing prep before the arXiv preprint. Every previously-named gap is now either a validatable rule, a reproducible measurement, or an honestly named scope boundary.

| Metric | Value |
|---|---|
| HEAD SHA | `b02a129` |
| Catalogued features | **156** (137 production / 18 scaffolded / 1 designed) |
| Compliance regimes shipped | **6** (EU AI Act Art 12, GDPR Art 30, HIPAA § 164.312(b), ISO 27001 A.8.15, SOC 2 CC7.2, PCI DSS v4.0 Req 10) |
| Compliance suite tests | **169** |
| Audit-log suite tests | **25** |
| Research (sheaf detector) tests | **86** |
| Full suite | **1767+ passing**, 0 collection errors, 0 outstanding failures (the 2 prior `test_concurrency_safety.py` flakes passed on the most recent run) |
| `bench_digest` reproducibility | **Unconditional** — no `PYTHONHASHSEED=0` required |
| Current bench digests | v3.2 validation: `b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a`<br>F3 diagnostic: `62b6e1878d1d12f36eb80e301304854a1a2c03386f0e872850d3461b2f733e7c` |
| `PROOF_BOUNDARY.md` | v1.6.0 (current at HEAD) |

---

## Initial self-check (first 5 minutes in this session — DO NOT SKIP)

Run these commands, in order, before doing any new work. Each step verifies a load-bearing claim from this handover. **If any step fails, stop and investigate before proceeding** — the substrate's truthfulness is the load-bearing property.

```bash
cd "/Users/ototao/Github Projects/SUM/SUM"

# 1. Confirm clean working tree at the expected HEAD.
git status                    # expect: clean
git log --oneline -1          # expect: b02a129 ... (#140)

# 2. Drift check — meta artifacts match doc state.
python3 -m pytest Tests/test_self_attestation.py -q
# expect: 7 passed

# 3. Compliance + audit-log + research suites green.
python3 -m pytest Tests/compliance/ Tests/test_audit_log.py Tests/research/ -q
# expect: 280 passed (169 + 25 + 86)

# 4. Bench reproducibility — unconditional, no env var.
for i in 1 2 3; do
  python3 -m scripts.research.sheaf_v3_2_validation 2>/dev/null | grep '"bench_digest"'
done
# expect: same digest 3x ("b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a")

# 5. Mechanical catalog count.
grep -cE "^### .*✅" docs/FEATURE_CATALOG.md   # expect: 137
grep -cE "^### "    docs/FEATURE_CATALOG.md   # expect: 156

# 6. Cross-regime registry consistency (the C1 contract).
python3 -c "
from sum_cli.main import _COMPLIANCE_REGIMES, _compliance_validators
assert set(_COMPLIANCE_REGIMES) == set(_compliance_validators()), 'registry drift'
print(f'OK: {len(_COMPLIANCE_REGIMES)} regimes wired')
"
# expect: OK: 6 regimes wired
```

If all six checks pass, the substrate is in the state this handover describes and you can begin work. If any fail, treat that as a stop-the-line signal and investigate.

---

## What landed on 2026-05-03 (today's seven-PR arc)

| PR | Title | What changed |
|---|---|---|
| **#134** | Catalog refresh + future-regime TODOs | Added entries 151–156 for the six compliance regimes (catalog had been silently drifted since PR #120); UK AI Bill + India DPDP dropped from candidate-future-regimes per operator decision; arXiv reordered to Sprint 7 (FINAL). |
| **#135** | Sprint 1: bench-digest substrate determinism | Single-line fix in `DeterministicSieve.extract_triplets` (`list(set(...))` → `sorted(set(...))`) — closed the `PYTHONHASHSEED=0` reproducibility caveat. Receipts rebased to 2026-05-03 with natural-determinism digests. |
| **#136** | Sprint 2: fastapi/starlette compat pin | Pinned `starlette<1.0` in `requirements-prod.txt` to fix `Router.__init__() got an unexpected keyword argument 'on_startup'`. **30 collection errors → 0**; 98 previously-erroring tests now pass. |
| **#137** | Sprint 3: shared compliance predicate library | Extracted `compliance/_predicates.py` with `is_iso8601_utc`. Six regime modules now import from one source (object-identity verified). 12 new predicate tests. |
| **#138** | Sprint 5b: sheaf detector library API doc | `docs/SHEAF_LIBRARY_API.md` — first latent capability surfaced. Documents the v2/v3/v3.2 module as a supported library surface (install path, stability tier, worked example, parameter reference, H16–H20 contracts, what's out of scope). |
| **#139** | PROOF_BOUNDARY v1.6.0 + close stale paths + close playbook TODOs | §1.10 expanded with the six-regime table; §2.9 expanded with v3.2 row + F1–F5 verdict ladder; §2.10 updated with substrate-determinism closure; CHANGELOG receipt paths refreshed; two outstanding playbook TODOs closed. |
| **#140** | Sprint 4: PCI DSS user-identification gap closed | Three optional identity fields (`user_id` / `host_id` / `ip_address`) added to `sum.audit_log.v1` (additive, backward-compatible). New env vars: `SUM_AUDIT_USER_ID` / `_HOST_ID` / `_IP_ADDRESS`. PCI validator gains **R7** firing on rows lacking `user_id`. The PR #133-named "structural gap" is now CLOSED at the substrate. |

The ordering is intentional. Each sprint either closed a load-bearing gap (1, 2, 4) or sharpened a substrate property (3, 5b) or reflected today's work into the canonical proof-boundary doc (139). Together they produce a substrate where the arXiv preprint can cite every claim without footnotes.

---

## What remains before the arXiv submit

**Three sprints remain.** Sprint 7 is the headline-shipping artifact; sprints 5a and 6 are *orthogonal additive* — they don't strengthen the preprint's load-bearing claims, only add capability the preprint could mention. Recommended ordering: **Sprint 7 first** (in this fresh-context session), then 5a and 6 alongside or after.

### Sprint 5a — Render-receipt aggregation CLI (orthogonal, optional)

`sum receipts aggregate <dir>` reads signed render receipts from a directory, verifies their Ed25519 signatures against known JWKS keys, and emits a `sum.receipt_aggregate.v1` JSON object listing counts by `kid` / schema / trust scope, the time range, and a per-`kid` signature-verification status. Surfaces the trust loop's signed-receipt history as a queryable artifact. Concrete second latent-capability surface (sister to PR #138).

### Sprint 6 — Production reference architecture doc (orthogonal, optional)

`docs/REFERENCE_ARCHITECTURE.md` — step-by-step deployment guide for a regulated AI pipeline. Sections: (1) `SUM_AUDIT_LOG` setup, (2) JSONL ingestion, (3) `sum compliance check` as CI gate for at least 2 regimes, (4) Worker render path with operator's own JWKS, (5) cross-runtime verifier setup. Working `docker-compose.yml` example. Converts "demo at sum-demo.ototao.workers.dev" into "documented production path."

### Sprint 7 — arXiv preprint v0.1 (FINAL)

`docs/arxiv/sheaf-detector-note-v0.md` v0 → v0.1. The synthesis task — every prior sprint exists to make the preprint's claims defensible. Fold in:

1. **v3 weighted Laplacian** (Hansen-Ghrist 2019 §3.2 weighted form anchoring)
2. **v3.1 boundary deviation** (Hansen-Ghrist Prop 4.1 / Thm 4.5 harmonic extension)
3. **F3 STRUCTURAL FAIL** as honest negative result (the 8-cell diagnostic refuted three competing hypotheses; the failure is mathematical, not parametric)
4. **v3.2 closure of F3 at the detector layer** (combined score with γ ≤ 0.1; subsumption to v3 at γ = 0; auto-calibration empirically wrong on this corpus — name it)
5. **`bench_digest` as a reproducibility primitive** (JCS-canonical SHA-256 over quantized payload; reproducibility is unconditional post-Sprint 1; signable with the project's existing JWKS keys → cryptographically-attested research artifacts)
6. **Six-regime compliance substrate as substrate-tightening evidence** (the same `sum.compliance_report.v1` shape carries six statutorily distinct regimes; PCI DSS R7 closes the load-bearing user-id gap)

**Citation anchors:** the three bench digests above. **Pre-circulation:** 1–2 readers before submit. **Categories:** cs.LG primary, cs.CR secondary.

**What makes Sprint 7 cognitively expensive:** synthesis. The preprint must hold the entire arc in mind — Gebhart-Hansen-Schrater 2023 → sheaf-Laplacian primitive → SUM's render-manifold mapping → contrastive training → corpus calibration → the F3 finding → the v3.2 fix → the substrate-determinism rebase → the compliance-regime evidence → the truth-first scope discipline that gives every claim its boundary. This is why it's last and why it gets a fresh context.

---

## Discipline reminders (the rules that produced today's clean state)

These aren't optional. They're the methods by which the substrate stays at the elevated standard.

1. **Truth-first labeling.** F3 was named STRUCTURAL FAIL, not "needs more work." The PCI user-id gap was named "load-bearing" until it was closed. Every wire-spec doc has a "what this does NOT pin" section. Refuse to soften failure modes — every honest negative becomes a load-bearing test that hardens the substrate.

2. **Falsification as a first-class artifact.** H1–H20 are pinned in `Tests/research/test_sheaf_laplacian_v{2,3,3,32,32_property}.py`. Each is a falsifiable claim — "if this fails, the spec is wrong." When you make a new claim, write the test that would falsify it. Universal-quantifier upgrades (Hypothesis property tests) are the strongest version.

3. **Mechanical counts over prose counts.** When a doc cites a count (features, tests, regimes), there must be a `grep` recipe that reproduces it. Drift between prose and reality is detected by `python3 -m pytest Tests/test_self_attestation.py`. Every doc-touching PR must end with `python3 -m scripts.attest_repo_docs` to refresh meta — the drift check fails otherwise (caught it twice today; pattern is reliable).

4. **Reproducibility over reproducibility-with-caveats.** Sprint 1's value was upgrading "bench reproducible with PYTHONHASHSEED=0" to "bench reproducible." Every caveat in a published claim is a future load-bearing-assumption-failure. Close caveats by fixing the substrate, not by adding footnotes.

5. **Receipts cite digests; digests cite code; code cites tests.** The chain is: external reader → bench receipt JSON → `bench_digest` value → `python3 -m scripts.research.<bench>` → identical digest → trust the receipt's measurements. Break any link and the chain doesn't carry. PR #135 was about closing a broken link.

6. **Substrate-tightening on natural cadence.** Six regimes meant the predicate duplication crossed the threshold from "nice DRY win" to "load-bearing technical debt" — PR #137 lifted it. The CLI dispatch `if regime == "..."` ladder was load-bearing at one regime, smelly at two — PR #130 lifted it. Don't force these ahead of need; do force them when need arrives. The signal is "this would require N-file lockstep edits."

7. **PR pattern.** Branch → write test → write code → ensure local pass → `python3 -m scripts.attest_repo_docs` → commit with detailed message → push → open PR with truthful body → watch CI → merge `--squash --delete-branch` → pull main. Every PR today followed this. CI failures (e.g. PR #127 manifest drift) are diagnostic, not punitive — fix and re-push.

8. **Truthful boundary-naming over false coverage claims.** The PCI wire-spec doc's "what this validator does NOT pin" section is the longest in the slate. That's correct, not a flaw — it's the honest report of what the validator can and cannot reach. A future regime that produces a 1-paragraph "NOT pin" section deserves more scrutiny, not less.

9. **Synthetic-vs-corpus discipline.** v3.1's H12 utility test passed on synthetic data; F3 corpus bench surfaced the structural fail. Synthetic tests are necessary but not sufficient. Every research claim that survives synthetic testing must also be exercised against a labeled corpus before the preprint cites it.

10. **The compression theorem.** Every primitive that compounds is a compression: specs compress requirements, signed receipts compress trust, `bench_digest` compresses reproducibility. Look for opportunities to compress; resist anything that uncompresses (vibe coding, threshold-based alerting, exact-match caching).

---

## Pointers (read order if you need to rebuild context)

1. **`CLAUDE.md`** — current handover pointer (item 0) is to *this* file
2. **`docs/PROOF_BOUNDARY.md`** v1.6.0 — proved-vs-measured discipline; six-regime table at §1.10; F1–F5 verdict ladder at §2.9; bench reproducibility at §2.10
3. **`docs/NEXT_SESSION_PLAYBOOK.md`** — open priorities; Sprint 7 framing at the bottom of the intensification roadmap
4. **`docs/FEATURE_CATALOG.md`** — 156 numbered features; entries 151–156 are the six compliance regimes (added today in PR #134)
5. **`docs/SHEAF_HALLUCINATION_DETECTOR.md`** — the research arc spec; §3.4.4 has the v3.2 closure with current digest values; §3.4.3 has the F3 STRUCTURAL FAIL diagnostic finding (preserved as historical record alongside v3.2's closure)
6. **`docs/SHEAF_LIBRARY_API.md`** — programmatic surface (PR #138)
7. **`docs/AUDIT_LOG_FORMAT.md`** — schema reference; "Optional identity fields (Sprint 4 / PR #140)" section documents the new env vars
8. **Compliance wire specs** — six docs: `docs/COMPLIANCE_EU_AI_ACT_ARTICLE_12.md`, `_GDPR_ARTICLE_30.md`, `_HIPAA_164_312_B.md`, `_ISO_27001_8_15.md`, `_SOC_2_CC_7_2.md`, `_PCI_DSS_4_REQ_10.md`. The PCI doc's §10.2.2 reads "CLOSED 2026-05-03 / PR #140" — operator-facing closure for the user-id gap.
9. **`CHANGELOG.md` `[Unreleased]`** — Sprint 4 entry at top describes the user-id closure; Sprint 1 entry covers the substrate-determinism rebase; further down covers Sprint 2/3/5b/6-regime arc

If picking up cold, read 1 → 2 → 3 in that order; sections 4-9 are reference material to consult on demand.

---

## When to /schedule a follow-up

Per the operator's prior decisions: don't schedule reminders for:
- Routine maintenance (no recurring sweeps in this project's discipline)
- Refactor follow-ups (close them in the same session or queue as a sprint)
- Future-regime monitoring (UK AI Bill, India DPDP — explicitly dropped on 2026-05-03)

DO offer to /schedule when there's a specific signal-bound trigger:
- arXiv submission deadline / response date
- A bench digest re-verification cycle (e.g., quarterly: confirm the receipts still reproduce after numpy / LAPACK / scipy upgrades)
- An external operator deploys SUM and reports back (would inform Sprint 6 reference architecture)

The bar is 85%+ odds the operator says yes. Most "follow up later" instincts fail this bar.

---

## Closing frame

The substrate as of HEAD `b02a129` is in a state where every claim resolves to either a test, a reproducible receipt, or an explicit boundary. The seven PRs that landed on 2026-05-03 closed every load-bearing prep before arXiv. Sprint 7 (the preprint) is now the synthesis task — it doesn't need further substrate work; it needs careful drafting that pulls together the math anchoring, the F3 negative result, the v3.2 closure, the `bench_digest` primitive, and the six-regime substrate-tightening evidence into a single coherent claim about reproducible-research-with-cryptographic-teeth.

The aspirational standard the operator named — "so good that no one ever questions the reason for their existence" — is the lens for Sprint 7's prose. Every sentence in the preprint should be answerable with a digest, a test, or a citation. No load-bearing prose. No aspirational claims presented as current state. The substrate is ready; the writing has to match the substrate's standard.

Begin with the self-check above. Once it passes, you have ground truth for everything in this handover and can start on Sprint 7.
