# Session handover — 2026-05-01 (research arc + v0.5.0 + compliance substrate)

**Read this first** if you are picking up this thread cold. Pair with
the docs in [`docs/`](.) and the codebase at HEAD; nothing here
contradicts the docs, this is the thin layer of *context that does
not appear in the code itself* — decisions made, things deliberately
deferred, gotchas the future-you would otherwise re-discover.

This supersedes [`docs/SESSION_HANDOVER_2026-05-01.md`](SESSION_HANDOVER_2026-05-01.md)
(which covered PRs #97–#102) as the canonical entry point. The
prior handover remains useful for the v0.4.0 → v0.4.1 publish-path
arc; this one carries the v0.5.0 publish + the sheaf-Laplacian
hallucination-detection research arc + the compliance substrate.

---

## What just shipped (15 PRs, in chronological merge order)

| #   | Subject                                                                                          | Why it matters |
|-----|--------------------------------------------------------------------------------------------------|----------------|
| 104 | MCP server `render` tool                                                                         | Bidirectional symmetry on the agent surface. CLI/MCP/HTTP × attest/verify/render 3×3 grid is fully populated. MCP-aware LLM clients drive both directions of the trust loop entirely from inside an LLM session. |
| 105 | Sheaf-Laplacian hallucination detector — research-direction spec                                 | Grounds SUM's primitives inside the peer-reviewed categorical-AI conversation (Gebhart 2023, Hansen-Ghrist 2019). v1/v2/v3 procedures specified; falsifiable predictions named. |
| 106 | v1 sheaf-Laplacian implementation + spec corrections                                             | Math verified via 7 sanity properties. Synthetic micro-benchmark (connected graphs): 18/30 catch, 100% top-1 localisation on caught classes. Empirical corrections to the spec landed alongside (A5 partially caught; P3 actually 100%; empty-render edge case). |
| 107 | v1 real-data falsification on naturalistic prose                                                 | **First documented falsification.** v1's density-dropout signal collapses to zero on disconnected source graphs (the textbook naturalistic case). Pinned in `test_disconnected_graph_density_dropout_invisible`. |
| 108 | `release: v0.5.0`                                                                                | Rotated `[Unreleased]` → `[0.5.0]`. Bidirectional MCP + sheaf-Laplacian research substrate. New `[research]` extras flag (numpy + scipy). |
| 109 | v1 audit-found foot-gun fixes                                                                    | Pre-tag-push audit found `stalk_dim` accept-mid-pipeline-error gap and `consistency_profile([])` IndexError. Fixed before v0.5.0 published. |
| 110 | v2 spec split into v2.0 / v2.1 / v2.2 after Hansen-Ghrist 2019 reading                            | Reading the foundational paper surfaced that the post-#108 v2 sketch ("swap stalk_dim to 384 with identity restriction maps") was mathematically insufficient. Spec corrected before any v2 code lands. |
| 111 | v2.1 scaffold + headline falsification                                                           | **Second documented falsification.** Math + training works; v2.1 with presence-style cochains *also* misses the disconnected-graph case (same structural issue v1 had — training amplifies present, not absent). |
| 112 | v2.2 combined detector closes disconnected-graph blindspot                                        | Anti-cochain analytically falsified (sign-flip preserves magnitude); absolute-presence regularizer collapses to v1's count. **Resolution: orthogonal-signal composition.** $V = \|\delta x\|^2 + \lambda \cdot d^2$. Deeper finding: Laplacian quadratic form is FUNDAMENTALLY a measure of cross-edge agreement, not entity presence — disconnected-graph dropout was a *category mismatch*. |
| 113 | A2 predicate-flip + A3 off-graph fabrication empirically verified                                  | First measured Laplacian-side wins. v2.1 trained restriction maps distinguish predicate-flip with 9–125× V ratios on synthetic data even though training only sampled tail-perturbation negatives. A3 caught structurally via OOV signal. |
| 114 | v2.x ROC bench on seed_long_paragraphs (overall AUC 0.948) + λ-calibration finding              | **Third documented falsification + first real-corpus ROC.** Per-class AUC: A1 1.000 / A2 1.000 / A3 1.000 / A4 0.801 (overall 0.948 — clears P1 target). KEY FINDING: v2.2's default λ=0.05 (calibrated on a 4-fact toy) was 38× too small for naturalistic-corpus scale; auto-calibrating from corpus statistics flipped A4 from 0.36 → 0.80. |
| 115 | Draft v0 of the arXiv working note (~3000 words)                                                  | Pre-arXiv working note positioning SUM's sheaf-Laplacian detector inside the categorical-AI conversation. Cites Gebhart 2023 / Hansen-Ghrist 2019 / Tull-Kleiner-Smithe 2023; honest attribution of mathematics to the cited literature; SUM's contribution framed as the cryptographic substrate + λ-calibration finding + empirical bench results. |
| 116 | Path 2 real-LLM bench + sum render User-Agent fix                                                 | 32 real LLM calls via the hosted Worker; 16/16 docs show stress-induced V increase (mean Δ +176.9). **HONEST INTERPRETATION:** v2.x measures FACT-SET DEVIATION, not "hallucination quality" — a faithful summary that shrinks the fact set scores high V too. Distinguishing requires labels. **Bonus: real production bug fix** — Cloudflare in front of hosted Workers was rejecting Python urllib's default UA with HTTP 403; sum render --use-worker now sends `sum-cli/<version>` UA. |
| 117 | Audit-log streaming primitive (`SUM_AUDIT_LOG`, `sum.audit_log.v1`)                              | **Path 3 / compliance foundation.** Regime-agnostic substrate: any compliance regime (GDPR, HIPAA, EU AI Act) implements its own validator on top by tailing JSONL. Fail-open semantics; concurrent-append serialisation; cross-referenceable across attest/verify/render via shared axiom_count + state_integer_digits. ✅ production. |

**Counts at HEAD: 150 features in FEATURE_CATALOG (131 production, 18
scaffolded, 1 designed).** Manifest + self-attestation drift gates
green. Cross-runtime K-matrix + A-matrix locked. Release machinery
green. v0.5.0 live on PyPI; `pip install sum-engine==0.5.0` and
`pip install 'sum-engine[research]==0.5.0'` both verified end-to-end.

---

## Open queue (priority-ordered)

The 2026-05-01-morning handover's open queue items (A PyPI cut /
B sum render CLI / C SUMequities portfolio fetch) were either closed
during this arc or remain explicitly out-of-scope. The new open queue:

### A. arXiv submission — pre-circulation review then submit

**Status:** draft v0 at `docs/arxiv/sheaf-detector-note-v0.md`.
3011 words; needs tightening to 2000–2500 for arXiv, plus pre-
circulation review on the ACT discourse / Topos Institute mailing
list / Quantinuum DisCoCat forum. **Operator decision:** when to
circulate; when to submit. The arXiv ID for cs.AI primary / math.CT
secondary is the natural cross-listing.

### B. v3 — receipt-weighted detector with harmonic-extension boundary

**Status:** specified in `docs/SHEAF_HALLUCINATION_DETECTOR.md` §3.4
+ §5.4. Not yet implemented. The SUM-specific extension — uses
Hansen-Ghrist 2019 Proposition 4.1 / Theorem 4.5 to make trusted-
issuer renders the *boundary* $B$ of the harmonic-extension problem,
untrusted renders the interior, and the boundary-value solution the
"most-consistent interpolation." Doesn't replicate elsewhere because
no other system has cross-runtime-verified render receipts to seed
the boundary.

### C. v4 — fact-set deviation vs hallucination-quality disambiguation

**Status:** open problem surfaced by PR #116. Current v2.x detector
flags ANY fact-set drift (faithful summary that shrinks the set OR
hallucination that adds wrong claims). Distinguishing requires
NLI-entailment per-rendered-triple from the source — probably a
v4 direction combining v2.x's per-rendered-triple V with NLI
entailment scores. Spec needs to land before code.

### D. Path 3 expansion — per-regime validators + multi-source connectors

**Status:** PR #117 shipped the regime-agnostic substrate. The
follow-ups are documented in `docs/AUDIT_LOG_FORMAT.md` "Use as a
compliance primitive" section: GDPR-Art-30, HIPAA-164.514, EU AI
Act Annex IV, internal-forensics validators. Each is a small
(~50–100 LOC) downstream consumer of the audit log.

Multi-source connectors (`sum attest --source wikidata://Q937`)
are a separate sub-direction — introduces external HTTP deps so
should be designed carefully (caching, offline mode, source-URI
provenance chains).

### Deferred (unchanged from prior handover)

- **Item C of 2026-04-30 handover:** SUMequities portfolio fetch of
  `meta/repo_manifest.json`. Cross-repo, deferred to a separate
  session per CLAUDE.md.
- **Audio + image-OCR adapters.** Deferred indefinitely; YAGNI
  per the 2026-04-30 audit.

---

## Decisions that aren't obvious from the code

These are the load-bearing judgment calls made this session-block.

### Three pinned-in-code falsifications, each surfacing a different design lesson

1. **PR #107: v1 disconnected-graph blindness.** Synthetic bench used
   connected graphs; real prose tends to produce sparse / disconnected
   fact-sets. v1 *cannot* detect density-dropout when both endpoints
   of the dropped edge are missing.
2. **PR #111: v2.1 presence-cochain inheritance.** Learned restriction
   maps don't help — training amplifies *present* entities, not
   *absent* ones. The fix had to be in cochain construction, not
   restriction maps.
3. **PR #114: v2.2 default λ at corpus scale.** λ=0.05 calibrated on
   a 4-fact toy graph is 38× too small for naturalistic corpora.
   Auto-calibrate from corpus statistics: `λ = mean(V_lap / |E|)`.

The discipline pattern: when a hypothesis falsifies, **pin the
falsification in code** rather than retry-until-it-works. Future
work that touches the same code path immediately surfaces the
re-introduced flaw via the pinned regression test. This pattern is
why the truthfulness contract held across 13 research-arc PRs.

### v2.x measures fact-set deviation, NOT hallucination quality

Surfaced by PR #116. A faithful summary that shrinks the fact set
to a coherent subset scores high V; a hallucinating summary that
adds wrong claims also scores high V. **They are different things.**
v2.x doesn't distinguish them without labels. This is the bounded
claim that lands in the arXiv note's §6 — *signal-to-deviation* is
honest; *signal-to-hallucination* is over-claim until labels are
introduced.

The v4 direction (per-rendered-triple V + NLI entailment from the
source) is the natural fix when prioritised.

### The Laplacian quadratic form is fundamentally a cross-edge-agreement measure

Surfaced analytically before PR #112. $x^T L_F x = \|\delta x\|^2$
sums *per-edge residuals*; it is structurally a measure of
cross-edge agreement under the restriction maps. It cannot detect
"facts missing entirely" by design — that's a presence-pattern issue
that needs a presence-pattern fix (the orthogonal deficit term).
v1's disconnected-graph blindspot was a *category mismatch*: trying
to use a relation-agreement signal to detect entity-dropout.

This realization changed how the arXiv note positions v2.x: it's
**not a single-signal detector**; it's a composition of
orthogonal signals (Laplacian for relation-aware drift; deficit
for presence-pattern issues; OOV for off-graph fabrication; per-
rendered-triple V for predicate-flip).

### Audit log = regime-agnostic substrate by design

The Path 3 design choice (PR #117). Inverting this — building per-
regime hooks into the CLI itself — would couple the trust loop to
specific compliance frameworks. The audit log records *what
happened* verbatim; per-regime validators are downstream consumers
that tail the JSONL file. This separation lets SUM ship one
substrate that supports any regime without committing to specific
ones.

---

## Gotchas the future-you should not re-learn

1. **Cloudflare-fronted hosted Workers reject Python's default urllib UA**
   with HTTP 403 / error 1010. Set
   `User-Agent: sum-cli/<version> (+...)` or similar identifier.
   Fixed in PR #116; pinned in `_post_render_to_worker` and the
   bench script.

2. **`sum render --use-worker` only triggers an LLM call when at
   least one non-density slider axis is non-neutral.** Neutral
   sliders → canonical-deterministic path → no LLM call →
   `model: "canonical-deterministic-v0"`. Path 2 bench had to set
   length=0.7 / formality=0.1 to get LLM-conditioned renders.

3. **The Laplacian quadratic form goes DOWN under LLM stress.**
   Counter-intuitive but explained: when the LLM omits source
   entities entirely, the corresponding edges drop to (0,0),
   zeroing per-edge contributions. The combined detector recovers
   via the deficit term — **same lesson** as the disconnected-graph
   case. This is *consistent behaviour*, not a regression.

4. **v2.x ROC numbers depend on auto-calibrated λ.** Hardcoded λ=0.05
   from the toy 4-fact calibration is 38× too small at corpus
   scale. Always auto-calibrate from corpus statistics before
   reporting numbers.

5. **The contrastive sheaf-embedding training only samples
   tail-perturbation negatives**, but the trained restriction
   maps still distinguish predicate-flips with 9–125× ratios on
   synthetic data (PR #113). The contrastive loss generalises
   beyond its sampling distribution. Don't assume a missed
   negative class means a missed detection class.

6. **The hosted Worker URL alias `sum.ototao.com` doesn't resolve
   yet** — README has stale references. Use
   `sum-demo.ototao.workers.dev`. Worth a doc-only PR to update
   the README's URL references.

7. **Wheel + sdist non-determinism + TestPyPI skip-existing**
   (carried over from v0.4.0 → v0.4.1 publish saga, still relevant
   for any future re-publish on an existing tag).

8. **The `pypi` environment has a deployment-protection rule.**
   Production publishes pause at `Publish to PyPI (production)` for
   manual reviewer approval. Reviewer is the user; the agent does
   not auto-approve.

---

## What "polished" looks like at the close of this arc

| Surface | Polish state at HEAD |
|---|---|
| Cold-install onboarding | ✅ |
| `pip install sum-engine` | ✅ v0.5.0 live |
| `pip install 'sum-engine[research]'` | ✅ numpy + scipy gated cleanly |
| Bidirectional 3×3 grid (CLI/MCP/HTTP × attest/verify/render) | ✅ fully populated |
| Cross-runtime trust triangle | ✅ K1–K4 + A1–A6 every PR |
| §2.5 LLM closure | ✅ vendor-independent across 3 model families |
| Self-attestation pipeline | ✅ all 5 canonical docs round-trip |
| Sheaf-Laplacian v1 / v2.1 / v2.2 | ✅ research-grade, behind `[research]` extras |
| Synthetic ROC AUC 0.948 | ✅ reproducible |
| Real-LLM bench (Path 2) | ✅ 32 calls, 16/16 stress-induced V increase |
| **Audit-log compliance primitive** | ✅ **production**, regime-agnostic |
| arXiv working note | 📝 draft v0; needs tightening + pre-circulation |
| v3 (receipt-weighted, harmonic-extension) | ⏸ specified, not implemented |
| v4 (fact-set vs hallucination-quality) | ⏸ open problem, no spec yet |
| Per-regime validators (GDPR, HIPAA, etc.) | ⏸ illustrative sketches in `docs/AUDIT_LOG_FORMAT.md` |
| Multi-source connectors (`--source wikidata://...`) | ⏸ design pending |

---

## Files to read in order on your first turn

1. This file
2. [`CHANGELOG.md`](../CHANGELOG.md) — `[0.5.0]` is the live PyPI release; `[Unreleased]` is empty until next work lands
3. [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — proved/measured/designed discipline; non-negotiable
4. [`docs/SHEAF_HALLUCINATION_DETECTOR.md`](SHEAF_HALLUCINATION_DETECTOR.md) — full v1/v2.0/v2.1/v2.2/v3 spec; the three pinned falsifications and the deeper analytical findings
5. [`docs/AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md) — `sum.audit_log.v1` wire spec + four illustrative regime-validator sketches
6. [`docs/arxiv/sheaf-detector-note-v0.md`](arxiv/sheaf-detector-note-v0.md) — pre-arXiv working note (3011 words; needs tightening)
7. [`fixtures/bench_receipts/sheaf_v2_roc_seed_long_paragraphs_2026-05-01.json`](../fixtures/bench_receipts/sheaf_v2_roc_seed_long_paragraphs_2026-05-01.json) — Path 1 synthetic ROC receipt (overall AUC 0.948)
8. [`fixtures/bench_receipts/sheaf_v2_path2_llm_bench_2026-05-01.json`](../fixtures/bench_receipts/sheaf_v2_path2_llm_bench_2026-05-01.json) — Path 2 real-LLM bench receipt (32 LLM calls; 16/16 stress signal)
9. [`docs/FEATURE_CATALOG.md`](FEATURE_CATALOG.md) — entries 144–150 are this arc's new surface
10. [`docs/SESSION_HANDOVER_2026-05-01.md`](SESSION_HANDOVER_2026-05-01.md) — prior handover (PRs #97–#102, the v0.4.0 → v0.4.1 publish-path arc)
11. [`docs/SESSION_HANDOVER_2026-04-30.md`](SESSION_HANDOVER_2026-04-30.md) — earlier handover (PRs #83–#95, the v0.3 → v0.4 substrate arc)

If you only have time for one: read this file. The rest are the
substrate.

— end of handover
