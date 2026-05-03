# Session handover — 2026-05-02 v3 / compliance / F3-diagnostic arc

**For a memory-less instance picking up cold.** Read this first. It bootstraps you into the substrate state at HEAD `bb7957d` after PRs #119–#125. Predecessor handover is [`docs/SESSION_HANDOVER_2026-05-01_research_arc.md`](SESSION_HANDOVER_2026-05-01_research_arc.md) — read it for the v0.5.0 release context + the initial sheaf-Laplacian arc (PRs #103–#117).

## What landed this arc — seven PRs, one bug caught, one structural finding

| PR | Title | Net effect |
|---|---|---|
| #119 | Audit-log gap closure | Substrate hardened: signed-bundle / multi-process / worker-mode / empty-string paths now under contract |
| #120 | EU AI Act Article 12 validator | First per-regime compliance utility on top of `sum.audit_log.v1`; regime-agnostic `ValidationReport` shape |
| #121 | v3 receipt-weighted sheaf-Laplacian | Math primitive + `weights_from_receipts` mapping trust-loop receipts to per-edge weights |
| #122 | v3.1 harmonic-extension boundary inference | Hansen-Ghrist Prop 4.1 / Thm 4.5; `boundary_deviation` + `boundary_from_weights` |
| #123 | Audit-tightening pass + λ double-counting fix | Independent audit caught **a real bug** in v3 combined formula. Bug fix and the test that surfaced it shipped together. |
| #124 | v3 corpus-scale ROC bench | F1 MARGINAL, F2 PASS, F3 FAIL on `seed_long_paragraphs` |
| #125 | F3 structural-fail diagnostic | 8/8 cells FAIL → F3 is *structural* in v3.1's design, not parametric. v3.2 must redesign. `bench_digest` substrate introduced. |

Tests: 1377 → 1450 (+73). `make xruntime` K1–K4 + adversarial A1–A6 still green. Meta drift gates green.

## Two findings that distinguish this arc

### 1. The audit caught a real bug

PR #123's audit found v3's `combined_detector_score_v3` was computing `λ² · deficit²` instead of `λ · deficit²` because v2.2's `v_deficit` field is already `presence_weight · deficit²` (post-λ-weighting). The fix is one line:

```python
# was: "v_combined_v3": v_laplacian_w + lambda_deficit * base["v_deficit"]
"v_combined_v3": v_laplacian_w + base["v_deficit"]
```

The test that caught it (`test_combined_v3_lambda_wiring_with_nonzero_deficit`) is now part of the contract. The pre-existing tests couldn't have caught this because they only exercised λ on clean renders (deficit = 0). **This is what "tests themselves need checking" looks like in practice.**

### 2. F3 STRUCTURAL FAIL — synthetic ≠ corpus

PR #122's v3.1 boundary deviation passed every synthetic utility test (H6–H15, including the headline H12 "interior-tampering increases deviation"). PR #124's corpus-scale ROC then surfaced F3 FAIL: trusted-mean AUC ≈ 0.50. PR #125's diagnostic refuted three competing hypotheses (graph too small / cochain produces zero-vectors / random partition too harsh) by running them as a 2×2×2 sweep — **all 8 cells FAIL**, including the all-three-axes-flipped cell.

Mathematical reason (now in [`docs/SHEAF_HALLUCINATION_DETECTOR.md`](SHEAF_HALLUCINATION_DETECTOR.md) §3.4.3): when a perturbation targets a trusted-edge vertex, the cochain change is at boundary positions; the harmonic extension formula `x_I^* = -L_II^{-1} L_IB x_B` recomputes the interior from the new boundary; the **actual** interior is unchanged → deviation ties between clean and perturbed by mathematical necessity. v3.1 has a structural blind spot for boundary perturbations.

**Implication for v3.2 (Priority 9 in NEXT_SESSION_PLAYBOOK):** v3.1 cannot be a standalone detector. Pair with a complementary boundary signal; redesign cochain to encode render information into the interior. Knob-tuning won't fix this.

## New substrate this arc introduced

### `sum.audit_log.v1` (PR #117 + #119)

Regime-agnostic JSONL audit stream emitted by every CLI op (`sum attest` / `sum verify` / `sum render`) when `SUM_AUDIT_LOG` is set. Fail-open: an unwritable destination doesn't break the trust loop. Wire spec: [`docs/AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md). Contract pinned by 17 tests in `Tests/test_audit_log.py` including multi-process O_APPEND atomicity (8 procs × 20 emits = 160 rows).

### `sum.compliance_report.v1` (PR #120)

Regime-agnostic ValidationReport shape. Future per-regime validators share consumers without per-regime adapters. Frozen dataclass with `Violation` records (rule_id, row_index, operation, message, row). The `sum compliance check --regime <id> --audit-log <path>` CLI verb wraps any validator and exits 0 iff `ok=true`, 1 otherwise (pipe-friendly for CI gates).

### `bench_digest` (PR #125)

JCS-canonical SHA-256 over a quantized research-bench payload (AUCs to 3 decimals; diagnostic floats to 4). Quantization absorbs the ~±0.02 LAPACK jitter `np.linalg.lstsq` introduces. Three intended uses, each a separate proof opportunity:

1. **Reproducibility canary** — same machine, same code → same digest. Currently exercised by `test_v3_1_f3_diagnostic_digest_is_quantization_stable` (two consecutive in-process runs → identical digest).
2. **Cross-runtime witness** — when a future Node/browser port reproduces these AUCs, the matching digest is the K-style portability proof. Not yet measured (no Node port for these detectors).
3. **Signable bench artifact** — Ed25519-sign with the project's existing JWKS keys; arXiv preprint can cite the digest, readers re-run and verify. Not yet exercised.

Same trust alphabet as `render_receipt.v1` — eats the project's own dog food (uses [`sum_engine_internal/infrastructure/jcs.py`](../sum_engine_internal/infrastructure/jcs.py), the project's RFC 8785 implementation).

## Pinned-in-code falsifications added this arc

A pattern from prior arcs: when an empirical finding contradicts a hypothesis, pin it in code rather than retry-until-success. Three new pins this arc:

1. **`test_v3_1_does_NOT_close_disconnected_graph_blindspot_with_presence_cochains`** (carried from prior arc; still pins the v2.1 falsification).
2. **`test_boundary_deviation_with_identity_maps_is_weight_invariant_on_chain_graphs`** (PR #122). Surfaced empirically: with a single bridge edge, the harmonic extension is weight-invariant for any positive weights, even on trained sheaves. Math reason: `x_I = -r · M(r)^{-1} (B x_B)` cancels for rank-1 B. Documented as a property of the toy graph, with a partner test (`test_boundary_deviation_with_weights_visible_with_multiple_bridge_edges`) showing weights *do* matter in non-chain topologies.
3. **F3 structural blind spot** (PR #125). Documented in [`docs/SHEAF_HALLUCINATION_DETECTOR.md`](SHEAF_HALLUCINATION_DETECTOR.md) §3.4.3, not pinned as a single test (it's a *negative* property — boundary perturbations CANNOT move v3.1's deviation). The diagnostic harness itself is the witness.

## Open queue at session-close

See [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) for full priority context. Three new priorities added 2026-05-02:

- **Priority 9 — v3.2 detector redesign.** Closes F3. Pair boundary deviation with a complementary boundary signal; redesign cochain to encode render into interior. Re-run [`scripts/research/sheaf_v3_1_f3_diagnostic.py`](../scripts/research/sheaf_v3_1_f3_diagnostic.py) against the v3.2 design; PASS criterion = trusted-mean AUC ≥ 0.55 on the PR #124 baseline cell.
- **Priority 10 — arXiv preprint v0.1.** `docs/arxiv/sheaf-detector-note-v0.md` exists as v0; needs to fold in v3, v3.1, F1/F2/F3 verdicts, F3 structural finding, and surface `bench_digest` as a reproducibility primitive in the methods.
- **Priority 11 — Second per-regime compliance validator.** Pick GDPR Art 30 OR HIPAA 164.514. Reuses `sum.compliance_report.v1` shape (zero adapter cost). Demonstrates the regime-agnostic claim earns a second instance.

Carried-over (still open from prior handovers):
- A2 predicate-flip is at chance for both v2.2 and v3 — known v2.x weakness; needs explicit predicate-perturbation negative sampling in training. Could be a v3.3 or a v2.3 patch.
- v4 (fact-set deviation vs hallucination-quality disambiguation) — still gated on labeled hallucination data.
- Audio + image-OCR adapters — not progressed.
- Wikidata QID SPARQL disambiguation, `sha256_128_v2` activation, browser-bench perf numbers — Priorities 3, 4, 2 in NEXT_SESSION_PLAYBOOK; unchanged in scope.

## Files and where they live

### Documentation updated this arc

- [`CLAUDE.md`](../CLAUDE.md) — read-first list item 0 now points at this handover.
- [`docs/SHEAF_HALLUCINATION_DETECTOR.md`](SHEAF_HALLUCINATION_DETECTOR.md) — §3.4 (v3 implementation status), §3.4.1 (v3.1 implementation status with H6–H15 falsifiable predictions), §3.4.2 (PR #124 corpus bench results — F1 MARGINAL, F2 PASS, F3 FAIL, A4 standout), §3.4.3 (PR #125 diagnostic — F3 STRUCTURAL with v3.2 redesign direction).
- [`docs/COMPLIANCE_EU_AI_ACT_ARTICLE_12.md`](COMPLIANCE_EU_AI_ACT_ARTICLE_12.md) — wire spec; six rules R1–R6; "what this validator does NOT pin" section.
- [`docs/AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md) — `sum.audit_log.v1` wire spec; pointer to the EU AI Act validator added this arc.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — v1.5.0; new §1.9 (audit-log substrate, mechanically pinned), §1.10 (compliance report shape), §2.9 (sheaf-Laplacian detector measurements with F3 STRUCTURAL FAIL named explicitly), §2.10 (`bench_digest` reproducibility under quantization).
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) — state-at-head refreshed; "Closed since" section adds the v0.5.0 / audit-log / compliance / sheaf-Laplacian arc; Priorities 9, 10, 11 added.
- [`docs/FEATURE_CATALOG.md`](FEATURE_CATALOG.md) — audit-log entry's expected pass count 11 → 17.
- [`README.md`](../README.md) — stale `sum.ototao.com` pointers (×2) replaced with `sum-demo.ototao.workers.dev` (carried-over deferred from prior handover; closed this arc).
- [`CHANGELOG.md`](../CHANGELOG.md) — `[Unreleased]` carries entries for all seven PRs.

### Code added this arc

- [`sum_cli/audit_log.py`](../sum_cli/audit_log.py) — `emit_audit_event` (from prior arc; hardened by PR #119).
- [`sum_engine_internal/compliance/`](../sum_engine_internal/compliance/) — new package. `report.py` (regime-agnostic shape), `eu_ai_act_article_12.py` (R1–R6 rules), `__init__.py`.
- [`sum_engine_internal/research/sheaf_laplacian_v3.py`](../sum_engine_internal/research/sheaf_laplacian_v3.py) — weighted Laplacian, weights_from_receipts, harmonic_extension, boundary_deviation, boundary_from_weights, combined_detector_score_v3.
- [`scripts/research/sheaf_v3_roc_bench.py`](../scripts/research/sheaf_v3_roc_bench.py) — corpus-scale ROC bench (`sum.sheaf_v3_roc_bench.v1`).
- [`scripts/research/sheaf_v3_1_f3_diagnostic.py`](../scripts/research/sheaf_v3_1_f3_diagnostic.py) — F3 hypothesis-sweep diagnostic (`sum.sheaf_v3_1_f3_diagnostic.v1`); introduces `bench_digest` field.
- [`Tests/compliance/test_eu_ai_act_article_12.py`](../Tests/compliance/test_eu_ai_act_article_12.py) — 27 tests including end-to-end through the real CLI.
- [`Tests/research/test_sheaf_laplacian_v3.py`](../Tests/research/test_sheaf_laplacian_v3.py) — 35 tests across v3 + v3.1 (after PR #123 tightening).
- [`Tests/research/test_sheaf_v3_roc_bench.py`](../Tests/research/test_sheaf_v3_roc_bench.py) — smoke + F2 PASS pin + A4 floor pin.
- [`Tests/research/test_sheaf_v3_1_f3_diagnostic.py`](../Tests/research/test_sheaf_v3_1_f3_diagnostic.py) — smoke + digest-quantization-stability pin.
- [`fixtures/bench_receipts/v3_roc_bench_2026-05-02.json`](../fixtures/bench_receipts/v3_roc_bench_2026-05-02.json) — PR #124 receipt.
- [`fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-02.json`](../fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-02.json) — PR #125 receipt; `bench_digest = 244423192cd88bb2864a9bf15a1aaf69b40a73f42bbd4082ba7aeb90e3ff5308`.

## Discipline notes for the next instance

- **Truth-first labeling is load-bearing.** PR #124's F1 verdict was MARGINAL, not PASS. PR #125's load_bearing_hypothesis was "none", not the optimistic "A". The arXiv note inherits these labels honestly.
- **Synthetic-data utility tests are necessary but not sufficient.** v3.1's H12 passed; F3 corpus-scale FAILED. The corpus-scale step is non-negotiable for category-defining software.
- **The audit pass that caught the v3 λ bug should run again next session-block.** Independent eyes on test files added this arc would be the natural follow-up. The pattern shipped (PR #123); reuse it.
- **Pinning falsifications in code beats retry-until-success.** Three new pins this arc preserve negative findings as part of the contract. The test names start with `test_..._does_NOT_...` or include "is_weight_invariant" — they read as load-bearing properties, not failed experiments.
- **`bench_digest` is novel substrate, not just a hash.** It's the project's first cryptographically-verifiable research artifact. Cite it. Build on it.

## Last action of this arc

Documentation refresh + this handover. After this PR merges, the substrate state at HEAD is consistent end-to-end:
- Every doc on the verify list has been read and is either current or updated this PR.
- `meta/` self-attestation + repo-manifest drift gates pass.
- Full pytest + xruntime + adversarial all green.
- `CLAUDE.md` item 0 points here.

The next instance should land on this file first, then proceed to whichever priority (9 v3.2 / 10 arXiv / 11 second compliance regime) it picks up.
