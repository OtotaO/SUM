# PRODUCT_DELIBERATION_2026-05-14.md

**Status: planning artifact, not a directive. No code changes follow from this document. The user's standing direction (memory `project_direction_2026-05-11`: wait + dogfood) remains in force.** This file exists so neither this engine-session nor the companion portfolio-session loses the deliberation across context compaction.

## Origin

Cross-session exchange on 2026-05-14 between the engine-repo Claude Code session and the portfolio (SUMequities) Claude Code session. The portfolio session proposed three strategic options for SUM's next 6–12 months and a unified framing. The engine session pushed back on missing context (grant timing, ICP feasibility, dogfood signal, services path). Both sessions converged on: **defer the strategic commitment for 4–12 weeks while grant signals, dogfood findings, and user signal land**, and use the deferral window structurally rather than passively.

Companion (portfolio) session's offer: write the planning artifact into this repo because engine-direction questions belong here, portfolio-curation questions belong in the SUMequities session. This file is that artifact.

## The unified framing — The Substrate Hub

**Core abstraction.** SUM is a *content-addressable semantic substrate* where every transformation between forms of the content is cryptographically verifiable. A single object in the substrate is the **CanonicalBundle** (or its modality-pluggable generalization). Around each bundle live:

- **Renderings** in various forms (prose tome, code, structured data, math, scene-graph-from-image, audio transcript)
- **Sliders** parameterizing transformations within a form
- **Bridges** parameterizing transformations across forms
- **Receipts** (`sum.transform_receipt.v1`) attesting to every transformation
- **Provenance graph** linking it all (DAG of bundles connected by signed transforms)

**Every dream item maps in:**

| Dream | Generalized realization |
|---|---|
| Reversible bidirectional | any-form → substrate → any-form |
| Sliders with values | per-modality slider configurations + cross-modality bridges |
| Verifiable / provable | receipt per transform; whole provenance DAG verifiable |
| Omni-format | modality registry; new modalities are pluggable |
| Universal compression | content-addressed substrate IS the compression; cross-modality is generalized transcoding with bounded fidelity loss; the slider is the rate-distortion knob |
| Everyday tool for writers | prose-modality specialization with workspace UI |
| De facto standard | receipt format IS the standard people adopt to participate |

**The honest universal-compression scoping:** *semantic compression of any modality with extractable structure, with cryptographically-attested bounded fact-loss measured by the modality's round-trip metric.* That covers nearly all human knowledge artifacts, excludes only random bytes (which zstd already handles optimally). Tighter than the original dream; the tightening is intellectually honest.

## The three options on the table

### Option A — Writer's Verifiable Knowledge Instrument

Document workspace where the Slider is the primary primitive and receipts are ambient. Three workflows the substrate already supports as parts:

- **Compress:** drag N PDFs in → triple-extract → merged graph → density-slider down → synthesis tome
- **Decompress:** outline / tags → length-slider up → drafted long-form
- **Reshape:** existing draft → move audience/formality/perspective → live re-renderings with NLI fact-preservation meter

Every transformation emits `sum.transform_receipt.v1`. Document history is a Merkle DAG of bundles + transforms (T4 source_chain_hash substrate). Publishing = ShareableRender → reader verifies offline against `/.well-known/jwks.json`.

**Wedge ICP under feasibility constraint:** journalists, newsletter writers, academic-survey writers — *not* legal/medical/pharma (those need enterprise sales motion solo founders don't have). The narrower wedge is bottom-up and reachable by one person.

**What's built vs. not:**
- ✓ Substrate (sliders, receipts, NLI audit, T2/T3/T6, ShareableRender, three-runtime verifiers, MCP server)
- ✗ Document model (persistence, versioning, multi-doc workspace)
- ✗ Real-time slider UX with live preview (current substrate is batch-oriented)
- ✗ Local-first storage strategy
- ✗ Identity model
- ✗ Export targets (Markdown ✓, PDF/EPUB/HTML render paths needed)
- ✗ Onboarding — "first 60 seconds" experience

### Option B — AI-Text Provenance Standard

Push `sum.render_receipt.v1` and `sum.transform_receipt.v1` toward formal standardization. Realistic surfaces:

- **C2PA text profile** — strongest play; $10M+ member investment; they need a text answer. Timeline: 18–30 months from first contact to v1 profile shipped.
- **W3C VC 2.0 profile** — already conformant (`eddsa-jcs-2022`); slower cycle.
- **IETF I-D** — faster cycle, lower prestige than C2PA.
- **Regulatory hooks** — EU AI Act Art 12, FTC AI disclosure rules, UK AI Bill. Standards bodies move when regulators force the issue.

**Verdict on B alone:** too risky as primary thrust (standards work without adoption is vaporware; adoption without standards is captive). Excellent as a thread running underneath A.

### Option C — Omni-Modal Canonical-Bundle Substrate

Per-modality tractability:

- **Prose** — ✓ done. The reference modality.
- **Structured data** — trivial extension. Rows are already triples. Weeks of work.
- **Code** — tractable but unidirectional. AST → triples works; triples → identical code doesn't. Useful for spec ↔ code documentation.
- **Math** — high-value, hard, possibly research. Lean/Coq integration is a year+.
- **Images** — partial. Vision models give scene-graph triples; round-trip to identical bits is lossy. Useful for image-grounded prose.
- **Audio** — similar to images. ASR + diarization triples; TTS round-trip is lossy on prosody.
- **Arbitrary binary** — honest scope: "compress any data with extractable semantic structure," not random bytes.

**Verdict on C alone:** each modality is months of focused work. Generalized substrate refactor (modality-pluggable CanonicalBundle) is the real win; modalities are then opportunistic.

### Comparison

| Axis | A (writer tool) | B (standard) | C (omni-modal) |
|---|---|---|---|
| Time-to-first-visible-value | 6–12 weeks | 12–30 months | per-modality, 3–12 months |
| Reuses existing substrate | 100% | 100% | ~70%, needs generalization layer |
| Risk of zero adoption | medium | high | medium |
| Maps to original dream | "writer tool" piece | "de facto standard" piece | "omni-format" + "universal compression" |
| Creates demand for the others | yes | weak | no |
| Cost of being wrong | moderate (lost time) | high (years lost) | high (built unused modalities) |
| What it asks of the user | product/UX work | diplomatic/political work | sustained engineering |

**A is the only one that creates demand for B and C.** Standards bodies want anchor users; modality requests come from users. A is the wedge that earns B and C the right to exist.

## Why we're not committing right now

Four pieces of missing context that make today the wrong day to choose:

1. **Grant timing is a strategy parameter, not a footnote.** Six grants out, ~$225K aggregate, decisions rolling May–August 2026. The "ship MVP in 8 weeks, commit to product-company shape in v0.7" path commits BEFORE any grant has decided. Different grant outcomes imply different optimal paths.

2. **No dogfood signal yet.** Per memory `user_origin_dream`, the user has not actually used SUM on his own writing in a long time. The cheapest source of "do writers want this" data is the user himself. Committing to product-company shape without that signal is committing without information you'll have this weekend if you dogfood deliberately.

3. **Wedge ICP feasibility wasn't checked.** "Regulated-domain writers" is uncontested AND requires enterprise sales motion that doesn't fit solo founder in Brooklyn. Narrower wedge (journalists / newsletter / academic survey) is bottom-up and reachable. Don't pick the wedge until you've confirmed you can reach it.

4. **The services / consulting path wasn't surfaced.** EU AI Act Art 12 + FTC AI disclosure + UK AI Bill window is 12–18 months. One or two enterprise engagements for cryptographically-attested provenance of AI-assisted content would fund runway, validate the C2PA-adoption story with real-customer data (which is *the* thing C2PA actually moves on), and skip the product-UX burden. Different shape than product or spec-author.

## The decision

**Defer the strategic commitment for 4–12 weeks.** Use the deferral window structurally:

### 1. Dogfood with a hypothesis, not just "see what's missing"

Pick the dogfood task deliberately. Different tasks teach different things. Candidates:

- **Writer→reader trust loop test.** *"If I distill three research papers into a 2-page brief and ship it, do I trust the receipt enough to publish without re-reading every source?"*
- **Reshape loop test.** *"If I take my draft newsletter and run it through three audience sliders, do any of the variants beat the version I'd have written by hand?"*
- **Compose test.** *"If I merge bundles from two existing pieces and re-render at low density, does the synthesis say anything I didn't already say?"*

Pick one with a falsifiable outcome. The deliberateness is the point.

### 2. Scope limits to name before dogfooding

The live Worker at `sum-demo.ototao.workers.dev` is a stripped-down render path — no multi-doc, no T3 compose UI, no ShareableRender UI. Dogfooding on the live Worker tests the per-doc reshape workflow only. CLI dogfooding via `sum transform apply slider` (now wired Python-side per PR #221) exercises a different surface but still no multi-doc workspace. Name the partial-coverage limit so any negative result isn't over-generalized.

### 3. The grant-outcome decision tree

Pre-decide so you don't re-deliberate when signal lands:

- **Foresight lands (~$60K, end of May):** substrate-research runway; defer product-company; double down on §4.7.x research arms or omni-modal preliminary work
- **Cloudflare BOOTSTRAPPED lands:** different runway shape (likely infra credits + cash); product-company viable but research arm also fundable
- **NLnet lands (~June 1–8):** mid-term runway; product-company on grant alone becomes viable
- **LTFF lands (July–August):** year of runway; product-company viable
- **SFF Speculation lands:** small but signal-positive; not enough alone
- **OpenAI Cybersec lands:** specific-use-case runway; channels the work toward security applications
- **Multiple land:** optionality; consider hiring one product engineer while the user stays on research
- **None land by August:** product-company pivot for cashflow OR services/consulting OR contract day-work

Each branch has a different next-action.

### 4. The services path — add to the live option set with a trap warning

EU AI Act Art 12 + FTC + UK AI Bill create regulatory demand within 12–18 months for cryptographically-attested provenance of AI-assisted content. One or two enterprise consulting engagements would:

- Generate cashflow without product UX burden
- Validate the C2PA-adoption story with real-customer data
- Sharpen the wedge ICP empirically

**Trap warning:** services work has high inertia. Many founders never escape services to product. If pursued, time-box: max 2 concurrent engagements, max 6 months, with the product-pivot trigger written into the contracts as a planning artifact. Without that pre-commitment, services revenue compounds and the product-pivot never happens.

## Re-entry conditions

Return to this document when any of the following triggers:

- First grant signal lands (acceptance or rejection)
- Dogfood session completes with a falsifiable outcome reported
- An unexpected pull from a named buyer (enterprise inquiry, journalist user feedback, C2PA inbound)
- The August 2026 deadline (if no grant has landed by then, the deferral is up regardless)

The job of THIS document is to make sure the deliberation that the engine session and portfolio session did on 2026-05-14 is available to whichever session re-enters the decision. Don't re-deliberate from scratch.

## What is explicitly NOT a decision in this document

- That SUM will become a product company
- That A is the right wedge
- That the unified framing is the right architectural commitment
- That services is the right path
- That bench-hardening T1–T5 should be implemented now

This document captures the analysis. The decisions wait for signal.

## Pointers

- Memory `project_direction_2026-05-11` — standing direction (wait + dogfood); supersedes this document for default behaviour.
- Memory `user_origin_dream` — bi-directional tags↔tomes; the slider IS the dream; substrate is downstream.
- Memory `feedback_buyer_or_dream_filter` — every PR must serve named buyer / funder commitment / dream element.
- Memory `project_grant_funnel` — the six grants and their timing.
- `docs/BENCH_HARDENING_FROM_QCVV.md` — independent planning artifact for empirical-benchmark hardening; orthogonal to product/standards/modality choice.
- The companion portfolio session at `~/SUMequities/` holds the curation-side deliberation; engine-direction questions live here.
