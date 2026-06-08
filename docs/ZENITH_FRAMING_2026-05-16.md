# ZENITH_FRAMING_2026-05-16.md

**Status: planning artifact, not a directive. No code changes follow from this document.** Companion to [`PRODUCT_DELIBERATION_2026-05-14.md`](PRODUCT_DELIBERATION_2026-05-14.md). The deliberation artifact captures the *tactical* deferral (wait 4–12 weeks for grant signal); this document captures the *destination* framing the engine session and a larger model converged on while the deliberation was in force. Both artifacts are required reading before re-engaging strategic-direction questions.

## Origin

On 2026-05-16, the engine repo Claude Code session received a strategic vision from a larger model (Claude / model-class-larger-than-Sonnet, via the user). The framing was bigger than any framing previously captured in repo docs. The engine session validated three concepts from it as genuinely new and worth persisting. The companion (SUMequities portfolio) Desktop Claude session asked the engine session to persist them so future sessions don't lose the framing across context compaction.

## The destination framing

> **SUM is most valuable as the "chain of custody" layer for knowledge moving between humans, LLMs, tools, agents, and institutions. Its zenith is not "better summaries." Its zenith is auditable semantic transport.**

Reframing of what SUM IS:

- **A receipt layer for machine trust** — signed render/transform receipts that prove *this issuer rendered this tome from these triples with these slider settings and this model at this time*. Explicitly NOT a truth oracle.
- **A semantic preservation layer for human trust** — transformations across density, length, formality, audience, and perspective bounded empirically and measurably, not magically guaranteed.
- **A cross-runtime verifier layer for swarm trust** — Python, Node, browser, CLI, MCP, Worker, all participating in the same trust loop. Agents need portable evidence, not merely fluent text.
- **A governance discipline layer** — PROOF_BOUNDARY.md arbitrates claims; THREAT_MODEL.md scopes the trust surface; FEATURE_CATALOG.md ships verification recipes. A rare and valuable epistemic architecture.

Reframing of what SUM PROMISES (candidate one-sentence opener):

> **SUM lets people and agents transform knowledge without losing the ability to verify what changed, what stayed the same, who signed it, and what remains unproven.**

The four-clause structure (what changed / what stayed / who signed / what remains unproven) maps cleanly onto the project's existing proof-boundary discipline.

## Three new concepts the engine session validated

### 1. Perspective Receipts

The five slider axes today are stylistic (length / formality / audience / perspective / density). Reframing them as named **perspectives** is a genuine conceptual upgrade with the same substrate:

```
same facts, rendered for:
- novice
- expert
- regulator
- engineer
- investor
- child
- adversarial reviewer
- affected community
- machine planner
```

The substrate doesn't change. The naming changes how the slider lands ethically and practically. Stylistic sliders are about prose feel; perspective receipts are about *whose epistemic frame is being served*. The NLI audit + per-axis drift threshold already provide the fact-preservation *measurement* within a perspective shift (measured, not a same-commit-replayable guarantee — see PROOF_BOUNDARY §2.6).

**Why this is genuinely new:** the original five-axis slider can be sold as "AI writing tool with style controls." Perspective Receipts can be sold as "verifiable epistemic translation between communities." Same code, different category.

### 2. Trust Profiles

Bundle the six-regime compliance work + the receipt-verification surface into named modes:

```
sum verify --profile research
sum verify --profile legal-discovery
sum verify --profile healthcare
sum verify --profile financial-recordkeeping
sum verify --profile agent-swarm-handoff
sum verify --profile public-web-citation
```

Each profile defines what must be present before an artifact is safe for that use. (E.g., `legal-discovery` requires audit-log presence + source-chain coverage ≥ 90% + Ed25519 signature + revoked-kid clean. `agent-swarm-handoff` may have looser source-chain requirements but tighter freshness/nonce requirements.)

**Why this is genuinely new:** compliance work today lives as separate validators (EU AI Act Art 12, GDPR Art 30, HIPAA, ISO 27001, SOC 2, PCI DSS). Trust Profiles make compliance a product feature — a buyer can ask "does this verify under `legal-discovery`?" without reading six docs.

### 3. Epistemic Nutrition Label

Productize the proof-boundary discipline as a user-visible per-artifact summary:

```
Signed:               yes
Source-backed:        83%   (12 of 14 claims)
Transform-preserved:  97%   (drift below threshold)
Model-generated:      yes
Truth asserted:       no
Known ambiguity:      2 entities, 1 unsupported claim
Recommended action:   safe to reuse as attested transform,
                      not safe as factual authority
```

**Why this is genuinely new:** the proof-boundary discipline today is `docs/PROOF_BOUNDARY.md` — invisible to end users. Surfacing it as a per-artifact label makes the discipline a customer-visible moat instead of internal hygiene.

## The destination layered output for `sum verify --explain`

For humans, "verified" is too binary. For agent swarms, binary verification is too lossy. The larger model proposed a layered output:

```
Cryptographic integrity:           pass
Canonical reconstruction:          pass
Cross-runtime compatibility:       pass
Source evidence coverage:          partial
Semantic preservation:             measured, within threshold
Truth of content:                  not asserted
Known gaps:                        QID ambiguity, extractor recall,
                                   source unavailable
Recommended action:                safe to reuse as attested transform,
                                   not safe as factual authority
```

This is the operational realization of the proof-boundary discipline. It would replace "verified: true" with a structured per-dimension report. **(Shipped v0.7.0 as `sum verify --explain` → `sum.verify_explained.v1`.)**

## Where the framing overreaches relative to current constraints

The destination is right; the timing is not. The engine session pushed back on three points where the larger model's plan ignored standing constraints:

- **Grant timing.** The 10-point plan is multi-year. The user has six grant decisions pending in May–August 2026. The deliberation artifact (PRODUCT_DELIBERATION_2026-05-14.md) explicitly defers product-shape commitment 4–12 weeks. The destination framing is correct; the build-it-now timing is not.
- **Standing direction was wait + dogfood.** The user has not yet dogfooded the slider on his own writing. Building "the SUM Viewer" before dogfood signal is the auto-pivot trap.
- **"Receipt-passing for agentic swarms"** is the seductive trap. Beautiful thesis, zero users today. The MCP server already exists; agent-builders will pull the next primitive when they hit friction. Supply-ahead-of-demand is the same trap Option C (omni-modal) carries.

## The synthesis (three time-horizons)

Three correct framings at three timescales, none contradicting each other:

- **This week (1–3 weeks):** receipt-replay window check; README 30-second-funder-read pass; one deliberate dogfood session. Small, reversible, grant-promise-aligned, no product-shape commitment.
- **4–12 weeks (grant-decision window):** continue per `PRODUCT_DELIBERATION_2026-05-14.md` deferral. Let grant signals arrive. Pre-decided decision tree per branch.
- **2026–2031 zenith:** chain-of-custody for knowledge in motion. Perspective Receipts, Trust Profiles, Epistemic Nutrition Labels. `sum verify --explain` as the layered UX primitive. The SUM Viewer. Standards adoption.

## What this document is NOT

- A commitment to execute the larger model's 10-point plan.
- A change to the standing direction in `project_direction_2026-05-11`.
- A claim that any of the three new concepts will ship in any specific PR or timeline.

## What this document IS

A snapshot of strategic clarity that the engine session and a larger model agreed on, captured so future sessions can pick it up cold. The three concepts (Perspective Receipts, Trust Profiles, Epistemic Nutrition Label) survive compaction via this file plus the memory pointer `project_zenith_framing_2026-05-16`.

## Re-entry conditions

Read this document when:

- A grant decision lands and the deferral window collapses
- A dogfood session produces a falsifiable signal
- An enterprise / journalist / academic user requests one of the three named concepts
- The `sum verify --explain` v1 work begins (the layered output design is here)
- The README's "what is SUM" surface needs a rewrite (the one-sentence elevator is here)

## Pointers

- [`PRODUCT_DELIBERATION_2026-05-14.md`](PRODUCT_DELIBERATION_2026-05-14.md) — tactical deferral; supersedes this document for default behaviour.
- [`BENCH_HARDENING_FROM_QCVV.md`](BENCH_HARDENING_FROM_QCVV.md) — orthogonal empirical-benchmark hardening plan.
- [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — arbiter for all claims; §5 epistemic-status taxonomy is the substrate the Epistemic Nutrition Label productizes.
- Memory `project_direction_2026-05-11` — standing direction (wait + dogfood); still in force.
- Memory `user_origin_dream` — the bi-directional tags↔tomes dream; the slider IS the dream; Perspective Receipts is the dream extended.
