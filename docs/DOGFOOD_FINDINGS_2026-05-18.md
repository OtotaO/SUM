# DOGFOOD_FINDINGS_2026-05-18.md

**Second dogfood pass — five-axis slider variant comparison via NVIDIA NIM with F7 fix live.** Companion to [`DOGFOOD_FINDINGS_2026-05-17.md`](DOGFOOD_FINDINGS_2026-05-17.md). The 2026-05-17 pass surfaced F1–F7 in mechanical execution; this pass tested whether the F7-fixed dispatch produces useful, audience-shifted prose AND whether the slider's per-axis claims hold up on a real user-supplied draft.

## Pass setup

- Provider: NVIDIA NIM via `nim:meta/llama-3.3-70b-instruct` (Tier 1 free credits)
- F7 fix in place (PR #241 — `make_chat_client` factory; compatible providers skip OpenAI's `beta.chat.completions.parse`)
- Source: 8 hand-extracted triples about World_sim (Nous Research), business-superintelligence simulation, media+telecommunications scope, expected-vs-unexpected output
- Five variants tested at distinct slider positions

## Variant matrix

| Variant | density | length | formality | audience | perspective | words out |
|---|---:|---:|---:|---:|---:|---:|
| novice-casual | 1.0 | 0.7 | 0.2 | 0.2 | 0.5 | 238 |
| expert-formal | 1.0 | 0.5 | 0.9 | 0.9 | 0.5 | 88 |
| engineer-precise | 1.0 | 0.3 | 0.7 | 0.7 | 0.5 | 46 |
| child-storyteller | 1.0 | 0.9 | 0.1 | 0.1 | 0.1 | 385 |
| adversarial-reviewer | 1.0 | 0.5 | 0.7 | 0.9 | 0.9 | 71 |

All five rendered coherently. NIM dispatch + F7 fix proven in production.

## Per-axis drift (slider_renderer.measure_drift output)

Same triple set, five variants. Density measured as `1 - retained_fraction`:

| Variant | density (target=0) | length | formality | audience | classification |
|---|---:|---:|---:|---:|---|
| novice-casual | **1.000** | 0.339 (ok) | 0.200 (ok) | 0.099 (ok) | density=fail; style=ok |
| expert-formal | **0.875** | 0.571 (warn) | 0.100 (ok) | 0.116 (warn) | density=fail; length+aud warn |
| engineer-precise | **1.000** | 0.324 (ok) | 0.200 (ok) | 0.026 (ok) | density=fail; style=ok |
| child-storyteller | **1.000** | 0.563 (warn) | 0.100 (ok) | 0.064 (ok) | density=fail; length warn |
| adversarial-reviewer | **1.000** | 0.268 (ok) | 0.200 (ok) | 0.078 (ok) | density=fail; style=ok |

**Density-axis fails on every variant.** Honest signal: the LLM elaborates beyond the 8 source triples on every render, and sieve re-extraction doesn't recover the original axiom set. This is the F2 finding propagating through the pipeline. The slider IS measuring it and IS flagging it. Style axes (length/formality/audience) are within tolerance.

This is a feature not a bug: the slider tells you when preservation broke. The receipt is honest. The fix (LLM extractor for re-extraction, or stricter generator prompts) is a separate worktrail.

## New findings

### F8 — `perspective` axis silently flipped third-person to first-person [severity: HIGH]

Source triples reference `"author"` as a third-person entity. The `child-storyteller` variant (perspective=0.1, the "first-person" pole) produced first-person prose ("I've been working with this thing called World_sim... I used World_sim..."). **The slider invented that the LLM-call-output narrator IS the author named in the source triples.**

Quote from the rendered tome:

> *"As the author of this little experiment, I used World_sim to try and simulate something really complex: business superintelligence. I mean, think about it..."*

The source triple `("author", "used", "World_sim")` was preserved structurally, but the entity reference shifted from third-person `author` to first-person `I`. A consumer of the rendered tome would reasonably believe the speaker is the author — which is not what the source says.

**Why this is load-bearing:** the SLIDER_CONTRACT.md claim that "axis changes do not lose facts" is preserved structurally (the `(author, used, World_sim)` fact survives re-extraction) but the *semantic identity* of `author` changed. NLI audit may or may not catch this depending on how the entailment model handles pronouns.

**Fix paths:**
- Per-axis preservation contract should explicitly cover entity-reference stability across perspective shifts.
- Prompt hardening on the perspective axis: "preserve the original entity references" when perspective moves toward first-person.
- Surface as a "perspective_drift" sub-metric measuring whether named entities maintained their syntactic-role across the transform.

### F9 — `perspective` axis at 0.9 doesn't produce adversarial-reviewer stance [severity: MEDIUM]

Intent: `perspective=0.9` is supposed to be the "omniscient/third-person" / external-observer pole. Combined with high audience+formality, I expected an adversarial-reviewer voice — skeptical, surfacing gaps, asking scope questions, noting what isn't said.

Actual output: a slightly more formal restatement of the same facts. No skepticism, no gap-surfacing, no scope questioning.

**Diagnosis:** the slider's perspective axis is documented as POV (first-person → omniscient), not stance (sympathetic → adversarial). My expectation was wrong; the contract is right. But this gap is operationally interesting — *"render this as an adversarial reviewer would"* is the kind of slider position a journalist or academic would value, and it doesn't exist as an axis today.

**Connection to zenith framing:** the ZENITH_FRAMING_2026-05-16's "Perspective Receipts" concept addresses exactly this — renaming axes from stylistic (length/formality/audience/perspective) to perspectival (novice/expert/regulator/engineer/adversarial-reviewer/...). The dogfood data confirms the need.

### F10 — formality and audience axes are highly collinear in practice [severity: MEDIUM]

The slider's contract treats `formality` (casual ↔ academic) and `audience` (novice ↔ expert) as independent axes. In practice, the LLM responds to *both* dropping together (novice/casual; child/storyteller) with a similar shift, and to *both* rising together (expert/formal) with a similar shift. Setting one high while the other is low (e.g. formality=0.9, audience=0.2 — "casual reader but formal language") was not tested but is the diagnostic position that would isolate the axes.

**Why this matters:** if the two axes are highly collinear, the slider effectively has 4 axes not 5. The compositional richness the SLIDER_CONTRACT claims is one axis weaker than advertised.

**Fix paths:**
- Bench task: explicitly measure axis-orthogonality across an N×M grid of (formality, audience) positions. Bench-hardening T2 (capability regions) is the right artifact.
- If the axes ARE collinear at scale, fold them or document the collinearity explicitly.
- If they're separable but only the LLM doesn't separate them, prompt hardening could help.

### F11 — `engineer-precise` produced a bulleted list, not paragraph prose [severity: LOW — interesting]

At `length=0.3` and `audience=0.7`, the LLM rendered as bullets:

> *"World_sim was made by Nous Research.* (newline)
> *World_sim is an online interface.* (newline)
> *It has a Large Language Model (LLM) backend..."*

The slider doesn't have a "format" axis (paragraph vs list vs table). The LLM inferred the format from the combination of low length + technical audience. This is *interesting* — the slider's compositional behavior produces emergent format choices the contract doesn't name. Could be a feature (the slider does "right thing" inference) or a bug (output format is uncontrolled, callers can't predict it).

**For dogfood downstream:** an `--output-format` slider axis (prose / list / table / nested-outline) would let publishing consumers (newsletter, journal, brief, slide-prep) control more precisely. Adds an axis; receipt-stable; non-breaking.

## What this dogfood validated

- **NIM dispatch via `nim:` prefix works** — five round-trips, zero errors, ~5 credits consumed (~$0).
- **F7 fix lands** — `make_chat_client` factory routes correctly; compatible providers produce coherent prose where v0.6.x produced single-token degenerate parses.
- **Slider axes ARE differentiating** — 46 to 385 words across the five variants; novice explains LLM, expert assumes it, engineer bullets it. The product surface is real.
- **Density measure is honest** — flags fail on every LLM-axis render. The proof-boundary discipline holds at run time, not just at spec time.
- **The slider IS the dream made code** — the user's draft text was reshaped across five distinct registers with measurable preservation telemetry. The wedge ICP claim ("journalists who would otherwise need a fact-checker") has empirical surface area now.

## Frontend gap surfaced separately

The `single_file_demo/index.html` user-facing demo:

**Has:** paste prose → bundle attest, slider UI (with all five axes wired to `/api/render`), BYO-key entry in localStorage, render receipt verification.

**Missing (relative to backend capability):**

| Backend capability | Frontend exposure |
|---|---|
| `/api/transform` endpoint (the transform-receipt substrate) | ✗ NOT used by frontend |
| `compose` transform (merge bundles) | ✗ |
| `extract` transform with multi-school | ✗ |
| `ShareableRender` (round-trip signed-render artifact) | ✗ |
| `sum verify --explain` layered output | ✗ |
| `source_chain_hash` evidence-chain display | ✗ |
| `signed_at_out_of_window` replay-defense indicator | ✗ |
| Negative-control bench surface | ✗ (offline only — correct) |
| `nim:` / `groq:` / `cerebras:` BYO-key UI | ✗ (only OpenAI + Anthropic in current panel) |
| Epistemic Nutrition Label (zenith concept) | ✗ |
| Perspective Receipts axis-rename (zenith concept) | ✗ |

**Operationally:** a funder hitting the demo today sees ~30% of what SUM ships. The slider works through the demo; everything else in the transform substrate (PRs #210–#241) is invisible. The README now lists it (PR #237) but the demo doesn't.

**Fix path:** a frontend roadmap PR that extends `single_file_demo/index.html` with:
1. Transform selector (slider / extract / compose) instead of just slider.
2. `/api/transform` invocation path alongside `/api/render`.
3. BYO-key panel extended with NIM / Groq / Cerebras / HF token fields.
4. Receipt display extended to show `extra.llm_endpoint` (so users see which tier served).
5. (Stretch) `verify --explain` layered output rendered as the Epistemic Nutrition Label.

Each is a small incremental addition; the cumulative effect closes the frontend↔backend gap. Estimated ~2–3 days for the full set. Deferred until grant signal or explicit pull.

## Summary table — F-findings so far

| # | Finding | Status |
|---|---|---|
| F1 | `FutureWarning` to stdout | fixed PR #239 |
| F2 | Sieve conservative — drops dates/specifics | known; mitigation via LLM extractor |
| F3 | PyPI v0.6.0 lacks `transform` subcommand | open; user-side (tag push) |
| F4 | `attest` output shape ≠ `compose` input | open; engine fix queued |
| F5 | Scenario A pipeline broken end-to-end | open; resolved by F3+F4 |
| F6 | Worker Anthropic key 401 | open; user-side `wrangler secret put` |
| F7 | `beta.chat.completions.parse` degenerate on non-OpenAI | fixed PR #241 |
| F8 | Perspective axis silently flips POV | open; spec clarification + prompt hardening |
| F9 | Perspective axis ≠ adversarial-reviewer stance | open; routes to Perspective Receipts (zenith concept) |
| F10 | Formality/audience axes collinear in practice | open; bench-hardening T2 measures it |
| F11 | engineer-precise rendered bullets, not prose | open; potential new `--output-format` axis |
| (FE) | Frontend missing ~9 backend surfaces | open; FE roadmap PR pending |

## Loop closure

This dogfood produced four new substantive findings (F8–F11) AND validated the F7 fix from yesterday in live production AND quantified frontend↔backend congruency. Per `CHARTER_2026-05-17.md` §5.1, dogfood is one of three load-bearing signals; this is the second deliberate dogfood pass since the charter landed and is producing more signal per minute than 14 internal substrate PRs combined.

## Pointers

- [`DOGFOOD_FINDINGS_2026-05-17.md`](DOGFOOD_FINDINGS_2026-05-17.md) — first pass; F1–F7
- [`FALLBACK_PROVIDER_CASCADE_2026-05-18.md`](FALLBACK_PROVIDER_CASCADE_2026-05-18.md) — provider routing the NIM run exercised
- [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) — the contract F8–F11 test
- [`ZENITH_FRAMING_2026-05-16.md`](ZENITH_FRAMING_2026-05-16.md) — Perspective Receipts concept that F9 + axis-rename resolves
- [`BENCH_HARDENING_FROM_QCVV.md`](BENCH_HARDENING_FROM_QCVV.md) — T2 capability regions is the right artifact to measure F10 axis-orthogonality
