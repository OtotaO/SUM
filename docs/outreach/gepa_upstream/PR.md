# Upstream PR draft — `gepa-ai/gepa`

> Status: **draft for operator review.** The `--selftest` path is verified (key-free, exercises the SUM-scoring core). The live `gepa.optimize` path is written against the documented `GEPAAdapter` protocol but **must be run once with a real model + key before sending** — confirm the `result.best_candidate` access and the `litellm` call shape against the installed `gepa` version, since their API may have moved.

---

## Target

- **Repo:** `gepa-ai/gepa`
- **Suggested path in their tree:** `examples/sum_meaning/optimize_faithful_summary.py` (+ a short `examples/sum_meaning/README.md` adapted from the "What it shows" section below).
- **Our source of truth:** `docs/outreach/gepa_upstream/optimize_faithful_summary.py` in the SUM repo — copy verbatim.

## Proposed PR title

`examples: optimise a summariser against a semantic-faithfulness metric (SUM meaning-loss)`

## Proposed PR body

> ### What this adds
>
> A worked example driving GEPA with a **semantic-faithfulness** metric instead of an exact-match one, via a custom `GEPAAdapter`.
>
> The metric is a SUM meaning-loss proxy (`pip install sum-engine[research]`): a bounded `[0, 1]` score **and** a per-document **kept / dropped / added** readout. That readout maps cleanly onto GEPA's `(score, textual-feedback)` design — `evaluate()` returns the numeric scores, `make_reflective_dataset()` routes the readout into each record's `"Feedback"`, so the reflection LM is told *which source claims a candidate summary dropped* and *which it added without grounding*, rather than reflecting on a bare number.
>
> ### Why it might be useful to GEPA users
>
> Faithfulness/meaning-preservation is a common optimisation target (summarisation, RAG answers, translation) where exact match is the wrong metric and an LLM-judge scalar throws away *why* an output was penalised. This example shows the textual-feedback channel doing real work on that class of task. The same proxy can additionally be turned into a signed, offline-verifiable certificate with a distribution-free `(1-δ)` bound over a corpus — so a team can optimise a prompt here and then certify the winner — but the example itself only needs the scorer.
>
> ### Runs with no key
>
> `python optimize_faithful_summary.py --selftest` exercises the scoring core (faithful summary scores `1.000`, a summary that drops two claims and adds an ungrounded one scores `0.350`, and the feedback names exactly what changed) with no model download and no API key. The full path is `python optimize_faithful_summary.py --task-lm openai/gpt-4.1-mini --reflection-lm openai/gpt-5`.
>
> ### Honest scope (kept in the file's docstring)
>
> The per-document readout is a **measurement under a named, swappable judge**, not "meaning" itself and not a certified bound. The self-test uses a deterministic token-overlap stand-in judge (blind to paraphrase, labelled as such); real use plugs an NLI/embedding/LLM judge. Nothing in the example overclaims.

## Talking points for the cross-post (their Discord / the PR thread)

- Lead with the *mechanism fit*, not the product: "GEPA's feedback channel is a natural home for a faithfulness metric that can say what was dropped/added — here's a worked adapter."
- Invite scrutiny of the metric: the meaning-loss proxy and its conformal bound are designed to be *recomputed* by a third party; a skeptic poking the bound is the best possible engagement (it's what our adoption sims found warms the rigorous-eval audience).
- Do **not** claim the example provides a guarantee — it provides a measurement-driven optimisation; the guarantee is a separate, signed artifact.

## Pre-send checklist

- [ ] Run the live path once with a real key; confirm `result.best_candidate["summarize"]` and the `litellm.completion(...)` shape against the pinned `gepa` version.
- [ ] Fork `gepa-ai/gepa`, add the example under `examples/sum_meaning/`, include a trimmed README.
- [ ] Open the PR from your account (this is your relationship to own — I can revise the example/body, but the PR is yours to submit).
