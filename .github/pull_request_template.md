<!--
SUM PR template. The "Justification" section is load-bearing —
the project's buyer-or-dream filter (`feedback_buyer_or_dream_filter`
in the auto-memory system) requires every PR to name a buyer, funder
commitment, or dream element it serves. "It's the next math-natural
step" / "closes the wires arc" / "the math is elegant" do not pass
the filter. The filter exists because the project drifted into math-
natural substrate growth from 2026-04 onward, producing orphan
primitives that no downstream surface consumes.

If you don't know the project's history here, read the auto-memory
system at ~/.claude/projects/<...>/memory/ or skim
docs/NEXT_SESSION_PLAYBOOK.md before opening this PR.

Delete this comment block before submitting.
-->

## Summary

<!-- 1–3 sentences. What is this PR? -->

## Justification (buyer-or-dream filter)

<!--
REQUIRED. Pick at least one and fill it in concretely. PRs without a
named justification fail the project's standing direction filter
(see `feedback_buyer_or_dream_filter` in auto-memory). If your work
doesn't fit any of these slots, the PR is likely orphan substrate
work and should be re-scoped or closed.
-->

- [ ] **Named buyer** — a specific audience pays/uses this:
      <!-- e.g. "GPAI providers needing EU AI Act Article 50 evidence",
           "regulated-content publishers needing fact-preservation receipts",
           "AI-safety teams needing audit-trail metadata", "SOC integrators",
           "Chief AI Officers needing ISO 42001 evidence" -->

- [ ] **Named funder commitment** — a deliverable from the grant
      applications (`~/SUMequities/.private-notes/SUM-applications-tracker-*.md`):
      <!-- e.g. "OpenAI adapter (OpenAI Cybersecurity / NLnet / SFF)",
           "local open-weights pathway (NLnet)",
           "integration documentation (OpenAI Cybersecurity)",
           "threat model + cryptographer review",
           "benchmark corpus expansion (NLnet 10× / OpenAI 50–100K)" -->

- [ ] **Named dream element** — `user_origin_dream` in auto-memory:
      <!-- e.g. "tags-from-tome direction surfaced as named output",
           "new render device (register / poetry / revelation axis)",
           "multi-school categorization (N extractors in tandem)",
           "incompressible-floor benchmark",
           "covers more than any other tag maker bench" -->

**Rejected justifications** (do not pass the filter on their own):
- "It's the next mathematically-natural step."
- "It closes the wires arc / completes the layer / extends the framework."
- "The math is interesting and the proof is elegant."
- "It might be useful later for X" (without X being one of the three above).

## Adversarial robustness

<!--
If this PR touches a funder-pitched surface (`sum-engine[openai]`,
the live demo, K1–K4, render-receipt format, six-regime compliance,
slider product, etc.), describe how it survives the
"would-an-adversarial-reviewer-score-this-as-a-win" test. If it
doesn't touch those surfaces, write N/A.
-->

## Test plan

<!--
Required for code changes. Empirical-first principle:
- [ ] `make pre-push` green locally (6/6 stages incl. xruntime + adversarial + fortress)
- [ ] If this PR touches the receipt path or live deploy:
      `make probe-live-trust-loop` green (10/10 fixtures)
- [ ] If this PR touches a code-bearing doc:
      every shell block in the doc was executed in a fresh venv
      before commit (no theoretical code samples; only verified ones)
- [ ] If this PR adds a public surface:
      a regression-guard test pins the surface against future drift
- [ ] CI green
-->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
