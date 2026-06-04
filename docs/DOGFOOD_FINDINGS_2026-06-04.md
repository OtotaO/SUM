# Dogfood findings — 2026-06-04 (Scenario A end-to-end smoke)

First end-to-end smoke of `docs/DOGFOOD_QUICKSTART.md` Scenario A (journalist
distill loop: `attest → compose → slider → verify`) on `main` post-PRs #261–#265.
Run with synthetic AI-policy sources (3 short docs) to verify the *path*, not the
prose quality. **The path works** — but the documented invocation was broken in
three places a real user (or a funder trying the demo) would hit immediately.
Continues the series: `DOGFOOD_FINDINGS_2026-05-17.md` (F1–F7),
`_2026-05-18.md` (F8–F11), `_2026-05-28.md` (F13), `_2026-05-29.md` (F14).

## Outcome

`attest` (3 sources → 2/3/2 axioms) → `compose` (7 merged axioms; **F4 confirmed**
— attest's `axioms` field consumed directly) → `slider` (density 0.5, canonical
path, no LLM) → brief: *"The act classify ai system. The article require record
keeping. The risk_system keep automatic log. The cryptographic_signature bind
provenance."* (4 of 7 facts kept at density 0.5). The loop is sound; the docs were
not.

## Findings

### F15 — `transform apply --input` is a file path / stdin, not inline JSON  [HIGH, doc]
Scenario A showed `sum transform apply compose --input '{"bundles":[...]}'`, but
the CLI (`sum_cli/main.py:2004`) opens `--input` as a **file path** (or `-` for
stdin). Inline JSON fails with `sum: cannot read --input '{...}'`. A user following
the doc verbatim dead-ends at step 4.
**Fix (shipped this finding):** write the input to a file first, pass the path.
**Why:** matters because Scenario A is the load-bearing journalist path and the
first hands-on surface a reviewer/funder touches.

### F16 — compose→slider shape mismatch (`axioms` key vs `triples` key)  [HIGH, doc]
`compose` emits `{"output": {"axioms": [[s,p,o], …], "state_integer": …}}`;
`slider` requires `{"triples": [[s,p,o], …]}` (`slider input: expected dict with
'triples' key`). Same inner shape, **different key**. The doc's
`--input "$(cat merged.json | jq '.output')"` passes the wrong shape and fails at
step 5.
**Fix (shipped this finding):** bridge `{"triples": output["axioms"]}` into a file.
**Candidate code-ergonomics follow-up (operator decision, NOT done):** have `slider`
also accept an `axioms`-keyed input, or have `compose` emit `triples`, so the two
transforms compose without a manual rename. Touches the transform I/O contract +
schemas + Worker; out of scope for a doc fix — flagged for the buyer/dream filter.

### F17 — editable dev install reports a stale version  [LOW, environment]
On this dev box the `sum` console script reported **0.6.0** while the repo /
`pyproject.toml` was 0.7.0 (`cli_version: "0.6.0"` baked into produced bundles),
because the editable install (`pip install -e .`) predated the version bump and was
never re-installed. Running the repo code via `python -m sum_cli.main` reported the
correct 0.7.0. Environment-specific, not a PyPI-user issue, but the doc now tells
you to confirm `sum --version` **and** the bundle's `cli_version`.

## What this earns (charter §3 / §7.2)

Dogfood findings "earn substrate work back from the wait." These are doc/UX, not
substrate — the slider loop itself is the dream made code and it runs. The one
substrate-adjacent item (F16's compose→slider ergonomics) is recorded as a
candidate, gated on buyer/dream pull, not auto-started. Next genuine signal would
be a *prose-quality* finding from a real source set (your writing), which only the
operator can produce.
