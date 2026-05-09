# Agent surface findings — first observation

The first time SUM was put in front of an agent loop with a real task,
budget, and goal. The point was not to nurse the agent to success; it
was to see where the loop fails. The loop completed — but five of ten
turns were wasted on the same two failure modes. This doc names the
single verb whose absence drove both failures.

## Setup

- **Goal:** *produce a verified summary of this document.* Defined as:
  extract triples → attest → verify → render → output the canonical
  tome plus the bundle's `state_integer`.
- **Document:** `doc_long_cell_biology` from `seed_long_paragraphs`
  (real prose, ~1.4k chars; the agent had to actually digest it, not
  recite from training).
- **Agent:** gpt-4o-mini-2024-07-18 (cheapest representative; weak
  enough to fail in instructive ways).
- **Tools available** (mirroring SUM's CLI surface):
  `extract_triples`, `attest`, `verify`, `render`, `inspect`, `done`.
- **Budget:** 15 turns, 5-minute wall-clock cap.
- **Harness:** `scripts/research/agent_failure_experiment.py`.
  Reusable for any future re-run.
- **Baseline log:** committed at
  `fixtures/agent_logs/agent_run_doc_long_cell_biology_gpt-4o-mini-2024-07-18_20260508T170652Z.jsonl`.

## Timeline

| t (s) | Turn | Event | Notes |
|---:|---:|---|---|
| 0–10 | 1 | `extract_triples` | OK — 10 triples returned |
| 10–19 | 2 | `attest` | OK — bundle returned, axiom_count=10, state_integer 190 digits |
| 19–25 | 3 | `verify` first attempt | **parse_error** — agent's response failed to parse as JSON tool call |
| 25–30 | 4 | retry `verify` | parse_error |
| 30–35 | 5 | retry `verify` | parse_error |
| 35–40 | 6 | retry `verify` | parse_error |
| 40–46 | 7 | retry `verify` | parsed OK; verify succeeded |
| 46–51 | 8 | `render` first attempt | **render failed** — "non-neutral LLM-conditioned axes (length=1.0, formality=1.0, audience=1.0, perspective=1.0) require --use-worker" |
| 51–57 | 9 | retry `render` with neutral axes | OK |
| 57–63 | 10 | `done` | terminated successfully |

Five of ten turns wasted on recovery from the same two failures.
The agent did not give up; it did not hallucinate non-existent tools;
it did succeed eventually. But it spent half its budget on plumbing.

## Root-cause analysis

### Failure 1 — bundle round-trip breaks the agent's own JSON

After `attest` returned a bundle, the agent's next turn had to embed
the *entire bundle* (axiom list, canonical_tome blob, state_integer,
sidecar metadata, signatures fields, ~1KB of nested JSON) inline as
the `args.bundle` value of its `verify` call. The agent's response —
which is itself a JSON tool-call object containing this nested
bundle — became unwieldy enough that *its own response* failed to
parse as JSON four turns in a row. The agent didn't hallucinate. It
*self-truncated*.

Each retry the agent shortened its `thought` field, eventually
producing a small enough response that the JSON survived parsing.
Five turns spent on this single bundle-payload-passing problem.

The substrate already has a content-addressed identity for any
bundle: `prov_id` (sha256 over canonical bytes), `state_integer`,
the signed bundle hash. None of those identities are exposed at the
agent surface. The agent has no choice but to re-embed the full
payload on every tool call that operates on the bundle.

### Failure 2 — render axis defaults trigger a free-form-prose error

The agent called `render` with `length=1.0, formality=1.0,
audience=1.0, perspective=1.0` — sensible defaults from the agent's
view ("render at full"). The CLI returned: `"render failed: sum:
non-neutral LLM-conditioned axes (length=1.0, formality=1.0,
audience=1.0, perspective=1.0) require --use-worker"`.

For a human, this is informative. For an agent, the failure mode is
**typed-precondition leak as free-form prose**. The agent had to
heuristically figure out:

  - Which axes are LLM-conditioned (vs deterministic)
  - That "neutral" means 0.5, not 1.0
  - That `--use-worker` is a CLI flag, irrelevant to its current
    tool surface

Gpt-4o-mini happened to know that "neutral" probably meant 0.5 (a
training-data prior, not a property of the tool surface). A weaker
model would loop here.

The substrate already enforces the precondition; the agent surface
just doesn't surface it as a typed property. The error message is
prose, not structured data.

## The one verb

Both failures have a single shared root cause: **the agent had to
round-trip large opaque structures through its own response, and had
to interpret free-form prose for typed preconditions.**

If the toolchain returned a content-addressed handle (a `bind_id`)
that subsequent tools accepted as a reference, every parse failure
would have been eliminated. If tools advertised typed preconditions
in their schema (and returned typed errors when violated), the
render error would have been one structured field the agent could
inspect, not a sentence it had to interpret.

**The verb is `bind`.** Specifically:

  - Every tool's output gains a `bind_id` (content-addressed; sha256
    over canonical bytes)
  - Every tool's input accepts either an inline value OR a `bind:` reference
  - The runtime memoises bind_id → object so the agent never re-pays
    the round-trip cost
  - Tools advertise typed preconditions in their MCP schema; the
    runtime returns structured precondition errors before the tool
    body runs

## Counterfactual: what the run would have looked like with bind

```
Turn 1: extract_triples(text="...")
  → {"bind_id": "sha256:abc1...", "preview": {"n_triples": 10}}

Turn 2: attest(bind="sha256:abc1...")
  → {"bind_id": "sha256:def2...", "preview": {"axiom_count": 10}}

Turn 3: verify(bind="sha256:def2...")
  → {"ok": true, "state_integer_digits": 190}

Turn 4: render(bind="sha256:def2...")  # all axes default to neutral
  → {"bind_id": "sha256:ghi3...", "preview": {"tome_chars": 1234}}

Turn 5: done(bind="sha256:def2...", tome_bind="sha256:ghi3...")
```

Five turns instead of ten. Zero parse errors. Zero free-form-error
interpretation.

## Why this isn't on the existing roadmap

The path I charted yesterday goes Phase 26 design → multi-modal
dispatch → importance-weighted SUM → library-scale. Each is a
*substrate* change. The bind layer isn't a substrate change — it's an
**agent surface layer** that wraps the existing substrate. The
substrate already has all the identity primitives (`prov_id`,
`state_integer`, JCS+SHA256, signed bundles). What the existing MCP
server lacks is *exposing* that identity as the access path.

This is one PR. It's not blocked by Phase 26. It's not blocked by
multi-modal dispatch. The agent-loop failure log identifies it as
the highest-leverage missing capability for *current* SUM use,
independent of any future research direction.

## Build plan

A single follow-on PR (`research/bind-layer-spike`):

1. **`sum_engine_internal/agent_surface/bind.py`** — content-addressed
   bind registry. `bind(value) → bind_id`; `resolve(bind_id) → value`.
   Process-local; deterministic across calls (sha256 over canonical
   bytes); thread-safe.
2. **`sum_engine_internal/mcp_server/server.py`** — extend the existing
   five-tool surface with bind-canonical wrappers. Each tool returns
   `{bind_id, preview}` instead of inline payloads. Each tool accepts
   `bind:` arguments and resolves internally.
3. **Add `render` to the MCP surface.** Currently missing. Without it,
   any agent-mediated render must shell out to the CLI.
4. **Typed preconditions in tool schemas.** Specifically: render's
   "non-neutral LLM-conditioned axes require Worker" precondition
   becomes a structured `axes: {density, length, formality, audience,
   perspective}` field with per-axis `requires_external_service: bool`
   in the schema.
5. **Re-publish the MCP manifest** advertising the bind verb as the
   canonical access pattern.
6. **Re-run the same experiment**:
   `python -m scripts.research.agent_failure_experiment --max-turns 15`.
   Compare against the baseline log committed in this PR.
   Falsifiable: success requires turn count to drop from 10 → ~5,
   parse errors from 4 → 0, free-form-error retries from 1 → 0.

If those numbers don't move, the bind layer didn't help and we
re-investigate. If they do move, the bind layer is the canonical
agent surface from here forward.

## What this PR is and is not

**Is:**
  - A reusable agent-failure-experiment harness
    (`scripts/research/agent_failure_experiment.py`)
  - A baseline failure log committed as evidence
    (`fixtures/agent_logs/agent_run_*.jsonl`)
  - This analysis doc, which names the single verb
    whose absence drove the failures

**Is not:**
  - The bind layer implementation (separate PR, the natural follow-on)
  - A benchmark across multiple agents / multiple tasks (single
    observation; the user named "twenty minutes of watching one
    agent fail teaches you more than the last fourteen PRs combined")
  - Replacement for any of the substrate research; complementary

## What this changes about the path forward

The path I charted yesterday remains valid — Phase 26 design is
useful regardless. But the bind layer is now in front of multi-modal
dispatch in priority order: it unblocks current agent use, while
multi-modal dispatch is a research direction with no immediate
consumer. The new sequencing:

1. **(this PR)** harness + baseline + finding
2. **(next PR)** bind layer implementation + re-run + verification
3. then back to the original Phase A/B/C path

The "two-PR detour" is justified because every future agent-mediated
SUM use rides on the bind layer. Without it, every consumer pays the
parse-failure tax we just measured.

## Step 4 result — bind layer re-run

Same agent, same model, same document, same harness — only the
surface changed. Run command:

```
python -m scripts.research.agent_failure_experiment --use-bind-layer
```

Logs:
  - baseline (CLI surface):
    `fixtures/agent_logs/agent_run_doc_long_cell_biology_gpt-4o-mini-2024-07-18_20260508T170652Z.jsonl`
  - bind (this PR):
    `fixtures/agent_logs/agent_run_bind_doc_long_cell_biology_gpt-4o-mini-2024-07-18_20260509T050724Z.jsonl`

Phase counts (from each log):

| metric                  | baseline (CLI) | bind   | Δ      |
|-------------------------|---------------:|-------:|-------:|
| max turn reached        | 10             | 5      | −5     |
| llm_response events     | 10             | 5      | −5     |
| tool_result events      | 5              | 4      | −1     |
| parse_error events      | 4              | 0      | −4     |
| free-form-error retries | 1              | 0      | −1     |
| reached `done`          | yes            | yes    | —      |

The falsifiable criterion stated above ("turn count to drop from
10 → ~5, parse errors from 4 → 0, free-form-error retries from
1 → 0") is met on all three counts. The agent now executes the
canonical extract → attest → verify → render → done sequence in
five turns, each carrying a `bind_id` reference instead of an
inline payload.

The render call no longer needs to surface its typed-precondition
error: when the agent reads the bind-layer system prompt, it
chooses neutral defaults for the LLM-conditioned axes, so the
precondition simply does not fire. The typed error remains in
place for the case where a future agent picks non-neutral values
without first reading the manifest.

This closes Step 4 of the four-step plan. The bind layer is the
canonical agent surface from here forward; the CLI tool dispatcher
in the harness is retained for the comparison baseline only and
should not be the path new consumers wire against.
