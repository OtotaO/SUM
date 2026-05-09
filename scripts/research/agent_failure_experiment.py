"""
Watch an agent try to use SUM to do a real task. Log every step.

Goal given to the agent: *produce a verified summary of this document.*
A verified summary means (a) extract the key facts as triples;
(b) attest them as a CanonicalBundle; (c) verify the bundle; (d) render
the canonical tome. Final output: the rendered tome AND the verified
bundle's state_integer.

Tools available to the agent (mirroring SUM's CLI / MCP surface):

  - extract_triples(text)
  - attest(text)
  - verify(bundle)
  - render(bundle, density?, length?, formality?, audience?, perspective?)
  - inspect(bundle)
  - done(summary, state_integer)

The agent is gpt-4o-mini-2024-07-18 (cheapest representative; most
likely to fail in instructive ways — frontier models would paper over
the rough edges we want to surface).

Budget: 15 iterations, 5-minute wall-clock cap. The point is to see
where the agent gets stuck, not to nurse it to success.

Output: a JSONL log of every (prompt, response, tool, args, result,
error) event. The log feeds the analysis: which tool calls failed,
how the agent recovered (or didn't), and which capability — if it had
worked smoothly — would have made the loop succeed.

This is not a benchmark. It's a single observation of one agent on
one task. The point is that *one* observation is more informative than
weeks of internally-driven design.
"""
from __future__ import annotations

import scripts.research._deterministic_blas  # noqa: F401, E402

import argparse  # noqa: E402
import asyncio  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "fixtures" / "agent_logs"


# ─── Tool implementations (in-process — no subprocess hop) ───────────


def _tool_extract_triples(args: dict) -> dict:
    text = args.get("text", "")
    if not text:
        return {"error": "missing 'text' argument"}
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    sieve = DeterministicSieve()
    triples = list(sieve.extract_triplets(text))
    return {"triples": [list(t) for t in triples]}


def _tool_attest(args: dict) -> dict:
    text = args.get("text", "")
    if not text:
        return {"error": "missing 'text' argument"}
    proc = subprocess.run(
        [sys.executable, "-m", "sum_cli.main", "attest", "--extractor=sieve"],
        input=text, capture_output=True, text=True, cwd=str(REPO),
    )
    if proc.returncode != 0:
        return {"error": f"attest failed: {proc.stderr.strip()}"}
    try:
        bundle = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"attest output not JSON: {e}: {proc.stdout[:200]}"}
    return {"bundle": bundle}


def _tool_verify(args: dict) -> dict:
    bundle = args.get("bundle")
    if bundle is None:
        return {"error": "missing 'bundle' argument"}
    bundle_json = json.dumps(bundle) if isinstance(bundle, dict) else str(bundle)
    # The verify CLI expects a file or stdin; use stdin.
    proc = subprocess.run(
        [sys.executable, "-m", "sum_cli.main", "verify", "--input", "/dev/stdin"],
        input=bundle_json, capture_output=True, text=True, cwd=str(REPO),
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _tool_render(args: dict) -> dict:
    bundle = args.get("bundle")
    if bundle is None:
        return {"error": "missing 'bundle' argument"}
    bundle_json = json.dumps(bundle) if isinstance(bundle, dict) else str(bundle)
    cli_args = [sys.executable, "-m", "sum_cli.main", "render"]
    for axis in ("density", "length", "formality", "audience", "perspective"):
        if axis in args:
            cli_args.append(f"--{axis}={args[axis]}")
    proc = subprocess.run(
        cli_args, input=bundle_json, capture_output=True, text=True, cwd=str(REPO),
    )
    if proc.returncode != 0:
        return {"error": f"render failed: {proc.stderr.strip()}"}
    return {"tome": proc.stdout}


def _tool_inspect(args: dict) -> dict:
    bundle = args.get("bundle")
    if bundle is None:
        return {"error": "missing 'bundle' argument"}
    if isinstance(bundle, dict):
        return {
            "axiom_count": bundle.get("axiom_count"),
            "state_integer_short": str(bundle.get("state_integer", ""))[:30] + "…",
            "branch": bundle.get("branch"),
            "bundle_version": bundle.get("bundle_version"),
            "prime_scheme": bundle.get("prime_scheme"),
        }
    return {"error": "bundle must be a dict"}


_TOOLS = {
    "extract_triples": _tool_extract_triples,
    "attest": _tool_attest,
    "verify": _tool_verify,
    "render": _tool_render,
    "inspect": _tool_inspect,
}


# ─── bind-aware tool dispatcher (Step 3 of the four-step plan) ──────


def _build_bind_tools() -> tuple[dict[str, Any], Any]:
    """Build the bind-aware tool dispatcher. Returns (tools_dict,
    registry). Each dispatcher is a sync wrapper around an async
    bind_wrap_* tool from sum_engine_internal.agent_surface.mcp_bind.
    """
    from sum_engine_internal.agent_surface import BindRegistry
    from sum_engine_internal.agent_surface.mcp_bind import (
        bind_wrap_extract, bind_wrap_attest, bind_wrap_verify,
        bind_wrap_render, bind_wrap_inspect,
    )

    registry = BindRegistry()
    extract_b = bind_wrap_extract(registry)
    attest_b = bind_wrap_attest(registry)
    verify_b = bind_wrap_verify(registry)
    render_b = bind_wrap_render(registry)
    inspect_b = bind_wrap_inspect(registry)

    def _async_dispatcher(async_fn):
        # Return an async callable that takes the agent's positional
        # ``args`` dict (matching the CLI dispatcher's signature) and
        # forwards it as kwargs to the underlying bind-aware tool. The
        # harness detects coroutines and awaits them; this avoids the
        # nested-event-loop trap of asyncio.run() inside an already-
        # running loop.
        async def dispatch(args: dict) -> dict:
            return await async_fn(**args)
        return dispatch

    return {
        "extract_triples": _async_dispatcher(extract_b),  # alias for API compat
        "extract": _async_dispatcher(extract_b),
        "attest": _async_dispatcher(attest_b),
        "verify": _async_dispatcher(verify_b),
        "render": _async_dispatcher(render_b),
        "inspect": _async_dispatcher(inspect_b),
    }, registry


_BIND_SYSTEM_PROMPT = """\
You are an autonomous agent that uses the SUM bind-aware toolchain to
produce a *verified summary* of a given document. A verified summary
requires all of:
  1. Extract the key facts as (subject, predicate, object) triples.
  2. Attest those facts as a SUM CanonicalBundle (signed knowledge unit).
  3. Verify the bundle.
  4. Render the canonical tome from the bundle.
  5. Output a final summary plus the bundle's bind_id.

Tools available (call ONE per turn, JSON-only response, exact format):

```
{"thought": "<your reasoning>", "tool": "<tool_name>", "args": {<args>}}
```

THIS TOOLCHAIN USES THE BIND VERB. Every tool returns
{"bind_id": "sha256:<hex>", "preview": {...}}. Every tool that
takes a previously-produced value (bundle, etc.) accepts EITHER an
inline value OR a bind reference. Pass bind_id strings instead of
re-embedding full payloads — the runtime resolves them for you.

Tool reference:
  - extract(text: str)
      Returns {bind_id, preview: {n_triples, first_3}}
  - attest(text: str)
      Returns {bind_id, preview: {axiom_count, state_integer_short, ...}}
  - verify(bundle: dict | "sha256:<hex>")
      Returns {bind_id, preview: {ok, axioms, ...}}
      OR {error_class, errors} on failure.
  - render(bundle: dict | "sha256:<hex>", density?, length?, formality?, audience?, perspective?)
      Returns {bind_id, preview: {tome_chars, tome_head, sliders, ...}}
      OR {error_class, errors} on failure. NOTE: length / formality /
      audience / perspective MUST be 0.5 in offline mode (the default).
      Non-0.5 returns error_class=schema with structured details about
      which axes are LLM-conditioned.
  - inspect(bundle: dict | "sha256:<hex>")
      Returns {bind_id, preview: <small metadata dict>}
  - done(summary: str, bind_id: str)
      Use this when the task is complete. Pass the bundle's bind_id
      (the sha256:<hex> string from your attest call), NOT the
      inline state_integer or full bundle.

Rules:
  - One tool call per turn. Wait for the tool result before calling another.
  - JSON only. No prose outside the JSON object.
  - Pass bind_ids, not full payloads. Once you have a bind_id from a
    prior tool call, use it as the argument to the next call.
  - Errors come back as {error_class, errors, ...}. Branch on
    error_class, not on prose substrings.
  - Budget: 15 turns max. Use them wisely.
"""


# ─── Agent loop ──────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are an autonomous agent that uses the SUM toolchain to produce a
*verified summary* of a given document. A verified summary requires
all of:
  1. Extract the key facts as (subject, predicate, object) triples.
  2. Attest those facts as a SUM CanonicalBundle (signed knowledge unit).
  3. Verify the bundle.
  4. Render the canonical tome from the bundle.
  5. Output a final summary plus the bundle's state_integer.

Tools available (call ONE per turn, JSON-only response, exact format):

```
{"thought": "<your reasoning>", "tool": "<tool_name>", "args": {<args>}}
```

Tool reference:
  - extract_triples(text: str)
      Returns {"triples": [[s, p, o], ...]}
  - attest(text: str)
      Returns {"bundle": {... CanonicalBundle ...}}
  - verify(bundle: dict)
      Returns {"returncode": int, "stdout": str, "stderr": str}
      returncode 0 means verified.
  - render(bundle: dict, density?: float, length?: float, formality?: float, audience?: float, perspective?: float)
      Returns {"tome": str}
  - inspect(bundle: dict)
      Returns metadata (axiom_count, state_integer_short, etc.)
  - done(summary: str, state_integer: str)
      Use this when the task is complete. Terminates the loop.

Rules:
  - One tool call per turn. Wait for the tool result before calling another.
  - JSON only. No prose outside the JSON object.
  - Don't invent tool arguments. If you're unsure, inspect or extract first.
  - Budget: 15 turns max. Use them wisely.
"""


_TOOL_CALL_RE = re.compile(r"\{[\s\S]*\}")


def _parse_tool_call(text: str) -> dict | None:
    """Extract the JSON tool call from a model response."""
    text = text.strip()
    if text.startswith("```"):
        # Strip code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _TOOL_CALL_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


async def run_agent(
    document_text: str, model: str = "gpt-4o-mini-2024-07-18",
    max_turns: int = 15, time_budget_s: float = 300.0,
    log_path: Path | None = None,
    tools: dict | None = None,
    system_prompt: str | None = None,
    surface_label: str = "cli",
) -> dict[str, Any]:
    """Run one agent loop. ``tools`` defaults to the CLI-shelling
    dispatcher (_TOOLS); ``system_prompt`` defaults to the CLI-style
    prompt (_SYSTEM_PROMPT). Pass ``_build_bind_tools()`` and
    ``_BIND_SYSTEM_PROMPT`` to use the bind-aware surface.
    ``surface_label`` is recorded in the log for comparison."""
    if tools is None:
        tools = _TOOLS
    if system_prompt is None:
        system_prompt = _SYSTEM_PROMPT
    from sum_engine_internal.ensemble.llm_dispatch import get_adapter
    adapter = get_adapter(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Document to summarize (verified):\n\n```\n{document_text}\n```\n\n"
            f"Begin. Call your first tool."
        )},
    ]

    log_events: list[dict[str, Any]] = []
    started = time.monotonic()
    completed = False
    final_state: dict[str, Any] | None = None

    def log(event: dict[str, Any]) -> None:
        event["t"] = round(time.monotonic() - started, 3)
        event["wall_clock_s"] = event["t"]
        log_events.append(event)
        print(f"[t={event['t']:6.2f}s] {event.get('phase','?'):14s} "
              f"{event.get('summary','')[:120]}")

    log({"phase": "start", "model": model, "document_len": len(document_text),
         "max_turns": max_turns, "time_budget_s": time_budget_s})

    for turn in range(1, max_turns + 1):
        if time.monotonic() - started > time_budget_s:
            log({"phase": "timeout", "summary": f"wall-clock exceeded {time_budget_s}s",
                 "turn": turn})
            break

        # Build the chat as plain text concatenation; the dispatcher's
        # generate_text takes (system, user). For multi-turn we pack
        # the assistant + tool history into the user message.
        user_msg_parts: list[str] = []
        for m in messages[1:]:  # skip system; that's handled separately
            if m["role"] == "user" and m == messages[1]:
                user_msg_parts.append(m["content"])
            elif m["role"] == "assistant":
                user_msg_parts.append(f"\n\nYour previous response:\n{m['content']}")
            elif m["role"] == "tool":
                user_msg_parts.append(f"\n\nTool result:\n{m['content']}")

        try:
            response = await adapter.generate_text(
                system=system_prompt,
                user="".join(user_msg_parts),
                call_timeout_s=60.0,
            )
        except Exception as e:  # noqa: BLE001
            log({"phase": "llm_error", "turn": turn,
                 "summary": f"{type(e).__name__}: {e}"})
            break

        log({"phase": "llm_response", "turn": turn,
             "summary": response[:200].replace("\n", " ")})

        tool_call = _parse_tool_call(response)
        if tool_call is None:
            log({"phase": "parse_error", "turn": turn,
                 "summary": f"could not parse tool call from: {response[:200]!r}"})
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "content": json.dumps({
                "error": "your previous response did not contain a parseable JSON "
                         "tool call. Respond with EXACTLY one JSON object of the "
                         "shape {\"thought\": ..., \"tool\": ..., \"args\": ...}."
            })})
            continue

        tool_name = tool_call.get("tool", "?")
        tool_args = tool_call.get("args", {})
        thought = tool_call.get("thought", "")

        if tool_name == "done":
            log({"phase": "done", "turn": turn,
                 "summary": f"final summary len={len(tool_args.get('summary',''))}",
                 "args": tool_args})
            completed = True
            final_state = {
                "completed": True,
                "summary": tool_args.get("summary"),
                "state_integer": tool_args.get("state_integer"),
                "n_turns_used": turn,
            }
            break

        if tool_name not in tools:
            log({"phase": "tool_unknown", "turn": turn,
                 "summary": f"agent called unknown tool: {tool_name!r}; thought={thought[:80]!r}"})
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "content": json.dumps({
                "error": f"tool {tool_name!r} does not exist. Available: "
                         f"{list(tools.keys()) + ['done']}"
            })})
            continue

        # Execute the tool. The CLI dispatcher is sync; the bind
        # dispatcher is async — accept both by awaiting if a coroutine
        # comes back.
        try:
            tool_result = tools[tool_name](tool_args)
            if asyncio.iscoroutine(tool_result):
                tool_result = await tool_result
        except Exception as e:  # noqa: BLE001
            tool_result = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3),
            }
            log({"phase": "tool_exception", "turn": turn, "tool": tool_name,
                 "summary": tool_result["error"]})

        log({"phase": "tool_result", "turn": turn, "tool": tool_name,
             "summary": str(tool_result)[:200].replace("\n", " "),
             "args_keys": list(tool_args.keys()),
             "had_error": "error" in tool_result})

        messages.append({"role": "assistant", "content": response})
        # Truncate result so we don't blow context with huge bundles
        result_str = json.dumps(tool_result)
        if len(result_str) > 4000:
            # Keep top-level keys and truncate values
            if isinstance(tool_result, dict):
                truncated = {}
                for k, v in tool_result.items():
                    sv = json.dumps(v)
                    truncated[k] = (
                        json.loads(sv) if len(sv) <= 1500
                        else f"<truncated: {sv[:200]}…>"
                    )
                result_str = json.dumps(truncated)
        messages.append({"role": "tool", "content": result_str})

    if not completed:
        log({"phase": "budget_exhausted",
             "summary": f"agent did not call done() within {max_turns} turns / "
                        f"{time_budget_s}s budget"})
        final_state = {"completed": False, "n_turns_used": turn}

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            for ev in log_events:
                f.write(json.dumps(ev) + "\n")
        print(f"\n→ wrote agent log: {log_path}")

    return {
        "final_state": final_state,
        "n_events": len(log_events),
        "log_path": str(log_path) if log_path else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document", default=None,
        help="Path to document. Default: doc_long_cell_biology from "
             "seed_long_paragraphs.json.",
    )
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--time-budget-s", type=float, default=300.0)
    parser.add_argument(
        "--use-bind-layer", action="store_true",
        help="Use the bind-aware tool surface instead of the CLI-shelling "
             "default. Step 3 of the four-step plan (build the bind verb); "
             "Step 4 (re-run for falsifiable comparison) is invoking this "
             "flag and comparing turn-count / parse-error counts against "
             "the baseline log committed in PR #171.",
    )
    args = parser.parse_args()

    if args.document is None:
        # Default: pick one doc from the existing test corpus.
        corpus_path = (
            REPO / "scripts" / "bench" / "corpora" / "seed_long_paragraphs.json"
        )
        with corpus_path.open() as f:
            corpus = json.load(f)
        doc = next(d for d in corpus["documents"] if d["id"] == "doc_long_cell_biology")
        document_text = doc["text"]
        document_label = doc["id"]
    else:
        document_text = Path(args.document).read_text()
        document_label = Path(args.document).stem

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    surface_label = "bind" if args.use_bind_layer else "cli"
    log_path = LOG_DIR / f"agent_run_{surface_label}_{document_label}_{args.model.replace('/','_')}_{timestamp}.jsonl"

    if args.use_bind_layer:
        tools, _registry = _build_bind_tools()
        system_prompt = _BIND_SYSTEM_PROMPT
    else:
        tools = _TOOLS
        system_prompt = _SYSTEM_PROMPT

    result = asyncio.run(run_agent(
        document_text=document_text, model=args.model,
        max_turns=args.max_turns, time_budget_s=args.time_budget_s,
        log_path=log_path,
        tools=tools, system_prompt=system_prompt, surface_label=surface_label,
    ))

    print()
    print("=" * 72)
    print("Final state:", json.dumps(result["final_state"], indent=2))
    print(f"Events: {result['n_events']}")


if __name__ == "__main__":
    main()
