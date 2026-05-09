"""Bind-aware wrappers for the MCP tool surface.

These wrappers sit *in front of* the existing MCP tools (extract,
attest, verify, render, inspect from
``sum_engine_internal.mcp_server.server``). They:

  - Resolve any ``bind:`` references in incoming arguments so the
    wrapped tool receives the inline value it expects.
  - Produce an output of shape ``{bind_id, preview, raw}`` instead of
    the inline result. ``raw`` is the original tool output; ``preview``
    is a small subset chosen to be useful for the agent without
    blowing context.
  - Pass typed errors through unchanged (the underlying MCP tools
    already return ``{error_class, errors}``).

The wrapped surface is the canonical agent access path. The original
tools remain available for callers that need inline payloads (e.g.
the Python CLI, programmatic library use).

Usage::

    from sum_engine_internal.agent_surface import DEFAULT_REGISTRY
    from sum_engine_internal.agent_surface.mcp_bind import (
        bind_wrap_extract, bind_wrap_attest, bind_wrap_verify,
        bind_wrap_render, bind_wrap_inspect,
    )

    extract_b = bind_wrap_extract(DEFAULT_REGISTRY)
    result = await extract_b(text="Alice likes cats.")
    # result == {"bind_id": "sha256:...", "preview": {"n_triples": 1}}

    attest_b = bind_wrap_attest(DEFAULT_REGISTRY)
    bundle_result = await attest_b(text="Alice likes cats.")
    # bundle_result == {"bind_id": "sha256:...", "preview": {"axiom_count": 1, ...}}

    verify_b = bind_wrap_verify(DEFAULT_REGISTRY)
    verify_result = await verify_b(bind=bundle_result["bind_id"])
    # verify_result == {"ok": True} (errors pass through as {error_class, errors})
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from sum_engine_internal.agent_surface.bind import (
    BindNotFoundError, BindRegistry,
)


# ─── argument resolution ─────────────────────────────────────────────


def _resolve_bind_arg(value: Any, registry: BindRegistry) -> Any:
    """If ``value`` looks like a bind reference, resolve it; else
    return as-is.

    Recognises two forms:
      - The bare bind_id string ``"sha256:abc..."``
      - The dict form ``{"bind": "sha256:abc..."}`` (allows a tool to
        accept "either inline value or bind reference" in a typed way)
    """
    if isinstance(value, str) and value.startswith("sha256:"):
        try:
            return registry.resolve(value)
        except BindNotFoundError as e:
            # Re-raise with a more agent-friendly message
            raise BindNotFoundError(
                f"bind reference {value!r} could not be resolved. "
                f"Use the bind_id returned from a prior tool call."
            ) from e
    if isinstance(value, dict) and set(value.keys()) == {"bind"}:
        return _resolve_bind_arg(value["bind"], registry)
    return value


def _resolve_kwargs(kwargs: dict[str, Any], registry: BindRegistry) -> dict[str, Any]:
    """Resolve any bind references in a kwargs dict."""
    return {k: _resolve_bind_arg(v, registry) for k, v in kwargs.items()}


# ─── result wrapping + previews ──────────────────────────────────────


def _is_error(result: Any) -> bool:
    return isinstance(result, dict) and "error_class" in result


def _preview_for(tool_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Produce a small, agent-useful preview of a tool result.

    Each tool has a hand-tuned preview shape. The preview NEVER
    inlines the full payload; it surfaces just the metadata an agent
    needs to decide its next call.
    """
    if tool_name == "extract":
        triples = raw.get("triples", [])
        return {
            "n_triples": len(triples),
            "first_3": triples[:3],
        }
    if tool_name == "attest":
        bundle = raw.get("bundle", raw)
        si = bundle.get("state_integer", "")
        return {
            "axiom_count": bundle.get("axiom_count"),
            "state_integer_short": (str(si)[:24] + "…") if si else None,
            "branch": bundle.get("branch"),
            "bundle_version": bundle.get("bundle_version"),
            "prime_scheme": bundle.get("prime_scheme"),
        }
    if tool_name == "verify":
        return {
            "ok": raw.get("ok"),
            "axioms": raw.get("axioms"),
            "state_integer_digits": raw.get("state_integer_digits"),
            "signatures": raw.get("signatures"),
        }
    if tool_name == "render":
        tome = raw.get("tome", "")
        return {
            "tome_chars": len(tome),
            "tome_head": tome[:120],
            "sliders": raw.get("sliders"),
            "axiom_count_input": raw.get("axiom_count_input"),
        }
    if tool_name == "inspect":
        return raw  # already small
    return {"_preview_unavailable": True, "_tool": tool_name}


def _wrap_result(
    tool_name: str, raw: Any, registry: BindRegistry,
) -> dict[str, Any]:
    """Return ``{bind_id, preview}`` for a successful tool result, or
    pass an error result through unchanged."""
    if _is_error(raw):
        return raw
    bind_id = registry.bind(raw)
    preview = _preview_for(tool_name, raw)
    return {"bind_id": bind_id, "preview": preview}


# ─── tool wrappers ───────────────────────────────────────────────────


def _bind_wrap(
    tool_name: str, tool_func: Callable[..., Awaitable[Any]],
    registry: BindRegistry,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Generic wrapper: resolve binds in args, call tool, wrap result."""
    async def wrapped(**kwargs):
        try:
            resolved = _resolve_kwargs(kwargs, registry)
        except BindNotFoundError as e:
            return {
                "error_class": "schema",
                "errors": [str(e)],
            }
        raw = await tool_func(**resolved)
        return _wrap_result(tool_name, raw, registry)
    wrapped.__name__ = f"bind_{tool_name}"
    wrapped.__doc__ = (
        f"Bind-aware wrapper around the MCP {tool_name!r} tool. "
        f"Args may include bind references; result is "
        f"{{bind_id, preview}} on success, {{error_class, errors}} on "
        f"failure (passed through unchanged from the underlying tool)."
    )
    return wrapped


# ─── direct in-process bindings to the underlying tool implementations ─
#
# Rather than going through FastMCP's transport layer (which would
# require a live server), the wrappers below call the underlying
# tool functions in-process. This keeps the bind-spike self-contained
# for the agent_failure_experiment harness; a future PR can mount
# these as actual FastMCP tools on the existing MCP server.


def _underlying_extract():
    """Return the underlying extract tool implementation as a callable."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.mcp_server.errors import ErrorClass

    sieve = DeterministicSieve()

    async def extract(text: str, extractor: str = "sieve") -> dict:
        if extractor not in ("sieve", "auto"):
            return {
                "error_class": ErrorClass.EXTRACTOR_UNAVAILABLE.value,
                "errors": [
                    f"only the offline sieve extractor is wired in this "
                    f"bind spike; got extractor={extractor!r}"
                ],
            }
        if not isinstance(text, str) or not text.strip():
            return {
                "error_class": ErrorClass.SCHEMA.value,
                "errors": ["text must be a non-empty string"],
            }
        triples = list(sieve.extract_triplets(text))
        return {
            "triples": [list(t) for t in triples],
            "n_triples": len(triples),
        }
    return extract


def _underlying_attest():
    """Return an in-process attest implementation matching the CLI's
    sieve path."""
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path
    REPO = _Path(__file__).resolve().parents[2]
    from sum_engine_internal.mcp_server.errors import ErrorClass

    async def attest(text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {
                "error_class": ErrorClass.SCHEMA.value,
                "errors": ["text must be a non-empty string"],
            }
        proc = _subprocess.run(
            [_sys.executable, "-m", "sum_cli.main", "attest", "--extractor=sieve"],
            input=text, capture_output=True, text=True, cwd=str(REPO),
        )
        if proc.returncode != 0:
            return {
                "error_class": ErrorClass.INTERNAL.value,
                "errors": [f"attest CLI failed: {proc.stderr.strip()[:300]}"],
            }
        try:
            bundle = _json.loads(proc.stdout)
        except _json.JSONDecodeError as e:
            return {
                "error_class": ErrorClass.INTERNAL.value,
                "errors": [f"attest CLI output not JSON: {e}"],
            }
        return {"bundle": bundle}
    return attest


def _underlying_verify():
    """In-process verify wrapping the CLI verify path."""
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path
    REPO = _Path(__file__).resolve().parents[2]
    from sum_engine_internal.mcp_server.errors import ErrorClass

    async def verify(bundle: dict) -> dict:
        if not isinstance(bundle, dict):
            # If bundle came in as the result of a prior bind-wrapped
            # attest call, the agent should pass the bundle's BIND
            # not the wrapper dict. This branch catches the case
            # where the agent passed the wrapper {bind_id, preview}
            # by mistake — give a typed error.
            if isinstance(bundle, dict) and "bind_id" in bundle:
                return {
                    "error_class": ErrorClass.SCHEMA.value,
                    "errors": [
                        "verify received a {bind_id, preview} wrapper. "
                        "Pass the bind_id string directly, not the wrapper."
                    ],
                }
            return {
                "error_class": ErrorClass.SCHEMA.value,
                "errors": [f"bundle must be a dict; got {type(bundle).__name__}"],
            }
        # The CLI's attest produces a bundle dict that's wrapped in
        # the underlying attest's {bundle: dict} envelope; unwrap if so.
        if "bundle" in bundle and isinstance(bundle["bundle"], dict) and "axiom_count" in bundle["bundle"]:
            bundle = bundle["bundle"]
        bundle_json = _json.dumps(bundle)
        proc = _subprocess.run(
            [_sys.executable, "-m", "sum_cli.main", "verify", "--input", "/dev/stdin"],
            input=bundle_json, capture_output=True, text=True, cwd=str(REPO),
        )
        # CLI prints both human + JSON; the JSON line is the last line of stdout
        last_json: dict[str, Any] = {}
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    last_json = _json.loads(line)
                    break
                except _json.JSONDecodeError:
                    pass
        if proc.returncode != 0:
            return {
                "error_class": ErrorClass.STRUCTURAL.value,
                "errors": [f"verify failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}"],
                "details": last_json,
            }
        return {
            "ok": last_json.get("ok", proc.returncode == 0),
            "axioms": last_json.get("axioms"),
            "state_integer_digits": last_json.get("state_integer_digits"),
            "signatures": last_json.get("signatures"),
        }
    return verify


def _underlying_render():
    """In-process render wrapping the CLI render path. Typed
    precondition errors for the LLM-conditioned-axes case (the
    failure 2 from the agent log)."""
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path
    REPO = _Path(__file__).resolve().parents[2]
    from sum_engine_internal.mcp_server.errors import ErrorClass

    LLM_CONDITIONED_AXES = ("length", "formality", "audience", "perspective")

    async def render(
        bundle: dict,
        density: float = 1.0,
        length: float = 0.5,
        formality: float = 0.5,
        audience: float = 0.5,
        perspective: float = 0.5,
    ) -> dict:
        if not isinstance(bundle, dict):
            return {
                "error_class": ErrorClass.SCHEMA.value,
                "errors": [f"bundle must be a dict; got {type(bundle).__name__}"],
            }
        if "bundle" in bundle and isinstance(bundle["bundle"], dict):
            bundle = bundle["bundle"]
        # Typed precondition: non-neutral LLM-conditioned axes
        # require an external service (the Worker). Surface as
        # SCHEMA error_class with structured per-axis info.
        # Capture into a name in enclosing scope; dict-comprehensions
        # have their own scope and would not see the function params
        # via locals().
        axis_values = {
            "length": length, "formality": formality,
            "audience": audience, "perspective": perspective,
        }
        non_neutral = {
            axis: axis_values[axis]
            for axis in LLM_CONDITIONED_AXES
            if abs(axis_values[axis] - 0.5) > 1e-6
        }
        if non_neutral:
            return {
                "error_class": ErrorClass.SCHEMA.value,
                "errors": [
                    "non-neutral LLM-conditioned axes require an "
                    "external Worker; this bind-spike runs offline only"
                ],
                "structured": {
                    "non_neutral_llm_axes": non_neutral,
                    "neutral_value": 0.5,
                    "axes_supported_offline": ["density"],
                    "axes_requiring_worker": list(LLM_CONDITIONED_AXES),
                    "remedy": "Set length=formality=audience=perspective=0.5 (the neutral default).",
                },
            }
        # Density-only render (offline-safe path)
        cli_args = [_sys.executable, "-m", "sum_cli.main", "render", f"--density={density}"]
        bundle_json = _json.dumps(bundle)
        proc = _subprocess.run(
            cli_args, input=bundle_json, capture_output=True, text=True, cwd=str(REPO),
        )
        if proc.returncode != 0:
            return {
                "error_class": ErrorClass.INTERNAL.value,
                "errors": [f"render failed: {proc.stderr.strip()[:300]}"],
            }
        return {
            "tome": proc.stdout,
            "sliders": {
                "density": density, "length": length, "formality": formality,
                "audience": audience, "perspective": perspective,
            },
            "axiom_count_input": bundle.get("axiom_count"),
        }
    return render


def _underlying_inspect():
    """In-process inspect — bundle metadata without running verification."""
    from sum_engine_internal.mcp_server.errors import ErrorClass

    async def inspect(bundle: dict) -> dict:
        if not isinstance(bundle, dict):
            return {
                "error_class": ErrorClass.SCHEMA.value,
                "errors": [f"bundle must be a dict; got {type(bundle).__name__}"],
            }
        if "bundle" in bundle and isinstance(bundle["bundle"], dict):
            bundle = bundle["bundle"]
        return {
            "axiom_count": bundle.get("axiom_count"),
            "state_integer": bundle.get("state_integer"),
            "branch": bundle.get("branch"),
            "bundle_version": bundle.get("bundle_version"),
            "prime_scheme": bundle.get("prime_scheme"),
            "canonical_format_version": bundle.get("canonical_format_version"),
        }
    return inspect


def bind_wrap_extract(registry: BindRegistry):
    return _bind_wrap("extract", _underlying_extract(), registry)


def bind_wrap_attest(registry: BindRegistry):
    return _bind_wrap("attest", _underlying_attest(), registry)


def bind_wrap_verify(registry: BindRegistry):
    return _bind_wrap("verify", _underlying_verify(), registry)


def bind_wrap_render(registry: BindRegistry):
    return _bind_wrap("render", _underlying_render(), registry)


def bind_wrap_inspect(registry: BindRegistry):
    return _bind_wrap("inspect", _underlying_inspect(), registry)


# ─── manifest for agent self-discovery ───────────────────────────────


BIND_TOOL_MANIFEST: dict[str, Any] = {
    "schema": "sum.agent_surface_manifest.v1",
    "verb": "bind",
    "summary": (
        "Bind-aware MCP tool surface. Every tool returns "
        "{bind_id, preview}; every tool accepts either inline values or "
        "bind: references. The bind_id is sha256:<hex> over "
        "JCS-canonical bytes (deterministic, content-addressed). "
        "The runtime memoises bind_id → object so the agent never "
        "round-trips full payloads."
    ),
    "tools": {
        "extract": {
            "args": {"text": "str"},
            "returns": {"bind_id": "sha256:<hex>", "preview": {"n_triples": "int"}},
            "errors": ["schema", "extractor_unavailable"],
        },
        "attest": {
            "args": {"text": "str"},
            "returns": {"bind_id": "sha256:<hex>", "preview": {
                "axiom_count": "int", "state_integer_short": "str",
            }},
            "errors": ["schema", "internal"],
        },
        "verify": {
            "args": {"bundle": "dict | bind:sha256:<hex>"},
            "returns": {"bind_id": "sha256:<hex>", "preview": {
                "ok": "bool", "axioms": "int",
            }},
            "errors": ["schema", "structural", "signature"],
        },
        "render": {
            "args": {
                "bundle": "dict | bind:sha256:<hex>",
                "density": "float in [0,1] (default 1.0)",
                "length": "float (must be 0.5 in offline mode)",
                "formality": "float (must be 0.5 in offline mode)",
                "audience": "float (must be 0.5 in offline mode)",
                "perspective": "float (must be 0.5 in offline mode)",
            },
            "returns": {"bind_id": "sha256:<hex>", "preview": {
                "tome_chars": "int", "tome_head": "str (first 120 chars)",
            }},
            "errors": ["schema (with structured.axes_requiring_worker)", "internal"],
            "preconditions": {
                "non_neutral_llm_axes_require_worker": (
                    "length / formality / audience / perspective MUST be 0.5 in "
                    "offline mode. Non-neutral values return error_class=schema "
                    "with structured.non_neutral_llm_axes telling you which axes."
                ),
            },
        },
        "inspect": {
            "args": {"bundle": "dict | bind:sha256:<hex>"},
            "returns": {"bind_id": "sha256:<hex>", "preview": "dict (small)"},
            "errors": ["schema"],
        },
    },
}
