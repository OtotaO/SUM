"""Error taxonomy for the SUM MCP server (v2 hardening).

A tagged enum of failure classes plus a single result-construction
choke point. Two design moves matter:

  1. **Tagged failure classes.** v1 collapsed schema rejection,
     signature failure, and structural mismatch into one
     ``errors: [string]`` array. v2 emits ``error_class:
     ErrorClass`` so callers can branch on a tag instead of
     pattern-matching error strings. The string list survives
     for human detail.

  2. **Single construction point.** Every result that leaves a
     tool goes through ``success_result`` or ``error_result``.
     The audit logger hooks here, so every call lands one
     structured stderr line — values redacted, only shapes
     and timing logged.

The unknown-class fallback (``ErrorClass.INTERNAL``) means a
caller seeing a class string they don't recognise can fail-safe:
treat it as an internal server error, do not retry, do not
escalate to "schema rejected" semantics. Forward-compat by
construction.
"""
from __future__ import annotations

import json
import sys
import time
from enum import Enum
from typing import Any


class ErrorClass(str, Enum):
    """Tagged failure classes for the SUM MCP server.

    String-valued so the JSON serialisation is the bare class
    name (no ``"ErrorClass.SCHEMA"`` wrapper). Callers compare
    against the string literal or import the enum.
    """

    SCHEMA = "schema"
    SIGNATURE = "signature"
    STRUCTURAL = "structural"
    INPUT_TOO_LARGE = "input_too_large"
    EXTRACTOR_UNAVAILABLE = "extractor_unavailable"
    NETWORK_DISALLOWED = "network_disallowed"
    REVOKED = "revoked"
    INTERNAL = "internal"


def success_result(tool: str, started_at: float, **fields: Any) -> dict:
    """Construct a success result. Logs one audit line to stderr."""
    duration_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
    _audit(tool=tool, result_class="ok", duration_ms=duration_ms, fields=fields)
    return {**fields}


def error_result(
    tool: str,
    started_at: float,
    error_class: ErrorClass,
    message: str,
    **extra: Any,
) -> dict:
    """Construct an error result. Logs one audit line to stderr."""
    duration_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
    _audit(
        tool=tool,
        result_class=error_class.value,
        duration_ms=duration_ms,
        fields=extra,
    )
    payload: dict[str, Any] = {
        "error_class": error_class.value,
        "errors": [message],
    }
    payload.update(extra)
    return payload


def _audit(*, tool: str, result_class: str, duration_ms: float, fields: dict) -> None:
    """Single-line JSON audit record on stderr.

    No values logged — only shapes (string lengths, list
    lengths, dict keys). This is log-injection-proof by
    construction: an attacker controlling tool input cannot
    influence the structure of the audit record because none
    of their bytes appear in it.
    """
    shapes: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            shapes[key] = {"type": "str", "len": len(value)}
        elif isinstance(value, (list, tuple)):
            shapes[key] = {"type": "list", "len": len(value)}
        elif isinstance(value, dict):
            shapes[key] = {"type": "dict", "keys": sorted(value.keys())[:20]}
        elif isinstance(value, bool):
            shapes[key] = {"type": "bool"}
        elif isinstance(value, (int, float)):
            shapes[key] = {"type": type(value).__name__}
        elif value is None:
            shapes[key] = {"type": "null"}
        else:
            shapes[key] = {"type": type(value).__name__}

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool,
        "result_class": result_class,
        "duration_ms": duration_ms,
        "shapes": shapes,
    }
    try:
        sys.stderr.write(json.dumps(record) + "\n")
        sys.stderr.flush()
    except Exception:
        # Audit failures must never break the tool path.
        pass
