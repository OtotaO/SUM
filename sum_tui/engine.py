"""Thin adapter from the SUM Workbench TUI to the shipped CLI surface.

The TUI never imports the heavy engine; it shells out, so the UI stays snappy and
dependency-light (only ``textual``) and degrades honestly when the live judge or
an LLM key is absent. The one thing that ALWAYS works offline is replaying the
bundled signed golden — verify + replay are cryptographic facts, no network, no
model.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

DEMO_NOTE = "BillSum binding-gate golden (CC0), replayed offline — verify + replay are REAL."


def _run(cmd: list[str], timeout: int = 90) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", str(exc)


def have_sum_cli() -> bool:
    """True if the `sum` binary is on PATH (live transforms become possible)."""
    return shutil.which("sum") is not None


def run_demo() -> dict:
    """Replay the bundled signed golden offline. Returns the parsed receipt dict.

    On success the dict carries verified/replayed/risk_upper_bound/scorer/n/
    not_covered/proxy_caveat. On failure it carries ``verified: False`` + an error.
    """
    rc, out, err = _run([sys.executable, "-m", "sum_verify", "--demo"])
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                continue
            data["_note"] = DEMO_NOTE
            return data
    return {"verified": False, "error": (err or out or "demo unavailable").strip()[:300]}
