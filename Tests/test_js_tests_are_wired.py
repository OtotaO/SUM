"""Every browser JS test file must run somewhere in CI.

Four files under single_file_demo/ (JCS, provenance, replay window, transform
verify) ran in no workflow until 2026-10-23; docs/FEATURE_CATALOG.md named
them, and a Makefile target one of them, only as manual commands, and no
workflow calls make. A test that never runs guards
nothing, so this fails when a single_file_demo test file is not named by a
workflow or by the package.json test script that CI runs.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEMO = _REPO / "single_file_demo"


def _ci_text() -> str:
    workflows = "\n".join(p.read_text("utf-8") for p in sorted((_REPO / ".github" / "workflows").glob("*.yml")))
    scripts = json.loads((_DEMO / "package.json").read_text("utf-8")).get("scripts", {})
    return workflows + "\n" + "\n".join(scripts.values())


def test_every_demo_js_test_runs_in_ci():
    ci = _ci_text()
    tests = sorted(p.name for p in _DEMO.iterdir() if p.name.startswith("test_") and p.suffix in {".js", ".mjs"})
    assert tests, "no demo JS tests found"
    unwired = [name for name in tests if name not in ci]
    assert not unwired, f"JS tests that no workflow or package.json script runs: {unwired}"
