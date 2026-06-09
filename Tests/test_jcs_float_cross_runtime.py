"""Cross-runtime byte-identity for FLOAT canonicalization — the render /
transform receipt signature path.

Render/transform receipts carry float slider values (``sliders_quantized``)
and are signed/verified over the **Erdtman ``canonicalize``** bytes (the
float-capable JS canonicalizer, which delegates to ``JSON.stringify`` →
``Number.prototype.toString``). So Python's ``jcs.py`` MUST emit the exact
ECMAScript ``Number::toString`` form (RFC 8785 §3.2.2.3) for every float, or a
JS-signed receipt fails Python verification.

Regression for the audit finding (2026-06-09): ``jcs.py`` previously fell back
to ``repr(f)``, which diverges from ECMAScript at the exponential-notation
boundary — e.g. ``density=1e-6`` → Python ``1e-06`` vs JS ``0.000001``. The
Worker passes ``density`` through unsnapped, so a value < 1e-4 was a *reachable*
in-range slider that made a Worker-signed render receipt fail the Python
verifier. ``_ecmascript_number_to_string`` fixes it; this test pins it cross-
runtime against the real Erdtman library.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sum_engine_internal.infrastructure.jcs import canonicalize

_REPO = Path(__file__).resolve().parent.parent
_ERDTMAN = _REPO / "single_file_demo" / "vendor" / "sum-verify-deps.js"

# The previously-divergent boundary + the slider grid + representative decimals.
_FLOAT_CASES = [
    0.5, 0.3, 0.1, 0.7, 0.9, 0.123, 1.5, 3.14159, 0.2, 0.25, 0.999999,
    1e-4, 9.9e-5, 5e-5, 2e-5, 1e-5, 1e-6, 0.000001, 1e-7, 5e-7, 0.0000005,
    0.00012345, -0.5, -1e-6, -0.000123,
]
_PAYLOAD_CASES = [
    # the exact shape + the exact value that broke the Worker→Python chain
    {"render_id": "r", "sliders_quantized": {
        "density": 1e-6, "length": 0.5, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5},
     "model": "m", "signed_at": "2026-06-09T00:00:00.000Z"},
    {"density": 1e-7, "length": 0.123},
]


def _erdtman_canonicalize(obj) -> bytes:
    src = (
        'import {canonicalize} from "./single_file_demo/vendor/sum-verify-deps.js";'
        'let d="";process.stdin.on("data",c=>d+=c);'
        'process.stdin.on("end",()=>process.stdout.write(canonicalize(JSON.parse(d))));'
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", src],
        input=json.dumps(obj).encode("utf-8"),
        capture_output=True, cwd=str(_REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    return proc.stdout


_skip = pytest.mark.skipif(
    shutil.which("node") is None or not _ERDTMAN.exists(),
    reason="node or the vendored Erdtman canonicalize unavailable",
)


@_skip
@pytest.mark.parametrize("value", _FLOAT_CASES)
def test_scalar_float_byte_identical(value):
    """Each float canonicalizes byte-identically in Python and JS (Erdtman)."""
    assert canonicalize(value) == _erdtman_canonicalize(value), (
        f"float {value!r} diverged: python={canonicalize(value)!r}"
    )


@_skip
@pytest.mark.parametrize("payload", _PAYLOAD_CASES)
def test_float_bearing_payload_byte_identical(payload):
    assert canonicalize(payload) == _erdtman_canonicalize(payload)


@_skip
def test_the_original_density_1e6_bug_is_fixed():
    """The exact reproduction from the audit: a render payload with
    density=1e-6 must canonicalize identically (so the Worker-signed
    signature verifies in Python)."""
    payload = _PAYLOAD_CASES[0]
    py = canonicalize(payload)
    assert b'"density":0.000001' in py        # ECMAScript decimal form, not 1e-06
    assert py == _erdtman_canonicalize(payload)
