"""JCS cross-runtime byte-identity harness.

For every fixture in a deliberately-tricky corpus, canonicalize in Python
(``sum_engine_internal.infrastructure.jcs.canonicalize``) and in JavaScript
(``single_file_demo/jcs.js`` via ``node``) and assert the emitted UTF-8
byte sequences are equal. Any divergence is a correctness regression
before merge — the load-bearing assumption for every signed bundle,
every prov_id, and every eventual VC 2.0 credential that the two
implementations agree on canonical form.

Fixtures are selected to probe the known difficult cases:
  - Key-sort edge (BMP vs supplementary-plane code points)
  - Nested objects (recursive sort)
  - Empty containers
  - Control-character escapes (short forms + lowercase \\uXXXX)
  - Unicode passthrough (café — multibyte UTF-8)
  - JSON-parse idempotence

If a future implementation diverges on any of these, the divergence is
reported with the specific input and the Python vs JS byte counts so the
gap can be debugged immediately rather than surfaced as a mysterious
signature-mismatch downstream.

Exit codes:
    0  all fixtures agree
    1  one or more byte-level divergences
    3  JS runner missing from the repo
    4  node not installed on PATH
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sum_engine_internal.infrastructure.jcs import canonicalize

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_RUNNER = REPO_ROOT / "single_file_demo" / "jcs_cli.js"


FIXTURES: list[tuple[str, Any]] = [
    ("null",                 None),
    ("true",                  True),
    ("false",                 False),
    ("zero",                  0),
    ("positive int",          42),
    ("negative int",          -7),
    # Integers on the wire stay within Number's safe range: SUM emits the
    # Gödel state integer as a STRING and receipt micro-units are ≤ 1e6, so
    # the >2^53 Python-int-exact / JS-Number-lossy divergence is never
    # reachable. We test agreement at the exact safe boundary.
    ("max safe integer",      2**53 - 1),
    # NOTE: floats are intentionally NOT tested here — this harness uses the
    # float-THROWING jcs_cli.js (the meaning/perspective family's float-free
    # enforcer). The float-capable render/transform canonicalization path
    # (Python jcs.py ↔ Erdtman canonicalize) is covered, including the
    # previously-divergent sub-1e-4 boundary, by
    # Tests/test_jcs_float_cross_runtime.py.
    ("empty string",          ""),
    ("ascii",                 "hello"),
    ("escape quote",          'a"b'),
    ("escape backslash",      "a\\b"),
    ("solidus not escaped",   "a/b"),
    ("short controls",        "\b\t\n\f\r"),
    ("hex control 0x01",      "\u0001"),
    ("unicode cafe",          "café"),
    ("empty object",          {}),
    ("single pair",           {"a": 1}),
    ("keys out of order",     {"b": 1, "a": 2}),
    ("nested",                {"outer": {"b": 1, "a": 2}}),
    ("empty array",           []),
    ("mixed array",           [1, "a", True, None]),
    ("arr of objects",        [{"b": 1}, {"a": 2}]),
    ("surrogate-pair sort",   {"\uE000": 1, "\U0001F600": 2}),
    ("deep nesting",          {"a": {"b": {"c": {"d": [1, {"e": True}]}}}}),
    ("realistic VC subject",  {
        "@context": ["https://www.w3.org/ns/credentials/v2"],
        "type": ["VerifiableCredential", "SumAxiomAttestation"],
        "issuer": "did:example:issuer",
        "credentialSubject": {
            "axiom": "alice||like||cat",
            "stateInteger": "12345",
            "primeScheme": "sha256_64_v1",
        },
        "validFrom": "2026-04-19T00:00:00Z",
    }),
    ("realistic provenance",  {
        "source_uri": "sha256:" + "a" * 64,
        "byte_start": 0,
        "byte_end": 17,
        "extractor_id": "sum.sieve:deterministic_v1",
        "timestamp": "2026-04-19T00:00:00+00:00",
        "text_excerpt": "Alice likes cats.",
        "schema_version": "1.0.0",
    }),
]


def _js_canonicalize(obj: Any) -> bytes:
    """Invoke the JS canonicalizer on an object serialized as JSON.

    The JS CLI reads one JSON value from stdin and writes the canonical
    UTF-8 bytes to stdout (raw, no trailing newline). This gives us a
    byte-exact return without having to escape anything through the
    shell.
    """
    result = subprocess.run(
        ["node", str(JS_RUNNER)],
        input=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"JS JCS runner exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def main() -> int:
    if not JS_RUNNER.exists():
        print(f"[harness] {JS_RUNNER} missing", file=sys.stderr)
        return 3

    try:
        node_v = subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        )
        if node_v.returncode != 0:
            return 4
        print(f"[harness] node {node_v.stdout.strip()}")
    except FileNotFoundError:
        print("[harness] node not on PATH", file=sys.stderr)
        return 4

    failures: list[str] = []
    for label, fixture in FIXTURES:
        payload: Any = fixture

        try:
            py_bytes = canonicalize(payload)
        except Exception as e:
            failures.append(f"  {label}: python threw {e!r}")
            continue

        try:
            js_bytes = _js_canonicalize(payload)
        except Exception as e:
            failures.append(f"  {label}: js threw {e!r}")
            continue

        if py_bytes == js_bytes:
            print(f"  [OK] {label}  ({len(py_bytes)} bytes)")
        else:
            failures.append(
                f"  {label}: py={py_bytes!r} ({len(py_bytes)}B) "
                f"js={js_bytes!r} ({len(js_bytes)}B)"
            )

    if failures:
        print("\nJCS BYTE-IDENTITY: REGRESSION", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        return 1
    print("\nJCS BYTE-IDENTITY: ALL FIXTURES AGREE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
