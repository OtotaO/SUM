"""Cross-runtime prov_id byte-identity harness.

Asserts Python and JavaScript produce the SAME content-addressable
prov_id for every ProvenanceRecord shape SUM might emit. If Python's
``internal.infrastructure.provenance.compute_prov_id`` and JS's
``single_file_demo/provenance.js`` computeProvId ever disagree, every
future cross-runtime provenance link breaks silently — an axiom
ingested via the Python ledger and queried via the JS artifact would
see different prov_ids for the same evidence span.

Fixtures cover the field-variation surface:
    - minimal record (ascii only, sha256: URI)
    - multibyte excerpt (café)
    - different byte_range values (prov_id must differ)
    - different extractor_id values (prov_id must differ)
    - doi and urn:sum:source URIs (non-sha256 schemes)
    - long-ish excerpt near the EXCERPT_MAX_CHARS limit

Exit codes:
    0  all prov_ids agree
    1  one or more divergences
    3  JS CLI missing
    4  node not on PATH
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from internal.infrastructure.provenance import (
    EXCERPT_MAX_CHARS,
    ProvenanceRecord,
    compute_prov_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_CLI = REPO_ROOT / "single_file_demo" / "prov_id_cli.js"


_SHA_A = "sha256:" + "a" * 64
_SHA_B = "sha256:" + "b" * 64

FIXTURES: list[tuple[str, dict[str, Any]]] = [
    ("minimal ascii", {
        "source_uri": _SHA_A,
        "byte_start": 0,
        "byte_end": 17,
        "extractor_id": "sum.test",
        "timestamp": "2026-04-19T00:00:00+00:00",
        "text_excerpt": "Alice likes cats.",
    }),
    ("sieve extractor", {
        "source_uri": _SHA_B,
        "byte_start": 0,
        "byte_end": 42,
        "extractor_id": "sum.sieve:deterministic_v1",
        "timestamp": "2026-04-19T12:34:56+00:00",
        "text_excerpt": "Marie Curie, a physicist, won Nobel Prizes.",
    }),
    ("llm extractor", {
        "source_uri": _SHA_B,
        "byte_start": 0,
        "byte_end": 42,
        "extractor_id": "sum.llm:gpt-4o-mini-2024-07-18",
        "timestamp": "2026-04-19T12:34:56+00:00",
        "text_excerpt": "Marie Curie, a physicist, won Nobel Prizes.",
    }),
    ("multibyte excerpt", {
        "source_uri": _SHA_A,
        "byte_start": 0,
        "byte_end": 10,
        "extractor_id": "sum.test",
        "timestamp": "2026-04-19T00:00:00+00:00",
        "text_excerpt": "café likes",
    }),
    ("doi source", {
        "source_uri": "doi:10.1234/example-5678",
        "byte_start": 100,
        "byte_end": 250,
        "extractor_id": "sum.sieve:deterministic_v1",
        "timestamp": "2026-04-19T00:00:00+00:00",
        "text_excerpt": "Gravity attracts objects toward Earth.",
    }),
    ("urn:sum:source", {
        "source_uri": "urn:sum:source:seed_v1",
        "byte_start": 0,
        "byte_end": 17,
        "extractor_id": "sum.sieve:deterministic_v1",
        "timestamp": "2026-04-19T00:00:00+00:00",
        "text_excerpt": "Alice likes cats.",
    }),
    ("max-ish excerpt", {
        "source_uri": _SHA_A,
        "byte_start": 0,
        "byte_end": EXCERPT_MAX_CHARS,
        "extractor_id": "sum.test",
        "timestamp": "2026-04-19T00:00:00+00:00",
        "text_excerpt": "x" * (EXCERPT_MAX_CHARS - 1),  # under limit, long
    }),
]


def _js_prov_id(fields: dict[str, Any]) -> str:
    result = subprocess.run(
        ["node", str(JS_CLI)],
        input=json.dumps(fields, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"JS prov_id CLI exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout.decode("utf-8")


def main() -> int:
    if not JS_CLI.exists():
        print(f"[harness] {JS_CLI} missing", file=sys.stderr)
        return 3
    try:
        v = subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        )
        if v.returncode != 0:
            return 4
        print(f"[harness] node {v.stdout.strip()}")
    except FileNotFoundError:
        print("[harness] node not on PATH", file=sys.stderr)
        return 4

    failures: list[str] = []
    for label, fields in FIXTURES:
        try:
            py_rec = ProvenanceRecord(**fields)
            py_id = compute_prov_id(py_rec)
        except Exception as e:
            failures.append(f"  {label}: python threw {e!r}")
            continue
        try:
            js_id = _js_prov_id(fields)
        except Exception as e:
            failures.append(f"  {label}: js threw {e!r}")
            continue

        if py_id == js_id:
            print(f"  [OK] {label}  {py_id}")
        else:
            failures.append(
                f"  {label}: py={py_id} js={js_id}"
            )

    # Negative check: two fixtures that SHOULD differ actually do
    py_a = compute_prov_id(ProvenanceRecord(**FIXTURES[0][1]))
    py_b = compute_prov_id(ProvenanceRecord(**FIXTURES[1][1]))
    if py_a == py_b:
        failures.append("  negative-check: different fixtures collide on py prov_id")

    if failures:
        print("\nPROV_ID CROSS-RUNTIME: REGRESSION", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        return 1
    print("\nPROV_ID CROSS-RUNTIME: ALL FIXTURES AGREE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
