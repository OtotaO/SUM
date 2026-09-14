#!/usr/bin/env python3
"""Behavioural deploy guard for the Worker's Unicode sort order.

``scripts/verify_frontend_bytes.py`` compares served static assets to the repo
and so detects that a deploy is *stale*. It cannot detect what a stale deploy
gets *wrong*, because the Worker script itself is bundled and never served as
bytes. On 2026-09-11 that gap hid a live correctness defect: the deployed build
predated PR #500, which added ``worker/src/unicode_order.ts``.

The defect. Python's ``str`` comparison orders by Unicode code point; JavaScript
``<`` orders by UTF-16 code unit. The two disagree whenever a supplementary
character (U+10000 and above, encoded as a surrogate pair beginning 0xD800) is
compared against a BMP character above U+DFFF -- fullwidth forms U+FF01..U+FF5E
(ordinary in Japanese and Chinese), the replacement character U+FFFD that
encoding errors leave behind, CJK compatibility ideographs, the private use
area. Before #500 the Worker sorted triples and source-chain links with bare
``<``, so for such text it derived ``input_hash``, ``triples_hash`` and
``source_chain_hash`` differently from the Python implementation. The signature
over those fields still verified; an independent Python verifier recomputing
them did not agree. That is the cross-runtime trust triangle of
``docs/PROOF_BOUNDARY.md`` 1.3.1 failing for one input class.

This script probes the property directly rather than inferring it from asset
bytes. It posts two triples whose relative order differs between the two
orderings through the canonical (no LLM) slider transform, and compares the
returned ``input_hash`` against both candidate values computed locally.

Note this is not a JCS bug and the JCS layer is not what is probed here. RFC
8785 3.2.3 requires object *keys* be sorted by UTF-16 code unit, and both
implementations do that correctly -- ``infrastructure/jcs.py`` via
``encode("utf-16-be")``, ``single_file_demo/jcs.js`` via default ``sort()``.
Only the application-level sorts over triples and source-chain links must
mirror Python's code-point order.

Usage:
    python -m scripts.verify_worker_unicode_order
    SUM_DEMO_URL=https://sum-demo.<account>.workers.dev python -m scripts.verify_worker_unicode_order

Exit codes:
    0  live hash matches the code-point ordering (deploy carries the #500 fix)
    1  live hash matches the UTF-16 ordering (deploy predates #500 -- redeploy)
    2  could not reach the endpoint, or the response matched neither candidate

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://sum-demo.ototao.workers.dev"
TIMEOUT_S = 30

# Fullwidth capital A (U+FF21) against an emoji (U+1F600). By code point
# U+FF21 < U+1F600; by UTF-16 code unit 0xD83D < 0xFF21, so the order inverts.
TRIPLES = [["Ａ", "p", "o"], ["\U0001f600", "p", "o"]]


def _candidates() -> tuple[str, str]:
    """Return (code_point_hash, utf16_hash) for the probe triples."""
    from sum_engine_internal.infrastructure.jcs import canonicalize

    def digest(rows: list[list[str]]) -> str:
        return hashlib.sha256(canonicalize([list(r) for r in rows])).hexdigest()

    by_code_point = sorted(TRIPLES, key=tuple)
    by_utf16 = sorted(
        TRIPLES, key=lambda r: tuple(c.encode("utf-16-be") for c in r)
    )
    return digest(by_code_point), digest(by_utf16)


def main() -> int:
    base = os.environ.get("SUM_DEMO_URL", DEFAULT_URL).rstrip("/")
    url = f"{base}/api/transform"
    code_point_hash, utf16_hash = _candidates()

    if code_point_hash == utf16_hash:  # pragma: no cover - guards the fixture
        print("FAIL: probe triples no longer distinguish the two orderings")
        return 2

    body = json.dumps(
        {
            "transform": "slider",
            "input": {"triples": TRIPLES},
            # Canonical path: every LLM axis at 0.5 means no provider call.
            "parameters": {
                "density": 0.5,
                "length": 0.5,
                "formality": 0.5,
                "audience": 0.5,
                "perspective": 0.5,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "content-type": "application/json",
            # A bare Python-urllib UA is refused with 403 by the edge.
            "User-Agent": "sum-worker-unicode-order-guard",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"FAIL: could not probe {url}: {exc}")
        return 2

    receipt = payload.get("transform_receipt") or {}
    live = (receipt.get("payload") or {}).get("input_hash")
    if not isinstance(live, str):
        print(f"FAIL: no transform_receipt.payload.input_hash in the response from {url}")
        return 2

    live_hex = live.split("-", 1)[-1]
    print(f"probe:      {url}")
    print(f"live:       {live}")
    print(f"code point: sha256-{code_point_hash}")
    print(f"utf-16:     sha256-{utf16_hash}")

    if live_hex == code_point_hash:
        print("\nok: deploy sorts by Unicode code point, matching Python (PR #500 present)")
        return 0
    if live_hex == utf16_hash:
        print(
            "\nDRIFT: deploy sorts by UTF-16 code unit. It derives input_hash, "
            "triples_hash and source_chain_hash differently from Python for text "
            "mixing supplementary and above-U+DFFF BMP characters. Redeploy the "
            "Worker to pick up worker/src/unicode_order.ts (PR #500)."
        )
        return 1
    print("\nFAIL: live hash matches neither candidate; the canonicalisation changed")
    return 2


if __name__ == "__main__":
    sys.exit(main())
