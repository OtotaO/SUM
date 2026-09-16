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
them did not agree. Note the scope carefully:
``docs/PROOF_BOUNDARY.md`` 1.3.1 claims the same *bundle bytes* verify
identically in all three runtimes, and that stayed true throughout. What broke
was upstream of it, in how the issuer *derived* the hash it then signed.

This script probes the property directly rather than inferring it from asset
bytes. It posts four triples through the canonical (no LLM) slider transform
and compares the returned ``input_hash`` against three candidate values
computed locally: the code-point ordering, the UTF-16 ordering, and the
as-posted ordering that a Worker which stopped sorting would produce.

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
    1  live hash matches the UTF-16 ordering (deploy predates #500), or the
       unsorted ordering (a different, worse defect). Exit 1 means drift and
       only drift; every error path returns 2.
    2  could not reach the endpoint, or the response matched no candidate

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

# Fullwidth capital A (U+FF21) against an emoji (U+1F600) is the pair that
# separates the orderings: by code point U+FF21 < U+1F600, but by UTF-16 code
# unit 0xD83D < 0xFF21, so their order inverts.
#
# Posted in an order that is none of the candidates, so "sorts by code point",
# "sorts by UTF-16 code unit" and "does not sort at all" each yield a distinct
# hash. A two-row fixture already in code-point order cannot do this: its
# unsorted and code-point hashes coincide, so a Worker that stopped sorting
# would have reported ok.
#
# The first two rows share the subject "a" and differ only in the predicate,
# which is what makes a comparator reading ONLY the first component visible:
# such a comparator leaves those two rows in posted order, so it cannot produce
# the code-point hash and cannot report ok. It aliases into one of the two
# drift buckets rather than getting its own verdict, which is enough -- the
# property being guarded is "agrees with Python", not "names the bug".
TRIPLES = [
    ["a", "\U0001f600", "o"],
    ["a", "Ａ", "o"],
    ["Ａ", "p", "o"],
    ["\U0001f600", "p", "o"],
]


def _candidates() -> dict[str, str]:
    """Return {ordering: hash} for the three hypotheses the probe separates."""
    from sum_engine_internal.infrastructure.jcs import canonicalize

    def digest(rows: list[list[str]]) -> str:
        return hashlib.sha256(canonicalize([list(r) for r in rows])).hexdigest()

    return {
        "code_point": digest(sorted(TRIPLES, key=tuple)),
        "utf16": digest(
            sorted(TRIPLES, key=lambda r: tuple(c.encode("utf-16-be") for c in r))
        ),
        "unsorted": digest(TRIPLES),
    }


def main() -> int:
    base = os.environ.get("SUM_DEMO_URL", DEFAULT_URL).rstrip("/")
    url = f"{base}/api/transform"
    candidates = _candidates()

    if len(set(candidates.values())) != len(candidates):  # pragma: no cover
        print("FAIL: probe triples no longer separate the three orderings")
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
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not probe {url}: {type(exc).__name__}: {exc}")
        return 2

    receipt = payload.get("transform_receipt") or {}
    live = (receipt.get("payload") or {}).get("input_hash")
    if not isinstance(live, str):
        print(f"FAIL: no transform_receipt.payload.input_hash in the response from {url}")
        return 2

    live_hex = live.split("-", 1)[-1]
    print(f"probe:      {url}")
    print(f"live:       {live}")
    for name, value in candidates.items():
        print(f"{name + ':':12}sha256-{value}")

    if live_hex == candidates["code_point"]:
        print("\nok: deploy sorts by Unicode code point, matching Python (PR #500 present)")
        return 0
    if live_hex == candidates["utf16"]:
        print(
            "\nDRIFT: deploy sorts by UTF-16 code unit. It derives input_hash, "
            "triples_hash and source_chain_hash differently from Python for text "
            "mixing supplementary and above-U+DFFF BMP characters. Redeploy the "
            "Worker to pick up worker/src/unicode_order.ts (PR #500)."
        )
        return 1
    if live_hex == candidates["unsorted"]:
        print(
            "\nDRIFT: deploy does not sort the triples at all, so input_hash "
            "depends on caller ordering and no longer identifies the triple set. "
            "This is a different defect from the UTF-16 one and is not fixed by "
            "redeploying #500 alone."
        )
        return 1
    print("\nFAIL: live hash matches no candidate ordering; the canonicalisation changed")
    return 2


if __name__ == "__main__":
    # Exit 1 is reserved for a drift verdict. Anything unexpected -- a failed
    # import in _candidates(), a response whose shape is not what we assume --
    # would otherwise surface as an uncaught traceback, which also exits 1 and
    # would be read as drift by the deploy workflow.
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: unexpected error: {type(exc).__name__}: {exc}")
        sys.exit(2)
