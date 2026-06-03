#!/usr/bin/env python3
"""Deployment byte-identity guard for the hosted SUM demo.

The Cloudflare Worker serves ``single_file_demo/index.html`` verbatim via
its ASSETS binding (``worker/wrangler.toml`` → ``directory = "../single_file_demo"``);
there is no build step that could legitimately diverge the two. So the
*only* way the live page can drift from the repo file is a missed/lagged
``wrangler deploy`` — which is exactly the failure mode that hid PR #243's
cascade panel from production until it was caught by manually opening the
page.

This script closes that gap with an automated check: fetch the deployed
``/`` and assert its SHA-256 equals the repo file's. Run it after every
deploy, or wire it into a (non-blocking, network-dependent) CI step.

Usage:
    python -m scripts.verify_frontend_bytes
    SUM_DEMO_URL=https://sum-demo.<account>.workers.dev python -m scripts.verify_frontend_bytes

Exit codes:
    0  live bytes are byte-identical to single_file_demo/index.html
    1  drift detected (live page lags the repo — redeploy needed)
    2  could not fetch the live page (network / DNS / 5xx)

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://sum-demo.ototao.workers.dev/"
REPO_FILE = Path(__file__).resolve().parent.parent / "single_file_demo" / "index.html"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    url = os.environ.get("SUM_DEMO_URL", DEFAULT_URL)
    if not url.endswith("/"):
        url += "/"

    if not REPO_FILE.exists():
        print(f"[verify-frontend-bytes] repo file missing: {REPO_FILE}", file=sys.stderr)
        return 2
    repo_bytes = REPO_FILE.read_bytes()
    repo_digest = _sha256(repo_bytes)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sum-frontend-byte-guard"})
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (trusted URL)
            live_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[verify-frontend-bytes] could not fetch {url}: {exc}", file=sys.stderr)
        return 2

    live_digest = _sha256(live_bytes)

    print(f"  url           : {url}")
    print(f"  repo file     : single_file_demo/index.html ({len(repo_bytes)} bytes)")
    print(f"  repo sha256   : {repo_digest}")
    print(f"  live bytes    : {len(live_bytes)} bytes")
    print(f"  live sha256   : {live_digest}")

    if live_digest == repo_digest:
        print("[verify-frontend-bytes] OK — live demo is byte-identical to the repo file.")
        return 0

    print(
        "[verify-frontend-bytes] DRIFT — live demo does NOT match the repo file.\n"
        "  The deployed Worker is serving stale frontend bytes. Redeploy with\n"
        "  `cd worker && npx wrangler deploy` from this commit, then re-run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
