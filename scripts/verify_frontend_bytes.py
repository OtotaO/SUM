#!/usr/bin/env python3
"""Deployment byte-identity guard for the hosted SUM demo.

The Cloudflare Worker serves ``single_file_demo/`` verbatim via its ASSETS
binding (``worker/wrangler.toml`` → ``directory = "../single_file_demo"``);
there is no build step that could legitimately diverge the served files from
the repo. So the *only* way the live demo can drift from the repo is a
missed/lagged ``wrangler deploy`` — which is exactly the failure mode that hid
PR #243's cascade panel from production until it was caught by manually opening
the page.

This script closes that gap for the load-bearing runtime surface — not just the
HTML, but the **in-browser verifier JS the trust triangle depends on**. The
HTML-only check (its original scope) would pass while the live demo ran a stale
``receipt_verifier.js`` (loaded directly by ``index.html``) after a verifier
change shipped to the repo but not to the Worker — exactly the drift the
2026-06-14 hardening introduced (JWKS/array guards in the repo, not yet
deployed). The check covers the whole served verifier surface, including helpers
that ``index.html`` does not itself import (``meaning_receipt_verifier.js``,
``transform_receipt_verifier.js``, ``jcs.js``): a served-but-unloaded asset
drifting is harmless, but a load-bearing dep drifting is the bug this guards
against, and it is cheaper to watch all of them than to track the import graph by
hand. Each asset is fetched from the deploy and its SHA-256 compared to the repo
file's.

Usage:
    python -m scripts.verify_frontend_bytes
    SUM_DEMO_URL=https://sum-demo.<account>.workers.dev python -m scripts.verify_frontend_bytes

Exit codes:
    0  every checked asset is byte-identical to its repo file
    1  drift detected on one or more assets (live lags the repo — redeploy)
    2  could not fetch a live asset (network / DNS / 5xx / 404)

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
DEMO_DIR = Path(__file__).resolve().parent.parent / "single_file_demo"

# (url_path, repo_relpath) for every load-bearing asset the Worker serves and
# the demo executes in the browser. url_path "" is the site root (== index.html).
# Keep this list aligned with index.html's transitive runtime imports + the
# browser verifier surface; a served-but-unloaded helper here is harmless, a
# MISSING runtime dep is the bug this guard exists to catch.
CHECKED_ASSETS: list[tuple[str, str]] = [
    ("", "index.html"),                                   # the page (root)
    ("sum_core_wasm.js", "sum_core_wasm.js"),             # wasm loader (script src)
    ("sum_core.wasm", "sum_core.wasm"),                   # the wasm binary itself
    ("receipt_verifier.js", "receipt_verifier.js"),       # render-receipt verifier (imported by index.html)
    ("meaning_receipt_verifier.js", "meaning_receipt_verifier.js"),    # meaning/perspective browser verifier
    ("transform_receipt_verifier.js", "transform_receipt_verifier.js"),  # transform browser verifier
    ("jcs.js", "jcs.js"),                                 # float-free canonicalizer
    ("vendor/sum-verify-deps.js", "vendor/sum-verify-deps.js"),  # vendored jose + float-capable canonicalize
    ("provenance.js", "provenance.js"),                   # prov_id helper
    ("godel.js", "godel.js"),                             # state-integer helper
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sum-frontend-byte-guard"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (trusted URL)
        return resp.read()


def main() -> int:
    base = os.environ.get("SUM_DEMO_URL", DEFAULT_URL)
    if not base.endswith("/"):
        base += "/"
    print(f"  url base : {base}")

    drift, fetch_err, ok = [], [], []
    for url_path, repo_rel in CHECKED_ASSETS:
        repo_file = DEMO_DIR / repo_rel
        if not repo_file.exists():
            print(f"  [skip] {repo_rel} — repo file missing", file=sys.stderr)
            continue
        repo_digest = _sha256(repo_file.read_bytes())
        try:
            live_bytes = _fetch(base + url_path)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  [FETCH-ERR] {repo_rel}: {exc}", file=sys.stderr)
            fetch_err.append(repo_rel)
            continue
        live_digest = _sha256(live_bytes)
        if live_digest == repo_digest:
            ok.append(repo_rel)
            print(f"  ✓ {repo_rel}")
        else:
            drift.append(repo_rel)
            print(f"  ✗ {repo_rel}  DRIFT (repo {repo_digest[:12]} != live {live_digest[:12]})")

    print(f"\n  {len(ok)} concordant, {len(drift)} drifted, {len(fetch_err)} unfetchable "
          f"(of {len(CHECKED_ASSETS)} assets)")

    if fetch_err:
        print(f"[verify-frontend-bytes] could not fetch: {fetch_err}", file=sys.stderr)
        return 2
    if drift:
        print(
            "[verify-frontend-bytes] DRIFT — the deployed Worker serves stale bytes for: "
            f"{drift}\n"
            "  The live in-browser surface lags the repo. Redeploy with\n"
            "  `cd worker && npx wrangler deploy` from this commit, then re-run.",
            file=sys.stderr,
        )
        return 1
    print("[verify-frontend-bytes] OK — live demo byte-identical to the repo on every checked asset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
