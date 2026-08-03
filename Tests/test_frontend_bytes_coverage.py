"""CHECKED_ASSETS alignment lock for the frontend byte guard.

``scripts/verify_frontend_bytes.py`` only protects the concordance of assets
that appear in its ``CHECKED_ASSETS`` list. Its own contract (the list comment)
is "keep this aligned with index.html's transitive runtime imports" — but until
this test nothing enforced it, so an asset index.html fetches/imports at runtime
could silently miss the list (that is exactly how the two sample_meaning_*.json
files slipped through, 2026-07-31 review #13).

This test parses index.html for its same-origin static references and asserts
each is covered by CHECKED_ASSETS. It is torch-free and network-free.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEMO = _REPO / "single_file_demo"
_INDEX = _DEMO / "index.html"


def _checked_url_paths() -> set[str]:
    import scripts.verify_frontend_bytes as v
    return {url for url, _repo in v.CHECKED_ASSETS}


# Same-origin static references the page pulls at runtime:
#   fetch("./foo.json") / fetch("/foo.json")   — data the page loads
#   import ... from "./foo.js"                 — ES module imports
#   <script src="foo.js">                      — classic scripts
_FETCH_RE = re.compile(r"""fetch\(\s*["']\.?/?([\w./-]+)["']""")
_IMPORT_RE = re.compile(r"""import\s+[^"']*?from\s*["']\.?/?([\w./-]+)["']""")
_SCRIPT_SRC_RE = re.compile(r"""<script[^>]*\bsrc\s*=\s*["']\.?/?([\w./-]+)["']""")

# References that are not repo static assets served by the Worker (same-origin
# API routes handled dynamically, not files). Extend deliberately, with a note.
_NOT_STATIC_ASSETS = {
    "api/complete",
    "api/render",
    "api/transform",
    "api/qid",
}


def _referenced_static_assets() -> set[str]:
    html = _INDEX.read_text("utf-8")
    refs: set[str] = set()
    for rx in (_FETCH_RE, _IMPORT_RE, _SCRIPT_SRC_RE):
        for m in rx.finditer(html):
            ref = m.group(1).lstrip("./")
            if ref in _NOT_STATIC_ASSETS:
                continue
            # Only same-origin repo files (no scheme, no protocol-relative).
            if "//" in ref or ref.startswith("http"):
                continue
            refs.add(ref)
    return refs


def test_every_runtime_referenced_asset_is_checked():
    """Each same-origin static file index.html fetches/imports at runtime must
    be present in CHECKED_ASSETS (as a url_path) AND exist on disk. A new
    reference added without updating the list fails here first."""
    checked = _checked_url_paths()
    referenced = _referenced_static_assets()
    # Only assert on refs that actually exist as repo files — a ref to a file
    # not in the repo is a different bug (a broken link), caught elsewhere; here
    # we lock the guard-coverage invariant for real served assets.
    served = {r for r in referenced if (_DEMO / r).exists()}
    uncovered = sorted(served - checked)
    assert not uncovered, (
        "index.html references these static assets at runtime but they are NOT "
        f"in verify_frontend_bytes.CHECKED_ASSETS: {uncovered}. Add them so the "
        "byte guard covers them (the drift class the guard exists to catch)."
    )


def test_checked_assets_all_exist_on_disk():
    """Every CHECKED_ASSETS repo path must exist — a stale entry means the guard
    silently dropped coverage (now also a runtime failure in the guard itself)."""
    import scripts.verify_frontend_bytes as v
    missing = [repo for _url, repo in v.CHECKED_ASSETS if not (_DEMO / repo).exists()]
    assert not missing, f"CHECKED_ASSETS names repo files that do not exist: {missing}"
