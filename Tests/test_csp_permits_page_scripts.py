"""CSP executability lock for the demo's own scripts.

The byte-concordance guard (``scripts/verify_frontend_bytes.py``) proves the
served bytes equal the repo bytes. It says nothing about whether the browser is
allowed to RUN them, and that gap shipped a real outage: the served CSP carried
``script-src 'unsafe-inline'`` with no ``'self'``, which permits inline blocks
but blocks every external script fetch — including ES-module imports. Both
in-page verify boxes therefore bound no handlers in production (the buttons
rendered and did nothing) while the byte guard stayed green at 13/13, because
nothing was wrong with the bytes.

This test closes that class. It parses index.html for the scripts the page
actually loads and asserts the CSP permits them, in BOTH copies of the header
(the Worker's ``BASELINE_HEADERS`` and the Pages ``_headers`` file, which are a
documented keep-in-sync pair that nothing previously enforced).

Torch-free and network-free: it reads repo files only. It cannot observe the
live site — deploying a stale Worker is a separate concern — but it makes the
header that WILL be deployed provably consistent with the page it serves.
"""
from __future__ import annotations

import re
from pathlib import Path

from Tests.test_frontend_bytes_coverage import _IMPORT_RE, _SCRIPT_SRC_RE

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "single_file_demo" / "index.html"
_HEADERS = _REPO / "single_file_demo" / "_headers"
_WORKER_INDEX = _REPO / "worker" / "src" / "index.ts"

_CSP_LINE_RE = re.compile(r"^\s*Content-Security-Policy:\s*(?P<csp>.+?)\s*$", re.M)


def _external_script_refs() -> set[str]:
    """Same-origin scripts index.html loads: classic <script src> + module imports.

    These are the references governed by ``script-src``. Data the page fetches
    (JSON) is governed by ``connect-src`` and is deliberately out of scope here.
    """
    html = _INDEX.read_text("utf-8")
    refs: set[str] = set()
    for rx in (_SCRIPT_SRC_RE, _IMPORT_RE):
        for m in rx.finditer(html):
            ref = m.group(1).lstrip("./")
            if "//" in ref or ref.startswith("http"):
                continue  # cross-origin would need an explicit host-source
            refs.add(ref)
    return refs


def _headers_csp() -> str:
    m = _CSP_LINE_RE.search(_HEADERS.read_text("utf-8"))
    assert m, "no Content-Security-Policy line found in single_file_demo/_headers"
    return m.group("csp")


def _worker_csp() -> str:
    """Reconstruct the CSP the Worker sends from its BASELINE_HEADERS array."""
    ts = _WORKER_INDEX.read_text("utf-8")
    m = re.search(
        r'"Content-Security-Policy":\s*\[(?P<body>.*?)\]\.join\("; "\)',
        ts,
        re.S,
    )
    assert m, "could not parse the Content-Security-Policy array in worker/src/index.ts"
    parts = re.findall(r'"([^"]+)"', m.group("body"))
    return "; ".join(parts)


def _directive(csp: str, name: str) -> list[str]:
    for chunk in csp.split(";"):
        tokens = chunk.split()
        if tokens and tokens[0] == name:
            return tokens[1:]
    return []


def test_worker_and_pages_csp_stay_in_sync():
    """The two header copies are documented as a keep-in-sync pair. Enforce it —
    a comment is not a guard, and they had already drifted in intent."""
    assert _worker_csp() == _headers_csp(), (
        "worker/src/index.ts and single_file_demo/_headers declare different "
        "Content-Security-Policy values; they must stay byte-identical.\n"
        f"  worker : {_worker_csp()}\n"
        f"  _headers: {_headers_csp()}"
    )


def test_csp_permits_the_pages_own_scripts():
    """If index.html loads any same-origin script, script-src must allow it.

    ``'unsafe-inline'`` alone does NOT: it whitelists inline blocks only and
    never permits a fetch, for classic tags and module imports alike. This is
    the exact assertion that would have caught the dead verify boxes.
    """
    refs = _external_script_refs()
    assert refs, (
        "expected index.html to load at least one same-origin script; if that is "
        "no longer true, this guard needs revisiting rather than deleting"
    )

    for csp_name, csp in (("worker", _worker_csp()), ("_headers", _headers_csp())):
        sources = _directive(csp, "script-src")
        assert sources, f"{csp_name}: no script-src directive; default-src would block {sorted(refs)}"
        permits_same_origin = "'self'" in sources or any(
            s.startswith(("'nonce-", "'sha256-", "'sha384-", "'sha512-")) for s in sources
        )
        assert permits_same_origin, (
            f"{csp_name}: script-src={sources!r} permits no same-origin script fetch, "
            f"but index.html loads {sorted(refs)}. Those scripts will be BLOCKED in the "
            "browser and any module block importing them never evaluates (the verify "
            "boxes bind no handlers). Add 'self'."
        )


def test_csp_permits_wasm_compilation_while_default_src_is_none():
    """The WASM fast path needs 'wasm-unsafe-eval' explicitly once default-src is
    'none' — otherwise WebAssembly compilation is blocked even with 'self'."""
    html = _INDEX.read_text("utf-8")
    if "WebAssembly" not in html:
        return  # page no longer compiles WASM; nothing to assert
    for csp_name, csp in (("worker", _worker_csp()), ("_headers", _headers_csp())):
        sources = _directive(csp, "script-src")
        assert "'wasm-unsafe-eval'" in sources, (
            f"{csp_name}: index.html compiles WebAssembly but script-src={sources!r} "
            "lacks 'wasm-unsafe-eval'; with default-src 'none' the WASM path is blocked."
        )


def test_referenced_scripts_exist_on_disk():
    """A script the CSP now permits should also be a file we actually ship."""
    demo = _REPO / "single_file_demo"
    missing = sorted(r for r in _external_script_refs() if not (demo / r).exists())
    assert not missing, f"index.html loads scripts that do not exist in the repo: {missing}"
