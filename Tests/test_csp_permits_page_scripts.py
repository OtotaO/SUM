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

import pytest

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
            ref = m.group(1)
            if "//" in ref or ref.startswith("http"):
                continue  # cross-origin would need an explicit host-source
            # removeprefix, not lstrip: lstrip strips CHARACTERS, so "../x.js"
            # and "/static/x.js" would silently collapse to a different path.
            if ref.startswith("./"):
                ref = ref[2:]
            refs.add(ref)
    return refs


def _headers_csp() -> str:
    """The CSP from _headers. Fails if the file grows a second CSP line: this
    file supports multiple path blocks, and a later block overriding the policy
    would be invisible to a first-match-only read."""
    found = _CSP_LINE_RE.findall(_HEADERS.read_text("utf-8"))
    assert found, "no Content-Security-Policy line found in single_file_demo/_headers"
    assert len(found) == 1, (
        f"_headers declares {len(found)} Content-Security-Policy lines; this guard "
        "assumes one global policy. Teach it about per-path blocks before adding more."
    )
    return found[0].strip()


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


def _directives(csp: str, name: str) -> list[list[str]]:
    """EVERY occurrence of a directive, so duplicates are visible.

    A policy that repeats a directive keeps the FIRST and ignores the rest, so a
    trailing ``script-src 'none'`` looks reassuring in the string and does
    nothing, while a trailing permissive copy looks permissive and does nothing.
    Either way the file no longer says what it appears to say, so we reject it.
    """
    out = []
    for chunk in csp.split(";"):
        tokens = chunk.split()
        if tokens and tokens[0] == name:
            out.append(tokens[1:])
    return out


def _directive(csp: str, name: str) -> list[str]:
    got = _directives(csp, name)
    return got[0] if got else []


def _permits_same_origin_fetch(sources: list[str]) -> tuple[bool, str]:
    """Does this source list permit fetching a same-origin script?

    Deliberately strict, and NOT a token sniff:
      * ``'self'`` is the only expression here that permits a same-origin fetch.
      * ``'strict-dynamic'`` makes browsers IGNORE ``'self'`` and every
        host-source, so its presence revokes the permission.
      * a nonce permits only elements carrying a matching ``nonce=`` attribute,
        and a hash permits only content whose digest matches. Treating either as
        proof of same-origin permission is exactly the bug this guard exists to
        prevent, so they count only when the page really carries nonces.
    """
    if "'strict-dynamic'" in sources:
        return False, "'strict-dynamic' present: browsers ignore 'self' and host-sources"
    if "'self'" in sources:
        return True, ""
    if any(s.startswith("'nonce-") for s in sources):
        if "nonce=" in _INDEX.read_text("utf-8"):
            return True, ""
        return False, "nonce source present but index.html carries no nonce= attributes"
    if any(s.startswith(("'sha256-", "'sha384-", "'sha512-")) for s in sources):
        return False, "hash sources do not permit external script fetches here"
    return False, "no 'self', no usable nonce"


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
        for directive in ("script-src", "script-src-elem"):
            occurrences = _directives(csp, directive)
            assert len(occurrences) <= 1, (
                f"{csp_name}: {directive} appears {len(occurrences)} times; only the "
                "first is honoured, so the policy does not mean what it reads as."
            )

        sources = _directive(csp, "script-src")
        assert sources, (
            f"{csp_name}: no script-src directive; default-src governs and would "
            f"block {sorted(refs)}"
        )

        # script-src-elem, when present, OVERRIDES script-src for element loads
        # and module imports. Checking only script-src would let a permissive
        # script-src sit next to a restrictive -elem and still read as green.
        elem = _directive(csp, "script-src-elem")
        governing, label = (elem, "script-src-elem") if elem else (sources, "script-src")

        ok, why = _permits_same_origin_fetch(governing)
        assert ok, (
            f"{csp_name}: {label}={governing!r} permits no same-origin script fetch "
            f"({why}), but index.html loads {sorted(refs)}. Those scripts will be "
            "BLOCKED in the browser, and any module block importing them never "
            "evaluates, so the verify boxes bind no handlers. Add 'self'."
        )


def test_csp_permits_wasm_compilation():
    """WebAssembly compilation needs 'wasm-unsafe-eval' (or 'unsafe-eval') in
    script-src. 'self' does not grant it.

    Looks for real compilation calls in the scripts the page actually loads, not
    for the bare word in index.html: index.html mentions WebAssembly only in a
    comment, so keying on that would let a reworded comment silently disarm the
    guard while the WASM path stayed live. Skips (visibly) rather than returning
    green if the page ever stops compiling WASM.
    """
    demo = _REPO / "single_file_demo"
    call_re = re.compile(r"WebAssembly\.(instantiateStreaming|instantiate|compile)")
    compiles = any(
        call_re.search((demo / ref).read_text("utf-8"))
        for ref in _external_script_refs()
        if (demo / ref).exists()
    )
    if not compiles:
        pytest.skip("no WebAssembly compilation in the page's loaded scripts")

    for csp_name, csp in (("worker", _worker_csp()), ("_headers", _headers_csp())):
        sources = _directive(csp, "script-src")
        assert "'wasm-unsafe-eval'" in sources or "'unsafe-eval'" in sources, (
            f"{csp_name}: the page compiles WebAssembly but script-src={sources!r} "
            "grants neither 'wasm-unsafe-eval' nor 'unsafe-eval', so compilation is "
            "blocked. Note 'self' does NOT cover WASM compilation."
        )


def test_referenced_scripts_exist_on_disk():
    """A script the CSP now permits should also be a file we actually ship."""
    demo = _REPO / "single_file_demo"
    missing = sorted(r for r in _external_script_refs() if not (demo / r).exists())
    assert not missing, f"index.html loads scripts that do not exist in the repo: {missing}"
