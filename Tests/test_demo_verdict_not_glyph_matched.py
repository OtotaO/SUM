"""The demo's security verdicts must never be re-derived from display text.

`single_file_demo/index.html` carries the live front door's only in-page
cryptographic gate. It previously decided pass/fail like this::

    if (signatures.ed25519 && signatures.ed25519.startsWith('✗')) {

where the inspected value was a **user-facing sentence** built 35 lines
earlier::

    out.ed25519 = '✗ INVALID — signature does not match public key';

`verifyEd25519InBrowser` already returned a clean enum
(``'verified' | 'invalid' | 'unsupported' | 'absent'``, its own contract
comment says so) and the code computed it correctly, then threw it away and
kept only the prose. So the fail-closed branch matched on the *first glyph of a
label*: rewording, localising, or re-glyphing that sentence would silently turn
a checked-and-FAILED signature into ``ok: true``. A fail-open in the page's only
crypto gate, reachable by a copy edit.

Nothing caught it. `test_csp_permits_page_scripts.py` checks CSP directives,
`test_frontend_bytes_coverage.py` checks static-asset reference coverage, and
`test_altitude_panel_data.py` checks the altitude JSON shape. None executes the
inline script, and `index.html` has no behavioural coverage at all.

These are deliberately STATIC checks over the source text. A real browser
harness for the front door is a bigger piece of work and is not what this pin
is for. The point is narrow: a security decision must read the machine-readable
`status`, never the human-readable `label`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "single_file_demo" / "index.html"


@pytest.fixture(scope="module")
def source() -> str:
    if not _INDEX.exists():  # pragma: no cover - repo layout guard
        pytest.skip("single_file_demo/index.html not present")
    return _INDEX.read_text("utf-8")


def test_ed25519_verdict_branches_on_status_not_on_a_glyph(source: str) -> None:
    """The fail-closed branch must compare the enum, not the label."""
    assert "signatures.ed25519.status === 'invalid'" in source, (
        "the Ed25519 fail-closed branch must test the machine-readable status; "
        "if this moved, make sure it did not move back to matching display text"
    )
    assert "signatures.ed25519.startsWith(" not in source, (
        "REGRESSION: the security verdict is being re-derived from the label "
        "again. Rewording the label would then flip a failed signature to ok."
    )


def test_no_verdict_branch_matches_on_a_check_or_cross_glyph(source: str) -> None:
    """No control-flow test anywhere in the page may key on the verdict glyphs.

    Catches the general shape, not just the one instance that was fixed:
    ``startsWith('✗')``, ``includes('✓')``, ``=== '✗ ...'`` and friends.
    The glyphs are fine in string LITERALS being assigned or rendered; they are
    not fine inside a comparison.
    """
    offenders: list[str] = []
    patterns = [
        r"\.startsWith\(\s*['\"`]\s*[✓✗]",
        r"\.includes\(\s*['\"`]\s*[✓✗]",
        r"\.indexOf\(\s*['\"`]\s*[✓✗]",
        r"[=!]==?\s*['\"`]\s*[✓✗]",
    ]
    for lineno, line in enumerate(source.splitlines(), start=1):
        for pat in patterns:
            if re.search(pat, line):
                offenders.append(f"{lineno}: {line.strip()[:110]}")
    assert not offenders, (
        "a verdict-bearing comparison keys on a display glyph, which makes "
        "security behaviour depend on user-facing copy:\n  "
        + "\n  ".join(offenders)
    )


def test_signature_entries_carry_both_status_and_label(source: str) -> None:
    """Every entry assigned into the signatures map must be the typed record.

    If a future edit assigns a bare string again, the render site would print
    ``[object Object]`` for the others or, worse, a security branch would start
    comparing strings again.
    """
    assignments = re.findall(r"out\.(?:ed25519|hmac)\s*=\s*([^;]+);", source)
    assert assignments, "expected assignments into the signatures map"
    for a in assignments:
        assert "status:" in a and "label:" in a, (
            f"signatures entry must be {{status, label}}, got: {a.strip()[:120]}"
        )


def test_render_site_prints_the_label_not_the_record(source: str) -> None:
    """Presentation reads `.label`; if it printed the record it would leak
    `[object Object]` onto the front door.

    Asserted on the `.label` accessor rather than on the whole template
    literal: the template now wraps both halves in ``escHtml`` (the key and
    the label both come out of a pasted bundle, see
    ``test_demo_html_sinks_escaped.py``). Pinning the exact literal made this
    test fail on an escaping fix that did not change what is printed, so it
    pinned formatting rather than the property it cares about.
    """
    assert "v.label" in source, "the signature-note render must print v.label"
    assert "${v}" not in source, (
        "REGRESSION: the whole signatures record is being interpolated; that "
        "prints [object Object] on the front door instead of the label"
    )


def test_the_enum_contract_is_still_documented(source: str) -> None:
    """The enum is the thing security depends on; its contract must stay stated."""
    for member in ("'verified'", "'invalid'", "'unsupported'", "'absent'"):
        assert member in source, f"enum member {member} missing from index.html"
