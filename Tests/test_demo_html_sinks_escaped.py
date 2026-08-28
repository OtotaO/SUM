"""Attacker-reachable values must be escaped before they reach ``innerHTML``.

`single_file_demo/index.html` is the live front door at
sum-demo.ototao.workers.dev. Two of its ``innerHTML`` templates interpolated
values an attacker controls:

1. The facts list wrote triple components straight into markup::

       li.innerHTML = `${s} <b>${p}</b> ${o} ...`

   ``s``/``p``/``o`` are extracted from prose the visitor pastes.

2. The verify panel wrote a pasted bundle's own strings into markup::

       `<span class="bad">✗ ${r.reason}</span>`
       `<div class="sig-note">• ${k}: ${v.label}</div>`

   ``k`` is an **object key** read out of the pasted JSON, so both the key and
   the label are wholly attacker-authored. The whole point of that panel is
   that you paste a bundle somebody else handed you.

This was not theoretical. The deployed CSP is::

    script-src 'unsafe-inline'

which permits inline event handlers, so an injected ``<img src=x onerror=...>``
executes. The page that runs it is the page rendering a trust verdict, which is
the single worst place in this project to lose integrity: an attacker who can
run script there can rewrite the verdict the human is reading.

Static checks over the source text, deliberately, matching the house style of
``test_demo_verdict_not_glyph_matched.py``. A browser harness for the front
door is a larger piece of work and is not what this pin is for. The point is
narrow: no attacker-reachable value reaches ``innerHTML`` unescaped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "single_file_demo" / "index.html"


@pytest.fixture(scope="module")
def source() -> str:
    if not _INDEX.exists():  # pragma: no cover - repo layout guard
        pytest.skip("single_file_demo/index.html not present")
    return _INDEX.read_text("utf-8")


def test_escape_helper_exists(source: str) -> None:
    """The helper must exist and cover the five HTML-significant characters."""
    assert "const escHtml =" in source, "the escHtml helper is gone"
    for ch, ent in (
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ('"', "&quot;"),
        ("'", "&#39;"),
    ):
        assert ent in source, f"escHtml no longer maps {ch!r} to {ent}"


def test_facts_list_escapes_triple_components(source: str) -> None:
    """Triple components come from pasted prose; they must be escaped."""
    assert "${escHtml(s)} <b>${escHtml(p)}</b> ${escHtml(o)}" in source, (
        "REGRESSION: the facts list is interpolating raw triple components "
        "into innerHTML again. Prose containing markup would execute."
    )


def test_verify_panel_escapes_bundle_controlled_strings(source: str) -> None:
    """Everything in the verify panel originates in a pasted bundle."""
    assert "${escHtml(r.reason)}" in source, (
        "REGRESSION: the verify verdict reason is unescaped."
    )
    assert "${escHtml(k)}: ${escHtml(v.label)}" in source, (
        "REGRESSION: signature-note key or label is unescaped. The key is an "
        "object key taken from the pasted JSON, so it is attacker-authored."
    )
    assert "${escHtml(e.message)}" in source, (
        "REGRESSION: the JSON parse-error message is unescaped."
    )


def test_no_raw_interpolation_left_in_those_templates(source: str) -> None:
    """Belt and braces: the exact pre-fix substrings must not reappear."""
    for bad in (
        "${s} <b>${p}</b> ${o}",
        "✓ ${r.reason}",
        "✗ ${r.reason}",
        "• ${k}: ${v.label}",
        "JSON parse error: ${e.message}",
    ):
        assert bad not in source, f"REGRESSION: unescaped template back: {bad!r}"


def test_module_blocks_define_their_own_escaper(source: str) -> None:
    """Both ``<script type="module">`` verifier panels escape locally.

    A module must not depend on a classic script's top-level ``const``
    surviving into its scope, so each block declares its own ``esc``.
    """
    assert source.count("const esc = v => String(v).replace") == 2, (
        "expected one local escaper in each of the two module verifier panels"
    )


def test_pasted_receipt_fields_are_escaped(source: str) -> None:
    """The paste panels take BOTH the receipt and the JWKS from the visitor.

    That means an attacker can mint a receipt that verifies cleanly and still
    carries markup in ``kid`` / ``scorer`` / ``not_covered`` / ``controlled``.
    """
    for frag in (
        "${esc(res.kid)}",
        "${esc(p.scorer ?? \"?\")}",
        "${esc(nc)}",
        "${esc(p.controlled)}",
        "${esc(schema)}",
        "${esc(result.kid)}",
        "${esc(ph.alg)}",
        "${esc(JSON.stringify(ph.crit))}",
    ):
        assert frag in source, f"REGRESSION: unescaped receipt field, expected {frag}"


def test_error_paths_are_escaped(source: str) -> None:
    """Error strings reach innerHTML on every failure branch."""
    assert "error class: <code>${esc(cls)}</code>" in source
    assert "${cls}</code>" not in source.replace("${esc(cls)}</code>", "")
    assert "opacity:0.75;\">${e.message}</div>" not in source, (
        "REGRESSION: a raw error message is being written into innerHTML"
    )
