"""End-to-end contract for ``sum attest --format auto`` on omni-format inputs.

Pinned behaviours:

  1. **HTML routing**: a .html file is converted to markdown via
     markitdown, the bundle's source_uri is the sha256: of the
     original HTML bytes (NOT the markdown), and the bundle's
     ``sum_cli`` sidecar carries enough metadata for a verifier
     to replay the conversion (``input_format``, ``converter``,
     ``markdown_sha256``).

  2. **Markdown pass-through**: a .md file routes through the
     ``passthrough`` converter; markdown_sha256 == source bytes
     hash (modulo CRLF normalisation).

  3. **--format raw**: even on a .html input, ``--format raw``
     skips markitdown and reads bytes verbatim. (Useful escape
     hatch when a user wants the literal HTML attested rather
     than its semantic content.)

  4. **--format auto with no markitdown**: the import probe
     succeeds (we have it installed), so this is exercised
     positively. Negative path is covered by a unit test in
     ``test_format_pivot.py``.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from sum_cli.main import cmd_attest


markitdown = pytest.importorskip("markitdown")
spacy = pytest.importorskip("spacy")


def _has_en_core_web_sm() -> bool:
    try:
        spacy.load("en_core_web_sm")
        return True
    except (OSError, ImportError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_en_core_web_sm(),
    reason="en_core_web_sm not installed",
)


SAMPLE_HTML = b"""\
<!DOCTYPE html>
<html><body>
<h1>Verifiable knowledge</h1>
<p>Marie Curie won two Nobel Prizes.</p>
<p>Einstein proposed relativity.</p>
<p>Shakespeare wrote Hamlet.</p>
<p>Water contains hydrogen.</p>
<p>Dolphins are mammals.</p>
</body></html>
"""


def _attest(input_path: Path, *, fmt: str = "auto") -> dict:
    args = argparse.Namespace(
        input=str(input_path),
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="Attested Tome",
        signing_key=None,
        ed25519_key=None,
        ledger=None,
        format=fmt,
        pretty=False,
        verbose=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_attest(args)
    finally:
        sys.stdout = old
    assert code == 0, f"attest failed: {buf.getvalue()}"
    return json.loads(buf.getvalue())


def test_html_input_routes_through_markitdown(tmp_path):
    p = tmp_path / "sample.html"
    p.write_bytes(SAMPLE_HTML)

    bundle = _attest(p)
    sidecar = bundle["sum_cli"]

    assert sidecar["input_format"] == "html"
    assert sidecar["converter"].startswith("markitdown@")
    # Source URI MUST be anchored to the ORIGINAL HTML bytes, not markdown.
    assert sidecar["source_uri"] == "sha256:" + hashlib.sha256(SAMPLE_HTML).hexdigest()
    assert sidecar["source_bytes_len"] == len(SAMPLE_HTML)
    assert "markdown_sha256" in sidecar
    # The markdown hash MUST differ from the source hash (different bytes).
    assert sidecar["markdown_sha256"] != sidecar["source_uri"].split(":", 1)[1]
    # Real triples were extracted from the converted markdown.
    assert bundle["axiom_count"] >= 3


def test_markdown_input_uses_passthrough(tmp_path):
    md_text = (
        "# Verifiable knowledge\n\n"
        "Marie Curie won two Nobel Prizes.\n"
        "Einstein proposed relativity.\n"
        "Shakespeare wrote Hamlet.\n"
    )
    p = tmp_path / "sample.md"
    p.write_text(md_text)

    bundle = _attest(p)
    sidecar = bundle["sum_cli"]

    assert sidecar["input_format"] == "markdown"
    assert sidecar["converter"] == "passthrough"
    # Pass-through: markdown_sha256 == sha256 of source bytes (after CRLF norm).
    assert sidecar["markdown_sha256"] == sidecar["source_uri"].split(":", 1)[1]


def test_format_raw_skips_markitdown_on_html(tmp_path):
    """Escape hatch: --format raw on a .html input MUST NOT route
    through markitdown. The CLI reads the bytes verbatim and the
    sidecar reports ``converter=raw-readthrough``.

    Test uses a file with .html extension but plaintext content so
    the sieve actually extracts triples after the noise filter
    (which rejects components containing ``<``, ``>``, ``[``, etc.,
    so feeding literal HTML through --format raw correctly produces
    zero triples as of the sieve-quality filter — that's a real
    behavioural change, not a regression). The point being tested
    here is that markitdown is NOT invoked when --format raw is
    set, which is orthogonal to whether the content parses well."""
    plaintext_with_html_ext = (
        b"Marie Curie won two Nobel Prizes. "
        b"Albert Einstein proposed the theory of relativity. "
        b"Shakespeare wrote Hamlet. "
        b"Water contains hydrogen."
    )
    p = tmp_path / "sample.html"
    p.write_bytes(plaintext_with_html_ext)

    bundle = _attest(p, fmt="raw")
    sidecar = bundle["sum_cli"]

    assert sidecar["input_format"] == "raw"
    assert sidecar["converter"] == "raw-readthrough"
    # Source URI still anchors to original bytes (plaintext here, but
    # the .html extension is what would normally trigger markitdown).
    assert sidecar["source_uri"] == "sha256:" + hashlib.sha256(plaintext_with_html_ext).hexdigest()
    # markdown_sha256 should be absent on the raw path (no conversion happened).
    assert "markdown_sha256" not in sidecar


def test_html_attest_is_deterministic(tmp_path):
    """Two invocations on the same HTML input MUST produce
    bundles with the same state_integer. The conversion path is
    deterministic for text-bearing formats with LLM hooks
    disabled (which is the default — markitdown is constructed
    without an llm_client)."""
    p = tmp_path / "sample.html"
    p.write_bytes(SAMPLE_HTML)

    b1 = _attest(p)
    b2 = _attest(p)
    assert b1["state_integer"] == b2["state_integer"]
    assert b1["sum_cli"]["markdown_sha256"] == b2["sum_cli"]["markdown_sha256"]
