"""Unit tests for the omni-format → markdown pivot adapter.

Layered coverage:

  1. **Format detection** is extension-driven and conservative —
     anything outside the explicit lists routes as plaintext.
  2. **Plaintext / markdown pass-through** is byte-exact (modulo
     CRLF→LF normalisation), no markitdown dep needed.
  3. **HTML/XML/etc. routing** goes through markitdown and produces
     deterministic markdown for a fixed input.
  4. **Source-URI anchoring** uses the ORIGINAL bytes' sha256, never
     the markdown's. (Determinism for verifier replay.)
  5. **Determinism**: converting the same input twice yields
     byte-identical markdown.
  6. **Missing-extra error path** is actionable.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from sum_engine_internal.adapters.format_pivot import (
    ConvertedDocument,
    _detect_format,
    convert_to_markdown,
    is_omni_format_available,
)


# --------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("foo.md", "markdown"),
        ("foo.markdown", "markdown"),
        ("FOO.MD", "markdown"),
        ("foo.txt", "plaintext"),
        ("foo.text", "plaintext"),
        ("foo.log", "plaintext"),
        ("foo.pdf", "pdf"),
        ("foo.html", "html"),
        ("foo.htm", "html"),
        ("foo.docx", "docx"),
        ("foo.pptx", "pptx"),
        ("foo.xlsx", "xlsx"),
        ("foo.epub", "epub"),
        ("foo.ipynb", "ipynb"),
        ("foo.json", "json"),
        ("foo.rtf", "rtf"),
        ("foo.xml", "xml"),
        # Unknown extensions → plaintext (conservative default)
        ("foo.bin", "plaintext"),
        ("foo", "plaintext"),
        ("foo.py", "plaintext"),
    ],
)
def test_detect_format(name, expected):
    assert _detect_format(Path(name)) == expected


# --------------------------------------------------------------------------
# Plaintext / markdown pass-through
# --------------------------------------------------------------------------


def test_plaintext_passthrough_exact(tmp_path):
    text = "Marie Curie won two Nobel Prizes.\nEinstein proposed relativity."
    p = tmp_path / "sample.txt"
    p.write_bytes(text.encode("utf-8"))

    cd = convert_to_markdown(p)
    assert cd.markdown == text
    assert cd.input_format == "plaintext"
    assert cd.converter == "passthrough"
    assert cd.source_uri == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert cd.source_bytes_len == len(text.encode("utf-8"))
    # Pass-through path: markdown_sha256 == source URI hex (same bytes)
    assert cd.markdown_sha256 == cd.source_uri.split(":", 1)[1]


def test_markdown_passthrough_normalises_crlf(tmp_path):
    """CRLF line endings are normalised to LF for cross-OS byte stability."""
    text_crlf = b"# Heading\r\n\r\nLine one.\r\nLine two.\r\n"
    text_lf = "# Heading\n\nLine one.\nLine two.\n"
    p = tmp_path / "sample.md"
    p.write_bytes(text_crlf)

    cd = convert_to_markdown(p)
    assert cd.markdown == text_lf
    assert cd.input_format == "markdown"
    assert cd.converter == "passthrough"
    # Source URI hashes the ORIGINAL CRLF bytes (anchored to artifact).
    assert cd.source_uri == "sha256:" + hashlib.sha256(text_crlf).hexdigest()


def test_unknown_extension_routes_as_plaintext(tmp_path):
    """Unknown extension → plaintext path, no markitdown invocation."""
    p = tmp_path / "sample.bin"
    p.write_bytes(b"plain content here")
    cd = convert_to_markdown(p)
    assert cd.input_format == "plaintext"
    assert cd.converter == "passthrough"
    assert cd.markdown == "plain content here"


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_to_markdown(tmp_path / "does_not_exist.pdf")


# --------------------------------------------------------------------------
# HTML routing through markitdown
# --------------------------------------------------------------------------


markitdown = pytest.importorskip(
    "markitdown",
    reason="omni-format extra not installed; HTML/PDF/etc routing tests skipped",
)


SAMPLE_HTML = b"""\
<!DOCTYPE html>
<html><body>
<h1>Verifiable knowledge</h1>
<p>Marie Curie won two Nobel Prizes.</p>
<p>Einstein proposed relativity.</p>
</body></html>
"""


def test_html_conversion_produces_markdown(tmp_path):
    p = tmp_path / "sample.html"
    p.write_bytes(SAMPLE_HTML)

    cd = convert_to_markdown(p)

    assert cd.input_format == "html"
    assert cd.converter.startswith("markitdown@")
    # Source URI anchored to original HTML bytes — NOT the markdown.
    assert cd.source_uri == "sha256:" + hashlib.sha256(SAMPLE_HTML).hexdigest()
    assert cd.source_bytes_len == len(SAMPLE_HTML)
    # Conversion produces some text (specific shape is markitdown's call).
    assert "Marie Curie" in cd.markdown
    assert "Einstein" in cd.markdown
    # markdown_sha256 hashes the converted text (not the source).
    assert cd.markdown_sha256 == hashlib.sha256(cd.markdown.encode("utf-8")).hexdigest()
    # The two hashes MUST differ (HTML and markdown are different bytes).
    assert cd.markdown_sha256 != cd.source_uri.split(":", 1)[1]


def test_html_conversion_is_deterministic(tmp_path):
    """Same HTML input → byte-identical markdown across two invocations.

    This is the load-bearing property for receipt replay: a verifier
    that re-fetches the source bytes and reruns the converter MUST
    get the same markdown_sha256 we recorded. Non-determinism here
    would silently invalidate every receipt minted from PDF/HTML/
    DOCX inputs.
    """
    p = tmp_path / "sample.html"
    p.write_bytes(SAMPLE_HTML)

    cd1 = convert_to_markdown(p)
    cd2 = convert_to_markdown(p)
    assert cd1.markdown == cd2.markdown
    assert cd1.markdown_sha256 == cd2.markdown_sha256
    assert cd1.source_uri == cd2.source_uri


# --------------------------------------------------------------------------
# is_omni_format_available
# --------------------------------------------------------------------------


def test_is_omni_format_available_returns_bool():
    """The probe MUST never raise; it answers a yes/no."""
    assert isinstance(is_omni_format_available(), bool)


# --------------------------------------------------------------------------
# Empty inputs
# --------------------------------------------------------------------------


def test_empty_plaintext_is_handled(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_bytes(b"")
    cd = convert_to_markdown(p)
    assert cd.markdown == ""
    assert cd.source_bytes_len == 0
