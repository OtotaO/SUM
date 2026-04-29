"""Omni-format → markdown pivot for the attest pipeline.

Single canonical pivot: every supported input format is converted
to markdown text, then fed to the existing extractor / state /
bundle pipeline. Markdown was chosen as the pivot because:

  - human-readable (debuggable when receipts disagree),
  - structurally lossless for the inputs that matter (headings,
    paragraphs, lists, tables — all preserved),
  - byte-deterministic for text-bearing formats given pinned
    converter versions (we hash the markdown to surface any
    drift if a dep upgrade changes output).

Source URI anchoring discipline (load-bearing):

  The bundle's source URI is computed from the **original input
  bytes**, NOT from the markdown. A receipt for a PDF says "this
  is the verifiable claim about the bytes of file.pdf"; the
  markdown is an intermediate representation, not the artifact.
  This lets a verifier re-run the conversion step and detect
  upstream drift (a markitdown bump that changes PDF text
  extraction would surface as a markdown-bytes mismatch even
  though the source URI still resolves the same file).

Determinism boundary (documented + tested):

  - text-bearing formats with NO LLM hooks → deterministic
    (HTML, DOCX, PDF text, plaintext, EPUB, .ipynb, .json)
  - LLM-vision image descriptions → NON-deterministic (disabled)
  - Cloud-API audio transcription → NON-deterministic (disabled)

  The adapter constructs ``MarkItDown()`` with no ``llm_client``
  argument so the LLM-vision path is never reached. Audio/
  transcription paths require explicit opt-in and are gated
  behind the ``audio`` extra (not the base ``omni-format`` one).

Public surface:

  - ``convert_to_markdown(path)`` → ``ConvertedDocument``
  - ``is_omni_format_available()`` → bool (markitdown installed?)
  - ``ConvertedDocument`` carries: markdown text, source URI
    (sha256 of original bytes), input format, converter id,
    converter version. All of these go into the bundle's
    ``sum_cli`` sidecar so verifiers can replay the conversion.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Heuristic: which file extensions we route through markitdown.
# Anything else is treated as plaintext. Kept narrow on purpose —
# adding an extension here is a deliberate scope decision because
# each one comes with a determinism story we have to maintain.
_MARKITDOWN_EXTENSIONS = frozenset(
    {
        # markup / structured text
        ".html", ".htm", ".xml", ".rtf", ".epub",
        # office
        ".docx", ".pptx", ".xlsx",
        # PDF
        ".pdf",
        # data / notebook
        ".ipynb", ".json",
        # already-markdown (no-op pass-through, but normalizes line endings)
        ".md", ".markdown",
    }
)

# Pure plaintext: read the bytes, decode UTF-8, no conversion.
_PLAINTEXT_EXTENSIONS = frozenset({".txt", ".text", ".log"})


@dataclass(frozen=True)
class ConvertedDocument:
    """Result of converting an input file to the markdown pivot.

    Carries the metadata a downstream attest path needs to anchor
    a verifiable receipt and to let a verifier replay the
    conversion.
    """

    markdown: str
    source_path: str
    source_uri: str           # sha256: of the ORIGINAL bytes
    source_bytes_len: int
    input_format: str         # e.g. "pdf", "html", "plaintext", "markdown"
    converter: str            # e.g. "markitdown@0.1.5", "passthrough"
    markdown_sha256: str      # sha256 of the markdown text (not URL-encoded)


def is_omni_format_available() -> bool:
    """True if ``markitdown`` is importable; False otherwise.

    The base `sum-engine` install does NOT pull markitdown. Users
    must `pip install 'sum-engine[omni-format]'` to enable the
    PDF / HTML / DOCX / EPUB / .ipynb / .json paths. Plaintext
    is always available.
    """
    try:
        import markitdown  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_format(path: Path) -> str:
    """Return a canonical format name for *path* based on extension.

    Conservative: the extension MUST be in one of the known sets
    or we route as plaintext. We do NOT sniff content here — that
    job is delegated to markitdown (which uses Google's magika
    under the hood) for the cases it owns.
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in _PLAINTEXT_EXTENSIONS:
        return "plaintext"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".docx":
        return "docx"
    if suffix == ".pptx":
        return "pptx"
    if suffix == ".xlsx":
        return "xlsx"
    if suffix == ".epub":
        return "epub"
    if suffix == ".ipynb":
        return "ipynb"
    if suffix == ".json":
        return "json"
    if suffix == ".rtf":
        return "rtf"
    if suffix == ".xml":
        return "xml"
    return "plaintext"


def convert_to_markdown(path: str | Path) -> ConvertedDocument:
    """Convert the file at *path* to markdown and return the
    result with anchoring metadata.

    Routing:

      - Known plaintext / .md / .markdown / .txt /.log:
        decode bytes as UTF-8 and pass through (no markitdown).
      - Known markitdown-routed extensions (.pdf / .html / .docx
        / .epub / .ipynb / .json / etc): call
        ``MarkItDown(enable_plugins=False).convert_local(path)``.
        Plugins are explicitly disabled to keep the conversion
        deterministic and side-effect-free.
      - Unknown extension: treated as plaintext (UTF-8 decode).

    Raises:
      FileNotFoundError if the path does not exist.
      RuntimeError if the format requires markitdown but the
        package is not installed (caller should install
        ``sum-engine[omni-format]``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"format_pivot: {p} does not exist")

    raw = p.read_bytes()
    source_uri = "sha256:" + hashlib.sha256(raw).hexdigest()
    fmt = _detect_format(p)

    if fmt in {"plaintext", "markdown"}:
        # Pass-through. Decode replace-on-error so a stray non-UTF-8
        # byte doesn't crash the pipeline; downstream extractors
        # are robust to mojibake but DO need a string.
        text = raw.decode("utf-8", errors="replace")
        # Normalize CRLF / CR → LF for byte-stable hashing across
        # operating systems (Git's autocrlf can rewrite endings).
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return ConvertedDocument(
            markdown=text,
            source_path=str(p),
            source_uri=source_uri,
            source_bytes_len=len(raw),
            input_format=fmt,
            converter="passthrough",
            markdown_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    if fmt not in {  # markitdown-handled
        "pdf", "html", "docx", "pptx", "xlsx",
        "epub", "ipynb", "json", "rtf", "xml",
    }:
        # Defensive — _detect_format only emits known names, but
        # if a future extension routes here, fail loudly.
        raise RuntimeError(f"format_pivot: unhandled format {fmt!r}")

    if not is_omni_format_available():
        raise RuntimeError(
            f"format_pivot: input format {fmt!r} requires the "
            f"omni-format extra. Run: "
            f"pip install 'sum-engine[omni-format]'"
        )

    import markitdown  # imported here so plaintext path stays dep-free

    converter_id = f"markitdown@{markitdown.__version__}"
    md = markitdown.MarkItDown(enable_plugins=False)
    # convert_stream is the most deterministic path: we feed bytes
    # we read, with the file extension, and don't let markitdown
    # touch the filesystem (no temp files, no follow-symlinks
    # surprises). The buffer is rewound in case markitdown
    # inspects bytes more than once.
    buf = io.BytesIO(raw)
    result = md.convert_stream(buf, file_extension=p.suffix)

    text = result.text_content or ""
    # Defensive line-ending normalization for cross-OS byte-stability.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return ConvertedDocument(
        markdown=text,
        source_path=str(p),
        source_uri=source_uri,
        source_bytes_len=len(raw),
        input_format=fmt,
        converter=converter_id,
        markdown_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
