"""`sum frontier` CLI subcommand tests — the version-cycler.

Covers the overview (sum.render_frontier.v1 with measured numbers + the
honest measurement_note), the --scrub cycler (print the version at a
position), stdin source, and the usage-error paths.
"""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _capture():
    out, err = io.StringIO(), io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        yield out, err


def run_cli(argv: list[str], stdin_text: str | None = None) -> tuple[int, str, str]:
    from sum_cli.main import main

    if stdin_text is not None:
        with _capture() as (out, err), patch("sys.stdin", io.StringIO(stdin_text)):
            rc = main(argv)
    else:
        with _capture() as (out, err):
            rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


SOURCE = (
    "The treaty was signed in Vienna in 1815. Delegates redrew the map of "
    "Europe. The settlement held for decades."
)
MEDIUM = "The treaty was signed in Vienna in 1815. The powers redrew Europe."
TAG = "Vienna treaty"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _files(tmp_path):
    return (
        _write(tmp_path, "src.txt", SOURCE),
        _write(tmp_path, "medium.txt", MEDIUM),
        _write(tmp_path, "tag.txt", TAG),
    )


# ── overview ──────────────────────────────────────────────────────────


def test_overview_emits_frontier_schema(tmp_path):
    src, medium, tag = _files(tmp_path)
    rc, out, _ = run_cli(
        ["frontier", "--source", src, "--version", src,
         "--version", medium, "--version", tag, "--pretty"]
    )
    assert rc == 0
    d = json.loads(out)
    assert d["schema"] == "sum.render_frontier.v1"
    assert d["n"] == 3
    assert d["scorer"] == "lexical-coverage-bidirectional"
    # honest: the per-doc number is a measurement, said so in the payload
    assert "measurement_note" in d


def test_overview_faithful_point_is_zero_loss(tmp_path):
    src, medium, tag = _files(tmp_path)
    rc, out, _ = run_cli(
        ["frontier", "--source", src, "--version", src,
         "--version", medium, "--version", tag]
    )
    assert rc == 0
    points = json.loads(out)["points"]
    assert points[0]["meaning_loss"] == 0.0   # version == source
    assert points[0]["position"] == 0.0
    assert points[-1]["position"] == 1.0
    assert points[-1]["meaning_loss"] > points[0]["meaning_loss"]


# ── scrub (the cycler) ────────────────────────────────────────────────


def test_scrub_most_compressed(tmp_path):
    src, medium, tag = _files(tmp_path)
    rc, out, _ = run_cli(
        ["frontier", "--source", src, "--version", src,
         "--version", medium, "--version", tag, "--scrub", "1.0"]
    )
    assert rc == 0
    assert out.strip() == TAG


def test_scrub_most_faithful(tmp_path):
    src, medium, tag = _files(tmp_path)
    rc, out, _ = run_cli(
        ["frontier", "--source", src, "--version", src,
         "--version", tag, "--scrub", "0.0"]
    )
    assert rc == 0
    assert out.strip() == SOURCE


def test_scrub_clamps(tmp_path):
    src, _, tag = _files(tmp_path)
    rc, out, _ = run_cli(
        ["frontier", "--source", src, "--version", src,
         "--version", tag, "--scrub", "9.0"]
    )
    assert rc == 0
    assert out.strip() == TAG


# ── stdin source ──────────────────────────────────────────────────────


def test_source_from_stdin(tmp_path):
    _, _, tag = _files(tmp_path)
    rc, out, _ = run_cli(
        ["frontier", "--source", "-", "--version", tag, "--scrub", "1.0"],
        stdin_text=SOURCE,
    )
    assert rc == 0
    assert out.strip() == TAG


# ── usage errors ──────────────────────────────────────────────────────


def test_no_version_is_usage_error(tmp_path):
    src, _, _ = _files(tmp_path)
    rc, _, err = run_cli(["frontier", "--source", src])
    assert rc == 2
    assert "at least one --version" in err


def test_missing_source_is_usage_error(tmp_path):
    _, _, tag = _files(tmp_path)
    rc, _, err = run_cli(
        ["frontier", "--source", str(tmp_path / "nope.txt"), "--version", tag]
    )
    assert rc == 2
    assert "cannot read" in err


def test_works_for_code_too(tmp_path):
    """The frontier is content-agnostic — code versions cycle the same."""
    src = _write(tmp_path, "full.py", "def add(a, b):\n    return a + b\n")
    terse = _write(tmp_path, "terse.py", "add = lambda a, b: a + b\n")
    rc, out, _ = run_cli(
        ["frontier", "--source", src, "--version", src, "--version", terse]
    )
    assert rc == 0
    assert json.loads(out)["n"] == 2
