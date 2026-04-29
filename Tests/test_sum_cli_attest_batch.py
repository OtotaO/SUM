"""End-to-end contract for ``sum attest-batch``.

Pinned behaviours:

  1. **JSONL output**: one compact bundle per line on stdout, in
     input order. No pretty-print, no leading metadata.

  2. **Per-file independence**: extraction failure on one file does
     NOT abort the run. The failing file is reported on stderr with
     a structured ``sum: file=<path> error=<reason>`` line; other
     files still produce bundles.

  3. **Aggregate exit code**: 0 only if every file succeeded; 1 if
     any failed. The number of stdout lines equals the number of
     succeeded files.

  4. **Per-bundle source provenance**: every bundle carries
     ``sum_cli.source_path`` and ``sum_cli.source_uri`` (sha256: of
     the file's bytes), letting downstream tooling map a bundle
     back to the file it came from.

  5. **State byte-identity**: the state_integer in a batch-bundle
     for file F equals the state_integer ``sum attest --input F``
     would produce — the batch is N independent attests, not a
     fused-state operation.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from sum_cli.main import cmd_attest, cmd_attest_batch


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


def _run_batch(files: list[str], **overrides) -> tuple[int, str, str]:
    args = argparse.Namespace(
        files=files,
        extractor="sieve",
        model=None,
        branch="main",
        title="Attested Tome",
        signing_key=None,
        ed25519_key=None,
        verbose=False,
        dedup_threshold=None,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    try:
        code = cmd_attest_batch(args)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
    return code, out_buf.getvalue(), err_buf.getvalue()


def _attest_one(path: Path) -> dict:
    """Reference: invoke ``sum attest --input path`` directly."""
    args = argparse.Namespace(
        input=str(path),
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="Attested Tome",
        signing_key=None,
        ed25519_key=None,
        ledger=None,
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
    assert code == 0
    return json.loads(buf.getvalue())


CORPUS_A = (
    "Marie Curie won two Nobel Prizes. "
    "Einstein proposed relativity."
)
CORPUS_B = (
    "Shakespeare wrote Hamlet. "
    "Water contains hydrogen. "
    "Dolphins are mammals."
)


def _make_files(tmp_path: Path, *texts: str) -> list[Path]:
    out = []
    for i, t in enumerate(texts):
        p = tmp_path / f"in_{i}.txt"
        p.write_text(t)
        out.append(p)
    return out


def test_batch_two_files_produces_two_jsonl_lines(tmp_path):
    """Two valid inputs → two bundles on stdout, exit 0."""
    files = _make_files(tmp_path, CORPUS_A, CORPUS_B)
    code, out, err = _run_batch([str(p) for p in files])

    assert code == 0, err
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 2, f"expected 2 bundles, got {len(lines)}: {out!r}"
    bundles = [json.loads(ln) for ln in lines]
    for b in bundles:
        # JSONL: each line MUST be a complete bundle. Required Phase 16 fields:
        for f in ("canonical_tome", "state_integer", "canonical_format_version"):
            assert f in b


def test_batch_state_integer_matches_single_attest(tmp_path):
    """For each file, the state_integer in the batch bundle MUST
    equal the state_integer ``sum attest --input <file>`` produces."""
    files = _make_files(tmp_path, CORPUS_A, CORPUS_B)
    _, out, _ = _run_batch([str(p) for p in files])
    batch_bundles = [json.loads(ln) for ln in out.strip().split("\n")]

    for f, batch_bundle in zip(files, batch_bundles):
        ref = _attest_one(f)
        assert batch_bundle["state_integer"] == ref["state_integer"], (
            f"batch state for {f.name} differs from single-file attest"
        )


def test_batch_carries_source_path_and_uri_per_bundle(tmp_path):
    """Each batch bundle MUST carry sum_cli.source_path (the file
    path argv used) and sum_cli.source_uri (sha256: of the file
    bytes) so downstream consumers can route bundles back to files."""
    import hashlib

    files = _make_files(tmp_path, CORPUS_A, CORPUS_B)
    _, out, _ = _run_batch([str(p) for p in files])
    bundles = [json.loads(ln) for ln in out.strip().split("\n")]

    for f, b in zip(files, bundles):
        sidecar = b.get("sum_cli", {})
        assert sidecar.get("source_path") == str(f)
        expected_uri = "sha256:" + hashlib.sha256(
            f.read_text().strip().encode("utf-8")
        ).hexdigest()
        assert sidecar.get("source_uri") == expected_uri
        assert sidecar.get("batch") is True


def test_batch_continues_on_per_file_failure(tmp_path):
    """One bad file (empty) MUST NOT abort the run. The good files
    still produce bundles; the bad file is reported on stderr.
    Exit code is 1."""
    good1 = tmp_path / "good1.txt"
    good1.write_text(CORPUS_A)
    bad = tmp_path / "empty.txt"
    bad.write_text("")
    good2 = tmp_path / "good2.txt"
    good2.write_text(CORPUS_B)

    code, out, err = _run_batch([str(good1), str(bad), str(good2)])

    assert code == 1, "any-file-failed → exit 1"
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 2, f"expected 2 bundles, got {len(lines)}"
    assert "empty.txt" in err and "empty_input" in err


def test_batch_continues_on_missing_file(tmp_path):
    """A nonexistent file MUST be reported on stderr but not abort
    the run; the present files still mint bundles."""
    good = tmp_path / "good.txt"
    good.write_text(CORPUS_A)
    missing = tmp_path / "does_not_exist.txt"

    code, out, err = _run_batch([str(missing), str(good)])

    assert code == 1
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 1
    assert "does_not_exist.txt" in err and "read_failed" in err


def test_batch_zero_files_returns_exit_2(tmp_path):
    """Defensive: argparse requires nargs='+' so this normally won't
    happen, but the function-level guard MUST also handle it."""
    code, out, err = _run_batch([])
    assert code == 2
    assert "requires at least one input file" in err


# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------


def test_batch_dedup_skips_identical_inputs(tmp_path):
    """Two byte-identical files with --dedup-threshold=0.85 → one
    bundle on stdout, the duplicate reported on stderr with
    jaccard=1.000."""
    files = _make_files(tmp_path, CORPUS_A, CORPUS_A)  # same text twice
    code, out, err = _run_batch(
        [str(p) for p in files], dedup_threshold=0.85,
    )
    assert code == 0  # no failures, just dedup
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 1, f"expected 1 bundle (one was dedup'd), got {len(lines)}"
    assert "dedup_skipped" in err
    assert "jaccard=1.000" in err
    assert f"duplicate_of={files[0]}" in err


def test_batch_dedup_keeps_distinct_inputs(tmp_path):
    """Two materially-different files with --dedup-threshold=0.85 →
    two bundles on stdout, no dedup_skipped on stderr."""
    files = _make_files(tmp_path, CORPUS_A, CORPUS_B)
    code, out, err = _run_batch(
        [str(p) for p in files], dedup_threshold=0.85,
    )
    assert code == 0
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 2, f"expected 2 bundles, got {len(lines)}"
    assert "dedup_skipped" not in err


def test_batch_dedup_default_disabled_keeps_duplicates(tmp_path):
    """Without --dedup-threshold, identical files are NOT dedup'd —
    backwards-compat check."""
    files = _make_files(tmp_path, CORPUS_A, CORPUS_A)
    code, out, err = _run_batch([str(p) for p in files])  # no dedup_threshold override
    assert code == 0
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 2, "without --dedup-threshold, both bundles should mint"


def test_batch_dedup_threshold_validation(tmp_path):
    """A threshold outside (0.0, 1.0] MUST exit 2 with diagnostic."""
    files = _make_files(tmp_path, CORPUS_A)
    for bad in (0.0, 1.1, -0.5):
        code, _, err = _run_batch([str(files[0])], dedup_threshold=bad)
        assert code == 2, f"threshold {bad} should have been rejected"
        assert "must be in" in err


def test_batch_dedup_strict_threshold_only_drops_byte_identical(tmp_path):
    """--dedup-threshold=1.0 means 'skip only true byte-identical
    duplicates'. Near-duplicates (one extra word) MUST mint normally."""
    near_dup_b = CORPUS_B + " Beethoven composed nine symphonies."
    files = _make_files(tmp_path, CORPUS_B, near_dup_b)
    code, out, _ = _run_batch(
        [str(p) for p in files], dedup_threshold=1.0,
    )
    assert code == 0
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 2, "near-duplicates at threshold=1.0 must both mint"
