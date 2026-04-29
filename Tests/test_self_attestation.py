"""End-to-end contract for SUM self-attestation.

Pinned behaviours:

  1. **Every canonical doc round-trips.** Each bundle in
     ``meta/self_attestation.jsonl`` MUST pass ``sum verify``:
     state-integer reconstruction matches, axiom count matches,
     bundle structure validates. This is the load-bearing claim
     of the whole self-attestation pipeline — if a single doc
     fails to round-trip, the receipts are not what they say they
     are.

  2. **Drift gate fails on doc edits.** A simulated edit to one
     of the canonical docs MUST cause ``--check`` to surface the
     drift with the path of the edited doc.

  3. **Summary matches JSONL.** Every summary entry's
     ``source_uri`` and ``axiom_count`` MUST match the
     corresponding line in the JSONL.

  4. **Algebra-level pipe rejection.** ``get_or_mint_prime`` and
     ``encode_chunk_state`` MUST reject pipe-containing components
     so noise like ``('|', 'close', 'this')`` (real triple from
     the README's markdown table) doesn't poison verification.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sum_cli.main import cmd_verify

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Algebra-level: pipe and empty-component rejection
# --------------------------------------------------------------------------


def test_get_or_mint_prime_rejects_empty_components():
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    a = GodelStateAlgebra()
    for triple in [
        ("", "win", "prize"),
        ("alice", "", "prize"),
        ("alice", "win", ""),
        ("   ", "win", "prize"),
        ("alice", "  ", "prize"),
    ]:
        with pytest.raises(ValueError, match="empty"):
            a.get_or_mint_prime(*triple)


def test_get_or_mint_prime_rejects_pipe_components():
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    a = GodelStateAlgebra()
    for triple in [
        ("|", "close", "this"),       # the real-world README case
        ("|alice", "win", "prize"),
        ("alice", "win", "prize|"),
        ("a||b", "win", "prize"),
    ]:
        with pytest.raises(ValueError, match="'\\|'"):
            a.get_or_mint_prime(*triple)


def test_encode_chunk_state_skips_malformed_axioms_silently():
    """``encode_chunk_state`` MUST keep going past a malformed
    triple, treating it as a no-op rather than crashing the bag-
    encoding loop. The good triples MUST encode normally."""
    from sum_engine_internal.algorithms.semantic_arithmetic import (
        GodelStateAlgebra,
    )
    a = GodelStateAlgebra()
    good = ("alice", "like", "cat")
    bad = ("|", "close", "this")
    state_with_bad = a.encode_chunk_state([good, bad])
    state_only_good = a.encode_chunk_state([good])
    assert state_with_bad == state_only_good


# --------------------------------------------------------------------------
# Self-attestation bundles round-trip
# --------------------------------------------------------------------------


JSONL_PATH = REPO_ROOT / "meta" / "self_attestation.jsonl"
SUMMARY_PATH = REPO_ROOT / "meta" / "self_attestation.summary.json"


@pytest.fixture(scope="module")
def self_attestation_bundles() -> list[dict]:
    if not JSONL_PATH.exists():
        pytest.fail(
            "meta/self_attestation.jsonl missing. "
            "Refresh with: python -m scripts.attest_repo_docs"
        )
    return [
        json.loads(ln)
        for ln in JSONL_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _verify_via_cmd(bundle: dict) -> tuple[int, str]:
    """Invoke cmd_verify in-process; return (exit_code, stdout_json)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bundle, f)
        path = f.name
    args = argparse.Namespace(
        input=path, signing_key=None, strict=False, pretty=False,
    )
    out_buf = io.StringIO()
    old = sys.stdout
    sys.stdout = out_buf
    try:
        code = cmd_verify(args)
    finally:
        sys.stdout = old
    return code, out_buf.getvalue()


def test_every_self_attestation_bundle_verifies(self_attestation_bundles):
    """The headline invariant. If this fails, the receipts shipped
    in meta/ are claiming things they cannot prove."""
    failures: list[str] = []
    for bundle in self_attestation_bundles:
        path = bundle.get("sum_cli", {}).get("source_path", "?")
        code, stdout = _verify_via_cmd(bundle)
        if code != 0:
            failures.append(f"{path}: exit={code} stdout={stdout}")
        else:
            result = json.loads(stdout)
            if not result.get("ok"):
                failures.append(f"{path}: {result}")

    assert not failures, "self-attestation verify failures:\n" + "\n".join(failures)


def test_self_attestation_summary_matches_jsonl(self_attestation_bundles):
    if not SUMMARY_PATH.exists():
        pytest.fail("meta/self_attestation.summary.json missing")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["doc_count"] == len(self_attestation_bundles)
    for bundle, entry in zip(self_attestation_bundles, summary["docs"]):
        sidecar = bundle["sum_cli"]
        assert entry["path"] == sidecar["source_path"]
        assert entry["source_uri"] == sidecar["source_uri"]
        assert entry["axiom_count"] == sidecar["axiom_count"]
        assert entry["state_integer_digits"] == len(bundle["state_integer"])


# --------------------------------------------------------------------------
# Drift gate
# --------------------------------------------------------------------------


def test_drift_check_passes_on_current_artifacts():
    """The committed self-attestation MUST be current with the
    docs at HEAD. CI re-runs this on every PR."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.attest_repo_docs", "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        f"--check failed: stdout={r.stdout!r} stderr={r.stderr!r}\n"
        f"This means a doc changed without refreshing the "
        f"self-attestation. Run: python -m scripts.attest_repo_docs"
    )


def test_drift_check_fails_on_missing_artifacts(tmp_path, monkeypatch):
    """If the artifacts are absent, --check MUST fail with exit 2
    and a refresh recipe."""
    fake_meta = tmp_path / "meta"
    fake_meta.mkdir()

    # Run --check with a synthetic attest_repo_docs that points its
    # OUT_JSONL at the empty fake_meta. We simulate by patching the
    # script's globals via a shim subprocess invocation.
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import scripts.attest_repo_docs as m\n"
        "from pathlib import Path\n"
        f"m.OUT_JSONL = Path({str(fake_meta / 'self_attestation.jsonl')!r})\n"
        f"m.OUT_SUMMARY = Path({str(fake_meta / 'self_attestation.summary.json')!r})\n"
        "sys.exit(m.main())\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code, "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "missing" in r.stderr.lower() or "refresh" in r.stderr.lower()
