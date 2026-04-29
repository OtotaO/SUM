"""Tests for `scripts/repo_manifest.py` — the cross-channel manifest.

Three layers of coverage:

  1. **Schema integrity.** The manifest emits exactly the fields the
     spec promises; the schema string is pinned.
  2. **Stable-fields determinism.** The drift check (``--check``)
     stripes out time-varying fields (``issued_at``, GitHub stars)
     so it doesn't false-positive on minute-by-minute repo state.
  3. **CI gate behaviour.** A stale manifest (with an outdated
     commits_last_30d count) MUST fail the drift check.

These tests deliberately do NOT call out to PyPI or GitHub —
the manifest builder gracefully handles offline (returns None for
those fields). The drift check only fails on substantive
divergences in the locally-derivable fields.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)


# --------------------------------------------------------------------------
# Schema integrity
# --------------------------------------------------------------------------


def test_manifest_emits_expected_schema_id():
    from scripts.repo_manifest import SCHEMA, build_manifest
    m = build_manifest()
    assert m["schema"] == SCHEMA
    assert SCHEMA == "sum.repo_manifest.v1"


def test_manifest_includes_load_bearing_fields():
    """Every field downstream consumers depend on MUST be in the
    manifest. If any is missing, the consumer either falls back
    silently or breaks; neither is acceptable."""
    from scripts.repo_manifest import build_manifest
    m = build_manifest()

    # Repo metadata
    assert m["repo"]["owner"] == "OtotaO"
    assert m["repo"]["name"] == "SUM"
    assert m["repo"]["url"] == "https://github.com/OtotaO/SUM"
    assert m["repo"]["license"] == "Apache-2.0"

    # Git state
    assert "head_sha" in m["git"]
    assert "head_short" in m["git"]
    assert "head_subject" in m["git"]
    assert "head_committer_date" in m["git"]
    assert "commits_last_30d" in m["git"]
    assert isinstance(m["git"]["commits_last_30d"], int)
    assert m["git"]["commits_last_30d"] >= 0

    # Release state
    assert "pyproject_version" in m["release"]
    assert isinstance(m["release"]["pyproject_version"], str)
    # pypi_published_version may be None if PyPI was unreachable

    # Feature counts
    feats = m["features"]
    assert all(k in feats for k in ("total", "production", "scaffolded", "designed"))
    # Sanity: total = production + scaffolded + designed (within ±1
    # for any future status emoji not in the count).
    counted = feats["production"] + feats["scaffolded"] + feats["designed"]
    assert abs(feats["total"] - counted) <= 2  # tolerance for future emoji

    # Hosted-demo URLs
    assert m["hosted_demo"]["worker_url"].startswith("https://")
    assert m["hosted_demo"]["jwks_url"].endswith("/.well-known/jwks.json")


def test_manifest_receipts_catalog_has_known_entries():
    """The receipt fixture catalogue MUST include the receipts shipped
    this session. If the fixture file moves or its schema changes,
    the catalogue should surface it."""
    from scripts.repo_manifest import build_manifest
    m = build_manifest()

    fixture_paths = {f["path"] for f in m["receipts"]["fixtures"]}
    # Known fixtures from this session
    expected = {
        "fixtures/bench_receipts/qid_accuracy_2026-04-28.json",
        "fixtures/bench_receipts/s25_canonicalization_replay_2026-04-28.json",
    }
    missing = expected - fixture_paths
    assert not missing, f"manifest missing receipts: {missing}"


# --------------------------------------------------------------------------
# Stable-fields determinism
# --------------------------------------------------------------------------


def test_stable_view_strips_time_varying_fields():
    """``_stable_view`` MUST remove fields that vary on every
    invocation, so the --check gate doesn't false-positive on
    issued_at."""
    from scripts.repo_manifest import _stable_view, build_manifest
    m = build_manifest()
    sv = _stable_view(m)
    assert "issued_at" not in sv
    assert "stars" not in sv.get("github", {})
    # Substantive fields preserved
    assert sv["git"]["commits_last_30d"] == m["git"]["commits_last_30d"]
    assert sv["features"] == m["features"]


def test_stable_view_is_idempotent():
    """Calling _stable_view on its own output produces the same dict."""
    from scripts.repo_manifest import _stable_view, build_manifest
    m = build_manifest()
    once = _stable_view(m)
    twice = _stable_view(once)
    assert once == twice


# --------------------------------------------------------------------------
# CI gate behaviour
# --------------------------------------------------------------------------


def test_check_passes_on_just_emitted_manifest(tmp_path):
    """`--check` against a manifest emitted seconds ago MUST pass
    even though `issued_at` differs (the stable view strips it)."""
    out = tmp_path / "manifest.json"
    r1 = _run([sys.executable, "-m", "scripts.repo_manifest",
               "--out", str(out)])
    assert r1.returncode == 0, r1.stderr

    r2 = _run([sys.executable, "-m", "scripts.repo_manifest",
               "--check", str(out)])
    assert r2.returncode == 0, (
        f"check failed against just-emitted manifest:\n"
        f"stdout={r2.stdout}\nstderr={r2.stderr}"
    )


def test_check_fails_on_stale_commit_count(tmp_path):
    """If a saved manifest has a stale `commits_last_30d`, the
    drift check MUST fail with exit code 1 and a refresh recipe in
    the error message."""
    from scripts.repo_manifest import build_manifest

    # Build a fresh manifest, then deliberately corrupt the commit count.
    m = build_manifest()
    m["git"]["commits_last_30d"] = -999  # impossible; surface drift
    stale = tmp_path / "stale_manifest.json"
    stale.write_text(json.dumps(m, indent=2), encoding="utf-8")

    r = _run([sys.executable, "-m", "scripts.repo_manifest",
              "--check", str(stale)])
    assert r.returncode == 1, (
        f"--check should have failed on stale manifest:\n"
        f"stdout={r.stdout}\nstderr={r.stderr}"
    )
    assert "MANIFEST DRIFT" in r.stderr
    assert "Refresh:" in r.stderr  # actionable recipe


def test_check_fails_on_missing_file(tmp_path):
    nonexistent = tmp_path / "does_not_exist.json"
    r = _run([sys.executable, "-m", "scripts.repo_manifest",
              "--check", str(nonexistent)])
    assert r.returncode != 0
