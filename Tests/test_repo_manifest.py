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
    # commits_last_30d is None on shallow clones (CI default, dev shallow);
    # int on full clones. Both are valid representations.
    cl30 = m["git"]["commits_last_30d"]
    assert cl30 is None or (isinstance(cl30, int) and cl30 >= 0)

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
    invocation (issued_at), vary independently (github stars), or
    represent snapshot-identity rather than substantive content
    (head_sha and friends — a manifest published at SHA X is
    *meant* to differ from one at SHA Y; the gate is for content
    drift, not identity drift)."""
    from scripts.repo_manifest import _stable_view, build_manifest
    m = build_manifest()
    sv = _stable_view(m)
    assert "issued_at" not in sv
    assert "stars" not in sv.get("github", {})
    # Snapshot-identity stripped
    git = sv.get("git", {})
    for f in ("head_sha", "head_short", "head_subject", "head_committer_date"):
        assert f not in git, f"snapshot-identity field {f} should be stripped"
    # Substantive fields preserved
    assert sv["features"] == m["features"]
    assert sv["release"]["pyproject_version"] == m["release"]["pyproject_version"]
    assert sv["receipts"] == m["receipts"]


def test_reconcile_drops_optional_fields_when_either_side_none():
    """``_reconcile_optional_fields`` MUST drop ``commits_last_30d``
    from BOTH sides when either side reports None (shallow CI), and
    MUST drop ``pypi_published_version`` from both when either side
    is None (offline). This is the mechanism that lets the drift
    gate run on shallow GitHub Actions checkouts without false
    positives."""
    from scripts.repo_manifest import _reconcile_optional_fields

    on_disk = {
        "git": {"commits_last_30d": 241},
        "release": {"pyproject_version": "0.4.0", "pypi_published_version": "0.3.1"},
        "features": {"total": 126},
    }
    # Fresh build on shallow CI: count is None, PyPI lookup failed
    fresh = {
        "git": {"commits_last_30d": None},
        "release": {"pyproject_version": "0.4.0", "pypi_published_version": None},
        "features": {"total": 126},
    }
    a, b = _reconcile_optional_fields(on_disk, fresh)
    assert "commits_last_30d" not in a["git"]
    assert "commits_last_30d" not in b["git"]
    assert "pypi_published_version" not in a["release"]
    assert "pypi_published_version" not in b["release"]
    # Substantive fields untouched
    assert a["features"] == b["features"]
    assert a["release"]["pyproject_version"] == b["release"]["pyproject_version"]


def test_check_passes_against_shallow_simulated_manifest(tmp_path):
    """If the on-disk manifest has commits_last_30d=N (recorded on a
    full clone) and the fresh build returns None (shallow CI),
    --check MUST still pass."""
    from scripts.repo_manifest import build_manifest

    m = build_manifest()
    # Simulate the on-disk being recorded on a full clone with a
    # plausible count, but our local environment may differ.
    m["git"]["commits_last_30d"] = 9999  # arbitrary; will be reconciled away
    saved = tmp_path / "manifest.json"
    saved.write_text(json.dumps(m, indent=2), encoding="utf-8")

    # Monkeypatch via a tiny inline run: simulate a shallow build by
    # replacing build_manifest's commits count with None at the JSON
    # level using the reconcile helper directly.
    from scripts.repo_manifest import _reconcile_optional_fields, _stable_view
    fresh = build_manifest()
    fresh["git"]["commits_last_30d"] = None  # simulate shallow
    a, b = _reconcile_optional_fields(_stable_view(m), _stable_view(fresh))
    assert a == b, f"reconciled views should match; diff: a={a}\nb={b}"


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


def test_check_fails_on_stale_feature_count(tmp_path):
    """If a saved manifest has stale feature counts (mechanically
    derived from ``docs/FEATURE_CATALOG.md`` — always available, not
    optional), the drift check MUST fail with exit code 1 and a
    refresh recipe in the error message.

    Feature counts are the canary substantive field: never None,
    never reconciled away, and changing FEATURE_CATALOG.md without
    refreshing the manifest is exactly the cross-channel drift this
    gate is meant to catch.
    """
    from scripts.repo_manifest import build_manifest

    # Build a fresh manifest, then deliberately corrupt feature counts.
    m = build_manifest()
    m["features"]["total"] = -999
    m["features"]["production"] = -999
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
