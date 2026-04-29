"""Repo manifest publisher — single source of truth for cross-channel state.

Publishes a JSON manifest describing the SUM repo's current state.
Downstream consumers (the SUMequities portfolio at
``https://www.sumequities.com/projects/sum/``, dashboards,
status pages, anyone) fetch this manifest from a stable URL
and compute their displayed values from it. Eliminates the
class of cross-channel drift surfaced by the 2026-04-29 audit
(portfolio said ``100 commits / 30d``; actual was ``236``).

The manifest fields are deliberately load-bearing — every one
maps to a public claim somewhere in the docs or downstream
surfaces. If a field becomes outdated, exactly one place
(this script) needs to change.

**Schema:** ``sum.repo_manifest.v1`` — versioned and forward-
compatible per ``docs/COMPATIBILITY_POLICY.md``.

**Reproducibility contract:** the manifest is computed from
git, the filesystem, the network (PyPI), and `gh` CLI. Same
inputs, same output. No timestamps inside the signed surface
(``issued_at`` is metadata; the body is deterministic given a
fixed point in time).

**Forward-compat:** the manifest can be Ed25519-signed with
the trust-root key (operator decision; not done in v1).
v1 callers are expected to verify the manifest's ``schema``
field and tolerate unknown additive fields per
``docs/COMPATIBILITY_POLICY.md``.

Usage::

    python -m scripts.repo_manifest
    python -m scripts.repo_manifest --out meta/repo_manifest.json
    python -m scripts.repo_manifest --check meta/repo_manifest.json
        # exits non-zero if the on-disk manifest is stale

The --check mode is the CI gate: if a PR changes anything that
the manifest reflects (commit count, feature counts, version)
without re-running the publisher, CI surfaces the drift.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "sum.repo_manifest.v1"

_USER_AGENT = "sum-repo-manifest/0.1 (+https://github.com/OtotaO/SUM)"


# ---------------------------------------------------------------------
# Field collectors — each is a pure-ish function over a single source
# ---------------------------------------------------------------------


def _git(*args: str) -> str:
    """Run a git command in the repo root and return stripped stdout."""
    r = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.strip()}")
    return r.stdout.strip()


def _is_shallow_repo() -> bool:
    """True if the working clone is a shallow checkout (e.g. CI default)."""
    try:
        return _git("rev-parse", "--is-shallow-repository") == "true"
    except RuntimeError:
        return False


def collect_git_state() -> dict:
    """Git-derived facts. Does not call out to GitHub.

    On shallow clones (default in GitHub Actions ``actions/checkout``)
    the 30-day commit count is meaningless — only commits in the
    shallow window are visible. We surface ``None`` in that case so
    the drift gate doesn't false-positive on shallow CI checkouts.
    """
    head_sha = _git("rev-parse", "HEAD")
    head_short = _git("rev-parse", "--short", "HEAD")
    head_subject = _git("log", "-1", "--format=%s")
    head_committer_date = _git("log", "-1", "--format=%cI")  # ISO 8601 with TZ

    if _is_shallow_repo():
        commits_last_30d: int | None = None
    else:
        # Commits in the last 30 days. Use --since=ISO timestamp for portability.
        thirty_d_ago = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        commits_last_30d = len(
            _git("log", f"--since={thirty_d_ago}", "--oneline").splitlines()
        )

    return {
        "head_sha": head_sha,
        "head_short": head_short,
        "head_subject": head_subject,
        "head_committer_date": head_committer_date,
        "commits_last_30d": commits_last_30d,
    }


def collect_feature_catalog_counts() -> dict:
    """Mechanically count Production/Scaffolded/Designed in FEATURE_CATALOG.md."""
    fc = (REPO_ROOT / "docs" / "FEATURE_CATALOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^### .*$", fc, flags=re.MULTILINE)
    return {
        "total": len(headings),
        "production": sum(1 for h in headings if "✅" in h),
        "scaffolded": sum(1 for h in headings if "🔧" in h),
        "designed": sum(1 for h in headings if "📄" in h),
    }


def collect_pyproject_version() -> str:
    """Extract version from pyproject.toml."""
    pp = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pp, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("pyproject.toml: version line not found")
    return m.group(1)


def collect_pypi_published_version(timeout_s: float = 10.0) -> str | None:
    """Fetch the latest version published to PyPI for sum-engine.

    Returns None on network failure; the manifest is still valid
    without this field.
    """
    try:
        req = urllib.request.Request(
            "https://pypi.org/pypi/sum-engine/json",
            headers={"user-agent": _USER_AGENT, "accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["info"]["version"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        return None


def collect_github_stars(timeout_s: float = 10.0) -> int | None:
    """Fetch GitHub star count via the `gh` CLI.

    Returns None if `gh` is unavailable or the API call fails.
    """
    try:
        r = subprocess.run(
            ["gh", "repo", "view", "OtotaO/SUM",
             "--json", "stargazerCount", "-q", ".stargazerCount"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def collect_receipt_fixtures() -> list[dict]:
    """Catalog the receipt fixtures shipped in fixtures/bench_receipts/."""
    fixtures_dir = REPO_ROOT / "fixtures" / "bench_receipts"
    if not fixtures_dir.exists():
        return []
    out: list[dict] = []
    for p in sorted(fixtures_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        # Both single-schema and schema_family receipts are catalogued.
        schema = data.get("schema") or data.get("schema_family")
        out.append({
            "path": str(p.relative_to(REPO_ROOT)),
            "schema": schema,
            "issued_at": data.get("issued_at"),
        })
    return out


# ---------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------


def build_manifest() -> dict:
    """Assemble the full manifest. Called from --emit and --check."""
    git_state = collect_git_state()
    return {
        "schema": SCHEMA,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": {
            "owner": "OtotaO",
            "name": "SUM",
            "url": "https://github.com/OtotaO/SUM",
            "license": "Apache-2.0",
        },
        "git": git_state,
        "github": {
            "stars": collect_github_stars(),
        },
        "release": {
            "pyproject_version": collect_pyproject_version(),
            "pypi_published_version": collect_pypi_published_version(),
        },
        "features": collect_feature_catalog_counts(),
        "receipts": {
            "fixtures": collect_receipt_fixtures(),
        },
        "hosted_demo": {
            "worker_url": "https://sum-demo.ototao.workers.dev",
            "jwks_url": "https://sum-demo.ototao.workers.dev/.well-known/jwks.json",
            "revoked_kids_url": (
                "https://sum-demo.ototao.workers.dev/.well-known/revoked-kids.json"
            ),
        },
    }


# ---------------------------------------------------------------------
# Stable comparison — exclude time-varying fields for --check
# ---------------------------------------------------------------------


_SNAPSHOT_IDENTITY_FIELDS = (
    "head_sha",
    "head_short",
    "head_subject",
    "head_committer_date",
)


def _stable_view(manifest: dict) -> dict:
    """Strip fields that vary by wall-clock time / by HEAD identity so
    --check only fails on substantive drift (features, versions,
    receipts, commit-velocity when measurable).

    Removed:
      - ``issued_at``                                   (wall-clock)
      - ``github.stars``                                (varies independently)
      - ``git.head_sha`` / ``head_short`` / ``head_subject`` /
        ``head_committer_date``                          (snapshot identity:
        a manifest published at SHA X is meant to differ from one published
        at SHA Y; the drift gate is for *content*, not identity)
      - ``git.commits_last_30d`` *if either side is None*  (shallow CI clones
        cannot count 30-day history; we don't false-positive on that)
      - ``release.pypi_published_version`` *if either side is None* (offline)
    """
    m = json.loads(json.dumps(manifest))  # deep copy via json
    m.pop("issued_at", None)
    if "github" in m and isinstance(m["github"], dict):
        m["github"].pop("stars", None)
    if "git" in m and isinstance(m["git"], dict):
        for f in _SNAPSHOT_IDENTITY_FIELDS:
            m["git"].pop(f, None)
    return m


def _reconcile_optional_fields(on_disk: dict, fresh: dict) -> tuple[dict, dict]:
    """Drop fields that are ``None`` on either side from BOTH copies so
    --check tolerates partial inputs (shallow CI, offline PyPI).

    The on-disk manifest is the canonical record; the fresh build runs
    in whatever environment CI gave us. If the canonical says
    ``commits_last_30d=241`` but the fresh build is shallow and reports
    ``None``, that's an environment limitation, not drift.
    """
    a, b = json.loads(json.dumps(on_disk)), json.loads(json.dumps(fresh))

    # git.commits_last_30d
    if isinstance(a.get("git"), dict) and isinstance(b.get("git"), dict):
        if a["git"].get("commits_last_30d") is None or b["git"].get("commits_last_30d") is None:
            a["git"].pop("commits_last_30d", None)
            b["git"].pop("commits_last_30d", None)

    # release.pypi_published_version
    if isinstance(a.get("release"), dict) and isinstance(b.get("release"), dict):
        if a["release"].get("pypi_published_version") is None or b["release"].get("pypi_published_version") is None:
            a["release"].pop("pypi_published_version", None)
            b["release"].pop("pypi_published_version", None)

    return a, b


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="Write the manifest to FILE (default: stdout).",
    )
    parser.add_argument(
        "--check",
        default=None,
        metavar="FILE",
        help=(
            "Compare the on-disk manifest at FILE against a fresh build. "
            "Exits 0 if the stable-fields portion matches; non-zero on drift "
            "(commit count, feature counts, version, receipts changed without "
            "the manifest being refreshed)."
        ),
    )
    args = parser.parse_args()

    if args.check:
        check_path = Path(args.check)
        if not check_path.exists():
            print(f"check: {check_path} does not exist", file=sys.stderr)
            return 2
        on_disk = json.loads(check_path.read_text(encoding="utf-8"))
        fresh = build_manifest()
        on_disk_stable, fresh_stable = _reconcile_optional_fields(
            _stable_view(on_disk), _stable_view(fresh)
        )
        if on_disk_stable != fresh_stable:
            # Compute a small diff for the failure message.
            diffs: list[str] = []
            for key in set(on_disk_stable.keys()) | set(fresh_stable.keys()):
                if on_disk_stable.get(key) != fresh_stable.get(key):
                    diffs.append(f"  {key}:")
                    diffs.append(f"    on-disk: {on_disk_stable.get(key)}")
                    diffs.append(f"    fresh:   {fresh_stable.get(key)}")
            print(
                "MANIFEST DRIFT: on-disk manifest is stale.\n"
                "Refresh: python -m scripts.repo_manifest --out " + str(check_path),
                file=sys.stderr,
            )
            print("\n".join(diffs), file=sys.stderr)
            return 1
        print(f"manifest: {check_path} is current (stable-fields match)", file=sys.stderr)
        return 0

    manifest = build_manifest()
    text = json.dumps(manifest, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"manifest written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
