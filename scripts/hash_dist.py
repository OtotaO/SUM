#!/usr/bin/env python3
"""Emit `sum.dist_hashes.v1` JSON: SHA-256 over each file in dist/.

Single source of truth for "the bytes we built locally" — consumed by:
  - check_long_description_sync.py (after a build, validates the wheel)
  - verify_pypi_attestation.py (TestPyPI + production verification)
  - the future R0 trust-root manifest (sum.trust_root.v1.json)

Usage:
    python scripts/hash_dist.py [--dist-dir DIR] [--out FILE] [--version V]

If --out is omitted, JSON is printed to stdout. If --version is omitted,
the script reads it from pyproject.toml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def read_project_version(repo_root: Path) -> str:
    pyproject = repo_root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not find version in {pyproject}")
    return match.group(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(dist_dir: Path, version: str) -> dict:
    artifacts = []
    for p in sorted(dist_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix not in (".whl", ".gz") and not p.name.endswith(".tar.gz"):
            continue
        artifacts.append(
            {
                "filename": p.name,
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
        )
    if not artifacts:
        raise SystemExit(f"no .whl or .tar.gz files found in {dist_dir}")
    return {
        "schema": "sum.dist_hashes.v1",
        "version": version,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="directory containing built wheel/sdist (default: dist/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path for the JSON manifest (default: stdout)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="version string (default: read from pyproject.toml)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.is_dir():
        raise SystemExit(f"dist directory not found: {dist_dir}")

    version = args.version or read_project_version(repo_root)
    manifest = build_manifest(dist_dir, version)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
