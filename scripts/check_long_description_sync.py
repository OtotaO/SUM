#!/usr/bin/env python3
"""Fail closed if the built distribution's `Description` metadata diverges
from `README.md` head — checks BOTH the wheel's `*.dist-info/METADATA` AND
the sdist's `PKG-INFO`.

Complements:
  - `twine check` — validates the long-description renders on Warehouse.
  - `check-wheel-contents` — validates the wheel's file tree.

This script answers a question neither does: "is this actually the README
we intended to ship?" Required because PyPA's metadata model freezes the
long-description at publish time; the wheel/sdist `Description` rots
independently of the GitHub README otherwise (this is the exact failure
mode v0.3.1 was cut to fix).

The release builds BOTH a wheel and an sdist. Both go to PyPI. Both
carry independently-encoded long-descriptions. This script checks both.

Usage:
    python scripts/check_long_description_sync.py
        [--wheel PATH] [--sdist PATH] [--readme PATH]
        [--no-wheel] [--no-sdist]

If --wheel is omitted, the script picks the most recently-modified .whl
under dist/. If --sdist is omitted, the most recent .tar.gz. If --readme
is omitted, README.md at the repo root is used. --no-wheel and --no-sdist
skip the respective check (useful when only one artifact has been built).
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path


METADATA_FIELD = "Description"


def find_default_wheel(dist_dir: Path) -> Path | None:
    candidates = sorted(
        (p for p in dist_dir.iterdir() if p.suffix == ".whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_default_sdist(dist_dir: Path) -> Path | None:
    candidates = sorted(
        (p for p in dist_dir.iterdir() if p.name.endswith(".tar.gz")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_pkg_metadata(raw: str) -> str:
    """Extract Description from RFC 822-shaped PKG-INFO / METADATA.

    The `Description` field can be either:
      - A header line `Description: ...` (rare for long descriptions).
      - A blank line followed by the description body (canonical for
        modern packages; `Description-Content-Type` declares the format).
    Per PEP 566 / PEP 643, the canonical form for non-trivial
    descriptions is the trailing-body form. We handle both.
    """
    if "\n\n" in raw:
        header, body = raw.split("\n\n", 1)
    else:
        header, body = raw, ""

    for line in header.splitlines():
        m = re.match(r"^Description:\s*(.*)$", line)
        if m:
            head_value = m.group(1)
            if head_value.strip():
                return head_value
            break

    return body


def read_wheel_description(wheel: Path) -> str:
    """Extract Description from a wheel's `*.dist-info/METADATA`."""
    with zipfile.ZipFile(wheel) as zf:
        meta_names = [
            n for n in zf.namelist()
            if n.endswith(".dist-info/METADATA")
        ]
        if not meta_names:
            raise SystemExit(f"no dist-info/METADATA found in {wheel}")
        if len(meta_names) > 1:
            raise SystemExit(f"multiple METADATA entries in {wheel}: {meta_names}")
        with zf.open(meta_names[0]) as f:
            raw = f.read().decode("utf-8")
    return parse_pkg_metadata(raw)


def read_sdist_description(sdist: Path) -> str:
    """Extract Description from an sdist's top-level `PKG-INFO`."""
    with tarfile.open(sdist, "r:gz") as tf:
        pkg_info_names = [
            m.name for m in tf.getmembers()
            if m.name.endswith("/PKG-INFO") and m.name.count("/") == 1
        ]
        if not pkg_info_names:
            raise SystemExit(f"no top-level PKG-INFO found in {sdist}")
        if len(pkg_info_names) > 1:
            raise SystemExit(f"multiple top-level PKG-INFO in {sdist}: {pkg_info_names}")
        member = tf.getmember(pkg_info_names[0])
        f = tf.extractfile(member)
        if f is None:
            raise SystemExit(f"could not extract PKG-INFO from {sdist}")
        raw = f.read().decode("utf-8")
    return parse_pkg_metadata(raw)


def normalise(text: str) -> str:
    """Newline-normalise + strip trailing whitespace per line.

    The wheel's METADATA encoder may emit \\r\\n or strip trailing spaces.
    Normalising both sides eliminates those harmless drifts so the diff
    catches actual content divergence only.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Trim trailing empty lines (PyPA's metadata writer adds them).
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def report_failure(label: str, source: Path, source_desc: str, readme: Path, readme_text: str) -> None:
    sys.stderr.write(
        f"FAIL: {label} `{METADATA_FIELD}` diverges from README.md.\n"
        f"  {label}:  {source} ({len(source_desc)} chars)\n"
        f"  readme: {readme} ({len(readme_text)} chars)\n"
        f"\n"
        f"This is the failure mode v0.3.1 was cut to fix. The {label}\n"
        f"will publish a stale long-description to PyPI; do not promote.\n"
        f"\n"
    )

    a_lines = source_desc.split("\n")
    b_lines = readme_text.split("\n")
    for i in range(min(len(a_lines), len(b_lines))):
        if a_lines[i] != b_lines[i]:
            sys.stderr.write(f"  first diff at line {i + 1}:\n")
            sys.stderr.write(f"    {label}:  {a_lines[i]!r}\n")
            sys.stderr.write(f"    readme: {b_lines[i]!r}\n")
            return
    if len(a_lines) != len(b_lines):
        sys.stderr.write(
            f"  identical through line {min(len(a_lines), len(b_lines))}, "
            f"differ in length ({len(a_lines)} vs {len(b_lines)} lines)\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", default=None, help="path to .whl (default: most recent in dist/)")
    parser.add_argument("--sdist", default=None, help="path to .tar.gz (default: most recent in dist/)")
    parser.add_argument("--readme", default="README.md", help="path to README.md (default: repo-root)")
    parser.add_argument("--no-wheel", action="store_true", help="skip wheel METADATA check")
    parser.add_argument("--no-sdist", action="store_true", help="skip sdist PKG-INFO check")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    readme = (repo_root / args.readme) if not Path(args.readme).is_absolute() else Path(args.readme)
    if not readme.is_file():
        raise SystemExit(f"readme not found: {readme}")
    readme_text = normalise(readme.read_text(encoding="utf-8"))

    failures = 0
    checks_run = 0

    # Wheel METADATA check
    if not args.no_wheel:
        wheel = Path(args.wheel) if args.wheel else find_default_wheel(repo_root / "dist")
        if wheel is None:
            raise SystemExit(
                "no .whl found in dist/ (pass --wheel PATH or --no-wheel to skip)"
            )
        if not wheel.is_file():
            raise SystemExit(f"wheel not found: {wheel}")
        wheel_desc = normalise(read_wheel_description(wheel))
        checks_run += 1
        if wheel_desc == readme_text:
            print(
                f"OK: wheel `{METADATA_FIELD}` matches README.md "
                f"({len(readme_text)} chars after normalisation)"
            )
        else:
            report_failure("wheel", wheel, wheel_desc, readme, readme_text)
            failures += 1

    # Sdist PKG-INFO check
    if not args.no_sdist:
        sdist = Path(args.sdist) if args.sdist else find_default_sdist(repo_root / "dist")
        if sdist is None:
            raise SystemExit(
                "no .tar.gz found in dist/ (pass --sdist PATH or --no-sdist to skip)"
            )
        if not sdist.is_file():
            raise SystemExit(f"sdist not found: {sdist}")
        sdist_desc = normalise(read_sdist_description(sdist))
        checks_run += 1
        if sdist_desc == readme_text:
            print(
                f"OK: sdist `{METADATA_FIELD}` matches README.md "
                f"({len(readme_text)} chars after normalisation)"
            )
        else:
            report_failure("sdist", sdist, sdist_desc, readme, readme_text)
            failures += 1

    if checks_run == 0:
        raise SystemExit("--no-wheel and --no-sdist both set; nothing to check")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
