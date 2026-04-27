#!/usr/bin/env python3
"""Fail closed if the built wheel's `Description` metadata diverges from
`README.md` head.

Complements:
  - `twine check` — validates the long-description renders on Warehouse.
  - `check-wheel-contents` — validates the wheel's file tree.

This script answers a question neither does: "is this actually the README
we intended to ship?" Required because PyPA's metadata model freezes the
long-description at publish time; the wheel's `Description` rots
independently of the GitHub README otherwise (this is the exact failure
mode v0.3.1 was cut to fix).

Usage:
    python scripts/check_long_description_sync.py [--wheel PATH] [--readme PATH]

If --wheel is omitted, the script picks the most recently-modified .whl
under dist/. If --readme is omitted, README.md at the repo root is used.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path


METADATA_FIELD = "Description"


def find_default_wheel(dist_dir: Path) -> Path:
    candidates = sorted(
        (p for p in dist_dir.iterdir() if p.suffix == ".whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"no .whl found in {dist_dir}")
    return candidates[0]


def read_wheel_description(wheel: Path) -> str:
    """Extract the Description body from `*.dist-info/METADATA`.

    METADATA is RFC 822-shaped. The `Description` field can be either:
      - A header line `Description: ...` (rare for long descriptions).
      - A blank line followed by the description body (canonical for
        modern wheels; `Description-Content-Type` declares the format).
    Per PEP 566 / PEP 643, the canonical form for non-trivial
    descriptions is the trailing-body form. We handle both.
    """
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

    # Split header from body at the first blank line (RFC 822 boundary).
    if "\n\n" in raw:
        header, body = raw.split("\n\n", 1)
    else:
        header, body = raw, ""

    # If a Description: header line exists, that's the canonical value.
    for line in header.splitlines():
        m = re.match(r"^Description:\s*(.*)$", line)
        if m:
            head_value = m.group(1)
            if head_value.strip():
                return head_value
            break

    # Otherwise the body IS the description.
    return body


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        default=None,
        help="path to .whl (default: most recent in dist/)",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="path to README.md (default: repo-root README.md)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    wheel = Path(args.wheel) if args.wheel else find_default_wheel(repo_root / "dist")
    readme = (repo_root / args.readme) if not Path(args.readme).is_absolute() else Path(args.readme)

    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")
    if not readme.is_file():
        raise SystemExit(f"readme not found: {readme}")

    wheel_desc = normalise(read_wheel_description(wheel))
    readme_text = normalise(readme.read_text(encoding="utf-8"))

    if wheel_desc == readme_text:
        print(
            f"OK: wheel `{METADATA_FIELD}` matches README.md "
            f"({len(readme_text)} chars after normalisation)"
        )
        return 0

    sys.stderr.write(
        f"FAIL: wheel `{METADATA_FIELD}` diverges from README.md.\n"
        f"  wheel:  {wheel} ({len(wheel_desc)} chars)\n"
        f"  readme: {readme} ({len(readme_text)} chars)\n"
        f"\n"
        f"This is the failure mode v0.3.1 was cut to fix. The wheel\n"
        f"will publish a stale long-description to PyPI; do not promote.\n"
        f"\n"
    )

    # Show first diverging line for debugging.
    a_lines = wheel_desc.split("\n")
    b_lines = readme_text.split("\n")
    for i in range(min(len(a_lines), len(b_lines))):
        if a_lines[i] != b_lines[i]:
            sys.stderr.write(f"  first diff at line {i + 1}:\n")
            sys.stderr.write(f"    wheel:  {a_lines[i]!r}\n")
            sys.stderr.write(f"    readme: {b_lines[i]!r}\n")
            break
    else:
        if len(a_lines) != len(b_lines):
            sys.stderr.write(
                f"  identical through line {min(len(a_lines), len(b_lines))}, "
                f"differ in length ({len(a_lines)} vs {len(b_lines)} lines)\n"
            )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
