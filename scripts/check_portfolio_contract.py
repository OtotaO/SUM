"""PORTFOLIO.md contract check — run in CI as a blocking gate.

The PORTFOLIO.md contract (see CLAUDE.md) requires every metric
claim on the consulting-prospect-facing page to be labelled with
its epistemic status: ``**proved**`` for mathematically proved
invariants and ``**empirical-benchmark**`` for measured numbers.
A portfolio that mixes labelled and unlabelled claims invites
the unlabelled ones to drift into marketing; this script keeps
the discipline mechanical.

Scope: only the **metric table** under ``## Current State``. That
section's rows are the ones directly read by prospects scanning
for numbers. Prose elsewhere in the document (Future Directions,
Technical Stack, the one-line hook, compatibility notes like
"Chrome 113+") is not a metric claim even when it contains digits,
and the contract intentionally does not gate it — the rule would
false-positive on too many non-claim strings.

Specifically:
  * Find the first Markdown table under ``## Current State``.
  * For every body row (not the header, not the separator):
      - If the row contains a digit (``\\d``),
      - Then it must also contain either the exact string
        ``**proved**`` or ``**empirical-benchmark**`` somewhere
        on that row.
  * Any violation → exit 1 with the offending row and a line
    number. Zero violations → exit 0.

Usage:
    python -m scripts.check_portfolio_contract
    python -m scripts.check_portfolio_contract --file PORTFOLIO.md
    python -m scripts.check_portfolio_contract --strict      # future use

CI wiring: .github/workflows/quantum-ci.yml :: portfolio-contract
job. Runs on every push to main and every PR. Blocks merge on a
non-zero exit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REQUIRED_LABELS = ("**proved**", "**empirical-benchmark**")
_CURRENT_STATE_HEADING = re.compile(r"^##\s+Current State\s*$")
_DIGIT = re.compile(r"\d")
# A Markdown table row starts with '|' after any leading whitespace;
# the separator row after the header consists of only pipes, dashes,
# colons, and whitespace (e.g. `|---|---|---|`).
_TABLE_ROW = re.compile(r"^\s*\|")
_SEPARATOR_ROW = re.compile(r"^\s*\|[\s\-:\|]+$")


def _find_metric_table(lines: list[str]) -> tuple[int, int] | None:
    """Return (first_body_row_index, end_exclusive) of the first
    markdown table under ``## Current State``, or None if either the
    heading or its table is missing.

    Indices are 0-based line numbers into ``lines``. The returned
    range covers the **body** rows only (not header, not separator).
    The table is considered to end at the first non-table line after
    the body rows begin.
    """
    in_section = False
    header_idx: int | None = None
    separator_idx: int | None = None
    for i, line in enumerate(lines):
        if _CURRENT_STATE_HEADING.match(line):
            in_section = True
            continue
        if not in_section:
            continue
        # Leaving the section at the next top-level-or-same heading.
        if line.startswith("## "):
            break
        if _TABLE_ROW.match(line):
            if header_idx is None:
                header_idx = i
                continue
            if separator_idx is None and _SEPARATOR_ROW.match(line):
                separator_idx = i
                continue
            # First body row found.
            if header_idx is not None and separator_idx is not None:
                body_start = i
                # Extend the body until the first non-table line.
                body_end = body_start
                while body_end < len(lines) and _TABLE_ROW.match(lines[body_end]):
                    body_end += 1
                return (body_start, body_end)
    return None


def _violations(lines: list[str], body_range: tuple[int, int]) -> list[tuple[int, str]]:
    start, end = body_range
    violations: list[tuple[int, str]] = []
    for i in range(start, end):
        row = lines[i]
        if not _DIGIT.search(row):
            # Row has no digits → not a metric claim, skipped.
            continue
        if any(label in row for label in _REQUIRED_LABELS):
            continue
        violations.append((i + 1, row.rstrip()))  # 1-based line number for humans.
    return violations


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="check_portfolio_contract",
        description="Verify PORTFOLIO.md metric-table rows are epistemically labelled.",
    )
    p.add_argument(
        "--file",
        default="PORTFOLIO.md",
        help="Path to the portfolio markdown file. Default: PORTFOLIO.md at cwd.",
    )
    args = p.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"check_portfolio_contract: {path} not found", file=sys.stderr)
        return 2

    lines = path.read_text(encoding="utf-8").splitlines()
    table = _find_metric_table(lines)
    if table is None:
        print(
            f"check_portfolio_contract: no metric table under '## Current State' in {path}. "
            f"PORTFOLIO.md must carry a pipe-delimited metric table there; see "
            f"CLAUDE.md → 'PORTFOLIO.md contract' → Required structure.",
            file=sys.stderr,
        )
        return 1

    violations = _violations(lines, table)
    if violations:
        print(
            f"check_portfolio_contract: {len(violations)} metric row(s) in {path} "
            f"lack '**proved**' or '**empirical-benchmark**':",
            file=sys.stderr,
        )
        for lineno, row in violations:
            print(f"  {path}:{lineno}: {row}", file=sys.stderr)
        print(
            "\nEvery metric-table row must be labelled. See CLAUDE.md → "
            "'PORTFOLIO.md contract' → Prohibited.",
            file=sys.stderr,
        )
        return 1

    body_count = table[1] - table[0]
    print(
        f"check_portfolio_contract: {path} metric table OK ({body_count} row(s), "
        f"all epistemically labelled)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
