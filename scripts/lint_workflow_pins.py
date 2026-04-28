#!/usr/bin/env python3
"""Fail closed if any GitHub Actions workflow pins a third-party action by
mutable tag instead of full commit SHA.

R0.3 hardening lint. Per ``docs/NEXT_SESSION_PLAYBOOK.md`` R0.3, every
``uses: <owner>/<repo>@<ref>`` line in ``.github/workflows/*.yml`` MUST
pin to a 40-char lowercase hex commit SHA. Tag pins (``@v4``,
``@release/v1``, ``@main``) are mutable: an attacker who compromises
the action's repo can re-point the tag at malicious code, and
SHA-stripped pulls execute it without further consent.

The convention this lint enforces:

    - uses: actions/checkout@<40-hex-sha>  # v4

The trailing ``# vX.Y.Z`` comment is human-readable; the SHA is the
load-bearing pin.

Allowed exceptions: GitHub-built-in path-style references (``./local``
composite actions, ``docker://image:tag``). These are not third-party
GitHub Actions and are out of scope for the SHA-pin rule.

Run:
    python scripts/lint_workflow_pins.py
Exit 0 on success, 1 on any tag-pinned action.

Re-run after a dependency bump:
    1. ``gh api repos/<org>/<repo>/git/ref/tags/<tag>`` to get the SHA.
    2. Update the workflow line: ``@<new-sha>  # <tag>``.
    3. Run this lint locally. CI re-runs on every push.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# Strict SHA shape: 40 lowercase hex chars.
_SHA_PIN_RE = re.compile(r"@[0-9a-f]{40}(?:\s|$)")
# Captures `uses: <something>@<ref>` where <ref> is a tag/branch
# (not a SHA). The regex tolerates leading whitespace + hyphens for
# both `- uses:` and `      uses:` forms.
_USES_LINE_RE = re.compile(r"^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)")


def _is_local_or_docker(target: str) -> bool:
    """`./local-action` and `docker://image` references aren't
    third-party GitHub Actions and don't fit the SHA-pin rule."""
    return target.startswith(("./", "docker://", "/"))


def lint_workflow(path: Path) -> list[tuple[int, str, str]]:
    """Return a list of (line_no, target, ref) for every non-SHA pin."""
    failures: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            # Skip lines that are clearly comments. The capture group
            # in _USES_LINE_RE doesn't match commented lines anyway,
            # but we also need to skip cases where `uses:` appears in
            # a multi-line YAML comment — `# uses: ...` would match
            # the regex without this guard.
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue

            m = _USES_LINE_RE.match(line)
            if not m:
                continue
            target, ref = m.group(1), m.group(2)
            if _is_local_or_docker(target):
                continue

            # Accept full-SHA pins (40 hex chars). Reject anything else.
            if re.fullmatch(r"[0-9a-f]{40}", ref):
                continue
            failures.append((lineno, target, ref))
    return failures


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        print(f"no workflows directory at {workflows_dir}", file=sys.stderr)
        return 0

    workflow_files = sorted(workflows_dir.glob("*.yml")) + sorted(
        workflows_dir.glob("*.yaml")
    )
    if not workflow_files:
        print("no workflow files found", file=sys.stderr)
        return 0

    total_failures = 0
    files_clean = 0
    files_failed = 0
    for path in workflow_files:
        failures = lint_workflow(path)
        if failures:
            files_failed += 1
            print(f"\n✗ {path.relative_to(repo_root)}: {len(failures)} unpinned action(s)")
            for lineno, target, ref in failures:
                print(f"    line {lineno:>3}: {target}@{ref}")
                print(f"             ↳ resolve via: "
                      f"gh api repos/{target}/git/ref/tags/{ref} | jq .object.sha")
            total_failures += len(failures)
        else:
            files_clean += 1
            print(f"✓ {path.relative_to(repo_root)}")

    print()
    if total_failures:
        print(
            f"FAIL: {files_failed} workflow file(s), {total_failures} "
            f"third-party action(s) pinned by mutable tag instead of SHA.\n"
            f"\n"
            f"Remediation: replace each `@<tag>` with `@<full-sha>  # <tag>`.\n"
            f"Resolve a tag's current SHA with:\n"
            f"    gh api repos/<owner>/<repo>/git/ref/tags/<tag> "
            f"| jq -r .object.sha\n"
            f"\n"
            f"This is R0.3's load-bearing supply-chain protection: a tag-\n"
            f"pinned action whose upstream repo is compromised executes\n"
            f"the new code on every push. SHA-pinning forces an explicit\n"
            f"opt-in to any new upstream code.\n"
        )
        return 1
    print(f"OK: {files_clean} workflow file(s), all third-party actions SHA-pinned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
