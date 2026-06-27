#!/usr/bin/env python3
"""Fail closed if outbound text contains an em dash, a leaked secret, or a
denylisted token; warn on a stray shell-command line.

"Outbound" = any text that leaves in the operator's voice: a PR/issue comment
body, a commit message, an email draft, release notes. This is a *draft-time*
gate: pipe a draft through it (or pass the file) BEFORE you post, so a known,
mechanically-checkable rule is enforced at the moment of emit instead of left to
in-the-moment recall.

Why this exists (session audit, 2026-06-25): the operator's "no em dashes"
preference lived only as a fact to recall, enforced by a grep I had to remember
to run — so an em dash leaked into a reply draft. A recalled rule has a non-zero
miss rate by construction; a gate does not. The rules live in one committed
config (``scripts/authoring_rules.json``), never in memory. This is deliberately
the ONLY authoring gate; per the same audit, formalising soft/judgement-shaped
process would be substrate-velocity.

Hard failures (exit 1):
  - forbidden glyphs (em dash, horizontal bar) and em-dash-substitute patterns
    (spaced " -- ", spaced en dash); unspaced en dashes in ranges are allowed
  - leaked secrets (GitHub/OpenAI/Anthropic/AWS keys, private-key headers,
    Bearer tokens, ``api_key=``/``token=``/``password=`` assignments)
  - any denylisted literal token from the config

Warnings (exit 1 only with --strict):
  - a stray shell-command line OUTSIDE a fenced code block, e.g. a trailing
    ``gh issue comment ...`` accidentally pasted into a comment body

Run:
    python scripts/lint_outbound_text.py DRAFT.md          # check a file
    pbpaste | python scripts/lint_outbound_text.py -        # check stdin
    python scripts/lint_outbound_text.py --selftest         # verify the linter
    python scripts/lint_outbound_text.py --list-rules        # show active rules
Exit 0 when clean, 1 on any hard failure (or any warning under --strict).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "authoring_rules.json"

# Universal secret shapes (baked in, NOT config — these must never be editable
# away by accident). Each is (label, compiled-regex). Kept deliberately tight to
# stay near-zero false-positive; when unsure, omit rather than over-match.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("OpenAI/Anthropic key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key header", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b")),
    (
        "secret assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[=:]\s*"
            r"['\"]?[A-Za-z0-9._\-/+]{12,}",
        ),
    ),
]

_FENCE_RE = re.compile(r"^\s*(```|~~~)")


class Finding:
    __slots__ = ("line", "col", "rule", "severity", "message", "snippet")

    def __init__(self, line: int, col: int, rule: str, severity: str, message: str, snippet: str):
        self.line, self.col = line, col
        self.rule, self.severity, self.message, self.snippet = rule, severity, message, snippet


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _command_line_re(prefixes: list[str]) -> re.Pattern[str]:
    # A line that, after markdown list/quote markers, begins with a known CLI
    # invocation — the "trailing terminal command swept into the body" near-miss.
    alts = "|".join(re.escape(p.rstrip()) for p in prefixes if p.strip())
    return re.compile(rf"^\s*(?:[-*>]\s*)?(?:{alts})\b")


def lint_text(text: str, cfg: dict) -> list[Finding]:
    findings: list[Finding] = []
    glyphs = cfg.get("forbidden_glyphs", [])
    patterns = cfg.get("forbidden_patterns", [])
    denylist = cfg.get("denylist_tokens", [])
    cmd_re = _command_line_re(cfg.get("command_line_prefixes", []))

    in_fence = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if _FENCE_RE.match(raw):
            in_fence = not in_fence

        # Forbidden glyphs — checked everywhere (hard).
        for g in glyphs:
            ch = g["glyph"]
            start = 0
            while (idx := raw.find(ch, start)) != -1:
                findings.append(Finding(lineno, idx + 1, "glyph", "error", g["name"], raw.strip()))
                start = idx + 1

        # Forbidden literal patterns (em-dash substitutes) — hard.
        for p in patterns:
            pat = p["pattern"]
            start = 0
            while (idx := raw.find(pat, start)) != -1:
                findings.append(Finding(lineno, idx + 1, "pattern", "error", p["name"], raw.strip()))
                start = idx + 1

        # Denylisted tokens — hard.
        for tok in denylist:
            if not tok:
                continue
            idx = raw.find(tok)
            if idx != -1:
                findings.append(Finding(lineno, idx + 1, "denylist", "error", f"denylisted token {tok!r}", raw.strip()))

        # Secrets — hard, checked everywhere.
        for label, rx in _SECRET_PATTERNS:
            m = rx.search(raw)
            if m:
                findings.append(Finding(lineno, m.start() + 1, "secret", "error", f"possible {label} leaked", "<redacted line>"))

        # Stray command line — warning, only OUTSIDE code fences.
        if not in_fence and cmd_re.match(raw):
            findings.append(Finding(lineno, 1, "stray-command", "warn", "line looks like a stray shell command", raw.strip()))

    return findings


def _report(label: str, findings: list[Finding]) -> tuple[int, int]:
    errors = [f for f in findings if f.severity == "error"]
    warns = [f for f in findings if f.severity == "warn"]
    if not findings:
        print(f"✓ {label}")
        return 0, 0
    mark = "✗" if errors else "⚠"
    print(f"\n{mark} {label}: {len(errors)} error(s), {len(warns)} warning(s)")
    for f in findings:
        tag = "ERROR" if f.severity == "error" else "warn "
        print(f"    {tag} {f.line}:{f.col} [{f.rule}] {f.message}")
        print(f"          ↳ {f.snippet}")
    return len(errors), len(warns)


_SELFTEST_CASES = [
    ("clean", "Good catch. The fix records the operator and replays from local files.\n", 0, 0),
    ("em dash", "This is fine — but this is not.\n", 1, 0),
    ("spaced double hyphen", "This is fine -- but this is not.\n", 1, 0),
    ("github token", "use this: ghp_0123456789ABCDEFabcdef0123456789ABCD\n", 1, 0),
    ("secret assignment", "api_key = sk_live_0123456789abcdef\n", 1, 0),
    ("stray command outside fence", "Thanks for the review.\ngh pr comment 382 --body hi\n", 0, 1),
    ("command inside fence is fine", "```\ngh pr comment 382 --body hi\n```\n", 0, 0),
    ("unspaced en dash range ok", "Coverage held across 2024–2025 runs.\n", 0, 0),
]


def selftest(cfg: dict) -> int:
    ok = True
    for name, text, want_err, want_warn in _SELFTEST_CASES:
        fs = lint_text(text, cfg)
        e = sum(1 for f in fs if f.severity == "error")
        w = sum(1 for f in fs if f.severity == "warn")
        passed = e == want_err and w == want_warn
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: got {e} err / {w} warn (want {want_err}/{want_warn})")
    print("\nselftest:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Lint outbound text (draft-time gate).")
    ap.add_argument("files", nargs="*", help="files to check; use '-' for stdin")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--selftest", action="store_true", help="run built-in self-test and exit")
    ap.add_argument("--list-rules", action="store_true", help="print active rules and exit")
    args = ap.parse_args()

    cfg = load_config()

    if args.selftest:
        return selftest(cfg)

    if args.list_rules:
        print(f"config: {_CONFIG_PATH}")
        print("forbidden glyphs:", ", ".join(f"{g['glyph']!r} ({g['name']})" for g in cfg.get("forbidden_glyphs", [])))
        print("forbidden patterns:", ", ".join(f"{p['pattern']!r}" for p in cfg.get("forbidden_patterns", [])) or "(none)")
        print("denylist tokens:", ", ".join(map(repr, cfg.get("denylist_tokens", []))) or "(none)")
        print("secret patterns:", ", ".join(label for label, _ in _SECRET_PATTERNS))
        print("stray-command prefixes:", ", ".join(cfg.get("command_line_prefixes", [])))
        return 0

    if not args.files:
        ap.error("no input: pass a file, or '-' to read stdin (see --help)")

    total_err = total_warn = 0
    for fname in args.files:
        if fname == "-":
            text, label = sys.stdin.read(), "<stdin>"
        else:
            p = Path(fname)
            if not p.is_file():
                print(f"✗ {fname}: not a file", file=sys.stderr)
                total_err += 1
                continue
            text, label = p.read_text(encoding="utf-8"), fname
        e, w = _report(label, lint_text(text, cfg))
        total_err += e
        total_warn += w

    print()
    fail = total_err > 0 or (args.strict and total_warn > 0)
    if fail:
        print(f"FAIL: {total_err} error(s), {total_warn} warning(s)"
              + (" (warnings fatal under --strict)" if args.strict and total_warn else ""))
        return 1
    print(f"OK: clean ({total_warn} warning(s))" if total_warn else "OK: clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
