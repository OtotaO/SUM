# SUM — Claude Code repo notes

Instructions that Claude Code inherits every session in this repo. Keep this
short. Repo-wide engineering conventions live in `docs/`; this file is for
rules a Claude Code session needs to know that are not otherwise obvious from
the code.

## Onboarding a memory-less session

If this is your first turn in this repo, read these files in order and you
will have the full picture. Each is under a contract to be honest (see
contracts below for PORTFOLIO.md specifically); stale claims are bugs.

1. **[`PORTFOLIO.md`](PORTFOLIO.md)** — current-state snapshot. What ships
   today, what proves it, what's next. Every metric row carries
   `**proved**` or `**empirical-benchmark**`.
2. **[`CHANGELOG.md`](CHANGELOG.md)** — release history. `[0.1.0]`
   was the first PyPI release (2026-04-22); `[0.2.0]` is the
   `internal/` → `sum_engine_internal/` hygiene rename (same date,
   later on). Anything after that lives under `[Unreleased]`.
3. **[`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md)** — proved-vs-
   measured discipline for every claim in the repo. Section 1.3.1 covers
   the cross-runtime Ed25519 trust triangle (Python ↔ Node ↔ Browser).
4. **[`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md)** — 103 numbered
   features, each with a reproducible verification command. Summary at
   the bottom gives the Production / Scaffolded / Designed counts.
5. **[`Makefile`](Makefile)** — every dev command canonicalised. `make help`
   renders the full list. Common ones: `make install`, `make test`,
   `make xruntime`, `make portfolio`, `make smoke`.

Shipping surface at the current HEAD: the `sum` binary on PyPI
(`pip install sum-engine[sieve]`), the Node verifier in
`standalone_verifier/`, and the browser demo in `single_file_demo/`.
All three verify Ed25519 on the same bundle bytes; the cross-runtime
harness (`make xruntime` → K1 / K1-mw / K2 / K3 / K4) proves this and
runs on every PR.

If you're about to make a change and want to know what's already deferred,
check the task list for items marked "deferred" (`internal/` →
`sum_engine_internal/` rename, Wikidata QIDs, AT Protocol Lexicon).

## PORTFOLIO.md contract

`PORTFOLIO.md` at the repo root is the body of `sumequities.com/projects/sum`.
The portfolio site pulls it via Astro 5 Live Collections on every push.
Frontmatter (title, hero image, priority) lives on the SUMequities side;
PORTFOLIO.md is body-only.

### When to update PORTFOLIO.md

- Any PR that changes a benchmark number, test count, completed horizon, or
  measured metric MUST also update PORTFOLIO.md in the same commit.
- Any PR that adds, deletes, or reframes a core capability (anything at the
  level of Core Capability 1–7 in the README) must touch PORTFOLIO.md.
- Tag pushes (`refs/tags/*`) trigger a rebuild regardless of PORTFOLIO.md.
  Use tags to mark milestone narratives.
- Pushes that do not touch PORTFOLIO.md refresh metrics only. Broken
  experiments pushed to main cannot corrupt the portfolio narrative.

### Prohibited in PORTFOLIO.md

- Marketing adjectives without substantiation ("blazing-fast",
  "production-ready", "revolutionary"). Numbers or nothing.
- Horizon claims without a measured artifact. A Horizon-III claim like
  "Zig core ships" requires a commit hash plus a benchmark number.
- Direct duplication of README.md. README is engineering-focused (how to
  run, how to extend); PORTFOLIO.md is consulting-prospect-focused (what
  ships today, what proves it, what's next).
- Inventing numbers. If `pytest --collect-only` reports 963, PORTFOLIO.md
  says "900+" (round down in public), not "963" and never "1000+".

### Required structure

1. One-line hook (what a consulting prospect sees in a link preview).
2. `## Current State` — verifiable claims only. Every metric line labelled
   `**proved**` or `**empirical-benchmark**` per `docs/PROOF_BOUNDARY.md`
   discipline. Numbers link to the script that produces them.
3. `## Future Directions` — ordered by leverage. Each item names concrete
   scaffolded code (file:line pointers) that exists today and what it
   becomes when wired.
4. `## Technical Stack` — what ships today. Not aspirational.

### Out of scope (do not cross-repo edit)

- Anything under `~/SUMequities` or `github.com/OtotaO/SUMequities`. That
  repo has its own Claude Code session. If PORTFOLIO.md needs frontmatter
  adjustments (different image, priority, alt text), raise it as an issue
  or PR comment — do not edit across repos.
- The webhook handler, Astro Live Collections loader, or CF Pages deploy
  hook. Those live in SUMequities.
- README.md rewrites motivated by PORTFOLIO.md edits. The two files serve
  different audiences and update on different cadences. They may disagree;
  that is fine.

### Enforcement

Two gates, one hard, one soft:

- **CI check (blocking):** [`scripts/check_portfolio_contract.py`](scripts/check_portfolio_contract.py)
  enforces the labelling rule on every row of the metric table under
  `## Current State` in `PORTFOLIO.md`. Any row containing a digit must
  also carry `**proved**` or `**empirical-benchmark**`. Wired into
  `.github/workflows/quantum-ci.yml` as the `portfolio-contract` job —
  blocks merge on violation.

- **Pre-commit hook (warn, not block):** [`scripts/hooks/pre-commit`](scripts/hooks/pre-commit)
  prints a reminder when files under `sum_cli/`, `internal/`,
  `scripts/bench/`, `Tests/`, `core-zig/`, `api/`, `single_file_demo/`,
  or `standalone_verifier/` change and `PORTFOLIO.md` does not. Human
  judgment decides whether the change is portfolio-relevant. Install
  once per clone with:

      bash scripts/install-hooks.sh

  Uninstall with `git config --unset core.hooksPath`. One-shot skip:
  `git commit --no-verify`.

The hard gate protects the portfolio narrative integrity on merge; the
soft gate nudges authors at commit time without ever blocking them.
