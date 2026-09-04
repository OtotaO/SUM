# SUM - repo notes for coding agents

This file exists so that an agent which reads `AGENTS.md` by convention
(Codex and others) does not start from stale instructions.

## Read `CLAUDE.md` instead

**[`CLAUDE.md`](CLAUDE.md) is the single maintained source of onboarding
instructions for this repo.** Read it now, in full, before doing anything
else. Everything that used to be duplicated here lives there: the operational
compass, the reading order, the shipping surface, the bench-hardening status,
and the out-of-scope rules.

This file is deliberately a pointer and not a copy. A parallel copy drifts:
the previous version of this file had accumulated 344 lines and linked nine
session-handover documents that no longer exist, which is the documentation
sprawl that `docs/NORTH_STAR.md` section 4f names as an anti-pattern. One
file is maintained; this one redirects to it.

## The three rules most often violated by a fresh session

These are in `CLAUDE.md` too, but they are cheap to repeat and expensive to
learn the hard way.

1. **Never `git add -A`.** Stage explicit paths. This repo has leaked private
   outreach drafts into a public commit exactly once, and that is how.
2. **`git fetch && git pull --ff-only` immediately before `git checkout -b`.**
   `meta/*` is refreshed on every PR, so skipping this guarantees a rebase
   conflict.
3. **Commit messages and PR bodies are outbound text.** No em dashes. Run
   `python scripts/lint_outbound_text.py <file>` before posting; it is not
   enforced by CI, so nothing else will catch it.

Editing `README.md`, `CHANGELOG.md`, `docs/PROOF_BOUNDARY.md`,
`docs/FEATURE_CATALOG.md`, `docs/RENDER_RECEIPT_FORMAT.md`, or
`docs/TRANSFORM_RECEIPT_FORMAT.md` additionally requires running
`python -m scripts.attest_repo_docs` and committing `meta/self_attestation.*`
in the same commit, or CI reds on the drift gate.

## Where the current state lives

`CHANGELOG.md` is the in-repo narrative. `docs/NEXT_SESSION_PLAYBOOK.md` is
the ordered work queue. Per-session handover documents are deliberately not
minted any more; the nine that used to be linked from this file were retired
on 2026-09-02 and are preserved at the `archive-2026-09-02` tag.
