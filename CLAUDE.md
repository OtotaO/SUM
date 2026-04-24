# SUM — Claude Code repo notes

Instructions that Claude Code inherits every session in this repo. Keep this
short. Repo-wide engineering conventions live in `docs/`; this file is for
rules a Claude Code session needs to know that are not otherwise obvious from
the code.

## Onboarding a memory-less session

If this is your first turn in this repo, read these files in order and you
will have the full picture.

1. **[`CHANGELOG.md`](CHANGELOG.md)** — release history. `[0.1.0]`
   was the first PyPI release (2026-04-22); `[0.2.0]` /`[0.2.1]`
   are hygiene fixes; `[0.3.0]` (2026-04-23) added the agentic
   introspection surface (`sum ledger`, `sum inspect`, `sum schema`).
   Anything after that lives under `[Unreleased]`.
2. **[`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md)** — proved-vs-
   measured discipline for every claim in the repo. Section 1.3.1 covers
   the cross-runtime Ed25519 trust triangle (Python ↔ Node ↔ Browser).
3. **[`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md)** — 103 numbered
   features, each with a reproducible verification command. Summary at
   the bottom gives the Production / Scaffolded / Designed counts.
4. **[`Makefile`](Makefile)** — every dev command canonicalised. `make help`
   renders the full list. Common ones: `make install`, `make test`,
   `make xruntime`, `make smoke`.
5. **[`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md)** —
   ordered work queue (Priorities 1–8), principles you must internalise
   before editing claims, stop-the-line triggers. Read first if you are
   picking the thread up cold. The ordering is precedence, not
   preference: earlier priorities harden existing claims; later ones
   extend the surface.

Shipping surface at the current HEAD: the `sum` binary on PyPI
(`pip install sum-engine[sieve]`), the Node verifier in
`standalone_verifier/`, and the browser demo in `single_file_demo/`.
All three verify Ed25519 on the same bundle bytes; the cross-runtime
harness (`make xruntime` → K1 / K1-mw / K2 / K3 / K4) proves this and
runs on every PR.

If you're about to make a change and want to know what's already deferred,
check the task list for items marked "deferred" (Wikidata QIDs SPARQL
disambiguation, `sha256_128_v2` activation, browser-bench perf numbers).

## Out of scope — do not cross-repo edit

- Anything under `~/SUMequities` or `github.com/OtotaO/SUMequities`. That
  is a personal-portfolio repo with its own Claude Code session; SUM the
  engine ships independently and does not maintain portfolio-narrative
  artifacts. If the portfolio needs a description of SUM, it authors one
  in its own repo or pulls from `README.md` via its own loader.
