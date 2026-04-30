# SUM — Claude Code repo notes

Instructions that Claude Code inherits every session in this repo. Keep this
short. Repo-wide engineering conventions live in `docs/`; this file is for
rules a Claude Code session needs to know that are not otherwise obvious from
the code.

## Onboarding a memory-less session

If this is your first turn in this repo, read these files in order and you
will have the full picture.

0. **[`docs/SESSION_HANDOVER_2026-04-30.md`](docs/SESSION_HANDOVER_2026-04-30.md)**
   — most recent session-block handover (PRs #83–#95). Read this first
   if you are picking the thread up cold; it carries the open queue,
   user authority/preferences, and gotchas that don't appear in the code.
1. **[`CHANGELOG.md`](CHANGELOG.md)** — release history. `[0.1.0]`
   was the first PyPI release (2026-04-22); `[0.2.0]` /`[0.2.1]`
   are hygiene fixes; `[0.3.0]` (2026-04-23) added the agentic
   introspection surface (`sum ledger`, `sum inspect`, `sum schema`).
   Anything after that lives under `[Unreleased]`.
2. **[`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md)** — proved-vs-
   measured discipline for every claim in the repo. Section 1.3.1 covers
   the cross-runtime Ed25519 trust triangle (Python ↔ Node ↔ Browser).
3. **[`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md)** — **143** numbered
   features (current at 2026-04-30), each with a reproducible verification
   command. Summary at the bottom gives the Production / Scaffolded /
   Designed counts (currently 129 / 13 / 1). Counts are mechanically
   refreshed; treat them as authoritative over any prose in this file.
4. **[`Makefile`](Makefile)** — every dev command canonicalised. `make help`
   renders the full list. Common ones: `make install`, `make test`,
   `make xruntime`, `make smoke`.
5. **[`docs/NEXT_SESSION_PLAYBOOK.md`](docs/NEXT_SESSION_PLAYBOOK.md)** —
   ordered work queue (Priorities 1–8) plus post-hardening platform
   trajectory (Phases A–D), principles you must internalise before
   editing claims, stop-the-line triggers. Read first if you are
   picking the thread up cold. The ordering is precedence, not
   preference: earlier priorities harden existing claims; later ones
   extend the surface. Phases B and C depend on Phase A priorities
   being closed first — do not start Phase B work while Phase A
   priorities are still open.
6. **[`docs/SLIDER_CONTRACT.md`](docs/SLIDER_CONTRACT.md)** — slider
   product contract. Five axes, per-axis drift formulas, fact-
   preservation thresholds, the v0.4 → v0.7 NLI-audit / scale-bench /
   prompt-hardening arc. Canonical source for the headline numbers
   (median LLM-axis fact preservation = 1.000; p10 = 0.818 short-doc
   n=8, 0.769 long-doc n=16; min lifted from 0.111 → 0.700 by v0.7
   prompt hardening; catastrophic outliers eliminated 2 → 0).
7. **[`docs/SLIDER_V02_RESEARCH.md`](docs/SLIDER_V02_RESEARCH.md)** —
   research/methodology behind the slider's v0.2+ substrate. Itself
   stale relative to the current head (pre-v0.7) but useful as
   context: which choices the survey validated as load-bearing
   (verifiable rewards, cycle-consistency, content-addressed
   provenance, IB Pareto frontier), MontageLie threat model,
   constrained-decoding rationale, NLI audit positioning.
8. **[`docs/RENDER_RECEIPT_FORMAT.md`](docs/RENDER_RECEIPT_FORMAT.md)** —
   wire spec for the trust loop. `sum.render_receipt.v1`: Ed25519
   (RFC 8032) over JCS-canonical bytes (RFC 8785), wrapped as
   detached JWS (RFC 7515 §A.5) with public keys distributed via
   JWKS (RFC 7517). Defines payload field semantics, six-step
   verifier algorithm, cross-runtime canonicalisation rule (the
   integer-vs-float-zero gotcha), trust scope (what a verified
   receipt does and does NOT prove), key rotation cadence, C2PA
   `digital_source_type` alignment. Source-of-truth for every
   receipt-related claim PR A or any future doc-pass writes.
9. **[`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]`** — the full
   v0.4 → v0.9.A.2 arc since the last tagged release. v0.4 NLI audit
   verified the slider product claim; v0.5 Worker render path + slider
   UI; v0.6 long-doc scale verification (n=16); v0.7 prompt hardening
   eliminated catastrophic outliers; v0.8 four-layer defence against
   `LengthFinishReasonError`; v0.9.A render receipts (signed JWS +
   JWKS); v0.9.A.1 review-pass triple-sort + doc-bytes regen; v0.9.A.2
   route `/.well-known/*` through Worker + keygen polish.

Shipping surface at the current HEAD: the `sum` binary (currently
`v0.4.0` on `pyproject.toml`; **PyPI is at `0.3.0` and is stale** —
PRs #85–#95 have not been published; cutting `v0.4.0` to PyPI is on
the operator-decision queue), the Node verifier in
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
