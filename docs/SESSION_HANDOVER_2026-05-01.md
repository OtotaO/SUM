# Session handover — 2026-05-01

**Read this first** if you are picking up this thread cold. Pair this
note with the docs in [`docs/`](.) and the codebase at HEAD; nothing
here contradicts the docs, this is the thin layer of *context that does
not appear in the code itself* — decisions made, things deliberately
deferred, gotchas the future-you would otherwise re-discover.

This supersedes [`docs/SESSION_HANDOVER_2026-04-30.md`](SESSION_HANDOVER_2026-04-30.md)
as the canonical entry point for picking up cold; the prior handover
remains useful for the v0.3 → v0.4 substrate arc but its open queue is
closed.

---

## What just shipped (in chronological merge order)

Six PRs landed this session-block plus the **v0.4.1 PyPI publish**.
Counts at HEAD `281f624`.

| #   | Subject                                                                                | Why it matters |
|-----|----------------------------------------------------------------------------------------|----------------|
| 97  | `sum render` CLI verb (bundle → tome under 5-axis slider control)                     | Closes "tags ↔ tomes" symmetry from the shell. Local path is deterministic (density only); `--use-worker URL` returns LLM-conditioned tome + signed `render_receipt`. The reverse direction was reachable from Python and HTTP before; now from a shell prompt. The README's bidirectional pitch lines up with the CLI surface for the first time. |
| 98  | `release: v0.4.0` rotation                                                            | Rotated `[Unreleased]` → `[0.4.0] — 2026-04-30`. Tag `v0.4.0` was pushed; publish workflow built and staged to TestPyPI cleanly, **fail-closed at the pre-promotion verify gate** with `no Sigstore certificates extractable from provenance`. **Production PyPI was correctly never touched.** v0.4.0 stays as a forever-untagged-on-PyPI git tag. |
| 99  | `verify_pypi_attestation: recognize flattened ``certificate``-string shape`           | Verifier-side script bug. PyPI's Integrity API now serialises the leaf certificate at `verification_material.certificate` directly as a base64 string, not under a `rawBytes` envelope. Walker only matched older shape ⇒ zero candidates ⇒ fail. Widened collector to also accept any `certificate` key; downstream `parse_certificates` already tolerated non-DER candidates by skipping, so widening cannot turn a tamper signal into an accept. Verified end-to-end against live TestPyPI provenance. +2 regression tests. |
| 100 | `publish-pypi: workflow_dispatch trigger`                                              | Added `workflow_dispatch:` so an operator can re-run an existing tag's publish flow without re-tagging. No workflow inputs; trigger contract identical to `push`. Has a documented caveat: re-running on the same tag requires deleting the existing TestPyPI release first (rebuild produces different bytes due to wheel non-determinism; `skip-existing: true` would silently skip and the same-bytes verify would then fail). |
| 101 | `release: v0.4.1 — verifier hygiene fix (Path A)`                                      | **Patch release. Wheel content byte-identical to never-published v0.4.0** (`scripts/` and `.github/` are excluded from dist). v0.4.0 stays as the honest "tagged but never published" git record; v0.4.1 is what users install. Documents the verifier fix, the workflow_dispatch trigger, and the "why no v0.4.0 on PyPI" note for users who check pypi.org and see no 0.4.0. |
| 102 | `docs: post-v0.4.1 polish`                                                            | Drops the "PyPI 0.3.0 stale" wording from README + CLAUDE.md now that 0.4.1 is live. README's "What ships today" row lists `pip install 'sum-engine[sieve]'` as `shipped on PyPI ≥ 0.4.1`. |

**v0.4.1 is live on PyPI.** Verified by `pip install sum-engine==0.4.1`
in a fresh venv → `sum --version` returns `0.4.1`. All five publish
jobs landed green including the post-publish production-PyPI verify
(`verify-pypi`), which exercised the #99 verifier fix end-to-end against
the live production attestation.

Counts at HEAD: **143 features in FEATURE_CATALOG (129 production, 13
scaffolded, 1 designed).** Manifest + self-attestation + repo manifest
all current; CI drift gates green; cross-runtime K-matrix + A-matrix
locked.

---

## Open queue (priority-ordered)

The 2026-04-30 audit's three concrete issues are **all closed.** What
remains is what was already deferred.

### Deferred (cross-repo): SUMequities portfolio fetch of `meta/repo_manifest.json`

**Status:** waits on a separate-session SUMequities Claude Code run.

`meta/repo_manifest.json` ships at a stable raw URL via PR #84. The
SUMequities portfolio at https://www.sumequities.com/projects/sum/
doesn't fetch it yet. CLAUDE.md says "do not cross-repo edit"; that
boundary still holds. When the user is ready, they spin up a
SUMequities session and have it fetch this manifest URL on a build
hook. Out of scope for this repo's session.

### Deferred (YAGNI): audio + image-OCR adapters

**Status:** unchanged from 2026-04-30 audit; deferred indefinitely.

No real user input has demanded audio or scanned-PDF attestation.
`faster-whisper` + `pytesseract` total ~700 MB of dep weight + require
a system tesseract binary; the cloud route violates the receipt-bearing
determinism contract. The omni-format adapter (PR #88) handles every
text-bearing format users have actually tried. Revisit when a real
user input demands it.

### Active research direction (specified, not yet implemented)

- **Sheaf-Laplacian hallucination detector** —
  [`docs/SHEAF_HALLUCINATION_DETECTOR.md`](SHEAF_HALLUCINATION_DETECTOR.md)
  is the spec, written 2026-05-01 after a session pass over the
  SCT (Stratified Cognitive Topos) synthesis the user shared.
  Grounded in Gebhart, Hansen & Schrater (2023, AISTATS,
  arXiv:2110.03789) "Knowledge Sheaves" — the
  sheaf-Laplacian quadratic form ``x^T L_F x`` is the
  mathematical primitive for measuring how badly a render
  manifold fails to glue. v1 (1-dim presence stalks), v2
  (text-embedding stalks), v3 (receipt-weighted) procedures
  specified; v1+v2 is generic Knowledge-Sheaves application;
  v3 is the SUM-specific extension that does not replicate
  elsewhere. Three-week scope to a published artifact;
  the wedges (agent-trust, C2PA-text, compliance) all
  benefit from the same artifact under different
  cover-computers. **No code shipped yet** — spec only.
  Read the spec doc before adding parallel research
  directions; SCT-flavoured framings should compose with
  this thread, not branch.

### Soft follow-ups (small, not load-bearing)

- ~~**`sum-mcp` tool surface gains `render`.**~~ ✓ Shipped in
  PR #104 — MCP server now exposes `extract / attest / verify /
  inspect / **render** / schema`. Tool count 5 → 6. The
  bidirectional 3×3 grid (CLI/MCP/HTTP × attest/verify/render) is
  fully populated. FEATURE_CATALOG entry 144.

- **`sum render --receipt-out PATH`.** Receipts arrive in the `--json`
  envelope when `--use-worker` is set; a separate file path would be
  more ergonomic for daily use. Cosmetic.

- **`sha256_128_v2` default-promotion.** Cross-runtime byte-identity
  is locked; flipping the default is a `bundle_version` minor bump per
  `docs/COMPATIBILITY_POLICY.md`. Operator decision; not blocked on
  any code.

- **§2.5 closure has a 2-doc residual on Opus 4.7** — traceable to
  upstream LLM source-extraction artifacts, not the intervention
  pattern. Documented in PROOF_BOUNDARY §2.5; honest open gap.

---

## Decisions that aren't obvious from the code

These are the load-bearing judgment calls made this session-block.
Future-you may not infer them from the diff alone.

### v0.4.0 → v0.4.1: Path A over Path B′

When the v0.4.0 publish failed at the pre-promotion verify gate (see
#99), two paths existed to actually publish v0.4.0's contents:

- **B′ (force-push v0.4.0 tag to current HEAD)** so the dispatched
  ref carries the workflow_dispatch trigger added in #100. Works
  technically, but **destructive**: rewrites tag history on a shared
  remote, against tag-immutability convention, and corrupts what
  v0.4.0 means in git history (would point at a commit containing
  unrelated commits). Anyone who already fetched v0.4.0 sees their
  tag diverge from origin.

- **A (cut v0.4.1 with byte-identical wheel content)**. Non-destructive;
  v0.4.0 stays as the honest "tagged but never published due to
  verifier shape drift" git record; v0.4.1 ships the same wheel
  content + the verifier fix in CI.

Path A was chosen. The "more elegant" framing originally favoured B
(workflow_dispatch on existing tag) when we believed it would just
work; once we saw it required a force-push, the elegance flipped — A
is now the lower-cost path. **The git rule "never force-push tags on
shared remotes" is not negotiable for elegance.** Document this when
the topic recurs.

### Fail-closed gates are working as designed; do not weaken them

The verify-attestation gate fired on v0.4.0 because of a verifier-side
shape drift, not an artifact-tamper signal. The temptation in the moment
is to "just bypass once" or to print a warning and continue. Don't.
Production PyPI publishes are irreversible; a fail-closed gate that
sometimes lets unverified bytes through is worse than one that
occasionally needs a verifier patch.

The fix in #99 was deliberately additive: the collector recognises one
more cert location, but the X.509 parse + SAN identity + repository +
workflow checks are all unchanged. Widening the collector cannot turn
a tamper signal into an accept (the parser tolerates non-DER candidates
by skipping them). Future verifier-shape evolution should follow this
same pattern: extend the inputs the collector recognises, never relax
the cryptographic checks downstream.

### Production-publish authority always asks

Per CLAUDE.md and saved memory, tag pushes / production publishes need
**explicit per-action user authorization**. Standing PR-merge authority
covers reversible code changes only. The publish workflow's deployment-
protection rule on the `pypi` environment (manual reviewer = OtotaO)
is the gate of last resort. Even when `current_user_can_approve: true`
is true via the API, *I do not click approve* — that defeats the
human-in-the-loop purpose the user explicitly configured.

---

## Gotchas the future-you should not re-learn

1. **`workflow_dispatch` reads from the workflow file at the
   dispatched ref, not from the default branch.** A trigger added on
   `main` does NOT make `gh workflow run --ref vX.Y.Z` work for a tag
   that predates the trigger. This is what bit us 2026-05-01 trying
   to re-run on v0.4.0 after #100 merged. Add the trigger BEFORE
   tagging, or accept Path A.

2. **TestPyPI release deletion is web-UI-only for non-staff
   maintainers.** No `twine` subcommand, no warehouse JSON endpoint, no
   `gh` shortcut. Project owner clicks "Delete release" on
   `https://test.pypi.org/manage/project/<name>/release/<version>/`.
   ~10 seconds; just has to be a human in a browser.

3. **Wheel + sdist builds are not deterministic by default.** Python
   wheel builds embed timestamps unless `SOURCE_DATE_EPOCH` is set.
   `skip-existing: true` on TestPyPI uploads compares filenames, not
   bytes — a re-run will silently skip the upload and then the
   downstream same-bytes verify will fail comparing new local hashes
   to stale TestPyPI bytes. The publish workflow's header comment now
   documents this.

4. **`scripts/` and `.github/` are excluded from the wheel.** Per
   `[tool.setuptools.packages.find].exclude`. Means CI tooling
   changes do not produce a different wheel and any patch release
   driven by CI-only edits has byte-identical user-visible content.
   This is what made Path A clean (v0.4.1 wheel ≡ never-published
   v0.4.0 wheel).

5. **The PyPI Integrity API JSON shape is not stable across time.**
   PR #99 was triggered by exactly this kind of evolution
   (`verification_material.certificate` flattened from sub-object to
   bare string). Future shape drifts will need similar additive
   widenings of the collector. The `pypi-attestations` external CLI
   already verified the cert chain cryptographically; our script's
   role is the SAN-identity check on the parsed cert. As long as
   `pypi-attestations rc=0` passes and SAN identity matches the
   expected workflow + ref, the artifact is real.

6. **The `pypi` environment has a deployment-protection rule.**
   Production publishes pause at `Publish to PyPI (production)` with
   status `waiting` until a reviewer approves. The reviewer is the
   user; I do not click approve myself even when API permissions
   technically allow it.

7. **`gh run view --json conclusion` returns empty string for
   in-progress runs.** Monitor scripts that don't account for that
   will exit early or report `unexpected:`. Use the case `*) sleep`
   pattern, not `case in_progress)`.

8. **Stagnant CLAUDE.md count claims are easy to miss.** Every PR
   that changes FEATURE_CATALOG.md should also update the counts in
   CLAUDE.md item 3 ("**N** numbered features ... currently P / S /
   D"). The catalog file's own summary section is mechanically
   grep'able; CLAUDE.md is hand-maintained. The two get checked
   against each other by the manifest drift gate (which reads from
   the catalog), so a CLAUDE.md prose drift will surface as a
   "manifest current but doc out of sync" reading on the next
   manual audit.

---

## What "polished" looks like at the close of this session

| Surface | Polish state at HEAD |
|---|---|
| Cold-install onboarding | ✅ Works in 60s end-to-end |
| README first 60 lines | ✅ Thesis + live demo URL + working curl example all in lead |
| Cross-runtime trust triangle | ✅ K1–K4 + A1–A6 green on every PR |
| §2.5 LLM closure | ✅ Vendor-independent across 3 model families |
| Self-attestation | ✅ All 5 canonical docs round-trip via `sum verify` |
| Omni-format text formats | ✅ HTML/PDF/DOCX/EPUB/JSON/IPYNB/RTF/XML |
| `attest-batch` | ✅ JSONL + MinHash dedup |
| MCP server | ✅ stdio, hardened, 29 tests; `render` tool TBD |
| Sieve quality | ✅ Rounds 1+2 noise filter; ~21% noise reduction on README |
| `sum render` CLI verb | ✅ Shipped (PR #97) |
| **PyPI release** | ✅ **0.4.1 live; verified end-to-end** |
| Verifier hygiene | ✅ Recognises current PyPI Integrity shape (PR #99) |
| Re-publish ergonomics | ✅ workflow_dispatch wired (PR #100) |
| Audio / OCR adapters | ⏸ Deferred until real user demand |
| Portfolio integration | ⏸ Cross-repo, waits on operator |

---

## Files to read in order on your first turn

1. This file ([`docs/SESSION_HANDOVER_2026-05-01.md`](SESSION_HANDOVER_2026-05-01.md))
2. [`CHANGELOG.md`](../CHANGELOG.md) — `[0.4.1]` is the most recent shipped release; `[Unreleased]` is empty until next work lands
3. [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — proved/measured/designed discipline; non-negotiable
4. [`docs/FEATURE_CATALOG.md`](FEATURE_CATALOG.md) — entry 143 is the most recent shipped surface (`sum render`)
5. [`meta/self_attestation.summary.json`](../meta/self_attestation.summary.json) + [`meta/repo_manifest.json`](../meta/repo_manifest.json) — current substantive state, machine-readable
6. [`docs/SESSION_HANDOVER_2026-04-30.md`](SESSION_HANDOVER_2026-04-30.md) — prior session-block; useful for the v0.3 → v0.4 substrate arc
7. [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) — long-term priority context (historical; superseded by these handovers for the active queue)

If you only have time for one: read this file. The rest are the
substrate.

— end of handover
