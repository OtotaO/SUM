# SUM next-session playbook

Updated 2026-09-09. This is the current queue after the comprehensive audit and the operator's instruction to implement needed repairs, including merging. The July queue is historical and is preserved in git; its expired dates, already-completed MCP work and old release numbers are not current blockers.

## Start from live evidence

Read NORTH_STAR, then CLAUDE.md, PRODUCT_VISION and the current CHANGELOG. Confirm the branch, open PRs, CI results, published package version and last successful deployment. A merged change, a deployed change, and a published package are separate facts. Do not copy a historic zero-adopter count into a current status without current evidence.

Fetch and fast-forward the base before branching (`git pull --ff-only origin main` is explicit when multiple worktrees fetch concurrently). Stage explicit paths. Regenerate self-attestation when editing its canonical documents, and the repo manifest when changing stable inputs. Lint workflow SHA pins and new outbound text. Independently review consequential changes and squash-merge only after the current head's checks pass. Existing session authorization to implement or merge persists; do not manufacture a new approval requirement from this playbook.

Keep signing secrets out of source and messages. Preserve historical signed receipts byte-for-byte; publish corrections in active documentation and new receipt metadata. Do not contact external people, submit papers as an author, or apply for grants without the relevant explicit authorization and actual human-supplied declarations.

## Finish the audited repair and release gates

Check the current implementation and tests for each item; the CHANGELOG records what has actually merged.

- Worker: classify by the selected serving credential; atomically admit paid work; fail closed when admission is unavailable; cap bytes, triples, models, time and active requests. Keep canonical routes usable without paid infrastructure.
- Agent integrity: immutable typed bind snapshots, oversized-value rejection before eviction, explicit chunk limits, no implicit model download from MCP.
- Trust: caller-supplied key pins, offline revocation snapshots and freshness/archival policy across families; clearly separate unchecked source identity and organization identity.
- Measurement: consistent empty-source scoring; canonical instrument/evaluation manifests; descriptive defaults unless sampling assumptions are asserted; visible truncation limits; no invariance claim from overlapping confidence intervals.
- Browser: retain the full source, apply density once, compare existing rewrites, verify displayed-content bindings, invalidate stale results, and export exact reviewed packets. Keep literal review separate from model measurement.
- Secondary surfaces: one maintained selection-capture client; no fake TUI zero score or Run action; no production claim for an unvalidated internal Docker recipe.

Run focused regressions plus the repository's trust and packaging gates. Run the new Worker suites and browser packet tests in CI. Add new runtime assets to the deployment byte guard. Check a built wheel outside the checkout with `[verify]` and the declared install extras; verify the historical golden and the new policy API there.

After green CI and merge, use the existing Cloudflare deployment workflow on the merged revision, then verify frontend bytes and a canonical signed render against its output hash and public keys. Confirm the SQLite Durable Object migration succeeds before declaring paid admission deployed. No paid generation request is needed for the canonical trust-loop check. Inspect a deployment failure before retrying; do not change workflow permissions or triggers to evade missing access.

For a package release, bump the version, update CHANGELOG, regenerate manifest/attestation, run release-byte checks and independently review the exact revision before tagging. Preserve the existing TestPyPI provenance gate and production post-publish verification. If dispatch, tag creation or an environment approval is unavailable, report the concrete blocked action and leave tested source ready; never describe prepared bytes as published.

## Evaluate and improve the real distiller

The next research evidence must concern actual source-to-SUM-output behavior. Dataset reference summaries are useful fixture material and must remain labelled as such. Retain ordered source/output hashes, extraction/generation/judge configuration, selections, per-example losses, human decisions, and an independently reviewed sampling plan. Separate within-corpus descriptive results from claims about future documents.

Prioritize qualifiers, negation, numbers, dates, attribution, long inputs and unsupported additions. For NLI, compare full source evidence through bounded support windows and expose uninspected spans; a truncation disclosure is not a substitute for that future capability. Prespecify practical error tolerances and paired tests before describing composition as stable. Evaluate the high-scoring embedding cases too; low-score rescue alone cannot measure high-confidence misses.

Then add a few useful depth candidates generated from the retained source and reviewed through the same packet path. Do not portray a sorted list as a continuous measured frontier. Restore detail from retained evidence; mark new generation when evidence is insufficient.

## Learn from actual use

Observe users completing a source review and an independent recipient checking the export. Record task, useful changes found, missed changes, effort, repeat use and concrete consequences. Compare with ordinary side-by-side editing and existing evaluation tools. Users, retention, buyers, pricing and grants remain external hypotheses until measured; simulations and green CI cannot close them.

Choose subsequent format support, team features or integrations from those observations. Avoid a large CLI rewrite, paid hosted judging platform, billing build, automated outreach or new research surface solely because the infrastructure is possible. Update this queue in place when evidence changes; do not create another handover document.
