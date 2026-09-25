# SUM next-session playbook

Updated 2026-09-25. The plan below comes from the 2026-09-23 adversarial review and roadmap, which the operator approved on 2026-09-25; finding IDs (H1, M1, ...) refer to that review. The 2026-09-09 repair queue further down still applies where the plan does not supersede it. The July queue is historical and is preserved in git.

## Plan from 2026-09-25: make trust true, test the wager by a date, ship the distiller either way

Operator-only actions are marked (operator). Claude prepares drafts and never contacts anyone.

### Phase 0: make trust true (to 2026-10-07)

Addressed in the 2026-09-25 change set: H1 (a supplied HMAC key now requires the HMAC signature); dependency floors reset to tested minimums, with floor jobs that fail on drift; the canonicalize 5.1.0 verifier bundle; dependabot #507, #516 and #517 merged, #511 to #515 superseded. Remaining, in order:

1. Release decision (operator) before the PyPI approval on run 35148438334 expires, about 2026-10-16. Package code at tag v0.11.0 equals main as of 2026-09-23, and H1 is also present in published 0.10.0. Approve with a known-issues note, or ship 0.11.1 with the H1 fix.
2. Key pin: the review-packet verifier checks an embedded key against the site JWKS and flags key-ID collisions. No packet is offered to anyone until this is live.
3. One fix PR, each fix with a test that fails on the old code: M3 (browser fails closed on a malformed key), M18 (explicit error instead of silent truncation above 100k characters), M4 (extractor fallback only on ImportError, with an egress notice), M17 (`inspect` field names). Then a Worker-only redeploy with a live-bytes check, targeted for 2026-09-30; it does not wait for 0.11.1.
4. M1: warn on the RFC 8032 test-vector key and rename the test key IDs `test-vector-zero-seed-DO-NOT-TRUST`. Operator key ceremony: offline key, two backups, a rotation date, revocation through TrustPolicy. The goldens stay byte-for-byte as test vectors and are re-minted once, after the scorer freeze.
5. M7, M6, M8: verdicts echo `statistical_scope`, `sampling_status`, delta and `simultaneous`; `controlled` appears only when replayed; `study --certify` moves behind the research flag with an in-sample warning; every "95%" on the BillSum golden is reworded as descriptive, in the README, fixture READMEs and paper sources together.
6. H2: clause guard at both sieve sites, a count of suppressed sentences, 20 or more conditional and chimeric cases in the T5 negative control, PROOF_BOUNDARY corrected. M19: rename the ZK prover and fix catalog entries #11 and #81. M23: the README says CI covers Python and Node only.
7. 0.11.1: one diff-scoped hostile audit of the exact revision, release, Worker redeploy, and the H1 reproduction run against the installed wheel.

Anything unfinished on 2026-10-07 goes on a dated, published known-issues list.

### Horizon 1: test the wager by hand (verdict Friday 2026-11-06)

- Pre-registration, committed here before any contact (operator approves): 15 named contacts, at least 10 in one setting (default: plain-language legal and consumer notices). Reach floor: 8 real conversations; below it, record INCONCLUSIVE-REACH and change channel. ALIVE: at least 3 dated second-party demands (2 in one setting) and at least 2 originators of second-party checks, one of them not a services client. FAILING: reach floor met, at most 1 demand, no checks. In between: one extension to 2026-12-04. Sensitive text: public or synthetic documents on a local-only page.
- Discovery (operator; Claude drafts): ask for the last dated time someone outside the team asked what an AI rewrite or translation changed. Drafts pass `scripts/lint_outbound_text.py`.
- Readout repair (M9 to M12): versioned unitizer with a `loss(x, x) == 0` test; recall and fidelity reported separately, empty output scored as loss 1; CHANGED / NOT CARRIED / ADDED using a prespecified contradiction threshold; claims beyond the judge window marked unchecked; regenerate `altitude_rungs.json`; freeze `scorer_version`. Then re-mint the BillSum and translation goldens once, from a pre-committed random draw, and publish the bound whatever it is.
- Workbench first-touch value, deterministic and in-browser with no network calls: flag changed or missing numbers, dates, negations, modals, conditions, exceptions and names, with a self-contained report. Pre-register the false-alarm and recall evaluation before results; if recall is below 50% or more than half of faithful rewrites are flagged, ship a neutral detail table instead.
- Benchmark preregistration before any test-split run: FRANK test split by error type (excluding calibration-card pairs) and the non-news LLM-AggreFact subsets; arms are SUM readout v2, MiniCheck-FT5, AlignScore-large and a lexical baseline; balanced accuracy and false-alarm rate with paired-bootstrap 95% intervals and n per type. The 42-case probe from 2026-09-23 is hypothesis-grade and is never cited as evidence.
- Paper: archive with a DOI after the re-mint; TMLR requires an anonymized manuscript. No further arXiv effort after 2026-11-06 unless an endorser appears.
- Thin distiller slice: depths built from verbatim spans of the user's own text, with no LLM.

Branches: ALIVE leads to a recipient packet view and a partner evaluation set; FAILING leads to a dated NORTH_STAR section 2 amendment, signing as an export button, and a zoom test with writers; INCONCLUSIVE-REACH re-runs on a new channel.

### Horizon 2 (2026-11 to 2027-03), in both branches

Claim map as `sum.review_packet.v2`: verbatim, clause-bounded source spans; labels signed as instrument outputs; human decisions kept separate. Minimum distiller: deterministic extractive depths under a distinct model ID (fixes M13), source highlighting, a "left out at this depth" panel; abstractive depths stay off until human-checked citation precision reaches 0.8 on 20 documents. Tripwire: if a stranger cannot read their own text at three or more depths with source links by 2027-01-31, other feature work freezes. Surface cut after 0.11.1; archiving `api/` or deleting the ZK prover needs explicit operator approval.

### Working rules

- The Friday number is cumulative originators of second-party checks: a real outside person reviews their own rewrite and sends the packet to a different person, who opens it. The ledger stays private; only counts are committed.
- Every task cites a finding, a person or a loop step. From 2026-10-07, at most one feature PR and one fix PR are open at a time.
- Until the verdict, stop: repo-wide multi-agent reviews (one diff-scoped audit per release instead), new receipt families, research imports, standards work without a partner, and "compliant", "certified", "faithful" or "guarantee" wording.
- Freeze: the Gödel/CanonicalBundle substrate, the sieve beyond H1 and H2, the compliance validators, the extension, Zig/WASM, the MCP tool count, and the size of `sum_cli/main.py`.
- `docs/` does not grow in net lines in a month. Three weeks without external contact, or more than two finished but unshipped artifacts, freezes Claude's feature work; fixes continue.

### Parking list (at most 10 lines; an item leaves only when a named user or funded deliverable asks)

- TypeSafe Jev judge (#518, #519): on hold; only as a disclosed baseline, after egress disclosure, abstention tests and a request-budget preflight.
- Review items not scheduled above: M5 legacy revocation string comparison; M15 and M16 Worker admission and KV fail-open (before any launch post); M24 catalog verify-line CI; 72 stale remote branches.

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
