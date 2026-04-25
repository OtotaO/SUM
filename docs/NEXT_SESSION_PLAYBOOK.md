# Next-Session Playbook

*Authored 2026-04-24 at the end of the session that landed v0.3.0, the
Cloudflare Worker (Pages→Workers migration), the WASM browser core, the
`/api/qid` Wikidata resolver, and the portfolio-coupling revert
(`63a7f5d`). Read this before touching code in a new Claude Code session.*

## Non-negotiable principles

Before any work: read [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md)
cover-to-cover and internalise the proved / measured / designed
distinction. Every claim emitted from this codebase — in docstrings,
README, commit messages, blog posts, tweets — carries one of those
three labels implicitly. Making the label explicit is discipline, not
paranoia. A "proved" claim survives hostile review; a "measured" claim
survives replication; a "designed" claim survives only in the author's
head until it earns a label upgrade.

**Measure before you assert performance.** Never write "fast,"
"efficient," "low-latency," or "scalable" in a commit message or README
without a benchmark in the same commit. If you don't have time to
benchmark, you don't have time to make the claim.

**Adversarial testing is first-class work, not a nice-to-have.** A
verifier that passes K1–K4 on well-formed bundles has been proved to
agree on validity; it has NOT been proved to agree on invalidity.
Those are two different proofs and only the first is done.

**Truthfulness over velocity.** If the honest answer to "does this
work?" is "on the happy path, yes; under adversarial input, unknown" —
write it that way. A repo that overclaims once loses the trust it took
eight months to build.

## Priority 1 — Fuzz the parser + verifier triple

**Problem.** `make xruntime` proves Python, Node, and Browser agree on
every bundle in the test corpus. That's an agreement proof on a curated
set of **valid** inputs. Disagreement on malformed / adversarial input
has not been proved. If the Python verifier rejects a crafted bundle
while the Node verifier accepts it, that's a protocol fork in disguise.

**Work.**
- Build an adversarial bundle corpus: malformed canonical tomes
  (missing headers, truncated, invalid UTF-8, injected comment lines,
  reordered sections), boundary-condition state integers (zero,
  negative as string, >10000 chars), signature mismatches, scheme
  downgrade attempts, nested JSON payload bombs.
- Run each adversarial bundle through all three verifiers. For each:
  does it reject? With what exit code / error? Are the three verifiers'
  rejection behaviours equivalent?
- Formalise the result as `make xruntime-adversarial` alongside the
  existing `make xruntime`, emitting the same K-style pass/fail matrix.

**Success criterion.** A matrix where every adversarial bundle is
rejected by all three verifiers with equivalent classifications
(structural / signature / scheme / version), OR — if any verifier
behaves differently — the discrepancy is documented, reproduced, and
either fixed or frozen into `docs/PROOF_BOUNDARY.md` as a known
asymmetry.

**Proof-boundary outcome.** Moves "the three verifiers agree" from
"proved on valid inputs" to "proved on adversarial inputs." Strongest
single thing you can do for robust-against-adversity.

## Priority 2 — Benchmark the WASM path, publish honest numbers

**Problem.** Phase 3b shipped the Zig → WASM core. No performance
claim has been made, which is correct discipline until a number exists.
But the implicit expectation of a WASM path is "faster than JS." That
expectation is not yet either proved or refuted.

**Work.**
- Build `Tests/benchmarks/browser_wasm_bench.html` that runs
  N ∈ {10, 100, 1000, 10000} axiom derivations via (a) the WASM path
  and (b) the JS path, records wall-clock per derivation + total,
  reports in a fixed format.
- Run in Chrome 113+, Firefox 129+, Safari 17+ (the supported matrix
  from the verifier). Different engines optimise WASM differently; the
  claim has to survive all three.
- Also run the Python CLI on the same input sizes for the cross-runtime
  comparison.
- Publish the numbers with methodology in `docs/WASM_PERFORMANCE.md`
  (new file). Label every number `measured`. Include a fallback
  statement: "if the WASM path is slower in your environment, here's
  how to fall back to JS" — regardless of whether measured.

**Success criterion.** Three measured data points per engine,
reproducible with a single make target, committed with the benchmark
code. No performance claims in README.md until this file exists.

**Proof-boundary outcome.** WASM path moves from "shipped but
unclaimed" to "measured." If the numbers are good, great. If they're
not, that's still truthful and informs the next decision.

## Priority 3 — Activate `sha256_128_v2`

**Problem.** `sha256_64_v1` is the default prime scheme. 64-bit primes
have a birthday-paradox collision probability that becomes non-trivial
at around 2^32 axioms per branch. Below that, it's fine. A large-scale
attestation user — say, someone attesting every fact in a Wikipedia
dump — is above that. The v2 scheme is "designed" per
[`docs/STAGE3_128BIT_DESIGN.md`](STAGE3_128BIT_DESIGN.md); it's not
activated.

**Work.**
- Land the 128-bit prime derivation in all three verifiers, covered by
  unit tests that prove Python / Node / Browser agree on the same
  128-bit primes for the same axiom keys.
- Extend the cross-runtime harness: K1 currently tests `sha256_64_v1`;
  add K1-v2 for `sha256_128_v2`. All bundles in the corpus should have
  a v2 counterpart.
- Keep `sha256_64_v1` as the default for 0.x (backward compat). Add a
  `--prime-scheme=sha256_128_v2` flag to `sum attest`. Document the
  migration path.
- CHANGELOG entry labels the transition: v1 is "proved" (as before),
  v2 becomes "proved" when the K-matrix passes.

**Success criterion.** K1-v2, K2-v2, K3-v2, K4-v2 all PASS. Two users
can attest the same corpus under v2 independently and get the same
state integer. Documentation warns that mixing schemes in a single
branch is undefined.

**Proof-boundary outcome.** v2 moves from "designed" to "proved."
Pre-empts a collision issue that grows worse with SUM's success.

## Priority 4 — SPARQL disambiguation for `/api/qid`

**Problem.** The current `/api/qid` endpoint hits `wbsearchentities`
alone. Commit `1f060c6` says ~80% accuracy on common-noun /
proper-name lookups. 80% accuracy on a lookup that's framed as "this
QID is this entity" is a truth claim with a 1-in-5 lie rate. That's
not acceptable for an attestation surface that's supposed to be
ground truth.

**Work.**
- When `wbsearchentities` returns multiple candidates with close
  confidence, fire a SPARQL query against `query.wikidata.org` that
  filters by predicate domain. For a subject with predicate "was born
  in" → filter candidates to those with the `P19` (place of birth)
  property present.
- If only one candidate passes the SPARQL filter, return it with
  `confidence: "sparql-disambiguated"`.
- If multiple still pass, return all with `confidence: "ambiguous"`
  and let the caller decide. Never pick one silently.
- Per the Wikidata operator-contact guidance, keep the User-Agent
  header with a contact URL. Rate-limit SPARQL to avoid hostile
  behaviour.

**Success criterion.** A held-out benchmark set (50+ known-
disambiguation cases, curated) where the pre-change resolver gets ~80%
right and the post-change resolver gets >95%. Wikipedia's
disambiguation-page list is a good oracle source for the benchmark.

**Proof-boundary outcome.** `/api/qid` moves from "best-effort entity
resolution" to "entity resolution with measured accuracy floor." The
confidence field becomes meaningful rather than heuristic.

## Priority 5 — Threat-model validation

**Problem.** [`docs/THREAT_MODEL.md`](THREAT_MODEL.md) exists. Every
documented threat is a claim of the form "SUM defends against X." If
the threat is documented but not tested, that defence is designed, not
proved.

**Work.**
- Read `docs/THREAT_MODEL.md` end-to-end. For each documented threat,
  write a test that embodies the threat and confirms the defence.
  File under `Tests/adversarial/test_threat_<n>.py` or equivalent.
- Threats that turn out to be undefended become either (a) new work
  items or (b) `docs/PROOF_BOUNDARY.md` entries labelled "known
  limitation."
- Threats not listed that should be — prompt injection in extracted
  axioms, replay across DIDs, partial-bundle truncation attacks,
  clock-skew signature validation — add them, test them, document
  them.

**Success criterion.** Every threat in the document has a
corresponding test. Every test in `Tests/adversarial/` maps to a
threat in the document. The two lists are kept in sync by a CI check.

**Proof-boundary outcome.** The threat model moves from prose to
prose + executable assertions. Claims are no longer just asserted;
they're exercised.

## Priority 6 — Delta-bundle semantics

**Problem.** `bundle.is_delta` exists in the schema
(`sum_cli/main.py`) and the verifier checks for it
(`standalone_verifier/verify.js`). What a verifier is supposed to DO
with a delta bundle — compose against a prior state, emit a combined
state integer, reject if the prior is missing — is under-specified.
Under-specified means different implementations will diverge.

**Work.**
- Write `docs/DELTA_SEMANTICS.md` specifying: what a delta bundle
  references (prior state integer? prior CID? prior bundle's
  signature?), how composition works mathematically (LCM of primes, as
  with full bundles, but with an ordering constraint?), when a delta
  is rejected (missing prior, scheme mismatch, non-monotone
  timestamp).
- Add K5 to the cross-runtime harness: delta-bundle composition. All
  three verifiers compose the same pair of (full-bundle, delta-bundle)
  to the same final state integer.
- Update the schema's `isDelta` description to point at the spec doc.

**Success criterion.** A composition test that produces
`combined_state_integer`, and all three verifiers compute the same
`combined_state_integer` from the same inputs. Doc says what happens
on every failure case.

**Proof-boundary outcome.** Delta bundles move from "supported in
schema but semantic undefined" to "proved composable across runtimes."
Unlocks append-only knowledge-graph use cases without fear of runtime
fork.

## Priority 7 — Supply-chain attestation

**Problem.** The PyPI wheel is OIDC-published — that's good. The
Cloudflare Worker is deployed via `wrangler deploy`, which trusts the
wrangler lockfile and the CF API token; the deployed JS is not
independently attested. The single-file demo is
`single_file_demo/index.html` committed to the repo, served via
`raw.githubusercontent.com`, with no signature on the served bytes.

**Work.**
- Sign release artifacts with Sigstore / cosign. The wheel + the
  single-file demo + the standalone verifier's Node entrypoint all get
  signed. Public key published in the repo and on the project site.
- A verifier wanting end-to-end trust can now check: the bytes I'm
  running came from a signed release, signed by a key traceable to the
  project's identity.
- Consider adding a SLSA provenance attestation to GitHub Actions for
  each release tag.
- Document the trust chain in `docs/SUPPLY_CHAIN.md`: "here's what we
  sign, here's how to verify, here's what we DON'T sign and why."

**Success criterion.** `pip install sum-engine` → verify signature →
run. `node standalone_verifier/verify.js` → verify signature of
`verify.js` itself before using it. All three runtimes have a
verifiable bytes-to-identity path.

**Proof-boundary outcome.** The trust model moves from "trust the
GitHub repo" to "verify the bytes." The first is reasonable; the
second is robust.

## Priority 8 — LLM-extraction honesty guardrails

**Problem.** `sum attest --extractor=llm` lets a user extract axioms
via an LLM. LLMs can hallucinate. The resulting bundle says "here are
the axioms, attested and signed" — but "attested and signed" applies
only to the computation (the state integer is correctly derived from
these axiom keys), not to the truth of the axioms themselves. A reader
might reasonably infer that an Ed25519-signed bundle with high
confidence means "these facts are true." That's not what it means.

**Work.**
- Distinguish clearly in the bundle schema between "extraction was
  deterministic" (sieve) and "extraction was best-effort" (LLM). A
  field like `extraction.verifiable: true | false` that makes the
  epistemic status of the axioms visible to downstream consumers.
- In `sum verify`, when verifying an LLM-extracted bundle, emit a
  prominent warning: "axioms were extracted by an LLM and are not
  independently verified." Don't hide this in a debug log.
- In the schema emitted by `sum schema bundle`, add documentation that
  signed ≠ true.
- Consider: can provenance spans (byte-level source-text references)
  constitute "light-touch verification" for LLM extractions? If the
  LLM said "Alice likes cats" and the source span literally says
  "Alice likes cats", the extraction is confirmed; if the source span
  says something else, flag it.

**Success criterion.** A consumer reading a signed bundle cannot
confuse "signed by whoever held the key" with "facts are true." The
bundle's epistemic status is a first-class, machine-readable field.

**Proof-boundary outcome.** The truth claim SUM makes — which has
always been "we bind the set of axioms, not their truth" — becomes
visible to downstream consumers rather than buried in docs.

## Ordering rationale

1–2 harden existing claims. Do them first. The worst outcome right now
is someone crafting a malformed bundle that the Python verifier rejects
but the Browser verifier accepts — it would contradict the public
"cross-runtime trust triangle" claim. Priority 1 closes that gap.
Priority 2 replaces an implicit performance expectation with a measured
number before anyone quotes SUM as "the fast verifier."

3–4 extend existing claims into new regions. 128-bit primes expand the
collision-safe frontier; SPARQL disambiguation expands the
entity-resolution truth floor. Both add capacity, not just correctness.

5–6 make the surface self-describing. Threat-model validation turns
prose into executable assertions. Delta semantics turns a schema field
into a specified behaviour.

7–8 broaden the trust base. Supply-chain attestation raises the
verification floor from "trust the repo" to "verify the bytes."
LLM guardrails make the signed-vs-true distinction visible where it
matters most — at the interface the consumer reads.

## Stop-the-line triggers

If you discover any of the following during the work above, pause and
surface it before continuing:

- **A cross-runtime disagreement on a valid bundle.** This would break
  K1–K4 and invalidates the central public claim. Fix before anything
  else.
- **A verifier that accepts a bundle the spec says it must reject.**
  Same severity.
- **A claim in README.md / CHANGELOG.md / docs that you cannot defend**
  with a test, a measurement, or an explicit "designed, not proved"
  label. These are the most common lies to discover, and every one
  corrodes trust by some small amount.

## On the discipline itself

The single best habit to carry into the next session is: when you're
about to write a claim, ask "what would adversarial review do with
this sentence?" If the answer is "catch me," rewrite it before you
commit.

SUM's whole thesis is that content-addressed, cryptographically-
attested, cross-runtime-verified bundles are a truth-telling surface
because they can't lie without being caught. The repo itself should
hold to the same standard the bundles do.

---

# Beyond the priorities — platform trajectory

Priorities 1–8 above are the **hardening playbook**: they close known
truth gaps in the current surface. Everything in this section is
**post-hardening** and depends on that surface being solid. Don't run
Phase B or C work while Phase A priorities are still open — a platform
built on an unstable foundation fails in a way that blames the
platform, not the foundation.

## The greater goal, stated plainly

SUM today is a cryptographically-attested knowledge-bundle engine with
a proved cross-runtime verifier triangle. "Extremely useful platform"
means three things it is *not* yet:

1. **Trustworthy end-to-end for a specific user in a specific
   adversarial context** (journalist verifying a claim, LLM operator
   auditing extraction, researcher replicating a cited result).
2. **Composable across publishers**, so bundles flow and aggregate
   rather than sit in isolation.
3. **Self-describing enough that a motivated skeptic can verify the
   engine itself**, not just the bundles it produces.

Every phase below is an answer to one of those three gaps. Phase A
finishes the foundation; Phase B puts the first useful surface on top
of it; Phase C turns the surface into a network; Phase D is the
stability commitment that makes the network usable as a dependency.

## Phase A — finish the hardening playbook (Priorities 3–8)

No new thinking — everything is already rationalised above. Execute in
sequence.

| Priority | Effect |
|---|---|
| P3 `sha256_128_v2` | Moves prime scheme from "safe for <2³² axioms" to "safe at knowledge-graph scale." |
| P4 SPARQL disambig | Moves entity resolution from 80% correct to a measured >95% floor. Kills the 1-in-5 lie rate on `/api/qid`. |
| P5 threat-model validation | Every documented defence gets an executable test. Prose → prose + assertions. |
| P6 delta semantics | Specifies what composition means. **Precondition for every Phase C item.** |
| P7 supply-chain attestation | Consumers verify the bytes, not just trust the repo. |
| P8 LLM honesty guardrails | Signed ≠ true becomes visible at the interface the consumer reads. |

**Gate to exit Phase A.** Every priority's success-criterion line
above passes. The CI matrix has K1–K5 + A1–A6 + threat-suite +
supply-chain verify + LLM-guardrail assertions all green on every
push.

## Phase B — platform surface

These items only make sense *after* A. Shipping them earlier adds
surface to a foundation still settling.

**B1 — source anchoring in the bundle schema.** Every axiom optionally
carries `{source_uri, content_hash, byte_range, retrieved_at}`. A
journalist's attestation survives link-rot because the hash pins the
content; a fact-checker can re-derive the axiom from the pinned source.
Backward-compat schema extension: `provenance` becomes required when
`extractor == "llm"`, optional when `"sieve"`. Builds directly on the
existing AkashicLedger byte-range machinery.

**B2 — bundle explorer / viewer.** `single_file_demo/index.html`
extended from "paste → attest" to "open → read → verify → trace
provenance." Drop-in file, paste-in URL, or QR-scan entry points.
Shows signer DID, DID resolvability status, per-axiom provenance span,
verification result. Primary surface through which a non-technical
reader encounters SUM.

**B3 — `sum verify --explain` on the CLI.** When verification fails,
the verifier tells the user exactly why in one sentence they can act
on. Current error output is proof-level detail; `--explain` is the UX
layer. Same input, new output format; the underlying verify logic is
unchanged.

**B4 — first-try onboarding.** `pip install sum-engine[sieve]` →
`echo "…" | sum attest | sum verify` must produce a visibly
interesting result in under 60 seconds with zero ambiguous steps.
The current path works but isn't rehearsed. A `sum tutorial`
subcommand walks a new user through attest → sign → verify →
view-in-browser in five numbered prompts.

### Process-intensification additions (B5 – B7)

The first four Phase B items polish each surface in isolation. The
next three collapse multi-step user flows into single gestures —
"process intensification" in the chemical-engineering sense:
combining steps so the artifact a user touches is the deliverable,
not an intermediate. Honest analysis lives in chat-archive only;
the durable form is below.

Each item names the foundation step that must land first. Do not
skip those dependencies. Doing so ships a more-usable surface on
top of an unsettled foundation, which is worse than a less-usable
surface without that risk.

**B5 — shareable bundle URLs (`/b/{hash}`).** Today a CanonicalBundle
is a JSON file. Civilians do not share JSON files casually. A new
Worker route `PUT /b/` accepts a bundle, content-addresses it
(`sha256` over JCS-canonical bytes), stores in R2, returns a short
URL. `GET /b/{hash}` renders a verified HTML view (signer DID,
DID resolvability status, axioms, signatures-verified summary)
using the same SubtleCrypto path the demo already uses. Idempotent
on identical bundles. Pinning policy must be explicit: do NOT
promise indefinite retrievability under v1 prime scheme — that
locks v1 in even after `sha256_128_v2` (Priority 3) lands. Spec a
soft-expiry of "best-effort 12 months" until 1.0 contract (Phase D)
freezes the pinning rules.
**Foundation.** B1 (source anchoring) lands first so URLs carry
useful provenance.
**Intensification.** Removes an entire artifact class (the JSON
file) from civilian awareness. Makes a one-URL-is-the-attestation
surface possible.

**B6 — PWA-installable demo.** Add `manifest.json` + a service
worker that caches the static assets (HTML / JS / `sum_core.wasm`)
to the existing `single_file_demo/` deployment. The demo becomes
installable on phones via "Add to Home Screen"; offline verify
works after first load. ~40 lines of code plus the manifest. No
backend change. **Foundation.** None — purely additive.
**Intensification.** A journalist at a press briefing can verify a
bundle from their phone without WiFi. Closes the mobile gap.

**B7 — `sum attest <url>` fetch mode.** Currently `sum attest`
takes prose on stdin or `--input <file>`. Add a positional URL
argument: `sum attest https://example.com/article > bundle.json`.
The CLI fetches with the existing `httpx` dependency, sets
`source_uri` to the URL and `retrieved_at` to wall-clock time,
records HTTP status + final URL after redirects in the bundle's
`sum_cli` sidecar. Per-request timeout default 10 s, configurable.
Does NOT follow `<noindex>` / `<meta name="robots" content="noindex">`
out of basic respect. **Foundation.** B1 (source anchoring) ships
the schema field first.
**Intensification.** Eliminates the "open browser → copy text →
switch to terminal → paste → run command" five-step pattern.
Single invocation from anywhere the URL appears.

### Out of Phase B (named so we don't lose them)

- **Browser extension: "Attest this selection".** Right-click on
  any web text → mints + signs a bundle with current URL + byte
  range as provenance → uploads to `/b/` → copies the shareable
  URL to clipboard. **Depends on B1 + B5 + B7.** Ship as v0.4
  feature in its own dedicated session — it is not in-scope for
  Phase B's "polish what already exists" frame.
- **Verify badges (Shields.io-shaped SVG).** `GET /badge/{hash}.svg`
  returns a live-verifying badge (✓ verified / ✗ invalidated /
  ⚠ unreachable). Publishers embed in READMEs, blog posts, papers.
  **Depends on B5.** Add to Phase C as `C5` once `B5` lands and
  there's anything to badge against.

**Gate to exit Phase B.** A fresh user, given only the README, can
produce a verified signed bundle with source-anchored axioms and
open it in the explorer without asking a question. With B5–B7
landed: that user can also produce + share a bundle by URL from
their phone, and a recipient can verify the URL from their phone
without installing anything.

## Phase C — network layer

SUM stops being a tool and starts being a platform. Conditional on
**B1** (source anchoring) and **P6** (delta semantics) both landing.

**C1 — bundle registry / discovery.** Open index any publisher can
push to, any consumer can pull from. Simplest shape: a well-known URL
pattern (`https://{domain}/.well-known/sum/bundles.json`) enumerating
recent bundles with CIDs + signer DIDs. No centralised server, no
network-effects mandate — a convention that makes discovery possible
without infrastructure.

**C2 — bundle composition UX.** B2's explorer learns to open a second
bundle as a delta over the first and show the composed state. A
fact-checker's flow becomes: open journalist's bundle → add own
attestations over a subset → export delta → publish at own well-known
URL.

**C3 — cross-attestation graph.** Given a root bundle and its deltas,
produce a traversable graph showing who attested what and when. This
is the primary artefact that makes SUM *valuable for fact-checking at
scale* — it surfaces consensus or disagreement between independent
attestors.

**C4 — standards interop.** Full W3C VC 2.0 emit/verify path (some
scaffolding exists). PROV-O export. Optional JSON-LD context. A bundle
survives as a W3C credential in W3C-credential tooling and round-trips
back to the SUM format unchanged.

**Gate to exit Phase C.** Two independent publishers can produce
bundles that compose into a third-party aggregate verifier's state,
and the aggregate verifier agrees with both publishers' local
verifiers. Disagreement surfaces as a named diagnostic, not as
silent drift.

## Phase D — 1.0 stability contract

Not a phase of new work so much as a decision point.

Before tagging `v1.0.0`: an explicit compatibility contract. What
bundles shipping 1.0 will continue to verify 10 years from now. What
schema fields are frozen. What CLI flags are stable. What wire-format
guarantees hold across minor versions. What requires a major bump.

This turns the implicit "we won't break things" into a documented
promise, backed by a corresponding CI gate that tests the promise
(e.g. a corpus of 1.0-minted bundles that every later release must
continue to verify). It is the hand-off point between "interesting
prototype" and "dependency anyone can build on."

Until this is in place, every `sum-engine` release is implicitly
saying "you should be fine, probably." 1.0 is when that becomes
"you will be fine, because the test suite asserts so."

## How to use this document

- **Reading order** if you are a memory-less session: Priorities 1–8,
  then this trajectory section. Don't skip ahead to Phase B work if
  Phase A priorities are still open.
- **Phase tags in commits**: optional but useful. Prefix
  `[P3] …` or `[B1] …` so `git log --grep` finds all work on a given
  item.
- **When you discover a dependency** between two items that isn't
  captured here, add the edge as an explicit sentence in the item
  text. The ordering here is the best I could make it; future
  learning refines it.
- **Scope pressure** comes from two directions — "ship faster, skip
  the gate" and "ship more, add an item." Both are wrong. The
  priorities close debts that already exist; the phases add surface
  that depends on those debts being closed. Shipping anything new
  before the gates it depends on pass moves the debt, not the
  value.
