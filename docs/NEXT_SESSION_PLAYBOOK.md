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
