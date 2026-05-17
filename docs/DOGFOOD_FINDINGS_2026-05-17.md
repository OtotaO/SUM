# DOGFOOD_FINDINGS_2026-05-17.md

**Status: planning artifact. Captures the first mechanical dogfood pass by the engine session on 2026-05-17.** The user asked the engine session to execute Scenario A from [`DOGFOOD_QUICKSTART.md`](DOGFOOD_QUICKSTART.md). Engine-session execution is *mechanical only* — it can run the pipeline, but the subjective evaluation ("would I publish this without re-reading every source?") is irreducibly user-side. What this doc captures is the **friction signal**, not the subjective trust judgment. The user's own pass produces the latter when they run.

## What was executed

Three short public-policy sources composed by hand (text-paste-equivalent of what a journalist would copy from PDFs):

- `src1.txt` — EU AI Act Article 12 logging requirement.
- `src2.txt` — FTC AI disclosure guidance.
- `src3.txt` — C2PA Content Credentials.

Pipeline attempted: `sum attest` → `sum transform apply compose` → `sum transform apply slider` → verify-via-receipt.

Plus Scenario B's live-Worker LLM-axis render via `/api/render`.

## Findings — six friction points

### F1 — `transformers` `FutureWarning` writes to stdout, poisons JSON pipe targets [severity: HIGH]

When running `sum attest --extractor=sieve < src.txt > bundle.json`, the resulting `bundle.json` is not valid JSON. The file starts with:

```
/opt/homebrew/Caskroom/miniforge/base/lib/python3.10/site-packages/transformers/utils/generic.py:441: FutureWarning: ...
```

…then the JSON envelope. Root cause: the `transformers` library (pulled in via the sieve extractor) emits a `FutureWarning` on import via `print()` to stdout, not stderr.

**Workaround:** prepend `PYTHONWARNINGS=ignore` to every `sum attest` invocation.

**Fix:** suppress the warning at attest-call entry with `warnings.filterwarnings("ignore", category=FutureWarning)` before the transformers import. Or downgrade the transformers warning to stderr explicitly.

**Why it matters:** every dogfood Scenario-A user hits this on step 1. They will not know to add `PYTHONWARNINGS=ignore`. The pipeline appears broken for no reason.

### F2 — Sieve extraction is too conservative for journalist use [severity: MEDIUM]

Source: *"The EU AI Act entered into force on August 1, 2024. Article 12 requires high-risk AI system providers to maintain automatic logs of events throughout the system's lifecycle."* (~50 words, several extractable facts.)

Sieve extracted: **1 axiom** — "article maintain automatic log."

Dropped: "EU AI Act entered into force 2024-08-01," "system_lifecycle scope," "high-risk_AI_system providers obligation," "6_months minimum retention," etc.

For a journalist's distill-three-sources workflow, the conservative sieve loses the load-bearing facts (dates, regulatory references, specific scopes). The distilled brief produced from this would be useless as a journalist's citation chain.

**Workaround:** wire the LLM extractor instead of sieve. Requires `OPENAI_API_KEY` and is currently undocumented in `DOGFOOD_QUICKSTART.md`.

**Fix paths (any one is sufficient):**
- Default scenarios in `DOGFOOD_QUICKSTART.md` to LLM extraction with sieve fallback.
- Improve sieve to handle ISO-8601 dates, named-entity multi-token preservation, and numeric quantities.
- Surface this gap as `extraction.verifiable: true | false` on the bundle so the consumer knows what was missed.

**Why it matters:** journalists are the named wedge ICP per the charter. If the wedge can't run the wedge use case, the wedge doesn't close.

### F3 — PyPI v0.6.0 lacks the `transform` subcommand [severity: CRITICAL]

`pip install --upgrade 'sum-engine[openai,sieve,receipt-verify]'` installs v0.6.0 from PyPI. The installed binary at `/opt/homebrew/Caskroom/miniforge/base/bin/sum` has subcommands: `attest, attest-batch, verify, render, resolve, ledger, inspect, schema, compliance` — **no `transform`**.

The transform-substrate arc (PRs #210–#228 — receipts, registry, fixtures, replay-window check, T1c-followup, etc.) is in repo HEAD's pyproject as v0.6.0 BUT lives entirely in CHANGELOG's `[Unreleased]` section. **The PyPI wheel hasn't been re-released since vendor-adapter v0.6.0.**

Effect: a user who follows `DOGFOOD_QUICKSTART.md` after `pip install --upgrade` cannot run Scenario A at all. `sum transform apply compose` returns:
```
sum: error: argument <command>: invalid choice: 'transform'
```

**Workaround:** `pip install -e /path/to/repo` makes the local repo the install source; `sum transform` then works (same v0.6.0 string but more features).

**Fix:** cut a 0.7.0 release. Bump `pyproject.toml`, close the `[Unreleased]` CHANGELOG section as `[0.7.0]`, run `make verify-release-bytes`, tag, push. The `publish-pypi.yml` workflow handles the rest via Trusted Publishing OIDC. User action — requires the tag push.

**Why it matters:** this is the operative release gap. Every dogfood reader, every funder pip-installing right now, every new user — they all get an incomplete SUM that's missing the substrate that the README, the CHANGELOG, and the grant applications all describe. The version string lies.

### F4 — `sum attest` output shape is incompatible with `sum transform apply compose` [severity: HIGH]

`sum attest` emits a CanonicalBundle with keys: `bundle_version / canonical_format_version / branch / axiom_count / canonical_tome / state_integer / timestamp / prime_scheme / state_integer_hex / is_delta / axiom_graph_entropy / axiom_consistency_check / sum_cli`.

`sum transform apply compose` requires each bundle to have a `triples` or `axioms` top-level key. Neither is present in attest's output.

```
sum: transform 'compose' failed: compose: bundle dict must have 'triples' or 'axioms' key
```

**Workaround:** re-extract triples from the bundle's `canonical_tome` string OR hand-build `{"triples": [...]}` shapes outside the attest path. Both defeat the value of attesting first.

**Fix paths:**
- `sum attest` adds an `axioms` field (a list of `{subject, predicate, object}` records) to its emitted bundle. The data exists internally — it's just not surfaced in the v1.1.0 bundle output schema.
- `compose` accepts the `canonical_tome` parse-back.

**Why it matters:** the dogfood quickstart's headline pipeline (`attest → compose → slider`) is **literally not runnable end-to-end today**. The user (you) will hit this immediately if you try Scenario A unmodified.

### F5 — Scenario A pipeline broken end-to-end [severity: HIGH; depends on F3, F4]

The dogfood quickstart's flagship scenario fails at step 4 (compose) for two compounding reasons (F3, F4). A new user running it cold gets a stack of errors with no clear remediation in the doc.

**Fix:** either ship a release that includes `transform` (F3 fix) AND wire attest's bundle shape compatible with compose (F4 fix), OR rewrite the scenario to use a different surface that's actually live today (e.g., direct `/api/render` like Scenario B, or live `/api/transform`).

### F6 — Live Worker's Anthropic key is invalid [severity: CRITICAL — production-impacting]

`POST https://sum-demo.ototao.workers.dev/api/render` with any off-centre LLM axis returns:

```json
{
  "error": "render failed: anthropic 401: {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"},\"request_id\":\"req_..\"}",
  "cache_key": "..."
}
```

Canonical-path renders (all axes at 0.5) still work — the deterministic path doesn't call Anthropic. LLM-axis renders are dead on production.

**Root cause (hypothesis):** the Anthropic key rotation noted in the prior session — the user rotated the key at console.anthropic.com but the Worker secret was not updated, OR the rotation was the user invalidating the old key without setting the new one on the Worker.

**Fix:** user-side, requires wrangler login.
```bash
wrangler secret put ANTHROPIC_API_KEY
# paste new key when prompted
```

`scripts/probes/operator_audit.sh` (landed in PR #222) is the empirical check tool for this class of issue. Running it would have surfaced the gap. The user has not yet run it.

**Why it matters:** every funder who clicks "live demo" on the README and tries an off-centre slider gets a 401 error. The README's "Slider fact preservation: median 1.000" claim is supported by a bench, not by the live demo currently serving. This is the single highest-priority external-visibility fix.

## Summary table

| # | Friction | Severity | Owner | Fix complexity |
|---|---|---|---|---|
| F1 | `FutureWarning` to stdout poisons pipes | high | engine | trivial (1-line filterwarnings call) |
| F2 | Sieve conservative — drops dates / specifics | medium | engine | medium (improve sieve or default to LLM extraction) |
| F3 | PyPI v0.6.0 lacks `transform` | **critical** | user (tag push) | trivial (cut 0.7.0 release) |
| F4 | `attest` output shape ≠ `compose` input | high | engine | trivial-to-medium (add `axioms` field to bundle) |
| F5 | Scenario A pipeline broken end-to-end | high (composite) | engine + user | resolved by F3 + F4 |
| F6 | Live Worker Anthropic key 401 | **critical** | user (wrangler) | trivial (one wrangler command) |

## What this dogfood DID NOT produce

- The user's subjective trust judgment ("would I publish this without re-reading every source?"). That requires the user running their own writing through the pipeline. The engine session's mechanical pass can't substitute.
- A finding on Scenario B's audience-slider reshape value (blocked by F6 — the LLM-axis path is dead on production).
- A finding on the receipt-as-product feel (Scenario C). Could be tested independently of F6 against the README's canonical-path positive control fixture, but wasn't run in this session.

## What this dogfood SHOULD trigger next

Two findings are buyer-or-dream-load-bearing and should be acted on before the next funder pip-installs:

1. **F3** — cut a 0.7.0 release. The version-string-lies issue means every funder who pip-installs is getting a misrepresented SUM. User action: tag and push. ~10 minutes.
2. **F6** — rotate the Anthropic key on the Worker. The live demo's LLM path is dead. User action: `wrangler secret put ANTHROPIC_API_KEY`. ~2 minutes.

The other findings (F1, F2, F4, F5) are engine-session work and should be triaged against the charter's constraints (no more than one substrate PR per week unless pulled). F1 is the smallest and lowest-risk; F4 is the highest-leverage for the dogfood loop to work.

## Loop closure

This document is the first artifact in the dogfood-finding-feedback class. Future dogfood sessions should append findings or supersede this with a dated successor. The user's own dogfood (Scenario A on their own writing) is still owed — it produces the subjective trust signal this mechanical pass cannot.

## Pointers

- [`DOGFOOD_QUICKSTART.md`](DOGFOOD_QUICKSTART.md) — the quickstart this session executed against.
- [`CHARTER_2026-05-17.md`](CHARTER_2026-05-17.md) §5.1 — dogfood is one of three load-bearing signals.
- [`ZENITH_FRAMING_2026-05-16.md`](ZENITH_FRAMING_2026-05-16.md) — many of these findings invoke the `verify --explain` layered output or the Epistemic Nutrition Label concept.
- `scripts/probes/operator_audit.sh` — empirical check for class F6 issues (operator-local; requires wrangler login).
