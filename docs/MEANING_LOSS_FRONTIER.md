# The meaning-loss frontier

*The layer below the fact. Status: research (behind the `[research]`
extra). First rung shipped — see `sum_engine_internal/research/meaning/`
and `docs/MEANING_RISK_RECEIPT_FORMAT.md`.*

---

## 1. The frontier, stated plainly

SUM's slider contract measures **fact-level** loss: did the
`(subject, predicate, object)` triples survive a transform? That is real
and load-bearing — but meaning lives *beneath* the proposition. Two
texts can carry the same facts and not the same meaning; one text can
lose no fact and still lose almost everything that mattered.

The layers between "fact preserved" and "meaning preserved":

| Layer | What lives here | Computable today? |
|---|---|---|
| **Fact** *(where the slider is)* | propositions, `(s,p,o)` triples | ✅ yes — triple-match (formally a degenerate Smatch) |
| **Logical / discourse** | negation, scope, modality, coreference, coherence | ✅ yes, with per-channel loss — AMR/Smatch → DRS/Counter → UMR/AnCast++ |
| **Implied / elided** | implicature, presupposition, what the reader reconstructs from an indicator | ⚠️ detectable in isolation; **no validated preservation metric** |
| **Connotation / register** | tone, stance, emotional colour | ⚠️ lexicon-level only (e.g. Allaway 6-aspect connotation frames) |
| **Arrangement / sound** | meaning that emerges only from word order and structure; prosody, rhyme | ❌ form *change* measurable; form *meaning* **unsolved, arguably ill-posed** |

The deepest layer has a name in the rhetorical scholarship of the
Abrahamic scriptural tradition — *naẓm*, the doctrine (al-Jurjānī) that
eloquence is not in the words or the meanings taken separately but in
the *arrangement* that binds them; the limiting case is a text held to
be inimitable precisely because its arrangement cannot be moved without
loss. We take that tradition as the **honest ceiling**, not the target:
it marks the layer any bag-of-units or even entailment measure is blind
to by construction.

## 2. The one finding that organises everything

Across the research sweep, one seam runs through the entire literature:

> **Every *computable* measure of meaning is a proxy** (embedding
> distance, NLI entailment, a downstream task variable). **Every
> framework that genuinely *models* meaning** (Bar-Hillel–Carnap,
> Kolmogorov structure functions) **is uncomputable or toy-scale.** No
> one has a computable, intrinsic measure of natural-language meaning.

So the prize is not a better proxy. **The prize is the honest seam
itself** — being the first to seal a *verifiable, bounded* claim over a
meaning-space proxy, with the boundary stated. That is SUM's existing
discipline (provable / certified / measured / not-asserted) pointed one
layer deeper.

## 3. The paradigm we are building toward

Compose three things, two of which SUM already owns:

```
  semantic-entropy-style meaning-loss          ← the loss substrate
  (cluster / judge meaning by bidirectional       (Farquhar, Kossen, Kuhn & Gal,
   entailment, not by tokens)                       Nature 2024)
            │
            ▼
  Conformal / risk-controlling bound            ← the guarantee
  (distribution-free, finite-sample,              (Bates et al. JACM 2021;
   marginal upper bound on expected loss)          Angelopoulos & Bates 2023)
            │
            ▼
  Signed, replayable receipt                     ← the verifiability
  (JCS + Ed25519 + detached JWS, plus a            (SUM's existing trust stack
   replay anchor over the committed losses)         + one new field)
```

The result is a *signed, same-commit-replayable certificate that bounds
a named proxy for meaning-loss, computed in **checkable text space**
(not from model internals)*. We are not aware of a prior artifact
combining a distribution-free meaning-loss-proxy bound with a replayable
signed receipt — but the load-bearing claim is the composition and its
caveats, not a priority race. It does **not** claim to have measured
meaning (the shipped lexical default is, by construction, a *lexical*
overlap measure — it over-reports loss on a faithful reword). It bounds
a *named proxy* for meaning-loss, *marginally* (on average over a named
target population), requiring independent calibration draws, a fixed policy
and no unaccounted adaptive selection. Exchangeability alone is insufficient.
Those caveats are the contract,
and they ride inside the receipt.

The decisive design choice — and the defensible contrast — is **checkable
text space, not model internals**: the loss is computed from text
(entailment between texts, lexical coverage of texts), so a third party
can recompute it. We deliberately reject model-internals
(SAEs, probes, steering vectors) for this surface: they are
model-dependent, non-deterministic (SAEs share only ~30 % of features
across seeds), and unverifiable — incompatible with a signed receipt.
They belong on the internal-research side, like the sheaf detector.

## 4. What shipped (first rung)

`sum_engine_internal/research/meaning/` — three composable pieces:

- **`meaning_loss.py`** — named, versioned, bounded `[0,1]` proxies.
  - `LexicalCoverageScorer`: deterministic, dependency-free. A runnable
    *placeholder* (bidirectional content-unit overlap distance) whose
    only job is to make the whole pipeline testable with zero model
    downloads. It **cannot see through paraphrase** — it over-reports
    loss on a faithful reword, by design and by name.
  - `EntailmentScorer`: the real path. Bidirectional-entailment loss
    over a caller-**injected** `entails(premise, hypothesis)` judge (an
    NLI model or LLM judge — the same shape as the slider bench's v0.4
    NLI audit). The model is never imported here, so the module stays
    dependency-free and the certified bound stays tied to a *named*
    judge.
- **`conformal_meaning.py`** — `certify_meaning_risk(...)`: a
  distribution-free **upper** bound on expected meaning-loss. It is the
  exact dual of the slider's rate kernel — `upper-bound on E[loss] =
  1 − lower-bound on E[1−loss]` — so it reuses the adversarially-hardened
  Hoeffding / Clopper–Pearson bounds (NaN-rejecting, clamped) rather
  than re-deriving a concentration inequality. A third method,
  **empirical-Bernstein** (`method="empirical_bernstein"`; Maurer &
  Pontil, COLT 2009), scales the deviation with the *observed* variance.
  Because faithful transforms cluster near zero loss — a low-variance
  regime — it certifies a materially **tighter** ceiling at realistic
  batch `n` (n ≳ 64): on a 200-pair faithful batch it lands a 0.073
  ceiling where Hoeffding gives ≈0.117 (the difference between a useful
  and a near-vacuous receipt — F22). It is **not** a universal upgrade:
  the additive `7·ln(2/δ)/(3(n−1))` term dominates at tiny `n`, so
  Hoeffding is tighter there — eB is the right tool for a real batch, and
  the Monte-Carlo coverage test (`test_empirical_bernstein_coverage_valid`,
  `empirical_risk_coverage(..., "empirical_bernstein")`) is the receipt
  that the tighter radius stays valid (≥ 1−δ coverage).
- **`receipt.py`** — `sum.meaning_risk_receipt.v1`: sign + verify +
  **replay**. The payload commits a `losses_hash`; a verifier handed the
  same loss vector side-band recomputes the hash *and* re-runs the
  certifier, reproducing the bound byte-for-byte on the same commit.
  That replay property is what turns "measured" into "measured *and*
  independently reproducible" — the gap the slider contract's T2/T3
  names. See `docs/MEANING_RISK_RECEIPT_FORMAT.md`.

## 5. The proof boundary (non-negotiable)

A verified meaning-risk receipt **proves**: the payload was signed by
the key holder; the committed losses hash to `losses_hash`; re-running
the named certifier reproduces the bound.

It does **not** prove:

- **that meaning was preserved** — only that a *named proxy* for
  meaning-loss is bounded *on average*. Swap the proxy, the number means
  something else; the scorer's name + version are in the payload for
  exactly this reason.
- **anything per-document** — the bound is *marginal*. Per-document
  control is provably not free (conditional conformal). A receipt must
  never be read as "this passage's meaning was certified".
- **anything about arrangement, sound, connotation, or implicature** —
  these are listed in the payload's `not_covered` field. The proxy is
  structurally blind to them, and the honest act is to *say so*, not to
  let silence imply coverage. (Validity also requires independent calibration draws from the target distribution
  with the named corpus; state the envelope with the number, always.)

This is the same discipline as `docs/PROOF_BOUNDARY.md`, one layer
down. No "guarantee" language without a same-commit replay receipt — the
rule the bench-hardening plan set for the slider applies here verbatim.

## 6. The natural test bed

The honest empirical bed for sub-factual meaning-loss is a corpus of
**maximally dense source held against many parallel translations** — the
translators were *forced* to expand what the source compressed, so the
loss is visible across renderings. The scriptural canon of the Abrahamic
traditions is the archetype, with mature open annotation layers (token
roles, pronoun→antecedent maps) that mark exactly where the elided
meaning lives. The precedent that "incompressibility of a canon" is
publishable already exists for sacred *images* (Kolmogorov-complexity on
Byzantine icons, *Scientific Reports* 2022); no one has done it for
canonical *text*. Measuring the loss is homage to the incompressibility,
not a claim to have captured it.

Wiring a real parallel-translation corpus + a real NLI judge into
`EntailmentScorer → certify_meaning_risk → sum.meaning_risk_receipt.v1`
is the next rung. It is `$`/operator-gated (judge cost, corpus curation)
and is **not** built until that pull arrives — per the charter's
scope-before-signal rule.

---

*References (load-bearing): Farquhar, Kossen, Kuhn & Gal, "Detecting
hallucinations in large language models using semantic entropy",
Nature 630 (2024). Bates, Angelopoulos, Lei, Malik & Jordan,
"Distribution-Free, Risk-Controlling Prediction Sets", JACM 68(6)
(2021). Angelopoulos & Bates, "Conformal Prediction: A Gentle
Introduction", FnT ML (2023). Beauchemin et al., "MeaningBERT", 2023.
Allaway & McKeown, connotation frames, EACL 2021. al-Jurjānī, Dalāʾil
al-Iʿjāz (on naẓm). Peptenatu et al., Scientific Reports 12 (2022),
Kolmogorov complexity of canonical iconography.*


## Candidate-path and inspection scope (2026-09-09)

`RenderFrontier` keeps its API name for compatibility. It stores a caller-ordered
candidate path, not a computed Pareto optimum. Position is an index; neither
length nor proxy quality is assumed monotone. The legacy `faithful` and
`compressed` accessors return the first and last candidates. JSON includes
`path_kind`, actual word counts and the scorer instrument when available.
Compare observed candidate outputs from the same original source before making
a dominance or quality claim.

The local NLI scorer keeps its historical longest-first finite-window algorithm.
Its explanatory readout now reports input-token coverage and identifies partial
judgments by direction and claim index. A partial score may have missed support
or part of the hypothesis and must not be presented as complete review.
`within_token_window` means only that token truncation was not needed; it says
nothing about judge accuracy, sentence segmentation or full meaning coverage.
Other callbacks without a coverage inspector report `not_inspected`.
Retrieval of support windows, richer source spans and independent human
validation remain required before promoting this proxy to a complete reviewer.

## Optional Jev judge (research, 2026-09-20)

`sum_engine_internal.research.meaning.jev_judge.JevJudge` adapts TypeSafe's
Noul support questions to the existing bidirectional `EntailmentScorer`.
`JevJudge.from_env(allow_network=True)` explicitly enables the fixed HTTPS
endpoint using `TYPESAFE_API_KEY`; `judge.as_scorer()` supplies the scorer.
Construction otherwise stays offline. This does not add a hosted browser/MCP
route or enable Jev for receipt issuance.

The default pinned version is `jev-1.13.0`. Probabilities at or below 0.1 map
to unsupported, at or above 0.9 to supported, and the intervening interval
raises `JevAbstention` instead of producing a Boolean loss. These thresholds
are experimental, not calibrated on human labels. `assess_batch()` exposes
the probabilities and abstentions directly. It evaluates narrow questions
over shared premises; it never silently truncates input. Byte caps are resource
limits, not tokenizer coverage measurements. Oversized inputs need an explicit
windowing/evidence policy outside this adapter.

Each instance bounds attempts, questions, request/response bytes, and socket
timeouts. Failed calls consume attempts; automatic retries are disabled. The
model identity, answer IDs/types, finite probabilities and usage are validated.
`observations()` returns detached, unsigned request/response records, including
uncertain responses, for a caller to save privately. These records contain
source text. They do not prove provider identity, semantic accuracy, or hosted
inference reproducibility. The existing offline verifier has no Jev dependency.

Contract: [TypeSafe API](https://docs.typesafe.ai/api) and
[current model limits](https://docs.typesafe.ai/models), checked 2026-09-20.
Tests use fake transports and establish protocol behavior only. A live,
human-labeled assessment of actual SUM transformations remains necessary.

### Run a descriptive evaluation or replay saved responses

The installed research module accepts exact source/output pairs and optional
sentence-aligned Boolean labels. `source_id` groups related originals; pair IDs
are unique. The fixture below is handwritten synthetic diagnostic material,
not a benchmark of actual SUM generations or independent human validation.
Labels and dataset metadata are never sent to the model.

```bash
# Live, paid evaluation: explicitly opt in and supply TYPESAFE_API_KEY securely.
python -m sum_engine_internal.research.meaning.jev_evaluation \
  --input fixtures/jev_evaluation/synthetic_pairs.json \
  --out /tmp/jev-live.json --allow-network

# Offline: recompute from the exact recorded requests and responses.
python -m sum_engine_internal.research.meaning.jev_evaluation \
  --input fixtures/jev_evaluation/synthetic_pairs.json \
  --out /tmp/jev-replayed.json --replay /tmp/jev-live.json
```

Output paths must be new. Files use mode 0600 and contain complete source,
output, claim-level findings, configuration, and unsigned observations. Each
direction is assessed even when the other abstains. An abstained document has
no numeric readout. Empty-input rules are deterministic and never masquerade
as model probabilities. Provider/protocol failure stops further calls, marks
the run incomplete, and retains completed directions and available responses.
Exit 0 means execution completed; it does not mean the document passed review.

Metrics are descriptive counts, Brier score, false-support fraction among
accepted labelled claims, and two separate coverage denominators: observed
labels versus requested labels, and non-abstained decisions versus observed
labels. Unlabelled input yields no accuracy estimate. Claims sharing an original
are not independent observations, and no population confidence bound is issued.

The command defaults to 100 attempted requests and a 64,000-byte response cap.
Input files are capped at 16 MB and replay reports at 64 MB. A conservative
report-size preflight reserves the maximum response storage before inference;
split large datasets when it rejects a run. Use `--max-requests`,
`--max-response-bytes`, `--accept-at-or-above`, and `--reject-at-or-below` to
declare a different experiment. Replay requires the same configuration and
scoring/evaluation implementations and checks recomputed results against the
saved report. It does not call the provider, reproduce original timing, verify
human labels, or authenticate an unsigned report. Saved source text requires
the same handling as the original documents.
