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
corpus), under *exchangeability*. Those three caveats are the contract,
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
  than re-deriving a concentration inequality.
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
  let silence imply coverage. (Validity also rests on exchangeability
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
