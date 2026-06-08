# The Translator — and the omnidirectional distiller

*Status: plan + proof-of-concept (the meaning-loss dial is proven
cross-lingual; the translation transform itself is operator/`$`-gated, like
the slider's LLM render path). Grounded in the 2026-06-07 extended
simulation (docs/DOGFOOD_FINDINGS_2026-06-07.md, F24–F26).*

---

## The proof first (so the vision isn't hand-waving)

The meaning-loss dial works **across languages**, today, through the
*existing* pipeline — `nli_entailment_scorer(model_id=<multilingual NLI>)`,
no new code (the scorer is already `model_id`-parameterised):

| translation of an EN source | meaning-loss |
|---|---|
| FR / ES / DE faithful | **0.000** |
| FR that drops a claim | **0.300** |
| ES mistranslation (negates source) | **1.000** |

So everything below is an *assembly of parts we have*, not an invention.

## "Translate known languages with all our dials present"

Translation is just a **transform** — `extract`/`slider`/`compose` already
prove the pattern. The full SUM stack is language-agnostic; point it at a
multilingual judge and every dial we built applies to translation:

| Dial (already shipped) | What it becomes for translation |
|---|---|
| **density / length** (slider) | translate *and* summarise/expand in one move |
| **formality / audience** (perspective) | translate **for** a novice vs an expert vs a regulator — a **Perspective Receipt per target audience** |
| **the frontier** | the faithful↔compressed path *in the target language* — scrub from full translation to a one-line gloss |
| **meaning-loss** (the judge) | a **multilingual** judge: how much of the source's meaning survived the crossing (proven above) |
| **the receipt** | a signed, replayable **translation-fidelity certificate**: "this FR rendering preserves ≥X of the EN source's meaning, per a named multilingual judge, controlled for the regulator audience" |
| **cross-runtime verify** | the JS verifier (`meaning_receipt_verifier.js`) checks the certificate in any runtime |

That is a translator no one else has: not "here's a translation," but
**"here's a translation *and* a verifiable receipt of exactly what it
preserved and for whom"** — chain-of-custody for meaning *across the
language barrier*.

## The omnidirectional distiller (the expansion you invited)

Translation is one *axis*. The deeper thing we're building is a
**meaning-preserving transformation engine** where *any* move through the
space of knowledge carries a measured, certified, signed receipt of what
it kept:

```
            EXPAND (tome)
                 ▲
                 │
  TRANSLATE ◀────●────▶ RE-PERSPECTIVE (novice ⇄ regulator)
  (EN⇄FR⇄…)      │      
                 │
            DISTILL (tag)
```

- **tags ↔ tomes** — compress and expand (the origin dream).
- **× languages** — translate in any direction (proven today).
- **× perspectives** — re-render for any audience (Perspective Receipts).
- **× modalities** — the later horizon (text → audio/image, decision: text first).
- at **every hop**, the meaning-loss is *measured* (per-document) and
  *certifiable* (per-batch), and the transformation carries a
  **chain-of-custody receipt** robust to rewriting.

"Omnidirectional" is literal: you can move in *any* direction in the
meaning-space — distil, expand, translate, re-frame, simplify — and the
engine always answers "**what did this lose, and can you prove it?**" That
is the truest sense of a data extrapolator/distiller: not a summariser,
not a translator, but a **trust layer over every transformation of
meaning**, with the receipts to back it.

## Architecture — what's there vs what's new

- **Reuses (built):** `EntailmentScorer` + the multilingual judge
  (`model_id` param), `RenderFrontier`, `certify_meaning_risk[_by_group]`,
  the meaning-risk + perspective receipts, the JS verifier, the
  `verify-meaning` CLI on-ramp.
- **New, small:** a `translate` transform (registers like slider/extract);
  a curated multilingual judge default; a `translation_fidelity` receipt
  flavour (or reuse `meaning_risk` with a `source_lang`/`target_lang`
  field — a schema decision).
- **Operator/`$`-gated (NOT built until pull):** the *translation itself*
  needs an MT/LLM model to **produce** the target text — exactly the same
  division as the slider (the engine *produces*, SUM *certifies*). The
  certifying half is no-`$` and proven; the producing half is gated.

## Open decisions this PoC surfaced (for the spec)

1. **F25 — normalization is load-bearing.** Stripped diacritics made the
   judge mis-score a faithful translation as a contradiction; restored
   accents → entailment 1.00. The translator MUST normalise/preserve
   Unicode before scoring. Pin a normalization (NFC) in the receipt.
2. **F26 — translation receipts are model-judge receipts, machine-pinned.**
   The lexical scorer is useless cross-lingual (≈0 word overlap), so a
   translation-fidelity receipt *must* use a model judge → its replay is
   machine-pinned (cross-hardware float drift). The receipt must pin the
   **model id + runtime**; the signature still verifies cross-runtime, the
   meaning recomputation is reproduced only on a matching stack. The
   measurement-audit's **coarse-quantise-the-judge-decision** technique
   (extend the integer-micro discipline to the entailment probability)
   shrinks the flip-band and should land before translation receipts ship.
3. **Tighter bounds (empirical-Bernstein).** Translation corpora will be
   low-variance (faithful translations ≈0 loss); the audit's
   empirical-Bernstein upgrade tightens the certified bound materially at
   realistic n — the difference between a useful and a vacuous
   translation receipt.
4. **Which multilingual judge.** `mDeBERTa-v3-base-xnli` works (proven);
   stronger multilingual NLI / faithfulness models (e.g. a MiniCheck-class
   multilingual checker, if one exists) are the upgrade axis.

## Honest boundary

**Proven today:** the meaning-loss dial measures translation fidelity
correctly across EN/FR/ES/DE through the existing pipeline. **Aspirational:**
the full translator product (the `translate` transform + MT integration +
the productised receipt) is the next build, and its *producing* half is
`$`-gated. As always: every number stays `measured` until a same-commit
replay receipt makes it `certified`, and the receipt never claims more
than the named proxy + named judge deliver.

The binding constraint hasn't moved: a real user. But the translator is
the most *legible* expression of the whole thesis — everyone understands
"a translation you can trust" — and it's the one that turns the
omnidirectional distiller from a beautiful abstraction into a thing a
person can hold.
