# DOGFOOD_QUICKSTART.md

**Five-minute guide to running SUM on your own writing.** Standing direction (memory `project_direction_2026-05-11`, charter §3, §7.2) names dogfood as the load-bearing signal that informs every strategic decision downstream of "is the substrate shipped." If you (the user, Umar) have not dogfooded recently, this is the document that should make it cheap.

This is the *user-facing* dogfood. The repo's own bench harness is a different surface; that runs against pinned corpora. Here we run against *your own writing* — the kind you'd publish.

## What dogfood is for

The charter names three signals as load-bearing:
1. **Grant signals** (mechanical — they arrive on a calendar).
2. **External pull** (rare — wait for it).
3. **Dogfood findings** (cheap, in your control, not yet captured).

A dogfood finding has the shape:

> *"I ran SUM on my real writing through scenario X with parameters Y. The output was Z. I [would | would not] publish this without re-reading every source. Specifically, what's missing or wrong is W."*

The result is binary in spirit: either the slider is the dream made code, or it's substrate without product-fit. Both outcomes are useful. The non-finding (no dogfood) is the worst outcome because it keeps the deliberation in theatre.

## Setup — one-time, ~3 minutes

```bash
# 1. Check your installed version. If you're below 0.6.0, upgrade.
pip show sum-engine | grep Version

# Expected: Version: 0.6.0
# If older: pip install --upgrade 'sum-engine[openai,sieve,receipt-verify]'
# (Last verified 2026-05-17: PyPI ships 0.6.0; if your `sum` binary
# is 0.1.0, that's the friction described in the standing direction.)

# 2. Make sure the sieve extractor is wired:
python -m spacy download en_core_web_sm

# 3. Set the API key for LLM-axis renders (canonical-path doesn't need it):
export OPENAI_API_KEY=sk-...
```

If you'd rather not install anything, **the live demo at `https://sum-demo.ototao.workers.dev` covers most of scenario B without local setup** — see Scenario B below.

## Scenario A — Distill (writer→reader trust loop)

**The hypothesis:** *"If I distill three sources into a 2-page brief and ship it with its receipt, do I trust the receipt enough to publish without re-reading every source?"*

This is the journalist-writing-on-AI-policy use case. The wedge ICP from the deliberation artifact. If the answer is "yes I'd publish," the writer's tool wedge is real. If "no," what's missing is the next thing to build.

### Run

```bash
# 1. Get three sources you'd actually cite — research papers, news
#    articles, policy briefs. Save each as plain text:
#       src1.txt  src2.txt  src3.txt
#    (Copy-paste from PDFs is fine. Don't waste time on extraction
#    edge cases — the dogfood is the slider, not the PDF parser.)

# 2. Attest each — produce a signed CanonicalBundle per source:
sum attest --extractor=sieve < src1.txt > src1.bundle.json
sum attest --extractor=sieve < src2.txt > src2.bundle.json
sum attest --extractor=sieve < src3.txt > src3.bundle.json

# 3. Inspect what was extracted. Sieve is lemma-aware and conservative;
#    LLM extraction would catch more. Pick the right knob for your test.
cat src1.bundle.json | python -m json.tool | head -20

# 4. Compose the three bundles into a merged knowledge graph:
#    (T3 compose transform — LCM-merge of state integers; commutative,
#    associative, idempotent. See docs/TRANSFORM_REGISTRY.md.)
sum transform apply compose \
  --input '{"bundles":[<src1.bundle.json contents>,<src2>,<src3>]}' \
  --parameters '{}' \
  > merged.json

# 5. Render the merged bundle at a density slider position you'd ship.
#    Density 0.3-0.5 = aggressive distillation; 0.7 = comprehensive.
#    All other axes neutral (0.5) for canonical-path. Off-centre on
#    length/formality/audience/perspective triggers LLM dispatch.
sum transform apply slider \
  --input "$(cat merged.json | jq '.output')" \
  --parameters '{"density":0.5,"length":0.5,"formality":0.5,"audience":0.5,"perspective":0.5}' \
  > brief.json

# 6. Read the brief. Read the receipt. Decide if you trust it enough.
cat brief.json | python -m json.tool
```

### What to capture (the falsifiable part)

- **Time to first useful output** (target: under 5 minutes once setup done).
- **Trust level 1–10** in the resulting brief without re-reading sources.
- **One specific thing missing or wrong**.
- **Would you publish without re-checking?** (Y / N.)
- **If N, what's the smallest change that would flip it to Y?**

The last bullet is the highest-information artifact this doc can produce.

---

## Scenario B — Reshape (audience-slider test)

**The hypothesis:** *"If I take my own draft newsletter / article and re-render at three audience slider positions, does any variant beat the version I'd have written by hand?"*

The dream's bidirectional axis. If the reshape produces *better* writing than the original draft for any audience, that's a product. If it produces *worse* writing every time, the slider is craft-aid not craft-replacement, and the wedge shifts.

### Run via live demo (no install needed)

```bash
# Pick a paragraph from your own writing — published or draft, doesn't
# matter. ~100-200 words is a good length. Get the triples first via
# the live extractor or hand-author them, then render.

# Hand-authored triples (faster for a one-shot test):
TRIPLES='[["climate","is","warming"],["sea_level","rises","2050"]]'

# Variant 1 — novice audience, casual formality
curl -s -X POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -d "{\"triples\":$TRIPLES,\"slider_position\":{\"density\":1.0,\"length\":0.7,\"formality\":0.2,\"audience\":0.2,\"perspective\":0.5}}" \
  | jq -r '.tome'

# Variant 2 — expert audience, formal
curl -s -X POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -d "{\"triples\":$TRIPLES,\"slider_position\":{\"density\":1.0,\"length\":0.5,\"formality\":0.8,\"audience\":0.9,\"perspective\":0.5}}" \
  | jq -r '.tome'

# Variant 3 — your typical position
curl -s -X POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -d "{\"triples\":$TRIPLES,\"slider_position\":{\"density\":1.0,\"length\":0.5,\"formality\":0.5,\"audience\":0.5,\"perspective\":0.5}}" \
  | jq -r '.tome'
```

### What to capture

- Does any variant beat your hand-written version?
- Which slider axis felt like it had the most product-value? (Hint from the zenith framing: the answer might motivate Perspective Receipts — renaming the axes from stylistic to perspectival.)
- Did the receipt give you anything actionable?

---

## Scenario C — Verify a render someone else made

**The hypothesis:** *"If I were a reader receiving a published piece with a SUM receipt, would the verify step actually convince me?"*

The reader-trust loop. Already covered in the README's 60-second-verify section. Worth re-running with adversarial intent — try to convince yourself the receipt is meaningful, not theatre.

```bash
# From README's verify example — confirms the full trust loop:
curl -s "https://sum-demo.ototao.workers.dev/api/render" \
  -H 'content-type: application/json' \
  -d '{"triples":[["alice","born","1990"]],"slider_position":{"density":1.0,"length":0.5,"formality":0.5,"audience":0.5,"perspective":0.5}}' \
  | python3 -c "
import json, sys, urllib.request
from sum_engine_internal.transform_receipt import verify_transform_receipt
d = json.load(sys.stdin)
# Note: /api/render produces render_receipt, not transform_receipt;
# adapt accordingly. For transform receipts, hit /api/transform.
print(d.get('tome'))
print('receipt schema:', (d.get('render_receipt') or {}).get('schema'))
"
```

### What to capture

- Does the receipt feel like load-bearing trust or feel like decoration?
- If a journalist published with a receipt, would you (as a reader) actually verify it? Why or why not?
- What would have to be different for the receipt to be *automatically* useful (browser extension, link preview, in-line indicator)?

---

## After dogfooding — what to do with findings

Capture in your own notes. The next session this repo opens, paste the findings; future-me will:

1. Triage against `docs/ZENITH_FRAMING_2026-05-16.md` — many dogfood findings invoke Perspective Receipts (axis-renaming) or Epistemic Nutrition Label (per-artifact summary) or `sum verify --explain` (layered output).
2. Check whether the finding maps to an existing constraint in `docs/CHARTER_2026-05-17.md` §6 (and update the charter if so).
3. Decide whether to ship a small thing or write a finding into the deliberation tree.

**Dogfood is the first non-grant signal that earns substrate work back from the wait.** Without a dogfood finding, every "let's start building X" is auto-pivot.

## Known friction points (don't waste energy on these)

- **Stale install.** If your `sum` binary is below 0.6.0 (last verified 2026-05-17: PyPI ships 0.6.0), `sum transform apply` will fail with `ModuleNotFoundError`. `pip install --upgrade 'sum-engine[openai,sieve,receipt-verify]'`.
- **Sieve extraction is conservative.** It will miss triples that an LLM extractor would catch. For the distill scenario, that's a feature (deterministic, no LLM). For dogfood-on-real-writing, you may want LLM extraction — wire it via the env vars in `sum_engine_internal/ensemble/live_llm_adapter.py`.
- **The transform CLI's `--input` flag wants JSON.** Wrap your text input accordingly (`{"triples": [...]}` shape for slider; `{"text": "..."}` shape for extract).
- **The CLI's output isn't a polished publishing format.** It's JSON. For dogfood, that's fine — the experience is the loop, not the typography. If the loop works, the publishing-format polish is a downstream product decision.

## Pointers

- [`CHARTER_2026-05-17.md`](CHARTER_2026-05-17.md) — why dogfood is load-bearing.
- [`ZENITH_FRAMING_2026-05-16.md`](ZENITH_FRAMING_2026-05-16.md) — the three concepts most dogfood findings invoke.
- [`TRANSFORM_REGISTRY.md`](TRANSFORM_REGISTRY.md) — the three registered transforms (slider / extract / compose) the scenarios above use.
- [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) — what the slider's fact-preservation contract is, per axis.
- [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §5 — the four epistemic statuses your dogfood findings should be tagged with when captured.
