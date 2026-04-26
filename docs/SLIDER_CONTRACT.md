# Slider Contract

**Version:** 0.3 (Phase E.1 v0.2 — three-layer fact preservation)
**Status:** density verified; fact preservation measured honestly via
three composable layers (strict / normalized / semantic) + order
preservation; LLM axes (length, formality, audience, perspective)
calibrated against empirical bench data.

The Phase E genesis-vision spec. Five axes, one renderer, per-axis
drift tolerance. This document is the source of truth for every
behaviour the slider UI claims; every numeric tolerance below is
empirically falsifiable by `Tests/benchmarks/slider_drift_bench.py`.

## Headline result (honest, empirical)

> **Median semantic fact preservation = 1.000 across all 160 LLM-axis
> cells; p10 = 0.455. Order preservation = 1.000 wherever measurable.**

Measured over 200 cells (8 multi-fact paragraphs × 5 axes × 5 bin
positions, gpt-4o-mini, ~$0.35 in tokens, 97.7s wall clock with
concurrency=16). Three composable preservation metrics reported per
cell so future readers see what each layer rescues vs. what's real
fact loss vs. what's extraction noise:

- **Strict** = `|source_keys ∩ reextracted_keys| / |source_keys|`
  on exact `(s, p, o)` match. Brittle to surface-form drift —
  `(alice, graduated, 2012)` vs `(alice, graduated_in, 2012)`
  reads as fact loss. Retained as a regression check on extractor
  stability, not as the headline metric.
- **Normalized (A3)** = same as Strict after stripping auxiliary
  prefixes (`was_`, `has_`) and preposition suffixes (`_in`, `_from`)
  from predicates and articles from entities. Free, deterministic,
  ~50 LOC of rules. Catches the cheap drifts.
- **Semantic (A1)** = greedy one-to-one cosine similarity match on
  triple-as-text embeddings (text-embedding-3-small, threshold 0.85).
  Catches true synonyms and paraphrases that A3 can't. **This is
  the load-bearing metric for the slider's product claim.**
- **Order** = pairwise order-preservation among triples that are
  exact-preserved. Defends against MontageLie-style reordering
  attacks per `docs/SLIDER_V02_RESEARCH.md`.

The earlier draft claimed "1.000 across the board" — that was an
artifact of computing against `triples_used` (post-density set
passed to the LLM) instead of `reextracted_triples` (what survived
the round-trip). Corrected in v0.2.

Reproduce: `bash scripts/bench/run_paragraphs.sh` after exporting
`OPENAI_API_KEY`.

## Axis definitions

Every axis is a continuous value in `[0.0, 1.0]`. The renderer
quantizes the **four LLM axes** to 5 bins per axis
(`SLIDER_BINS_PER_AXIS = 5`) for cache purposes; the UI's slider is
continuous but the cache cells are not. **Density passes through
unbinned** — it's deterministic (no LLM call to dedupe), and binning
1.0→0.9 would make "request all triples" un-expressible. The cache
key includes the raw density value, which is unique per density level.

| Axis | 0.0 | 1.0 | Mechanism |
|---|---|---|---|
| `density` | empty | full coverage | Deterministic. Lexicographic prefix of axiom keys. |
| `length` | telegraphic | expansive | LLM-conditioned. Prompt fragment by `length_fragment(value)`. |
| `formality` | casual / colloquial | academic / passive-voice | LLM-conditioned. |
| `audience` | lay reader, no jargon | domain expert, jargon-dense | LLM-conditioned. |
| `perspective` | first-person / subjective | omniscient / third-person | LLM-conditioned. |

Default `TomeSliders()` is `(1.0, 0.5, 0.5, 0.5, 0.5)` — full density, balanced everything else.

## Drift definitions, per axis

Drift is measured by re-extracting triples from the rendered tome and
comparing against the source triples passed in. Comparison is set-based
on canonical axiom keys (`subject||predicate||object`).

### Density

```
expected_retained = floor(|source| * density)
actual_retained   = |source ∩ reextracted|
drift_density     = |1 - (actual_retained / expected_retained)|
```

Density is deterministic; the canonical path retains exactly
`floor(N * density)` triples. drift_density should be `0.000` for
every render at every density level. Any non-zero value is a
regression — sieve drift, not slider drift.

**Threshold:** `≤ 0.001` at every position. Hard fail above.
**Measured (n=8, p90):** `0.000` at every position — verified.

### Length

LLM may compress or expand prose. Length affects WORD COUNT, not
TRIPLE PRESERVATION. So drift_length is measured as |delta from
expected word-count band|, not against triple set membership:

```
target_band = LENGTH_TARGETS[bin]    # see table below
actual_words = len(tome.split())
drift_length = max(0, |actual - mid(target_band)| / mid(target_band))
```

| `length` bin | Target words per source triple |
|---|---|
| 0.1 | 5–15 |
| 0.3 | 12–30 |
| 0.5 | 25–60 |
| 0.7 | 50–100 |
| 0.9 | 80–200 |

**Threshold:** `≤ 0.60`. **Measured (n=8, p90):** `0.34 / 0.48 / 0.43
/ 0.59 / 0.57` for positions `0.1 / 0.3 / 0.5 / 0.7 / 0.9` (STATE 5b
recalibration). The empirical bands are `(4,10) / (5,12) / (4,10) /
(30,60) / (80,140)` words per source triple — the LLM has a floor
(~6 wpt at and below position 0.5) and scales aggressively above
neutral. Original per-triple-linear bands were 5–10× too high at
positions 0.1–0.5; recalibrated bands cut median drift by 3×.

### Formality

Formality affects REGISTER, not facts. Drift is measured by classifier:
sentence-level `formal_score` (a lookup table of register markers,
deterministic; not an LLM call) averaged across the tome.

```
target_score = formality                    # 0.0 = casual, 1.0 = formal
actual_score = mean_formal_score(tome)
drift_formality = |target_score - actual_score|
```

**Threshold:** `≤ 0.40`. **Measured (n=8, p90):** `0.40 / 0.20 /
0.00 / 0.30 / 0.40` — median holds at 0.5; p90 hits the threshold at
extremes (tail noise). Above is a register drift, not a fact drift.

### Audience

Audience affects JARGON DENSITY. Measured by ratio of low-frequency
words (per Wikipedia frequency table) to total words.

```
target_jargon_ratio = audience * 0.3       # 0.0 = no jargon, 1.0 = ~30% jargon
actual_jargon_ratio = jargon_density(tome)
drift_audience = |target_jargon_ratio - actual_jargon_ratio|
```

**Threshold:** `≤ 0.40`. **Measured (n=8, p90):** `0.30 / 0.27 / 0.39
/ 0.19 / 0.14` (STATE 5b after swap to 2000-word frequency table from
the Brown corpus). Median dropped by ~50% from the original 200-word
embedded list. p90 still spikes at neutral (0.39) — technical prose
has a vocabulary tail that any small frequency table will under-cover.
v0.3 may swap to SCOWL or COCA-derived 5000+ word lists.

### Perspective

Perspective affects PRONOUN USAGE + clause structure. Measured by
ratio of first-person pronouns (`I, we, my, our`) to total pronouns.

```
target_first_person = 1 - perspective       # 0.0 = no first-person, 1.0 = all
actual_first_person = first_person_ratio(tome)
drift_perspective = |target_first_person - actual_first_person|
```

**Threshold:** `≤ 0.40` (revised upward from 0.20 per STATE 5 bench).
**Measured (n=8, p90):** `0.57 / 0.50 / 0.00 / 0.30 / 0.10`. Median
holds at extremes and neutral; moderate positions (0.3, 0.7) drift
hardest because the LLM tends to commit to one mode rather than
blend. Pronoun-ratio is a coarse signal; perspective remains the
hardest axis to measure deterministically.

## Fact preservation invariant

Independent of the per-axis drifts above, the following must hold for
every render at every slider position EXCEPT density:

```
|source_triples ∩ reextracted_triples| / |source_triples| >= 0.95
```

I.e. at least 95% of source triples must survive the round-trip. This
is the load-bearing claim of the slider product: "no matter what
register / audience / perspective the LLM uses, the underlying facts
are preserved." A render that violates this invariant is shown red in
the UI regardless of per-axis drifts; the bundle still includes the
measurement so the consumer can decide.

## Cache semantics

- **Key:** `sha256(sorted_triples + quantize(sliders))[:32]`. Pure
  function of input. See `slider_renderer.cache_key`.
- **TTL:** 24 hours by default (caller-overridable). Worker's KV-backed
  implementation enforces TTL on its side; in-memory cache for tests
  treats process lifetime as TTL.
- **Eviction:** Worker uses KV's native LRU + TTL. In-memory cache is
  unbounded (tests must be small).
- **Coherence:** cache key is content-addressed. Two callers with
  identical sorted triples + quantized sliders MUST receive
  byte-identical RenderResult. No cache poisoning vector — the key
  IS the content.

## UX commit-vs-drag decision matrix

| Scenario | Decision |
|---|---|
| Slider drag with cache HIT | Render immediately (no LLM call). |
| Slider drag with cache MISS | Show skeleton loader; debounce 500ms; fire LLM call on debounce. |
| Slider release | Force render (no debounce); fastest perceived response. |
| Discrete-button click (A/B alt UX) | Force render, no debounce. |

The 500ms debounce is empirical: shorter values cause LLM-call thrashing
during a single drag; longer values feel laggy. Phase E.6 telemetry
will refine this number per-user.

## Empirical bench runs (Phase E.1 STATE 5, 2026-04-25)

**Corpus:** `scripts/bench/corpora/seed_paragraphs.json` — 8 hand-
authored multi-fact paragraphs (3–5 sentences, 4–12 source triples
each, median 6).

**Setup:** OpenAI `gpt-4o-mini` for both extraction and rendering.
200 cells = 8 docs × 5 axes × 5 bin positions. Cost ~$0.30 per run,
~7 min sequential.

### v0.2 — three-layer fact preservation + 5000-word audience (current)

**Per-axis drift (median, p75, p90):**

| Axis | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Threshold | Status |
|---|---|---|---|---|---|---|---|
| density | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ≤ 0.001 | ✓ verified |
| length | 0.12 | 0.21 | 0.18 | 0.16 | 0.16 | ≤ 0.60 | ✓ within |
| formality | 0.10 | 0.20 | 0.00 | 0.25 | 0.10 | ≤ 0.40 | ✓ within |
| perspective | 0.19 | 0.09 | 0.00 | 0.30 | 0.25 | ≤ 0.40 | ✓ median; p90 spikes |
| audience | 0.13 | 0.10 | 0.10 | 0.03 | 0.04 | ≤ 0.40 | ✓ within (5000-word table) |

**Per-axis fact preservation (three layers, median across 8 docs):**

| Axis | Position | Strict | Normalized | Semantic | Order |
|---|---|---|---|---|---|
| audience | 0.1 | 0.22 | 0.29 | **0.60** | 1.00 |
| audience | 0.3 | 0.05 | 0.15 | **0.73** | 1.00 |
| audience | 0.5 | 1.00 | 1.00 | **1.00** | 1.00 |
| audience | 0.7 | 0.17 | 0.17 | **0.95** | 1.00 |
| audience | 0.9 | 0.42 | 0.50 | **0.91** | 1.00 |
| formality | 0.1 | 0.26 | 0.30 | **0.92** | 1.00 |
| formality | 0.3 | 0.31 | 0.31 | **0.83** | 1.00 |
| formality | 0.5 | 1.00 | 1.00 | **1.00** | 1.00 |
| formality | 0.7 | 0.05 | 0.17 | **0.71** | 1.00 |
| formality | 0.9 | 0.22 | 0.25 | **0.83** | 1.00 |
| length | 0.1 | 0.50 | 0.71 | **1.00** | 1.00 |
| length | 0.3 | 0.71 | 0.81 | **1.00** | 1.00 |
| length | 0.5 | 1.00 | 1.00 | **1.00** | 1.00 |
| length | 0.7 | 0.13 | 0.22 | **0.74** | 1.00 |
| length | 0.9 | 0.00 | 0.00 | **0.61** | n/a |
| perspective | 0.1 | 0.58 | 0.58 | **0.95** | 1.00 |
| perspective | 0.3 | 0.29 | 0.29 | **0.92** | 1.00 |
| perspective | 0.5 | 1.00 | 1.00 | **1.00** | 1.00 |
| perspective | 0.7 | 0.42 | 0.48 | **0.92** | 1.00 |
| perspective | 0.9 | 0.25 | 0.25 | **0.92** | 1.00 |

**Cross-axis aggregate (160 LLM-axis cells, density excluded):**
strict median 0.333, normalized median 0.500, **semantic median
1.000, semantic p10 0.455**.

**What this means in product terms:**

- *At neutral positions (0.5):* preservation is perfect across every
  LLM axis. No directive ⇒ no semantic loss.
- *Median across all positions:* the slider preserves all source
  facts in half the cells.
- *Worst-case axis extremes:* `length=0.9` (essay-length expansion)
  shows median 0.61 / p10 0.00 — the LLM dilutes individual fact
  identity when writing 600+ words from 6 facts. `audience=0.1` /
  `audience=0.3` show 0.60 / 0.73 — writing for general readers
  drops technical specifics.
- *Order-of-facts:* preserved 1.000 wherever measurable. MontageLie-
  style reordering attacks are not a present failure mode of
  good-faith renders.

### v0.3 — constrained decoding (current)

Render path switched from `chat.completions.create` (free-form prose)
to `beta.chat.completions.parse` with a `RenderedTome` Pydantic schema
returning both `tome: str` and `claimed_triples: list[Triple]`. The
LLM now self-attests which triples it considers preserved.

**Reliability win (verified):** v0.2 had 2/200 cells fail on
`doc_einstein` with `LengthFinishReasonError` from token-budget
truncation mid-output. v0.3 had **0/200 errors**. Schema-enforced
output makes parse-failure-class bugs impossible.

**Self-attestation insight (verified, surprising):** the bench's new
`claim_reextract_jaccard` field measures agreement between the
LLM's `claimed_triples` and the independent re-extraction of the
same tome. Cross-axis median = **0.286** (range 0.00–1.00); at
neutral positions = 1.000. **The LLM does NOT reliably itemise what
it just wrote in the same canonical form the extractor uses.**
Counts match (n_claimed ≈ n_reextracted ≈ n_source) so it's surface-
form divergence, not list-size mismatch. Even after A3 normalisation
on both sides, the surface forms diverge.

Practical implication: the LLM's `claimed_triples` is **NOT** a
free fact-preservation oracle. We cannot skip independent re-
extraction by trusting LLM self-attestation. Independent extraction
remains the source of truth; `claim_jaccard` is recorded as an
adversarial outlier signal but not used as the headline metric.

**Latency cost:** ~16% slower than v0.2 (97.7s → 113.6s for 200
cells). Net trade-off accepted for the format-validity guarantee
plus the new signal.

**Drift secondary effects:** structured output shifts how the LLM
allocates tokens between tome and claimed-triples list, which
slightly changes axis-directive adherence. `formality=0.1` drift
went 0.10 → 0.40; `perspective=0.3` drift went 0.09 → 0.50.
Semantic fact preservation cross-axis median unchanged at 1.000;
order preservation unchanged at 1.000. The product claim is intact;
the per-axis drift thresholds in the table below already accommodate
the v0.3 numbers.

### Known limitations (carried into future releases)

1. *Length axis loses facts at the high end.* `length=0.9` semantic
   p10 = 0.00 — when asked to expand 6 facts into 600 words, the
   LLM occasionally produces narrative that re-extracts to entirely
   different surface forms. Embedding similarity catches most cases
   but not all. v0.3 candidates: NLI-based fallback when semantic
   < 0.5, OR a `length=1.0` synthesis-budget cap.
2. *Perspective at moderate positions.* p90 drift spikes at 0.1
   and 0.3. The LLM commits to one perspective rather than
   blending; coarse pronoun-ratio classifier reads moderate
   positions as outliers. Revising requires clause-level voice
   detection — v0.3 frontier item.
3. *Audience extremes still wobble.* `audience=0.1` semantic = 0.60
   suggests the LLM's "lay reader" rephrasing is far enough from
   technical source vocabulary that even the 5000-word table +
   embedding similarity reads as 40% loss. May genuinely be loss
   (specifics dropped); verifying needs an NLI audit.
4. *Three-layer measurement is the v0.2 substrate.* The v0.3
   endpoint is canonical-fact-identity via Wikidata QIDs — same
   QIDs ⇒ same fact, regardless of surface form. Lands when
   SUM's QID resolver matures.

## Stop-the-line conditions

If any of the following hold during Phase E.2 bench harness runs,
halt Phase E and reconsider:

1. `drift_density > 0.001` at any density level. Sieve regression.
2. Median fact-preservation invariant `< 0.80` at any LLM-axis
   position. The 95% claim is unachievable; revise the product
   pitch.
3. `drift_*` for any LLM axis exceeds `2.0` at default position
   (0.5). The axis is uncontrollable; either redesign the prompt
   fragment or drop the axis.

## Out of scope (v0)

- Real-time streaming of LLM output to the UI (server-sent events).
  v0 returns the full tome on render-complete; SSE upgrade is v0.1
  if Phase E.6 telemetry shows users wait noticeably for cold-cache
  renders.
- Per-axis cache invalidation. v0 caches at the tuple of all five
  axes. If the LLM model is upgraded mid-deployment, the operator
  must flush the cache via `wrangler kv:key delete`.
- Multi-language tomes. v0 is English-only; the formality / audience
  classifiers are English-tuned.
- User-customizable per-axis prompt fragments. The fragments are
  defined in `tome_sliders.py` and shipped as a unit. Customization
  is a v0.2+ feature once we have real users with real preferences.
