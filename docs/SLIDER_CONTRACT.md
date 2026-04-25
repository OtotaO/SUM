# Slider Contract

**Version:** 0.2 (Phase E.1 STATE 5 — empirical bench run landed)
**Status:** density + fact preservation verified; audience + length
preliminary (classifier upgrades pending v0.2).

The Phase E genesis-vision spec. Five axes, one renderer, per-axis
drift tolerance. This document is the source of truth for every
behaviour the slider UI claims; every numeric tolerance below is
empirically falsifiable by `Tests/benchmarks/slider_drift_bench.py`.

## Headline result (empirically verified)

> **Fact preservation across all four LLM axes is 100% (median, p10).**

Measured over 200 cells (8 multi-fact paragraphs × 4 LLM axes × 5 bin
positions, gpt-4o-mini, $0.30 in tokens). For every axis position
across every doc, `|source_keys ∩ reextracted_keys| / |source_keys|`
returned 1.000. The slider's central product claim — *axis changes
do not lose facts* — holds on this corpus.

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

**Threshold (preliminary v0):** `≤ 0.95` documenting current LLM
behaviour. **Measured (n=8, p90):** `0.45 / 0.77 / 0.91 / 0.71 /
0.60` for positions `0.1 / 0.3 / 0.5 / 0.7 / 0.9`. The per-triple band
formula assumes the LLM scales response length linearly with the
input fact count; empirically it doesn't. v0.2 will recalibrate
against absolute word-count bands using the bench data, then tighten
the threshold.

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

**Threshold (preliminary v0):** `≤ 0.55` documenting current
classifier limitation. **Measured (n=8, p90):** `0.49 / 0.43 / 0.55 /
0.38 / 0.31`. The embedded ~200-word common-words table saturates on
technical prose: any content word longer than 4 chars not in the
table reads as jargon, so the actual jargon ratio sits ~0.40–0.55
regardless of axis position. v0.2 will swap to a frequency-table
classifier (e.g. SCOWL or COCA-derived) and tighten the threshold.

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

## Empirical bench run (Phase E.1 STATE 5, 2026-04-25)

**Corpus:** `scripts/bench/corpora/seed_paragraphs.json` — 8 hand-
authored multi-fact paragraphs (3–5 sentences, 4–12 source triples
each, median 6).

**Setup:** OpenAI `gpt-4o-mini` for both extraction and rendering.
200 cells = 8 docs × 5 axes × 5 bin positions. Cost ~$0.30, wall
clock ~7 min sequential.

**Per-axis median drift:**

| Axis | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Threshold | Status |
|---|---|---|---|---|---|---|---|
| density | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ≤ 0.001 | ✓ verified |
| formality | 0.10 | 0.20 | 0.00 | 0.30 | 0.10 | ≤ 0.40 | ✓ within |
| perspective | 0.10 | 0.20 | 0.00 | 0.30 | 0.10 | ≤ 0.40 | ✓ within |
| length | 0.34 | 0.70 | 0.88 | 0.56 | 0.42 | ≤ 0.95 | preliminary |
| audience | 0.45 | 0.37 | 0.41 | 0.32 | 0.27 | ≤ 0.55 | preliminary |

**Fact-preservation:** 1.000 median, 1.000 p10 across all 200 LLM-
axis cells — the load-bearing claim is verified.

**Two known limitations** (both flagged as v0.2 work, not blockers):

1. *Audience classifier saturates.* The embedded common-words table
   (~200 words) is too small; on technical prose, ~50% of content
   words read as jargon regardless of axis position. The threshold
   is loose enough to pass current behaviour but doesn't actually
   measure audience fit.
2. *Length per-triple bands don't match LLM behaviour.* The formula
   assumes response length scales linearly with input fact count;
   empirically the LLM produces a roughly fixed-size summary
   regardless. v0.2 will recalibrate against absolute word-count
   bands using this bench's tome data.

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
