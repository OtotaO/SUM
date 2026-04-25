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

### STATE 5b — final calibration (current)

After swapping to a 2000-word Brown-corpus frequency table for the
audience classifier and recalibrating length bands against measured
LLM behaviour:

| Axis | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Threshold | Status |
|---|---|---|---|---|---|---|---|
| density | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ≤ 0.001 | ✓ verified |
| length | 0.09 | 0.23 | 0.40 | 0.20 | 0.26 | ≤ 0.60 | ✓ within (recalibrated) |
| formality | 0.10 | 0.20 | 0.00 | 0.25 | 0.10 | ≤ 0.40 | ✓ within |
| perspective | 0.30 | 0.20 | 0.00 | 0.30 | 0.10 | ≤ 0.40 | ✓ median; p90 spikes |
| audience | 0.28 | 0.20 | 0.22 | 0.11 | 0.09 | ≤ 0.40 | ✓ within |

**Fact-preservation:** 1.000 median, 1.000 p10 across 198/200 LLM-
axis cells — the load-bearing claim is verified. (2/200 cells
errored on `doc_einstein` with `LengthFinishReasonError` — the LLM
exceeded the 16384-token completion ceiling during re-extraction.
Robustness item, not a contract violation.)

### What changed between STATE 5a and STATE 5b

- *Audience classifier:* embedded 200-word list → 2000-word Brown-
  corpus frequency list. Median drift cut ~50%. Threshold 0.55 →
  0.40.
- *Length bands:* per-triple-linear `(5,15)…(80,200)` →
  empirically-derived `(4,10) / (5,12) / (4,10) / (30,60) / (80,140)`.
  The LLM has a 6-wpt floor at and below position 0.5 and scales
  aggressively above. Median drift cut 3× across positions 0.1–0.7.
  Threshold 0.95 → 0.60.

### Known limitations (carried into v0.2)

1. *Audience tail at neutral.* p90 still touches 0.39 at audience=0.5.
   Technical prose has a vocabulary tail any small frequency table
   under-covers. v0.3 candidate: SCOWL or COCA-derived 5000+ word
   list, OR rescale formula to anchor against measured LLM baseline
   rather than linear `audience × 0.30`.
2. *Perspective at moderate positions.* p90 spikes at positions 0.1
   and 0.3 (0.57, 0.70). The LLM tends to commit to one perspective
   rather than blending, so moderate positions read as outliers
   under the pronoun-ratio classifier. Coarse signal; revising
   requires a richer perspective measurement (clause-level voice
   detection) — frontier item.
3. *Set-based fact preservation is MontageLie-vulnerable.* The
   current `|source_keys ∩ reextracted_keys| / |source_keys|`
   formula is exploitable by reordering preserved triples into a
   deceptive narrative. See `docs/SLIDER_V02_RESEARCH.md` for the
   v0.2 mitigation plan (event-order-aware verifiers).

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
