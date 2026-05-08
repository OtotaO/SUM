# Recursive compression and the SUM

The first measured pass at the operator's original-vision claim:
*the SUM is the incompressible point of a corpus, beyond which
compression causes degradation.* This document reports what we
measured, what compresses cleanly, what doesn't, and where the
boundary actually sits in this substrate.

Receipts:

  - Deterministic walk:
    [`fixtures/bench_receipts/recursive_compression_walk_deterministic_2026-05-08.json`](../fixtures/bench_receipts/recursive_compression_walk_deterministic_2026-05-08.json)
    — `bench_digest 778cd95d…d85090`
  - LLM-mediated walk:
    [`fixtures/bench_receipts/recursive_compression_walk_llm_gpt-4o-mini-2024-07-18_2026-05-08.json`](../fixtures/bench_receipts/recursive_compression_walk_llm_gpt-4o-mini-2024-07-18_2026-05-08.json)
    — `bench_digest 6d779aa4…edeb26`
  - LLM-arm capture-once snapshot:
    [`fixtures/bench_renders/recursive_walk_<corpus>_gpt-4o-mini-2024-07-18.json`](../fixtures/bench_renders/)

Schema for both walks: `sum.recursive_compression_walk.v1`.

## The walk, formally

Given source prose $T_0$ and a render operator $R$ (compressor):

$$
A_k = \text{sieve}(R(A_{k-1})), \qquad A_0 = \text{sieve}(T_0)
$$

The walk halts when $\text{signature}(A_k)$ has been seen before — a
fixed point or short cycle. Two metrics:

  - $\text{recall}_k = \dfrac{|A_k \cap A_0|}{|A_0|}$ — fraction of
    *original* axioms still present at step $k$.
  - $|A_k|$ — axiom count at step $k$.

The **SUM at threshold $\tau$** is the smallest $A_k$ along the walk
satisfying $\text{recall}_k \ge \tau$. The **incompressible point**
is the fixed-point axiom-set; whether it preserves the original facts
depends on $R$.

## Two compressors, two findings

### Arm 1 — Deterministic canonical tome

$R$ = `AutoregressiveTomeGenerator.generate_canonical`. Renders each
axiom as a bare-lemma sentence ("The alice like cat."). Empirical
result on the two corpora:

| Corpus | $n$ | median fp step | median fp $n_\text{axioms}$ | median fp recall | docs collapsed to ∅ |
|---|---:|---:|---:|---:|---:|
| `seed_long_paragraphs` | 16 | 3 | 3 | 0.333 | 0 / 16 |
| `seed_news_briefs` | 16 | 2 | 1 | 0.225 | 5 / 16 |

The deterministic canonical tome's lemmatized output is *only
partially sieve-extractable*. The walk does iterate non-trivially —
some grammatical scaffolding from the markdown structure (`## Subject`
chapters) survives sieve re-extraction. But recall vs the original
axiom-set drops fast: the median doc retains only ~22-33% of its
original facts at the fixed point.

**This is a substantive finding the project did not have before.**
The "lossless round-trip" claim in PROOF_BOUNDARY §1.1 holds at the
*state-integer* level (parse(canonical_tome(S)) decodes to S's
state) but does NOT hold at the *sieve* level
(sieve(canonical_tome(S)) ≠ S's triples in general). The two
representations of S — the canonical tome and the triple-set — are
*not mutually inverse* under the sieve. The asymmetry is invisible
at the substrate's verifier layer; this walk surfaces it.

### Arm 2 — LLM-mediated grammatical render

$R$ = call to `gpt-4o-mini-2024-07-18` via
`llm_dispatch.get_adapter`. Render axioms as simple declarative
sentences with proper noun forms and verb conjugation. Sieve
re-extracts cleanly.

| Corpus | $n$ | median fp step | median fp $n_\text{axioms}$ | median fp recall | docs collapsed to ∅ |
|---|---:|---:|---:|---:|---:|
| `seed_long_paragraphs` | 16 | 2.5 | 7 | 0.59 | 0 / 16 |
| `seed_news_briefs` | 16 | 2 | 4 | **0.80** | 0 / 16 |

The LLM walk preserves substantially more of the original axiom-set
across iterations. **News-brief docs converge with 80% median recall
at a 4-axiom fixed point — most facts survive iteration.**

## The "SUM" identified per-doc

For each doc and threshold $\tau \in \{0.5, 0.7, 0.9, 0.99\}$, the
receipt records the smallest fixed-point-reachable axiom-set
satisfying $\text{recall} \ge \tau$. Three categories of doc emerged:

**Category A — at-fixed-point already.** The original axiom-set is
*its own SUM* under the LLM compressor. Examples:
`doc_news_amazon_fire`, `doc_news_apple_event` — recall stays at
1.000 across the walk. The prose was already maximally compressed
relative to this round-trip; no further iteration is informative.

**Category B — gradual compression to a stable subset.** Recall
decreases monotonically over 1-2 steps and then stabilises. Example:
`doc_long_cryptography` — n=9 → 8 → 8 (stable), recall 1.0 → 0.89 →
0.78. The SUM at $\tau=0.7$ is the 8-axiom subset reached at step 1.

**Category C — sharp drop then stable.** Recall plummets after step 1
and stays flat. Example: `doc_long_climate_change` — n=7 → 5 → 5,
recall 1.0 → 0.14 → 0.14. The LLM's paraphrase of these axioms
re-extracts to mostly-different triples — the predicate or object
canonicalisation is sufficiently different that *technically* most
original axioms drop out, but a different axiom-set of similar size
emerges. The fixed-point exists but its recall against the original
is low. **At $\tau=0.5$, no SUM exists for this doc** (no step
satisfies the threshold beyond step 0).

## Implications for the operator's original vision

The vision: *hand SUM a library, get back its succinct summary; the
incompressible point after which degradation is experienced.*
Three measured contributions toward this:

1. **The walk machinery exists and produces measurable curves.** Per
   doc, per compressor, per threshold, you can identify a SUM (or
   establish that none exists below the threshold). Receipt-pinned,
   reproducible.

2. **The SUM is corpus- and doc-dependent.** News briefs preserve
   more axioms across LLM iteration than encyclopedic prose;
   short-form factual claims (events, dates, named entities)
   compress more cleanly than definitional prose. *Different content
   types have different SUMs.*

3. **The substrate has a sieve↔canonical-tome asymmetry that
   matters.** The deterministic round-trip *parses back to the same
   state-integer* (PROOF_BOUNDARY §1.1) but does *not* sieve back to
   the same triples. The deterministic canonical tome is a viable
   compressor at the *state-integer* level only. Recursive
   compression at the *triple* level requires LLM-mediated
   grammatical rendering — confirmed empirically and named in the
   receipt.

## What this is not

This is not yet:

- **Multi-modal compression dispatch** (axioms vs parables vs
  poetry vs quotes vs emoji). The current arm renders all axiom
  types as simple declarative sentences — appropriate for factual
  claims but not for moral/aesthetic content. v0.5+ candidate.
- **Cross-LLM-family generalisation.** Only gpt-4o-mini was tested.
  The §4.7.x cross-family pattern says we should expect different
  models to produce different fixed-point sets; the absolute SUM
  size and recall trajectory may be model-dependent. v0.5+
  candidate (and trivial extension via the existing dispatcher).
- **Library-scale measurement.** Two 16-doc corpora is a starting
  measurement. Per the §4.9 performance-characterisation envelope,
  walks at ~1k axioms (book-length) are feasible at ~0.6 s per
  iteration — a 5-step walk on a real book takes ~3 seconds. *The
  recursive-compression vision is computationally feasible up to
  medium-book corpora today; library-scale is gated on Phase 26.*
- **An importance-weighted SUM.** The current "SUM at $\tau$"
  selects by `n_axioms`, breaking ties arbitrarily. A real
  importance-weighted SUM (which axioms are *most informative* per
  unit count?) requires either an axiom-importance scorer or
  Kolmogorov-style description-length search. v0.6+ research arc.

## Reproducing

```bash
# Deterministic arm — instant, no API calls
python -m scripts.research.recursive_compression_walk \
    --compressor deterministic \
    --corpora seed_long_paragraphs seed_news_briefs

# LLM arm — Phase 1 capture (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python -m scripts.research.recursive_compression_walk \
    --compressor llm \
    --model gpt-4o-mini-2024-07-18 \
    --corpora seed_long_paragraphs seed_news_briefs

# LLM arm — Phase 2 replay (no API calls; reads cached snapshots)
python -m scripts.research.recursive_compression_walk \
    --compressor llm \
    --model gpt-4o-mini-2024-07-18 \
    --corpora seed_long_paragraphs seed_news_briefs
```

The capture-once-replay-forever architecture is identical to
Path 2 (§4.7.x): the per-step LLM responses are committed to
`fixtures/bench_renders/recursive_walk_<corpus>_<model>.json` and
re-running the walk against the cached snapshot produces the same
`bench_digest` byte-for-byte.

## Honest scope

- One LLM (gpt-4o-mini); two corpora (16 docs each); one compressor
  per arm. The §4.7.4.1 lesson — *small-n thresholded measurements
  produce extremal-Goodhart artifacts* — applies here too. The
  fixed-point recall numbers (0.59 / 0.80 medians) are point
  estimates with corpus-sample variance; 5-corpora extension would
  tighten them.
- The "incompressible point" identified here is a fixed point of
  *this particular* compressor (LLM-mediated grammatical render).
  Different operators (parable-style, poetry-style, importance-
  weighted) would identify different points. This is the
  beginning of a measurement program, not its conclusion.
- The deterministic-arm finding (sieve↔canonical-tome asymmetry)
  is itself a load-bearing result and should not be presented as a
  bug — it surfaces an architectural asymmetry the project should
  document. Both rendering paths have valid uses (canonical tome
  for state-integer round-trip; LLM tome for sieve round-trip).
