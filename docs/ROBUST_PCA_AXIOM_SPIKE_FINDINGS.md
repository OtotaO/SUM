# Robust PCA axiom-corruption spike — Phase A findings

The deep-research article's §9.1 ranked Robust PCA on axioms as
the highest-leverage move not currently in SUM, with a [provable]
kernel (Candès, Li, Ma & Wright, JACM 58(3):11, 2011), a labeled
failing test case (`seed_v2 doc_015`), and a 1-2-day cost
estimate. This is the spike.

## What landed

  - `sum_engine_internal/research/robust_pca/pcp.py` —
    Principal Component Pursuit via ADMM (Lin et al. arXiv:1009.5055,
    2010, Algorithm 5). ~140 lines including doc + edge cases.
    Default λ = 1/√(max(n,d)) and μ₀ = 1.25/‖M‖₂ from the
    literature.
  - `sum_engine_internal/research/robust_pca/axiom_embedding.py` —
    Deterministic sha256-bucketed one-hot embedding of
    `Triple(s, p, o)` into ℝ^{3·B}. No ML dependency.
  - `scripts/research/robust_pca_axiom_spike.py` — Two-experiment
    measurement harness emitting `sum.robust_pca_axiom_spike.v1`
    receipt.
  - `Tests/test_robust_pca.py` — 18 contract tests covering math
    correctness, edge cases, and embedding determinism.

## Experiment 1 — Synthetic ground-truth recovery (PROVABLE)

Build M = L_true + S_true with known low-rank L_true (rank r) and
sparse S_true (5 % entry density, magnitude 5). PCP must recover
both exactly. Receipt:
`fixtures/bench_receipts/robust_pca_axiom_spike_*.json`.

| n   | d   | r_true | r_recov | L_rel_err | S_rel_err | iters |
|----:|----:|-------:|--------:|----------:|----------:|------:|
|  50 |  80 |      3 |       3 |    0.0000 |    0.0000 |    22 |
| 200 | 200 |      5 |       5 |    0.0000 |    0.0000 |    17 |
| 500 | 500 |     10 |      10 |    0.0000 |    0.0000 |    16 |

  - **All three workload sizes recover exact rank** (no over- or
    under-estimation of L's complexity).
  - **L_rel_err = S_rel_err = 0** to numerical precision (≤ 10⁻⁶
    tolerance pinned by `test_pcp_recovers_L_to_high_precision`
    and `test_pcp_recovers_S_to_high_precision`).
  - **Convergence in 16-22 iterations** with the Lin et al. 2010
    initial μ. Wrong μ initialisation converges to a "fits-but-
    doesn't-separate" high-rank L; the Lin choice is load-bearing
    and is documented in the docstring + pinned by tests.

This is the **[provable]** core kernel from Candès et al. 2011
verified on our implementation. PCP works.

## Experiment 2 — Real-corpus corruption detection (HONEST NEGATIVE
RESULT for the simplest setup)

Take the deterministic-sieve-extracted axioms from a real corpus
(`seed_long_paragraphs`, 120 axioms), inject 12 corrupted axioms in
two classes:
  - **Type (a)** — off-corpus entities: `Triple("junk_X", "glorp_X", "frobozz_X")`
  - **Type (b)** — miswired triples: real corpus entities in random
    (s, p, o) combinations

Run PCP on two embeddings × two ranking strategies:

| corpus               | embedding         | strategy     | precision | recall |
|----------------------|-------------------|-------------:|----------:|-------:|
| seed_long_paragraphs | hashed_buckets_64 | top-K score  |     0.08  |  0.08  |
| seed_long_paragraphs | hashed_buckets_64 | top-K \|anomaly\| | 0.08  |  0.08  |
| seed_long_paragraphs | corpus_vocab      | top-K score  |     0.08  |  0.08  |
| **seed_long_paragraphs** | **corpus_vocab**  | **top-K \|anomaly\|** | **0.50** | **0.50** |
| seed_news_briefs     | corpus_vocab      | top-K \|anomaly\| |  0.58  |  0.58  |

Random baseline is 12/132 ≈ 0.09. The corpus-vocab embedding +
bidirectional-anomaly ranking lifts precision to **0.50-0.58** —
meaningful signal but well below the article's stated success
target (precision ≥ 0.95, recall ≥ 0.80).

### What this tells us

The article's "smallest experiment" framing assumed RPCA on a
single-extraction axiom matrix would discriminate corruption out
of the box. **It does not, with the simplest deterministic
embeddings.** Two reasons surface from the measurements:

1. **Single-extraction lacks consensus signal.** The article's
   alternative phrasing ("across re-extractions of the same
   document under varied prompts/seeds") would build a matrix
   where rows are extractions of one document and PCP's L₀ is
   the *consensus* of axioms appearing in many extractions.
   Single-extraction-with-injected-corruption doesn't have this
   structure — every "axiom" appears exactly once, in one row.
2. **Sparse one-hot embeddings give PCP no manifold to recover.**
   With ~3 ones in a 192-dim row, the matrix is uniformly sparse;
   neither L nor S has obvious structure for PCP to extract. The
   corpus-vocab embedding is denser and works better, but a
   single hot column per role still doesn't capture
   corruption-relevant features (e.g., "this entity-predicate
   pairing makes no sense in this corpus").

### Why ship this anyway

The math kernel is **[provable]** and verified — that's the
load-bearing part. The application result is honest data the
substrate didn't have before this PR. Future iterations can
build on a known-correct foundation; alternative iterations can
revisit the embedding without re-litigating the math.

## Path forward (iteration 2+)

Three concrete directions in priority order:

  1. **Multi-extraction matrix.** Generate K extractions per
     document (varying LLM seeds, prompts, or — cheaper —
     applying small text perturbations through the deterministic
     sieve). Rows = (document × extraction); columns = unique
     (s, p, o). PCP separates consensus axioms (L) from rare
     extraction-specific noise (S). This is the article's
     alternative framing and the closer match to its 0.95
     precision target.
  2. **Learned axiom embedding.** Replace the one-hot/vocab
     embedding with a small sentence-encoder embedding of
     `f"{s} {p} {o}"`. Trades [empirical] for [empirical with a
     larger dependency], may give PCP a richer manifold. Risk:
     introduces an ML dependency in the load-bearing path
     (PROOF_BOUNDARY discipline applies).
  3. **Reframe the failure class.** RPCA is a strong tool for
     *consensus-vs-outlier* problems. If single-extraction
     corruption isn't its sweet spot, the substrate may need a
     different mechanism (e.g., entity-resolution drift via
     persistent homology — article §9.3 — or SMT-backed
     consistency checking — substantive-math agent's #1 rec).

The core math (`sum_engine_internal/research/robust_pca/`) stays
in place across iterations. Whichever embedding or framing wins,
PCP is the convex solver under it.

## Honest tier table

| component                                | tier        |
|------------------------------------------|-------------|
| PCP/ADMM convergence + recovery          | [provable]  |
| ‖M − L − S‖_F → 0 in 16-22 iters         | [certified] |
| Synthetic exact recovery on (n, d, r)    | [empirical] |
| Corpus corruption detection P=0.50-0.58  | [empirical, scope-limited] |
| RPCA at substrate scale for `seed_v2 doc_015` | [open question — needs iteration 2+] |
