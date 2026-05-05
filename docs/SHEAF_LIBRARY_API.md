# Sheaf-Laplacian detector — library API

The `sum_engine_internal.research` package ships the sheaf-Laplacian
hallucination detector as an importable Python library. This doc
turns the previously-implementation-only modules into a **supported
library surface**: external code can import the detectors and call
them with the contracts pinned here.

The detector ships behind the `[research]` extras flag. Install with:

```bash
pip install 'sum-engine[research]'
```

This installs `numpy` and the sheaf-Laplacian modules. Without the
extras flag, `import sum_engine_internal.research.sheaf_laplacian_v32`
raises `ImportError`.

## Stability tier

This is a **research library surface**, not a production CLI:

- The function signatures, return shapes, and rule names are stable
  *within a major version*. Breaking changes ship as a new module
  (e.g. `sheaf_laplacian_v33`) rather than mutating an existing one.
- The empirical numbers (AUCs, γ_auto values, calibration constants)
  are pinned in `fixtures/bench_receipts/` with `bench_digest`
  reproducibility — the digests are a public claim that the
  numbers reproduce.
- Test contracts pinning the falsifiable predictions (H1–H20) live
  in `Tests/research/`. Library users should refer to these as
  the canonical "what does this function guarantee" reference.

The stability tier is one notch below `sum_engine_internal.render_receipt`
or the cross-runtime trust triangle (which carry stronger backward-
compat guarantees). Treat it accordingly.

## Quick start — score a render against a sheaf

```python
import numpy as np
from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    train_restriction_maps,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_from_weights,
    weights_from_receipts,
)
from sum_engine_internal.research.sheaf_laplacian_v32 import (
    combined_detector_score_v32,
)

# 1. Build a sheaf from your knowledge base.
source_triples = [
    ("alice", "graduated_from", "mit"),
    ("alice", "knows", "bob"),
    ("bob", "owns", "dog"),
]
trained, embeddings, _history = train_restriction_maps(
    source_triples,
    stalk_dim=8,
    epochs=200,
    learning_rate=0.005,
    margin=0.5,
    n_negatives_per_positive=3,
    seed=0,
)

# 2. Build per-edge weights from your trust signals (receipts,
#    issuer reputation, key revocation status, etc.).
trusted_edges = [t for t in source_triples if t[1] == "graduated_from"]
weights = weights_from_receipts(trained, trusted_edges=trusted_edges)
# weights is a numpy array of shape (|edges|,) with values in [0, 1]:
#   trusted_weight (default 1.0) for trusted edges
#   default_weight (default 0.1) for everything else
#   revoked_weight (default 0.0) for explicitly-revoked edges

# 3. Derive a boundary partition. Vertices incident only to high-
#    weight edges form the boundary; the rest is the interior.
boundary = boundary_from_weights(trained, weights, threshold=0.5)

# 4. Score a candidate render (a list of triples produced by the
#    LLM or other downstream system) against the trained sheaf.
candidate_render = [
    ("alice", "graduated_from", "mit"),
    ("alice", "knows", "bob"),
    ("bob", "owns", "dog"),
]
score = combined_detector_score_v32(
    trained,
    embeddings,
    candidate_render,
    weights,
    lambda_deficit=0.05,
    gamma_deviation=0.1,        # see "Choosing γ" below
    boundary_indices=boundary,
)

# 5. Inspect the score components.
print(score["v_combined_v32"])  # the headline detector score
print(score["v_laplacian_w"])   # weighted Laplacian quadratic form
print(score["deviation_w"])     # harmonic-extension deviation
print(score["v_deficit"])       # presence-deficit term
```

A render that's **consistent** with the trained sheaf produces a
*low* `v_combined_v32`. A render that **introduces an entity swap,
predicate flip, or fact drop** that the trust frame doesn't support
produces a *higher* score. The threshold is calibration-dependent;
see the corpus-scale validation receipt at
`fixtures/bench_receipts/v3_2_validation_2026-05-03.json` for the
threshold values used in the published F4/F5 verdicts.

## API surface — what each function does

### `train_restriction_maps(triples, stalk_dim, ...) → (sheaf, embeddings, history)`

Module: `sum_engine_internal.research.sheaf_laplacian_v2`

Trains the sheaf's restriction maps (edge-relation transformations)
and per-vertex embeddings against the input triple set. Pre-training
step — build a sheaf once, then score many renders against it.

Hyperparameters:

| Param | Type | Default | Notes |
|---|---|---|---|
| `triples` | `list[Triple]` | required | Source-of-truth triples |
| `stalk_dim` | `int` | 32 (v3.2 corpus uses 8) | Per-vertex embedding dim |
| `epochs` | `int` | 200 | Contrastive training epochs |
| `learning_rate` | `float` | 0.01 | SGD step size |
| `margin` | `float` | 1.0 | Triplet loss margin |
| `n_negatives_per_positive` | `int` | 3 | Negative samples per positive |
| `seed` | `int` | 0 | RNG seed (reproducibility) |

Returns `(KnowledgeSheafV2, np.ndarray of shape (|V|, stalk_dim), list[float])`
— sheaf, trained embeddings, training-loss history.

### `weights_from_receipts(sheaf, trusted_edges, revoked_edges, ...) → np.ndarray`

Module: `sum_engine_internal.research.sheaf_laplacian_v3`

Builds per-edge weights from the operator's trust signals. The
"signals" can be anything — verified Ed25519 signatures from a known
issuer, a reputation score from a downstream reviewer, a manual
allow-list — the function abstracts that to "trusted vs revoked
vs default."

| Param | Default | Meaning |
|---|---|---|
| `trusted_weight` | 1.0 | High weight for verified-trusted edges |
| `default_weight` | 0.1 | Floor for unsigned / unknown edges |
| `revoked_weight` | 0.0 | Zero weight for revoked-key signatures |

Returns a numpy array of shape `(|sheaf.edges|,)` with non-negative
weights (PSD requirement).

### `boundary_from_weights(sheaf, weights, threshold) → list[int]`

Module: `sum_engine_internal.research.sheaf_laplacian_v3`

Derives the boundary partition: vertices whose incident edges all
have weight ≥ `threshold`. Default threshold 0.5 separates trusted
neighbourhoods from the rest under the default weight contract
(trusted=1.0, default=0.1).

Returns a sorted list of vertex indices on the boundary.

### `combined_detector_score_v32(sheaf, embeddings, render, weights, ...) → dict`

Module: `sum_engine_internal.research.sheaf_laplacian_v32`

The v3.2 combined detector — strict generalization of v3 that adds
harmonic-extension deviation as a complementary signal:

```
v_combined_v32 = v_laplacian_w + γ · deviation_w + λ · v_deficit
```

| Param | Default | Meaning |
|---|---|---|
| `lambda_deficit` | 0.05 | Coefficient on presence-deficit term |
| `gamma_deviation` | 1.0 | Coefficient on harmonic-extension deviation |
| `boundary_indices` | `None` | Vertex indices forming the boundary B; `None` means empty B (graceful fall-back) |

**Choosing γ**. The empirical finding from the v3.2 validation
bench: on the seed_long_paragraphs corpus, the magnitude-matching
auto-calibration heuristic (γ_auto ≈ 1.0) is **wrong** —
deviation's signal-to-noise ratio is below what its magnitude
suggests. F5 (no regression vs v3) PASSES only at γ ≤ 0.1. **For
production use, start with `gamma_deviation=0.1` and tune via
your own labeled corpus.** With γ=0 v3.2 reduces to v3 numerically
(subsumption — H16); that's a safe default if you're not sure.

Returns a dict with these keys:

```python
{
    # v3 keys (carried through):
    "v_laplacian":          float,  # unweighted Laplacian quadratic form
    "v_deficit":            float,  # λ · presence_deficit²
    "v_combined":           float,  # v3 unweighted-laplacian combined score
    "v_laplacian_w":        float,  # weighted Laplacian quadratic form
    "v_combined_v3":        float,  # v3 combined score
    "edge_weights":         list[float],

    # v3.2 additions:
    "deviation_w":          float,  # ‖x_I_actual − x_I*‖² (weighted)
    "v_combined_v32":       float,  # v_laplacian_w + γ·deviation_w + v_deficit
    "boundary_indices":     list[int],
    "interior_size":        int,
    "gamma_deviation":      float,
}
```

### Sprint-7.5 additions — per-rendered-triple V channel + entity-set baselines + Borda fusion

These functions were added in the Sprint 7.5 hardening arc to recover the detector's competitive position vs trivial baselines. The cochain channel (`combined_detector_score_v32` above) is mathematically blind to entity-set-preserving perturbations (predicate-flip A2); the per-rendered-triple V channel restores that signal directly. Entity-set baselines + Borda fusion combine the two complementary signals. The complementary hybrid is the **published WIN** at trusted-mean AUC 0.876 (Δ=+0.043 vs B2 alone, `bench_digest dc6e0260…`).

#### `score_v32_with_per_triple(...) → float`

Module: `scripts.research.sheaf_per_triple_integration_experiment`

Adds the §3.5 per-rendered-triple V channel to the v3.2 cochain score:

```
v_combined = v_laplacian_w + γ·deviation_w + λ·v_deficit
           + α·max_in_vocab_v_triple + β·n_oov
```

| Param | Default | Meaning |
|---|---|---|
| `lambda_` | (caller) | Coefficient on presence-deficit term |
| `gamma` | (caller) | Coefficient on harmonic-extension deviation |
| `alpha` | 1.0 | Coefficient on max in-vocab per-triple V |
| `beta` | 1.0 | Coefficient on out-of-vocab triple count |
| `global_sheaf` | `None` | Trained sheaf for per-triple OOV detection (if `None`, uses doc-local sheaf which by construction has no OOV) |
| `global_embeddings` | `None` | Embeddings paired with `global_sheaf` |

Returns a single float — the additive-combined score.

The per-triple channel uses `score_rendered_triples_v2` from `sum_engine_internal.research.sheaf_laplacian_v2` to score each rendered triple individually under the trained restriction maps; OOV detection requires passing the GLOBAL trained sheaf (per-doc sheaves have no OOV by construction). Empirically lifts A2 trusted-mean AUC from 0.500 to 0.671 on `seed_long_paragraphs`.

#### `borda_fuse(scores_a, scores_b) → list[float]`

Module: `scripts.research.sheaf_hybrid_comparison`

Per-pool average-rank fusion of two detectors' scores. Given `n` paired scores, returns the rank-sum per index:

```
fused[i] = average_rank(scores_a[i] | scores_a) + average_rank(scores_b[i] | scores_b)
```

Ties get mean rank. Magnitude-invariant — works regardless of detector score scales. Parameter-free.

Used by the complementary-hybrid bench (`scripts/research/sheaf_complementary_hybrid_experiment.py`) to fuse `score_v32_with_per_triple` with `score_b2_jaccard_distance`. The fused score's per-cell AUC strictly beats either component on `seed_long_paragraphs`.

#### `score_b1_entity_presence_deficit(source_triples, rendered_triples) → float`
#### `score_b2_jaccard_distance(source_triples, rendered_triples) → float`

Module: `scripts.research.sheaf_baseline_comparison`

Two trivial reproducible baselines, scored on the same (clean, perturbed) triple-set pairs the v3.x detectors consume:

- **B1**: `1.0 − |source_entities ∩ rendered_entities| / |source_entities|`. Higher = more source entities missing from render.
- **B2**: `1.0 − |source_entities ∩ rendered_entities| / |source_entities ∪ rendered_entities|`. Symmetric variant of B1; penalises spurious entities too.

Both pure set ops on entity sets (predicates excluded). No floating-point, no LAPACK, no randomness — AUCs reproduce exactly across runs. B2 alone trusted-mean AUC = 0.833 on `seed_long_paragraphs` (catches A1/A4 at 1.000; blind to A2 at 0.500). The baselines exist as the **minimum-defensible reproducible comparison** for the cochain-channel detectors; LM-based baselines (sequence log-prob, MiniCheck-FT5) are deferred to v0.2.

#### Cross-machine `bench_digest` verification

Module: `scripts.research.cross_machine_verify_modal`

Modal app with two `@app.function`s sharing one `Image`: `verify_v32_validation_digest` and `verify_complementary_hybrid_digest`. Builds an image pinned to a specific commit SHA, installs `'.[research,sieve]'` extras, downloads `en_core_web_sm`, runs each bench and returns the digest plus environment metadata. Compares against operator-side digests; writes a `sum.cross_machine_verification.v1` receipt.

Run: `modal run scripts/research/cross_machine_verify_modal.py`. Requires `modal` CLI authenticated (`modal token new`). Cost ~$0.01 per full run on Modal's default CPU; container build amortizes per image hash.

## Falsifiable predictions pinned in tests

These are the contracts the v3.2 module's tests pin (`Tests/research/
test_sheaf_laplacian_v32.py` for example-based + `test_sheaf_laplacian_v32_property.py`
for Hypothesis-driven). Library users can rely on these:

- **H16 (subsumption).** `gamma_deviation = 0` → `v_combined_v32`
  byte-equals `v_combined_v3`. Setting γ=0 is the safe-default
  no-op upgrade path.
- **H17 (L_IB ≠ 0 visibility).** On a graph with cross-partition
  edges, `deviation_w` differs between clean and a boundary-only
  perturbation. The harmonic-extension signal is informative
  whenever the topology has boundary↔interior coupling.
- **H18 (F3 fall-back).** On a graph with `L_IB = 0` (the F3
  failure topology), `v_laplacian_w` still surfaces the
  perturbation — even when `deviation_w` is structurally blind,
  the combined score is informative.
- **H19 (no λ double-counting).** Doubling `lambda_deficit`
  doubles `v_deficit` only — does not double-multiply through
  the v3.2 wrapper.
- **H20 (degenerate-boundary fall-back).** Empty `B` or full
  `B` → `deviation_w = 0`; combined score reduces to v3.

## Falsifiable predictions about the *math* (Hypothesis property tests)

`Tests/research/test_sheaf_laplacian_v32_property.py` lifts H16/H17/H18
to universal-quantifier strength via Hypothesis-generated random
graphs and cochains:

- **Universal H17.** For any 3-vertex chain sheaf with v0 ∈ B and
  edges (v0→v1), (v1→v2), a non-trivial change to v0's stalk
  produces a non-zero Δdeviation_w. Tested across 30 random
  (stalk_dim, cochain, perturbation, weight) combinations.
- **Universal H18.** For any 4-vertex sheaf with B = {v0, v1},
  I = {v2, v3} and edges entirely within partitions, a non-
  trivial change to v0's stalk produces a non-zero Δv_laplacian_w.
- **Universal H16.** v3.2 with γ=0 equals v3 across 20 randomly
  weighted toy sheaves with random embeddings.

## Out of scope for the library

This library does **not** ship:

- **Production calibration tooling.** γ and λ calibration is the
  caller's responsibility. The v3.2 validation script
  (`scripts/research/sheaf_v3_2_validation.py`) is a *bench*, not a
  calibrator — it scans a fixed γ grid against a fixed corpus.
- **A decision threshold.** The detector returns a continuous score;
  the operator must pick a cutoff for their use case (precision/
  recall tradeoff). Bench AUCs are a guide but corpus-specific.
- **A2 predicate-flip detection.** All current detectors (v22, v3,
  v3.1, v3.2) score 0.500 on relation perturbations across the
  validation corpus. This is a known structural gap, named in
  `docs/SHEAF_HALLUCINATION_DETECTOR.md`. Future v3.3 work named
  in `docs/NEXT_SESSION_PLAYBOOK.md`.
- **A CLI surface.** The detector is library-only by design — it's
  research-tier, and exposing it via `sum` would imply a stronger
  stability tier than the substrate currently supports. CLI users
  who want hallucination detection on their own audit logs should
  call the library from their own Python process.

## Relationship to the rest of SUM

The detector consumes a knowledge source (triples or extracted
sheaf) and a candidate render (triples again, or one of the
canonical bundle / render-receipt artifacts). It is *orthogonal*
to the trust loop:

- **Trust loop** (`sum.render_receipt.v1`, JWKS, Ed25519): proves
  *who signed* a given render and *what bundle it derives from*.
- **Sheaf detector**: proves whether the render is *consistent*
  with the source-of-truth sheaf — independent of whether anyone
  signed it.

In a deployment where signed render receipts feed back into the
detector's `weights_from_receipts`, the two compose: the trust
loop tells you which edges to weight up; the detector tells you
whether the render makes sense given those weights.

## Pointers

- **Math + receipt-weighted theory:** `docs/SHEAF_HALLUCINATION_DETECTOR.md`
- **Bench & validation receipts:** `fixtures/bench_receipts/v3_*.json`
- **Module sources:** `sum_engine_internal/research/sheaf_laplacian_v{2,3,32}.py`
- **Test contracts:** `Tests/research/test_sheaf_laplacian_v{2,3,32}{,_property}.py`
- **Spec doc with the F3 STRUCTURAL FAIL finding and v3.2 closure:**
  `docs/SHEAF_HALLUCINATION_DETECTOR.md` §3.4.4
