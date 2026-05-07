# Sheaf-Laplacian hallucination detector — research direction

> *"The legitimate frameworks already give us 80% of what we need; what is
> missing is not more terminology but rigorous integration, implementation,
> and experiment."* — from the SCT synthesis note (2026-05-01)

This document specifies the **first concrete research artifact** SUM
will ship that grounds the project inside the peer-reviewed
categorical-AI conversation: a sheaf-Laplacian-based hallucination
score over signed render receipts. Status: **research direction** —
specification, plan, and bounded claims; **not** a guarantee of working
performance until the v1 prototype lands and benchmarks separate.

---

## 1. Executive summary

`sum render` produces signed receipts for LLM-conditioned tomes.
Each receipt binds *(triples, slider position, model, output)* under
Ed25519 over JCS-canonical bytes. Three runtimes verify these
receipts byte-for-byte. What no SUM surface measures yet: **whether
the rendered tome's content is internally consistent across paraphrase
or rendering variation** — i.e. whether re-rendering the same triple
set under different conditions produces outputs whose re-extracted
triple sets *glue*, in the precise mathematical sense of agreeing on
overlaps.

A consistency obstruction in this gluing is, structurally, what
hallucination *is*: a locally-plausible output that fails to be
globally consistent with its source axioms or with sibling renderings
of those same axioms.

The hallucination detector takes a SUM bundle plus N renderings of
the same triples (varying slider positions, paraphrase prompts, or
model selections) and computes a **scalar consistency score**: the
sheaf-Laplacian quadratic form on the cochain induced by the
re-extracted-triples cover. Low score ↔ the rendering set glues;
high score ↔ at least one rendering disagrees with siblings under
the relation's restriction-map structure ↔ hallucination signal.

The detector is grounded in Gebhart, Hansen & Schrater (2023,
AISTATS) and the sheaf-Laplacian theory of Hansen & Ghrist (2019).
It does **not** claim to solve hallucination; it claims to make a
specific, mathematically clean obstruction-class signal computable
on SUM's existing receipt-bearing outputs.

---

## 2. Theoretical foundation

### 2.1 Cellular sheaves on knowledge graphs

Following Curry (2014) and Gebhart, Hansen & Schrater (2023):

**Definition (cellular sheaf, Gebhart et al. Def. 4).**
A cellular sheaf $\mathcal{F}$ on a directed graph $G = (V, E)$
consists of:

- a vector space $\mathcal{F}(v)$ for each vertex $v \in V$,
- a vector space $\mathcal{F}(e)$ for each edge $e \in E$,
- linear restriction maps $\mathcal{F}_{v \trianglelefteq_h e} :
  \mathcal{F}(v) \to \mathcal{F}(e)$ for each
  head-incidence pair $(v, e)$ with $h(e) = v$,
- linear restriction maps $\mathcal{F}_{v \trianglelefteq_t e} :
  \mathcal{F}(v) \to \mathcal{F}(e)$ for each
  tail-incidence pair $(v, e)$ with $t(e) = v$.

**0-cochain space.** $C^0(G; \mathcal{F}) = \prod_{v\in V} \mathcal{F}(v)$ —
one entity-embedding per vertex.

**1-cochain space.** $C^1(G; \mathcal{F}) = \prod_{e\in E} \mathcal{F}(e)$.

**Coboundary operator.** $\delta : C^0 \to C^1$. For an edge $e: u \to v$,
$$(\delta x)_e = \mathcal{F}_{v \trianglelefteq e}\, x_v
- \mathcal{F}_{u \trianglelefteq e}\, x_u.$$

**Global sections.** $H^0(G; \mathcal{F}) = \ker(\delta)$ — cochains
satisfying *every* restriction-map equality across edges
simultaneously. A global section is the strict gluing condition.

### 2.2 The sheaf Laplacian as continuous consistency measure

From Hansen & Ghrist (2019) and reproduced as Equation 1 of
Gebhart et al. (2023):

$$L_{\mathcal{F}} := \delta^T \delta, \qquad
x^T L_{\mathcal{F}} x = \sum_{e=u\sim v \in E}
\|\mathcal{F}_{u \trianglelefteq e}\, x_u
- \mathcal{F}_{v \trianglelefteq e}\, x_v\|^2.$$

Properties:

- $x^T L_{\mathcal{F}} x \geq 0$ for all $x \in C^0(G; \mathcal{F})$.
- $x^T L_{\mathcal{F}} x = 0$ iff $x \in H^0(G; \mathcal{F})$, i.e.
  $x$ is a global section.
- Quadratic in $x$, sums to a per-edge contribution
  $\|\mathcal{F}_{u \trianglelefteq e}\, x_u
  - \mathcal{F}_{v \trianglelefteq e}\, x_v\|^2$ — which is exactly
  the Structured Embedding scoring function (Bordes et al. 2011).

This gives us a continuous, differentiable, decomposable consistency
score on any 0-cochain over any cellular sheaf on a knowledge graph.
**This is the math we will use.**

### 2.3 SUM-to-Knowledge-Sheaves mapping

Mechanical correspondence between SUM primitives and the Knowledge
Sheaves framework. Not aspirational — these are direct identifications.

| SUM primitive | Knowledge-Sheaves equivalent |
|---|---|
| Triple set $T = \{(s_i, p_i, o_i)\}$ in a CanonicalBundle | A knowledge graph $G_T$ with $V = \{$entities$\}$, $E = \{$predicates$\}$, schema $\mathcal{Q}$ a single-type directed multigraph (matching FB15k-237 / NELL-995 convention) |
| Per-entity prime $p_{\text{entity}}$ minted in `sha256_64_v1` | A 1-dimensional "presence" stalk $\mathcal{F}_{1d}(v) = \mathbb{R}$ with the prime acting as a content-address label |
| State integer $S = \mathrm{LCM}(p_{\text{axioms}})$ | The characteristic Yoneda token for the support set of axioms ; not a sheaf section per se but a content-addressed digest of *which* axioms are active |
| Cross-runtime byte-identity (K1–K4 + A1–A6) | A descent / gluing condition under a covering family $\{\text{Python}, \text{Node}, \text{Browser}\}$, both for accept and adversarial-reject classes |
| `render_receipt.v1` $(\text{triples\_hash}, \text{tome\_hash}, \text{sliders\_quantized}, \text{kid}, \text{signed\_at})$ | A signed witness that *one specific* 0-cochain $x_n$ on $G_T$ was the one minted by issuer $\text{kid}$ at time $\text{signed\_at}$ under slider position $\text{sliders\_quantized}$ |
| 5-axis slider | A parameterization of a (non-strictly) compositional rendering functor; honest qualification: not provably a strong-monoidal DisCoCat functor, but the right shape |

The detector lives downstream of these primitives: it consumes
receipts and re-extracted-triples cochains; it does not modify the
attestation or rendering layers.

---

## 3. The artifact

### 3.1 Inputs

- **Source bundle** $B$: a SUM CanonicalBundle for a fact-set,
  carrying $T$, $S$, $\text{state\_integer}$.
- **Render manifold** $\{R_n\}_{n=1}^N$: a set of `sum render`
  outputs derived from $B$. Variability axes:
  - slider position (different LLM-conditioned axes)
  - paraphrase prompt (different surface forms of "render these
    facts")
  - model selection (different LLM backends; ablation only when
    receipts are issued by trusted issuers)
- **Re-extraction function** $\Phi : \text{tome} \to T'$ (sieve or
  constrained-extractor LLM with vocab pinning per §2.5 of
  PROOF_BOUNDARY).

### 3.2 Procedure (v1, 1-dim presence stalks)

1. Build $G_T = (V, E)$ from the source bundle's triple set. $V$ =
   subjects ∪ objects; $E$ = predicate-labelled edges.
2. Choose stalks $\mathcal{F}_{1d}$: $\mathcal{F}(v) = \mathbb{R}$,
   $\mathcal{F}(e) = \mathbb{R}$. Restriction maps
   $\mathcal{F}_{h \trianglelefteq_h r} = \mathcal{F}_{t \trianglelefteq_t r} = 1$
   (identity on $\mathbb{R}$).
3. For each rendering $R_n$:
   a. extract triples $T'_n = \Phi(R_n)$,
   b. construct a 0-cochain $x_n \in C^0(G_T; \mathcal{F}_{1d})$
      with $x_n(v) = 1$ iff $v$ appears in any triple of $T'_n$, else 0.
4. Compute $V_n = x_n^T L_{\mathcal{F}} x_n$ for each $n$.
5. **Detector score** (variance-based): $\bar{V} =
   \frac{1}{N}\sum_n V_n$ and $\sigma_V^2 = \frac{1}{N}\sum_n (V_n -
   \bar{V})^2$. The pair $(\bar{V}, \sigma_V)$ is the consistency
   profile of the rendering manifold.
6. **Decision threshold** $\tau$ (calibrated): label as
   *hallucination-suspect* if $\bar{V} > \tau_\mu$ or $\sigma_V >
   \tau_\sigma$.

The 1-dim presence version is intentionally crude — it asks "do all
renders mention all entities consistently?" That's a low bar but
catches gross hallucination (entity dropped or substituted entirely).

### 3.3 Procedure (v2 — split into v2.0 / v2.1 / v2.2 after Hansen-Ghrist 2019 reading on 2026-05-01)

The original §3.3 sketched v2 as "swap stalk_dim from 1 to 384 with
identity restriction maps." Reading Hansen & Ghrist (2019),
*Toward a Spectral Theory of Cellular Sheaves* (arXiv:1808.01513,
JACT 2019) §3.2 surfaced that this is **structurally insufficient**:

  With identity restriction maps and per-vertex semantic embeddings,
  the global-section condition becomes ``x_v = x_u for every edge
  (u, e, v)`` — i.e., entities connected by *any* relation should
  have *identical* embeddings. That's a wrong constraint: an
  entity's embedding shouldn't equal another's just because they're
  connected by a predicate. ``V`` would be uniformly large on every
  render, the per-render *variance* would be the only signal, and
  the detector collapses back toward something v1-shaped.

What *actually* makes ``d > 1`` stalks meaningful (per Gebhart 2023
§4): **per-relation learned restriction maps**, trained via the
contrastive sheaf-embedding loss (Gebhart Def. 11, Eq. 4) so that
``F_{s ⊵ r} x_h = F_{t ⊵ r} x_t`` holds *exactly* on known triples
``(h, r, t)`` and *fails* on negative samples. With those learned
maps, the Laplacian quadratic form measures: "does the rendered
cochain agree under the learned per-relation projections?"

The honest split:

#### v2.0 — d-dim stalks with identity restriction maps (sanity baseline only)

  Stalks: $\mathcal{F}(v) = \mathbb{R}^d$, $\mathcal{F}(e) =
  \mathbb{R}^d$. Restriction maps: identity. Cochain $x_n[v]$:
  one-hot or compact-encoded presence indicator embedded in
  $\mathbb{R}^d$.

  This is *structurally equivalent to v1 lifted into $\mathbb{R}^d$*
  — it confirms the math machinery works at $d > 1$ but does not
  address the v1 blindspots (predicate-flip, off-graph fabrication,
  empty-render false-negative, disconnected-graph density-dropout).

  **Implementation: skip as a research artifact; ship only as a
  smoke test that the d>1 quadratic form numerically matches the
  d=1 case when the cochain is presence-equivalent.** No separate
  bench, no publishable claim, no separate `v2.0` versus `v1` ROC.

#### v2.1 — learned restriction maps (the real meaningful step)

  Stalks: $\mathcal{F}(v) = \mathbb{R}^d$ for $d \in \{8, 32, 64\}$
  initially (parameter count dominated by per-relation restriction
  maps, scales as $|\mathcal{R}| \cdot d^2$).

  Restriction maps: $\mathcal{F}_{s \trianglelefteq r}$ for each
  relation $r$, *trained* per Gebhart Def. 11 / Eq. 4:
  $\gamma$-gapped contrastive loss with positive triples from the
  source bundle and negative triples from a local-closed-world-
  assumption sample.

  Cochain $x_n[v]$: still presence-style for v2.1 (one-hot in
  $\mathbb{R}^d$), but now the per-relation restriction maps mean
  the per-edge residual $F_{s \trianglelefteq r} x_h - F_{t
  \trianglelefteq r} x_t$ encodes *relation-aware* disagreement
  rather than just naked entity mismatch.

  **What v2.1 with presence-style cochains addresses (with empirical
  results, all measurements 2026-05-01):**
  - Predicate-flip (A2): **VERIFIED** ✓. Per-rendered-triple V
    via ``score_rendered_triple_v2``: trained restriction maps
    distinguish (alice, knows, bob) at V = 0.0163 from the
    predicate-flipped (alice, owns, bob) at V = 2.0338 — a
    **~125× ratio**. Three additional clean / flipped pairs
    measured: ratios 9×, 40×, 9×. Even though the LCWA negative
    sampler in the current training loop perturbs only the tail
    (not the predicate), the Laplacian's per-relation structure
    suffices to amplify predicate-flip residuals strongly.
    Pinned in
    ``test_a2_predicate_flip_caught_with_meaningful_margin``
    (margin > 0.05 on the smallest pair).
  - Off-graph fabrication (A3): **VERIFIED** ✓ structurally.
    A rendered triple with an out-of-vocabulary relation or
    entity surfaces ``oov_signal=True`` from
    ``score_rendered_triple_v2`` before any V is computed — no
    statistical inference needed. Pinned in
    ``test_a3_off_graph_fabrication_via_oov_relation_caught``
    and ``test_a3_off_graph_fabrication_via_oov_entity_caught``.
  - **Disconnected-graph density-dropout: NOT closed by v2.1
    with presence-style cochains.** Empirically (the v2.1 test
    suite, ``test_v2_1_does_NOT_close_disconnected_graph_blindspot_with_presence_cochains``):
    trained 4-fact disconnected source, clean V = 0.4377,
    dropout V = 0.3270, margin = -0.1108 (dropout *lower* than
    clean, the wrong direction). Why: when a render drops a
    whole component, BOTH endpoints of the dropped edge zero
    out in the cochain, the trained restriction maps multiply
    by zero on both sides, the per-edge residual vanishes —
    same structural issue as v1. v2.2's semantic-context-window
    cochains are the proposed fix.
  - **Empty-render false-negative: NOT closed by v2.1 with
    presence-style cochains.** Same structural reason —
    all-zero cochain gives V = 0 regardless of restriction
    maps. v2.2's semantic-context-window cochains close this.

  Three candidate cochain redesigns for v2.2 — analytically
  evaluated 2026-05-01:

    (a) **Anti-cochain** (FALSIFIED analytically): $x_n[v] =
        +\text{trained\_emb}[v]$ if $v$ mentioned, $-\text{trained\_emb}[v]$
        if missing. For an edge with both endpoints missing,
        the residual becomes $F_h(-\text{emb}_h) - F_t(-\text{emb}_t)
        = -(F_h \text{emb}_h - F_t \text{emb}_t)$ — *same magnitude*
        as the positive case. V is unchanged. Anti-cochain
        does NOT close the blindspot.
    (b) **Semantic-context cochain:** $x_n[v] = \text{embed}
        (\text{context}(v, R_n))$ via sentence-transformer.
        Catches semantic drift in *present* mentions; missing
        entities still give zero, so disconnected-graph dropout
        still falsifies. v2.3 problem (sentence-transformer
        dep), not v2.2.
    (c) **Combined detector** ✓ (implemented as v2.2,
        ``combined_detector_score`` in
        ``sum_engine_internal/research/sheaf_laplacian_v2.py``):
        $V_{\text{total}} = \|\delta x\|^2 + \lambda \cdot
        (\text{presence\_deficit})^2$ where presence_deficit is
        the count of source vertices missing from the render.
        The Laplacian term carries the relation-aware signal
        (catches A2 / A3 after training); the deficit term
        carries the presence-pattern signal (catches density-
        dropout, including on disconnected source graphs which
        the Laplacian alone cannot detect by design). The two
        terms are **orthogonal** — combining them is the
        publishable v2.2 artifact, not a workaround.
        $\lambda = 0.05$ default, calibrated on the
        v2.1-falsification 4-fact disconnected-graph data
        (clean V = 0.438; dropout V = 0.327 + $\lambda \cdot 4$
        = 0.527; correct sign).

  **The deeper finding (added 2026-05-01) underlying (c):** the
  Laplacian quadratic form $x^T L_F x = \|\delta x\|^2$ is
  *fundamentally* a measure of cross-edge agreement under
  restriction maps, not entity presence. It cannot detect "facts
  missing entirely" by design — that's a separate problem with
  a separate fix (the deficit term). v1 / v2.1's
  disconnected-graph blindspot was a *category mismatch*: trying
  to use a relation-agreement signal to detect entity-dropout.
  v2.2's combined detector resolves this by running both signals
  in parallel.

  Training data: SUM already has ``seed_v1`` (50 docs) and
  ``seed_v2`` (20 docs) corpora. No new data to acquire.

  Compute: parameter-light at $d = 32$ (~$|\mathcal{R}| \cdot
  10^3$ parameters); CPU-only. No external API spend.

#### v2.2 — semantic-embedding cochains (the publishable step)

  As v2.1 except $x_n[v] = \mathrm{embed}(\text{context}(v, R_n))$
  via a sentence-transformer (e.g. `all-MiniLM-L6-v2`, $d = 384$).
  $\text{context}$ extracts a window around $v$'s mention in $R_n$.

  This addresses the case the spec originally framed v2 around:
  *paraphrase variation that preserves entity presence but drifts
  semantically* (e.g., real-data paraphrase 3's `python_code` /
  `python` divergence; verbose paraphrasing that introduces
  side-claims).

  v2.2 is the artifact the arXiv note is written around. v2.1 is
  the prerequisite that makes the math meaningful; v2.2 is the
  artifact that scales to naturalistic prose.

### 3.4 Procedure (v3, receipt-weighted) — IMPLEMENTED 2026-05-02

The weighted sheaf Laplacian (Hansen-Ghrist 2019 §3.2 weighted
generalization):

$$L_F^w \;=\; \delta^T W \delta, \qquad
  x^T L_F^w x \;=\; \sum_e w_e \cdot \|F_h^{(r)} x_u - F_t^{(r)} x_v\|^2$$

where $W$ is a non-negative diagonal $|E| \times |E|$ matrix of
edge weights. Each edge's weight is a function of whether that
edge's source-of-record carries a verified Ed25519-signed render
receipt:

| Receipt status | Weight |
|---|---|
| Signed by a key in the trusted-issuer JWKS | $w_{\text{trusted}} = 1.0$ |
| Unsigned / unknown issuer | $w_{\text{default}} = 0.1$ |
| Signed by a revoked key (per `/.well-known/revoked-kids.json`) | $w_{\text{revoked}} = 0.0$ |

The math claim from §3.2 carries through: $W^{1/2}\delta$ is a
coboundary of a sheaf with scaled stalks, so
$L_F^w = (W^{1/2}\delta)^T(W^{1/2}\delta)$ remains symmetric PSD.
Implementation: `sum_engine_internal.research.sheaf_laplacian_v3`.

**Why this is fractal in the architectural sense the project
calls out:** the weights come from the system's own trust
artifacts. The cross-runtime trust triangle (K1–K4) attests that
a receipt's Ed25519 signature is byte-identically verifiable in
Python, Node, and the browser. v3 takes those receipts and feeds
them into the detector's confidence weighting. The audit-log
substrate (PR #117) records every render's receipt KID, so
backfilling weights from a logged history is straightforward.
Higher trust → higher weight → sharper detection signal in regions
the system already verifies; unsigned regions get a lower-weight
floor that doesn't silence them entirely.

**Falsifiable predictions pinned in code:**

  - **H1 (linearity).** $V$ is linear in the weights — doubling all
    weights doubles $V$; setting one edge's weight to 0 zeros that
    edge's contribution exactly. Pinned at
    `test_h1_doubling_weights_doubles_quadratic_form` and
    `test_h1_zero_weight_kills_edge_contribution`.
  - **H2 (v2 reduction).** Uniform weights $w_e = c$ give
    $V_{v3}(x; w=c) = c \cdot V_{v2}(x)$. v3 is a strict
    generalization of v2. Pinned at
    `test_h2_uniform_weights_v3_equals_scaled_v2`.
  - **H3 (per-edge weighting).** The localization ranker's per-edge
    contribution scales with weight ($w_e \cdot \|residual_e\|^2$).
    Pinned at `test_h3_per_edge_contribution_scales_with_weight`.
  - **H4 (trust amplifies signal).** Tampering a trusted edge yields
    a sharper $\Delta V$ than tampering an untrusted edge — that is,
    receipt-weighting amplifies signal where the system already
    trusts. **This is the utility claim**; if it inverts, v3 is
    well-defined but useless. Pinned at
    `test_tampering_trusted_edge_yields_sharper_v_jump_than_untrusted`.
  - **H5 (revocation overrides trust).** An edge in both the
    trusted and revoked sets resolves to revoked. Pinned at
    `test_weights_from_receipts_revocation_overrides_trust`.

**Out of scope (v3 limits, named honestly):**

  - **Harmonic-extension boundary inference.** Implemented as v3.1
    on 2026-05-02; see §3.4.1 below. v3 only weights the quadratic
    form; v3.1 adds the boundary/interior partition and the
    most-consistent interpolation on the interior.
  - **JWKS verification round-trip.** v3's `weights_from_receipts`
    takes the trusted-edge set as a parameter; mapping receipts →
    JWKS-verified-edges is the caller's responsibility (a thin
    wrapper around the existing render-receipt verifier suffices).
  - **Empirical bench.** The synthetic-data utility test (H4) is
    pinned, but a corpus-scale bench (analogous to the v2.2 ROC
    bench in PR #114) is a follow-up. The math + falsifiability +
    utility pin is the v3 PR's scope.

v3 connects the obstruction class to SUM's existing trust-and-
verification primitives — the cross-runtime trust triangle, the
render receipts, the audit log, the JWKS / revoked-kids surface.
This is the SUM-specific extension that doesn't replicate
elsewhere: no other system has cross-runtime-verified render
receipts to feed into a sheaf-Laplacian's edge weights.

### 3.4.1 Harmonic extension over a (boundary, interior) partition (v3.1) — IMPLEMENTED 2026-05-02

Hansen-Ghrist 2019, Proposition 4.1 / Theorem 4.5: given a sheaf
$F$ on $G$, a partition $V = B \cup I$, and a cochain $x_B$
specified on the boundary $B$, the **harmonic extension** is the
unique cochain $x \in C^0(G; F)$ that

  (i)  agrees with $x_B$ on $B$
  (ii) minimizes $\|\delta x\|^2$ over the interior $I$.

Block-decompose the Laplacian by the $(B, I)$ partition:

$$L_F = \begin{bmatrix} L_{BB} & L_{BI} \\ L_{IB} & L_{II} \end{bmatrix}$$

Setting $\partial \|\delta x\|^2 / \partial x_I = 0$ gives the
closed-form interior cochain

$$x_I^* = -L_{II}^{-1} L_{IB} \, x_B$$

(when $L_{II}$ is invertible; v3.1's implementation uses
`np.linalg.lstsq` so a rank-deficient $L_{II}$ — disconnected
interior, or interior with a global section — yields the
minimum-norm solution rather than crashing).

For SUM: trusted-receipt-backed vertices (those whose every
incident edge is signed by a known-issuer JWKS key) form the
boundary $B$; the rest fall to the interior $I$. Given a render's
cochain $x$, restrict it to the boundary, compute the harmonic
extension on the interior, and compare:

$$\text{deviation}(x) = \|x_I - x_I^*\|^2$$

A render whose interior matches the harmonic extension is
*consistent with the trust frame*; one that diverges is flagged.

**Falsifiable predictions pinned in code:**

  - **H6 (boundary preservation).** `harmonic_extension` returns
    only the interior; reconstructing the full cochain from
    $(x_B, x_I^*)$ preserves $x_B$ byte-identically on $B$. Pinned
    at `test_harmonic_extension_agrees_with_x_B_on_boundary_by_construction`.
  - **H7 (minimization — the defining property).** No perturbation
    of the interior cochain (off the harmonic extension) gives a
    smaller $V$. Pinned at
    `test_harmonic_extension_minimizes_v_subject_to_boundary_constraint`.
  - **H8 (uniqueness when $L_{II}$ has full rank).** Two calls
    with the same boundary cochain yield byte-identical interior.
    Pinned at `test_harmonic_extension_unique_when_L_II_invertible`.
  - **H9 (degenerate full-boundary).** When every vertex is on the
    boundary, the interior is empty — function returns a `(0, d)`
    array, not a crash. Pinned at
    `test_harmonic_extension_full_boundary_returns_empty_interior`.
  - **H10/H11 (defensive boundary).** Invalid indices raise
    `ValueError`; wrong $x_B$ shape raises `ValueError`. Pinned at
    `test_harmonic_extension_rejects_invalid_boundary_indices` and
    `test_harmonic_extension_rejects_wrong_x_B_shape`.
  - **H12 (utility — the headline claim).** Tampering an interior
    vertex (boundary held fixed) increases the deviation. This is
    the hallucination-detection use case. Pinned at
    `test_boundary_deviation_detects_interior_tampering`.
  - **H13 ($v_{\text{ext}} \le v_{\text{actual}}$).** By the
    minimization property, the Laplacian quadratic form at the
    harmonic-extended cochain is no larger than at the actual
    cochain. Pinned at
    `test_boundary_deviation_v_at_extension_is_minimum`.
  - **H14 (chain-topology weight invariance — surfaced 2026-05-02).**
    With a *single* bridge edge connecting boundary to interior
    (chain topology), the harmonic extension is weight-invariant
    even on a trained sheaf. Analytic reason: $x_I = -r \cdot
    M(r)^{-1} (B x_B)$ with $r = w_{\text{bridge}}/w_{\text{interior}}$
    cancels for rank-1 $B$. The weight effect IS visible with
    multiple bridge edges. Both pinned: chain-invariance at
    `test_boundary_deviation_with_identity_maps_is_weight_invariant_on_chain_graphs`,
    multi-bridge visibility at
    `test_boundary_deviation_with_weights_visible_with_multiple_bridge_edges`.
  - **H15 (boundary_from_weights).** A vertex is on the boundary
    iff every incident edge has weight $\ge$ threshold. Vertices
    with even one untrusted incident edge fall to the interior.
    Pinned at
    `test_boundary_from_weights_picks_only_fully_trusted_vertices`.

**Out of scope for v3.1 (named honestly):**

  - **JWKS verification round-trip.** `boundary_from_weights` works
    on a weight vector that the caller has built from a
    trusted/revoked partition; mapping receipts → JWKS-verified
    edges is the caller's responsibility (see v3 §3.4 same point).
  - **Adaptive threshold for "deviation > θ → hallucination".**
    Decision threshold is calibrated per-corpus; v3.1 ships the
    primitive, not the threshold.

### 3.4.2 Corpus-scale ROC bench (2026-05-02) — what we measured

The v3 bench at `scripts/research/sheaf_v3_roc_bench.py` runs all
three detectors (v2.2 baseline, v3 receipt-weighted, v3.1 boundary
deviation) over the 16-document `seed_long_paragraphs` corpus
under deterministic 50/50 trust partitioning per doc. Receipt
JSON: `fixtures/bench_receipts/v3_roc_bench_2026-05-03.json`.

**Headline AUC numbers (mean across runs; ±0.02 LAPACK jitter):**

| Class × target | v2.2 | v3 | v3.1 |
|---|---|---|---|
| A1 entity-swap @ trusted   | 0.62 | 0.63 | 0.50 |
| A1 entity-swap @ untrusted | 0.60 | 0.69 | 0.37 |
| A2 predicate-flip @ trusted    | 0.50 | 0.50 | 0.50 |
| A2 predicate-flip @ untrusted  | 0.50 | 0.50 | 0.50 |
| A4 triple-drop @ trusted   | 0.86 | 0.94 | 0.50 |
| A4 triple-drop @ untrusted | 0.84 | 0.97 | 0.20 |

**Three falsification verdicts (named in code):**

  - **F1 MARGINAL.** v3 mean AUC on trusted-target (0.685) vs
    v2.2 mean (0.663): $\Delta = +0.022$. v3 is slightly better
    than v2.2 on trusted-target perturbations, but the margin is
    inside the noise floor. The H4 hypothesis ("trust amplifies
    signal") holds dramatically on synthetic-data H4 (10/10 wins,
    ~10× ratio) but only marginally at corpus scale on
    `seed_long_paragraphs`.

  - **F2 PASS.** v3 doesn't collapse on untrusted-target — no
    class drops more than 0.10 from v2.2. The 0.1 default weight
    is a viable floor for naturalistic-prose corpora.

  - **F3 FAIL.** v3.1 boundary-deviation mean AUC: 0.50 on
    trusted, 0.34 on untrusted. **Synthetic H12 passed
    (`test_boundary_deviation_detects_interior_tampering`);
    corpus-scale FAILS.** This is a real falsification of v3.1's
    utility on naturalistic prose with random 50/50 partitioning.

  Likely causes for the F3 failure (open hypotheses for v3.2):
  - Per-doc graphs are small (5–10 triples). 50/50 partition
    leaves ~3 trusted edges → `boundary_from_weights` often
    returns degenerate boundaries (full or empty), forcing the
    fallback path.
  - The cochain-construction `cochain_one_hot_v2` may not produce
    a meaningful boundary embedding when most boundary vertices
    are zero-vectors (vertices not in the trained vocabulary
    via the per-doc sub-vocabulary).
  - Random 50/50 partition is harsh: real-world deployments would
    have receipt distributions concentrated by source (e.g.,
    one document = one issuer = uniform trust within the doc).

  **F3 directs the v3.2 work**: the boundary-inference primitive
  is mathematically correct (proven by the synthetic H12 pin) but
  its practical utility requires either (a) larger graphs, (b)
  better cochain construction at vertex boundaries, or (c)
  corpus selection where the trust partition is structurally
  meaningful rather than random. A future v3.2 should test (a)
  and (b) explicitly; a future deployment study should test (c).

  Calling F3 a FAIL rather than a TODO is the truth-first
  discipline: the bench surfaced a finding the synthetic tests
  could not, and burying it under "future work" would be
  dishonest about the current state. **Synthetic-data utility
  testing is necessary but not sufficient for category-defining
  software.**

This is the corpus-scale bench v3 listed as "out-of-scope" at PR
#121. v3.X work continues from here.

### 3.4.3 F3 diagnostic harness (2026-05-02) — F3 is structural, not parametric

A 2×2×2 diagnostic over (graph_size, cochain_strategy,
partition_strategy) at
``scripts/research/sheaf_v3_1_f3_diagnostic.py`` was built to
isolate which of the §3.4.2 hypotheses (A graph too small,
B cochain produces zero-vectors, C random partition too harsh) is
load-bearing. Receipt:
``fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-03.json``.
Schema: ``sum.sheaf_v3_1_f3_diagnostic.v1`` with ``bench_digest``
field — JCS-canonical SHA-256 over the quantized payload (AUCs
to 3 decimals; diagnostic floats to 4); the digest lets future
runs prove reproducibility, lets a Node port prove cross-runtime
byte-identity, and is signable with the project's existing
Ed25519 keys (same trust alphabet as ``render_receipt.v1``).

**Result: load_bearing_hypothesis = "none"**. All 8 cells
FAIL the F3 PASS threshold (trusted-mean AUC ≥ 0.55). Every
single-axis flip of the PR #124 baseline cell still FAILs; even
the all-three-axes-flipped cell FAILs. **None of the three
hypotheses is load-bearing.**

The diagnostic's per-cell AUC structure reveals *why*:

  - All 4 cells using ``cochain_strategy = trained_embedding``
    produce uniform AUC = 0.500 across every (class, target).
    Reason: the strategy as I designed it returned the same
    cochain regardless of render — the cochain is a pure function
    of sheaf vertices. This was a confused axis. It does serve as
    a sanity-check negative control: AUC = 0.500 everywhere
    confirms the harness's AUC machinery is correct (a render-
    invariant cochain MUST give AUC 0.500).
  - All 4 cells using ``cochain_strategy = one_hot_default``
    (matches PR #124) produce AUC = 0.500 on every
    *trusted-target* perturbation across A1/A2/A4. This is
    structural: when a perturbation targets a trusted edge, the
    perturbation's vertices lie on the boundary, the cochain
    change is at boundary positions, the harmonic extension
    formula $x_I^* = -L_{II}^{-1} L_{IB} x_B$ recomputes the
    interior from the new boundary, but the actual interior is
    unchanged — so the deviation $\|x_I^{\text{actual}} -
    x_I^*\|^2$ ties between clean and perturbed by mathematical
    necessity.
  - The same 4 cells produce AUC ≈ 0.36–0.43 on untrusted-target
    (anti-correlated). Reason: ``cochain_one_hot_v2`` zeros out
    the swapped-out entity's vertex, lowering the interior
    cochain's energy; the harmonic extension also lowers; the
    deviation can shrink rather than grow.

**Truth-first conclusion: v3.1 boundary deviation has a
structural blind spot for perturbations on the trust frame's
boundary**. This is not a parameter-sweep problem. The detector,
as currently formulated, cannot distinguish clean from perturbed
when the perturbation's vertices are in $B$.

**Implications for v3.2 design:**

  1. v3.1's boundary deviation as a *standalone* detector is not
     a viable path. It must be paired with a separate signal that
     handles boundary perturbations (e.g., x_B's direct
     contribution to the weighted Laplacian quadratic form).
  2. The cochain construction must encode render information
     into the *interior* of the partition for boundary deviation
     to be informative. A cochain that's translation-invariant
     under boundary-only perturbations is mathematically blind
     by construction.
  3. The PR #124 §3.4.2 hypotheses (A/B/C) need retirement —
     they were partial misdiagnoses. The diagnostic refuted
     them empirically; the root cause is upstream of all three.

The diagnostic settled the question: F3 is structural, not
parametric. v3.2 should redesign the detector, not tune
parameters of v3.1.

### 3.4.4 v3.2 (IMPLEMENTED 2026-05-02) — F3 STRUCTURAL FAIL closer

v3.2 (``sum_engine_internal/research/sheaf_laplacian_v32.py``,
``Tests/research/test_sheaf_laplacian_v32.py``) is a **strict
generalization of v3** that adds the harmonic-extension deviation
as a complementary signal:

  $$v_\text{combined}^{v3.2} = v_{\text{laplacian}}^w + \gamma \cdot \text{deviation}_w + \lambda \cdot v_{\text{deficit}}$$

The two cochain-side terms catch complementary things:

  - $v_\text{laplacian}^w$ (from v3) sums residuals over *every*
    edge — informative anywhere on the graph, including under
    boundary-only perturbations regardless of $L_{IB}$ topology.
  - $\text{deviation}_w$ (from v3.1) is informative *only* when
    $L_{IB} \neq 0$. Under the F3 failure topology ($L_{IB} = 0$,
    edges live entirely within $B$ or entirely within $I$),
    deviation is structurally invariant to boundary perturbations
    by linear algebra.

When $\gamma = 0$, v3.2 reduces to v3 numerically (subsumption — the
H16 contract). When $\gamma > 0$, deviation contributes additively
where it has signal; falls back to a constant where it's blind.
The combined score is informative either way — that's the F3
**fall-back** guarantee (H18).

**Falsifiable predictions** (pinned in ``test_sheaf_laplacian_v32.py``):

  - **H16. Subsumption.** $\gamma_\text{deviation} = 0$ → v3.2
    numerically equals v3.
  - **H17. $L_{IB} \neq 0$ visibility.** On a graph with cross-
    partition edges, $\text{deviation}_w$ changes under boundary
    perturbation.
  - **H18. F3 fall-back.** On a graph with $L_{IB} = 0$,
    $v_\text{laplacian}^w$ still surfaces the perturbation; the
    combined score is informative even when deviation is blind.
  - **H19. No λ double-counting** at the v3.2 wrapper layer.
  - **H20. Degenerate-boundary fall-back.** Empty $B$ or full $B$
    → $\text{deviation}_w = 0$; combined score reduces to v3.

**Corpus-scale validation** (PRs #126/#127/#129 + Sprint 1
substrate-determinism rebase; receipt
``fixtures/bench_receipts/v3_2_validation_2026-05-03.json``,
``bench_digest = b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a``):

| γ        | trusted-mean AUC | F4 (≥0.55) | Δ vs v3 | F5 (Δ ≥ −0.02) |
|----------|------------------|------------|---------|----------------|
| 0.0      | 0.663            | PASS       | 0.000   | PASS           |
| 0.1      | 0.659            | PASS       | −0.004  | PASS           |
| 1.0      | 0.635            | PASS       | −0.028  | FAIL           |
| auto (1.0167) | 0.635       | PASS       | −0.028  | FAIL           |
| **v3** (ref) | **0.663**    | —          | —       | —              |

Three honest readings:

  1. **F3 STRUCTURAL FAIL is closed at the detector layer.** v3.2
     at any γ ≥ 0 produces trusted-mean AUC ≥ 0.55 (vs PR #124's
     v3.1 standalone deviation: 0.499). The "blind spot" was
     scoring against deviation alone; pairing with $v_\text{laplacian}^w$
     restores robust signal regardless of $L_{IB}$ topology.
  2. **Calibration finding (truth-first):** the magnitude-matching
     auto-calibration heuristic (γ_auto ≈ 1.0) is **wrong on this
     corpus** — F5 fails at γ ∈ {1.0, auto}. Optimal γ is small
     (≤ 0.1). Deviation's signal-to-noise ratio is worse than its
     magnitude suggests; on the seed_long_paragraphs distribution
     it functions as a small modulator, not a co-leader.
  3. **H16 verified at corpus scale.** γ = 0 produces trusted-mean
     AUC = 0.663, byte-identical to v3's. v3.2 is genuinely a
     strict generalization, not a different detector wearing a
     similar mask.

**Reproducibility (closed under Sprint 1 substrate-determinism fix).**
``bench_digest`` reproduces across runs unconditionally. The earlier
``PYTHONHASHSEED=0`` requirement was caused by a single load-bearing
site in ``DeterministicSieve.extract_triplets``
(``return list(set(triplets))`` — set iteration is hash-randomized);
fixed in the same PR as this section update by sorting at the
dedup step. Verified by running the validation script three times in
fresh Python processes and confirming identical digests. The fix
shifted the digest values slightly from their PYTHONHASHSEED=0-
conditional form (e.g. trusted-mean AUC moved from 0.661 to 0.663
at γ=0); substantive verdicts are unchanged.

**v3.3 candidate directions** (named, not yet investigated):

  - Per-doc graph-structure-aware γ: when $L_{IB}$ has high mass,
    raise γ; when near zero, set γ = 0. The combined score then
    uses deviation only where it's structurally informative.
  - Cochain redesign that propagates render content into the
    interior (the original PR #125 §3.4.3 implication 2). v3.2
    works around this by combining with $v_\text{laplacian}^w$;
    a cochain redesign would address the root cause.
  - A2 weakness — relation perturbations score 0.500 across all
    detectors (v22, v3, v31, v32). This needs predicate-perturbation
    negative sampling at training time, separate from the v3.2 arc.
    **(Sprint 7.5 update: predicate-perturbation training was
    tested in `aa34b6e8…` and did NOT lift A2 — the cochain
    construction is mathematically blind to entity-set-preserving
    perturbations regardless of training-distribution. The
    recovery is via the §3.5 per-rendered-triple V channel —
    see §3.4.5 below.)**

### 3.4.5 Sprint-7.5 baseline-comparison gate + recovery arc (2026-05-04)

The substrate's truth-first discipline includes a baseline-comparison
gate before any detector ships into a public claim: how does the
detector fare against trivial reproducible baselines computed from
the same render bundle?

Two such baselines, both pure set ops on entity sets and both
fully deterministic (no LAPACK, no randomness — AUC reproduces
exactly across runs):

  - **B1 entity-presence-deficit** —
    $1 - |\text{src}_\text{ent} \cap \text{render}_\text{ent}| /
    |\text{src}_\text{ent}|$
  - **B2 jaccard-distance** —
    $1 - J(\text{src}_\text{ent}, \text{render}_\text{ent})$

Module: `scripts.research.sheaf_baseline_comparison`.
Receipt: `fixtures/bench_receipts/baseline_comparison_2026-05-04.json`,
`bench_digest cb32c617…`. Pinned in
`Tests/research/test_sheaf_baseline_comparison.py`.

**STOP-THE-LINE finding: trivial baselines beat v3.2 alone.**
B2 trusted-mean AUC = 0.833 vs v3.2 cochain-only at 0.659
(Δ = −0.174). Per-cell: B2 catches A1 trusted at 1.000, A4
trusted at 1.000; v3.2 cochain only sees 0.578 and 0.898
respectively. A2 is at chance for both (0.500).

This loss is structural, not parametric. The two recovery
hypotheses we tested:

  1. **Predicate-perturbation training negatives** — adding
     $(h, r', t)$ negatives to the LCWA contrastive sampler
     during training, hoping the trained restriction maps would
     learn to score predicate-flips as high V. Result: **A2
     stayed at 0.500.** Receipt:
     `fixtures/bench_receipts/predicate_negatives_experiment_2026-05-04.json`,
     `bench_digest aa34b6e8…` (operator-environment, Python 3.10;
     digest is Python-version-sensitive because the bench uses a
     local copy of the v2 training loop — see the bench script
     header. The substantive finding holds invariant; the pinned
     test asserts verdict + A2-at-chance rather than byte-digest).
     The trained sheaf with predicate
     negatives does score the negatives as high V *during
     training*, but A2 detection at scoring time depends on the
     *cochain construction* — and the cochain at vertex $v$ is
     `trained_embedding(v) if v ∈ rendered_triples else zero`.
     When A2 swaps $(h, r, t) \to (h, r', t)$, both $h$ and
     $t$ remain in the rendered entity set; the cochain is
     identical between clean and perturbed; the per-edge
     residual under the source-graph topology computes the same
     value. Predicate doesn't enter the cochain. Adding training
     negatives can't fix what the scoring path discards.

     This is a **structural finding** of the same shape as F3
     STRUCTURAL FAIL: a synthetic-utility expectation
     (predicate negatives should help A2) refuted by a corpus-
     scale measurement, with the failure traceable to a
     mathematical blindness in the scoring architecture.

  2. **Per-rendered-triple V channel integration** — restoring
     the §3.5 channel that v2.2 §4.3 ROC bench used to hit
     A1/A2/A3 = 1.000. The per-triple channel scores each
     rendered triple $(h, r, t)$ directly under
     $F_h(r), F_t(r)$, so an A2 perturbation $r \to r'$ produces
     high $V_\text{triple}$ at $F_h(r')$ — without depending on
     the cochain construction. Implementation:
     `scripts.research.sheaf_per_triple_integration_experiment.score_v32_with_per_triple`.

     Result: **A2 LIFTED 0.500 → 0.671** on trusted target,
     0.500 → 0.678 on untrusted. v3.2 + per-triple trusted-mean
     AUC = 0.759 (vs cochain-only 0.659). Receipt:
     `fixtures/bench_receipts/per_triple_integration_2026-05-04.json`,
     `bench_digest 7025436f…`.

     Still loses to B2 alone (0.833), but the structure of the
     loss is now informative: v3.2 + per-triple is the *only*
     detector that catches A2.

The two are **structurally complementary**: B2 catches entity-
set-changing perturbations (A1/A4); v3.2 + per-triple catches
predicate-flips (A2). The complementary hybrid (§3.4.6 below)
combines them via Borda fusion to beat both.

### 3.4.6 Complementary hybrid — Borda(v3.2 + per-triple, B2) — IMPLEMENTED 2026-05-04

The substrate-enabled headline detector. Borda rank-fusion at
the per-(class, target) cell level of the two complementary
signals from §3.4.5:

  - $s^{\text{v3.2 + per-triple}}_i$ — the §3.4.5 channel
    composition (cochain V + γ·deviation_w + λ·v_deficit +
    α·max_in_vocab_v_triple + β·n_oov)
  - $s^{\text{B2 jaccard}}_i$ — entity-set jaccard distance

Borda fusion:

$$s^\text{borda}_i = \mathrm{rank}(s^{(\text{v32+pt})}_i \mid s^{(\text{v32+pt})}_*) +
\mathrm{rank}(s^{(\text{B2})}_i \mid s^{(\text{B2})}_*)$$

Module: `scripts.research.sheaf_complementary_hybrid_experiment`.
Receipt: `fixtures/bench_receipts/complementary_hybrid_2026-05-04.json`,
**`bench_digest dc6e0260…343ce`**. Pinned in
`Tests/research/test_recovery_experiment_digests.py::test_complementary_hybrid_digest_pinned`
(test additionally asserts the verdict label
`HYBRID_BEATS_BASELINE`).

**Per-cell AUC comparison (2026-05-04):**

| Class × target | v3.2 + per-triple | B2 jaccard | **Borda hybrid** |
|---|---:|---:|---:|
| A1 entity-swap @ trusted    | 0.698 | **1.000** | 0.967 |
| A1 entity-swap @ untrusted  | 0.751 | 1.000 | 0.991 |
| A2 predicate-flip @ trusted | **0.671** | 0.500 | **0.671** |
| A2 predicate-flip @ untrusted | **0.678** | 0.500 | 0.678 |
| A4 triple-drop @ trusted    | 0.907 | 1.000 | 0.991 |
| A4 triple-drop @ untrusted  | 0.969 | 1.000 | 1.000 |
| **Trusted-mean**            | 0.759 | 0.833 | **0.876** |

**Δ(borda − b2) = +0.043** trusted-mean. Above the +0.030
"real win" threshold. **Verdict: HYBRID_BEATS_BASELINE.**

The fusion preserves both contributions: A2 lift from the
sheaf-Laplacian channel; A1/A4 dominance from B2's perfect 1.000
on entity-set changes. Magnitude-invariance of Borda means the
component scale differences (sheaf-Laplacian scores in
$O(10^1{-}10^2)$, jaccard in $[0, 1]$) don't bias the fusion.

**Cross-machine verification (Sprint 7.5 T3.M, 2026-05-04).** The
complementary hybrid bench was re-run on Modal x86_64 (Linux 4.4 /
glibc 2.31 / Python 3.10.8 / numpy 1.25.0 / OpenBLAS-via-PyPI /
AVX2) at the pinned commit SHA. The bench_digest `dc6e0260…`
reproduced byte-for-byte and the substantive verdict
`HYBRID_BEATS_BASELINE` (Δ=+0.043) reproduced cross-machine.
Apple Accelerate (operator's Apple Silicon) and OpenBLAS-via-PyPI
(Modal x86_64) are two distinct LAPACK environments — see
`docs/PROOF_BOUNDARY.md` §2.10 for the full cross-machine
boundary. Receipt:
`fixtures/bench_receipts/cross_machine_verification_2026-05-04.json`.

**Out of scope (named honestly):**

  - **Cross-corpus generalisation.** The +0.043 margin is on
    `seed_long_paragraphs` with the synthetic A1/A2/A4
    perturbation harness. v0.2 follow-up: second corpus.
  - **Real-LLM-rendered hallucinations.** Closed by Path 2 (PR
    #156, preprint §4.7.2), cross-corroborated across six LLM
    lineages on `seed_long_paragraphs` by Path 2 multi-LLM
    (PRs #157 / #158 / #161, preprint §4.7.3
    `STRUCTURAL_GAP_NO_MODEL_BEATS`), extended across three
    corpora by Path 2 cross-corpus (PR #163, preprint §4.7.4
    initially `CROSS_CORPUS_VERDICTS_DIVERGE`), then resolved by
    §4.7.4.1 (PR #164) as extremal-Goodhart at small n. The lone
    BEATS cell on the 8-doc `seed_paragraphs` × gpt-4o-mini
    (Δ=+0.032) does not survive at 16 docs in the same
    encyclopedic style (Δ=−0.013 TIES). At controlled sample sizes
    (n ≥ 16) across three corpora and 4-6 LLM lineages, the joint
    finding is `STRUCTURAL_GAP_NO_MODEL_BEATS` in 3/3 corpora.
    Honest reading: the synthetic-bench WIN (+0.043) is best read
    as a Goodhart artifact — the hybrid was selected to compose
    well on a measure (the synthetic harness), and the measure
    stops being a good measure once it is the target of
    optimisation. v0.4+ candidates remaining: real-LLM-aware
    per-triple V training, naturalistic perturbation synthesis on
    the source TRIPLE set rather than the rendered prose, deeper
    corpus sampling (5-10 stylistically distinct corpora at n ≥
    16).
  - **Composition tuning.** Borda is parameter-free but is one
    composition choice among many. Z-score additive, gated
    (B2 fires → use B2 else fall to v3.2 + per-triple), and
    weighted-linear are alternatives we didn't exhaustively
    evaluate.

### 3.5 Output shape

Stable JSON envelope `sum.hallucination_consistency.v1`:

```json
{
  "schema": "sum.hallucination_consistency.v1",
  "source_bundle_hash": "sha256-...",
  "render_count": 12,
  "stalk_dim": 384,
  "version": "v2-learned-embeddings",
  "consistency_profile": {
    "mean_laplacian": 0.083,
    "std_laplacian": 0.041,
    "max_per_render": 0.197,
    "argmax_render_id": "deadbeef00000000",
    "per_edge_top10": [
      {"edge": "alice|graduated|2012", "discrepancy": 0.083, "render": "deadbeef..."},
      ...
    ]
  },
  "decision": {
    "threshold_mu": 0.15,
    "threshold_sigma": 0.10,
    "verdict": "consistent",
    "calibration_set": "synthetic_v1_n100_2026-05-01"
  }
}
```

---

## 4. Falsifiable predictions

Any of these failing falsifies the detector design at the v1/v2/v3
specified.

**P1.** On a synthetic benchmark of 50 triple-sets with controlled
adversarial perturbations (entity substitution, predicate flip, fact
fabrication), the mean Laplacian on adversarial-rendering manifolds
is at least $2\times$ the mean on clean-rendering manifolds.
*Strong signal threshold; falls back to weaker $1.3\times$ for v1.*

**P2.** ROC curve over 50 mixed clean/adversarial sets achieves
AUC $\geq 0.75$ for v2 with text-embedding-3-small stalks. Below
that, the detector is barely better than confidence-calibration
heuristics.

**P3.** Per-edge top-$k$ discrepancy localization correctly
identifies the perturbed triple in $\geq 70\%$ of single-perturbation
adversarial examples. (Localization is what makes the detector
*useful* downstream, not just a binary signal.)

**P3 update (2026-05-01):** the synthetic micro-benchmark
([`scripts/research/sheaf_microbench.py`](../scripts/research/sheaf_microbench.py))
established this empirically at **18/18 = 100%** top-1 localization on
catchable classes (A1 entity-swap, A4 triple-drop, A5
consistent-entity-swap) across 6 fact-sets. The 70% threshold is
preserved as the bar v1 must clear to be useful; v1's actual
performance is a strict superset of that bar on synthetic data.
Real-world benchmark calibration is Week 3 of the plan.

**P4.** Render receipts issued by trusted issuers concentrate
$x^T L x$ at the low end; unsigned renderings or renderings from
unknown issuers concentrate at the high end (v3 only, when receipt
metadata is available).

If P1–P3 fail, the detector design needs revision — likely toward
v2 with learned restriction maps rather than identity, or different
cochain construction. If P4 fails, the receipt-weighting in v3
adds no signal beyond v2 and should be dropped.

---

## 5. Plan

Total scope: **~3 weeks one-engineer**, gated on no further
emergent constraints.

### 5.1 Week 1 — references and prototype scaffold

- [x] Read Gebhart, Hansen & Schrater (2023) — *complete*; this doc
  is the synthesis output.
- [ ] Read Hansen & Ghrist (2019), *Toward a Spectral Theory of
  Cellular Sheaves* — foundational; used for Laplacian properties
  and pseudoinverse / harmonic-extension subtleties.
- [ ] Read Tull, Kleiner & Smithe (2023, arXiv:2308.00861) end-to-end
  — for the categorical-active-inference compositionality theorem
  and the connection to hierarchical free-energy bounds.
- [ ] Skim Boudourides; Phillips & Wilson 2018; sheaf-cohomology
  cognitive-systematicity papers — cite as adjacent references, not
  load-bearing.
- [ ] Scaffold module `sum_engine_internal/research/sheaf_laplacian.py`
  with type stubs and signatures only (~200 LOC). Tests dir:
  `Tests/research/test_sheaf_laplacian.py`.

### 5.2 Week 2 — v1 prototype + synthetic benchmark

- [ ] Implement v1 (1-dim presence stalks). Pure-stdlib + numpy;
  sparse-matrix Laplacian via scipy.sparse if needed.
- [ ] Synthesize benchmark: 50 fact-sets generated from a small
  ontology (people / places / events / properties). For each,
  generate (a) 8 clean renderings via `sum render` at varying
  slider positions, (b) 8 adversarial renderings with one of
  *entity-swap*, *predicate-flip*, *fact-fabrication*,
  *negation-injection* applied at a known location.
- [ ] Compute the consistency profile for each. Plot
  $(\bar{V}, \sigma_V)$ scatter. Compute ROC vs. ground-truth labels.
- [ ] If v1 ROC AUC $\geq 0.65$ (P2-relaxed for the unlearned case),
  proceed to v2; else investigate cochain construction and re-derive.

### 5.3 v2 — split per Hansen-Ghrist 2019 reading

#### 5.3a Week 3 — v2.0 sanity test (skip as standalone artifact)

- [ ] Smoke test: `coboundary_matrix_v2` and `sheaf_laplacian_v2`
  return the correct values on a 1-dim sheaf (i.e., reduce to v1
  output when restriction maps and stalk dim are degenerate). Pin
  in `Tests/research/test_sheaf_laplacian_v2.py` as a numerical compatibility
  guarantee, not as a ROC benchmark.

#### 5.3b Week 3-4 — v2.1 learned restriction maps (the real artifact)

- [ ] Implement `sum_engine_internal/research/sheaf_laplacian_v2.py`
  with d-dim stalks (initially $d \in \{8, 32, 64\}$) and per-relation
  restriction maps as trainable parameters. Use scipy.sparse.bsr
  for the block-sparse Laplacian. ~300 LOC.
- [ ] Implement the contrastive sheaf-embedding training loop
  (Gebhart Def. 11 / Eq. 4): $\gamma$-gapped margin ranking loss
  with positive triples from source bundles and negative triples
  via local-closed-world-assumption (LCWA) sampling. CPU-only.
- [ ] Synthetic bench: 6 fact-sets × 5 perturbation classes (the
  v1 connected-graph corpus) PLUS 6 disconnected fact-sets that
  v1 missed. Train restriction maps on the 12-graph corpus, run
  detector on held-out renders.
- [ ] Compare v2.1 vs v1 on the disconnected-graph corpus
  specifically. Honest result: does v2.1 close the v1 blindspot
  the real-data test of 2026-05-01 surfaced?

#### 5.3c Week 5+ — v2.2 semantic-embedding cochains (publishable)

- [ ] Replace presence-style cochains with sentence-transformer
  embeddings of context windows around each entity's mention.
  Default model: `all-MiniLM-L6-v2` (384-dim, CPU-friendly).
  Optional: text-embedding-3-small via OpenAI API (per-experiment
  spend authorization).
- [ ] Re-run the v2.1 synthetic + disconnected bench with v2.2
  cochains. Measure ROC AUC against P1 target ($\geq 0.75$).
- [ ] Run a small *real* benchmark: 20 documents from a curated
  corpus where ground-truth-clean and known-tampered renderings
  are both available.
- [ ] Write a 2000-word note: *"Cohomological consistency scoring
  for signed bidirectional render receipts."* Include the
  SUM-to-Knowledge-Sheaves mapping (§2.3), the v1/v2/v3 procedure,
  the synthetic + real benchmarks, and the bounded-claims section.
- [ ] Submit to arXiv (`cs.AI` primary, `math.CT` secondary).
- [ ] Cross-post to the ACT (Applied Category Theory) discourse,
  the Topos Institute mailing list, and the Coecke / Quantinuum
  DisCoCat / lambeq community.

### 5.4 v3 (post-publication, contingent on v2.2 passing P1–P3)

If v2.2 publishes cleanly, v3 (receipt-weighted + harmonic
extension over trusted-issuer boundary) becomes the SUM-specific
extension that doesn't replicate elsewhere because no other system
has cross-runtime-verified render receipts. v3 is the positioning
anchor — *"this is the bit that requires SUM."* See Hansen-Ghrist
2019 Proposition 4.1 + Theorem 4.5 for the harmonic-extension
machinery the boundary-value formulation rests on.

---

## 6. Bounded claims

**Note (2026-05-01):** which version's claims hold is now version-
explicit. v1 is shipped (PR #106) and partially falsified on
naturalistic data (PR #107). v2.0 is a sanity test, not a research
artifact. v2.1 / v2.2 / v3 are unimplemented; their claims are
predictions, not measurements.

What this artifact, if successful, claims:

- **Specific:** the sheaf-Laplacian quadratic form on a 0-cochain
  induced by re-extracted triples from a render manifold separates
  consistent from inconsistent rendering sets at the ROC AUCs
  specified in §4.
- **Localized:** per-edge discrepancy correctly identifies the
  perturbed triple at the rates specified in §4.
- **Trust-coupled (v3 only):** receipt-issuer trust weighting
  improves the detector's signal-to-noise on adversarial
  rendering manifolds.

What this artifact does **not** claim:

- That it solves hallucination. It is one signal among many; in
  practice it would compose with confidence-calibration,
  retrieval-grounded checks, NLI verifiers, and human review.
- That it provides correctness proofs. It detects inconsistency in
  the *output manifold*; it does not certify that any single
  output is *correct*.
- **Consistent-hallucination behaviour (corrected 2026-05-01).**
  An earlier draft of this section claimed "a perfectly consistent
  manifold of uniformly-wrong renderings would score zero." That
  was over-cautious. The empirical decomposition, pinned by
  ``Tests/research/test_sheaf_laplacian.py``:
    * Consistent hallucination via *entity substitution* (A5-via-swap)
      **is caught**. Each render fails the same edges by the same
      amount; the *mean* Laplacian is positive even though per-render
      variance is zero. The detector signals on the mean, not just
      the variance. Verified 6/6 on the synthetic micro-benchmark.
    * Consistent hallucination via *predicate flip* (A2)
      **is missed**. v1's presence stalks carry no predicate
      information; consistency across renders is irrelevant because
      no per-render signal exists in the first place. v2
      (learned-embedding stalks) is required.
    * Consistent hallucination via *off-graph fabrication* (A3)
      **is missed**. Entities outside the source vertex set are
      silently ignored by the cochain construction.

  Honest one-line summary: consistent hallucination via classes v1
  catches (A1 entity-swap, A4 triple-drop, A5-via-swap) is caught
  by the mean signal; consistent hallucination via classes v1 is
  structurally blind to (A2 predicate-flip, A3 off-graph fabrication)
  remains uncaught regardless of consistency.
- That it generalises across all knowledge-graph schemas. The
  schema choices (single-type vs. typed; restriction-map structure)
  affect the score; calibration is per-schema until robustness is
  established.
- That it is computable on arbitrary bundle sizes. The Laplacian
  is dense in the worst case; for $|V| > 10^4$ vertices we expect
  to need sparse approximations or graph partitioning.
- That the v1 simple-presence variant captures semantic drift.
  v1 is a coarse pre-filter; v2 is the semantic version.
- **Empty-render false negative (added 2026-05-01).** A render that
  extracts zero triples yields the all-zero cochain x = 0, hence
  $x^T L_{\mathcal{F}} x = 0$ — the same score as a perfectly
  consistent render. Pinned by
  ``test_empty_render_maximizes_laplacian``. Callers must treat
  ``n_extracted == 0`` as a separate signal; the Laplacian alone
  cannot distinguish "all entities present everywhere" from "no
  entities present anywhere." Addressing this requires either an
  absolute-presence regulariser ($\|x\|^2$ term) or v2's learned
  stalks, which carry positive activation on mention.
- **Disconnected source-graph blindspot (added 2026-05-01,
  surfaced by the real-prose test
  ``scripts/research/sheaf_real_test.py``).** When the source
  bundle's induced graph $G_T$ has multiple disconnected
  components and a density-controlled render drops *entire
  components* (both endpoints of an edge), every remaining
  edge is in $\{(1,1), (0,0)\}$ — never $(1,0)$ — so the
  Laplacian quadratic form is identically zero even though
  facts have been dropped. The synthetic micro-benchmark
  (which used connected fact-sets) did not surface this,
  but naturally-occurring prose often produces sparse,
  disconnected fact-sets (4 unrelated facts about 4
  unrelated entities is the textbook case). Real-data
  measurement: density=0.5 of a 4-fact disconnected graph
  yielded V = 0, identical to density=1.0. **v1 cannot
  detect density-controlled axiom dropout on disconnected
  graphs.** v2's per-vertex learned-embedding stalks
  address this because each vertex's embedding contributes
  independently to the cochain even when its graph
  neighbourhood vanishes. Pinned by
  ``test_disconnected_graph_density_dropout_invisible``.
- **Identity-restriction-map d>1 stalks do NOT address v1
  blindspots (added 2026-05-01 from Hansen-Ghrist 2019 reading).**
  An earlier sketch of v2 in §3.3 framed it as "swap stalk_dim
  to 384 with identity restriction maps." That's mathematically
  insufficient: with identity restriction maps and per-vertex
  semantic embeddings, the global-section condition becomes
  $x_v = x_u$ for every edge, which means entities connected
  by *any* relation should have *identical* embeddings — a
  wrong constraint. v1's blindspots (predicate-flip, off-graph
  fabrication, disconnected-graph density-dropout, empty-render
  false-negative) are addressed only by **per-relation learned
  restriction maps** (Gebhart Def. 11 / Eq. 4 contrastive
  training). The v2.1 / v2.2 / v3 path in §3.3 / §5.3 reflects
  this. v2.0 (identity restriction maps) is now framed as a
  numerical-compatibility smoke test only, not a research
  artifact.

What is honestly speculative, pending the benchmark:

- Whether the receipt-weighting in v3 carries the trust signal
  cleanly. P4 is a hypothesis, not a result.
- Whether the detector composes well with retrieval-grounded
  checks (HHEM, FactScore, etc.). Compositionality with existing
  hallucination detectors is a follow-on study, not part of v1.

---

## 7. Position vs. the three monetisable wedges

The detector itself is a *research artifact*. It earns its keep
operationally by sharpening each of the three wedges identified in
the post-#104 zenith conversation:

**Agent-to-agent trust protocol.** Two agents render the same
triple-set; if their consistency profiles diverge, the disagreement
is *localizable* to specific predicates. Multi-agent reconciliation
becomes "compute the discrepancy edge, surface the conflicting
axiom, request a re-render or a human arbiter." This converts
inter-agent disagreement from a vibes-based problem into a
mathematical one.

**C2PA-text-equivalent.** Every signed publication-release receipt
can be paired with a consistency profile against the source
documents' triples. A publisher who claims "this AI-rendered
summary is consistent with these source documents" exposes a
falsifiable claim — the consistency profile is computable by any
verifier with the source bundle.

**Compliance / regulated-LLM audit.** A regulator's audit pass
*is* a request for the consistency profile of the regulated party's
rendering manifold against an approved source set. The sheaf
Laplacian is a defensible quantitative metric in adversarial
settings (legal, medical, financial) where confidence-calibration
heuristics are not.

The detector is the same artifact in all three settings; the wedges
differ only in who computes the cover and who reads the score.

---

## 8. References

- **Gebhart, T., Hansen, J., & Schrater, P.** (2023). *Knowledge
  Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph
  Embedding*. AISTATS 2023, PMLR 206. arXiv:2110.03789. ★ Primary
  reference; Equation 1 is the detector's core math.
- **Hansen, J. & Ghrist, R.** (2019). *Toward a spectral theory of
  cellular sheaves*. Journal of Applied and Computational Topology
  3(4):315–358. The foundational sheaf-Laplacian paper. Used here
  for spectral properties and pseudoinverse subtleties.
- **Curry, J.** (2014). *Sheaves, Cosheaves, and Applications*.
  PhD thesis, University of Pennsylvania. The discretized
  cellular-sheaf theory the above papers build on.
- **Tull, S., Kleiner, J., & Smithe, T. St C.** (2023). *Active
  Inference in String Diagrams: A Categorical Account of Predictive
  Processing and Free Energy*. arXiv:2308.00861. Categorical
  active-inference; cited for the compositionality-of-free-energy
  theorem we expect to use in v3 / future hierarchical extensions.
- **Bordes, A., Weston, J., Collobert, R., & Bengio, Y.** (2011).
  *Learning Structured Embeddings of Knowledge Bases*. AAAI.
  Original SE model; per-edge $\|R_h x_h - R_t x_t\|^2$ scoring is
  the primitive Gebhart et al. 2023 generalises and that we apply.
- **Coecke, B., Sadrzadeh, M., & Clark, S.** (2010). *Mathematical
  Foundations for a Compositional Distributional Model of Meaning*.
  Linguistic Analysis 36. DisCoCat foundational; cited as the
  parallel categorical compositional-semantics line, not directly
  used in the detector but the natural follow-on for v3+ work.
- **Lawvere, F. W. & Tierney, M.** Topos theory; Lawvere-Tierney
  topology underpins the "topos of cognitive sheaves" framing in
  the SCT note.

---

## 9. Status and next concrete step

**Status (2026-05-01, updated):** specification complete with
empirical corrections from the v1 prototype run; references read
and verified; SUM-to-Knowledge-Sheaves mapping charted; bounded
claims set; falsifiable predictions named **and partially verified
on synthetic data**. **v1 implementation shipped** in
`sum_engine_internal/research/sheaf_laplacian.py` behind the
`[research]` extras flag, with 12 pinned tests
(`Tests/research/test_sheaf_laplacian.py`) covering 7 math-sanity
properties + 5 micro-benchmark assertions, and a reproducible
synthetic micro-benchmark in `scripts/research/sheaf_microbench.py`.

**Empirical signal on synthetic data (6 fact-sets × 5
perturbation classes = 30 trials, all CONNECTED graphs):**

  | Class | Detect rate | Top-1 localization |
  |---|---|---|
  | A1 entity-swap | 6/6 ✓ | 6/6 |
  | A2 predicate-flip | 0/6 — known blind | n/a |
  | A3 off-graph fabrication | 0/6 — known blind | n/a |
  | A4 triple-drop | 6/6 ✓ | 6/6 |
  | A5 consistent-swap (×3) | 6/6 ✓ via mean signal | 6/6 |
  | **Total caught** | **18/30** | **18/18 = 100%** |

The 60% catch rate is precisely the v1 design's claim on
connected graphs: catch entity-presence-affecting perturbations
cleanly, defer predicate- and off-graph-sensitive perturbations
to v2. Localization is strictly better than the spec's P3
prediction (≥70% target; actual 100% on caught classes).

**Real-data falsification (2026-05-01, on 4-fact disconnected
graph from naturalistic prose). The synthetic benchmark missed
this:**

  - density=1.0 render: V = 0 ✓
  - density=0.7 render (drops 2 axioms): V = 0 ✗ — should be > 0
  - density=0.5 render (drops 2 axioms): V = 0 ✗ — should be > 0
  - paraphrase 2 (reordered): V = 0 ✓
  - paraphrase 3 (verbose; sieve drift to ``python_code``): V = 3 ✓
  - paraphrase 4 (lexical variation): V = 0 ✓

**On disconnected source-graphs, v1's density-dropout signal
collapses to zero** (because dropping a whole component leaves
every remaining edge in {(1,1), (0,0)} — both endpoints
either present or both absent, never partially present, so no
edge contributes to V). The synthetic benchmark used connected
fact-sets which masked this. The honest one-line summary of v1's
real-data behaviour:

> **v1 is a connected-graph entity-presence drift detector.** It
> catches sieve canonicalisation divergence across paraphrases
> (verbose → predicate/object drift, the paraphrase 3 case). It
> does NOT catch density-controlled axiom dropout when the source
> graph is disconnected. Naturally-occurring prose tends to
> produce sparse, often-disconnected fact-sets, so v1 is **not
> generally useful as a hallucination detector on naturalistic
> input** — its useful regime is restricted to source-graphs with
> high cross-edge connectivity.

This is reproducible: ``PYTHONPATH=. python
scripts/research/sheaf_real_test.py``. Pinned in code by
``test_disconnected_graph_density_dropout_invisible``.

**Next concrete step:** v2. Replace 1-dim presence stalks with
text-embedding-3-small (1536-dim) stalks; train (or fix as
identity) the per-relation restriction maps. v2 addresses every
v1 blindspot named so far (A2 predicate-flip, A3 off-graph
fabrication, empty-render false negative, **disconnected-graph
density-dropout blindness**) because per-vertex embeddings
contribute independently to the cochain regardless of graph
neighbourhood. Targeted at Week 2 of the original three-week
plan; v3 (receipt-weighted, the SUM-specific extension) follows
v2.

— end of research direction
