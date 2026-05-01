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

### 3.3 Procedure (v2, learned-embedding stalks)

Same as v1 except:

- $\mathcal{F}(v) = \mathbb{R}^d$ for $d \in \{384, 768, 1536\}$
  (text-embedding model dim).
- Per-render cochain $x_n(v) = \mathrm{embed}(\text{context}(v, R_n))$
  where $\text{context}$ extracts a window around $v$'s mention in $R_n$.
- Restriction maps $\mathcal{F}_{h \trianglelefteq r}$ trained
  per-relation under the contrastive sheaf-embedding loss
  (Gebhart et al. Def. 11, Eq. 4) on the existing knowledge graph,
  *or* fixed as identity for the unlearned baseline.

v2 captures *semantic* drift (entity mentioned consistently but
described inconsistently across renderings) which v1 misses.

### 3.4 Procedure (v3, receipt-weighted)

Same as v2 except: each render $R_n$'s cochain contribution to the
Laplacian is weighted by an **issuer-trust score** derived from the
render receipt's signing key. Receipts from a trusted-issuer JWKS
contribute weight 1; unsigned renderings or renderings from unknown
issuers contribute lower weight. This couples the cohomology to the
trust loop — adversarial renderings issued by an attacker can be
included in the cover but downweighted.

v3 connects the obstruction class to SUM's existing
trust-and-verification primitives. It is the version that, if it
benches well, becomes a publishable artifact tied to SUM's
infrastructure rather than a generic Knowledge-Sheaves application.

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

### 5.3 Week 3 — v2 + write-up

- [ ] Implement v2 with text-embedding-3-small (1536-dim) and
  identity restriction maps as default; learned restriction maps as
  optional via the contrastive sheaf-embedding loss.
- [ ] Re-run synthetic benchmark.
- [ ] Run a small *real* benchmark: 20 documents from a hand-curated
  corpus where ground-truth-clean and known-tampered renderings are
  both available. Measure ROC.
- [ ] Write a 2000-word note: *"Cohomological consistency scoring
  for signed bidirectional render receipts."* Include the
  SUM-to-Knowledge-Sheaves mapping (§2.3 above), the
  v1/v2/v3 procedure, the synthetic + real benchmarks, and the
  bounded-claims section verbatim from §6.
- [ ] Submit to arXiv (`cs.AI` primary, `math.CT` secondary).
- [ ] Cross-post to the ACT (Applied Category Theory) discourse,
  the Topos Institute mailing list, and the
  Coecke / Quantinuum DisCoCat / lambeq community.

### 5.4 v3 (post-publication, contingent on v2 passing P1–P3)

If v2 publishes cleanly, v3 (receipt-weighted) becomes the
SUM-specific extension: it doesn't replicate elsewhere because no
other system has cross-runtime-verified render receipts. v3 is the
positioning anchor — *"this is the bit that requires SUM."*

---

## 6. Bounded claims

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
