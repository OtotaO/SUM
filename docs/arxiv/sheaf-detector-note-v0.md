# Sheaf-Laplacian hallucination detection on signed render receipts

**Status: draft v0 (2026-05-01)** — pre-arXiv working note. Audience: ACT / Topos Institute / Quantinuum DisCoCat / categorical-active-inference researchers and engineers building verifiable LLM systems. Not yet submitted; circulating internally and in research forums for review before arXiv submission to `cs.AI` (primary) / `math.CT` (secondary).

---

## Abstract

We describe a sheaf-Laplacian-based hallucination detector that scores
the consistency of LLM-rendered text against a signed source bundle.
The detector lives on top of cross-runtime-verified render receipts
(`sum.render_receipt.v1`, an Ed25519-over-JCS detached-JWS witness
binding triples to tomes through a parameterised rendering functor)
and reduces to three orthogonal signals: (i) per-rendered-triple
score $\|F_h(r)x_h - F_t(r)x_t\|^2$ against a contrastively-trained
knowledge sheaf (Gebhart, Hansen & Schrater 2023), catching entity
substitution, predicate-flip, and off-graph fabrication;
(ii) presence-deficit count of source vertices missing from the
render, catching density-dropout including on disconnected source
graphs; (iii) out-of-vocabulary structural detection at extraction
time. On a 16-document multi-fact corpus with 120 sieve-extracted
source triples and synthetic perturbations, per-class ROC AUC is
1.000 / 1.000 / 1.000 / 0.801 across A1 entity-swap / A2
predicate-flip / A3 off-graph fabrication / A4 triple-drop
(overall mean 0.948). The detector does not solve hallucination;
it provides one mathematically clean signal among many, with three
pinned-in-code falsifications testifying to which signals it does
*not* carry. We position the work inside the program of
Hansen-Ghrist 2019 sheaf-Laplacian spectral theory, Gebhart 2023
contrastive sheaf-embedding training, and Tull-Kleiner-Smithe 2023
categorical active inference. The render-receipt binding is the
SUM-specific contribution — it is what allows the consistency claim
to be cryptographically auditable across independent verifiers.

---

## 1. Introduction

LLM hallucination — the generation of plausible-but-wrong content —
remains the load-bearing reliability problem of generative AI.
Existing detection approaches partition into (broadly) three
families: token-level uncertainty estimation (sequence
log-probability, semantic entropy), retrieval-grounded
verification (RAG with source citation), and consistency checking
across multiple generations. Each family is partial: the first
captures the model's own confidence but is blind to systematic
bias; the second catches retrieval-supported claims but cannot
score what isn't retrievable; the third has no canonical metric
for what counts as "consistent."

The work below contributes to the third family, with a specific
constraint that distinguishes it: the consistency claim is
*cryptographically auditable*. Each rendered text is bound to its
source triples through a signed receipt that any third party can
verify offline. This means: a detector running on a render's
consistency profile produces a verdict that is itself an artifact
in a chain of attestations, not just a heuristic confidence score.

The mathematical machinery we use is not new. Knowledge graphs as
free categories on directed multigraphs (Spivak & Kent 2012);
sheaves on the resulting graph as cellular sheaves (Curry 2014);
the sheaf Laplacian as a spectral consistency measure
(Hansen & Ghrist 2019); contrastive sheaf-embedding training
(Gebhart, Hansen & Schrater 2023). What we add is the *application*
to signed render receipts — and the engineering finding that the
default weighting parameter from a toy 4-fact calibration is
38× too small for naturalistic-corpus use, surfaced by a per-class
ROC bench at scale.

## 2. Mathematical preliminaries

### 2.1 Cellular sheaves on knowledge graphs

Following Curry 2014 and Gebhart, Hansen & Schrater 2023:

**Definition (cellular sheaf, Gebhart 2023 Def. 4).** A cellular
sheaf $\mathcal{F}$ on a directed graph $G = (V, E)$ consists of
a vector space $\mathcal{F}(\sigma)$ for each cell $\sigma$ (vertex
or edge), and linear restriction maps $\mathcal{F}_{v \trianglelefteq e} :
\mathcal{F}(v) \to \mathcal{F}(e)$ for each incident pair. Restriction
maps come in two flavours, head $\mathcal{F}_{v \trianglelefteq_h e}$
and tail $\mathcal{F}_{v \trianglelefteq_t e}$, distinguished only
when $e$ is a self-loop.

For a knowledge graph $G$ instantiating a schema $\mathcal{Q}$,
each entity type carries a vertex stalk and each relation type
carries an edge stalk. Restriction maps $\mathcal{F}_{s \trianglelefteq r}$,
$\mathcal{F}_{t \trianglelefteq r}$ for a relation $r$ from type
$s$ to type $t$ are matrices of shape $d_r \times d_s$, $d_r \times d_t$.

**Definition (cochains).** $C^0(G; \mathcal{F}) = \prod_v \mathcal{F}(v)$
is the space of 0-cochains; $C^1(G; \mathcal{F}) = \prod_e \mathcal{F}(e)$
the space of 1-cochains. The coboundary operator
$\delta : C^0 \to C^1$ acts on edge $e: u \to v$ by
$(\delta x)_e = \mathcal{F}_{v \trianglelefteq e} x_v - \mathcal{F}_{u \trianglelefteq e} x_u$.

**Definition (global sections).** $H^0(G; \mathcal{F}) = \ker \delta$.

### 2.2 The sheaf Laplacian (Hansen & Ghrist 2019)

The degree-0 Hodge Laplacian is $L_{\mathcal{F}} := \delta^T \delta$.
It is symmetric and positive-semidefinite, with kernel
$\ker L_{\mathcal{F}} = H^0(G; \mathcal{F})$ — the space of
global sections. It admits a per-edge factorisation:

$$x^T L_{\mathcal{F}} x \;=\; \|\delta x\|^2 \;=\; \sum_{e = u \sim v \in E}
\|\mathcal{F}_{v \trianglelefteq e} x_v - \mathcal{F}_{u \trianglelefteq e} x_u\|^2$$

Each summand is the squared $\ell^2$ norm of the per-edge residual;
the total quadratic form is zero exactly when $x$ is a global section.
This is the continuous consistency score we will use.

For the d-dim case, $L_{\mathcal{F}}$ is a symmetric block matrix
indexed by vertices, with block at $(v,v)$ equal to
$\sum_{v \trianglelefteq e} \mathcal{F}^{*}_{v \trianglelefteq e}
\mathcal{F}_{v \trianglelefteq e}$ and off-diagonal block at
$(u, v)$ equal to $-\mathcal{F}^{*}_{u \trianglelefteq e}
\mathcal{F}_{v \trianglelefteq e}$ for the edge $e$ between them.
Computing $x^T L_{\mathcal{F}} x$ via the per-edge factorisation
costs $O(|E| \cdot d^2)$ operations and avoids materialising $L$.

### 2.3 Contrastive sheaf embeddings (Gebhart 2023 §4.1)

A *consistent sheaf embedding* is a knowledge sheaf $\mathcal{F}$
together with a 0-cochain $x \in H^0(G; k^{*}\mathcal{F})$ —
i.e., the embedding is in the kernel of $\delta$ for the source
graph's pullback sheaf. To distinguish positive triples from
negatives, Gebhart 2023 Def. 11 introduces the
$\gamma$-gapped contrastive sheaf embedding via the margin ranking loss:

$$L_m = \sum_{(H, \tilde{H}) \in A} \max(0,\, V_{H, \mathcal{F}^H}(x^H) +
\gamma - V_{\tilde{H}, \mathcal{F}^{\tilde{H}}}(x^{\tilde{H}}))$$

where $V_{G, \mathcal{F}}(x) = x^T L_{\mathcal{F}} x$ is the
sheaf Laplacian quadratic form and $A$ is a set of pairs
(positive subgraph, negative subgraph). Training learns
restriction maps $\mathcal{F}_{s \trianglelefteq r}$,
$\mathcal{F}_{t \trianglelefteq r}$ that satisfy positive triples
and reject negatives.

We use stochastic gradient descent on the LCWA negative sampler
(per-positive draw $k$ tail-perturbed negatives) for training.
At $d = 8$ stalk dim with seed_long_paragraphs's 95-relation
vocabulary, training converges in ~200 epochs of CPU computation.

## 3. The detector

### 3.1 Render-receipt binding

We assume a verifiable signed-render system that produces, for each
rendered text $R$:
- the source triple set $T = \{(s_i, p_i, o_i)\}$ from which $R$ was rendered,
- the rendering parameters (e.g., a 5-axis stylistic slider),
- a render receipt $\rho = (\text{schema}, \text{kid}, \text{payload}, \text{jws})$
  signing $(triples\_hash, tome\_hash, sliders, model, kid, signed\_at)$
  with Ed25519 over JCS-canonical bytes.

Concretely: SUM (`sum-engine`, MIT-licensed Apache 2.0; PyPI 0.5.0)
provides the substrate. The cross-runtime trust triangle (Python /
Node / browser WebCrypto, locked by the K-matrix gate on every
release) ensures the receipt-verification claim is realiser-
independent. Other receipt-bearing systems (C2PA-text, future
standards) could substitute.

### 3.2 v1 — 1-dim presence stalks (the baseline)

**Cochain construction.** $\mathcal{F}(v) = \mathcal{F}(e) = \mathbb{R}$;
all restriction maps are identity. For a render with re-extracted
triples $T_n$, the cochain $x_n$ has $x_n[v] = 1$ if $v$ appears in
$T_n$, $0$ otherwise.

**Detection.** $V_n = x_n^T L_{\mathcal{F}} x_n$ measures cross-edge
agreement of entity presence. Empirically, on a 6-fact-set ×
5-perturbation synthetic micro-benchmark, v1 catches A1 entity-swap
6/6, A4 triple-drop 6/6, and (via the *mean* signal over a 3-render
manifold) consistent-A1 6/6, with 100% top-1 edge localisation on
caught cases. Per-class detect rate 18/30; A2 predicate-flip 0/6
(by design — predicates are invisible in $\mathbb{R}^1$ stalks),
A3 off-graph fabrication 0/6 (entities outside the source vertex
set are silently dropped from the cochain).

**Falsification.** On a 4-fact disconnected source graph (real
human-authored prose, sieve-extracted), v1's density-dropout signal
collapses to zero: when a render drops a whole component, both
endpoints vanish, every remaining edge sits in $\{(0,0), (1,1)\}$,
no edge contributes to $V$. The Laplacian quadratic form is
*structurally* a measure of cross-edge agreement, not entity
presence — it cannot detect "facts missing entirely" by design.
Pinned in `test_disconnected_graph_density_dropout_invisible`.

### 3.3 v2.1 — d-dim stalks with learned restriction maps

**Stalks.** $\mathcal{F}(v) = \mathcal{F}(e) = \mathbb{R}^d$ for $d \in \{8, 32, 64\}$.

**Restriction maps.** Per-relation $\mathcal{F}_{h \trianglelefteq r},
\mathcal{F}_{t \trianglelefteq r} \in \mathbb{R}^{d \times d}$ trained
under the $\gamma$-gapped contrastive loss above on the source
bundle's triples (LCWA tail-perturbation negatives).

**Cochain construction (presence-style).** $x_n[v] = $ trained entity
embedding if $v \in T_n$, else zero.

**Falsification (pinned).** On the same disconnected-graph corpus,
v2.1 with presence-style cochains *also* misses dropout: when a
component vanishes, both endpoints zero out, and the trained
restriction maps are then multiplied by zero on both sides — the
per-edge residual at the dropped component vanishes regardless of
how $\mathcal{F}$ was trained. The Laplacian's category mismatch
is unchanged; learned restriction maps amplify the contributions of
*present* entities, not the absence of absent ones.

### 3.4 v2.2 — combined detector

The fix is orthogonal-signal composition rather than Laplacian
modification. Define:

$$V_{\text{combined}}(x) \;=\; \|\delta x\|^2 \;+\; \lambda \cdot
(\text{presence\_deficit})^2$$

where presence_deficit is the count of source vertices not appearing
in the render. The Laplacian term carries the relation-aware signal;
the deficit term carries the presence-pattern signal. Combining them
is the publishable artifact, not a workaround.

**Principled $\lambda$ calibration.** Per a corpus-scale ROC bench
(see §4), the default $\lambda = 0.05$ — calibrated on a 4-fact
toy graph where the Laplacian magnitude is ~0.4 — is 38× too small
for corpora where the Laplacian magnitude is 9–21 per doc. The
principled fix:

$$\lambda_{\text{auto}} \;=\; \frac{1}{|D|}\sum_{d \in D}
\frac{V^{(d)}_{\text{clean\_laplacian}}}{|E_d|}$$

— the mean over docs of the per-edge Laplacian contribution. On
seed_long_paragraphs this gives $\lambda \approx 1.92$. Auto-
calibration recovers A4 detection from anti-correlation (AUC
0.36) to clean signal (AUC 0.80).

### 3.5 Per-rendered-triple scoring (A2 / A3)

Independent of the cochain-on-source-graph machinery, we also
score each rendered triple $(h, r, t)$ individually against the
trained sheaf:

- If $r$ is not in the trained relation vocabulary: out-of-vocab signal (A3 catch).
- If $h$ or $t$ is not in the trained vertex set: out-of-vocab signal (A3 catch).
- Otherwise: $V_{\text{triple}} = \|\mathcal{F}_{h \trianglelefteq r}
  \mathrm{emb}(h) - \mathcal{F}_{t \trianglelefteq r} \mathrm{emb}(t)\|^2$
  — small for trained-positive triples, large for predicate-flips
  and other relation-violating claims.

Importantly, the contrastive training samples only *tail*
perturbations as negatives; predicate-flips were not in the negative
set. Empirically the trained restriction maps nevertheless
distinguish predicate-flips strongly: on 4 clean/flipped pairs from
a 4-triple training set with two relations, V ratios were
**125×, 9×, 40×, 9×.** The contrastive loss generalises beyond its
sampling distribution.

## 4. Empirical results

### 4.1 Synthetic micro-benchmark (connected graphs)

Six hand-built fact-sets × five perturbation classes = 30 trials.
v1 catches 18/30 entity-presence-affecting perturbations with 100%
top-1 localisation on caught classes. Predicate-flip and off-graph
fabrication are 0/6 each — by design.

### 4.2 Disconnected-graph falsification

A 4-fact disconnected source graph (4 unrelated facts about 4
entity-pairs) from human-authored paraphrase data exposed v1's
structural blindspot. v2.1 with presence-style cochains
*also* misses, surfacing the deeper category mismatch. Both
falsifications are pinned in code.

### 4.3 ROC benchmark on a 16-document corpus

We sieve-extracted triples from each of 16 multi-fact paragraphs
(seed_long_paragraphs corpus, 120 source triples, 229-entity /
95-relation transductive vocabulary), trained one v2.1 sheaf on
the union, generated four perturbation classes per doc (A1 entity-
swap, A2 predicate-flip, A3 off-graph fabrication, A4 triple-drop),
scored, and computed per-class ROC AUC.

| Class | AUC | Detection signal |
|---|---|---|
| A1 entity-swap | **1.000** | max in-vocab $V$ from per-rendered-triple scoring |
| A2 predicate-flip | **1.000** | max in-vocab $V$ from per-rendered-triple scoring |
| A3 off-graph fabrication | **1.000** | n_oov from per-rendered-triple scoring |
| A4 triple-drop | **0.801** | combined-detector $V$ (Laplacian + auto-λ deficit) |
| **Overall (mean)** | **0.948** | |

The receipt is at `fixtures/bench_receipts/sheaf_v2_roc_seed_long_paragraphs_2026-05-01.json`;
reproducible via `PYTHONPATH=. python scripts/research/sheaf_v2_roc_bench.py`.

## 5. Position vs. existing work

**vs. token-level uncertainty (sequence log-probability, semantic
entropy):** complementary. Uncertainty estimates measure the
model's own confidence; we measure cross-claim consistency under
a verifiable-source binding. The two should compose.

**vs. retrieval-grounded verification (RAG with citations):**
complementary. RAG confirms support for retrievable claims; we score
consistency of any rendered claim against an attested source bundle,
including claims whose support is in the sheaf but not in a
retrieval index.

**vs. existing knowledge-graph hallucination detectors (HalluGraph,
KG-grounded checks):** structurally adjacent, with two
distinguishing features. First, the *math* is grounded in
Hansen-Ghrist sheaf Laplacian theory and Gebhart contrastive
sheaf-embedding training, not ad-hoc consistency rules. Second, the
*provenance* is cryptographic — the source bundle is a signed
artifact whose verifier is realiser-independent (Python / Node /
browser byte-identity locked in CI). The detector's verdict is
itself part of an audit trail, not a vendor-locked confidence score.

**vs. Tull, Kleiner & Smithe 2023 categorical active inference:**
the compositionality theorem (Theorem 45/46) says free energy is
additive across sequential and parallel composition of generative
models. Our v2.2 combined detector is mathematically a sum of
two orthogonal terms — the same shape — over per-doc subgraphs
of a multi-document corpus. Federated multi-agent scoring is the
natural application; v3 (receipt-weighted, harmonic extension over
trusted-issuer boundary; Hansen-Ghrist Prop. 4.1 / Thm. 4.5) is
the SUM-specific instantiation.

## 6. Bounded claims

What the detector claims:

- **Specific:** the per-class detection signals defined above
  separate clean from perturbed at the AUCs reported in §4 on the
  synthetic and seed_long_paragraphs corpora.
- **Localised:** per-edge / per-rendered-triple discrepancy
  identifies the perturbed claim at high precision (100% top-1 on
  the synthetic micro-benchmark; the per-rendered-triple V signal
  on A1/A2/A3 reports the offending triple directly).
- **Cryptographically-auditable:** because the source binding is a
  signed receipt, the detector's verdict can be reproduced offline
  by any verifier with the source bundle and the trained sheaf
  parameters — no vendor-locked confidence score.

What the detector does *not* claim:

- **Not** a solution to hallucination. One signal among many; in
  practice would compose with confidence calibration, retrieval-
  grounded checks, NLI verifiers, and human review.
- **Not** a correctness proof. It detects inconsistency in the
  *output manifold* under cross-edge agreement and presence
  patterns; it does not certify that any single output is *correct*.
  A consistent manifold of uniformly-wrong renderings (where the
  trained sheaf has no internal contradiction with the wrong claims)
  remains undetected.
- **Not** generalising across all knowledge-graph schemas without
  re-calibration. The $\lambda$ auto-calibration is per-corpus;
  schema-typed sheaves (multiple entity types, typed restriction
  maps) require additional design work.
- **Not** computable on arbitrary corpus sizes. Sparse-block storage
  of the d-dim Laplacian is straightforward but not yet implemented;
  scaling to $> 10^4$ vertices needs the sparsification machinery
  of Hansen-Ghrist 2019 §6 (Theorem 6.4).

Three pinned-in-code falsifications testify to which signals the
detector does not carry: (a) v1 disconnected-graph density-dropout
blindness; (b) v2.1 presence-cochain inheritance of the same
blindness; (c) v2.2 default $\lambda$ at corpus scale. The fixes
are documented in §3 and tested at every PR.

## 7. Future work

- **Path 2 — real LLM-rendered adversarial bench.** Synthetic
  perturbations are existence-proofs; adversarial LLM renderings
  stress the detector differently. Generate clean and adversarial
  variants via the hosted Worker render path and re-run §4.3.

- **v3 — harmonic extension over trusted-issuer boundary.**
  Hansen-Ghrist Proposition 4.1 / Theorem 4.5: given boundary
  values on a subset $B \subseteq V$, the harmonic extension to
  $V \setminus B$ is unique. Receipts from a trusted-issuer JWKS
  form $B$; untrusted renders form the interior; renders that
  match the harmonic extension pass, those that don't are flagged.
  This is the SUM-specific instantiation of the categorical-
  active-inference framing — no other system has cross-runtime-
  verified render receipts to seed the boundary.

- **Compositionality at scale.** Tull-Kleiner-Smithe Thm. 45/46
  give per-component additivity of free energy under sequential
  and parallel composition. Our combined detector $V = \|\delta x\|^2 +
  \lambda d^2$ has the same shape; for a multi-doc corpus, per-doc
  scores compose to a corpus-level score with bounded total. The
  free-energy compositionality theorem is the rigorous statement
  of "minimising local consistency achieves global consistency."

- **Schema-typed sheaves.** Multi-type entity schemas (people /
  films / events) with typed restriction maps; Spivak/Kent
  ologs as the schema language.

- **Multi-source connectors.** Each external source of authority
  (Wikidata, DOI registry, ORCID, regulatory text) provides a
  partial sheaf; the combined sheaf over a federation of sources
  is the Grothendieck-topology gluing. Compliance-regime tags
  (GDPR, HIPAA, EU AI Act) become per-regime predicates with
  required-field validators.

## References

- **Curry, J.** (2014). *Sheaves, Cosheaves, and Applications.*
  PhD thesis, University of Pennsylvania.
- **Gebhart, T., Hansen, J., & Schrater, P.** (2023). *Knowledge
  Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding.*
  AISTATS 2023, PMLR 206. arXiv:2110.03789.
- **Hansen, J. & Ghrist, R.** (2019). *Toward a spectral theory of
  cellular sheaves.* Journal of Applied and Computational Topology
  3(4):315–358. arXiv:1808.01513.
- **Tull, S., Kleiner, J., & Smithe, T. St C.** (2023). *Active
  Inference in String Diagrams: A Categorical Account of Predictive
  Processing and Free Energy.* arXiv:2308.00861.
- **Bordes, A., Weston, J., Collobert, R., & Bengio, Y.** (2011).
  *Learning Structured Embeddings of Knowledge Bases.* AAAI.
- **Spivak, D. I. & Kent, R. E.** (2012). *Ologs: A Categorical
  Framework for Knowledge Representation.* PLOS ONE 7(1):e24274.
- **Coecke, B., Sadrzadeh, M., & Clark, S.** (2010). *Mathematical
  Foundations for a Compositional Distributional Model of Meaning.*
  Linguistic Analysis 36.
- **C2PA** (2023+). *Content Authenticity Coalition* technical
  specifications, including `digital_source_type` ontology.
- **OtotaO/SUM repository.** `sum-engine` v0.5.0 on PyPI;
  source at https://github.com/OtotaO/SUM. Spec doc at
  `docs/SHEAF_HALLUCINATION_DETECTOR.md`. Bench scripts at
  `scripts/research/sheaf_v2_roc_bench.py`.

---

*Acknowledgements.* This work would not exist without the
foundational categorical-AI program of Spivak, Kent, Coecke,
Curry, Hansen, Ghrist, Gebhart, Schrater, Tull, Kleiner, Smithe,
and the broader Topos Institute / Quantinuum / ACT community.
SUM contributes the cryptographic substrate and the engineering
plumbing; the mathematics is theirs.

*Reproducibility.* All code Apache-2.0, all benchmarks reproducible
by `pip install 'sum-engine[research]'` and the bench scripts in
the repository. Receipt JSON schemas are stable and versioned
(`sum.render_receipt.v1`, `sum.sheaf_v2_roc_bench.v1`).

*Status of claims.* §4.1 (synthetic micro-bench) and §4.3 (ROC
bench on seed_long_paragraphs) are measured, pinned, and
reproducible at the commit hash of this draft. §4.2's
falsifications are pinned in code by named regression tests.
§5 positioning claims are this author's reading of the cited
literature; corrections welcomed before arXiv submission.

*Authors and contact.* Draft authored 2026-05-01 by the
SUM project. For corrections / contributions:
https://github.com/OtotaO/SUM/issues. Pre-arXiv comments welcome
on the ACT discourse and the Topos Institute mailing list.
