# Sheaf-Laplacian hallucination detection on signed render receipts

**Status: draft v0.1 (2026-05-04)** — pre-arXiv working note. Audience:
applied-ML and security/cryptography researchers and engineers building
verifiable LLM systems; categorical-AI / Topos Institute / Quantinuum
DisCoCat / categorical-active-inference readers as a secondary
audience. Targeted submission: `cs.LG` (primary) / `cs.CR` (secondary).
Pre-circulation to 1–2 readers before submission.

---

## Abstract

We describe a sheaf-Laplacian-based hallucination detector that scores
the consistency of LLM-rendered text against a signed source bundle.
The detector lives on top of cross-runtime-verified render receipts
(`sum.render_receipt.v1`, an Ed25519-over-JCS detached-JWS witness
binding source triples to rendered tomes through a parameterised
rendering functor) and reduces, in its current form, to three
orthogonal signals: (i) the weighted sheaf Laplacian quadratic form
$x^T L_\mathcal{F}^w x$ over a contrastively-trained knowledge sheaf
(Gebhart, Hansen & Schrater 2023), with per-edge weights driven by
each source-of-record's render-receipt trust status against a JWKS
allow/revocation list (Hansen & Ghrist 2019 §3.2 weighted
generalization); (ii) the harmonic-extension boundary deviation
$\|x_I - x_I^*\|^2$ over the (trusted, untrusted) partition (Hansen
& Ghrist 2019 Prop. 4.1 / Thm. 4.5); (iii) per-rendered-triple
out-of-vocabulary structural detection at extraction time.

Three findings from corpus-scale evaluation on
`seed_long_paragraphs` (16 documents, 120 source triples) are
load-bearing for the design:

1. **Receipt-weighting helps modestly** at corpus scale (F1
   verdict: MARGINAL, $\Delta = +0.034$ trusted-mean AUC vs the
   v2.2 baseline) and does not collapse on untrusted-target
   perturbations (F2 verdict: PASS).
2. **Boundary deviation as a standalone signal fails structurally**
   at corpus scale (F3 verdict: STRUCTURAL FAIL — trusted-mean AUC
   $0.499$). A 2×2×2 diagnostic over (graph_size,
   cochain_strategy, partition_strategy) shows none of the three
   parameter axes is load-bearing: the failure is mathematical
   (under $L_{IB} = 0$ topology, boundary perturbations are
   invisible to the harmonic extension by linear algebra), not
   parametric. We name this STRUCTURAL FAIL rather than "needs
   tuning" — the synthetic single-edge utility test (H12) passed,
   but the corpus bench refuted the claim, and naming the
   refutation honestly is what the substrate's truth-first
   discipline requires.
3. **Combining the two signals closes F3** at the detector layer
   (v3.2): $v_\text{combined} = v_\text{laplacian}^w + \gamma \cdot
   \text{deviation}_w + \lambda \cdot v_\text{deficit}$ produces
   trusted-mean AUC $\geq 0.55$ at every $\gamma$ tested (F4
   PASS) and stays within $-0.02$ of v3 at $\gamma \leq 0.1$ (F5
   PASS). The magnitude-matching auto-calibration heuristic
   ($\gamma_\text{auto} \approx 1.02$) is **empirically wrong on
   this corpus** — F5 fails at $\gamma \in \{1.0, \text{auto}\}$;
   deviation's signal-to-noise ratio is below what its magnitude
   suggests.

The detector does not solve hallucination; it provides one
mathematically clean signal among many, with five named
falsification verdicts (F1 MARGINAL, F2 PASS, F3 STRUCTURAL FAIL,
F4 PASS, F5 PASS at $\gamma \leq 0.1$) testifying to which signals
it does and does *not* carry.

The SUM-specific contribution is the *substrate* — render
receipts as cryptographic anchors for the detector's edge weights;
a bench-reproducibility primitive (`bench_digest`, JCS-canonical
SHA-256 over the quantized payload, signable with the project's
existing JWKS keys); and a six-regime compliance evidence layer
(EU AI Act Art 12, GDPR Art 30, HIPAA § 164.312(b), ISO 27001
A.8.15, SOC 2 CC7.2, PCI DSS v4.0 Req 10) demonstrating that the
rendering pipeline whose outputs the detector scores carries
audit-grade record-keeping by construction.

We position the work inside the program of Hansen-Ghrist 2019
sheaf-Laplacian spectral theory, Gebhart 2023 contrastive
sheaf-embedding training, and Tull-Kleiner-Smithe 2023 categorical
active inference. The bench digest and the JWKS-signed render
receipts together constitute "reproducible-research-with-
cryptographic-teeth" — claims that an external reader can re-run
locally and verify byte-for-byte against the published digest.

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

The work below contributes to the third family, with two
constraints that distinguish it. First, the consistency claim is
*cryptographically auditable*: each rendered text is bound to its
source triples through a signed receipt that any third party can
verify offline against a published JWKS. A detector running on a
render's consistency profile produces a verdict that is itself an
artifact in a chain of attestations, not just a heuristic
confidence score. Second, the consistency claim is
*reproducibility-anchored*: every numeric result in this note is
captured in a versioned `fixtures/bench_receipts/*.json` file
carrying a `bench_digest` field — a JCS-canonical SHA-256 over
the quantized payload that an external reader can match
byte-for-byte after re-running the bench locally.

The mathematical machinery is not new. Knowledge graphs as free
categories on directed multigraphs (Spivak & Kent 2012); sheaves
on the resulting graph as cellular sheaves (Curry 2014); the sheaf
Laplacian as a spectral consistency measure (Hansen & Ghrist 2019);
the weighted generalization (§3.2 of the same paper); the
harmonic extension over a (boundary, interior) partition (Prop.
4.1 / Thm. 4.5); contrastive sheaf-embedding training (Gebhart,
Hansen & Schrater 2023). What we add is the *application* to
signed render receipts, the corpus-scale evaluation that surfaces
both a viable detector (v3) and an honest negative result (v3.1
F3 STRUCTURAL FAIL), the F3 closer (v3.2) that combines the two
signals, and the engineering discipline that pins every claim to
either a mechanical test, a digest-anchored measurement, or an
explicit boundary.

## 2. Mathematical preliminaries

### 2.1 Cellular sheaves on knowledge graphs

Following Curry 2014 and Gebhart, Hansen & Schrater 2023:

**Definition (cellular sheaf, Gebhart 2023 Def. 4).** A cellular
sheaf $\mathcal{F}$ on a directed graph $G = (V, E)$ consists of
a vector space $\mathcal{F}(\sigma)$ for each cell $\sigma$
(vertex or edge), and linear restriction maps
$\mathcal{F}_{v \trianglelefteq e} : \mathcal{F}(v) \to \mathcal{F}(e)$
for each incident pair. Restriction maps come in two flavours,
head $\mathcal{F}_{v \trianglelefteq_h e}$ and tail
$\mathcal{F}_{v \trianglelefteq_t e}$, distinguished only when
$e$ is a self-loop.

For a knowledge graph $G$ instantiating a schema $\mathcal{Q}$,
each entity type carries a vertex stalk and each relation type
carries an edge stalk. Restriction maps
$\mathcal{F}_{s \trianglelefteq r}$,
$\mathcal{F}_{t \trianglelefteq r}$ for a relation $r$ from type
$s$ to type $t$ are matrices of shape $d_r \times d_s$,
$d_r \times d_t$.

**Definition (cochains).**
$C^0(G; \mathcal{F}) = \prod_v \mathcal{F}(v)$
is the space of 0-cochains;
$C^1(G; \mathcal{F}) = \prod_e \mathcal{F}(e)$ the space of
1-cochains. The coboundary operator $\delta : C^0 \to C^1$ acts
on edge $e: u \to v$ by
$(\delta x)_e = \mathcal{F}_{v \trianglelefteq e} x_v -
\mathcal{F}_{u \trianglelefteq e} x_u$.

**Definition (global sections).**
$H^0(G; \mathcal{F}) = \ker \delta$.

### 2.2 The sheaf Laplacian (Hansen & Ghrist 2019)

The degree-0 Hodge Laplacian is
$L_{\mathcal{F}} := \delta^T \delta$. It is symmetric and
positive-semidefinite, with kernel
$\ker L_{\mathcal{F}} = H^0(G; \mathcal{F})$ — the space of
global sections. It admits a per-edge factorisation:

$$x^T L_{\mathcal{F}} x \;=\; \|\delta x\|^2 \;=\; \sum_{e = u \sim v \in E}
\|\mathcal{F}_{v \trianglelefteq e} x_v -
\mathcal{F}_{u \trianglelefteq e} x_u\|^2$$

Each summand is the squared $\ell^2$ norm of the per-edge
residual; the total quadratic form is zero exactly when $x$ is a
global section. This is the continuous consistency score we will
use.

For the $d$-dim case, $L_{\mathcal{F}}$ is a symmetric block matrix
indexed by vertices, with block at $(v,v)$ equal to
$\sum_{v \trianglelefteq e} \mathcal{F}^{*}_{v \trianglelefteq e}
\mathcal{F}_{v \trianglelefteq e}$ and off-diagonal block at
$(u, v)$ equal to $-\mathcal{F}^{*}_{u \trianglelefteq e}
\mathcal{F}_{v \trianglelefteq e}$ for the edge $e$ between them.
Computing $x^T L_{\mathcal{F}} x$ via the per-edge factorisation
costs $O(|E| \cdot d^2)$ operations and avoids materialising $L$.

### 2.3 The weighted sheaf Laplacian (Hansen & Ghrist 2019 §3.2)

Hansen & Ghrist (2019) generalise the Laplacian to allow per-edge
weights: given a non-negative diagonal $|E| \times |E|$ matrix
$W$, the weighted Laplacian is
$L_{\mathcal{F}}^w := \delta^T W \delta$. Equivalently,
$L_{\mathcal{F}}^w = (W^{1/2}\delta)^T (W^{1/2}\delta)$, so
$L_{\mathcal{F}}^w$ is itself the (unweighted) Laplacian of the
sheaf with stalks scaled by $W^{1/2}$ on the edge side; symmetry
and positive-semidefiniteness carry through. The per-edge
factorisation becomes

$$x^T L_{\mathcal{F}}^w x \;=\; \sum_{e = u \sim v \in E}
w_e \cdot
\|\mathcal{F}_{v \trianglelefteq e} x_v -
\mathcal{F}_{u \trianglelefteq e} x_u\|^2.$$

Mechanically pinned: linearity in weights and zero-weight
annihilation in
`Tests/research/test_sheaf_laplacian_v3.py::test_h1_doubling_weights_doubles_quadratic_form`
and
`test_h1_zero_weight_kills_edge_contribution`; uniform-weight
reduction to v2 in `test_h2_uniform_weights_v3_equals_scaled_v2`.

### 2.4 Harmonic extension over a (boundary, interior) partition

Hansen & Ghrist 2019 Proposition 4.1 / Theorem 4.5: given a
sheaf $\mathcal{F}$ on $G$, a vertex partition
$V = B \sqcup I$, and a cochain $x_B \in \prod_{v \in B}
\mathcal{F}(v)$, the **harmonic extension** is the unique cochain
$x \in C^0(G; \mathcal{F})$ that

  (i)  agrees with $x_B$ on $B$, and
  (ii) minimises $\|\delta x\|^2$ over the interior $I$.

Block-decomposing $L_{\mathcal{F}}$ by the $(B, I)$ partition:

$$L_{\mathcal{F}} = \begin{bmatrix} L_{BB} & L_{BI} \\ L_{IB} & L_{II} \end{bmatrix},$$

setting $\partial \|\delta x\|^2 / \partial x_I = 0$ gives the
closed-form interior cochain

$$x_I^* = -L_{II}^{-1} L_{IB} \, x_B$$

when $L_{II}$ is invertible. (Our implementation uses
`numpy.linalg.lstsq`, so a rank-deficient $L_{II}$ — disconnected
interior, or interior carrying a global section — yields the
minimum-norm solution rather than crashing.) Mechanically pinned:
defining minimisation property in
`Tests/research/test_sheaf_laplacian_v31.py::test_harmonic_extension_minimizes_v_subject_to_boundary_constraint`;
boundary preservation, uniqueness, and full-boundary degeneracy
in adjacent tests.

The signal we extract from this is the *boundary deviation*

$$\text{deviation}_w(x; B) \;=\; \|x_I - x_I^*\|^2,$$

i.e. the gap between the actual interior cochain and the unique
minimum-energy interior cochain consistent with the given
boundary. A render whose interior matches the boundary's
harmonic extension is *consistent with the trust frame*; one
that diverges is flagged.

### 2.5 Contrastive sheaf embeddings (Gebhart 2023 §4.1)

A *consistent sheaf embedding* is a knowledge sheaf $\mathcal{F}$
together with a 0-cochain $x \in H^0(G; k^{*}\mathcal{F})$ —
i.e., the embedding is in the kernel of $\delta$ for the source
graph's pullback sheaf. To distinguish positive triples from
negatives, Gebhart 2023 Def. 11 introduces the
$\gamma$-gapped contrastive sheaf embedding via the margin
ranking loss:

$$L_m = \sum_{(H, \tilde{H}) \in A} \max(0,\, V_{H, \mathcal{F}^H}(x^H) +
\gamma_\text{margin} - V_{\tilde{H}, \mathcal{F}^{\tilde{H}}}(x^{\tilde{H}}))$$

where $V_{G, \mathcal{F}}(x) = x^T L_{\mathcal{F}} x$ is the
sheaf Laplacian quadratic form and $A$ is a set of pairs
(positive subgraph, negative subgraph). Training learns
restriction maps $\mathcal{F}_{s \trianglelefteq r}$,
$\mathcal{F}_{t \trianglelefteq r}$ that satisfy positive triples
and reject negatives.

We use stochastic gradient descent on the LCWA negative sampler
(per-positive draw $k$ tail-perturbed negatives) for training.
At $d = 8$ stalk dim with the corpus's 95-relation vocabulary,
training converges in ~200 epochs of CPU computation.

(N.B. — we use $\gamma_\text{margin}$ for the contrastive-loss
margin and $\gamma$ for the v3.2 deviation-mixing weight in §3.8.
The two are unrelated.)

## 3. The detector

### 3.0 Threat model

The detector and substrate together address a specific, narrow threat
model. Stating it precisely is necessary because vague threat-model
descriptions are the most common failure mode in adjacent literature.

We consider four attacker capabilities:

  - **T1. Adversarial render.** Attacker controls the LLM rendering
    process — prompt injection, jailbreak, fine-tuned adversarial
    model — but does NOT control the source bundle, the signing key,
    or the verifier.
    *Defence:* the consistency signal of §3.6–§3.8. A render that
    diverges from its bound source bundle scores higher; the receipt
    binds render output to the source-bundle hash so the binding
    itself cannot be retroactively edited without invalidating the
    JWS.

  - **T2. Adversarial source bundle.** Attacker controls the source
    bundle BEFORE signing — i.e., the attacker is the rendering
    operator and chooses what to attest. The receipt is valid; the
    bundle's claims are the attacker's choice.
    *Defence:* partial. The detector helps only if the rendered
    claims diverge from the attacker's own bundle (catches sloppy
    attackers, not careful ones). Substrate response: this is where
    the compliance-regime layer of §5 carries weight — a careful
    attacker controlling the source bundle still produces an audit
    trail whose provenance is verifiable, shifting the trust
    question from "is this claim true" to "who attested it."

  - **T3. Stolen signing key.** Attacker possesses a previously-
    trusted JWKS private key.
    *Defence:* revocation list at `/.well-known/revoked-kids.json`.
    v3 receipt-weighting sets $w_e = 0$ for revoked keys; revoked-
    key edges contribute nothing to the Laplacian quadratic form.
    Revocation latency is the residual gap (operator-dependent;
    out of scope for this preprint).

  - **T4. Compromised verifier.** Attacker controls the verifying
    client.
    *Defence: OUT OF SCOPE.* Trust roots are operator-provisioned;
    the cross-runtime trust triangle (§3.1) attests realiser-
    independence among honest verifiers but does not defend against
    a verifier whose code has been replaced.

The detector's contribution is concentrated in T1; the receipt
substrate's contribution is concentrated in T2 and T3. Neither
addresses T4.

### 3.1 Render-receipt binding

We assume a verifiable signed-render system that produces, for
each rendered text $R$:

- the source triple set $T = \{(s_i, p_i, o_i)\}$ from which $R$
  was rendered,
- the rendering parameters (e.g., a 5-axis stylistic slider),
- a render receipt
  $\rho = (\text{schema}, \text{kid}, \text{payload}, \text{jws})$
  signing $(triples\_hash, tome\_hash, sliders, model, kid,
  signed\_at)$ with Ed25519 over JCS-canonical bytes (RFC 8032
  signature; RFC 8785 canonicalisation; RFC 7515 §A.5 detached-JWS
  envelope; public keys distributed per RFC 7517 JWKS).

Concretely: SUM (`sum-engine`, MIT-licensed; PyPI 0.5.0) provides
the substrate. The cross-runtime trust triangle (Python / Node /
browser WebCrypto, locked by the K-matrix gate on every release)
ensures the receipt-verification claim is realiser-independent.
Other receipt-bearing systems (C2PA-text, future standards) could
substitute.

### 3.2 v1 — 1-dim presence stalks (the baseline)

**Cochain construction.** $\mathcal{F}(v) = \mathcal{F}(e) =
\mathbb{R}$; all restriction maps are identity. For a render with
re-extracted triples $T_n$, the cochain $x_n$ has $x_n[v] = 1$
if $v$ appears in $T_n$, $0$ otherwise.

**Detection.** $V_n = x_n^T L_{\mathcal{F}} x_n$ measures
cross-edge agreement of entity presence. Empirically, on a
6-fact-set × 5-perturbation synthetic micro-benchmark, v1 catches
A1 entity-swap 6/6, A4 triple-drop 6/6, and (via the *mean*
signal over a 3-render manifold) consistent-A1 6/6, with 100 %
top-1 edge localisation on caught cases. Per-class detect rate
18/30; A2 predicate-flip 0/6 (by design — predicates are
invisible in $\mathbb{R}^1$ stalks), A3 off-graph fabrication
0/6 (entities outside the source vertex set are silently dropped
from the cochain).

**Falsification.** On a 4-fact disconnected source graph (real
human-authored prose, sieve-extracted), v1's density-dropout
signal collapses to zero: when a render drops a whole component,
both endpoints vanish, every remaining edge sits in $\{(0,0),
(1,1)\}$, no edge contributes to $V$. The Laplacian quadratic
form is *structurally* a measure of cross-edge agreement, not
entity presence — it cannot detect "facts missing entirely" by
design. Pinned in
`test_disconnected_graph_density_dropout_invisible`.

### 3.3 v2.1 — d-dim stalks with learned restriction maps

**Stalks.** $\mathcal{F}(v) = \mathcal{F}(e) = \mathbb{R}^d$ for
$d \in \{8, 32, 64\}$.

**Restriction maps.** Per-relation
$\mathcal{F}_{h \trianglelefteq r}, \mathcal{F}_{t \trianglelefteq r}
\in \mathbb{R}^{d \times d}$ trained under the
$\gamma_\text{margin}$-gapped contrastive loss above on the
source bundle's triples (LCWA tail-perturbation negatives).

**Cochain construction (presence-style).** $x_n[v] = $ trained
entity embedding if $v \in T_n$, else zero.

**Falsification (pinned).** On the same disconnected-graph
corpus, v2.1 with presence-style cochains *also* misses dropout:
when a component vanishes, both endpoints zero out, and the
trained restriction maps are then multiplied by zero on both
sides — the per-edge residual at the dropped component vanishes
regardless of how $\mathcal{F}$ was trained. The Laplacian's
category mismatch is unchanged; learned restriction maps amplify
the contributions of *present* entities, not the absence of
absent ones.

### 3.4 v2.2 — combined detector (deficit + Laplacian)

The fix for v2.1's blindspot is orthogonal-signal composition
rather than Laplacian modification. Define:

$$V_{\text{combined}}^{v2.2}(x) \;=\; \|\delta x\|^2 \;+\;
\lambda \cdot (\text{presence\_deficit})^2$$

where presence_deficit is the count of source vertices not
appearing in the render. The Laplacian term carries the
relation-aware signal; the deficit term carries the
presence-pattern signal. Combining them is the publishable
artifact, not a workaround.

**Principled $\lambda$ calibration.** Per a corpus-scale ROC
bench (see §4.3), the default $\lambda = 0.05$ — calibrated on a
4-fact toy graph where the Laplacian magnitude is ~0.4 — is
38× too small for corpora where the Laplacian magnitude is
9–21 per doc. The principled fix:

$$\lambda_\text{auto} \;=\; \frac{1}{|D|}\sum_{d \in D}
\frac{V^{(d)}_\text{clean\_laplacian}}{|E_d|}$$

— the mean over docs of the per-edge Laplacian contribution. On
`seed_long_paragraphs` this gives $\lambda \approx 1.92$. Auto-
calibration recovers A4 detection from anti-correlation (AUC
0.36) to clean signal (AUC 0.80).

### 3.5 Per-rendered-triple scoring (A2 / A3)

Independent of the cochain-on-source-graph machinery, we also
score each rendered triple $(h, r, t)$ individually against the
trained sheaf:

- If $r$ is not in the trained relation vocabulary: out-of-vocab
  signal (A3 catch).
- If $h$ or $t$ is not in the trained vertex set: out-of-vocab
  signal (A3 catch).
- Otherwise: $V_\text{triple} = \|\mathcal{F}_{h \trianglelefteq r}
  \mathrm{emb}(h) - \mathcal{F}_{t \trianglelefteq r} \mathrm{emb}(t)\|^2$
  — small for trained-positive triples, large for predicate-flips
  and other relation-violating claims.

Importantly, the contrastive training samples only *tail*
perturbations as negatives; predicate-flips were not in the
negative set. Empirically the trained restriction maps
nevertheless distinguish predicate-flips strongly on a small
worked example: on 4 clean/flipped pairs from a 4-triple training
set with two relations, $V$ ratios were **125×, 9×, 40×, 9×**.
The contrastive loss generalises beyond its sampling distribution
*on small worked examples*. On the 95-relation
`seed_long_paragraphs` vocabulary the same generalisation does
not appear at corpus scale (see §4.3 — A2 sits at AUC 0.50 across
every detector); see also §7 bounded claims and §8 future work.

### 3.6 v3 — receipt-weighted Laplacian

v3 instantiates §2.3's weighted Laplacian by deriving each edge's
weight from whether that edge's source-of-record carries a
verified Ed25519-signed render receipt:

| Receipt status | Weight |
|---|---|
| Signed by a key in the trusted-issuer JWKS | $w_\text{trusted} = 1.0$ |
| Unsigned / unknown issuer | $w_\text{default} = 0.1$ |
| Signed by a revoked key (per `/.well-known/revoked-kids.json`) | $w_\text{revoked} = 0.0$ |

The fractal property worth naming: the weights come from the
system's own trust artifacts. The cross-runtime trust triangle
(K1–K4 in SUM's release-gate matrix) attests that a receipt's
Ed25519 signature is byte-identically verifiable in Python, Node,
and the browser. v3 takes those receipts and feeds them into the
detector's confidence weighting. The audit-log substrate (see §5)
records every render's receipt KID, so backfilling weights from a
logged history is straightforward. Higher trust → higher weight
→ sharper detection signal in regions the system already
verifies; unsigned regions get a lower-weight floor that doesn't
silence them entirely.

**Falsifiable predictions pinned in code**
(`Tests/research/test_sheaf_laplacian_v3.py`):

  - **H1 (linearity).** $V$ is linear in the weights — doubling
    all weights doubles $V$; setting one edge's weight to 0 zeros
    that edge's contribution exactly.
  - **H2 (v2 reduction).** Uniform weights $w_e = c$ give
    $V_{v3}(x; w=c) = c \cdot V_{v2}(x)$. v3 is a strict
    generalisation of v2.
  - **H3 (per-edge weighting).** The localisation ranker's
    per-edge contribution scales with weight ($w_e \cdot
    \|\text{residual}_e\|^2$).
  - **H4 (trust amplifies signal).** Tampering a trusted edge
    yields a sharper $\Delta V$ than tampering an untrusted edge
    — receipt-weighting amplifies signal where the system
    already trusts. **This is the utility claim**; if it inverts,
    v3 is well-defined but useless.
  - **H5 (revocation overrides trust).** An edge in both the
    trusted and revoked sets resolves to revoked.

### 3.7 v3.1 — harmonic-extension boundary deviation

v3.1 instantiates §2.4's harmonic extension as a hallucination
signal. Trusted-receipt-backed vertices (those whose every
incident edge is signed by a known-issuer JWKS key) form the
boundary $B$; the rest fall to the interior $I$.
`boundary_from_weights(w, threshold)` realises this map: a vertex
is on the boundary iff every incident edge has weight $\geq$
threshold.

Given a render's cochain $x$, restrict it to the boundary,
compute the harmonic extension $x_I^*$ on the interior, and
report

$$\text{deviation}_w(x; B) \;=\; \|x_I - x_I^*\|^2.$$

**Falsifiable predictions pinned in code**
(`Tests/research/test_sheaf_laplacian_v31.py`):

  - **H6 (boundary preservation).** `harmonic_extension` returns
    only the interior; reconstructing the full cochain from
    $(x_B, x_I^*)$ preserves $x_B$ byte-identically on $B$.
  - **H7 (minimisation — the defining property).** No perturbation
    of the interior cochain off the harmonic extension gives a
    smaller $V$.
  - **H8 (uniqueness when $L_{II}$ has full rank).** Two calls
    with the same boundary cochain yield byte-identical interior.
  - **H9 (degenerate full-boundary).** When every vertex is on
    the boundary, the interior is empty — function returns a
    `(0, d)` array, not a crash.
  - **H10 / H11 (defensive boundary).** Invalid indices raise
    `ValueError`; wrong $x_B$ shape raises `ValueError`.
  - **H12 (utility — the headline claim).** Tampering an interior
    vertex (boundary held fixed) increases the deviation. This is
    the hallucination-detection use case.
  - **H13 ($v_\text{ext} \le v_\text{actual}$).** By the
    minimisation property, the Laplacian quadratic form at the
    harmonic-extended cochain is no larger than at the actual
    cochain.
  - **H14 (chain-topology weight invariance).** With a *single*
    bridge edge connecting boundary to interior (chain topology),
    the harmonic extension is weight-invariant even on a trained
    sheaf. Analytic reason: $x_I = -r \cdot M(r)^{-1} (B x_B)$
    with $r = w_\text{bridge}/w_\text{interior}$ cancels for
    rank-1 $B$. The weight effect IS visible with multiple bridge
    edges.
  - **H15 (`boundary_from_weights`).** A vertex is on the
    boundary iff every incident edge has weight $\geq$ threshold.

H12's pin uses a single-bridge-edge synthetic graph and passes.
The headline finding of this preprint (§4.5–§4.6) is that on a
naturalistic corpus with random 50/50 trust partitioning, H12's
synthetic pin does not transfer — boundary deviation as a
standalone signal **fails structurally**. This is what the
substrate's truth-first discipline calls a STRUCTURAL FAIL: a
real falsification of the standalone-deviation utility claim,
named honestly rather than relegated to a "future work" footnote.

### 3.8 v3.2 — combined detector closing F3 STRUCTURAL FAIL

v3.2
(`sum_engine_internal/research/sheaf_laplacian_v32.py`,
`Tests/research/test_sheaf_laplacian_v32.py`) is a **strict
generalisation of v3** that adds the harmonic-extension deviation
as a complementary signal:

$$v_\text{combined}^{v3.2}(x; w, \gamma, \lambda) \;=\;
v_\text{laplacian}^w(x; w) \;+\; \gamma \cdot \text{deviation}_w(x; B(w)) \;+\;
\lambda \cdot v_\text{deficit}(x).$$

The two cochain-side terms catch complementary things:

  - $v_\text{laplacian}^w$ (from v3) sums residuals over *every*
    edge — informative anywhere on the graph, including under
    boundary-only perturbations regardless of $L_{IB}$ topology.
  - $\text{deviation}_w$ (from v3.1) is informative *only* when
    $L_{IB} \neq 0$. Under the F3 failure topology
    ($L_{IB} = 0$, edges live entirely within $B$ or entirely
    within $I$), deviation is structurally invariant to boundary
    perturbations by linear algebra.

When $\gamma = 0$, v3.2 reduces to v3 numerically (subsumption —
the H16 contract). When $\gamma > 0$, deviation contributes
additively where it has signal; falls back to a constant where
it's blind. The combined score is informative either way —
that's the F3 fall-back guarantee (H18).

**Falsifiable predictions pinned in code**
(`Tests/research/test_sheaf_laplacian_v32.py` and
`test_sheaf_laplacian_v32_property.py`):

  - **H16 (subsumption).** $\gamma = 0$ → v3.2 numerically equals
    v3.
  - **H17 ($L_{IB} \neq 0$ visibility).** On a graph with
    cross-partition edges, $\text{deviation}_w$ changes under
    boundary perturbation.
  - **H18 (F3 fall-back).** On a graph with $L_{IB} = 0$,
    $v_\text{laplacian}^w$ still surfaces the perturbation; the
    combined score is informative even when deviation is blind.
  - **H19 (no $\lambda$ double-counting).** The deficit term
    appears once and only once in the v3.2 wrapper layer.
  - **H20 (degenerate-boundary fall-back).** Empty $B$ or full
    $B$ → $\text{deviation}_w = 0$; combined score reduces to v3.

H16, H17, H18, and H20 carry universal-quantifier upgrades via
Hypothesis property tests (`test_sheaf_laplacian_v32_property.py`):
the contracts hold for arbitrary inputs in the tested domain, not
just the seed examples.

## 4. Empirical results

All numbers in this section are captured in versioned
`fixtures/bench_receipts/*.json` files; each receipt carries the
configuration, the per-cell raw values, and (for v3.1 and v3.2)
a `bench_digest` SHA-256 over the JCS-canonical quantized
payload. §4.7 documents the digest as a reproducibility primitive.

### 4.1 Synthetic micro-benchmark (connected graphs)

Six hand-built fact-sets × five perturbation classes = 30 trials.
v1 catches 18/30 entity-presence-affecting perturbations with
100 % top-1 localisation on caught classes. Predicate-flip and
off-graph fabrication are 0/6 each — by design.

### 4.2 Disconnected-graph falsification

A 4-fact disconnected source graph (4 unrelated facts about 4
entity-pairs) from human-authored paraphrase data exposes v1's
structural blindspot. v2.1 with presence-style cochains *also*
misses, surfacing the deeper category mismatch. Both
falsifications are pinned in code.

### 4.3 ROC bench v2.2 on `seed_long_paragraphs` (baseline)

We sieve-extracted triples from each of 16 multi-fact paragraphs
(`seed_long_paragraphs` corpus, 120 source triples, 229-entity /
95-relation transductive vocabulary), trained one v2.1 sheaf on
the union, generated four perturbation classes per doc (A1
entity-swap, A2 predicate-flip, A3 off-graph fabrication, A4
triple-drop), scored, and computed per-class ROC AUC with v2.2's
combined detector at $\lambda_\text{auto} \approx 1.92$.

| Class | AUC | Detection signal |
|---|---|---|
| A1 entity-swap | **1.000** | max in-vocab $V$ from per-rendered-triple scoring |
| A2 predicate-flip | **1.000** | max in-vocab $V$ from per-rendered-triple scoring |
| A3 off-graph fabrication | **1.000** | $n_\text{oov}$ from per-rendered-triple scoring |
| A4 triple-drop | **0.801** | combined-detector $V$ (Laplacian + auto-$\lambda$ deficit) |
| **Overall (mean)** | **0.948** | |

Receipt:
`fixtures/bench_receipts/sheaf_v2_roc_seed_long_paragraphs_2026-05-01.json`.

The A1 / A2 / A3 detection routes through the per-rendered-triple
scoring of §3.5 (out-of-vocab + relation-residual signals), which
is unrelated to the cochain-on-source-graph machinery the v3.x
arc focuses on. The A4 channel (triple-drop, scored by the
cochain-on-source-graph $V$) is the one §4.4–§4.7 below
investigate.

### 4.4 ROC bench v3 on `seed_long_paragraphs` — F1 MARGINAL, F2 PASS

The v3 bench at `scripts/research/sheaf_v3_roc_bench.py` runs all
three detectors (v2.2 baseline, v3 receipt-weighted, v3.1
boundary deviation) over the 16-document `seed_long_paragraphs`
corpus under deterministic 50/50 trust partitioning per doc.
Receipt: `fixtures/bench_receipts/v3_roc_bench_2026-05-03.json`.

**Headline AUC numbers (mean across runs; ±0.02 LAPACK jitter):**

| Class × target | v2.2 | v3 | v3.1 |
|---|---|---|---|
| A1 entity-swap @ trusted   | 0.62 | 0.63 | 0.50 |
| A1 entity-swap @ untrusted | 0.60 | 0.69 | 0.37 |
| A2 predicate-flip @ trusted    | 0.50 | 0.50 | 0.50 |
| A2 predicate-flip @ untrusted  | 0.50 | 0.50 | 0.50 |
| A4 triple-drop @ trusted   | 0.86 | 0.94 | 0.50 |
| A4 triple-drop @ untrusted | 0.84 | 0.97 | 0.20 |

Two of the five named verdicts resolve from this bench:

  - **F1 MARGINAL.** v3 mean AUC on trusted-target (0.685) vs
    v2.2 mean (0.663): $\Delta = +0.022$. v3 is slightly better
    than v2.2 on trusted-target perturbations, but the margin is
    inside the noise floor on this corpus. The H4 hypothesis
    ("trust amplifies signal") holds dramatically on synthetic
    data H4 (10/10 wins, ~10× ratio) but only marginally at
    corpus scale on `seed_long_paragraphs`.
  - **F2 PASS.** v3 doesn't collapse on untrusted-target — no
    class drops more than 0.10 from v2.2. The 0.1 default weight
    is a viable floor for naturalistic-prose corpora.

The v3 A4 column (0.94 trusted / 0.97 untrusted) is the channel
where receipt-weighting visibly helps. A2 sits at AUC 0.50 across
every detector; this is the known v2.x predicate-flip weakness
(predicate-perturbation negatives are not in the contrastive
training distribution) and addressing it requires changes to the
training loss, orthogonal to the v3.x arc.

### 4.5 v3.1 corpus-scale: F3 STRUCTURAL FAIL

The third row of the v3 bench is the load-bearing negative
result. v3.1 boundary deviation, scored as a *standalone* signal:
trusted-mean AUC 0.499; untrusted-mean AUC 0.343 (anti-correlated
on A4 untrusted). The synthetic H12 pin
(`test_boundary_deviation_detects_interior_tampering`) passed,
yet the corpus bench refutes the standalone-deviation utility
claim.

  - **F3 STRUCTURAL FAIL.** v3.1 standalone trusted-mean AUC
    $0.499 < 0.55$ threshold. Synthetic single-edge utility test
    passed; corpus-scale random-50/50 fails.

We name this STRUCTURAL FAIL rather than "needs tuning" or
"future work" — the synthetic test's pass and the corpus test's
fail together signal that the *single-bridge-edge* topology the
H12 synthetic exercises is not representative of naturalistic
graphs. Burying that finding under "future work" would let the
detector ship into a downstream system on the strength of a
synthetic that does not correspond to the production
distribution. Truth-first labelling makes the load-bearing
finding visible to readers and to the design space.

### 4.6 F3 diagnostic — none of three hypotheses load-bearing

The first response to F3 was to enumerate plausible parametric
fixes and test each. We named three candidate causes ("graph too
small"; "cochain produces zero-vectors"; "random partition too
harsh") and built a 2×2×2 diagnostic at
`scripts/research/sheaf_v3_1_f3_diagnostic.py` that flips each
axis around the PR-#124 baseline cell, producing 8 cells over
(graph_size, cochain_strategy, partition_strategy). Receipt:
`fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-03.json`,
`bench_digest`
`62b6e1878d1d12f36eb80e301304854a1a2c03386f0e872850d3461b2f733e7c`.

**Result:** `load_bearing_hypothesis = "none"`. All 8 cells
FAIL the F3 PASS threshold (trusted-mean AUC ≥ 0.55). Every
single-axis flip of the baseline FAILs; the all-three-axes-flipped
cell FAILs.

The per-cell AUC structure reveals *why*:

  - All 4 cells using `cochain_strategy = trained_embedding`
    produce uniform AUC = 0.500 across every (class, target).
    Reason: the strategy as designed returned the same cochain
    regardless of render — the cochain was a pure function of
    sheaf vertices. This was a confused axis. It does serve as a
    sanity-check negative control: a render-invariant cochain
    MUST give AUC 0.500.
  - All 4 cells using `cochain_strategy = one_hot_default`
    produce AUC = 0.500 on every *trusted-target* perturbation
    across A1/A2/A4. This is structural: when a perturbation
    targets a trusted edge, the perturbation's vertices lie on
    the boundary, the cochain change is at boundary positions,
    the harmonic extension formula
    $x_I^* = -L_{II}^{-1} L_{IB} x_B$ recomputes the interior
    from the new boundary, but the actual interior is
    unchanged — so the deviation
    $\|x_I^\text{actual} - x_I^*\|^2$ ties between clean and
    perturbed by mathematical necessity.
  - The same 4 cells produce AUC ≈ 0.36–0.43 on untrusted-target
    (anti-correlated). Reason: the one-hot cochain zeros out the
    swapped-out entity's vertex, lowering the interior cochain's
    energy; the harmonic extension also lowers; the deviation can
    *shrink* rather than grow.

**Truth-first conclusion: v3.1 boundary deviation has a
structural blind spot for perturbations on the trust frame's
boundary.** This is not a parameter-sweep problem. The detector,
as currently formulated, cannot distinguish clean from perturbed
when the perturbation's vertices are in $B$ and $L_{IB} = 0$.
The diagnostic settled the question — F3 is structural, not
parametric — and pointed v3.2 at *redesigning the detector*
rather than tuning v3.1.

### 4.7 v3.2 validation — F4 PASS, F5 PASS at $\gamma \leq 0.1$, auto-cal wrong

The v3.2 validation bench at
`scripts/research/sheaf_v3_2_validation.py` runs the combined
detector at four $\gamma$ values $\{0.0, 0.1, 1.0,
\gamma_\text{auto}\}$ over the same 16-doc corpus, measuring
trusted-mean AUC and the gap to v3 baseline. Receipt:
`fixtures/bench_receipts/v3_2_validation_2026-05-03.json`,
`bench_digest`
`b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a`.

| $\gamma$ | trusted-mean AUC | F4 ($\geq 0.55$) | $\Delta$ vs v3 | F5 ($\Delta \geq -0.02$) |
|---|---|---|---|---|
| 0.0 | 0.663 | PASS | 0.000 | PASS |
| 0.1 | 0.659 | PASS | $-0.004$ | PASS |
| 1.0 | 0.635 | PASS | $-0.028$ | FAIL |
| auto ($\approx 1.02$) | 0.635 | PASS | $-0.028$ | FAIL |
| **v3** (ref) | **0.663** | — | — | — |

Three honest readings:

  1. **F3 STRUCTURAL FAIL is closed at the detector layer.** v3.2
     at every $\gamma \geq 0$ produces trusted-mean AUC $\geq
     0.55$ (vs PR #124's v3.1 standalone deviation: $0.499$).
     The "blind spot" was scoring against deviation alone;
     pairing with $v_\text{laplacian}^w$ restores robust signal
     regardless of $L_{IB}$ topology. **F4 PASS.**
  2. **Calibration finding (truth-first).** The
     magnitude-matching auto-calibration heuristic
     ($\gamma_\text{auto} \approx 1.02$) is **empirically wrong
     on this corpus** — F5 fails at $\gamma \in \{1.0,
     \text{auto}\}$. Optimal $\gamma$ is small ($\leq 0.1$).
     Deviation's signal-to-noise ratio is below what its
     magnitude suggests; on the `seed_long_paragraphs`
     distribution it functions as a small modulator, not a
     co-leader. **F5 PASS at $\gamma \leq 0.1$, FAIL above.**
     We name this finding rather than retiring the auto-cal
     heuristic silently — the magnitude-matching intuition is
     itself the kind of plausible-but-wrong move the substrate
     should leave a record of, so future readers don't reinvent
     it.
  3. **H16 verified at corpus scale.** $\gamma = 0$ produces
     trusted-mean AUC $= 0.663$, byte-identical to v3's. v3.2 is
     genuinely a strict generalisation, not a different detector
     wearing a similar mask.

### 4.8 Reproducibility: `bench_digest` as a primitive

The two `bench_digest` values cited above
(`b4d26c01…ed5a` for v3.2 validation;
`62b6e187…733e7c` for the F3 diagnostic) are JCS-canonical
SHA-256 hashes (RFC 8785 canonicalisation) over the *quantized*
payload of each bench's receipt — AUCs to 3 decimals; diagnostic
floats to 4. Quantisation absorbs the ~$\pm 0.02$ LAPACK jitter
that `numpy.linalg.lstsq` introduces across runs; the digest is
byte-stable across fresh Python invocations.

The reproducibility property is **unconditional**: no
`PYTHONHASHSEED=0` or other environment-variable manipulation is
required. The earlier conditional form (each receipt previously
carried a `reproducibility_requires: "PYTHONHASHSEED=0"` field)
was the consequence of one load-bearing site —
`DeterministicSieve.extract_triplets` returning
`list(set(triplets))`, whose iteration order is hash-randomised
across Python invocations — fixed by `sorted(set(triplets))` in
the same arc as this draft. Verification protocol:
`python3 -m scripts.research.sheaf_v3_2_validation 2>/dev/null |
grep '"bench_digest"'` run three times in fresh processes yields
the identical hash.

The digest is built on the same canonicalisation primitive
(JCS / RFC 8785) the project's render receipts use, and is
therefore signable with the project's existing JWKS keys. An
arXiv preprint citing the digest gives external readers a
byte-level fixed point: rerun the bench, recompute the digest,
match against the published value, and any divergence is either
upstream code change or environment drift — both load-bearing
findings in their own right.

This makes the `bench_digest` field a small but novel substrate
for what we call *reproducible-research-with-cryptographic-teeth*:
the published numeric claim is not just textual prose in a paper
but a hash that an external reader can match offline. We are not
aware of comparable substrate in the LLM-eval literature; the
closest neighbours are the DOI-anchored evaluation bundles in
classical ML benchmarks, which carry source data hashes but not
quantised-result digests.

## 5. Substrate context — six-regime audit-grade record-keeping

The detector scores the consistency of a render against a source
bundle. The substrate that produces the rendering pipeline whose
outputs the detector scores is a separate, audit-grade
record-keeping layer. Six per-regime validators ship in the same
codebase as the detector, each consuming the same JSONL audit-log
schema (`sum.audit_log.v1`) and emitting the same regime-agnostic
report shape (`sum.compliance_report.v1`):

| Regime | Statute | Per-row rules | Tests |
|---|---|---|---|
| EU AI Act Article 12 | Reg (EU) 2024/1689 Art 12 | 6 | 32 |
| GDPR Article 30 | Reg (EU) 2016/679 Art 30 | 5 | 25 |
| HIPAA § 164.312(b) | 45 CFR § 164.312(b) | 6 | 27 |
| ISO/IEC 27001:2022 A.8.15 | ISO/IEC 27001:2022 A.8.15 | 5 | 19 |
| SOC 2 CC7.2 | AICPA TSP §100A CC7.2 | 5 | 19 |
| PCI DSS v4.0 Req 10 | PCI DSS v4.0 Req 10 | 7 | 25 |

Each regime's wire-spec doc carries an explicit "what this
validator does NOT pin" section naming the operational
obligations the validator can't reach (organisational policy,
retention duration, log-file protection, review processes). That
honesty is itself the load-bearing property: a green
`ValidationReport` says the per-row form floor is satisfied, not
that the deployment is operationally compliant.

The PCI DSS validator's R7 rule
(`pci-dss-4-req-10.user-identification`) closed the previously
load-bearing user-identification gap by adding three additive
optional fields (`user_id` / `host_id` / `ip_address`) to
`sum.audit_log.v1` (backward-compatible under the schema's
"consumers should ignore unknown keys" convention) plus three env
vars (`SUM_AUDIT_USER_ID` etc.) that operators source from their
authenticating proxy's session identity. The closure means a
PCI-relevant deployment can pass the per-row floor for the most
complex statute in the slate without an out-of-band
schema-extension exercise.

Why this matters to the preprint's claim. The detector's $w_e$
weights derive from cryptographic render receipts; the audit log
carries every render's receipt KID; the compliance validators
attest that the audit log meets a per-row record-keeping floor
across six statutorily-distinct regimes. The chain of evidence
from "an external auditor's record-keeping requirement" to "the
edge weight in $L_\mathcal{F}^w$" runs through artifacts whose
shapes and signatures are byte-stable across the trust triangle.
The detector is a research artifact; the substrate around it is
production-shaped by the demands of regulated audit.

## 6. Position vs. existing work

**vs. token-level uncertainty (sequence log-probability,
semantic entropy):** complementary. Uncertainty estimates measure
the model's own confidence; we measure cross-claim consistency
under a verifiable-source binding. The two should compose.

**vs. retrieval-grounded verification (RAG with citations):**
complementary. RAG confirms support for retrievable claims; we
score consistency of any rendered claim against an attested
source bundle, including claims whose support is in the sheaf
but not in a retrieval index.

**vs. existing knowledge-graph hallucination detectors
(HalluGraph, KG-grounded checks):** structurally adjacent, with
three distinguishing features. First, the *math* is grounded in
Hansen-Ghrist sheaf-Laplacian theory (and its weighted /
harmonic-extension generalisations) and Gebhart contrastive
sheaf-embedding training, not ad-hoc consistency rules. Second,
the *provenance* is cryptographic — the source bundle is a
signed artifact whose verifier is realiser-independent
(Python / Node / browser byte-identity locked in CI). Third, the
*reproducibility* is hash-anchored — every published numeric
claim is a `bench_digest` an external reader can match offline.
The detector's verdict, the source bundle's provenance, and the
benchmark's reproducibility are each a byte-level fixed point in
an audit trail.

**vs. Tull, Kleiner & Smithe 2023 categorical active inference:**
the compositionality theorem (Theorems 45/46) says free energy
is additive across sequential and parallel composition of
generative models. Our v3.2 combined detector is mathematically a
sum of orthogonal terms — the same shape — over per-doc subgraphs
of a multi-document corpus. Federated multi-agent scoring is the
natural application; v3.1's harmonic extension over a trusted-
issuer boundary is the SUM-specific instantiation, with v3.2
serving as the F3-aware fall-back that recovers when boundary
deviation is structurally blind.

## 7. Bounded claims

What the detector claims, at the v3.2 layer evaluated on
`seed_long_paragraphs`:

- **Specific.** Per-class detection signals defined above
  separate clean from perturbed at the AUCs reported in §4.4 /
  §4.7 on the synthetic micro-bench and the
  `seed_long_paragraphs` corpus.
- **Localised.** Per-edge / per-rendered-triple discrepancy
  identifies the perturbed claim at high precision (100 % top-1
  on the synthetic micro-benchmark; the per-rendered-triple $V$
  signal on A1 / A2 / A3 reports the offending triple directly).
- **Cryptographically auditable.** Because the source binding is
  a signed receipt, the detector's verdict can be reproduced
  offline by any verifier with the source bundle and the trained
  sheaf parameters — no vendor-locked confidence score.
- **Reproducibility-anchored.** Each numeric result is captured
  in a versioned receipt with a `bench_digest` field; rerunning
  the bench locally and comparing the digest is a byte-level
  match operation.

What the detector does *not* claim:

- **Not** a solution to hallucination. One signal among many; in
  practice would compose with confidence calibration,
  retrieval-grounded checks, NLI verifiers, and human review.
- **Not** a correctness proof. It detects inconsistency in the
  *output manifold* under cross-edge agreement and presence
  patterns; it does not certify that any single output is
  *correct*. A consistent manifold of uniformly-wrong renderings
  (where the trained sheaf has no internal contradiction with
  the wrong claims) remains undetected.
- **Not** generalising across all knowledge-graph schemas without
  re-calibration. The $\lambda$ auto-calibration is per-corpus;
  schema-typed sheaves (multiple entity types, typed restriction
  maps) require additional design work.
- **Not** computable on arbitrary corpus sizes. Sparse-block
  storage of the $d$-dim Laplacian is straightforward but not
  yet implemented; scaling to $> 10^4$ vertices needs the
  sparsification machinery of Hansen-Ghrist 2019 §6 (Theorem
  6.4).
- **Not** giving the boundary-deviation signal of v3.1 a
  standalone use. F3 STRUCTURAL FAIL is true at the v3.1 layer
  and remains true; v3.2 closes the *detector-layer* problem by
  combining with $v_\text{laplacian}^w$ but does not make
  standalone deviation informative.
- **Not** confirming the magnitude-matching auto-calibration of
  $\gamma$. F5 at $\gamma_\text{auto}$ FAILs on this corpus;
  optimal $\gamma$ is small. Future deployments must measure
  $\gamma$ rather than auto-derive it from term magnitudes.
- **Not** detecting A2 predicate-flip at corpus scale. AUC $=
  0.50$ across every detector (v2.2 / v3 / v3.1 / v3.2) on
  `seed_long_paragraphs`. Closing this requires
  predicate-perturbation negative sampling at training time,
  orthogonal to the v3.x detector arc.
- **Not** generalising across machines. The `bench_digest`
  reproducibility property holds on the same machine + same code
  (and across fresh Python invocations on that machine);
  cross-machine reproducibility (different LAPACK builds,
  different numpy versions) is unmeasured.

Five named falsification verdicts (F1 MARGINAL, F2 PASS,
F3 STRUCTURAL FAIL, F4 PASS, F5 PASS at $\gamma \leq 0.1$)
together with the synthetic-corpus disconnect (H12 synthetic-pass
vs F3 corpus-fail) testify to which signals the detector does
and does not carry. Three pinned-in-code falsifications from
earlier in the arc — (a) v1 disconnected-graph density-dropout
blindness; (b) v2.1 presence-cochain inheritance of the same
blindness; (c) v2.2 default-$\lambda$ at corpus scale — remain
as regression tests at every PR.

## 8. Future work

- **Path 2 — real LLM-rendered adversarial bench.** Synthetic
  perturbations are existence-proofs; adversarial LLM renderings
  stress the detector differently. Generate clean and adversarial
  variants via the hosted Worker render path and re-run §4.4 /
  §4.7.
- **A2 closure — predicate-perturbation training.** Add
  predicate-flip negatives to the contrastive sampler. This is
  orthogonal to the v3.x arc (a change to §2.5 / §3.3) but is
  the single biggest open class-AUC gap in §4.4.
- **Cross-machine reproducibility — Node port.** A Node
  reimplementation of the v3 / v3.2 detectors that reproduces
  the AUCs and matches the `bench_digest` would extend the
  cross-runtime trust triangle from the verifier layer (which
  already holds K1–K4) to the research bench layer. The digests
  in §4.6 / §4.7 are the published fixed points such a port
  would have to match.
- **Per-doc graph-structure-aware $\gamma$.** When $L_{IB}$ has
  high mass, raise $\gamma$; when near zero, set $\gamma = 0$.
  The combined score then uses deviation only where it is
  structurally informative. This addresses the §4.7 finding that
  $\gamma_\text{auto}$ over-weights deviation on average.
- **Cochain redesign that propagates render content into the
  interior.** v3.2 works around the F3 blind spot by combining
  with $v_\text{laplacian}^w$; a cochain redesign would address
  the root cause directly (the cochain at present is
  translation-invariant under boundary-only perturbations). Both
  paths remain open.
- **Compositionality at scale.** Tull-Kleiner-Smithe Thm. 45/46
  give per-component additivity of free energy under sequential
  and parallel composition. Our combined detector
  $V = \|\delta x\|^2 + \gamma \cdot \text{deviation}_w +
  \lambda \cdot d^2$ has the same shape; for a multi-doc corpus,
  per-doc scores compose to a corpus-level score with bounded
  total. The free-energy compositionality theorem is the rigorous
  statement of "minimising local consistency achieves global
  consistency."
- **Schema-typed sheaves.** Multi-type entity schemas (people /
  films / events) with typed restriction maps; Spivak / Kent
  ologs as the schema language.
- **Multi-source connectors.** Each external source of authority
  (Wikidata, DOI registry, ORCID, regulatory text) provides a
  partial sheaf; the combined sheaf over a federation of sources
  is the Grothendieck-topology gluing. Compliance-regime tags
  (the six in §5) become per-regime predicates with required-
  field validators.

## References

- **Curry, J.** (2014). *Sheaves, Cosheaves, and Applications.*
  PhD thesis, University of Pennsylvania.
- **Gebhart, T., Hansen, J., & Schrater, P.** (2023). *Knowledge
  Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph
  Embedding.* AISTATS 2023, PMLR 206. arXiv:2110.03789.
- **Hansen, J. & Ghrist, R.** (2019). *Toward a spectral theory
  of cellular sheaves.* Journal of Applied and Computational
  Topology 3(4):315–358. arXiv:1808.01513.
- **Tull, S., Kleiner, J., & Smithe, T. St C.** (2023). *Active
  Inference in String Diagrams: A Categorical Account of
  Predictive Processing and Free Energy.* arXiv:2308.00861.
- **Bordes, A., Weston, J., Collobert, R., & Bengio, Y.** (2011).
  *Learning Structured Embeddings of Knowledge Bases.* AAAI.
- **Spivak, D. I. & Kent, R. E.** (2012). *Ologs: A Categorical
  Framework for Knowledge Representation.* PLOS ONE 7(1):e24274.
- **Coecke, B., Sadrzadeh, M., & Clark, S.** (2010).
  *Mathematical Foundations for a Compositional Distributional
  Model of Meaning.* Linguistic Analysis 36.
- **C2PA** (2023+). *Content Authenticity Coalition* technical
  specifications, including `digital_source_type` ontology.
- **OtotaO/SUM repository.** `sum-engine` v0.5.0 on PyPI;
  source at https://github.com/OtotaO/SUM. Spec doc at
  `docs/SHEAF_HALLUCINATION_DETECTOR.md`. Library API surface at
  `docs/SHEAF_LIBRARY_API.md`. Bench scripts at
  `scripts/research/sheaf_v3_roc_bench.py`,
  `scripts/research/sheaf_v3_1_f3_diagnostic.py`,
  `scripts/research/sheaf_v3_2_validation.py`. Receipts at
  `fixtures/bench_receipts/v3_roc_bench_2026-05-03.json`,
  `fixtures/bench_receipts/v3_1_f3_diagnostic_2026-05-03.json`,
  `fixtures/bench_receipts/v3_2_validation_2026-05-03.json`.

---

*Acknowledgements.* This work would not exist without the
foundational categorical-AI program of Spivak, Kent, Coecke,
Curry, Hansen, Ghrist, Gebhart, Schrater, Tull, Kleiner, Smithe,
and the broader Topos Institute / Quantinuum / ACT community.
SUM contributes the cryptographic substrate, the
reproducibility-with-digests engineering, the corpus-scale
evaluation, and the F3 STRUCTURAL FAIL / v3.2 closure arc; the
underlying mathematics is theirs.

*Reproducibility.* All code Apache-2.0, all benchmarks
reproducible by `pip install 'sum-engine[research]'` and the
bench scripts in the repository. Receipt JSON schemas are stable
and versioned (`sum.render_receipt.v1`,
`sum.sheaf_v2_roc_bench.v1`,
`sum.sheaf_v3_roc_bench.v1`,
`sum.sheaf_v3_1_f3_diagnostic.v1`,
`sum.sheaf_v3_2_validation.v1`). Each of the v3.1 / v3.2 receipts
carries a `bench_digest` field — JCS-canonical SHA-256 over the
quantised payload — that an external reader can match
byte-for-byte after re-running the bench.

*Status of claims.* §4.1 (synthetic micro-bench), §4.3 (v2.2 ROC
bench), §4.4 (v3 ROC bench), §4.6 (F3 diagnostic) and §4.7 (v3.2
validation) are measured, pinned, and reproducible at the
commit hash of this draft. §4.2's falsifications are pinned in
code by named regression tests. §5's compliance evidence is
mechanically verified by the per-regime test suites and the
cross-regime CLI dispatch test. §6 positioning claims are this
author's reading of the cited literature; corrections welcomed
before arXiv submission.

*Authors and contact.* Draft authored 2026-05-01 (v0) and
revised 2026-05-04 (v0.1) by the SUM project. For corrections /
contributions: https://github.com/OtotaO/SUM/issues. Pre-arXiv
comments welcome; pre-circulation review (1–2 readers) sought
before submission to `cs.LG` (primary) / `cs.CR` (secondary).
