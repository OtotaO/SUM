# A cryptographically-anchored substrate for hallucination detection on signed render bundles

**Status: draft v0.1 (2026-05-04)** — pre-arXiv working note.
Targeted submission: `cs.CR` (substrate / threat model /
cryptographically-anchored reproducibility) primary, `cs.LG`
(sheaf-Laplacian detection methods, honest negative results,
complementary-signal Borda composition) secondary. Pre-circulation
to 1–2 readers before submission.

---

## Abstract

We describe a substrate for cryptographically-anchored,
byte-reproducible hallucination detection on signed render
bundles, and a complementary-signal hybrid detector built on it
that strictly beats trivial entity-set baselines on a 16-document
corpus's full perturbation space — including
entity-set-preserving predicate flips that no entity-set baseline
can detect.

The substrate composes four primitives, each verifiable
independently: (i) cross-runtime-verified render receipts
(`sum.render_receipt.v1`, Ed25519 over JCS-canonical bytes;
Python / Node / browser byte-identical; locked by a K-matrix
gate on every release); (ii) `bench_digest` — a JCS-canonical
SHA-256 over each bench's quantized payload, byte-stable across
fresh Python invocations *and* across two distinct LAPACK
environments (Apple Accelerate on Apple Silicon and OpenBLAS via
the numpy PyPI wheel on Modal x86_64; cross-machine measurement
in §4.8); (iii) six per-regime audit-grade compliance validators
(EU AI Act Art 12 / GDPR Art 30 / HIPAA § 164.312(b) / ISO 27001
A.8.15 / SOC 2 CC7.2 / PCI DSS v4.0 Req 10) consuming a
regime-agnostic `sum.audit_log.v1` schema; (iv) an explicit
threat model (§3.0) naming exactly the attacker capabilities
each substrate component defends.

The detector arc is the substrate's first published application.
Built on top of the verified render bundles, it composes three
orthogonal signals: (a) the weighted sheaf-Laplacian quadratic
form $x^T L_\mathcal{F}^w x$ over a contrastively-trained
knowledge sheaf (Hansen-Ghrist 2019 §3.2; Gebhart-Hansen-Schrater
2023), with edge weights driven by per-edge receipt-trust
status; (b) the per-rendered-triple $V$ channel
(Gebhart 2023 §4) scored against the trained restriction maps;
(c) entity-set Jaccard distance between source and rendered
triples. The three signals are *structurally complementary*:
(a) and (b) catch entity-set-preserving predicate perturbations
that no entity-set baseline detects (cochain blindness to
predicate flip is mathematical, not parametric); (c) catches
entity-set-changing perturbations trivially at AUC $1.000$.
Borda rank-fusion across the channels yields trusted-mean
AUC $0.876$ on `seed_long_paragraphs` (16 docs, 120 source
triples), strictly above the strongest single-signal baseline
($\Delta = +0.043$ vs B2 jaccard alone, above the $+0.030$
"real win" threshold).

The substrate's truth-first discipline produced the hybrid the
hard way. The first published v3.x detector (sheaf-Laplacian
cochain channel only) lost to trivial entity-set baselines by
$-0.174$ trusted-mean. Naming that loss a STOP-THE-LINE finding,
not "future work," led to a structural diagnosis (the cochain is
mathematically blind to entity-set-preserving perturbations
because predicate doesn't enter the cochain), then to the
per-rendered-triple channel restoration (which lifted A2 from
$0.500$ to $0.671$), then to the Borda fusion that closed the
gap. Five named falsification verdicts (F1 MARGINAL, F2 PASS,
F3 STRUCTURAL FAIL, F4 PASS, F5 PASS at $\gamma \leq 0.1$) and
four cryptographically-anchored recovery-experiment digests —
`a7965803…` (Borda(v3.2_only, B2) loses), `aa34b6e8…`
(predicate-perturbation training fails to lift A2), `7025436f…`
(per-triple integration lifts A2), `dc6e0260…` (complementary
Borda WINS) — pin every step in mechanically-verifiable
artifacts.

We position the work inside the program of Hansen-Ghrist 2019
sheaf-Laplacian spectral theory, Gebhart 2023 contrastive
sheaf-embedding training, and Tull-Kleiner-Smithe 2023
categorical active inference; the substrate sits in the
adjacent space of cryptographically-anchored ML evaluation
(zkML provers; verifiable-computation pipelines) at a different
threat-model layer (§6). The contribution is the substrate plus
the methodology that produced the hybrid; the detector is the
worked example.

---

## 1. Introduction

The reliability layer of generative AI has two compounding
problems. The first is hallucination — LLMs producing
plausible-but-wrong content. The second is the *evaluation
crisis* around it: published hallucination-detection benchmarks
typically lack reproducible numerics (the model is pinned, the
prompts are pinned, but the result is a number in a paper),
lack threat models (an attacker controlling which inputs?),
and lack provenance (whose source bundle, signed by whom?).
A reader who cares about deploying a detector in a regulated
context cannot, today, take a published AUC and say "I have
audit-grade evidence this number reproduces in my environment
under a specific attacker model." The substrate this paper
describes addresses that gap.

The substrate composes four primitives, each verifiable without
trusting the others: cross-runtime signed render receipts (§3.1);
a `bench_digest` reproducibility primitive (§4.8) measured to
hold across two distinct LAPACK environments; six per-regime
audit-grade compliance validators consuming a regime-agnostic
audit-log schema (§5); and an explicit four-capability threat
model (§3.0). On top of this substrate we build a
complementary-signal hybrid hallucination detector (§3.9) that
strictly beats trivial entity-set baselines on the corpus's full
synthetic perturbation space — the first published *competitive*
detector on this substrate, but more importantly the first one
whose competitive claim can be verified end-to-end against
mechanically-pinned artifacts (§4.7.1 documents the recovery
arc that produced it).

The mathematical machinery in the detector is not new. Knowledge
graphs as free categories on directed multigraphs (Spivak & Kent
2012); sheaves on the resulting graph as cellular sheaves (Curry
2014); the sheaf Laplacian as a spectral consistency measure
(Hansen & Ghrist 2019); the weighted generalization (§3.2 of the
same paper); the harmonic extension over a
(boundary, interior) partition (Prop. 4.1 / Thm. 4.5);
contrastive sheaf-embedding training (Gebhart, Hansen & Schrater
2023). What we add at the *detector* layer is the
complementary-signal Borda fusion (§3.9) plus the recovery
methodology (§4.7.1) that distinguishes the hybrid from any
single component. What we add at the *substrate* layer is the
composition of result-anchored bench digests with JWKS-signed
render receipts — making aggregate ML-evaluation claims
verifiable end-to-end, not just at the input boundary.

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
`Tests/research/test_sheaf_laplacian_v3.py::test_harmonic_extension_minimizes_v_subject_to_boundary_constraint`;
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

Concretely: SUM (`sum-engine`, Apache-2.0; PyPI 0.6.0) provides
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
(`Tests/research/test_sheaf_laplacian_v3.py`):

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

**The cochain channel alone is insufficient against trivial
baselines** (§4.7.1 shows this empirically). The full v3.x
deployment composes the v3.2 cochain channel above with the
§3.5 per-rendered-triple channel and an entity-set baseline
(§3.9 below); the published WIN comes from this composition,
not from the cochain channel evaluated in isolation.

### 3.9 Complementary-signal hybrid

The §3.6–§3.8 sheaf-Laplacian cochain channel and the §3.5
per-rendered-triple channel together form one detector class;
entity-set Jaccard distance forms another. The two classes are
*structurally complementary*:

  - **Sheaf-Laplacian detectors** (cochain channel, §3.6–§3.8;
    per-triple channel, §3.5) are sensitive to per-edge
    restriction-map residuals. The per-triple channel detects
    predicate violations directly: a rendered triple
    $(h, r', t)$ scores high $V_\text{triple}$ under the trained
    $F_h(r'), F_t(r')$ when $r'$ is not the correct relation
    for $(h, \cdot, t)$. The cochain channel, by contrast, is
    *mathematically blind* to entity-set-preserving
    perturbations because its cochain construction encodes only
    entity *presence*, not the predicates connecting present
    entities. §4.7.1 documents the discovery of this blindspot
    via a corpus-scale comparison and the recovery arc that
    addressed it.
  - **Entity-set baselines** (§4.7.1: B1 entity-presence
    deficit, B2 jaccard distance) are sensitive to set
    differences between source entities and rendered entities.
    They detect entity-swap (A1) and triple-drop (A4) perfectly
    when those perturbations change the entity set
    ($\text{AUC} = 1.000$), but are completely blind
    ($\text{AUC} = 0.500$) to predicate flips (A2) that
    preserve the entity set.

Define the hybrid via Borda rank-fusion at the per-(class,
target) cell level. Given two detectors' scores on the same $n$
pairs, the fused score for pair $i$ is

$$s^\text{borda}_i = \mathrm{rank}\!\left(s^{(d_1)}_i \mid s^{(d_1)}_*\right)
\;+\; \mathrm{rank}\!\left(s^{(d_2)}_i \mid s^{(d_2)}_*\right)$$

where $\mathrm{rank}(\cdot \mid s_*)$ is the average rank within
the pool (ties at mean rank). Borda is parameter-free and
magnitude-invariant; it preserves any detector's perfect ranking
of a pool ($\text{AUC} = 1.000 \Rightarrow$ the perturbed pair
always outranks the clean pair) while letting the other
detector contribute signal where the first is at chance.

The complementary hybrid we publish is

$$\text{Borda}\!\left(s^{\text{v3.2 + per-triple}},\; s^{\text{B2 jaccard}}\right),$$

evaluated per (class, target) cell on `seed_long_paragraphs`
(§4.7.1 has the per-cell numbers). The component
$s^{\text{v3.2 + per-triple}}$ score combines the §3.5
per-triple channel additively with the §3.8 v3.2 cochain
score:

$$s^{\text{v3.2 + per-triple}} \;=\; v_\text{laplacian}^w \;+\;
\gamma \cdot \text{deviation}_w \;+\; \lambda \cdot v_\text{deficit}
\;+\; \alpha \cdot \max_i V^{(i)}_\text{triple} \;+\;
\beta \cdot n_\text{oov}$$

with $\alpha = \beta = 1.0$ chosen at moderate scale;
per-deployment composition tuning (Z-score additive, gated,
weighted-linear alternatives to Borda) is an open follow-up.

**The hybrid is a substrate-enabled claim, not just a detector
choice.** Each of the three component scores is computable
because the substrate provides a verifiable bundle: the source
triples for the per-triple channel come from an Ed25519-signed
render receipt; the cochain channel's per-edge weights come from
the same receipt's trust status; the entity-set baseline operates
on the same render bundle as the other two. Without the verified
bundle, the composition's reproducibility claim collapses — there
is no shared, attested input to compose against. The substrate's
contribution is making this composition trustworthy end-to-end,
not the choice of Borda fusion specifically.

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

### 4.7.1 Recovery arc — how the WIN was found

§4.4–§4.7 chart the v3.x detector arc as it was developed. By
the end of §4.7, v3.2 closes the F3 STRUCTURAL FAIL within the
detector layer — but only against the v3.x family's internal
baseline, not against any external comparison. As part of the
substrate's pre-publication discipline (a baseline-comparison
gate), we then asked: how does v3.2 fare against trivial
reproducible baselines computed from the same render bundles?

The answer was: **it loses, decisively, on entity-set-changing
perturbations.** Two minimum-defensible baselines, both pure
set operations on entity sets — B1 entity-presence-deficit
($1 - \text{recall}_\text{src→render}$) and B2 jaccard distance
($1 - J(\text{src}_\text{ent}, \text{render}_\text{ent})$) —
produce trusted-mean AUC $0.824$ and $0.833$ respectively,
against v3.2's $0.659$. The losing margin is $-0.174$.

Naming this loss a STOP-THE-LINE finding rather than "future
work" led to four engineering recovery experiments, each pinned
in `Tests/research/test_recovery_experiment_digests.py`. Two of
the four pins are byte-digest (anchored to a specific 64-hex
hash that any reader can match); two are behavior-shape pins
(verdict label + observable invariant) — see the per-experiment
notes in the table below for which is which and why:

| Experiment | Outcome (trusted-mean) | Pin | bench_digest (operator-side) |
|---|---|---|---|
| **Borda(v3.2_only, B2)** | LOSES — $0.808 < 0.833$, $\Delta \approx -0.025$. v3.2 anti-correlated with B2 on A1 trusted; rank fusion degrades B2's perfect 1.000 to 0.933. | byte-digest (post-v0.3 deterministic-BLAS fix: `scripts/research/_deterministic_blas.py` sets BLAS thread-count env vars at module-import time, eliminating the Apple Accelerate / OpenBLAS thread-pool-size variance that previously caused two-outcome non-determinism). Verified 10× in fresh procs. | `a7965803…6c2003` |
| **Predicate-perturbation training negatives** | A2 STAYED at $0.500$. Surfaced cochain-channel structural blindness to entity-set-preserving perturbations: predicate doesn't enter the cochain; adding training negatives can't fix what scoring discards. Same shape as F3 STRUCTURAL FAIL. | byte-digest (post-v0.2 Issue 3 refactor: bench refactored to call production `train_restriction_maps` with new `n_predicate_negatives_per_positive` parameter, replacing the local v2-training-loop copy. Verified 5× in fresh procs on operator + 3-environment Modal v6 ALL_MATCH, including Python 3.10 ↔ Python 3.12). | `ddf41484…001f59c3` |
| **Per-rendered-triple V channel integration** | A2 LIFTED $0.500 \to 0.671$ (trusted) and $0.678$ (untrusted); trusted-mean $0.659 \to 0.759$. Restored the §3.5 channel that v2.2 §4.3 used to hit A1/A2/A3 = 1.000. Still $< $ B2 alone, but now informative about *why*: this composition is the only detector that catches A2. | byte-digest (verified 5× in fresh procs; 3-environment Modal v6 MATCH; per-rendered-triple V channel adds magnitude that breaks ties cleanly) | `7025436f…fd4fa` |
| ★ **Complementary Borda(v3.2 + per-triple, B2)** | **WINS** — trusted-mean $0.876 > 0.833$, $\Delta = +0.043$ (above the $+0.030$ "real win" threshold). | byte-digest (verified 5× in fresh procs; 3-environment Modal v6 MATCH; per-triple V magnitude breaks LAPACK ties cleanly) | `dc6e0260…343ce` |

**Per-cell AUC of the headline composition** (compared to its
two components):

| Class × target | v3.2 + per-triple | B2 jaccard | **Borda hybrid** |
|---|---:|---:|---:|
| A1 entity-swap @ trusted    | 0.698 | 1.000 | **0.967** |
| A1 entity-swap @ untrusted  | 0.751 | 1.000 | **0.991** |
| A2 predicate-flip @ trusted | **0.671** | 0.500 | **0.671** |
| A2 predicate-flip @ untrusted | **0.678** | 0.500 | **0.678** |
| A4 triple-drop @ trusted    | 0.907 | 1.000 | **0.991** |
| A4 triple-drop @ untrusted  | 0.969 | 1.000 | **1.000** |
| **Trusted-mean**            | 0.759 | 0.833 | **0.876** |

The hybrid wins where the components are complementary: A2 lift
comes entirely from the v3.2 + per-triple channel (B2 has zero
signal there); A1/A4 dominance comes entirely from B2 (the
sheaf-Laplacian channels are noisy on these). Borda fusion
preserves both contributions.

**The methodology, not the specific detector, is what we claim
generalizes.** A sheaf-Laplacian detector trained on a different
corpus or evaluated against a different perturbation regime
might require different recovery experiments. The substrate's
role is to make every such experiment *reproducible*
(`bench_digest`, including across LAPACK environments per §4.8),
*comparable* (same render-bundle inputs feeding every detector),
and *trustworthy* (cryptographic provenance via the JWKS-signed
receipts). The discipline can run end-to-end without ad-hoc
trust assumptions.

Receipts:
`fixtures/bench_receipts/baseline_comparison_2026-05-04.json`
(B1, B2 alone),
`fixtures/bench_receipts/hybrid_comparison_2026-05-04.json`
(Borda(v3.2_only, B2) — losing),
`fixtures/bench_receipts/predicate_negatives_experiment_2026-05-04.json`
(A2 stays at chance),
`fixtures/bench_receipts/per_triple_integration_2026-05-04.json`
(A2 lifted),
`fixtures/bench_receipts/complementary_hybrid_2026-05-04.json`
(the WIN). Verification:
`scripts/research/sheaf_*.py` modules under research/.

### 4.7.2 Real-LLM-rendered adversarial bench (Path 2)

The §4.7.1 hybrid WIN was measured against the synthetic A1/A2/A4
perturbation harness. §7's bounded claims explicitly named this as
the most-load-bearing open asterisk: "Not generalising to real-LLM-
rendered hallucinations." This section closes that asterisk.

**Setup.** For each doc in `seed_long_paragraphs`, source triples
are rendered via `gpt-4o-mini-2024-07-18` (matching the project's
existing pinned-snapshot convention) at four prompt classes:

  - `neutral` — faithful baseline rendering.
  - `a1_adversarial` — instruct the LLM to subtly substitute one
    subject or object with a plausible-but-wrong entity.
  - `a2_adversarial` — instruct the LLM to subtly change one
    predicate to a different relation of similar type.
  - `a4_adversarial` — instruct the LLM to silently omit one fact.

Each rendered tome is re-extracted via the deterministic sieve;
the re-extracted triples become the detector's input. (Clean,
adversarial) pairs feed the same four detectors as §4.7.1.

The architecture is **capture-once-replay-forever**: the LLM
responses are committed to
`fixtures/bench_renders/path2_seed_long_paragraphs.json` at one
specific snapshot date; downstream scoring is fully deterministic
given that snapshot. Any reader rerunning Phase 2 against the
checked-in snapshot reproduces the digest byte-identically. Phase 1
(LLM capture) requires `OPENAI_API_KEY` and is operator-gated; the
bench only re-runs Phase 1 if explicitly invoked with
`--regenerate-snapshot`.

Receipt:
`fixtures/bench_receipts/path2_v3_bench_2026-05-05.json`,
**`bench_digest 7b364fc6…cc4b75e`**. Pinned in
`Tests/research/test_sheaf_path2_v3.py`.

**Per-cell AUC (real LLM perturbations):**

| Detector | A1 | A2 | A4 | Mean |
|---|---:|---:|---:|---:|
| v3.2 + per-triple | 0.535 | 0.586 | 0.578 | 0.566 |
| **B2 jaccard** | **0.602** | **0.643** | **0.736** | **0.660** |
| Borda hybrid (v3.2 + per-triple, B2) | 0.598 | 0.631 | 0.699 | 0.643 |

**Verdict:** `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM`,
$\Delta = -0.0208$ trusted-mean. The hybrid loses to B2 alone by a
small margin on real LLM perturbations.

**Honest framing.** Three findings:

  1. **The synthetic-bench WIN does NOT generalise.** §4.7.1's
     $\Delta = +0.043$ on synthetic perturbations becomes
     $\Delta = -0.021$ on real LLM perturbations. The hybrid does
     not maintain its competitive edge on real-LLM-rendered prose.

  2. **All detectors substantially weaken on real LLM**: B2's
     trusted-mean drops from $0.833$ (synthetic) to $0.660$ (real
     LLM). The hybrid drops from $0.876$ to $0.643$. The synthetic
     A1 perturbation produced a clean entity-set diff that B2
     caught at $1.000$; the LLM's "subtle entity swap" produces
     a noisier perturbation distribution where B2 is only mildly
     informative ($0.602$).

  3. **A2 is the cell where v3.x narrows the gap.** On synthetic
     A2 (predicate flip with preserved entity set), B2 was at
     chance ($0.500$) and v3.x + per-triple was the only
     informative detector ($0.671$). On real LLM A2, B2 picks up
     some signal ($0.643$) — the LLM's "subtle predicate change"
     prompt seems to perturb the entity set more than the
     synthetic harness did. v3.x + per-triple gets slightly less
     ($0.586$) because the per-triple V channel is calibrated for
     entity-set-preserving perturbations.

**The synthetic-vs-real gap is itself the load-bearing finding.**
Synthetic A1/A2/A4 perturbations have a structural property —
A1/A4 cleanly change the entity set, A2 cleanly preserves it —
that real LLM perturbations don't share. A useful frame from the
biological literature on signal-reward contracts (Schiestl et al.,
*Nature* 399:421, 1999; Cook & Rasplus, *TREE* 18:241, 2003) is the
*deception register* of an adversarial signal. Synthetic A1/A4 are
**structurally legible deceptions**: the entity-set boundary is
cleanly violated and B2's set-distance metric reads them at AUC
$1.000$. Real-LLM A1/A4 are **naturalistically embedded
deceptions**: the LLM smuggles a substitution into a plausible
narrative whose entity set overlaps the source's, and B2's signal
weakens accordingly. The hybrid's $+0.043$ synthetic-bench WIN can
be read in this language as the consequence of a measure (the
synthetic harness) becoming a target — Goodhart's law in its
*regressional* form (Manheim & Garrabrant, arXiv:1803.04585,
2018): the hybrid was selected to compose well with B2 on signals
of the structurally-legible class, and stops being a good measure
once the deception-register changes. The substrate's truth-first
discipline catches this exactly the way it caught the original
baseline-vs-v3.2-only loss in §4.7.1's STOP-THE-LINE gate: by
running the comparison and reporting whatever it shows.

### 4.7.3 Cross-family corroboration: Path 2 across six LLM lineages

The Path 2 finding could in principle have been gpt-4o-mini-specific:
that one LLM's adversarial-prompt compliance distribution might have
happened to dilute B2's signal less than the hybrid's. To test
whether the synthetic-vs-real gap is *structural* or
*model-specific*, we ran the same `seed_long_paragraphs` corpus
through six organisationally distinct LLM lineages — two closed
(OpenAI, Anthropic) and four open-weights (Meta, Alibaba/Qwen,
DeepSeek, Google/Gemma) — at the same four prompt classes
(neutral + a1/a2/a4 adversarial), produced one capture-once
snapshot per model, and re-ran the deterministic Phase 2 scorer.
The four open-weights members route through Hugging Face Inference
Providers via the OpenAI-compatible router; the closed pair use
their respective vendor APIs.

**No model produces a HYBRID_BEATS verdict on real-LLM
perturbations:**

| Model | $\Delta$(borda − b2) | Verdict |
|---|---:|---|
| meta-llama/Llama-3.3-70B-Instruct | $-0.047$ | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` |
| claude-haiku-4-5-20251001 | $-0.032$ | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` |
| google/gemma-3-27b-it | $-0.028$ | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` |
| gpt-4o-mini-2024-07-18 | $-0.021$ | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` |
| Qwen/Qwen3.6-35B-A3B | $+0.003$ | `HYBRID_TIES_BASELINE_ON_REAL_LLM` |
| deepseek-ai/DeepSeek-V3-0324 | $+0.018$ | `HYBRID_TIES_BASELINE_ON_REAL_LLM` |
| **Δ-spread across the six** | **0.065** | — |

**Joint finding: `STRUCTURAL_GAP_NO_MODEL_BEATS`.** Four of six
families have the hybrid LOSING; two TIE; **zero BEAT**. The
synthetic-bench WIN ($\Delta = +0.043$, §4.7.1) does not generalise
to *any* of the six LLM families in the cross-organisational
sample, closed or open-weights. The synthetic-vs-real gap is
structural rather than gpt-4o-mini-specific.

**Texture in the deltas.** Qwen ($+0.003$) and DeepSeek ($+0.018$)
are the only two models where the hybrid's $\Delta$ is even
weakly positive, and both fall short of the BEATS threshold
($+0.030$). The remaining four cluster between $-0.021$ and
$-0.047$ with roughly even spacing. The 0.065 spread between best
and worst is enough to suggest that *which* LLM renders the
perturbation matters quantitatively — Qwen and DeepSeek's
adversarial-prompt outputs may produce a perturbation distribution
slightly more like the synthetic harness's (where the hybrid wins)
than gpt-4o-mini, claude, Llama, and Gemma do — but not enough to
flip any model into BEATS territory.

Receipt:
`fixtures/bench_receipts/path2_multi_llm_compare_2026-05-05.json`,
schema `sum.sheaf_path2_multi_llm_compare.v1`. Per-model
`bench_digest`s pinned in
`Tests/research/test_sheaf_path2_multi_llm_compare.py`:
`7b364fc6…cc4b75e` (gpt-4o-mini), `d0f9f175…2f6f7` (claude-haiku),
`f1c17c3e…aac29b31` (Llama), `23da3ecb…461b8ea2` (Qwen),
`619a413f…2fe22c9f` (DeepSeek), `fe76913e…0318b9b5` (Gemma).

**Honest scope.** Six LLM lineages is meaningfully wider than
"a thin sample" but still bounded. The 16-doc corpus is held
constant across all six; results may differ on other corpora.
The four open-weights models are routed through one provider's
default selection (HF Inference Providers); other deployments
(local inference, alternative providers, different quantisation)
may produce slightly different perturbation distributions. The
load-bearing claim — *no LLM family in the sample makes the hybrid
BEAT the baseline on real-LLM perturbations* — is robust to those
caveats; the precise per-model $\Delta$ values are not.

### 4.7.4 Cross-corpus extension: §4.7.3 is corpus-specific

The §4.7.3 sample held the corpus constant
(`seed_long_paragraphs`, n=16 docs, encyclopedic prose). A natural
skeptical question: *is the structural-gap finding
corpus-independent?* To test, we extended the cross-family compare
to two additional corpora at n=5 commensurable models (open-weights
+ gpt-4o-mini; the Anthropic key was unavailable for the
extension). The two extensions are:

  - `seed_paragraphs` (8 docs, encyclopedic prose, shorter
    paragraphs) — same domain as `seed_long_paragraphs` but at
    half the scale.
  - `seed_news_briefs` (16 docs, news-wire prose, event-focused
    with named individuals/dates/organisations) — stylistically
    distinct: tense, structure, vocabulary all differ from the
    encyclopedic baseline.

**Per-corpus joint findings disagree:**

| Corpus | $n_\text{docs}$ | Joint finding |
|---|---:|---|
| `seed_long_paragraphs` | 16 | `STRUCTURAL_GAP_NO_MODEL_BEATS` |
| `seed_paragraphs` | 8 | `MIXED_VERDICTS_MODEL_DEPENDENT` |
| `seed_news_briefs` | 16 | `STRUCTURAL_GAP_NO_MODEL_BEATS` |

**Cross-corpus joint finding: `CROSS_CORPUS_VERDICTS_DIVERGE`.**

The per-(corpus, model) verdict matrix is jagged:
`seed_long_paragraphs` carries the full §4.7.3 n=6 set
(claude-haiku-4.5 is included via the committed snapshot from
PR #158); `seed_paragraphs` and `seed_news_briefs` are at n=5
because the operator's Anthropic key was unavailable during the
§4.7.4 capture. Total cells: 6 + 5 + 5 = 16.

| Model | `seed_long_paragraphs` | `seed_paragraphs` | `seed_news_briefs` |
|---|---:|---:|---:|
| gpt-4o-mini-2024-07-18 | $-0.021$ LOSES | **$+0.032$ BEATS** | $-0.023$ LOSES |
| claude-haiku-4-5-20251001 | $-0.032$ LOSES | — | — |
| meta-llama/Llama-3.3-70B-Instruct | $-0.047$ LOSES | $+0.005$ TIES | $+0.025$ TIES |
| Qwen/Qwen3.6-35B-A3B | $+0.003$ TIES | $+0.027$ TIES | $-0.016$ TIES |
| deepseek-ai/DeepSeek-V3-0324 | $+0.018$ TIES | $-0.042$ LOSES | $-0.007$ TIES |
| google/gemma-3-27b-it | $-0.028$ LOSES | $-0.014$ TIES | $-0.038$ LOSES |

Across the initial three-corpus, 16-cell matrix: **1 BEATS**
(`seed_paragraphs` × gpt-4o-mini), **8 TIES**, **7 LOSES**. The lone
BEATS cell sits right at the threshold ($+0.032$ vs $+0.030$ floor);
after partition filtering the effective sample is $n=6$ docs, near
the regime where AUC variance dominates. The hypothesis at this
point: extremal Goodhart on a sharp threshold at small $n$.

### 4.7.4.1 Resolving the lone BEATS cell: extremal-Goodhart confirmed

The §4.7.4 narrative as initially written did not discriminate
between two readings of the lone BEATS cell — *real corpus-dependent
advantage of the hybrid on shorter encyclopedic paragraphs* versus
*sample-size noise around a sharp threshold at $n=6$ effective docs*.
To discriminate, we authored an extended corpus
`scripts/bench/corpora/seed_paragraphs_16.json` (16 docs, all eight
originals from `seed_paragraphs` retained verbatim plus eight new
docs in the same encyclopedic voice — Mount Everest, Marie Curie,
the Great Wall of China, Titanic, Renaissance, atomic structure,
jet stream, blockchain). Same model set, same prompt classes, same
deterministic Phase-2 scorer; the only experimental variable is
sample size.

**The BEATS verdict does not survive at 16 docs:**

| Model | seed_paragraphs (8) | seed_paragraphs_16 (16) | Direction |
|---|---:|---:|---|
| gpt-4o-mini-2024-07-18 | **$+0.032$ BEATS** | $-0.013$ TIES | **flipped** |
| meta-llama/Llama-3.3-70B | $+0.005$ TIES | $-0.039$ LOSES | toward LOSES |
| Qwen/Qwen3.6-35B-A3B | $+0.027$ TIES | $-0.019$ TIES | sign flip, still TIES |
| deepseek-ai/DeepSeek-V3-0324 | $-0.042$ LOSES | $-0.022$ LOSES | toward TIES |
| google/gemma-3-27b-it | $-0.014$ TIES | $-0.027$ LOSES | toward LOSES |
| **Joint** | `MIXED_VERDICTS_MODEL_DEPENDENT` | **`STRUCTURAL_GAP_NO_MODEL_BEATS`** | resolved |

Doubled $n$ pulls every per-model $\Delta$ closer to its likely
population-mean value, the BEATS cell collapses into TIES, and the
joint finding consolidates to `STRUCTURAL_GAP_NO_MODEL_BEATS` —
which is precisely what extremal-Goodhart predicts when an
optimisation pressure (the $+0.030$ BEATS threshold) acts on a
high-variance estimator at small sample sizes (Manheim & Garrabrant,
arXiv:1803.04585, 2018).

**Updated four-corpus aggregate.** With `seed_paragraphs_16`
included, the cross-corpus aggregate spans four corpora, jagged 21
cells: 1 BEATS, 10 TIES, 10 LOSES. **Three of four corpora at $n
\geq 16$ docs reproduce `STRUCTURAL_GAP_NO_MODEL_BEATS`**; the
fourth (`seed_paragraphs`, $n=8$) is the small-$n$ canary the
extension was designed to test, and its lone BEATS cell is now
explained.

| Corpus | $n_\text{docs}$ | $n_\text{models}$ | Joint finding |
|---|---:|---:|---|
| `seed_long_paragraphs` | 16 | 6 | `STRUCTURAL_GAP_NO_MODEL_BEATS` |
| `seed_paragraphs` | 8 | 5 | `MIXED_VERDICTS_MODEL_DEPENDENT` (small-$n$) |
| `seed_paragraphs_16` | 16 | 5 | `STRUCTURAL_GAP_NO_MODEL_BEATS` |
| `seed_news_briefs` | 16 | 5 | `STRUCTURAL_GAP_NO_MODEL_BEATS` |

Aggregate joint finding remains `CROSS_CORPUS_VERDICTS_DIVERGE`
because the 8-doc and 16-doc per-corpus joint findings disagree —
but this is a *measurement-artifact* divergence, not a
substantively different claim about the underlying detector. The
honest top-level reading is now:

  1. **At controlled sample sizes ($n \geq 16$), the
     synthetic-bench WIN does not generalise to any LLM family in
     the cross-organisational sample.** Three of three corpora at
     $n \geq 16$ produce `STRUCTURAL_GAP_NO_MODEL_BEATS` across
     OpenAI, Anthropic (where measured), Meta, Alibaba, DeepSeek,
     Google.
  2. **At small samples ($n = 8$), AUC variance dominates and
     thresholded verdicts can flip by chance.** The $n=8$
     `seed_paragraphs` cell with gpt-4o-mini at $\Delta = +0.032$
     is the worked example. The same model on the same style at
     $n=16$ gives $\Delta = -0.013$.
  3. **The synthetic-vs-real *magnitude* gap is robust to corpus
     and to LLM family.** $\Delta = +0.043$ on the synthetic
     harness sits substantially above every real-LLM cell at
     $n \geq 16$; the synthetic harness was a measure-that-became-
     a-target whose discriminative power does not survive contact
     with naturalistic perturbation distributions, exactly as
     Goodhart's law predicts in its regressional form.

**Honest scope.** Four corpora is still bounded; only `seed_news_briefs`
is stylistically distinct from the encyclopedic baseline. A robust
*corpus-independent* claim would want 5–10 corpora spanning genres
(scientific abstracts, fiction, legal/policy, code commentary,
spoken transcripts). The 5-model $n$ on three of four corpora is
less than the §4.7.3 $n=6$ because the Anthropic key was unavailable
during the §4.7.4 / §4.7.4.1 captures; if a future run adds
claude-haiku-4.5 on the additional corpora the matrix could shift,
but the *direction* of the consolidated finding (no consistent
BEATS at $n \geq 16$) is unlikely to flip with a single additional
model.

Receipts: per-corpus
`fixtures/bench_receipts/path2_multi_llm_compare_<corpus>_<date>.json`
and aggregate
`fixtures/bench_receipts/path2_cross_corpus_compare_2026-05-06.json`
(schema `sum.sheaf_path2_cross_corpus_compare.v1`). All 21 per-(corpus,
model) `bench_digest`s pinned in
`Tests/research/test_sheaf_path2_cross_corpus.py`.

**Remaining v0.4+ candidate directions:**

  - **Real-LLM-aware detector**: train the per-triple V channel on
    a corpus of LLM-rendered perturbations rather than synthetic
    tail-perturbations, so the trained restriction maps reflect
    the real-LLM perturbation distribution.
  - **Naturalistic perturbation synthesis**: have an LLM generate
    A1/A2/A4-class perturbations on the source TRIPLE set
    directly (not the rendered prose), so the perturbation
    structure matches synthetic but the perturbation choice is
    LLM-natural. Decouples "what gets perturbed" from "how it
    propagates through rendering."
  - **Deeper corpus sampling**: expand to 5-10 stylistically
    distinct corpora at $n \geq 16$ docs to harden the
    *no consistent BEATS at controlled sample sizes* finding
    against further objections.

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

**Cross-machine verification on Modal x86_64
(three environments × three benches, all MATCH).** We re-ran each
load-bearing bench inside a `modal.Image.debian_slim(...)` container
at the pinned commit SHA, in two Python environments simultaneously —
Python 3.10 with numpy 1.25 and Python 3.12 with numpy 2.x. Both
container environments differ from the operator's reference machine
across every dimension that matters for floating-point
reproducibility:

| | Operator reference | Modal Py 3.10 | Modal Py 3.12 |
|---|---|---|---|
| Architecture | Apple Silicon (`arm64`) | x86_64 | x86_64 |
| OS | Darwin 25.0.0 | Linux 4.4.0 / glibc 2.31 | Linux 4.4.0 / glibc 2.31 |
| Python | 3.10 (miniforge) | 3.10.8 (Debian slim) | 3.12 (Debian slim) |
| numpy | (operator-side) | 1.25.0 | 2.x |
| LAPACK provider | Apple Accelerate | OpenBLAS-via-PyPI-wheel | OpenBLAS-via-PyPI-wheel |
| SIMD | NEON | AVX2 | AVX2 |

All four load-bearing bench digests reproduced byte-for-byte across
all three environments:

| Bench | Operator | Modal Py 3.10 | Modal Py 3.12 |
|---|---|---|---|
| v3.2 validation | `b4d26c01…` | ✓ | ✓ |
| complementary hybrid | `dc6e0260…` | ✓ | ✓ |
| predicate negatives | `ddf41484…` | ✓ | ✓ |
| hybrid comparison | `a7965803…` | ✓ | ✓ |

The substantive verdicts also reproduce: `HYBRID_BEATS_BASELINE`
($\Delta = +0.043$ trusted-mean vs B2 alone; trusted-mean AUC
$0.876$) holds in all three environments; `A2_STILL_CHANCE` (the
cochain-blindness diagnosis underlying §4.7.1's structural finding)
also holds. Outcome label:
**`BRANCH_A_THREE_ENVIRONMENTS_DIGESTS_MATCH`**.

The `predicate_negatives` cross-version digest stability was
specifically verified after a v0.2 latent-fix refactor that replaced
a local v2-training-loop copy with a call to production
`train_restriction_maps(..., n_predicate_negatives_per_positive=3)`.
The pre-refactor bench produced different digests between Modal
Python 3.10 (`aa34b6e8…`) and Modal Python 3.12 (`8638253903…`)
because the local SGD trajectory accumulated ULP-level differences
across 200 epochs on different LAPACK/numpy builds; the post-
refactor bench (single training-loop source) produces identical
`ddf41484…` across both Python versions.

Receipt: `fixtures/bench_receipts/cross_machine_verification_2026-05-05.json`,
schema `sum.cross_machine_verification.v1`. Verification harness:
`scripts/research/cross_machine_verify_modal.py` — any reader with
Modal credits can rerun via `modal run` against the pinned SHA
(currently `5715c40` post-latent-fixes) and verify all three digests
match across both Modal Python versions.

The digest is built on the same canonicalisation primitive
(JCS / RFC 8785) the project's render receipts use, and is
therefore signable with the project's existing JWKS keys. An
arXiv preprint citing the digest gives external readers a
byte-level fixed point: rerun the bench, recompute the digest,
match against the published value, and any divergence is either
upstream code change or environment drift — both load-bearing
findings in their own right.

We use a hash to anchor reproducibility, drawing on the project's
existing JCS canonicalisation machinery; this is good engineering
practice rather than a novel research primitive, and we name it
explicitly so external readers have a byte-level fixed point to
verify against. The closest neighbours in the broader literature
are the DOI-anchored evaluation bundles in classical ML benchmarks
(which carry source data hashes but not quantised-result digests)
and verifiable-computation pipelines (zkML provers, Sumcheck-based
result attestation; these address a stronger threat model than
ours but at substantially higher cost). Our specific contribution
is composing the result-digest with the JWKS-signed render-receipt
substrate so the *aggregate* claim is verifiable, not just the
*input* — and demonstrating that this composition reproduces across
two distinct LAPACK environments (Apple Accelerate and OpenBLAS).
Cross-machine reproducibility beyond these two environments is
unmeasured (v0.2 candidates: ARM Linux, Intel MKL builds).

### 4.9 Scaling characterisation

The substrate's load-bearing operations were measured at corpus
sizes $N \in \{100, 500, 1000, 2000, 5000, 10000\}$ axioms with
single-threaded BLAS and the Zig FFI bridge active. Receipt:
`fixtures/bench_receipts/performance_characterisation_2026-05-07.json`,
schema `sum.performance_characterisation.v1`,
`bench_digest 839f4a7f…3ad74`. Pinned in
`Tests/research/test_performance_audit.py`. Full analysis with
candidate accelerations and recursive-compression feasibility envelope
in `docs/PERFORMANCE_CHARACTERISATION.md`.

**Empirical scaling exponents (log-log fit, p50_us vs N):**

| Operation | Fitted $k$ | Coefficient $a$ (µs) | Reading |
|---|---:|---:|---|
| `ingest` (per-triple) | $0.001$ | $45.6$ | constant per-triple |
| `entail` | $0.981$ | $0.144$ | linear in $N$ |
| `merge` | $1.497$ | $19.05$ | sub-quadratic empirical |
| `encode` | $1.909$ | $0.048$ | near-quadratic |

**`merge` is the bottleneck at every measured size**, growing from
24.6 ms at N=100 to **23.9 s at N=10000**. The PROOF_BOUNDARY §4
prior quote (519 ms at N=1000) reproduces within 10% locally
(476 ms). The $k=1.497$ exponent is sub-quadratic relative to the
naive O(N²) big-int LCM bound — the Zig FFI path's GCD
implementation is a real win — but at N=10000 the constant has
already dominated. The cProfile output (in the receipt's
`bottleneck_profile_truncated` field) shows the hot path is
concentrated in `math.gcd` calls under LCM.

**Library-scale feasibility envelope (extrapolated above N=10000):**

| Target | $N$ | per-iter estimate | extrapolated? |
|---|---:|---:|---|
| small book | 1,000 | 0.62 s | measured |
| medium book | 5,000 | 7.12 s | measured |
| large book | 10,000 | 20.6 s | measured |
| small library | 50,000 | 4.2 min | extrapolated |
| modest library | 100,000 | 12.5 min | extrapolated |

Reading: the substrate scales comfortably to the **low thousands**
of axioms; acceptably for batch use to the **low tens of thousands**;
and requires architectural change (the Phase 26 property-graph
backing store proposed in `PROOF_BOUNDARY.md` §3) for **library
scale** above ~50k. The recursive-compression workload (iterated
re-extraction → axiom-set compression) is feasible *today* on the
current substrate up to medium-book corpora; library-scale
recursive compression is gated on Phase 26.

### 4.10 Recursive compression and the SUM (first measured pass)

The original-vision claim — *the SUM is the incompressible point
of a corpus, beyond which compression causes degradation* — was
measured for the first time in this work. Two compressors were
applied iteratively: $A_k = \text{sieve}(R(A_{k-1}))$, $A_0 =
\text{sieve}(\text{prose}_0)$, until the axiom signature reaches a
fixed point or a short cycle. The SUM at threshold $\tau$ is the
smallest $A_k$ along the walk satisfying
$\text{recall}_k = |A_k \cap A_0| / |A_0| \ge \tau$.

Receipts:
[`fixtures/bench_receipts/recursive_compression_walk_deterministic_2026-05-08.json`](../../fixtures/bench_receipts/recursive_compression_walk_deterministic_2026-05-08.json)
(`bench_digest 778cd95d…d85090`) and
[`fixtures/bench_receipts/recursive_compression_walk_llm_gpt-4o-mini-2024-07-18_2026-05-08.json`](../../fixtures/bench_receipts/recursive_compression_walk_llm_gpt-4o-mini-2024-07-18_2026-05-08.json)
(`bench_digest 6d779aa4…edeb26`). Schema
`sum.recursive_compression_walk.v1`. Pinned in
`Tests/research/test_recursive_compression_walk.py`. Full analysis
in `docs/RECURSIVE_COMPRESSION.md`.

**Two arms, two findings.**

| Compressor $R$ | Corpus | median fp step | median fp $n_\text{axioms}$ | median fp recall |
|---|---|---:|---:|---:|
| deterministic canonical | `seed_long_paragraphs` | 3 | 3 | $0.333$ |
| deterministic canonical | `seed_news_briefs` | 2 | 1 | $0.225$ |
| LLM grammatical (gpt-4o-mini) | `seed_long_paragraphs` | 2.5 | 7 | $0.59$ |
| LLM grammatical (gpt-4o-mini) | `seed_news_briefs` | 2 | 4 | **$0.80$** |

The deterministic arm reveals a **sieve↔canonical-tome asymmetry**
not previously documented: $\text{sieve}(\text{canonical\_tome}(A))
\ne A$ in general, because the canonical tome renders bare lemmas
("The alice like cat.") that the sieve cannot fully re-extract.
PROOF_BOUNDARY §1.1's "lossless round-trip" claim holds at the
state-integer level (`parse(canonical_tome(S))` decodes to $S$'s
state) but does *not* hold at the triple level. This is a
substantive architectural finding the project did not have before.

The LLM arm shows the recursive-compression vision is empirically
real: median fixed-point recall of 0.59-0.80 vs the original axiom
set, fixed points reached in 2-3 steps, and **per-doc SUM
identification at multiple thresholds**. Three doc-categories
emerged: *at-fixed-point* (recall stays at 1.0 — prose is its own
SUM), *gradual decay to a stable subset*, *sharp drop to a
different stable subset of similar size* (the LLM-paraphrase
canonicalises to different triples that re-extract stably).

**Implications for the original vision.** The SUM is corpus- and
doc-dependent, computationally feasible on the current substrate
up to medium-book corpora (per §4.9), and gates on Phase 26 for
library scale. Multi-modal compression dispatch (axioms vs parables
vs poetry vs quotes vs emoji) is a v0.5+ research direction this
arm does not yet enter; cross-LLM-family extension would mirror
the §4.7.x methodology directly. The substrate's
sieve↔canonical-tome asymmetry should be explicit in any future
narrative about the bidirectional pipeline — both rendering paths
have valid uses, and which one is appropriate depends on whether
the downstream consumer needs state-integer round-trip or
triple-level round-trip.

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

We position the contribution at two layers, with the substrate
as the primary claim (`cs.CR`) and the detector arc as the
worked example (`cs.LG`).

### 6.1 Substrate layer (cs.CR primary)

**vs. published reproducibility primitives.** The closest
neighbour to the `bench_digest` substrate is the
DOI-anchored evaluation-bundle pattern in classical ML
benchmarks (e.g. ML reproducibility checklists; HuggingFace
datasets). Those carry source-data hashes but typically not
quantised-result digests; they ensure the *input* is fixed but
not that the output reproduces byte-for-byte. Our digest commits
to *the result*, not just the substrate. Combined with the
JWKS-signed render receipts (§3.1), the result-digest forms an
end-to-end claim: an external reader can rerun the bench,
reproduce the digest, and verify the provenance of the source
bundle through Ed25519 signature validation against the
project's published JWKS — without trusting any intermediate
party. We are not claiming this specific composition is novel
in the abstract; we *are* claiming it is missing from the
LLM-eval literature, where reproducibility claims typically stop
at the prompt level. §4.8 demonstrates the digest reproduces
across two distinct LAPACK environments (Apple Accelerate +
OpenBLAS-via-PyPI), so the claim is not "byte-stable on one
machine" but "byte-stable across distinct floating-point stacks
that share the project's pinned numerics."

**vs. cryptographically-anchored ML-evaluation pipelines.** The
substrate is comparable in spirit to verifiable-computation
pipelines (zkML provers, Sumcheck-based result attestation),
but at a different threat-model layer. zkML proves *the model
executed correctly on a committed input* — a strong claim
under T1 (adversarial render) of §3.0. Our substrate proves
*the bench executed and produced this specific aggregate* — a
weaker claim than zkML's, but at substantially lower cost and
without the model-architecture restrictions that current zkML
implementations impose. The two compose naturally: a future
v0.X could anchor the model-execution layer in zkML and the
evaluation-aggregate layer in `bench_digest`, defending T1
cryptographically rather than statistically.

**vs. compliance-only audit-log substrates.** Industry
compliance tools (SIEM aggregators, PCI-attestation packages)
typically pin record-keeping schemas without integrating result
attestation. The §5 six-regime layer of the substrate is shaped
by their conventions, but composed with the receipt and digest
machinery so a passing compliance report and a verified
bench_digest can be cross-referenced through a shared
`audit_log.v1` row. We are not claiming the compliance
validators themselves are novel; the integration into the
substrate (so detector outputs and compliance checks share an
auditable provenance chain) is.

### 6.2 Detector layer (cs.LG secondary)

**vs. token-level uncertainty (sequence log-probability,
semantic entropy):** complementary. Uncertainty estimates
measure the model's own confidence; we measure cross-claim
consistency under a verifiable-source binding. The two should
compose; our substrate makes the composition's reproducibility
verifiable.

**vs. retrieval-grounded verification (RAG with citations):**
complementary. RAG confirms support for retrievable claims; we
score consistency of any rendered claim against an attested
source bundle, including claims whose support is in the sheaf
but not in a retrieval index.

**vs. existing knowledge-graph hallucination detectors
(HalluGraph, KG-grounded checks):** structurally adjacent, with
three distinguishing features. First, the *math* of the cochain
and per-triple channels is grounded in Hansen-Ghrist
sheaf-Laplacian theory (and its weighted / harmonic-extension
generalisations) and Gebhart contrastive sheaf-embedding
training, not ad-hoc consistency rules. Second, the *provenance*
is cryptographic — the source bundle is a signed artifact whose
verifier is realiser-independent (Python / Node / browser
byte-identity locked in CI). Third, the *evaluation* is
honest about the detector's competitive position: §4.7.1
documents the recovery arc that produced the WIN, including the
intermediate experiments that lost (predicate-perturbation
training failed; cochain-only Borda fusion failed). The hybrid
itself — Borda(v3.2 + per-triple, B2) — is one composition
choice among many, presented with the discipline that surfaced
its limits, not as a competitive-claim headline.

**vs. Tull, Kleiner & Smithe 2023 categorical active inference:**
the compositionality theorem (Theorems 45/46) says free energy
is additive across sequential and parallel composition of
generative models. Our v3.2 combined detector and the §3.9
hybrid are both mathematically sums of orthogonal terms — the
same shape — over per-doc subgraphs of a multi-document corpus.
Federated multi-agent scoring is the natural application;
v3.1's harmonic extension over a trusted-issuer boundary is the
SUM-specific instantiation, with v3.2 serving as the F3-aware
fall-back, and the §3.9 hybrid providing the substrate-level
composition primitive that the categorical-active-inference
framing identifies as additive in free-energy terms.

## 7. Bounded claims

This section uses a four-tier audit framework: (1) claims that hold
up under verification across the full §4.7.x extension arc, (2)
claims now corrected from prior overstatements (the §4.7.x extension
generated negative-going findings the original text had to
incorporate honestly), (3) claims that are real but narrow in scope,
(4) limits and future-work hooks. The framework borrows its shape
from the empirical-claim audit any peer reviewer would want to do
on a paper of this type; we apply it to ourselves first.

### 7.1 Claims that hold up under verification

**Substrate-layer (load-bearing, pinned in CI):**

- **Cross-machine `bench_digest` reproducibility.** Four benches
  (`v3_2_validation`, `complementary_hybrid`, `predicate_negatives`,
  `hybrid_comparison`) reproduce byte-for-byte across three
  environments — Apple Accelerate on Apple Silicon (operator),
  OpenBLAS Py 3.10 / numpy 1.25 on Modal x86_64, and OpenBLAS Py
  3.12 / numpy 2.x on Modal x86_64; §4.8.
- **Cryptographic auditability.** Receipts are RFC 8785 JCS-canon
  + Ed25519 signed; an external verifier with the source bundle
  and trained sheaf parameters reproduces every numeric result
  offline.
- **Continuous-enforcement discipline catches drift.** The PR #160
  dict-order fix is the worked example: a determinism bug
  (in-memory snapshots in corpus order, cached snapshots in
  alphabetical order — different trained sheaves, different
  digests) was caught by a probe-style investigation around the
  digest pin and root-caused as a one-line fix
  (`docs_with_src.sort(key=lambda x: x[0])`). A new regression
  test asserts Phase 2 is invariant to snapshot dict order.
  PROOF_BOUNDARY §2.10 frames this as continuous enforcement
  against the analogue of mutualism breakdown (Sachs, Mueller,
  Wilcox & Bull, *QRB* 79:135, 2004).

**Detector-layer (load-bearing on `seed_long_paragraphs`):**

- **Synthetic Borda hybrid Δ = +0.043** above the strongest
  single-signal baseline B2 jaccard ($0.876$ vs $0.833$
  trusted-mean AUC) on the §4.4 / §4.7 synthetic A1/A2/A4 harness.
- **Localised top-1** on the synthetic micro-benchmark.
- **Five named falsification verdicts** (F1 MARGINAL, F2 PASS,
  F3 STRUCTURAL FAIL, F4 PASS, F5 PASS at $\gamma \leq 0.1$)
  reproduce across the cross-machine matrix.

**Detector-layer (load-bearing on the §4.7.x extension at $n \geq
16$ docs):**

- **No hybrid BEATS verdict on real-LLM perturbations across the
  cross-organisational sample at controlled sample sizes.** Across
  three corpora at $n \geq 16$ docs (`seed_long_paragraphs`,
  `seed_paragraphs_16`, `seed_news_briefs`) and six LLM lineages
  where measured (OpenAI gpt-4o-mini, Anthropic claude-haiku-4.5
  on `seed_long_paragraphs` only, Meta Llama-3.3-70B, Alibaba
  Qwen3.6-35B-A3B, DeepSeek V3-0324, Google Gemma-3-27B), the
  hybrid never produces `HYBRID_BEATS_BASELINE_ON_REAL_LLM`. Joint
  finding `STRUCTURAL_GAP_NO_MODEL_BEATS` reproduces in 3/3
  corpora at $n \geq 16$.

### 7.2 Claims corrected from prior overstatements

The §4.7.x extension produced two negative-going findings the
preprint absorbs honestly rather than rewrites away:

- **§4.7.3 originally read as "no LLM family makes the hybrid BEAT
  the baseline on real LLM."** The §4.7.4 cross-corpus extension
  produced one BEATS cell on `seed_paragraphs` × gpt-4o-mini at
  $\Delta = +0.032$ — right at the $+0.030$ threshold, with $n=6$
  effective docs post-partition. §4.7.4.1 controlled for sample
  size by extending the corpus to 16 docs (same encyclopedic
  voice, eight new docs added) and the BEATS verdict flipped to
  TIES at $\Delta = -0.013$ for the same model and style. The
  corrected reading: **at controlled sample sizes ($n \geq 16$),
  no model BEATS; at small $n$ ($n=8$), AUC variance around the
  threshold can produce isolated BEATS cells via extremal
  Goodhart**.
- **Synthetic-bench WIN as a universal-detector claim.** The
  $+0.043$ advantage is on one specific corpus's synthetic
  perturbation distribution. The full §4.7.x extension shows the
  hybrid's edge over B2 does not generalise to naturalistic
  perturbations from any LLM family at $n \geq 16$. This is
  consistent with Goodhart's law in its regressional form
  (Manheim & Garrabrant, arXiv:1803.04585, 2018): the hybrid was
  selected to compose well on the synthetic measure, and that
  measure stopped being a good measure once it became the target
  of optimisation. The substrate's truth-first discipline catches
  this exactly the way it caught the v3.x cochain-only loss in
  §4.7.1's STOP-THE-LINE gate.

### 7.3 Claims that are real but narrow

- **v3.1 boundary-deviation has no standalone use.** F3 STRUCTURAL
  FAIL is mathematical, not parametric. v3.2 closes the
  *detector-layer* problem only via composition with
  $v_\text{laplacian}^w$; standalone v3.1 deviation remains
  uninformative.
- **Cochain channel is not standalone-competitive.** §4.7.1
  documents the cochain-only loss to trivial baselines
  ($-0.174$ trusted-mean vs B2). The §4.7.1 WIN comes from
  composition with the §3.5 per-triple channel and the B2
  baseline. Operators expecting cochain-only detection (e.g. for
  closed-vocab schemas where per-triple OOV signal is
  uninformative) should not rely on the cochain channel alone.
- **Predicate-perturbation training cannot fix cochain
  blindness.** `aa34b6e8…` digest pin in
  `Tests/research/test_recovery_experiment_digests.py`. The
  structural blindness to entity-set-preserving perturbations is
  mathematical, not parametric. Predicate sensitivity must come
  from the per-triple channel, not the cochain.
- **Cross-machine reproducibility is verified for two LAPACK
  families only.** Apple Accelerate + OpenBLAS-via-PyPI cover the
  three measured environments. Intel MKL, ARM Linux OpenBLAS, AMD
  AOCL remain unmeasured.
- **$\lambda$ auto-calibration is per-corpus.** The hybrid's
  $+0.043$ margin is calibrated for `seed_long_paragraphs`'s
  specific perturbation distribution. Cross-corpus deployments
  need to recalibrate.
- **F5 ($\gamma_\text{auto}$) FAILs.** Optimal $\gamma$ is
  small ($\leq 0.1$); future deployments must measure $\gamma$
  rather than auto-derive it from term magnitudes.
- **Borda is one composition choice.** Z-score additive, gated
  (B2 fires → use B2; else use v3.2 + per-triple), and
  weighted-linear are alternatives unevaluated.

### 7.4 Limits and future-work hooks

- **Not a solution to hallucination.** The hybrid is one signal
  composition among many; production would compose with
  confidence calibration, retrieval-grounded checks, NLI
  verifiers, and human review.
- **Not a correctness proof.** The detector measures
  inconsistency in the *output manifold* under cross-edge
  agreement, predicate violations, and entity-set patterns; it
  does not certify that any single output is *correct*. A
  consistent manifold of uniformly-wrong renderings remains
  undetected.
- **Not computable on arbitrary corpus sizes.** Sparse-block
  storage of the $d$-dim Laplacian is straightforward but
  unimplemented; scaling to $> 10^4$ vertices needs the
  sparsification machinery of Hansen-Ghrist 2019 §6.
- **v0.4+ candidates** named in §4.7.4.1: real-LLM-aware
  per-triple V training; naturalistic perturbation synthesis on
  the source TRIPLE set rather than the rendered prose;
  deeper corpus sampling (5-10 corpora at $n \geq 16$ to harden
  the *no consistent BEATS at controlled n* finding against
  further objections).

Five named falsification verdicts (F1 MARGINAL, F2 PASS,
F3 STRUCTURAL FAIL, F4 PASS, F5 PASS at $\gamma \leq 0.1$)
together with the synthetic-corpus disconnect (H12 synthetic-pass
vs F3 corpus-fail) and the four recovery-experiment digests
(`a7965803…` Borda(v3.2_only, B2) loses; `aa34b6e8…`
predicate-perturbation training fails to lift A2; `7025436f…`
per-triple integration lifts A2; `dc6e0260…` complementary
Borda WINS) testify to which signals the substrate-enabled
detector does and does not carry. Three pinned-in-code
falsifications from earlier in the arc — (a) v1
disconnected-graph density-dropout blindness; (b) v2.1
presence-cochain inheritance of the same blindness; (c) v2.2
default-$\lambda$ at corpus scale — remain as regression tests
at every PR.

## 8. Future work

- **Path 2 — real LLM-rendered adversarial bench.** Synthetic
  perturbations are existence-proofs; adversarial LLM renderings
  stress the detector differently. Generate clean and adversarial
  variants via the hosted Worker render path and re-run §4.4 /
  §4.7.
- **A2 closure already partial via §3.5 per-triple channel
  composition** (v0.1). Predicate-perturbation training negatives
  alone do NOT lift A2 for the cochain channel
  (`aa34b6e8…` digest pin) — the cochain is mathematically blind
  to entity-set-preserving perturbations, so training-distribution
  changes can't fix what scoring discards. The recovered A2 signal
  comes from the per-triple channel directly (lifting A2 from
  $0.500 \to 0.671$); v0.2 candidates: predicate-perturbation
  training to *strengthen* the per-triple channel further; cochain
  redesign to make the cochain itself sensitive (per below).
- **Cross-machine reproducibility — additional LAPACK
  environments.** §4.8 covers Apple Accelerate (operator) and
  OpenBLAS-via-PyPI (Modal x86_64). v0.2 candidates: Intel MKL,
  ARM Linux OpenBLAS, AMD AOCL. A future Node / browser
  reimplementation of the v3 / v3.2 detectors that matches the
  `bench_digest` would extend the cross-runtime trust triangle
  from the verifier layer (K1–K4) to the research bench layer;
  the digests in §4.6 / §4.7 / §4.7.1 are the published fixed
  points such a port would have to match.
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
- **OtotaO/SUM repository.** `sum-engine` v0.6.0 on PyPI;
  source at https://github.com/OtotaO/SUM. Spec doc at
  `docs/SHEAF_HALLUCINATION_DETECTOR.md`. Library API surface at
  `docs/SHEAF_LIBRARY_API.md`. v3.x bench scripts at
  `scripts/research/sheaf_v3_roc_bench.py`,
  `scripts/research/sheaf_v3_1_f3_diagnostic.py`,
  `scripts/research/sheaf_v3_2_validation.py`.
  Recovery-arc bench scripts at
  `scripts/research/sheaf_baseline_comparison.py`,
  `scripts/research/sheaf_hybrid_comparison.py`,
  `scripts/research/sheaf_predicate_negatives_experiment.py`,
  `scripts/research/sheaf_per_triple_integration_experiment.py`,
  `scripts/research/sheaf_complementary_hybrid_experiment.py`.
  Cross-machine verification harness at
  `scripts/research/cross_machine_verify_modal.py`. Receipts under
  `fixtures/bench_receipts/`: `v3_roc_bench_2026-05-03.json`,
  `v3_1_f3_diagnostic_2026-05-03.json`,
  `v3_2_validation_2026-05-03.json`,
  `baseline_comparison_2026-05-04.json`,
  `hybrid_comparison_2026-05-04.json`,
  `predicate_negatives_experiment_2026-05-04.json`,
  `per_triple_integration_2026-05-04.json`,
  `complementary_hybrid_2026-05-04.json`,
  `cross_machine_verification_2026-05-04.json`. Test pins at
  `Tests/research/test_sheaf_baseline_comparison.py` and
  `Tests/research/test_recovery_experiment_digests.py`.

---

*Acknowledgements.* This work would not exist without the
foundational categorical-AI program of Spivak, Kent, Coecke,
Curry, Hansen, Ghrist, Gebhart, Schrater, Tull, Kleiner, Smithe,
and the broader Topos Institute / Quantinuum / ACT community.
SUM contributes the cryptographic substrate
(receipts + bench_digest + audit-log + threat model), the
recovery-arc methodology (§4.7.1) that produced the
complementary-signal hybrid, and the cross-machine reproducibility
measurement (§4.8); the underlying mathematics is theirs.

*Reproducibility.* All code Apache-2.0. Local reproduction:
`pip install 'sum-engine[research,sieve]'` plus the bench scripts
listed in §References. Cross-machine reproduction via Modal:
`modal run scripts/research/cross_machine_verify_modal.py` against
the pinned commit SHA — any reader with Modal credits can verify
both load-bearing digests match. Receipt JSON schemas are stable
and versioned (`sum.render_receipt.v1`,
`sum.sheaf_v2_roc_bench.v1`,
`sum.sheaf_v3_roc_bench.v1`,
`sum.sheaf_v3_1_f3_diagnostic.v1`,
`sum.sheaf_v3_2_validation.v1`,
`sum.sheaf_baseline_comparison.v1`,
`sum.sheaf_hybrid_comparison.v1`,
`sum.sheaf_predicate_negatives_experiment.v1`,
`sum.sheaf_per_triple_integration.v1`,
`sum.sheaf_complementary_hybrid.v1`,
`sum.cross_machine_verification.v1`). Each bench receipt carries
a `bench_digest` field — JCS-canonical SHA-256 over the quantised
payload — that an external reader can match byte-for-byte after
re-running the bench.

*Status of claims.* §4.1 (synthetic micro-bench), §4.3 (v2.2 ROC
bench), §4.4 (v3 ROC bench), §4.6 (F3 diagnostic), §4.7 (v3.2
validation), §4.7.1 (recovery arc + complementary hybrid WIN), and
§4.8 (cross-machine bench_digest) are measured, pinned, and
reproducible at the commit hash of this draft. §4.2's
falsifications are pinned in code by named regression tests.
§5's compliance evidence is mechanically verified by the
per-regime test suites and the cross-regime CLI dispatch test.
§6 positioning claims are this author's reading of the cited
literature; corrections welcomed before arXiv submission.

*Authors and contact.* Draft authored 2026-05-01 (v0); revised
2026-05-04 (v0.1) by the SUM project, with the substrate-first
reframe and recovery-arc methodology added in the same revision.
For corrections / contributions: https://github.com/OtotaO/SUM/issues.
Pre-arXiv comments welcome; pre-circulation review (1–2 readers)
sought before submission to `cs.CR` (primary) / `cs.LG`
(secondary).
