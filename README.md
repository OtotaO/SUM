# SUM — The Semantic Understanding Machine

[![SUM Knowledge OS CI](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml/badge.svg)](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml)

> **From Tags to Tomes and Back Again — canonical round-trip proven, full pipeline measured.**

SUM began as a humble bidirectional knowledge distillation engine: turn **structured facts** (tags) into **coherent narratives** (tomes) and vice versa, with adjustable knobs for density, length, formality, audience, and perspective. What emerged is a **semantic algebra** that represents knowledge as prime-factored integers, giving a mathematically proven round-trip on the canonical representation and an empirical faithfulness score on the full text→structure→text loop.

**The core insight remains unchanged**: knowledge should flow fluidly between structured and narrative forms. That flow is **cryptographically verified** in every signed bundle, **mathematically guaranteed** on the canonical layer (round-trip drift = 0.00%, proven), and **continuously measured** on real prose (96% FActScore on `seed_v1`, tracked by the bench harness). Every claim in this repo is labelled with an explicit epistemic status — see [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) for the separation of proved from measured.

---

## 📊 Current Measured State

Every headline number below is reproducible via `python -m scripts.bench.run_bench`. See [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) for the full truthfulness document and per-claim epistemic status.

| Axis | Value | Epistemic Status |
|---|---|---|
| Canonical round-trip drift | **0.00 %** | **provable** (Ouroboros protocol, §1.1) |
| Extraction F1 on `seed_v1` (50 docs) | **1.000** | empirical-benchmark |
| Regeneration FActScore on `seed_v1` | **0.960** | empirical-benchmark |
| Sieve re-extract of canonical (known ceiling) | 54 % drift | empirical-benchmark |
| Merge p50 @ N=1000 primes | ~518 ms (~O(n²)) | empirical-benchmark |
| Test suite | **756+** tests | continuous |

---

## 🌱 The Evolution: From Simple to Sublime

### Genesis: The Original Vision (2023)
```
Text → Structured Facts → Text
  ↑                        ↓
Adjustable Parameters & Style Control
```

**Problem**: How do we bidirectionally transform between:
- **Tags** (structured semantic facts: `alice||age||30`)
- **Tomes** (natural language narratives: "Alice is 30 years old...")
- With **adjustable knobs** for tone, detail, perspective, and focus

### Evolution: The Mathematical Substrate (2024-2026)
```
Perspectival Text → Prime-Encoded Facts → Verified Narratives
        ↑                    ↓                        ↓
   Multi-Viewpoint    Gödel Integer          Round-Trip Verified
    Classification   (Single Source of Truth)    Generation
```

**Breakthrough**: Every fact becomes a **unique prime number**. The entire knowledge state is a **single integer** (product of all active primes). This enables:
- **Verified Generation**: `State % Prime == 0` proves a fact is grounded
- **Zero-Cost Sync**: Send one integer, use GCD to compute exact deltas
- **Git for Truth**: Branch = copy integer; Merge = LCM operation
- **Multi-Perspectival Views**: Same facts, different narrative styles per viewpoint

### Future: The Polytaxis Integration (2026+)
```
Multi-Perspective Classification ←→ Verified Multi-Narrative Generation
            ↑                                    ↓
    Formal Ontological Bridges          Category-Theoretic Alignment
         ↑                                    ↓
Semantic Branching & Temporal Evolution ←→ Accountable Pluralism
```

**Vision**: SUM's mathematical substrate becomes the foundation for **Polytaxis** — a meta-classification system that hosts multiple perspectives simultaneously, bridges them through category theory, and generates perspective-aware narratives with formal guarantees.

---

## 🧮 Core Transformation Capabilities

### 1. **Tomes → Tags** (Semantic Extraction)
```python
# Natural language → Structured knowledge
POST /ingest
{
  "text": "Alice met Bob at Stanford in 1995. She studied AI while he focused on databases.",
  "perspective": "academic"  # Optional: academic, personal, legal, etc.
}

# Returns: Crystallized facts
{
  "delta_axioms": [
    "alice||met||bob",
    "alice||location||stanford",
    "bob||location||stanford",
    "alice||study_focus||ai",
    "bob||study_focus||databases",
    "alice||met_bob_year||1995"
  ],
  "new_global_state": "89471829047120947812904...",  # Prime-encoded
  "perspective": "academic"
}
```

### 2. **Tags → Tomes** (Generation with measured faithfulness)
```python
# Structured knowledge → Natural narrative
POST /extrapolate
{
  "target_axioms": ["alice||met||bob", "alice||study_focus||ai"]
}

# Returns: a narrative rendered by LiveLLMAdapter.generate_text
{
  "narrative": "Alice met Bob; Alice's focus is on artificial intelligence...",
  "canonical_appendix": "The alice met bob.\nThe alice study_focus ai.",
  "state_integer": "89471829047120947812904..."
}

# For independent faithfulness measurement, run the bench harness:
#   python -m scripts.bench.run_bench --corpus scripts/bench/corpora/seed_v1.json
# → FActScore per corpus, currently 0.960 on seed_v1
# Note: per-request entailment gating is a roadmap item; today the LLM
# adapter generates faithful prose most of the time but is not hallucination-
# gated at the endpoint level.
```

### 3. **Round-Trip Conservation** (The Ouroboros Protocol)
```python
# Prove semantic conservation: Text → Tags → Text → Tags
POST /ouroboros/verify
{
  "text": "Original narrative about Alice and Bob..."
}

# Returns: Mathematical proof of conservation
{
  "round_trip_verified": true,
  "original_state": "894718290471...",
  "reconstructed_state": "894718290471...",  # Must be identical
  "information_preserved": 1.0,
  "canonical_tome": "Verified canonical reconstruction..."
}
```

---

## 🔄 Sliding-Scale Rendering (currently density-actionable; others LLM-gated)

The `TomeSliders` interface parameterizes rendering across five `[0.0, 1.0]` axes:

```python
from internal.ensemble.tome_sliders import TomeSliders
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator

sliders = TomeSliders(
    density=0.5,       # actioned on canonical path (deterministic axiom subsetting)
    length=0.8,        # LLM-gated (no-op without extrapolator)
    formality=0.3,     # LLM-gated
    audience=0.7,      # LLM-gated
    perspective=0.5,   # LLM-gated
)
tome = generator.generate_controlled(state, sliders)
# Output includes slider metadata in the header for reproducibility.
```

**What ships today:** the 5-axis type, validation, `requires_extrapolator()` gate, `header_line()` serialization, and the density axis actioned on the deterministic canonical path via lexicographic subsetting. 21 tests pin the contract. See `internal/ensemble/tome_sliders.py`.

**What is roadmap:** the other four axes (length, formality, audience, perspective) require an LLM extrapolator and are no-ops on the canonical path today — their values are captured in the output header as metadata so a future LLM-backed renderer can honour them. Per-perspective functorial bridges (category-theoretic mappings between perspective ontologies) are Phase 26 vision, not current capability.

---

## 🌌 Mathematical Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    SUM: Semantic Understanding Machine              │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────┐ │
│  │ Perspective │  │ Transform   │  │ Verification & Sync          │ │
│  │ Management  │  │ Engine      │  │ • Round-Trip Conservation    │ │
│  │ • Viewpoints│  │ • Tags→Tomes│  │ • Mathematical Proof         │ │
│  │ • Bridges   │  │ • Tomes→Tags│  │ • Decentralized P2P          │ │
│  │ • Evolution │  │ • Style Ctrl│  │ • Temporal Branching         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────────┬───────────────┘ │
│         │                │                        │                 │
│  ┌──────┴────────────────┴────────────────────────┴──────────────┐  │
│  │                 Gödel Semantic Algebra                        │  │
│  │  • Prime Encoding: Facts → Unique Primes                      │  │
│  │  • State Integer: Global_State = ∏(all active primes)         │  │
│  │  • Verification: State % Prime == 0 (fact exists)             │  │
│  │  • Sync Protocol: GCD(State_A, State_B) = exact delta         │  │
│  │  • Branching: Branch = Integer Copy (O(1) operation)          │  │
│  │  • Merging: LCM(Branch_A, Branch_B) = unified truth           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Akashic Ledger                              │ │
│  │     Event-sourced • Merkle Chain • Time Travel • Git-like      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Concrete Use Cases: From Personal to Planetary

| Transformation Pattern | Input | Output | Why SUM |
|----------------------|--------|---------|---------|
| **Personal Knowledge Assistant** | "I learned about quantum computing today..." | Structured facts + retrievable Q&A | **Verified answers** - no hallucination, mathematical proof of groundedness |
| **Multi-Perspective Research** | Academic papers on climate change | Structured facts per perspective (scientific, economic, political) | **Accountable pluralism** - same facts, different valid interpretations |
| **Collaborative Documentation** | Team meeting notes | Canonical knowledge base + personalized summaries | **Conflict resolution** - contradictions detected and mediated mathematically |
| **Cross-Cultural Knowledge** | Same events described by different cultures | Multiple valid narrative representations | **Perspective bridges** - formal mappings between worldviews |
| **Temporal Knowledge Evolution** | Historical document corpus | Time-indexed knowledge with evolution tracking | **Git for truth** - branch, merge, and time-travel through knowledge states |
| **Decentralized Truth Networks** | Global news and research papers | Synchronized planetary knowledge graph | **Zero-JSON sync** - one integer communicates exact knowledge delta |

---

## 🚀 Quick Start: From Zero to Semantic Algebra

### 1. Boot the Transformation Engine
```bash
git clone https://github.com/OtotaO/SUM.git
cd SUM
pip install -r requirements-prod.txt

# Verify mathematical correctness
python -m pytest Tests/ -v
python scripts/verify_fortress.py --json

# Launch with LLM integration
export OPENAI_API_KEY="sk-..."
uvicorn quantum_main:app --reload --port 8000
```

### 2. Try Basic Transformations
```bash
# Text → Facts
curl -X POST http://localhost:8000/api/v1/quantum/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice is 30 years old and works at Stanford."}'

# Facts → Text
curl -X POST http://localhost:8000/api/v1/quantum/extrapolate \
  -H "Content-Type: application/json" \
  -d '{"target_axioms": ["alice||age||30", "alice||works_at||stanford"]}'

# Round-trip verification
curl -X POST http://localhost:8000/api/v1/quantum/ouroboros/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice is 30 years old and works at Stanford."}'
```

### 3. Explore the Live Interface
Open `http://localhost:8000` to access:
- **Knowledge Graph Visualization** - See facts crystallize in real-time
- **Ask Bar** - Natural language queries with verified answers
- **Perspective Switcher** - Generate different narratives from same facts
- **Live Telemetry** - Watch the semantic algebra work
- **WASM Offline Mode** - Local knowledge processing without servers

---

## 📡 API Reference: The Transformation Endpoints

### Core Transformations
| Method | Endpoint | Transform | Description |
|--------|----------|-----------|-------------|
| `POST` | `/ingest` | Tomes → Tags | Extract structured facts from natural language |
| `POST` | `/ingest/math` | Direct → Tags | Insert structured triplets directly (zero LLM cost) |
| `POST` | `/extrapolate` | Tags → Tomes | Generate verified narratives from facts |
| `POST` | `/rehydrate` | State → Tome | Convert entire knowledge state to readable form |
| `POST` | `/ask` | Query → Verified Answer | Natural language Q&A with proof |

### Verification & Integrity
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/ouroboros/verify` | Prove round-trip conservation |
| `POST` | `/zk/prove` | Generate zero-knowledge semantic proofs |
| `GET` | `/provenance/{axiom}` | Full provenance chain for any fact |

### Perspective & Collaboration
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/branch` | Create new perspective (fork knowledge) |
| `POST` | `/merge` | Combine perspectives (LCM operation) |
| `POST` | `/time-travel` | Rebuild knowledge at historical point |
| `POST` | `/sync/state` | Decentralized knowledge synchronization |

---

## 🌅 Future Horizons (honestly-scoped roadmap)

These are **roadmap items**, not current capabilities. Each is a concrete piece of work with a defined entry in `docs/PROOF_BOUNDARY.md` §3. They are listed in approximate order of prerequisite dependence.

### Near-term (next 1–2 milestones)
- **LLM wiring for the 4 remaining sliders** — length / formality / audience / perspective. Requires attaching an extrapolator (e.g. `LiveLLMAdapter` via `AutoregressiveTomeGenerator.extrapolator`). Interface is already shipped; what's missing is the prompt-conditioning layer that honours each axis.
- **Per-doc logging in the regeneration runner** — surface which specific claims fail entailment (closes the 4% gap currently aggregated in the 0.960 FActScore).
- **LLM narrative full round-trip runner** — composes existing LLM generator + LLM re-extractor + drift metric. Measures real prose conservation end-to-end.
- **Calibration fixture authoring for Venn-Abers** — turns zero-width confidence intervals into meaningful bounds. Needs a labelled (score, was_correct) set.

### Medium-term (Polytaxis Bucket A completion)
- **SHACL structural validation** via pySHACL — W3C-standard replacement for the hand-rolled `ExtractionValidator`.
- **W3C Verifiable Credentials 2.0** emission with `eddsa-jcs-2022` Data Integrity proofs — makes SUM bundles consumable by any VC-compliant ecosystem.
- **RFC 3161 timestamping anchor** — external witness on the Merkle chain.
- **RFC 9162 CT v2 inclusion proofs** — third-party verifiability of the audit log.
- **Full polyglot emission** — Turtle and RDF/XML beyond the JSON-LD already shipped for PROV-O.

### Long-term (aspirational, requires user-pull to prioritise)
- **Category-theoretic perspective bridges** — functorial mappings between classification perspectives (Polytaxis §1). Currently no crisp use case in SUM; surface it when multi-perspective users ask.
- **Lean 4 meta-theorems** — machine-checked proofs of algebra invariants. Currently unit-tested; Lean would upgrade to `certified` epistemic status.
- **Property-graph primary store** (TerminusDB or Oxigraph) with Gödel integer demoted to attestation witness. Justified by measured merge perf (~O(n²) confirmed) above ~10k axioms.
- **Phase 19B adversarial corpus integration** into the bench harness (currently Phase 19B is separately maintained).

Items *not* on this roadmap that earlier drafts suggested: perspective-as-functor as a core mechanic (it's a classification abstraction, wrong category for the distillation product); zero-knowledge entailment proofs via Halo2/Plonky2 (too early; added only when a specific user needs it); multi-formal-method specification stack (Alloy + TLA+ + Lean 4 — three formalisms is more than a small team can maintain).

---

## 🌐 Cloudflare Deployment Architecture

### Static Edge Deployment
```yaml
# wrangler.toml
name = "sum-semantic-engine"
main = "dist/worker.js"
compatibility_date = "2024-01-01"

[env.production]
vars = { ENVIRONMENT = "production" }
kv_namespaces = [
  { binding = "SUM_KNOWLEDGE", id = "knowledge_store" }
]
```

The **static interface** (quantum.html) deploys to Cloudflare Pages, with:
- **WASM Module** (`sum_core.wasm`) - Offline semantic algebra
- **Knowledge Sync** - WebRTC P2P + Cloudflare KV for state caching
- **Global Edge** - Sub-100ms latency worldwide via Cloudflare's network
- **Serverless Backend** - FastAPI → Cloudflare Workers via serverless functions

### Hybrid Architecture Benefits
- **Offline-first** - WASM enables local knowledge processing
- **Global sync** - Cloudflare KV provides planetary knowledge state
- **Zero latency** - Edge computing for instant semantic queries
- **Cost efficiency** - Only pay for compute used, scale to zero

---

## 🛡️ Verification: 756+ Test Suite

The test suite covers both proven invariants and empirically-measured properties; each assertion is scoped to the epistemic status of the thing it tests.

```text
Provable (deterministic code + tests that enforce the proof):
  ✓ Canonical Round-Trip Conservation — 0.00 % drift (Ouroboros §1.1)
  ✓ Algebra Invariants — LCM commutativity / associativity, merge idempotency,
    entailment correctness, delta correctness, deletion correctness
  ✓ Akashic Ledger Durability — event-sourced replay, branch isolation
  ✓ Merkle Hash-Chain Integrity — SHA-256 chain (Phase 19C)
  ✓ Cross-Runtime State Equivalence — Python ↔ Node.js witness on the
    non-colliding derivation path

Empirically measured (reported by the bench harness):
  ✓ Extraction F1 on seed_v1 — 1.000 on 50 SVO docs
  ✓ Regeneration FActScore — 0.960 (LLM narrative + entailment checker)
  ✓ Operation performance — p50 / p99 at N ∈ {100, 500, 1000} axioms
  ✓ Sieve re-extract of canonical — 54 % drift (known ceiling)

Cryptographic integrity:
  ✓ HMAC-SHA256 signatures + Ed25519 key rotation
  ✓ Bundle tamper detection
  ✓ Adversarial bundle handling

Interop (Polytaxis Bucket A absorption):
  ✓ Epistemic Status Taxonomy — {provable, certified, empirical-benchmark,
    expert-opinion} on every metric
  ✓ Venn-Abers Conformal Intervals — distribution-free confidence bounds
  ✓ PROV-O JSON-LD Emission — Akashic Ledger events → W3C PROV-O
  ✓ TomeSliders — 5-axis slider interface (density actioned)
```

**Every claim carries an explicit epistemic status.** The canonical round-trip is mathematically proven; the broader text→structure→text pipeline is empirically measured and reported honestly. See [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) for the separation of proved from measured and the list of what's still aspirational.

---

## 🎨 Interface Vision: Beyond CRUD to Semantic Flow

### The Transformation-Centric UI
```
┌─────────────────────────────────────────────────────┐
│  ┌─────────────┐    Transform    ┌─────────────────┐ │
│  │    TOMES    │ ← ← ← ← ← ← ← ← │      TAGS       │ │
│  │  Narrative  │                 │   Structured    │ │
│  │   Content   │ → → → → → → → → │     Facts       │ │
│  │             │                 │                 │ │
│  │ • Stories   │     Verified    │ • Triplets      │ │
│  │ • Reports   │   Bidirectional │ • Relations     │ │
│  │ • Essays    │  Transformation │ • Properties    │ │
│  │ • Docs      │                 │ • Assertions    │ │
│  └─────────────┘                 └─────────────────┘ │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │            PERSPECTIVE SELECTOR                 │ │
│  │  [Academic] [Personal] [Legal] [Cultural]       │ │
│  │  [Scientific] [Historical] [+Custom]            │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │              STYLE CONTROLS                     │ │
│  │  Detail: [●────────] Granularity               │ │
│  │  Tone:   [Formal] [Casual] [Technical]         │ │
│  │  Focus:  [Overview] [Deep Dive] [Summary]      │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Key Interface Innovations
1. **Bi-directional transformation as primary interaction**
2. **Live perspective switching** - same content, different viewpoints
3. **Mathematical verification indicators** - visual proof of groundedness
4. **Collaborative conflict resolution** - merge perspectives gracefully
5. **Temporal sliders** - explore knowledge evolution over time

---

## 🤝 Contributing to the Semantic Revolution

1. **Fork & Branch** - Create your perspective branch
2. **Test Mathematically** - `python -m pytest Tests/ -v`
3. **Verify Fortress** - `python scripts/verify_fortress.py --json`
4. **Transform & Submit** - Your PR becomes part of the verified knowledge base

**Join us in building the first mathematically rigorous knowledge transformation engine.**

---

## 📜 License & Philosophy

Apache 2.0 — **Built for the future of human knowledge.**

*"The best way to predict the future is to invent it. The best way to preserve knowledge is to make it mathematically eternal."*

---

<p align="center">
<strong>SUM — From Tags to Tomes and Back Again.</strong><br>
<em>Canonical round-trip mathematically proven. Full pipeline continuously measured. Every claim labelled.</em>
</p>