# SUM — The Quantum Knowledge OS

[![Quantum Knowledge OS CI](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml/badge.svg)](https://github.com/OtotaO/SUM/actions/workflows/quantum-ci.yml)

> **From Atomic Tags to Structured Books. Mathematically Verified within the Canonical Semantic Boundary.**

SUM is a **Gödel-State Knowledge Engine** — a system that encodes semantic content as prime-factored integers and performs semantic operations (merge, entailment, paradox detection, graph traversal) using pure number theory, avoiding corpus-scale scanning.

Built on the formalisms of the **Semantic Prime Number Theorem** and **Gauge-Theoretic CRUD**, SUM replaces probabilistic vector databases with deterministic integer arithmetic for core knowledge operations. It provides **mechanically verifiable round-trip conservation** within the canonical semantic representation, cross-runtime state verification, signed knowledge transport, and decentralized P2P syncing.

> **Complexity note:** operations described below as O(1) are O(1) **in axiom count** — they do not scan the corpus. They scale with integer **bit length** (sub-quadratic via GMP). See [`docs/PROOF_BOUNDARY.md`](docs/PROOF_BOUNDARY.md) for precise complexity analysis.

---

## 🌌 Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      Quantum Knowledge OS (SUM)                        │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ Quantum UI   │  │ Quantum API  │  │ Autonomous Daemons           │  │
│  │ quantum.html │  │ /state /sync │  │ • Subconscious Crystallizer  │  │
│  │ vis-network  │  │ /ingest /zk  │  │ • P2P Holographic Mesh       │  │
│  │ Telemetry HUD│  │ /branch      │  │ • Epistemic Arbiter          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬───────────────┘  │
│         │                 │                         │                  │
│  ┌──────┴─────────────────┴─────────────────────────┴───────────────┐  │
│  │                   GlobalKnowledgeOS Singleton                    │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ GodelStateAlgebra       │ Epistemic Engine    │ Multiverse       │  │
│  │ • SHA-256 Primes        │ • Paradox Collapse  │ • Branching      │  │
│  │ • LCM Merge / GCD Sync  │ • ZK Semantic Proof │ • Merging        │  │
│  │ • GraphRAG              │ • Causal Cascades   │ • Time Travel    │  │
│  ├─────────────────────────┴─────────────────────┴──────────────────┤  │
│  │                      Akashic Ledger (SQLite)                     │  │
│  │         Event-sourced • Crash-safe • Historically Replayable     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🧮 The 10-Phase Capabilities

### 1. Gödel-State Algebra & SPNT

Every irreducible fact ("Alice age 30") is assigned a unique, deterministic **prime number** via SHA-256 hashing. The entire knowledge state of the universe is a single massive integer — the product of all active primes.

| Semantic Operation | Mathematical Equivalent | Complexity† |
|--------------------|-------------------------|-------------|
| **Branch / Fork** | Integer Copy: `Branch_B = Branch_A` | O(n) copy |
| **Merge States** | Least Common Multiple: `math.lcm(A, B)` | O(n²) GCD |
| **Verify a Fact** | Modulo Entailment: `Global_State % Prime == 0` | O(n) modulo |
| **Delete a Fact** | Integer Division: `Global_State // Prime` | O(n) division |
| **Update a Fact** | `lcm(State // Old_Prime, New_Prime)`| O(n²) GCD |
| **GraphRAG Traversal** | `math.gcd(Global, Node_Integer)` | O(n²) GCD |

†n = bit length of the integer, NOT axiom count. No corpus scanning required.

### 2. The Epistemic Feedback Loop

LLM-generated text is **mathematically verified** against the Gödel state. If the modulo check fails, `math.gcd` isolates the exact hallucinated primes. The LLM is forced into a self-correcting loop with strict negative constraints until mathematical fidelity is achieved.

### 3. Temporal Evolution & The Vector Bridge

Maps the absolute certainty of discrete Gödel Primes to the fuzzy continuous space of Vector Embeddings. Semantic search (`cosine_similarity`) is mathematically filtered by `State % Prime == 0`, meaning deleted facts vanish instantly from search results.

### 4. The Akashic Ledger (Fidelity Persistence)

An append-only SQLite event log (`MINT`, `MUL`, `DIV`). Provides **crash recovery** — the RAM-based Gödel BigInt can be perfectly reconstructed by replaying the mathematical trace.

### 5. The Gödel Sync Protocol

Clients hold a local `BigInt` state. Syncing over the network requires sending *one integer*. The server uses `math.gcd` to identify exactly what the client needs to add or delete. **Single-integer network delta**, zero JSON diffing.

### 6. Fractal Crystallization (Semantic Zooming)

  - **Zoom Out**: Replace a cluster of 100 micro-primes with 1 Macro-Prime, storing the cluster product as provenance.
  - **Zoom In**: Divide out the Macro-Prime and multiply the provenance back in to decompress the cluster (single division + multiplication).

### 7. Quantum GraphRAG & The Subconscious Daemon

Standard RAG requires O(N) vector scans. Quantum GraphRAG maintains a "Node Integer" for every entity. Querying a node's multi-hop topological neighborhood is `math.gcd(Global_State, Node_Integer)` — **no corpus scan required**. A background daemon autonomously crystallizes dense clusters while the system sleeps.

### 8. Epistemic Superposition & Wave Function Collapse

When mutually exclusive facts are ingested ("Alice lives in NY" vs "London"), the system detects Level 3 Curvature (a topological paradox) and enters Superposition. An LLM Arbiter acts as a logical judge, mathematically collapsing the wave function to a single verified truth, streamed live via Server-Sent Events (SSE).

### 9. The Multiverse of Meaning (Git for Truth)

Because the state is just an integer, branching a timeline is a simple integer copy. We support **Semantic Smart Contracts** (Causal Triggers): learning one fact automatically cascades through the integer, deducing logical consequences until equilibrium is reached.

### 10. The Chronos Engine & Holographic Mesh

  - **Time Travel**: Rebuild the Akashic Ledger to any historical tick into a parallel branch.
  - **ZK Semantic Proofs**: Prove you know a fact using salted SHA-256 hashes of the quotient, without revealing your state integer.
  - **Decentralized P2P**: A background gossip daemon constantly syncs Gödel Integers with peer nodes, achieving a planetary hive-mind.

---

## 🌅 Future Horizons

SUM has completed its genesis phase. The road ahead expands into three frontiers:

1.  **Horizon I: Project EXOCORTEX (Continuous Human Digitization):** Wiring desktop and browser companion clients to stream daily human life directly into the `/ingest` API, compressing personal reality into a single, ever-growing Gödel Integer.
2.  **Horizon II: The Babel Protocol (Planetary-Scale Truth):** Deploying thousands of headless `SUM` nodes to ingest Wikipedia, ArXiv, and global news—forcing the Epistemic Arbiter to collapse historical and scientific paradoxes into a Single Master Integer (Planetary Truth).
3.  **Horizon III: Shor's Horizon (Quantum Hardware Execution):** Porting the core `GodelStateAlgebra` to GMP (C++/Rust) and eventually executing the Semantic Sieve natively on Physical Quantum Computers to achieve instantaneous semantic routing using Shor's Algorithm.

---

## 🚀 Quick Start

### 1. Install & Boot the Knowledge OS

```bash
git clone https://github.com/OtotaO/SUM.git
cd SUM
pip install -r requirements-prod.txt

# Run the 231+ test verification suite
python -m pytest Tests/ -v

# Launch the Quantum UI & OS
export OPENAI_API_KEY="sk-..."
uvicorn quantum_main:app --reload --port 8000
```

Open `http://localhost:8000` to access the **Dashboard of Truth** (Vis-Network Graph, BigInt State Mirror, Live Telemetry).

---

## 📡 Quantum API Endpoints

All endpoints are mounted under `/api/v1/quantum/`. Every endpoint accepts an optional `?branch=main` parameter.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/state` | Returns the current Gödel integer |
| `POST` | `/sync` | Network delta sync via GCD |
| `POST` | `/ingest` | Tomes → Tags (Text to Math pipeline) |
| `POST` | `/extrapolate`| Tags → Tomes (Verified generation) |
| `POST` | `/query` | GraphRAG retrieval |
| `POST` | `/search` | Fuzzy semantic search via Vector Bridge |
| `GET` | `/telemetry` | SSE Stream of internal monologues & wave collapses |
| `POST` | `/branch` | Semantic Branching (fork state) |
| `POST` | `/merge` | Semantic Merge via LCM |
| `POST` | `/time-travel`| Rebuilds the universe at a historical tick |
| `POST` | `/zk/prove` | Generate a Zero-Knowledge Semantic Proof |
| `POST` | `/peers` | Add a node to the P2P Holographic Mesh |

---

## 📂 Project Structure

```text
SUM/
├── internal/
│   ├── algorithms/
│   │   ├── semantic_arithmetic.py    # GodelStateAlgebra, SPNT, Fractal Zoom
│   │   └── zk_semantics.py           # Zero-Knowledge Entailment Proofs
│   ├── ensemble/
│   │   ├── epistemic_loop.py         # Hallucination-proof generation
│   │   ├── vector_bridge.py          # Continuous↔Discrete bridge
│   │   ├── autonomous_agent.py       # Subconscious Crystallizer daemon
│   │   ├── epistemic_arbiter.py      # Wave Function Collapse & SSE
│   │   ├── gauge_orchestrator.py     # L1/L2/L3 Commutativity Hierarchy
│   │   ├── causal_triggers.py        # Semantic Smart Contracts
│   │   └── live_llm_adapter.py       # OpenAI structured outputs adapter
│   └── infrastructure/
│       ├── akashic_ledger.py         # Event-sourced crash recovery & Time Travel
│       ├── p2p_mesh.py               # Decentralized Gossip Protocol
│       ├── canonical_codec.py        # Signed bundle transport (HMAC + Ed25519)
│       └── key_manager.py             # Ed25519 keypair lifecycle management
├── api/
│   └── quantum_router.py             # FastAPI routing and GlobalKnowledgeOS
├── mass_semantic_engine.py           # MapReduce Tomes↔Tags pipeline
├── quantum_main.py                   # ASGI entrypoint (FastAPI + lifespan boot)
├── standalone_verifier/
│   └── verify.js                     # Independent Node.js semantic witness
├── docs/
│   ├── CANONICAL_ABI_SPEC.md         # Normative protocol specification
│   ├── PROOF_BOUNDARY.md             # What is proven vs aspirational
│   ├── THREAT_MODEL.md               # Security analysis and attack surfaces
│   └── COMPATIBILITY_POLICY.md       # Version semantics and guarantees
├── static/
│   ├── quantum.html                  # Quantum UI (vis-network, Telemetry HUD)
│   └── js/godel_client.js            # Browser-side Gödel sync client
└── Tests/
    ├── fixtures/                      # Frozen golden reference vectors
    └── test_phase[1-17]_*.py         # 226+ Passing Verification Tests
```

---

## 🛡️ Mathematical Verification Suite

```text
226+ passed

Ed25519 Attestation ................... 11 ✓  (dual-sig, tamper, compat, key mgmt)
Witness Matrix Hardening .............. 7  ✓  (frozen vectors, cross-runtime)
Property & Adversarial Tests .......... 43 ✓  (algebra invariants, bundle hardening)
Phase 16 — Independent Witness ........ 21 ✓  (cross-runtime verification)
Phase 15 — Canonical Semantic ABI ..... 22 ✓  (versioning, bundles, JWT, multi-hop)
Phase 14 — Ouroboros Round-Trip ....... 16 ✓  (encode/decode conservation)
Phase 13 — JWT Multi-Tenancy .......... 3  ✓
Phase 10 — Chronos & Mesh ............. 17 ✓
Phase 9 — Multiverse & Causal Triggers  21 ✓
Phase 8 — Wave Function Collapse ...... 8  ✓
Phase 7 — GraphRAG + Daemon ........... 6  ✓
Phase 6 — Fractal Crystallization ..... 5  ✓
Phase 5 — Gödel Sync Protocol + API ... 7  ✓
Phase 4 — Akashic Ledger .............. 3  ✓
Phase 3 — Temporal CRUD + Vectors ..... 12 ✓
Phase 2 — Epistemic Feedback Loop ..... 7  ✓
Phase 1 — SPNT + Gödel Algebra ........ 15 ✓
```

---

## ⚙️ Configuration

```bash
OPENAI_API_KEY=sk-...    # Required for LLM features (ingest/extrapolate/search/arbitrate)
                         # Math-only CRUD, GraphRAG, and P2P Sync work without it
```

## 🤝 Contributing

1.  Fork it
2.  Create your feature branch (`git checkout -b feature/holographic-expansion`)
3.  Run the test suite (`python -m pytest Tests/ -v`)
4.  Commit your changes
5.  Push and open a Pull Request

## 📜 License

Apache 2.0 — Built for the future of Man and Machine.

---

<p align="center">
<strong>SUM — Canonical Semantic Compression. Mechanically Verified.</strong>
</p>
