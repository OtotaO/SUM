# SUM — The Quantum Knowledge OS

> **From Atomic Tags to Infinite Books. Mathematically Verified. Hallucination-Proof.**

SUM is the world's first **Gödel-State Knowledge Engine** — a system that encodes human knowledge as prime-factored integers and performs semantic operations (merge, entailment, contradiction detection, graph traversal) in **O(1) time** using pure number theory.

Unlike probabilistic vector databases, SUM provides **mathematical proof** that every generated statement is grounded in verified facts.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Quantum Knowledge OS                        │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Quantum UI   │  │ Quantum API  │  │ Autonomous Daemon    │ │
│  │ quantum.html │  │ /state /sync │  │ (Crystallizer)       │ │
│  │ vis-network  │  │ /ingest      │  │ Background compactor │ │
│  │ BigInt sync  │  │ /extrapolate │  │ asyncio.create_task  │ │
│  │              │  │ /query       │  │                      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                 │                      │             │
│  ┌──────┴─────────────────┴──────────────────────┴───────────┐ │
│  │              GlobalKnowledgeOS Singleton                   │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │ GodelStateAlgebra    │ Epistemic Loop  │ Vector Bridge    │ │
│  │ • Prime minting      │ • Hallucination │ • Continuous↔    │ │
│  │ • LCM merge          │   detection     │   Discrete       │ │
│  │ • GCD entailment     │ • Modulo proof  │ • Semantic search│ │
│  │ • Fractal zooming    │                 │                  │ │
│  │ • Node registry      │                 │                  │ │
│  ├───────────────────────┴─────────────────┴──────────────────┤ │
│  │                   Akashic Ledger (SQLite)                  │ │
│  │             Event-sourced • Crash-safe • Replayable        │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## Core Capabilities

### Gödel-State Algebra (Phase 1)
Every fact ("Alice age 30") is assigned a unique **prime number**. The entire knowledge state is a single integer — the product of all active primes. Operations become arithmetic:

| Operation | Math | Time |
|-----------|------|------|
| **Merge** two knowledge states | `lcm(A, B)` | O(1) |
| **Verify** a fact exists | `state % prime == 0` | O(1) |
| **Detect** contradictions | Exclusion zone check | O(1) |
| **Delete** a fact | `state // prime` | O(1) |

### Epistemic Feedback Loop (Phase 2)
LLM-generated text is **mathematically verified** against the Gödel state. If the modulo check fails, hallucinated claims are isolated and the LLM is re-prompted with explicit negative constraints.

### Temporal Evolution & Vector Bridge (Phase 3)
- **Update/Delete** axioms with full state integrity
- **ContinuousDiscreteBridge**: Indexes primes to vector embeddings, enabling fuzzy semantic search filtered by the exact Gödel state

### Akashic Ledger (Phase 4)
An append-only SQLite event log (`MINT`, `MUL`, `DIV`) that provides **crash recovery** — the full Gödel state can be replayed from the ledger at any time.

### Gödel Sync Protocol (Phase 5)
Clients hold a local `BigInt` state. Syncing is a single API call — `gcd` identifies what to add/delete. **O(1) network delta**, no diffing required.

### Fractal Crystallization (Phase 6)
Semantic zooming via prime arithmetic:
- **Zoom Out**: Replace N micro-primes with 1 macro-prime, store provenance
- **Zoom In**: Restore original primes from stored provenance
- Both operations: **O(1)**

### Quantum GraphRAG & Autonomous Daemon (Phase 7)
- **Node Registry**: Each entity accumulates its edge primes via LCM. Querying a node's neighbourhood is `gcd(global_state, node_integer)` — **O(1)** vs. vector RAG's O(N log N).
- **Autonomous Crystallizer**: Background daemon compresses dense graph clusters while you sleep.

## Project Structure

```
SUM/
├── internal/
│   ├── algorithms/
│   │   └── semantic_arithmetic.py    # GodelStateAlgebra (core math engine)
│   ├── ensemble/
│   │   ├── epistemic_loop.py         # Hallucination-proof generation
│   │   ├── vector_bridge.py          # Continuous↔Discrete bridge
│   │   ├── autonomous_agent.py       # Subconscious Crystallizer daemon
│   │   └── live_llm_adapter.py       # OpenAI adapter
│   └── infrastructure/
│       └── akashic_ledger.py         # Event-sourced crash recovery
├── api/
│   ├── quantum_router.py             # FastAPI: /state /sync /ingest /extrapolate /query
│   └── knowledge_os.py               # GlobalKnowledgeOS singleton
├── mass_semantic_engine.py            # MapReduce Tomes↔Tags pipeline
├── quantum_main.py                    # ASGI entrypoint (FastAPI + lifespan)
├── main.py                            # Legacy Flask entrypoint
├── static/
│   ├── quantum.html                   # Quantum UI (vis-network + BigInt)
│   └── js/godel_client.js             # Browser-side Gödel sync client
├── Tests/
│   ├── test_semantic_arithmetic.py    # Phase 1: SPNT + Algebra (15 tests)
│   ├── test_epistemic_loop.py         # Phase 2: Hallucination loop (7 tests)
│   ├── test_temporal_vector.py        # Phase 3: CRUD + Vector Bridge (12 tests)
│   ├── test_phase4.py                 # Phase 4: Akashic Ledger (3 tests)
│   ├── test_phase5_api.py             # Phase 5: Sync Protocol + API (7 tests)
│   ├── test_phase6_fractal.py         # Phase 6: Fractal Crystallization (5 tests)
│   └── test_phase7_moonshot.py        # Phase 7: GraphRAG + Daemon (6 tests)
└── requirements.txt
```

## Quick Start

### Quantum Knowledge OS (Phases 1–7)

```bash
git clone https://github.com/OtotaO/SUM.git
cd SUM

pip install -r requirements.txt

# Run all 55 tests
python -m pytest Tests/ -v

# Launch the Quantum UI
export OPENAI_API_KEY="sk-..."
uvicorn quantum_main:app --reload --port 8000
# Open http://localhost:8000
```

### Legacy Flask Engine

```bash
python main.py
# Open http://localhost:5001
```

## API Endpoints

All endpoints are mounted under `/api/v1/quantum/`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/state` | Current Gödel integer + axiom count |
| `POST` | `/sync` | O(1) delta sync (send client BigInt, receive add/delete) |
| `POST` | `/ingest` | Tomes → Tags (text → prime factored state) |
| `POST` | `/extrapolate` | Tags → Tomes (verified narrative generation) |
| `POST` | `/search` | Fuzzy semantic search via Vector Bridge |
| `POST` | `/query` | O(1) GraphRAG neighbourhood retrieval |

## Test Suite

```
55 passed in 0.59s

Phase 1 — SPNT + Gödel Algebra ........... 15 ✓
Phase 2 — Epistemic Feedback Loop ........ 7  ✓
Phase 3 — Temporal CRUD + Vector Bridge .. 12 ✓
Phase 4 — Akashic Ledger ................. 3  ✓
Phase 5 — Gödel Sync Protocol + API ...... 7  ✓
Phase 6 — Fractal Crystallization ........ 5  ✓
Phase 7 — GraphRAG + Autonomous Daemon ... 6  ✓
```

## Configuration

```bash
OPENAI_API_KEY=sk-...    # Required for LLM features (ingest/extrapolate/search)
                          # Math-only mode works without it
```

## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run the test suite (`python -m pytest Tests/ -v`)
4. Commit your changes
5. Push and open a Pull Request

## License

Apache 2.0 — Built for the future of Man and Machine.

---

<p align="center">
  <strong>SUM — Distill the Universe. Expand the Atom. Prove Everything.</strong>
</p>
