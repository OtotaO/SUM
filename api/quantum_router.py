"""
Quantum Router — The Global Knowledge OS API

Exposes the Gödel-State Engine to the outside world via FastAPI:
    /state           – current global integer
    /sync            – O(1) delta synchronisation (Gödel Sync Protocol)
    /search          – semantic search over alive primes
    /ingest          – ingest text into the global state via live LLM
    /extrapolate     – hallucination-proof narrative generation

The ``GlobalKnowledgeOS`` singleton boots from the Akashic Ledger on
startup, rebuilding the exact Gödel BigInt from the SQLite trace.

Author: ototao
License: Apache License 2.0
"""

import logging
import math
from typing import List

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.vector_bridge import ContinuousDiscreteBridge
from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.ensemble.epistemic_loop import QuantumExtrapolator
from mass_semantic_engine import MassSemanticEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/quantum", tags=["Gödel State"])


# ─── Global Knowledge OS (Singleton) ─────────────────────────────────

class GlobalKnowledgeOS:
    """
    Central nervous system connecting Math, DB, and LLMs in RAM.

    Boots from the Akashic Ledger on startup so the Gödel state survives
    server restarts.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.algebra = GodelStateAlgebra()
            cls._instance.ledger = AkashicLedger("production_akashic.db")
            cls._instance.global_state = 1
            cls._instance.is_booted = False
        return cls._instance

    async def boot_sequence(self, llm_adapter=None):
        """
        Replay the Akashic Ledger to restore the full Gödel state, then
        wire up the LLM adapter for live extraction and search.
        """
        if self.is_booted:
            return

        # Crash recovery
        self.global_state = await self.ledger.rebuild_state(self.algebra)
        logger.info(
            "KOS boot complete — state bit-length=%d, axioms=%d",
            self.global_state.bit_length(),
            len(self.algebra.prime_to_axiom),
        )

        # Wire live LLM (optional — tests can boot without it)
        if llm_adapter is not None:
            self.mass_engine = MassSemanticEngine(llm_adapter.extract_triplets)
            self.mass_engine.algebra = self.algebra

            self.vector_bridge = ContinuousDiscreteBridge(
                self.algebra, llm_adapter.get_embedding
            )
            await self.vector_bridge.index_new_primes()

            self.extrapolator = QuantumExtrapolator(
                self.algebra,
                llm_adapter.generate_text,
                llm_adapter.extract_triplets,
            )

        self.is_booted = True


# Singleton instance used by route handlers
kos = GlobalKnowledgeOS()


# ─── Request / Response models ────────────────────────────────────────

class SyncRequest(BaseModel):
    """Client sends its Gödel integer as a string (BigInts exceed JS limits)."""
    client_state_integer: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestRequest(BaseModel):
    text: str


class ExtrapolateRequest(BaseModel):
    target_axioms: List[str]


# ─── Routes ──────────────────────────────────────────────────────────

@router.get("/state")
async def get_global_state():
    """Returns the current global Gödel integer and axiom count."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    return {
        "global_state_integer": str(kos.global_state),
        "axiom_count": len(kos.algebra.prime_to_axiom),
    }


@router.post("/sync")
async def sync_client_state(request: SyncRequest):
    """
    O(1) Delta Sync (Gödel Sync Protocol).

    Returns exactly which axioms the client needs to add and delete
    to match the server's state.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    client_state = int(request.client_state_integer)
    delta = kos.algebra.calculate_network_delta(
        kos.global_state, client_state
    )

    return {
        "new_global_state": str(kos.global_state),
        "delta": delta,
    }


@router.post("/search")
async def semantic_search(request: SearchRequest):
    """Fuzzy query against the exact Gödel state via the Vector Bridge."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    if not hasattr(kos, "vector_bridge"):
        raise HTTPException(
            status_code=503,
            detail="Vector bridge not initialised (no LLM adapter)",
        )

    results = await kos.vector_bridge.semantic_search_godel_state(
        kos.global_state, request.query, request.top_k
    )
    return {
        "verified_axioms": [
            {"axiom": axiom, "similarity": round(float(sim), 4)}
            for axiom, sim in results
        ]
    }


@router.post("/ingest")
async def ingest_document(
    request: IngestRequest, background_tasks: BackgroundTasks
):
    """
    Tomes → Tags. Ingests raw text into the Gödel state via MapReduce
    extraction, mathematical merge, and Akashic trace logging.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    if not hasattr(kos, "mass_engine"):
        raise HTTPException(
            status_code=503,
            detail="Mass engine not initialised (no LLM adapter)",
        )

    # 1. Chunk the text
    chunk_size = 2000
    chunks = [
        request.text[i : i + chunk_size]
        for i in range(0, len(request.text), chunk_size)
    ]
    raw_claims_estimate = max(len(request.text) // 100, 1)

    # 2. MapReduce extraction
    result = await kos.mass_engine.tomes_to_tags(raw_claims_estimate, chunks)
    new_state = result["global_state"]

    # 3. Mathematical merge
    kos.global_state = math.lcm(kos.global_state, new_state)

    # 4. Akashic trace — log new primes
    for prime, axiom in kos.algebra.prime_to_axiom.items():
        if new_state % prime == 0:
            await kos.ledger.append_event("MINT", prime, axiom)
            await kos.ledger.append_event("MUL", prime)

    # 5. Background vector indexing
    if hasattr(kos, "vector_bridge"):
        background_tasks.add_task(kos.vector_bridge.index_new_primes)

    return {
        "status": "success",
        "new_global_state": str(kos.global_state),
        "axioms_ingested": result["total_unique_primes"],
        "paradoxes": result["paradoxes"],
    }


@router.post("/extrapolate")
async def extrapolate_tome(request: ExtrapolateRequest):
    """
    Tags → Tomes and Back. Generates a hallucination-proof narrative
    strictly from verified axioms in the global state.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    if not hasattr(kos, "extrapolator"):
        raise HTTPException(
            status_code=503,
            detail="Extrapolator not initialised (no LLM adapter)",
        )

    try:
        narrative = await kos.extrapolator.extrapolate_with_proof(
            kos.global_state, request.target_axioms
        )
        return {"verified_narrative": narrative}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Quantum GraphRAG ────────────────────────────────────────────────

class QuantumQueryRequest(BaseModel):
    nodes: List[str]
    hops: int = 1


@router.post("/query")
async def quantum_graph_rag(request: QuantumQueryRequest):
    """
    O(1) Topological GraphRAG Context Retrieval.

    Returns the exact axioms in the mathematical neighbourhood of the
    requested nodes, filtered to only include alive axioms.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    context_integer = kos.algebra.get_quantum_neighborhood(
        kos.global_state, request.nodes, request.hops
    )

    # Extract axioms from the context integer
    context_axioms: list[str] = []
    temp_int = context_integer
    for prime, axiom in kos.algebra.prime_to_axiom.items():
        if temp_int % prime == 0:
            context_axioms.append(axiom)
            while temp_int % prime == 0:
                temp_int //= prime
        if temp_int == 1:
            break

    return {
        "context_integer": str(context_integer),
        "axioms": context_axioms,
    }
