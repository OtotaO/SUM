"""
Quantum Router — The Global Knowledge OS API

Exposes the Gödel-State Engine to the outside world via FastAPI:
    /state           – current global integer (branch-aware)
    /sync            – O(1) delta synchronisation (Gödel Sync Protocol)
    /search          – semantic search over alive primes
    /ingest          – ingest text into the global state via live LLM
    /extrapolate     – hallucination-proof narrative generation
    /query           – O(1) GraphRAG neighbourhood retrieval
    /branch          – O(1) epistemic branching (Git for Truth)
    /merge           – O(1) LCM-based branch merging

The ``GlobalKnowledgeOS`` singleton boots from the Akashic Ledger on
startup, rebuilding the exact Gödel BigInt from the SQLite trace.

Author: ototao
License: Apache License 2.0
"""

import logging
import math
import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.vector_bridge import ContinuousDiscreteBridge
from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.ensemble.epistemic_loop import QuantumExtrapolator
from internal.ensemble.causal_triggers import CausalTriggerMap
from mass_semantic_engine import MassSemanticEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/quantum", tags=["Gödel State"])


# ─── Global Knowledge OS (Singleton) ─────────────────────────────────

class GlobalKnowledgeOS:
    """
    Central nervous system connecting Math, DB, and LLMs in RAM.

    Boots from the Akashic Ledger on startup so the Gödel state survives
    server restarts.  Now supports multiple branch integers for
    Zero-Cost Semantic Branching (Git for Truth).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.algebra = GodelStateAlgebra()
            cls._instance.ledger = AkashicLedger("production_akashic.db")
            cls._instance.branches = {"main": 1}  # Multiverse State
            cls._instance.is_booted = False
        return cls._instance

    @property
    def global_state(self):
        """Backward-compatible accessor for the main branch."""
        return self.branches["main"]

    @global_state.setter
    def global_state(self, value):
        self.branches["main"] = value

    async def boot_sequence(self, llm_adapter=None):
        """
        Replay the Akashic Ledger to restore the full Gödel state, then
        wire up the LLM adapter for live extraction and search.
        """
        if self.is_booted:
            return

        # Crash recovery
        self.branches["main"] = await self.ledger.rebuild_state(self.algebra)
        logger.info(
            "KOS boot complete — state bit-length=%d, axioms=%d",
            self.branches["main"].bit_length(),
            len(self.algebra.prime_to_axiom),
        )

        # Interacting Theory Setup (Causal Engine)
        self.trigger_map = CausalTriggerMap(self.algebra, self.ledger)

        # Built-in Rule: Symmetric relationships (married_to)
        def cond_symmetric(s, p, o, state, alg):
            return p == "married_to"

        def infer_symmetric(s, p, o, state, alg):
            return [(o, "married_to", s)]

        self.trigger_map.register_rule(cond_symmetric, infer_symmetric)

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
    branch: str = "main"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    branch: str = "main"


class IngestRequest(BaseModel):
    text: str
    branch: str = "main"


class ExtrapolateRequest(BaseModel):
    target_axioms: List[str]
    branch: str = "main"


class QuantumQueryRequest(BaseModel):
    nodes: List[str]
    hops: int = 1
    branch: str = "main"


class BranchRequest(BaseModel):
    source_branch: str
    new_branch: str


class MergeRequest(BaseModel):
    source_branch: str
    target_branch: str


# ─── Helper ──────────────────────────────────────────────────────────

def _get_branch_state(branch: str) -> int:
    """Retrieve the state integer for a branch, raising 404 if missing."""
    if branch not in kos.branches:
        raise HTTPException(
            status_code=404, detail=f"Branch '{branch}' not found"
        )
    return kos.branches[branch]


# ─── Routes ──────────────────────────────────────────────────────────

@router.get("/state")
async def get_global_state(branch: str = Query("main")):
    """Returns the current global Gödel integer and axiom count."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    state = _get_branch_state(branch)
    return {
        "branch": branch,
        "global_state_integer": str(state),
        "axiom_count": len(kos.algebra.prime_to_axiom),
        "branch_count": len(kos.branches),
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

    branch_state = _get_branch_state(request.branch)
    client_state = int(request.client_state_integer)
    delta = kos.algebra.calculate_network_delta(branch_state, client_state)

    return {
        "branch": request.branch,
        "new_global_state": str(branch_state),
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

    branch_state = _get_branch_state(request.branch)
    results = await kos.vector_bridge.semantic_search_godel_state(
        branch_state, request.query, request.top_k
    )
    return {
        "branch": request.branch,
        "verified_axioms": [
            {"axiom": axiom, "similarity": round(float(sim), 4)}
            for axiom, sim in results
        ],
    }


@router.post("/ingest")
async def ingest_document(
    request: IngestRequest, background_tasks: BackgroundTasks
):
    """
    Tomes → Tags. Ingests raw text into the Gödel state via MapReduce
    extraction, mathematical merge, and Akashic trace logging.
    Now triggers causal cascades via the Interacting Theory engine.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    if not hasattr(kos, "mass_engine"):
        raise HTTPException(
            status_code=503,
            detail="Mass engine not initialised (no LLM adapter)",
        )

    branch = request.branch
    branch_state = _get_branch_state(branch)

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
    kos.branches[branch] = math.lcm(branch_state, new_state)

    # 4. Akashic trace — log new primes
    delta_axioms = []
    for prime, axiom in kos.algebra.prime_to_axiom.items():
        if new_state % prime == 0:
            await kos.ledger.append_event("MINT", prime, axiom)
            await kos.ledger.append_event("MUL", prime)
            delta_axioms.append(axiom)

    # 5. Apply Interacting Theory (Causal Cascades)
    if hasattr(kos, "trigger_map") and delta_axioms:
        kos.branches[branch] = await kos.trigger_map.apply_cascade(
            kos.branches[branch], delta_axioms
        )

    # 6. Background vector indexing
    if hasattr(kos, "vector_bridge"):
        background_tasks.add_task(kos.vector_bridge.index_new_primes)

    return {
        "status": "success",
        "branch": branch,
        "new_global_state": str(kos.branches[branch]),
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

    branch_state = _get_branch_state(request.branch)

    try:
        narrative = await kos.extrapolator.extrapolate_with_proof(
            branch_state, request.target_axioms
        )
        return {"branch": request.branch, "verified_narrative": narrative}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Quantum GraphRAG ────────────────────────────────────────────────

@router.post("/query")
async def quantum_graph_rag(request: QuantumQueryRequest):
    """
    O(1) Topological GraphRAG Context Retrieval.

    Returns the exact axioms in the mathematical neighbourhood of the
    requested nodes, filtered to only include alive axioms.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    branch_state = _get_branch_state(request.branch)

    context_integer = kos.algebra.get_quantum_neighborhood(
        branch_state, request.nodes, request.hops
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
        "branch": request.branch,
        "context_integer": str(context_integer),
        "axioms": context_axioms,
    }


# ─── Git for Truth (Epistemic Branching) ─────────────────────────────

@router.post("/branch")
async def create_branch(req: BranchRequest):
    """
    O(1) Epistemic Branching.

    Creates a parallel universe of knowledge by copying a single integer.
    This is literally an integer assignment — the most efficient fork
    operation in the history of knowledge management.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    if req.source_branch not in kos.branches:
        raise HTTPException(
            status_code=404, detail=f"Source branch '{req.source_branch}' not found"
        )
    if req.new_branch in kos.branches:
        raise HTTPException(
            status_code=409, detail=f"Branch '{req.new_branch}' already exists"
        )

    kos.branches[req.new_branch] = kos.branches[req.source_branch]

    from internal.ensemble.epistemic_arbiter import kos_telemetry
    await kos_telemetry.broadcast(
        f"🔗 Branch Created: '{req.new_branch}' from '{req.source_branch}'"
    )

    return {
        "message": f"Branch '{req.new_branch}' created.",
        "state_integer": str(kos.branches[req.new_branch]),
        "branch_count": len(kos.branches),
    }


@router.post("/merge")
async def merge_branches(req: MergeRequest):
    """
    O(1) Merge via LCM.

    Merges two parallel universes of knowledge by computing their
    Least Common Multiple.  Any axiom in either branch survives.
    Paradoxes from conflicting facts are detected and reported.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    if req.source_branch not in kos.branches:
        raise HTTPException(
            status_code=404, detail=f"Source branch '{req.source_branch}' not found"
        )
    if req.target_branch not in kos.branches:
        raise HTTPException(
            status_code=404, detail=f"Target branch '{req.target_branch}' not found"
        )

    source_state = kos.branches[req.source_branch]
    target_state = kos.branches[req.target_branch]

    merged_state = math.lcm(source_state, target_state)

    # Audit for Paradoxes resulting from the merge
    paradoxes = kos.algebra.detect_curvature_paradoxes(merged_state)

    kos.branches[req.target_branch] = merged_state

    from internal.ensemble.epistemic_arbiter import kos_telemetry
    await kos_telemetry.broadcast(
        f"🔗 Branch Merged: '{req.source_branch}' → '{req.target_branch}' "
        f"| Paradoxes: {len(paradoxes)}"
    )

    return {
        "message": "Merge successful",
        "new_state": str(merged_state),
        "paradoxes_detected": len(paradoxes),
        "paradoxes": paradoxes,
    }


@router.get("/branches")
async def list_branches():
    """Lists all active epistemic branches and their state bit-lengths."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    return {
        "branches": {
            name: {
                "state_integer": str(state),
                "bit_length": state.bit_length(),
            }
            for name, state in kos.branches.items()
        }
    }


# ─── SSE Telemetry ────────────────────────────────────────────────────

@router.get("/telemetry")
async def telemetry_stream():
    """
    Server-Sent Events endpoint for real-time internal monologue.

    Streams paradox detection, superposition entry, wave function
    collapse, and causal inference events to the frontend.
    """
    from internal.ensemble.epistemic_arbiter import kos_telemetry

    async def event_generator():
        queue = kos_telemetry.subscribe()
        try:
            while True:
                message = await queue.get()
                yield f"data: {message}\n\n"
        except asyncio.CancelledError:
            kos_telemetry.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
