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
from typing import List

from fastapi import APIRouter, HTTPException
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
    chunks: List[str]


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
            {"axiom": axiom, "similarity": round(sim, 4)}
            for axiom, sim in results
        ]
    }
