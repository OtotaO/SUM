import sys
sys.set_int_max_str_digits(0)

"""
Quantum Router — The Global Knowledge OS API

Exposes the Gödel-State Engine to the outside world via FastAPI:
    /state           – current global integer (branch-aware)
    /sync            – delta synchronisation (Gödel Sync Protocol)
    /search          – semantic search over alive primes
    /ingest          – ingest text into the global state via live LLM
    /extrapolate     – verified narrative generation
    /query           – GraphRAG neighbourhood retrieval
    /branch          – epistemic branching (Git for Truth)
    /merge           – LCM-based branch merging
    /time-travel     – Chronos Engine (historical state rebuild)
    /peers           – P2P Holographic Mesh peer management
    /zk/prove        – Zero-Knowledge Semantic Proofs
    /tick            – Current Akashic Ledger tick
    /auth/token      – Quantum Passport (JWT multi-tenancy)

The ``GlobalKnowledgeOS`` singleton boots from the Akashic Ledger on
startup, rebuilding the exact Gödel BigInt from the SQLite trace.

Phase 13: Zenith of Process Intensification — JWT Sovereign Tenancy.

Author: ototao
License: Apache License 2.0
"""

import logging
import math
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.algorithms.syntactic_sieve import DeterministicSieve
from internal.ensemble.vector_bridge import ContinuousDiscreteBridge
from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.infrastructure.canonical_codec import CanonicalCodec, InvalidSignatureError
from internal.ensemble.epistemic_loop import QuantumExtrapolator
from internal.ensemble.causal_triggers import CausalTriggerMap
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.ensemble.ouroboros import OuroborosVerifier
from internal.algorithms.zk_semantics import ZKSemanticProver
from internal.infrastructure.p2p_mesh import EpistemicMeshNetwork
from internal.ensemble.mass_semantic_engine import MassSemanticEngine
from internal.ensemble.confidence_calibrator import ConfidenceCalibrator

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

            # Dynamic DB allocation for the P2P Swarm
            db_path = os.getenv("AKASHIC_DB", "production_akashic.db")
            cls._instance.ledger = AkashicLedger(db_path)

            cls._instance.branches = {"main": 1}  # Multiverse State
            cls._instance.mesh = None
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

        # Phase 14: Deterministic Sieve + Tome Generator + Ouroboros
        # These work in math-only mode (no LLM required)
        self.sieve = DeterministicSieve()
        self.tome_generator = AutoregressiveTomeGenerator(
            self.algebra,
            extrapolator=getattr(self, 'extrapolator', None),
        )
        self.ouroboros = OuroborosVerifier(
            self.algebra, self.sieve, self.tome_generator,
        )

        # Phase 15: Canonical Codec for signed knowledge transport
        self.codec = CanonicalCodec(
            self.algebra,
            self.tome_generator,
            signing_key=SECRET_KEY,
        )

        # P2P Holographic Mesh
        self.mesh = EpistemicMeshNetwork(
            self.algebra,
            lambda b: self.branches.get(b, 1),
            self._update_branch_state,
        )

        # Phase 19: Automated Scientist Daemon
        from internal.ensemble.automated_scientist import AutomatedScientistDaemon
        self.scientist_daemon = AutomatedScientistDaemon(self, interval_seconds=15)
        asyncio.create_task(self.scientist_daemon.start_dreaming())

        self.is_booted = True

    def _update_branch_state(self, branch: str, new_state: int):
        """Callback for the P2P mesh to update branch states."""
        self.branches[branch] = new_state


# Singleton instance used by route handlers
kos = GlobalKnowledgeOS()


# ─── JWT Sovereign Tenancy ────────────────────────────────────────────

SECRET_KEY = os.getenv("SUM_JWT_SECRET", "quantum_supremacy_secret_key_minimum_32b")
ALGORITHM = "HS256"


def create_access_token(user_id: str) -> str:
    """Create a JWT Quantum Passport for multi-tenant branch isolation."""
    expire = datetime.utcnow() + timedelta(days=7)
    return jwt.encode(
        {"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM
    )


async def get_current_user(authorization: str = Header(None)) -> str:
    """
    Validates JWT and returns the User ID (which acts as their
    semantic branch in the Gödel multiverse).

    Falls back to ``"main"`` if no token is provided, preserving
    backward compatibility for local dev and unauthenticated usage.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return "main"  # Legacy / unauthenticated

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub", "main")
        if user_id not in kos.branches:
            kos.branches[user_id] = 1
        return user_id
    except Exception:
        raise HTTPException(
            status_code=401, detail="Invalid or expired Quantum Passport"
        )



# ─── Request / Response models ────────────────────────────────────────

class TokenRequest(BaseModel):
    """Request a Quantum Passport (JWT) for multi-tenant access."""
    username: str


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
    source_url: str = ""


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


class TimeTravelRequest(BaseModel):
    target_tick: int
    new_branch_name: str


class DirectMathRequest(BaseModel):
    """Zero-cost ingestion: bypass LLM, inject triplets directly into the math engine."""
    triplets: List[List[str]]  # [[subject, predicate, object]]
    branch: str = "main"
    source_url: str = ""
    confidence_mode: str = "auto"  # "auto", "manual", or "llm"
    confidence: Optional[float] = None  # only used when confidence_mode="manual"

class PeerRequest(BaseModel):
    peer_url: str


class ZKProofRequest(BaseModel):
    axiom_key: str
    branch: str = "main"


class RehydrateRequest(BaseModel):
    """Unpack a Gödel Integer into a structured Tome."""
    title: str = "The Rehydrated Codex"
    mode: str = "proof"  # "proof" (canonical) or "narrative" (LLM)


class LearnRequest(BaseModel):
    """Generate an Epistemic Delta Tome for personalized learning."""
    target_topic_node: str


class SyncStateRequest(BaseModel):
    """P2P sync: accept a foreign Gödel Integer for O(1) LCM merge."""
    peer_state_integer: str


class OuroborosRequest(BaseModel):
    """Verify semantic conservation via round-trip encoding."""
    text: str


class AskRequest(BaseModel):
    """Natural-language knowledge retrieval over the Gödel state."""
    question: str
    branch: str = "main"
    top_k: int = 10


# ─── Helper ──────────────────────────────────────────────────────────

def _get_branch_state(branch: str) -> int:
    """Retrieve the state integer for a branch, raising 404 if missing."""
    if branch not in kos.branches:
        raise HTTPException(
            status_code=404, detail=f"Branch '{branch}' not found"
        )
    return kos.branches[branch]


# ─── Routes ──────────────────────────────────────────────────────────

@router.post("/auth/token")
async def generate_token(req: TokenRequest):
    """Generates a Quantum Passport for Multi-Tenancy."""
    token = create_access_token(req.username)
    if req.username not in kos.branches:
        kos.branches[req.username] = 1
    return {
        "access_token": token,
        "token_type": "bearer",
        "branch": req.username,
    }


@router.get("/state")
async def get_global_state(
    user_id: str = Depends(get_current_user),
    branch: str = Query(None),
):
    """Returns the current global Gödel integer and axiom count."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    # JWT user_id takes precedence; fall back to query param for legacy usage
    effective_branch = user_id if user_id != "main" else (branch or "main")
    state = _get_branch_state(effective_branch)
    return {
        "branch": effective_branch,
        "global_state_integer": str(state),
        "axiom_count": len(kos.algebra.prime_to_axiom),
        "branch_count": len(kos.branches),
        "user_id": user_id,
    }


@router.post("/sync")
async def sync_client_state(
    request: SyncRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Delta Sync (Gödel Sync Protocol).

    Returns exactly which axioms the client needs to add and delete
    to match the server's state.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    effective_branch = user_id if user_id != "main" else request.branch
    branch_state = _get_branch_state(effective_branch)
    client_state = int(request.client_state_integer)
    delta = kos.algebra.calculate_network_delta(branch_state, client_state)

    return {
        "branch": effective_branch,
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

    # 4. Akashic trace — log new primes (Phase 22: with provenance)
    from datetime import datetime as _dt
    _now = _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    delta_axioms = []
    for prime, axiom in kos.algebra.prime_to_axiom.items():
        if new_state % prime == 0:
            await kos.ledger.append_event(
                "MINT", prime, axiom,
                source_url=request.source_url,
                ingested_at=_now,
            )
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


@router.post("/ingest/math")
async def ingest_math_direct(
    request: DirectMathRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    """
    Zero-Cost Ingestion: Bypasses the LLM entirely and injects strict
    mathematical triplets directly into the Gödel-State Engine.

    Each triplet is a [subject, predicate, object] array that gets
    deterministically mapped to a Semantic Prime, merged via LCM,
    and cascaded through the Interacting Theory engine.

    Cost: $0. Speed: microseconds. Truth: absolute.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    branch = user_id if user_id != "main" else request.branch
    current_state = kos.branches.get(branch, 1)
    new_state_product = 1
    added_axioms = []

    for t in request.triplets:
        if len(t) == 3:
            s, p, o = [x.strip().lower() for x in t]
            prime = kos.algebra.get_or_mint_prime(s, p, o)
            axiom = f"{s}||{p}||{o}"

            # Only process if genuinely new to both current state and this batch
            if current_state % prime != 0 and new_state_product % prime != 0:
                new_state_product = math.lcm(new_state_product, prime)
                added_axioms.append((prime, axiom))

    if added_axioms:
        new_state = math.lcm(current_state, new_state_product)

        # Phase 24: Auto-calibrate confidence
        from datetime import datetime as _dt
        _now = _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        calibrator = ConfidenceCalibrator()

        for prime, axiom in added_axioms:
            if request.confidence_mode == "manual" and request.confidence is not None:
                conf = max(0.0, min(1.0, request.confidence))
            else:
                conf = await calibrator.calibrate(
                    axiom_key=axiom,
                    source_url=request.source_url,
                    current_state=current_state,
                    algebra=kos.algebra,
                    ledger=kos.ledger,
                )
            await kos.ledger.append_event(
                "MINT", prime, axiom,
                source_url=request.source_url,
                confidence=conf,
                ingested_at=_now,
            )
            await kos.ledger.append_event("MUL", prime)

        # Apply Interacting Theory (Causal Cascades)
        axioms_strs = [a[1] for a in added_axioms]
        if hasattr(kos, "trigger_map"):
            new_state = await kos.trigger_map.apply_cascade(
                new_state, axioms_strs
            )

        kos.branches[branch] = new_state

        # Background vector indexing
        if hasattr(kos, "vector_bridge"):
            background_tasks.add_task(kos.vector_bridge.index_new_primes)

    return {
        "status": "success",
        "branch": branch,
        "new_global_state": str(kos.branches.get(branch, 1)),
        "axioms_added": len(added_axioms),
        "user_id": user_id,
    }

@router.post("/extrapolate")
async def extrapolate_tome(
    request: ExtrapolateRequest,
    user_id: str = Depends(get_current_user),
):
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

    effective_branch = user_id if user_id != "main" else request.branch
    branch_state = _get_branch_state(effective_branch)

    try:
        narrative = await kos.extrapolator.extrapolate_with_proof(
            branch_state, request.target_axioms
        )
        return {"branch": effective_branch, "verified_narrative": narrative}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Quantum GraphRAG ────────────────────────────────────────────────

@router.post("/query")
async def quantum_graph_rag(
    request: QuantumQueryRequest,
    user_id: str = Depends(get_current_user),
):
    """
    GraphRAG Topological Context Retrieval.

    Returns the exact axioms in the mathematical neighbourhood of the
    requested nodes, filtered to only include alive axioms.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    effective_branch = user_id if user_id != "main" else request.branch
    branch_state = _get_branch_state(effective_branch)

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
        "branch": effective_branch,
        "context_integer": str(context_integer),
        "axioms": context_axioms,
    }


# ─── Git for Truth (Epistemic Branching) ─────────────────────────────

@router.post("/branch")
async def create_branch(req: BranchRequest):
    """
    Epistemic Branching.

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
    Merge via LCM.

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


# ─── Chronos Engine (Time Travel) ────────────────────────────────────

@router.post("/time-travel")
async def time_travel(
    req: TimeTravelRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Chronos Engine (historical state rebuild).

    Rebuilds the exact state of the universe at a historical ledger
    tick into an alternate timeline branch.  This is a full BigInt
    reconstruction from the Akashic trace, filtered to ``max_seq_id``.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    if req.new_branch_name in kos.branches:
        raise HTTPException(
            status_code=409,
            detail=f"Branch '{req.new_branch_name}' already exists",
        )

    # Rebuild a fresh algebra for the historical snapshot
    from internal.algorithms.semantic_arithmetic import GodelStateAlgebra as GSA
    historical_algebra = GSA()
    past_state = await kos.ledger.rebuild_state(
        historical_algebra, max_seq_id=req.target_tick
    )
    kos.branches[req.new_branch_name] = past_state

    from internal.ensemble.epistemic_arbiter import kos_telemetry
    await kos_telemetry.broadcast(
        f"⏳ Chronos Time Travel: Branch '{req.new_branch_name}' "
        f"created at tick {req.target_tick}"
    )

    return {
        "message": (
            f"Time travel successful. Branch '{req.new_branch_name}' "
            f"created at tick {req.target_tick}."
        ),
        "historical_state_integer": str(past_state),
        "branch_count": len(kos.branches),
    }


@router.get("/tick")
async def get_latest_tick():
    """Returns the current Akashic Ledger tick (latest seq_id)."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    tick = await kos.ledger.get_latest_tick()
    return {"latest_tick": tick}


# ─── Zero-Knowledge Semantic Proofs ──────────────────────────────────

@router.post("/zk/prove")
async def generate_zk_proof(
    req: ZKProofRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Generates a Zero-Knowledge proof that this node knows a specific
    axiom without revealing the full state integer.

    Returns a salted hash commitment over the quotient.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    prime = kos.algebra.axiom_to_prime.get(req.axiom_key)
    if not prime:
        raise HTTPException(status_code=404, detail="Axiom not known")

    effective_branch = user_id if user_id != "main" else req.branch
    state = _get_branch_state(effective_branch)
    try:
        proof = ZKSemanticProver.generate_proof(state, prime)
        return proof
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/zk/verify")
async def verify_zk_proof(proof: dict):
    """Verifies a Zero-Knowledge semantic proof."""
    valid = ZKSemanticProver.verify_proof(proof)
    return {"valid": valid}


# ─── P2P Holographic Mesh ────────────────────────────────────────────

@router.post("/peers")
async def add_network_peer(req: PeerRequest):
    """
    Add a remote KOS node to the Holographic Mesh.

    Immediately triggers an initial sync with the new peer.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    if not kos.mesh:
        raise HTTPException(status_code=503, detail="P2P mesh not initialised")

    kos.mesh.add_peer(req.peer_url)
    asyncio.create_task(kos.mesh._sync_with_peer(req.peer_url))

    return {
        "message": f"Peer {req.peer_url} added to Epistemic Mesh.",
        "active_peers": list(kos.mesh.peers),
    }


@router.get("/peers")
async def list_peers():
    """Lists all known peers in the Holographic Mesh."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")
    peers = list(kos.mesh.peers) if kos.mesh else []
    return {"peers": peers, "count": len(peers)}


# ─── Phase 14: Ouroboros Protocol ────────────────────────────────────

@router.post("/rehydrate")
async def rehydrate_tome(
    req: RehydrateRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Lossless Semantic Rehydration.

    Unpacks the user's Gödel branch into a structured Tome.

    Modes:
        * ``proof``     — deterministic canonical output (round-trips)
        * ``narrative``  — LLM-enhanced prose (requires LLM adapter)
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    current_state = kos.branches.get(user_id, 1)
    if current_state == 1:
        return {
            "tome": "Knowledge state is empty. Nothing to rehydrate.",
            "axiom_count": 0,
        }

    tome_text = await kos.tome_generator.generate_tome(
        current_state, req.title, mode=req.mode
    )
    active_axioms = kos.tome_generator.extract_active_axioms(current_state)

    return {
        "tome": tome_text,
        "mode": req.mode,
        "axiom_count": len(active_axioms),
        "state_digits": len(str(current_state)),
        "user_id": user_id,
    }


@router.post("/learn")
async def generate_epistemic_delta(
    req: LearnRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Epistemic Delta Generation (Personalized Learning).

    Computes the exact mathematical gap between the user's knowledge
    and a target topic, then generates a custom Tome containing only
    the facts the user does not already know:

        delta = topic_integer // gcd(topic_integer, user_integer)

    This is the perfectly optimized educational curriculum.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    user_state = kos.branches.get(user_id, 1)

    # Pull topic knowledge via GraphRAG from the global main branch
    main_state = kos.branches.get("main", 1)
    topic_integer = kos.algebra.get_quantum_neighborhood(
        main_state, [req.target_topic_node.strip().lower()], hops=1
    )

    if topic_integer == 1:
        return {
            "target_topic": req.target_topic_node,
            "educational_tome": (
                f"No knowledge found about '{req.target_topic_node}' "
                f"in the global state."
            ),
            "delta_axiom_count": 0,
        }

    # The Epistemic Delta: Topic ÷ gcd(Topic, User)
    shared_knowledge = math.gcd(topic_integer, user_state)
    delta_integer = topic_integer // shared_knowledge

    if delta_integer == 1:
        return {
            "target_topic": req.target_topic_node,
            "educational_tome": (
                f"You already know everything about "
                f"'{req.target_topic_node}'."
            ),
            "delta_axiom_count": 0,
        }

    # Generate the educational Tome from the delta
    delta_axioms = kos.tome_generator.extract_active_axioms(delta_integer)
    tome = kos.tome_generator.generate_canonical(
        delta_integer,
        f"Learning Delta: {req.target_topic_node.title()}",
    )

    return {
        "target_topic": req.target_topic_node,
        "educational_tome": tome,
        "delta_axiom_count": len(delta_axioms),
        "user_id": user_id,
    }


@router.post("/ouroboros/verify")
async def verify_semantic_conservation(req: OuroborosRequest):
    """
    Prove Semantic Conservation (The Ouroboros Protocol).

    Accepts raw text and runs the full round-trip:
        Text → Sieve → Integer A → Canonical Tome → Sieve → Integer B

    Returns a proof object demonstrating whether A == B
    (lossless conservation) and detailed diagnostics if not.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    proof = kos.ouroboros.verify_from_text(req.text)
    result = kos.ouroboros.proof_to_dict(proof)
    result["canonical_tome"] = proof.canonical_tome

    return result


@router.post("/export")
async def export_knowledge_bundle(
    user_id: str = Depends(get_current_user),
):
    """
    Export a branch's state as a signed, self-contained bundle.

    The bundle includes the canonical tome, state integer, axiom count,
    timestamp, and an HMAC-SHA256 signature for tamper detection.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    current_state = kos.branches.get(user_id, 1)
    bundle = kos.codec.export_bundle(
        current_state, branch=user_id, title=f"Export: {user_id}"
    )
    return bundle


class ImportRequest(BaseModel):
    """Import a signed knowledge bundle."""
    bundle: dict


@router.post("/import")
async def import_knowledge_bundle(
    req: ImportRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Import and verify a signed bundle into the user's branch.

    Validates HMAC-SHA256 signature, then merges the imported state
    via LCM into the user's branch.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    try:
        imported_state = kos.codec.import_bundle(req.bundle)
    except InvalidSignatureError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Merge via LCM
    current = kos.branches.get(user_id, 1)
    new_state = math.lcm(current, imported_state)
    kos.branches[user_id] = new_state

    return {
        "imported": True,
        "user_id": user_id,
        "previous_axiom_count": len(
            kos.tome_generator.extract_active_axioms(current)
        ),
        "new_axiom_count": len(
            kos.tome_generator.extract_active_axioms(new_state)
        ),
    }


# ─── SSE Telemetry ────────────────────────────────────────────────

@router.get("/telemetry")
async def telemetry_stream():
    """
    Server-Sent Events endpoint for real-time internal monologue.

    Streams paradox detection, superposition entry, wave function
    collapse, causal inference, time travel, and P2P sync events.
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

# ─── Phase 19: Sovereign Edge & Machine Synthesis ─────────────────

@router.post("/sync/state")
async def sync_peer_state(
    req: SyncStateRequest,
    user_id: str = Depends(get_current_user),
):
    """O(1) P2P Merge. Accepts a foreign Gödel Integer and merges via LCM."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    effective_branch = user_id if user_id != "main" else "main"
    current_state = kos.branches.get(effective_branch, 1)

    try:
        peer_int = int(req.peer_state_integer)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid peer state integer.")

    # Use Zig FFI if available, fallback to Python
    try:
        from internal.infrastructure.zig_bridge import zig_engine
        if zig_engine and hasattr(zig_engine, 'bigint_lcm'):
            new_state = zig_engine.bigint_lcm(current_state, peer_int)
        else:
            new_state = math.lcm(current_state, peer_int)
    except ImportError:
        new_state = math.lcm(current_state, peer_int)

    if new_state != current_state:
        kos.branches[effective_branch] = new_state
        await kos.ledger.append_event(
            "SYNC", 1, f"Merged peer state from {user_id}"
        )

        from internal.ensemble.epistemic_arbiter import kos_telemetry
        await kos_telemetry.broadcast(
            f"🌐 Sovereign Edge Sync: merged state from {user_id}"
        )

    return {
        "status": "synchronized",
        "branch": effective_branch,
        "global_state_integer": str(new_state),
    }


@router.get("/discoveries")
async def get_autonomous_discoveries():
    """Returns knowledge autonomously deduced by the Automated Scientist."""
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    total = 0
    if hasattr(kos, 'scientist_daemon'):
        total = kos.scientist_daemon.total_discoveries

    # Query ledger for DEDUCED events
    import sqlite3
    try:
        with sqlite3.connect(kos.ledger.db_path) as conn:
            rows = conn.execute(
                "SELECT axiom_key, rowid FROM semantic_events "
                "WHERE operation = 'DEDUCED' ORDER BY rowid DESC LIMIT 50"
            ).fetchall()
        discoveries = [
            {"axiom": r[0].replace("||", " "), "id": r[1]} for r in rows
        ]
    except Exception:
        discoveries = []

    return {
        "total_discoveries": total,
        "recent": discoveries,
    }


# ─── Phase 21+22: Knowledge Retrieval + Provenance ──────────────────

@router.post("/ask")
async def ask_knowledge(
    req: AskRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Natural-Language Knowledge Retrieval over the Gödel State.

    Closes the ingest↔retrieve loop. Works in two modes:

    1. **Math-only (always available):** Tokenizes the question into
       keywords, scans all axioms alive in the current state for
       substring matches, and returns ranked results.

    2. **Semantic (when VectorBridge is booted):** Uses embedding
       cosine similarity for fuzzy matching, then optionally
       synthesizes a narrative answer via the EpistemicLoop.

    Phase 22: Results are weighted by confidence × recency, and
    provenance metadata (source_url, confidence, ingested_at) is
    included in every match.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    effective_branch = user_id if user_id != "main" else req.branch
    current_state = kos.branches.get(effective_branch, 1)

    if current_state == 1:
        return {
            "question": req.question,
            "matches": [],
            "answer": "No knowledge has been ingested yet.",
            "mode": "empty",
        }

    # ── 1. Get all active axioms ─────────────────────────────────
    active_axioms = kos.algebra.get_active_axioms(current_state)

    # ── 2. Keyword matching (always works, no LLM needed) ────────
    from internal.algorithms.predicate_canon import canonicalize
    question_lower = req.question.lower()
    keywords = [
        w for w in question_lower.replace("?", "").replace(".", "").split()
        if len(w) > 2 and w not in {
            "the", "and", "for", "are", "was", "were", "has", "have",
            "this", "that", "with", "from", "what", "who", "how",
            "why", "when", "where", "does", "did", "can", "could",
            "will", "would", "should", "about", "which",
        }
    ]

    scored_axioms = []
    for ax in active_axioms:
        parts = ax.split("||")
        if len(parts) == 3:
            s, p, o = parts
            # Score by keyword overlap
            ax_text = f"{s} {p} {o}".lower()
            score = sum(1 for kw in keywords if kw in ax_text)
            if score > 0:
                prime = kos.algebra.axiom_to_prime.get(ax, 0)
                scored_axioms.append({
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "axiom_key": ax,
                    "prime": prime,
                    "relevance_score": float(score),
                })

    # ── 2b. Provenance-weighted scoring (Phase 22) ────────────────
    if scored_axioms:
        axiom_keys = [m["axiom_key"] for m in scored_axioms]
        provenance = await kos.ledger.get_provenance_batch(axiom_keys)

        from datetime import datetime as _dt
        _now = _dt.utcnow()
        for m in scored_axioms:
            prov = provenance.get(m["axiom_key"])
            if prov:
                m["source_url"] = prov["source_url"]
                m["confidence"] = prov["confidence"]
                m["ingested_at"] = prov["ingested_at"]

                # Recency factor: halve score every 30 days
                recency = 1.0
                if prov["ingested_at"]:
                    try:
                        ts = _dt.fromisoformat(prov["ingested_at"])
                        age_days = max((_now - ts).days, 0)
                        recency = 0.5 ** (age_days / 30.0)
                    except (ValueError, TypeError):
                        pass

                # Weighted score = keyword_hits × confidence × recency
                m["relevance_score"] = (
                    m["relevance_score"] * prov["confidence"] * recency
                )
            else:
                m["source_url"] = ""
                m["confidence"] = 0.5
                m["ingested_at"] = ""

    # Sort by weighted relevance
    scored_axioms.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_matches = scored_axioms[:req.top_k]

    # ── 3. Semantic search (if VectorBridge is available) ─────────
    semantic_matches = []
    mode = "keyword"

    if hasattr(kos, "vector_bridge") and kos.vector_bridge is not None:
        try:
            results = await kos.vector_bridge.semantic_search_godel_state(
                query=req.question,
                global_state=current_state,
                top_k=req.top_k,
            )
            semantic_matches = [
                {
                    "axiom_key": r["axiom_key"],
                    "similarity": round(r["similarity"], 4),
                }
                for r in results
            ]
            mode = "semantic"
        except Exception as e:
            logger.warning("VectorBridge search failed: %s", e)

    # ── 4. Synthesize answer from matched axioms ─────────────────
    answer = None
    if top_matches:
        # Build a concise answer from matched axioms
        facts = [
            f"{m['subject']} {m['predicate']} {m['object']}"
            for m in top_matches[:5]
        ]
        answer = "Based on known axioms:\n" + "\n".join(
            f"  • {fact}" for fact in facts
        )

        # If LLM is available, generate a synthesized narrative
        if hasattr(kos, "extrapolator") and kos.extrapolator is not None:
            try:
                axiom_keys = [m["axiom_key"] for m in top_matches[:5]]
                narrative = await kos.extrapolator.extrapolate_with_proof(
                    current_state, axiom_keys
                )
                answer = narrative
                mode = "verified_narrative"
            except Exception as e:
                logger.warning("Narrative synthesis failed: %s", e)

    return {
        "question": req.question,
        "matches": top_matches,
        "semantic_matches": semantic_matches,
        "answer": answer or "No matching knowledge found.",
        "mode": mode,
        "total_active_axioms": len(active_axioms),
        "branch": effective_branch,
    }


# ─── Phase 22: Provenance Query ──────────────────────────────────────

@router.get("/provenance/{axiom_key:path}")
async def get_provenance(
    axiom_key: str,
    user_id: str = Depends(get_current_user),
):
    """
    Retrieve the full provenance chain for a specific axiom.

    Returns every source that contributed this axiom, with its
    confidence score and ingestion timestamp. The axiom_key is
    the normalised ``subject||predicate||object`` string.
    """
    if not kos.is_booted:
        raise HTTPException(status_code=503, detail="KOS booting")

    chain = await kos.ledger.get_axiom_provenance(axiom_key)

    # Check if the axiom actually exists in the algebra
    prime = kos.algebra.axiom_to_prime.get(axiom_key)

    return {
        "axiom_key": axiom_key,
        "prime": prime,
        "provenance_chain": chain,
        "total_sources": len(chain),
    }

