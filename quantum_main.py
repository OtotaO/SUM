import sys
# Zenith of Process Intensification: Uncap the maximum integer string
# conversion limit.  The Gödel Integer will grow into hundreds of
# thousands of digits as the harvester devours the internet.
sys.set_int_max_str_digits(0)

"""
Quantum Knowledge OS — ASGI Entrypoint

Mounts the Quantum API router, serves the frontend, and initialises
the GlobalKnowledgeOS from the Akashic Ledger on startup.

Usage:
    export OPENAI_API_KEY="sk-..."
    uvicorn quantum_main:app --reload --port 8000

Author: ototao
License: Apache License 2.0
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from api.quantum_router import router as quantum_router, kos
from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter
from sum_engine_internal.ensemble.autonomous_agent import AutonomousCrystallizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot the KOS on startup, start daemon, cleanup on shutdown."""
    logger.info("Initializing Quantum Knowledge OS...")
    api_key = os.getenv("OPENAI_API_KEY")
    llm_adapter = None
    if api_key:
        llm_adapter = LiveLLMAdapter(api_key=api_key)
        await kos.boot_sequence(llm_adapter)
    else:
        logger.warning(
            "OPENAI_API_KEY not set — booting in math-only mode "
            "(no LLM extraction or search)."
        )
        await kos.boot_sequence()

    state_str = str(kos.global_state)
    preview = state_str[:30] + "..." if len(state_str) > 30 else state_str
    logger.info("🚀 KOS Booted. Global State: %s", preview)

    # Start the Autonomous Crystallizer daemon
    async def summarize_cluster(axioms):
        if llm_adapter:
            return await llm_adapter.generate_text(
                axioms, negative_constraints=[]
            )
        return "has_complex_history"

    kos.crystallizer = AutonomousCrystallizer(
        kos.algebra, kos.ledger, summarize_cluster
    )
    daemon_task = asyncio.create_task(
        kos.crystallizer.start_daemon(
            get_state_func=lambda: kos.global_state,
            set_state_func=lambda s: setattr(kos, "global_state", s),
            interval=30,
        )
    )

    # Start the P2P Holographic Mesh daemon
    mesh_task = None
    if kos.mesh:
        mesh_task = asyncio.create_task(
            kos.mesh.start_gossip_daemon(interval=15)
        )

    # Wire the Epistemic Arbiter (Wave Function Collapse)
    from sum_engine_internal.ensemble.epistemic_arbiter import EpistemicArbiter

    async def llm_judge(prompt: str) -> str:
        if llm_adapter and hasattr(llm_adapter, "client"):
            response = await llm_adapter.client.chat.completions.create(
                model=llm_adapter.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content
        return prompt.split("Claim B:")[-1].strip().split()[-1]

    kos.arbiter = EpistemicArbiter(llm_judge)

    yield  # App is running

    # Graceful shutdown
    kos.crystallizer.stop_daemon()
    daemon_task.cancel()
    if kos.mesh:
        kos.mesh.stop_daemon()
    if mesh_task:
        mesh_task.cancel()
    logger.info("Quantum Knowledge OS shutting down.")


app = FastAPI(
    title="SUM: Quantum Knowledge OS",
    description="Gödel-State Engine with O(1) semantic operations",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
os.makedirs("static/js", exist_ok=True)

# Mount routes and static files
app.include_router(quantum_router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Redirect to the Quantum UI."""
    return RedirectResponse(url="/static/quantum.html")
