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
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from api.quantum_router import router as quantum_router, kos
from internal.ensemble.live_llm_adapter import LiveLLMAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot the KOS on startup, cleanup on shutdown."""
    logger.info("Initializing Quantum Knowledge OS...")
    api_key = os.getenv("OPENAI_API_KEY")
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

    yield  # App is running

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
