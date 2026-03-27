"""
Automated Scientist Daemon — Machine Synthesis

Background asyncio task that continuously evaluates the Gödel Integer,
discovers novel topological relationships via transitive closure,
batch-mints primes via the Zig C-ABI, and permanently expands the
system's intelligence.

Every discovery is logged to the Akashic Ledger as a ``DEDUCED`` event,
providing a complete provenance trail of machine-generated knowledge.

Author: ototao
License: Apache License 2.0
"""

import math
import asyncio
import logging
from typing import Optional

from internal.algorithms.causal_discovery import CausalDiscoveryEngine

logger = logging.getLogger("sum.scientist")


class AutomatedScientistDaemon:
    """
    Horizon V: The Dreaming Machine.

    Runs in the background, sweeping the global Gödel state for
    logically entailed but unminted axioms.  Uses Zig FFI batch
    minting when available.
    """

    def __init__(self, kos_instance, interval_seconds: int = 15):
        self.kos = kos_instance
        self.interval = interval_seconds
        self.discovery_engine = CausalDiscoveryEngine(self.kos.algebra)
        self.running = False
        self.total_discoveries = 0

    async def start_dreaming(self):
        """Begin the autonomous deduction loop."""
        self.running = True

        # Broadcast startup
        try:
            from internal.ensemble.epistemic_arbiter import kos_telemetry
            await kos_telemetry.broadcast(
                "🔬 Automated Scientist initialized. Awaiting REM sleep cycles..."
            )
        except Exception:
            pass

        while self.running:
            await asyncio.sleep(self.interval)
            try:
                await self._dream_cycle()
            except Exception as e:
                logger.error("Dream cycle error: %s", e)

    async def stop(self):
        """Graceful shutdown."""
        self.running = False

    async def _dream_cycle(self):
        """
        Single deduction sweep:
        1. Get current state
        2. Discover novel transitive relationships
        3. Batch-mint primes (Zig if available, Python fallback)
        4. LCM into global state
        5. Log to Akashic Ledger
        """
        current_state = self.kos.branches.get("main", 1)
        if current_state == 1:
            return  # Nothing to analyze

        novel_triplets = self.discovery_engine.sweep_for_discoveries(current_state)
        if not novel_triplets:
            return

        # Broadcast discovery
        try:
            from internal.ensemble.epistemic_arbiter import kos_telemetry
            await kos_telemetry.broadcast(
                f"🧠 EUREKA! Synthesizing {len(novel_triplets)} novel discoveries..."
            )
        except Exception:
            pass

        axiom_strings = [f"{s}||{p}||{o}" for s, p, o in novel_triplets]

        # Batch mint — Zig bare-metal if available
        try:
            from internal.infrastructure.zig_bridge import zig_engine
            if zig_engine and hasattr(zig_engine, 'batch_mint_primes'):
                new_primes = zig_engine.batch_mint_primes(axiom_strings)
            else:
                new_primes = None
        except ImportError:
            new_primes = None

        if new_primes is None:
            new_primes = [
                self.kos.algebra.get_or_mint_prime(s, p, o)
                for s, p, o in novel_triplets
            ]

        async with self.kos.branch_lock("main"):
            current_state = self.kos.branches.get("main", 1)
            new_state = current_state
            for prime, axiom_str in zip(new_primes, axiom_strings):
                new_state = math.lcm(new_state, prime)
                await self.kos.ledger.append_event("DEDUCED", prime, axiom_str)
                await self.kos.ledger.append_event("MUL", prime)

            self.kos.branches["main"] = new_state
            # 19D coherence: daemon mutation must update index
            if hasattr(self.kos, 'prime_index'):
                self.kos.prime_index.rebuild("main", new_state, self.kos.algebra)
                self.kos.prime_index.assert_coherent(
                    "main", new_state, self.kos.algebra, context="automated_scientist"
                )
        self.total_discoveries += len(novel_triplets)

        logger.info(
            "Scientist: %d new discoveries (total: %d)",
            len(novel_triplets), self.total_discoveries
        )

        try:
            from internal.ensemble.epistemic_arbiter import kos_telemetry
            await kos_telemetry.broadcast(
                f"🧬 State multiplied. Total autonomous discoveries: {self.total_discoveries}"
            )
        except Exception:
            pass
