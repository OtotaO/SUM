"""
Autonomous Crystallizer — The Subconscious Graph Optimizer

A background daemon that monitors the Gödel State and autonomously
compresses dense topological clusters into Macro-Primes using the
Fractal Crystallization method from Phase 6.

When a node's topological degree exceeds a configurable threshold,
the daemon asks the LLM to summarize the cluster and collapses the
micro-primes into a single macro-prime — all while you sleep.

Author: ototao
License: Apache License 2.0
"""

import asyncio
import math
import logging
from typing import Callable, Optional

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger

logger = logging.getLogger(__name__)


def _zig():
    try:
        from sum_engine_internal.infrastructure.zig_bridge import zig_engine
        return zig_engine
    except ImportError:
        return None


class AutonomousCrystallizer:
    """
    The Subconscious Graph Optimizer.

    Monitors the Gödel State and autonomously compresses dense
    topological clusters into Macro-Primes.
    """

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        ledger: AkashicLedger,
        summarizer_llm: Callable,
    ):
        self.algebra = algebra
        self.ledger = ledger
        self.summarize = summarizer_llm  # async func(list[str]) -> str
        self.is_running = False

    async def run_compaction_cycle(
        self, global_state: int, threshold: int = 5
    ) -> int:
        """
        Scans the node registry for highly-connected nodes and
        crystallises their edge clusters into macro-primes.

        Args:
            global_state: Current Gödel integer.
            threshold:    Minimum edges before a node triggers compaction.

        Returns:
            Potentially compressed global state.
        """
        new_state = global_state

        for node, node_integer in list(self.algebra.node_registry.items()):
            z = _zig()
            zg = z.bigint_gcd(new_state, node_integer) if z else None
            alive_node_integer = zg if zg is not None else math.gcd(new_state, node_integer)

            # Extract alive axioms connected to this node
            cluster_axioms: list[str] = []
            temp_int = alive_node_integer
            for prime, axiom in self.algebra.prime_to_axiom.items():
                if temp_int % prime == 0:
                    cluster_axioms.append(axiom)
                    while temp_int % prime == 0:
                        temp_int //= prime
                if temp_int == 1:
                    break

            # If the node has too many connections, compress it
            if len(cluster_axioms) >= threshold:
                logger.info(
                    "[Subconscious] Compacting dense node '%s' "
                    "(%d edges)...",
                    node,
                    len(cluster_axioms),
                )

                # Ask LLM to summarize the cluster
                macro_predicate = await self.summarize(cluster_axioms)
                macro_key = (
                    f"{node}||"
                    f"{macro_predicate.lower().replace(' ', '_')}||"
                    f"compressed_cluster"
                )

                # Fractal Compression (Phase 6)
                new_state = self.algebra.crystallize_axioms(
                    new_state, cluster_axioms, macro_key
                )

                # Log to Akashic Ledger
                if macro_key in self.algebra.axiom_to_prime:
                    macro_prime = self.algebra.axiom_to_prime[macro_key]
                    await self.ledger.append_event(
                        "MINT", macro_prime, macro_key
                    )
                    await self.ledger.append_event("MUL", macro_prime)
                    for axiom in cluster_axioms:
                        p = self.algebra.axiom_to_prime.get(axiom)
                        if p:
                            await self.ledger.append_event("DIV", p)

        return new_state

    async def start_daemon(
        self,
        get_state_func: Callable[[], int],
        set_state_func: Callable[[int], None],
        interval: int = 60,
    ):
        """
        Background loop that runs compaction cycles at a fixed interval.

        Args:
            get_state_func: Callable returning the current global state.
            set_state_func: Callable to update the global state.
            interval:       Seconds between cycles.
        """
        self.is_running = True
        logger.info(
            "[Subconscious] Daemon started — compaction every %ds", interval
        )
        while self.is_running:
            await asyncio.sleep(interval)
            try:
                current_state = get_state_func()
                new_state = await self.run_compaction_cycle(current_state)
                if new_state != current_state:
                    set_state_func(new_state)
                    logger.info(
                        "[Subconscious] State compacted: %d → %d bits",
                        current_state.bit_length(),
                        new_state.bit_length(),
                    )
            except Exception as e:
                logger.error("[Subconscious] Compaction error: %s", e)

    def stop_daemon(self):
        """Gracefully stop the background daemon."""
        self.is_running = False
