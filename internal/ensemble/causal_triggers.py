"""
Causal Trigger Map — Interacting Theory Engine (Yaroslavtsev §5.7)

Transitions the Knowledge OS from "Free Theory" (isolated axiom minting)
to "Interacting Theory" where logical rules cascade deductive inferences
in ACID-compliant transactions.

If A implies B, minting A automatically cascades and mints B in the same
operation.  Idempotency is guaranteed by Gödel LCM arithmetic — attempting
to mint an already-entailed prime is a no-op (LCM(state, p) == state).

Author: ototao
License: Apache License 2.0
"""

import math
from typing import Callable, List, Tuple

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.ensemble.epistemic_arbiter import kos_telemetry


class CausalTriggerMap:
    """
    Implements Yaroslavtsev's Interacting Theory (Section 5.7).

    Expands primitive axioms into ACID-compliant causal cascades of
    deductive reasoning.  Rules are registered as (condition, inference)
    pairs.  When an axiom matching the condition is minted, the inference
    function produces consequent axioms that are automatically locked
    into the state.
    """

    def __init__(self, algebra: GodelStateAlgebra, ledger: AkashicLedger):
        self.algebra = algebra
        self.ledger = ledger
        self.rules: List[Tuple[Callable, Callable]] = []

    def register_rule(self, condition_func: Callable, inference_func: Callable):
        """
        Registers an ontological rule (e.g., Transitivity, Syllogisms).

        Args:
            condition_func:  (s, p, o, state, algebra) -> bool
            inference_func:  (s, p, o, state, algebra) -> List[(s, p, o)]
        """
        self.rules.append((condition_func, inference_func))

    async def apply_cascade(
        self, current_state: int, delta_axioms: List[str]
    ) -> int:
        """
        Cascades inferences recursively until the semantic state
        reaches equilibrium.

        O(1) Idempotency is guaranteed by Gödel LCM arithmetic —
        re-minting an existing prime changes nothing.

        Args:
            current_state: The current Gödel integer.
            delta_axioms:  Newly added axiom keys (``s||p||o`` format).

        Returns:
            The updated Gödel integer with all cascaded inferences.
        """
        new_state = current_state
        cascade_queue = list(delta_axioms)
        visited: set = set()

        while cascade_queue:
            current_axiom = cascade_queue.pop(0)
            if current_axiom in visited:
                continue
            visited.add(current_axiom)

            parts = current_axiom.split("||")
            if len(parts) != 3:
                continue
            s, p, o = parts

            for condition, infer in self.rules:
                if condition(s, p, o, new_state, self.algebra):
                    inferred_triplets = infer(s, p, o, new_state, self.algebra)

                    for inf_s, inf_p, inf_o in inferred_triplets:
                        inf_axiom = f"{inf_s}||{inf_p}||{inf_o}"
                        inf_prime = self.algebra.get_or_mint_prime(
                            inf_s, inf_p, inf_o
                        )

                        # Novel deductive leap — not yet in state
                        if new_state % inf_prime != 0:
                            await kos_telemetry.broadcast(
                                f"🔗 Causal Inference: "
                                f"[{current_axiom}] ⟹ [{inf_axiom}]"
                            )

                            # Lock the inference into the state
                            new_state = math.lcm(new_state, inf_prime)

                            # Persist to Akashic Ledger
                            await self.ledger.append_event(
                                "MINT", inf_prime, inf_axiom
                            )
                            await self.ledger.append_event("MUL", inf_prime)

                            # Push to queue for further domino effects
                            cascade_queue.append(inf_axiom)

        return new_state
