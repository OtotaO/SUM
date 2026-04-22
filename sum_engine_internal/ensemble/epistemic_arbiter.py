"""
Epistemic Arbiter & Event Broadcaster — Wave Function Collapse Engine

The Arbiter resolves Level 3 Curvature (semantic contradictions) by
invoking an LLM judge to determine which conflicting fact survives.
The EventBroadcaster streams the internal "thinking" process to the
frontend via Server-Sent Events (SSE).

Author: ototao
License: Apache License 2.0
"""

import asyncio
import logging
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Streams internal mathematical thoughts to the frontend via SSE."""

    def __init__(self):
        self.queues: List[asyncio.Queue] = []

    async def broadcast(self, message: str):
        """Push a message to all connected SSE subscribers."""
        logger.info("[KOS Telemetry] %s", message)
        for queue in self.queues:
            await queue.put(message)

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscriber queue."""
        q: asyncio.Queue = asyncio.Queue()
        self.queues.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        if queue in self.queues:
            self.queues.remove(queue)


# Global singleton — imported by quantum_router for SSE streaming
kos_telemetry = EventBroadcaster()


class EpistemicArbiter:
    """
    Resolves Level 3 Curvature (Semantic Contradictions).

    Collapses the wave function of conflicting primes into
    absolute truth by invoking an LLM judge.
    """

    def __init__(self, llm_judge: Callable):
        self.judge = llm_judge  # async func(prompt: str) -> str

    async def collapse_wave_function(
        self, conflicts: List[Tuple[str, str, str, str]]
    ) -> Dict[Tuple[str, str], str]:
        """
        Takes a list of conflicts: (subject, predicate, object_a, object_b).
        Returns the winning mapping: {(subject, predicate): winning_object}.
        """
        resolutions: Dict[Tuple[str, str], str] = {}

        for subject, predicate, obj_a, obj_b in conflicts:
            await kos_telemetry.broadcast(
                f"⚠️ Level 3 Curvature Detected: "
                f"{subject} {predicate} [{obj_a} OR {obj_b}]"
            )
            await kos_telemetry.broadcast(
                "🌀 Entering Epistemic Superposition..."
            )

            prompt = (
                f"You are a strict logic arbiter. We have a contradiction "
                f"regarding '{subject}'.\n"
                f"Claim A: {subject} {predicate} {obj_a}\n"
                f"Claim B: {subject} {predicate} {obj_b}\n"
                f"Analyze standard logical precedence, general knowledge, "
                f"or temporal recency, and return ONLY the correct object "
                f"value. If tied, pick the most specific."
            )

            # Call the LLM Judge
            winner = await self.judge(prompt)
            winner_clean = winner.strip().lower()

            # Fallback if the judge hallucinates an entirely new answer
            if winner_clean not in [obj_a.lower(), obj_b.lower()]:
                winner_clean = obj_a.lower()

            resolutions[(subject, predicate)] = winner_clean
            await kos_telemetry.broadcast(
                f"⚡ Wave Function Collapsed: "
                f"{subject} {predicate} → {winner_clean}"
            )

        return resolutions


class DeterministicArbiter:
    """
    Deterministic contradiction resolution without LLM.

    Resolves Level 3 Curvature using SHA-256 lexicographic ordering:
    for each conflict (subject, predicate, obj_a, obj_b), the winner
    is whichever object has the lower SHA-256 hash of
    ``f"{subject}||{predicate}||{object}"``.

    This guarantees:
      - Identical resolution on every node (deterministic)
      - No LLM cost or latency
      - Consistent ordering regardless of minting order
      - Reproducibility across runtimes (SHA-256 is universal)
    """

    @staticmethod
    def _canonical_hash(subject: str, predicate: str, obj: str) -> str:
        """SHA-256 of the canonical triplet key."""
        import hashlib
        return hashlib.sha256(
            f"{subject}||{predicate}||{obj}".encode()
        ).hexdigest()

    async def collapse_wave_function(
        self, conflicts: List[Tuple[str, str, str, str]]
    ) -> Dict[Tuple[str, str], str]:
        """
        Resolve conflicts deterministically via SHA-256 ordering.

        For each (subject, predicate, obj_a, obj_b), the object with
        the lexicographically lower SHA-256 hash wins.
        """
        resolutions: Dict[Tuple[str, str], str] = {}

        for subject, predicate, obj_a, obj_b in conflicts:
            hash_a = self._canonical_hash(subject, predicate, obj_a)
            hash_b = self._canonical_hash(subject, predicate, obj_b)

            winner = obj_a if hash_a <= hash_b else obj_b

            await kos_telemetry.broadcast(
                f"⚠️ Level 3 Curvature: "
                f"{subject} {predicate} [{obj_a} OR {obj_b}]"
            )
            await kos_telemetry.broadcast(
                f"🔬 Deterministic Resolution: SHA-256({obj_a})={hash_a[:8]}… "
                f"vs SHA-256({obj_b})={hash_b[:8]}…"
            )
            await kos_telemetry.broadcast(
                f"⚡ Collapsed → {subject} {predicate} → {winner}"
            )

            resolutions[(subject, predicate)] = winner

        return resolutions
