"""
Akashic Ledger — Event-Sourced Fidelity Persistence

Implements Yaroslavtsev's Fidelity Axiom (§7.1):  RAM holds the massive
Gödel BigInt, while disk holds O(1) append-only mathematical operations
(traces) that can reconstruct it exactly after a crash.

Operations:
    MINT — a new prime was assigned to a semantic axiom
    MUL  — a prime was multiplied into the global state (LCM)
    DIV  — a prime was divided out of the global state (deletion)

Author: ototao
License: Apache License 2.0
"""

import math
import sqlite3
import asyncio
import logging



from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

logger = logging.getLogger(__name__)


class AkashicLedger:
    """
    Crash-safe persistence for the Gödel state via event sourcing.

    Instead of serialising a million-digit integer to disk on every
    change, we append a single O(1) mathematical trace (``MINT``,
    ``MUL``, ``DIV``).  The exact BigInt can be rebuilt at any time by
    replaying the trace in order.
    """

    def __init__(self, db_path: str = "akashic.db"):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the event table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_events (
                    seq_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation   TEXT    NOT NULL,   -- 'MINT', 'MUL', 'DIV'
                    prime       TEXT    NOT NULL,   -- Stored as TEXT for arbitrary-precision
                    axiom_key   TEXT
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    async def append_event(
        self, operation: str, prime: int, axiom_key: str = ""
    ) -> None:
        """
        Append a single mathematical trace to the ledger.

        Args:
            operation: One of ``'MINT'``, ``'MUL'``, ``'DIV'``.
            prime:     The semantic prime involved.
            axiom_key: Human-readable axiom key (for ``MINT`` events).
        """
        def _write():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO semantic_events "
                    "(operation, prime, axiom_key) VALUES (?, ?, ?)",
                    (operation, str(prime), axiom_key),
                )

        await asyncio.to_thread(_write)
        logger.debug("Ledger ← %s prime=%s axiom=%s", operation, prime, axiom_key)

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    async def rebuild_state(self, algebra: GodelStateAlgebra) -> int:
        """
        Replay the full event trace to reconstruct:
          1. The ``axiom_to_prime`` / ``prime_to_axiom`` mappings.
          2. The exact global Gödel BigInt.

        Primes are now deterministic (SHA-256 seeded), so no sequential
        watermark needs to be tracked.

        Args:
            algebra: A **fresh** GodelStateAlgebra to populate.

        Returns:
            The reconstructed global state integer.
        """
        def _read():
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute(
                    "SELECT operation, prime, axiom_key "
                    "FROM semantic_events ORDER BY seq_id ASC"
                ).fetchall()

        events = await asyncio.to_thread(_read)

        global_state = 1

        for op, prime_str, axiom in events:
            prime = int(prime_str)
            if op == "MINT":
                algebra.axiom_to_prime[axiom] = prime
                algebra.prime_to_axiom[prime] = axiom
            elif op == "MUL":
                global_state = math.lcm(global_state, prime)
            elif op == "DIV":
                while global_state % prime == 0:
                    global_state //= prime

        logger.info(
            "Akashic recovery complete: %d events replayed, state bit-length=%d",
            len(events),
            global_state.bit_length(),
        )
        return global_state
