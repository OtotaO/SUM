"""
Akashic Ledger — Event-Sourced Fidelity Persistence

Implements Yaroslavtsev's Fidelity Axiom (§7.1):  RAM holds the massive
Gödel BigInt, while disk holds O(1) append-only mathematical operations
(traces) that can reconstruct it exactly after a crash.

Operations:
    MINT — a new prime was assigned to a semantic axiom
    MUL  — a prime was multiplied into the global state (LCM)
    DIV  — a prime was divided out of the global state (deletion)

The Chronos Engine (Phase 10) adds time-travel capability by allowing
historical state rebuilds up to any specific ledger tick (``max_seq_id``).

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

    The Chronos Engine enables O(1) Time Travel by replaying only up
    to a given ``max_seq_id``, reconstructing the universe as it
    existed at any historical tick.
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
    # Crash recovery & Time Travel (Chronos Engine)
    # ------------------------------------------------------------------

    async def rebuild_state(
        self, algebra: GodelStateAlgebra, max_seq_id: int = None
    ) -> int:
        """
        Replay the event trace to reconstruct the Gödel BigInt.

        When ``max_seq_id`` is provided, only events up to that tick
        are replayed, enabling **O(1) Time Travel** — the exact state
        of knowledge at any historical moment can be reconstructed
        into an alternate timeline branch.

        Primes are deterministic (SHA-256 seeded), so no sequential
        watermark needs to be tracked.

        Args:
            algebra:    A GodelStateAlgebra to populate.
            max_seq_id: Optional tick limit for time-travel rebuilds.

        Returns:
            The reconstructed global state integer.
        """
        def _read():
            with sqlite3.connect(self.db_path) as conn:
                query = (
                    "SELECT seq_id, operation, prime, axiom_key "
                    "FROM semantic_events"
                )
                params = []
                if max_seq_id is not None:
                    query += " WHERE seq_id <= ?"
                    params.append(max_seq_id)
                query += " ORDER BY seq_id ASC"
                return conn.execute(query, params).fetchall()

        events = await asyncio.to_thread(_read)

        global_state = 1

        for seq_id, op, prime_str, axiom in events:
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

    # ------------------------------------------------------------------
    # Chronos Engine — Tick Query
    # ------------------------------------------------------------------

    async def get_latest_tick(self) -> int:
        """
        Return the latest sequence ID (tick) in the ledger.

        This represents the current "time" of the knowledge universe
        and can be used as a bookmark for time-travel operations.

        Returns:
            The highest ``seq_id``, or 0 if the ledger is empty.
        """
        def _read():
            with sqlite3.connect(self.db_path) as conn:
                res = conn.execute(
                    "SELECT MAX(seq_id) FROM semantic_events"
                ).fetchone()
                return res[0] if res[0] else 0

        return await asyncio.to_thread(_read)
