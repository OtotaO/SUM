"""
Akashic Ledger — Event-Sourced Fidelity Persistence

Implements Yaroslavtsev's Fidelity Axiom (§7.1):  RAM holds the massive
Gödel BigInt, while disk holds append-only mathematical operations
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


def _zig():
    try:
        from internal.infrastructure.zig_bridge import zig_engine
        return zig_engine
    except ImportError:
        return None


class AkashicLedger:
    """
    Crash-safe persistence for the Gödel state via event sourcing.

    Instead of serialising a million-digit integer to disk on every
    change, we append a single mathematical trace (``MINT``,
    ``MUL``, ``DIV``).  The exact BigInt can be rebuilt at any time by
    replaying the trace in order.

    The Chronos Engine enables Time Travel by replaying only up
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
        """Create the event table if it does not exist, then migrate."""
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
            # Phase 22: Provenance + Confidence migration
            self._migrate_provenance(conn)

    def _migrate_provenance(self, conn: sqlite3.Connection) -> None:
        """Idempotent migration: add provenance columns if missing."""
        columns_to_add = [
            ("source_url",  "TEXT DEFAULT ''"),
            ("confidence",  "REAL DEFAULT 0.5"),
            ("ingested_at", "TEXT DEFAULT ''"),
        ]
        for col_name, col_def in columns_to_add:
            try:
                conn.execute(
                    f"ALTER TABLE semantic_events ADD COLUMN {col_name} {col_def}"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — safe to ignore
        conn.commit()

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    async def append_event(
        self,
        operation: str,
        prime: int,
        axiom_key: str = "",
        *,
        source_url: str = "",
        confidence: float = 0.5,
        ingested_at: str = "",
    ) -> None:
        """
        Append a single mathematical trace to the ledger.

        Args:
            operation:   One of ``'MINT'``, ``'MUL'``, ``'DIV'``,
                         ``'SYNC'``, ``'DEDUCED'``.
            prime:       The semantic prime involved.
            axiom_key:   Human-readable axiom key (for ``MINT``/``DEDUCED`` events).
            source_url:  Origin URL or identifier for this axiom.
            confidence:  Trust score in [0.0, 1.0], default 0.5.
            ingested_at: ISO timestamp (YYYY-MM-DDTHH:MM:SS).
        """
        def _write():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO semantic_events "
                    "(operation, prime, axiom_key, source_url, confidence, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (operation, str(prime), axiom_key,
                     source_url, confidence, ingested_at),
                )

        await asyncio.to_thread(_write)
        logger.debug(
            "Ledger ← %s prime=%s axiom=%s source=%s conf=%.2f",
            operation, prime, axiom_key, source_url, confidence,
        )

    # ------------------------------------------------------------------
    # Provenance queries (Phase 22)
    # ------------------------------------------------------------------

    async def get_axiom_provenance(self, axiom_key: str) -> list:
        """
        Retrieve provenance metadata for a specific axiom.

        Queries all MINT events matching the axiom_key to build
        a full provenance chain (an axiom may be re-ingested from
        multiple sources).

        Args:
            axiom_key: The normalised axiom string (``subj||pred||obj``).

        Returns:
            List of dicts with ``source_url``, ``confidence``,
            ``ingested_at``, ``seq_id``.
        """
        def _read():
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT seq_id, source_url, confidence, ingested_at "
                    "FROM semantic_events "
                    "WHERE operation = 'MINT' AND axiom_key = ? "
                    "ORDER BY seq_id ASC",
                    (axiom_key,),
                ).fetchall()
                return [
                    {
                        "seq_id": r[0],
                        "source_url": r[1] or "",
                        "confidence": r[2] if r[2] is not None else 0.5,
                        "ingested_at": r[3] or "",
                    }
                    for r in rows
                ]

        return await asyncio.to_thread(_read)

    async def get_provenance_batch(self, axiom_keys: list) -> dict:
        """
        Batch provenance lookup for multiple axioms.

        Returns the *latest* (highest-confidence, most-recent) provenance
        record for each axiom, keyed by axiom_key.

        Args:
            axiom_keys: List of axiom key strings.

        Returns:
            Dict mapping axiom_key → provenance dict (or None if not found).
        """
        if not axiom_keys:
            return {}

        def _read():
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" for _ in axiom_keys)
                rows = conn.execute(
                    f"SELECT axiom_key, source_url, confidence, ingested_at "
                    f"FROM semantic_events "
                    f"WHERE operation = 'MINT' AND axiom_key IN ({placeholders}) "
                    f"ORDER BY seq_id DESC",
                    axiom_keys,
                ).fetchall()

                # Keep the latest record per axiom_key
                result = {}
                for axiom_key, source_url, confidence, ingested_at in rows:
                    if axiom_key not in result:
                        result[axiom_key] = {
                            "source_url": source_url or "",
                            "confidence": confidence if confidence is not None else 0.5,
                            "ingested_at": ingested_at or "",
                        }
                return result

        return await asyncio.to_thread(_read)

    # ------------------------------------------------------------------
    # Crash recovery & Time Travel (Chronos Engine)
    # ------------------------------------------------------------------

    async def rebuild_state(
        self, algebra: GodelStateAlgebra, max_seq_id: int = None
    ) -> int:
        """
        Replay the event trace to reconstruct the Gödel BigInt.

        When ``max_seq_id`` is provided, only events up to that tick
        are replayed, enabling **Time Travel** — the exact state
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
                z = _zig()
                r = z.bigint_lcm(global_state, prime) if z else None
                global_state = r if r is not None else math.lcm(global_state, prime)
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
