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

Phase 0 Durability Contract:
    All branch state changes are now persisted via:
    - ``branch`` column on events for branch-scoped replay
    - ``branch_heads`` snapshot table for instant boot
    Model C (Hybrid): event log = source of truth, snapshots = fast cache.

Phase 19C Merkle Hash-Chain:
    Each event stores ``prev_hash = SHA-256(prev_hash + payload)``.
    The chain is verified on boot for tamper detection.
    Genesis seed: SHA-256("SUM_GENESIS_BLOCK").

Author: ototao
License: Apache License 2.0
"""

import math
import hashlib
import json
import sqlite3
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.provenance import (
    ProvenanceRecord,
    compute_prov_id,
)

logger = logging.getLogger(__name__)

# Phase 19C: Merkle Hash-Chain
GENESIS_HASH = hashlib.sha256(b"SUM_GENESIS_BLOCK").hexdigest()


def compute_event_hash(prev_hash: str, operation: str, prime: str,
                       axiom_key: str, branch: str) -> str:
    """Compute SHA-256 hash for an event in the Merkle chain."""
    payload = f"{prev_hash}|{operation}|{prime}|{axiom_key}|{branch}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _zig() -> Optional[Any]:
    """Return the Zig engine module if available on this host, else None.

    Typed as ``Any`` because ``zig_bridge.zig_engine`` is a C-FFI-backed
    object with attributes defined at runtime; pinning a stricter type
    would require stubs for the Zig core. Callers gate on truthiness.
    """
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
            # Phase 0: Branch head snapshot table for instant boot
            conn.execute("""
                CREATE TABLE IF NOT EXISTS branch_heads (
                    branch_name   TEXT PRIMARY KEY,
                    state_integer TEXT NOT NULL DEFAULT '1',
                    last_event_id INTEGER,
                    is_ephemeral  INTEGER DEFAULT 0,
                    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            # Phase 22: Provenance + Confidence migration
            self._migrate_provenance(conn)
            # Phase 0: Branch column migration
            self._migrate_branch_column(conn)
            # Phase 19C: Merkle hash-chain migration
            self._migrate_hash_chain(conn)
            # M1: Structured ProvenanceRecord side-table
            self._migrate_structured_provenance(conn)

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

    def _migrate_branch_column(self, conn: sqlite3.Connection) -> None:
        """Phase 0: Idempotent migration — add branch column if missing."""
        try:
            conn.execute(
                "ALTER TABLE semantic_events ADD COLUMN branch TEXT DEFAULT 'main'"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    def _migrate_hash_chain(self, conn: sqlite3.Connection) -> None:
        """Phase 19C: Idempotent migration — add prev_hash column if missing."""
        try:
            conn.execute(
                "ALTER TABLE semantic_events ADD COLUMN prev_hash TEXT DEFAULT ''"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    def _migrate_structured_provenance(self, conn: sqlite3.Connection) -> None:
        """M1: Structured ProvenanceRecord side-table + axiom linking.

        Normalises evidence into a content-addressable table keyed by
        ``prov_id``. The legacy flat columns (``source_url``, ``confidence``,
        ``ingested_at``) stay in place so existing callers and PROV-O
        emission are untouched; new provenance-aware code writes to both.

        Tables:
          provenance_records(prov_id PRIMARY KEY, record_json TEXT) — one
            row per unique evidence span. Dedup is automatic via PRIMARY
            KEY because prov_id is content-addressable.
          axiom_provenance(axiom_key, prov_id, PRIMARY KEY(axiom_key, prov_id))
            — many-to-many. An axiom extracted from multiple sources has
            one entry per source. Queryable both directions.

        Idempotent — safe to run on an existing DB.
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance_records (
                prov_id     TEXT PRIMARY KEY,
                record_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS axiom_provenance (
                axiom_key TEXT NOT NULL,
                prov_id   TEXT NOT NULL,
                PRIMARY KEY (axiom_key, prov_id),
                FOREIGN KEY (prov_id) REFERENCES provenance_records(prov_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_axiom_provenance_prov "
            "ON axiom_provenance(prov_id)"
        )
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
        branch: str = "main",
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
            branch:      Branch this event belongs to (default: 'main').
            source_url:  Origin URL or identifier for this axiom.
            confidence:  Trust score in [0.0, 1.0], default 0.5.
            ingested_at: ISO timestamp (YYYY-MM-DDTHH:MM:SS).
        """
        def _write() -> None:
            with sqlite3.connect(self.db_path) as conn:
                # Merkle-chain integrity under concurrent writers requires
                # that the SELECT of the previous prev_hash and the INSERT
                # of the new event happen atomically under a reserved
                # write-lock. Python's sqlite3 module defaults to
                # autocommit for SELECTs (the transaction begins only
                # on the first INSERT/UPDATE/DELETE), which means two
                # concurrent writers can both observe the SAME prev_hash
                # before either commits — both then chain on stale state
                # and verify_chain() subsequently reports divergence.
                # BEGIN IMMEDIATE acquires the reserved lock NOW, queuing
                # other writers at the SQLite boundary until this
                # transaction commits. Verified by
                # Tests/test_ledger_concurrency.py.
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT prev_hash FROM semantic_events "
                    "ORDER BY seq_id DESC LIMIT 1"
                ).fetchone()
                prev_hash = row[0] if row and row[0] else GENESIS_HASH
                event_hash = compute_event_hash(
                    prev_hash, operation, str(prime), axiom_key, branch
                )

                conn.execute(
                    "INSERT INTO semantic_events "
                    "(operation, prime, axiom_key, branch, source_url, "
                    "confidence, ingested_at, prev_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (operation, str(prime), axiom_key, branch,
                     source_url, confidence, ingested_at, event_hash),
                )

        await asyncio.to_thread(_write)
        logger.debug(
            "Ledger ← %s prime=%s axiom=%s branch=%s source=%s conf=%.2f",
            operation, prime, axiom_key, branch, source_url, confidence,
        )

    # ------------------------------------------------------------------
    # Structured provenance (M1)
    # ------------------------------------------------------------------

    async def record_provenance(
        self, provenance: ProvenanceRecord, axiom_key: str
    ) -> str:
        """Persist a structured ProvenanceRecord and link it to an axiom.

        Returns the content-addressable ``prov_id``. Safe to call repeatedly
        with identical (record, axiom_key) pairs — both underlying inserts
        are idempotent (``INSERT OR IGNORE``), so the same evidence span is
        never double-counted regardless of ingestion retries.

        Args:
            provenance: A validated ProvenanceRecord. Its shape is the
                authoritative evidence unit; see provenance.py.
            axiom_key:  The normalised axiom string (``subject||predicate||object``)
                whose prime this evidence attests.
        """
        prov_id = compute_prov_id(provenance)
        record_json = json.dumps(
            provenance.to_dict(), separators=(",", ":"), sort_keys=True
        )

        def _write() -> None:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO provenance_records "
                    "(prov_id, record_json) VALUES (?, ?)",
                    (prov_id, record_json),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO axiom_provenance "
                    "(axiom_key, prov_id) VALUES (?, ?)",
                    (axiom_key, prov_id),
                )

        await asyncio.to_thread(_write)
        logger.debug(
            "Ledger ← provenance axiom=%s prov_id=%s extractor=%s",
            axiom_key, prov_id, provenance.extractor_id,
        )
        return prov_id

    async def get_provenance_record(
        self, prov_id: str
    ) -> Optional[ProvenanceRecord]:
        """Fetch a single ProvenanceRecord by its content-addressable id.

        Returns None if the id is unknown. The returned record re-validates
        on construction via ``ProvenanceRecord.__post_init__``, so any
        corrupted row in the DB will raise rather than silently return bad
        evidence.
        """
        def _read() -> Optional[str]:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT record_json FROM provenance_records WHERE prov_id = ?",
                    (prov_id,),
                ).fetchone()
                return row[0] if row else None

        record_json = await asyncio.to_thread(_read)
        if record_json is None:
            return None
        payload = json.loads(record_json)
        return ProvenanceRecord(**payload)

    async def get_structured_provenance_for_axiom(
        self, axiom_key: str
    ) -> list[ProvenanceRecord]:
        """Return every ProvenanceRecord linked to ``axiom_key``.

        Order: ascending prov_id (stable across runs; content-addressed).
        An axiom extracted from the same span by the same extractor at the
        same instant with the same excerpt collapses to one row by
        construction — no duplicate evidence ever appears in the result.
        """
        def _read() -> list[str]:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT pr.record_json FROM axiom_provenance ap "
                    "JOIN provenance_records pr ON ap.prov_id = pr.prov_id "
                    "WHERE ap.axiom_key = ? "
                    "ORDER BY ap.prov_id ASC",
                    (axiom_key,),
                ).fetchall()
                return [r[0] for r in rows]

        rows = await asyncio.to_thread(_read)
        return [ProvenanceRecord(**json.loads(j)) for j in rows]

    # ------------------------------------------------------------------
    # Provenance queries (Phase 22 — flat-column legacy surface)
    # ------------------------------------------------------------------

    async def get_axiom_provenance(
        self, axiom_key: str
    ) -> List[Dict[str, Any]]:
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
        def _read() -> List[Dict[str, Any]]:
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

    async def get_provenance_batch(
        self, axiom_keys: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch provenance lookup for multiple axioms.

        Returns the *latest* (highest-confidence, most-recent) provenance
        record for each axiom, keyed by axiom_key.

        Args:
            axiom_keys: List of axiom key strings.

        Returns:
            Dict mapping axiom_key → provenance dict. Axioms with no
            recorded provenance are omitted from the result (not mapped
            to ``None``).
        """
        if not axiom_keys:
            return {}

        def _read() -> Dict[str, Dict[str, Any]]:
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
                result: Dict[str, Dict[str, Any]] = {}
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
    # PROV-O Export (Polytaxis Bucket A §4)
    # ------------------------------------------------------------------

    async def to_prov_jsonld(
        self, branch: str = "main", graph_iri: str = "urn:sum:audit"
    ) -> str:
        """Serialize every event on a branch as a W3C PROV-O JSON-LD graph.

        Convenience wrapper that joins the ledger's event table with the
        stateless PROV-O adapter in ``internal.infrastructure.prov_o``.
        The returned string is a valid JSON-LD document any PROV-compliant
        tool can consume without SUM-specific knowledge.

        Args:
            branch:    Branch to export (default 'main').
            graph_iri: IRI for the outer named graph.

        Returns:
            A pretty-printed JSON-LD document string.
        """
        from internal.infrastructure.prov_o import dump_prov_jsonld

        def _read() -> List[Dict[str, Any]]:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT seq_id, operation, prime, axiom_key, branch, "
                    "source_url, confidence, ingested_at, prev_hash "
                    "FROM semantic_events WHERE branch = ? "
                    "ORDER BY seq_id ASC",
                    (branch,),
                ).fetchall()
                events: List[Dict[str, Any]] = []
                prev_seq_id: Optional[int] = None
                for r in rows:
                    event: Dict[str, Any] = {
                        "seq_id": r[0],
                        "operation": r[1],
                        "prime": r[2],
                        "axiom_key": r[3] or "",
                        "branch": r[4] or "main",
                        "source_url": r[5] or "",
                        "confidence": r[6],
                        "ingested_at": r[7] or "",
                        "prev_hash": r[8] or "",
                    }
                    if prev_seq_id is not None:
                        event["prev_seq_id"] = prev_seq_id
                    prev_seq_id = r[0]
                    events.append(event)
                return events

        events = await asyncio.to_thread(_read)
        return dump_prov_jsonld(events, graph_iri=graph_iri)

    # ------------------------------------------------------------------
    # Branch Head Snapshots (Phase 0: Durability Contract)
    # ------------------------------------------------------------------

    async def save_branch_head(
        self,
        branch_name: str,
        state_integer: int,
        is_ephemeral: bool = False,
    ) -> None:
        """
        Upsert a branch head snapshot for instant boot recovery.

        Args:
            branch_name:    Name of the branch.
            state_integer:  Current Gödel BigInt for this branch.
            is_ephemeral:   If True, branch is transient (e.g. time-travel)
                            and will NOT be restored on boot.
        """
        def _write() -> None:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO branch_heads "
                    "(branch_name, state_integer, is_ephemeral) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(branch_name) DO UPDATE SET "
                    "state_integer = excluded.state_integer, "
                    "is_ephemeral = excluded.is_ephemeral",
                    (branch_name, str(state_integer), int(is_ephemeral)),
                )

        await asyncio.to_thread(_write)
        logger.debug("Branch head saved: %s (ephemeral=%s)", branch_name, is_ephemeral)

    async def load_branch_heads(self) -> Dict[str, int]:
        """
        Load all durable (non-ephemeral) branch head snapshots.

        Returns:
            Dict mapping branch_name → state integer.
        """
        def _read() -> Dict[str, int]:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT branch_name, state_integer FROM branch_heads "
                    "WHERE is_ephemeral = 0"
                ).fetchall()
                return {name: int(state) for name, state in rows}

        return await asyncio.to_thread(_read)

    async def delete_branch_head(self, branch_name: str) -> None:
        """Remove a branch head snapshot (e.g. cleanup of ephemeral branches)."""
        def _write() -> None:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM branch_heads WHERE branch_name = ?",
                    (branch_name,),
                )

        await asyncio.to_thread(_write)
        logger.debug("Branch head deleted: %s", branch_name)

    # ------------------------------------------------------------------
    # Crash recovery & Time Travel (Chronos Engine)
    # ------------------------------------------------------------------

    async def rebuild_state(
        self, algebra: GodelStateAlgebra,
        max_seq_id: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> int:
        """
        Replay the event trace to reconstruct the Gödel BigInt.

        When ``max_seq_id`` is provided, only events up to that tick
        are replayed, enabling **Time Travel** — the exact state
        of knowledge at any historical moment can be reconstructed
        into an alternate timeline branch.

        When ``branch`` is provided, only events for that branch are
        replayed (Phase 0 Durability Contract).

        Primes are deterministic (SHA-256 seeded), so no sequential
        watermark needs to be tracked.

        Args:
            algebra:    A GodelStateAlgebra to populate.
            max_seq_id: Optional tick limit for time-travel rebuilds.
            branch:     Optional branch filter (default: replay all).

        Returns:
            The reconstructed global state integer.
        """
        def _read() -> List[Tuple[Any, ...]]:
            with sqlite3.connect(self.db_path) as conn:
                query = (
                    "SELECT seq_id, operation, prime, axiom_key "
                    "FROM semantic_events"
                )
                conditions: List[str] = []
                # SQLite accepts int or str bind parameters interchangeably, so
                # params is heterogeneous by design — mypy's naive inference
                # from the first append would lock it to list[int] and then
                # flag the str append below. Annotate explicitly.
                params: List[Any] = []
                if max_seq_id is not None:
                    conditions.append("seq_id <= ?")
                    params.append(max_seq_id)
                if branch is not None:
                    conditions.append("(branch = ? OR branch = 'main' OR branch IS NULL)")
                    params.append(branch)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY seq_id ASC"
                return list(conn.execute(query, params).fetchall())

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
        def _read() -> int:
            with sqlite3.connect(self.db_path) as conn:
                res = conn.execute(
                    "SELECT MAX(seq_id) FROM semantic_events"
                ).fetchone()
                return int(res[0]) if res and res[0] else 0

        return await asyncio.to_thread(_read)

    # ------------------------------------------------------------------
    # Phase 19C: Merkle Hash-Chain Verification
    # ------------------------------------------------------------------

    async def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """
        Walk the entire event hash chain and verify integrity.

        Returns:
            (is_valid: bool, break_seq_id: int | None)
            If valid, break_seq_id is None.
            If tampered, break_seq_id is the first corrupted event.
        """
        def _verify() -> Tuple[bool, Optional[int]]:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT seq_id, operation, prime, axiom_key, branch, prev_hash "
                    "FROM semantic_events ORDER BY seq_id ASC"
                ).fetchall()

            if not rows:
                return (True, None)

            prev_hash = GENESIS_HASH
            for seq_id, operation, prime, axiom_key, branch, stored_hash in rows:
                expected = compute_event_hash(
                    prev_hash, operation, prime, axiom_key or "", branch or "main"
                )
                if stored_hash != expected:
                    return (False, int(seq_id))
                prev_hash = stored_hash

            return (True, None)

        return await asyncio.to_thread(_verify)

    async def get_chain_tip(self) -> str:
        """
        Return the hash of the most recent event (chain tip).

        Useful for sync protocols and integrity checks.

        Returns:
            The latest prev_hash, or GENESIS_HASH if the ledger is empty.
        """
        def _read() -> str:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT prev_hash FROM semantic_events "
                    "ORDER BY seq_id DESC LIMIT 1"
                ).fetchone()
                return str(row[0]) if row and row[0] else GENESIS_HASH

        return await asyncio.to_thread(_read)
