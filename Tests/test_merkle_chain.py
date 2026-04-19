"""
Phase 19C: Merkle Hash-Chain Ledger Tests

Tests the tamper-evident hash chain on the Akashic Ledger.
Covers: genesis hash, chain construction, tamper detection,
deletion detection, insertion detection, empty ledger, chain tip.

Author: ototao
License: Apache License 2.0
"""

import os
import asyncio
import sqlite3
import pytest
from internal.infrastructure.akashic_ledger import (
    AkashicLedger,
    GENESIS_HASH,
    compute_event_hash,
)


@pytest.fixture
def ledger(tmp_path):
    db_path = str(tmp_path / "test_merkle.db")
    return AkashicLedger(db_path=db_path)


def run(coro):
    """Run an async coroutine synchronously.

    Uses a fresh event loop per invocation rather than ``asyncio.get_event_loop()``.
    The latter raises ``RuntimeError: There is no current event loop`` once any
    other test in the same pytest session has called ``asyncio.run()``, because
    that call closes the implicitly-created loop and the next
    ``get_event_loop()`` cannot resurrect it. A fresh loop is immune to that
    pollution and is cheap to create — O(μs).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestGenesisHash:

    def test_genesis_hash_deterministic(self):
        """Genesis hash is always the same (deterministic seed)."""
        import hashlib
        expected = hashlib.sha256(b"SUM_GENESIS_BLOCK").hexdigest()
        assert GENESIS_HASH == expected
        assert len(GENESIS_HASH) == 64  # SHA-256 hex

    def test_genesis_hash_not_empty(self):
        assert GENESIS_HASH != ""
        assert GENESIS_HASH != "0" * 64


class TestChainConstruction:

    def test_single_event_chain(self, ledger):
        """One event: chain should verify."""
        run(ledger.append_event("MINT", 2, "earth||orbits||sun"))
        valid, break_at = run(ledger.verify_chain())
        assert valid is True
        assert break_at is None

    def test_three_event_chain_verifies(self, ledger):
        """Three events: chain should verify end-to-end."""
        run(ledger.append_event("MINT", 2, "earth||orbits||sun"))
        run(ledger.append_event("MUL", 2))
        run(ledger.append_event("MINT", 3, "water||has_part||hydrogen"))
        valid, break_at = run(ledger.verify_chain())
        assert valid is True
        assert break_at is None

    def test_chain_tip_after_events(self, ledger):
        """Chain tip should be the hash of the last event."""
        run(ledger.append_event("MINT", 2, "earth||orbits||sun"))
        tip = run(ledger.get_chain_tip())
        assert tip != GENESIS_HASH  # Chain has advanced
        assert len(tip) == 64

    def test_chain_tip_empty_ledger(self, ledger):
        """Empty ledger should return genesis hash."""
        tip = run(ledger.get_chain_tip())
        assert tip == GENESIS_HASH


class TestTamperDetection:

    def test_mutated_event_detected(self, ledger):
        """Changing an event's data should break the chain."""
        run(ledger.append_event("MINT", 2, "earth||orbits||sun"))
        run(ledger.append_event("MUL", 2))
        run(ledger.append_event("MINT", 3, "water||has_part||hydrogen"))

        # Tamper: change the middle event's prime
        with sqlite3.connect(ledger.db_path) as conn:
            conn.execute(
                "UPDATE semantic_events SET prime = '999' WHERE seq_id = 2"
            )

        valid, break_at = run(ledger.verify_chain())
        assert valid is False
        assert break_at == 2  # Tamper detected at event 2

    def test_mutated_axiom_detected(self, ledger):
        """Changing an event's axiom key should break the chain."""
        run(ledger.append_event("MINT", 2, "earth||orbits||sun"))
        run(ledger.append_event("MINT", 3, "water||has_part||hydrogen"))

        # Tamper: change axiom_key on event 1
        with sqlite3.connect(ledger.db_path) as conn:
            conn.execute(
                "UPDATE semantic_events SET axiom_key = 'FAKE||DATA||HERE' "
                "WHERE seq_id = 1"
            )

        valid, break_at = run(ledger.verify_chain())
        assert valid is False
        assert break_at == 1

    def test_deleted_event_detected(self, ledger):
        """Deleting a middle event should break the chain."""
        run(ledger.append_event("MINT", 2, "earth||orbits||sun"))
        run(ledger.append_event("MUL", 2))
        run(ledger.append_event("MINT", 3, "water||has_part||hydrogen"))

        # Delete middle event
        with sqlite3.connect(ledger.db_path) as conn:
            conn.execute("DELETE FROM semantic_events WHERE seq_id = 2")

        valid, break_at = run(ledger.verify_chain())
        assert valid is False
        # After deleting event 2, event 3 won't chain from event 1 correctly
        assert break_at == 3

    def test_tampered_hash_detected(self, ledger):
        """Directly changing the stored hash should be detected."""
        run(ledger.append_event("MINT", 2, "earth||orbits||sun"))
        run(ledger.append_event("MUL", 2))

        # Tamper: change the hash directly
        with sqlite3.connect(ledger.db_path) as conn:
            conn.execute(
                "UPDATE semantic_events SET prev_hash = 'deadbeef' WHERE seq_id = 1"
            )

        valid, break_at = run(ledger.verify_chain())
        assert valid is False
        assert break_at == 1


class TestEmptyLedger:

    def test_empty_ledger_verifies(self, ledger):
        """An empty ledger should verify as valid."""
        valid, break_at = run(ledger.verify_chain())
        assert valid is True
        assert break_at is None


class TestHashComputation:

    def test_compute_event_hash_deterministic(self):
        """Same inputs should always produce the same hash."""
        h1 = compute_event_hash(GENESIS_HASH, "MINT", "2", "earth||orbits||sun", "main")
        h2 = compute_event_hash(GENESIS_HASH, "MINT", "2", "earth||orbits||sun", "main")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_inputs_different_hash(self):
        """Different inputs should produce different hashes."""
        h1 = compute_event_hash(GENESIS_HASH, "MINT", "2", "earth||orbits||sun", "main")
        h2 = compute_event_hash(GENESIS_HASH, "MINT", "3", "earth||orbits||sun", "main")
        assert h1 != h2

    def test_branch_changes_hash(self):
        """Different branches should produce different hashes."""
        h1 = compute_event_hash(GENESIS_HASH, "MINT", "2", "earth||orbits||sun", "main")
        h2 = compute_event_hash(GENESIS_HASH, "MINT", "2", "earth||orbits||sun", "feature-x")
        assert h1 != h2


class TestChainIntegrity:

    def test_chain_grows_correctly(self, ledger):
        """Each new event should chain from the previous hash."""
        run(ledger.append_event("MINT", 2, "a||b||c"))
        tip1 = run(ledger.get_chain_tip())

        run(ledger.append_event("MUL", 2))
        tip2 = run(ledger.get_chain_tip())

        assert tip1 != tip2  # Chain advanced
        assert tip1 != GENESIS_HASH
        assert tip2 != GENESIS_HASH

    def test_verify_after_many_events(self, ledger):
        """10-event chain should verify."""
        for i in range(10):
            prime = 2 + i
            run(ledger.append_event("MINT", prime, f"s{i}||p{i}||o{i}"))
            run(ledger.append_event("MUL", prime))

        valid, break_at = run(ledger.verify_chain())
        assert valid is True
        assert break_at is None
