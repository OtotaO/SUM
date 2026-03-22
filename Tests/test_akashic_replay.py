"""
Akashic Ledger Replay Correctness Tests

Verifies that the event-sourced ledger correctly rebuilds the Gödel
state from append-only traces, including time-travel to intermediate
ticks and crash recovery.

Author: ototao
License: Apache License 2.0
"""

import math
import os
import tempfile
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger


@pytest.fixture
def ledger_env():
    """Provide a fresh ledger + algebra in a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_akashic.db")
        ledger = AkashicLedger(db_path=db_path)
        algebra = GodelStateAlgebra()
        yield ledger, algebra, db_path


class TestLedgerAppendAndReplay:

    @pytest.mark.asyncio
    async def test_mint_mul_replay_roundtrip(self, ledger_env):
        """Mint + MUL events → rebuild produces exact state."""
        ledger, algebra, _ = ledger_env

        p1 = algebra.get_or_mint_prime("alice", "likes", "cats")
        p2 = algebra.get_or_mint_prime("bob", "knows", "python")

        await ledger.append_event("MINT", p1, "alice||likes||cats")
        await ledger.append_event("MUL", p1)
        await ledger.append_event("MINT", p2, "bob||knows||python")
        await ledger.append_event("MUL", p2)

        expected_state = math.lcm(p1, p2)

        # Rebuild from scratch
        fresh_algebra = GodelStateAlgebra()
        rebuilt = await ledger.rebuild_state(fresh_algebra)
        assert rebuilt == expected_state

    @pytest.mark.asyncio
    async def test_div_event_removes_axiom(self, ledger_env):
        """DIV event removes a prime from the state."""
        ledger, algebra, _ = ledger_env

        p1 = algebra.get_or_mint_prime("fact", "to", "delete")
        p2 = algebra.get_or_mint_prime("fact", "to", "keep")

        await ledger.append_event("MINT", p1, "fact||to||delete")
        await ledger.append_event("MUL", p1)
        await ledger.append_event("MINT", p2, "fact||to||keep")
        await ledger.append_event("MUL", p2)
        await ledger.append_event("DIV", p1)

        fresh = GodelStateAlgebra()
        rebuilt = await ledger.rebuild_state(fresh)

        assert rebuilt % p2 == 0, "Kept axiom should be present"
        assert rebuilt % p1 != 0, "Deleted axiom should be absent"

    @pytest.mark.asyncio
    async def test_empty_ledger_produces_state_one(self, ledger_env):
        """Empty ledger rebuilds to state=1 (no axioms)."""
        ledger, _, _ = ledger_env
        fresh = GodelStateAlgebra()
        rebuilt = await ledger.rebuild_state(fresh)
        assert rebuilt == 1

    @pytest.mark.asyncio
    async def test_replay_populates_algebra_maps(self, ledger_env):
        """MINT events populate the algebra's axiom↔prime maps."""
        ledger, algebra, _ = ledger_env

        p = algebra.get_or_mint_prime("test", "map", "population")
        await ledger.append_event("MINT", p, "test||map||population")
        await ledger.append_event("MUL", p)

        fresh = GodelStateAlgebra()
        await ledger.rebuild_state(fresh)

        assert "test||map||population" in fresh.axiom_to_prime
        assert fresh.axiom_to_prime["test||map||population"] == p


class TestChronosTimeTravel:

    @pytest.mark.asyncio
    async def test_time_travel_to_tick(self, ledger_env):
        """Rebuild to an intermediate tick returns historical state."""
        ledger, algebra, _ = ledger_env

        p1 = algebra.get_or_mint_prime("first", "is", "early")
        p2 = algebra.get_or_mint_prime("second", "is", "late")

        await ledger.append_event("MINT", p1, "first||is||early")
        await ledger.append_event("MUL", p1)
        # Tick 2 is the last event with only p1

        tick_after_first = await ledger.get_latest_tick()

        await ledger.append_event("MINT", p2, "second||is||late")
        await ledger.append_event("MUL", p2)

        # Full rebuild includes both
        full_algebra = GodelStateAlgebra()
        full_state = await ledger.rebuild_state(full_algebra)
        assert full_state % p1 == 0
        assert full_state % p2 == 0

        # Time-travel to tick_after_first — should only have p1
        past_algebra = GodelStateAlgebra()
        past_state = await ledger.rebuild_state(past_algebra, max_seq_id=tick_after_first)
        assert past_state % p1 == 0
        assert past_state % p2 != 0

    @pytest.mark.asyncio
    async def test_get_latest_tick_empty(self, ledger_env):
        """Empty ledger returns tick=0."""
        ledger, _, _ = ledger_env
        tick = await ledger.get_latest_tick()
        assert tick == 0

    @pytest.mark.asyncio
    async def test_get_latest_tick_increments(self, ledger_env):
        """Each append increments the tick."""
        ledger, algebra, _ = ledger_env
        p = algebra.get_or_mint_prime("tick", "test", "axiom")

        await ledger.append_event("MINT", p, "tick||test||axiom")
        tick1 = await ledger.get_latest_tick()

        await ledger.append_event("MUL", p)
        tick2 = await ledger.get_latest_tick()

        assert tick2 > tick1


class TestLedgerCrashRecovery:

    @pytest.mark.asyncio
    async def test_new_ledger_instance_sees_old_data(self, ledger_env):
        """A new AkashicLedger instance on same DB file sees prior events."""
        ledger, algebra, db_path = ledger_env

        p = algebra.get_or_mint_prime("persist", "across", "instances")
        await ledger.append_event("MINT", p, "persist||across||instances")
        await ledger.append_event("MUL", p)

        # Simulate crash: create new instance on same DB
        ledger2 = AkashicLedger(db_path=db_path)
        fresh = GodelStateAlgebra()
        rebuilt = await ledger2.rebuild_state(fresh)

        assert rebuilt == p
        assert "persist||across||instances" in fresh.axiom_to_prime
