"""
Concurrency Safety Tests

Verifies that concurrent branch mutations do not lose updates.
"""

import asyncio
import math
import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra


class TestBranchLockSafety:

    @pytest.mark.asyncio
    async def test_concurrent_mutations_no_lost_updates(self):
        """Simulate concurrent LCM merges — all primes must survive."""
        from api.quantum_router import kos
        kos.algebra = GodelStateAlgebra()
        kos.branches = {"main": 1}
        kos.is_booted = True

        primes = []
        for i in range(20):
            p = kos.algebra.get_or_mint_prime(f"concurrent_{i}", "test", f"val_{i}")
            primes.append(p)

        async def merge_prime(p):
            async with kos.branch_lock("main"):
                current = kos.branches["main"]
                await asyncio.sleep(0.001)
                kos.branches["main"] = math.lcm(current, p)

        await asyncio.gather(*[merge_prime(p) for p in primes])

        final = kos.branches["main"]
        for i, p in enumerate(primes):
            assert final % p == 0, f"Prime {i} lost in concurrent merge"

    @pytest.mark.asyncio
    async def test_branch_lock_is_per_branch(self):
        """Mutations on different branches should not block each other."""
        from api.quantum_router import kos
        kos.algebra = GodelStateAlgebra()
        kos.branches = {"a": 1, "b": 1}
        kos.is_booted = True

        order = []

        async def mutate_branch(name, delay):
            async with kos.branch_lock(name):
                order.append(f"{name}_start")
                await asyncio.sleep(delay)
                order.append(f"{name}_end")

        await asyncio.gather(
            mutate_branch("a", 0.05),
            mutate_branch("b", 0.01),
        )

        assert order.index("b_end") < order.index("a_end")
