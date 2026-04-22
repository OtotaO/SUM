"""
Cross-Instance Collision Resolution Tests

Verifies that two independent GodelStateAlgebra instances assign
the same prime to the same axiom key, even when axioms are minted
in different orders. This is critical for P2P correctness.

Author: ototao
License: Apache License 2.0
"""

import math
import pytest
from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra


class TestCrossInstanceConsistency:

    def test_same_axiom_same_prime(self):
        """Two fresh instances assign identical primes for identical axioms."""
        a = GodelStateAlgebra()
        b = GodelStateAlgebra()

        p1 = a.get_or_mint_prime("alice", "likes", "cats")
        p2 = b.get_or_mint_prime("alice", "likes", "cats")
        assert p1 == p2

    def test_different_order_same_primes(self):
        """Minting in different order produces same primes."""
        a = GodelStateAlgebra()
        b = GodelStateAlgebra()

        # Instance A: mint in order X, Y, Z
        pa_x = a.get_or_mint_prime("x", "is", "1")
        pa_y = a.get_or_mint_prime("y", "is", "2")
        pa_z = a.get_or_mint_prime("z", "is", "3")

        # Instance B: mint in order Z, X, Y
        pb_z = b.get_or_mint_prime("z", "is", "3")
        pb_x = b.get_or_mint_prime("x", "is", "1")
        pb_y = b.get_or_mint_prime("y", "is", "2")

        assert pa_x == pb_x
        assert pa_y == pb_y
        assert pa_z == pb_z

    def test_merged_state_order_independent(self):
        """LCM of same axioms is identical regardless of mint order."""
        a = GodelStateAlgebra()
        b = GodelStateAlgebra()

        axioms = [
            ("earth", "orbits", "sun"),
            ("mars", "has", "moons"),
            ("water", "boils", "100c"),
        ]

        # Instance A: forward order
        state_a = 1
        for s, p, o in axioms:
            state_a = math.lcm(state_a, a.get_or_mint_prime(s, p, o))

        # Instance B: reverse order
        state_b = 1
        for s, p, o in reversed(axioms):
            state_b = math.lcm(state_b, b.get_or_mint_prime(s, p, o))

        assert state_a == state_b

    def test_collision_resolution_deterministic(self):
        """If two axioms hash to nearby seeds, collision resolution is deterministic."""
        a = GodelStateAlgebra()
        b = GodelStateAlgebra()

        # Mint many axioms to increase collision probability
        primes_a = {}
        primes_b = {}
        for i in range(100):
            key = (f"s{i}", f"p{i}", f"o{i}")
            primes_a[key] = a.get_or_mint_prime(*key)

        # Mint same axioms in reverse on instance B
        for i in range(99, -1, -1):
            key = (f"s{i}", f"p{i}", f"o{i}")
            primes_b[key] = b.get_or_mint_prime(*key)

        for key in primes_a:
            assert primes_a[key] == primes_b[key], f"Divergence at {key}"

    def test_disjoint_axiom_sets_composable(self):
        """Two instances with non-overlapping axioms can merge states."""
        a = GodelStateAlgebra()
        b = GodelStateAlgebra()

        pa = a.get_or_mint_prime("alice", "likes", "cats")
        pb = b.get_or_mint_prime("bob", "knows", "python")

        merged = math.lcm(pa, pb)

        # Merged state entails both
        assert merged % pa == 0
        assert merged % pb == 0

    def test_1000_axioms_no_divergence(self):
        """Stress test: 1000 axioms produce identical primes across instances."""
        a = GodelStateAlgebra()
        b = GodelStateAlgebra()

        for i in range(1000):
            pa = a.get_or_mint_prime(f"entity_{i}", "has_property", f"value_{i}")
            pb = b.get_or_mint_prime(f"entity_{i}", "has_property", f"value_{i}")
            assert pa == pb, f"Divergence at axiom {i}"
