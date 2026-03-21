"""
Phase 6 Tests — Fractal Crystallization (Semantic Zooming)

Validates:
    - Crystallize (zoom out): micro-primes removed, macro-prime present
    - Decrystallize (zoom in): macro-prime removed, micro-primes restored
    - Round-trip preserves mathematical equivalence
    - Empty cluster is a no-op
    - Invalid macro key raises ValueError
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
import pytest


class TestFractalCrystallization:

    def test_crystallize_and_decrystallize_roundtrip(self):
        """Full zoom-out → zoom-in round-trip preserves all micro-facts."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("empire", "builds", "death_star")
        p2 = algebra.get_or_mint_prime("rebels", "steal", "plans")

        global_state = p1 * p2

        # Zoom Out (Crystallize)
        macro_key = "star_wars||plot||episode_iv"
        crystallized = algebra.crystallize_axioms(
            global_state,
            ["empire||builds||death_star", "rebels||steal||plans"],
            macro_key,
        )

        macro_prime = algebra.axiom_to_prime[macro_key]

        # Verify: micro-facts gone, macro-fact present
        assert crystallized % p1 != 0
        assert crystallized % p2 != 0
        assert crystallized % macro_prime == 0

        # Zoom In (Decrystallize)
        restored = algebra.decrystallize_axiom(crystallized, macro_key)

        # Verify: macro-fact gone, micro-facts restored
        assert restored % macro_prime != 0
        assert restored % p1 == 0
        assert restored % p2 == 0

    def test_crystallize_empty_cluster_is_noop(self):
        """Crystallizing axioms not in the state is a no-op."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("a", "is", "1")
        state = p1

        result = algebra.crystallize_axioms(
            state, ["nonexistent||key||here"], "macro||is||test"
        )
        assert result == state

    def test_decrystallize_non_macro_is_noop(self):
        """Decrystallizing a non-macro axiom is a no-op."""
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("a", "is", "1")
        state = p

        result = algebra.decrystallize_axiom(state, "a||is||1")
        assert result == state  # Not a macro, so no-op

    def test_crystallize_invalid_macro_key_raises(self):
        """Invalid macro key format raises ValueError."""
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("a", "is", "1")

        with pytest.raises(ValueError, match="subject||predicate||object"):
            algebra.crystallize_axioms(p, ["a||is||1"], "bad_key")

    def test_crystallize_preserves_unrelated_axioms(self):
        """Crystallizing a subset doesn't affect other axioms in the state."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("a", "is", "1")
        p2 = algebra.get_or_mint_prime("b", "is", "2")
        p3 = algebra.get_or_mint_prime("c", "is", "3")

        state = p1 * p2 * p3

        # Only crystallize p1 and p2
        crystallized = algebra.crystallize_axioms(
            state, ["a||is||1", "b||is||2"], "ab||summary||12"
        )

        # p3 should still be present
        assert crystallized % p3 == 0
        # p1 and p2 should be gone
        assert crystallized % p1 != 0
        assert crystallized % p2 != 0
