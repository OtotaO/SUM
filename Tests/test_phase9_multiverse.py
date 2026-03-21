"""
Phase 9 Tests — The Multiverse of Meaning & Interacting Theory

Validates:
    - Universal Deterministic Primes (SHA-256 seeded, globally consistent)
    - Semantic Smart Contracts (Causal Trigger Map cascading inferences)
    - Epistemic Branching (O(1) fork via integer copy)
    - O(1) LCM Merge across branches
    - Branch isolation guarantees
    - Causal cascade idempotency
    - Collision-safe prime minting
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.causal_triggers import CausalTriggerMap


# ─── Mock Ledger ──────────────────────────────────────────────────────

class MockLedger:
    """In-memory ledger stub for unit tests."""

    def __init__(self):
        self.events = []

    async def append_event(self, op, prime, axiom=""):
        self.events.append((op, prime, axiom))


# ─── 1. Universal Deterministic Primes ───────────────────────────────

class TestUniversalDeterministicPrimes:

    def test_two_instances_same_prime(self):
        """Completely isolated instances generate identical primes."""
        algebra1 = GodelStateAlgebra()
        algebra2 = GodelStateAlgebra()

        p1 = algebra1.get_or_mint_prime("apple", "is", "red")
        p2 = algebra2.get_or_mint_prime("apple", "is", "red")

        assert p1 == p2

    def test_deterministic_prime_is_large(self):
        """SHA-256 seeded primes are large cryptographic primes."""
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("apple", "is", "red")

        # SHA-256 seeds into 64-bit space → primes are much larger than sequential
        assert p > 1000

    def test_different_axioms_different_primes(self):
        """Different axioms still produce distinct primes."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("apple", "is", "red")
        p2 = algebra.get_or_mint_prime("apple", "is", "green")
        p3 = algebra.get_or_mint_prime("banana", "is", "yellow")

        assert len({p1, p2, p3}) == 3

    def test_idempotency_preserved(self):
        """Same axiom always returns the same prime on same instance."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("sky", "color", "blue")
        p2 = algebra.get_or_mint_prime("sky", "color", "blue")
        assert p1 == p2

    def test_case_normalization_with_deterministic(self):
        """Case and whitespace normalisation works with deterministic primes."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("Alice", "Age", "30")
        p2 = algebra.get_or_mint_prime("  alice  ", "  age  ", "  30  ")
        assert p1 == p2

    def test_cross_instance_encoding_consistency(self):
        """Two instances produce identical chunk states for identical axioms."""
        alg1 = GodelStateAlgebra()
        alg2 = GodelStateAlgebra()

        axioms = [("X", "is", "1"), ("Y", "is", "2"), ("Z", "is", "3")]

        state1 = alg1.encode_chunk_state(axioms)
        state2 = alg2.encode_chunk_state(axioms)

        assert state1 == state2

    def test_deterministic_primes_are_prime(self):
        """Generated 'primes' are actually prime numbers."""
        import sympy
        algebra = GodelStateAlgebra()

        for s, p, o in [("a", "b", "c"), ("x", "y", "z"), ("1", "2", "3")]:
            prime = algebra.get_or_mint_prime(s, p, o)
            assert sympy.isprime(prime), f"{prime} is not prime"


# ─── 2. Semantic Smart Contracts (Causal Trigger Map) ────────────────

class TestSemanticSmartContracts:

    @pytest.mark.asyncio
    async def test_symmetric_rule_cascades(self):
        """Symmetric relationship rule cascades the inverse."""
        algebra = GodelStateAlgebra()
        ledger = MockLedger()
        contracts = CausalTriggerMap(algebra, ledger)

        # Register symmetric rule
        def cond_sym(s, p, o, state, alg):
            return p == "is_married_to"

        def infer_sym(s, p, o, state, alg):
            return [(o, "is_married_to", s)]

        contracts.register_rule(cond_sym, infer_sym)

        p1 = algebra.get_or_mint_prime("alice", "is_married_to", "bob")
        initial_state = p1

        new_axioms = ["alice||is_married_to||bob"]
        new_state = await contracts.apply_cascade(initial_state, new_axioms)

        # The smart contract should have automatically minted: bob is_married_to alice
        p_cascade = algebra.axiom_to_prime.get("bob||is_married_to||alice")
        assert p_cascade is not None
        assert new_state % p_cascade == 0

    @pytest.mark.asyncio
    async def test_cascade_idempotency(self):
        """Running the cascade twice produces the same state."""
        algebra = GodelStateAlgebra()
        ledger = MockLedger()
        contracts = CausalTriggerMap(algebra, ledger)

        def cond_sym(s, p, o, state, alg):
            return p == "is_married_to"

        def infer_sym(s, p, o, state, alg):
            return [(o, "is_married_to", s)]

        contracts.register_rule(cond_sym, infer_sym)

        p1 = algebra.get_or_mint_prime("alice", "is_married_to", "bob")
        state1 = await contracts.apply_cascade(p1, ["alice||is_married_to||bob"])
        state2 = await contracts.apply_cascade(state1, ["alice||is_married_to||bob"])

        assert state1 == state2  # Idempotent

    @pytest.mark.asyncio
    async def test_cascade_logs_to_ledger(self):
        """Cascaded inferences are persisted to the Akashic Ledger."""
        algebra = GodelStateAlgebra()
        ledger = MockLedger()
        contracts = CausalTriggerMap(algebra, ledger)

        def cond_sym(s, p, o, state, alg):
            return p == "parent_of"

        def infer_sym(s, p, o, state, alg):
            return [(o, "child_of", s)]

        contracts.register_rule(cond_sym, infer_sym)

        p1 = algebra.get_or_mint_prime("alice", "parent_of", "bob")
        await contracts.apply_cascade(p1, ["alice||parent_of||bob"])

        # Should have MINT + MUL events for the cascaded inference
        mint_events = [e for e in ledger.events if e[0] == "MINT"]
        mul_events = [e for e in ledger.events if e[0] == "MUL"]

        assert len(mint_events) >= 1
        assert len(mul_events) >= 1
        assert any("bob||child_of||alice" in e[2] for e in mint_events)

    @pytest.mark.asyncio
    async def test_no_cascade_without_matching_rule(self):
        """Axioms without matching rules don't cascade."""
        algebra = GodelStateAlgebra()
        ledger = MockLedger()
        contracts = CausalTriggerMap(algebra, ledger)

        def cond_sym(s, p, o, state, alg):
            return p == "is_married_to"

        def infer_sym(s, p, o, state, alg):
            return [(o, "is_married_to", s)]

        contracts.register_rule(cond_sym, infer_sym)

        p1 = algebra.get_or_mint_prime("sky", "color", "blue")
        state = await contracts.apply_cascade(p1, ["sky||color||blue"])

        # No cascade should occur — state unchanged
        assert state == p1

    @pytest.mark.asyncio
    async def test_multi_hop_cascade(self):
        """Cascades can chain across multiple rule applications."""
        algebra = GodelStateAlgebra()
        ledger = MockLedger()
        contracts = CausalTriggerMap(algebra, ledger)

        # Rule 1: parent_of → child_of
        def cond_parent(s, p, o, state, alg):
            return p == "parent_of"

        def infer_child(s, p, o, state, alg):
            return [(o, "child_of", s)]

        # Rule 2: child_of → has_parent (tests multi-hop)
        def cond_child(s, p, o, state, alg):
            return p == "child_of"

        def infer_has_parent(s, p, o, state, alg):
            return [(s, "has_parent", o)]

        contracts.register_rule(cond_parent, infer_child)
        contracts.register_rule(cond_child, infer_has_parent)

        p1 = algebra.get_or_mint_prime("alice", "parent_of", "bob")
        state = await contracts.apply_cascade(p1, ["alice||parent_of||bob"])

        # First hop: bob child_of alice
        p_child = algebra.axiom_to_prime.get("bob||child_of||alice")
        assert p_child is not None
        assert state % p_child == 0

        # Second hop: bob has_parent alice
        p_has = algebra.axiom_to_prime.get("bob||has_parent||alice")
        assert p_has is not None
        assert state % p_has == 0


# ─── 3. Epistemic Branching & Merging ────────────────────────────────

class TestEpistemicBranching:

    def test_branch_is_o1_copy(self):
        """Branching is a simple integer copy."""
        algebra = GodelStateAlgebra()

        p_base = algebra.get_or_mint_prime("sky", "color", "blue")
        branches = {"main": p_base}

        # O(1) branch
        branches["experiment"] = branches["main"]

        assert branches["experiment"] == branches["main"]

    def test_branch_isolation(self):
        """Changes to one branch don't affect the other."""
        algebra = GodelStateAlgebra()

        p_base = algebra.get_or_mint_prime("sky", "color", "blue")
        branches = {"main": p_base}

        branches["experiment"] = branches["main"]

        # Evolve experiment independently
        p_exp = algebra.get_or_mint_prime("grass", "color", "green")
        branches["experiment"] = math.lcm(branches["experiment"], p_exp)

        # Main branch should NOT contain experiment's axiom
        assert branches["main"] % p_exp != 0
        # Experiment SHOULD contain the base axiom
        assert branches["experiment"] % p_base == 0

    def test_merge_via_lcm(self):
        """LCM merge produces the union of both branches."""
        algebra = GodelStateAlgebra()

        p_base = algebra.get_or_mint_prime("sky", "color", "blue")
        branches = {"main": p_base}

        # Fork
        branches["experiment"] = branches["main"]

        # Evolve experiment
        p_exp = algebra.get_or_mint_prime("grass", "color", "green")
        branches["experiment"] = math.lcm(branches["experiment"], p_exp)

        # Evolve main independently
        p_main = algebra.get_or_mint_prime("sun", "color", "yellow")
        branches["main"] = math.lcm(branches["main"], p_main)

        # Verify isolation before merge
        assert branches["main"] % p_exp != 0
        assert branches["experiment"] % p_main != 0

        # O(1) Merge via LCM
        branches["main"] = math.lcm(branches["main"], branches["experiment"])

        # Verify successful merge — all three axioms present
        assert branches["main"] % p_base == 0
        assert branches["main"] % p_main == 0
        assert branches["main"] % p_exp == 0

    def test_merge_is_commutative(self):
        """LCM(A, B) == LCM(B, A)."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("a", "is", "1")
        p2 = algebra.get_or_mint_prime("b", "is", "2")

        assert math.lcm(p1, p2) == math.lcm(p2, p1)

    def test_merge_is_associative(self):
        """LCM(LCM(A, B), C) == LCM(A, LCM(B, C))."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("a", "is", "1")
        p2 = algebra.get_or_mint_prime("b", "is", "2")
        p3 = algebra.get_or_mint_prime("c", "is", "3")

        assert math.lcm(math.lcm(p1, p2), p3) == math.lcm(p1, math.lcm(p2, p3))

    def test_rebase_via_division(self):
        """Integer division can remove specific axioms (rebase operation)."""
        algebra = GodelStateAlgebra()

        p_a = algebra.get_or_mint_prime("a", "is", "1")
        p_b = algebra.get_or_mint_prime("b", "is", "2")

        state = p_a * p_b

        # Rebase: remove axiom A
        rebased = algebra.delete_axiom(state, "a||is||1")
        assert rebased % p_a != 0
        assert rebased % p_b == 0

    def test_merge_detects_paradoxes(self):
        """Merging branches with contradictory facts triggers paradox detection."""
        algebra = GodelStateAlgebra()

        # Branch A says Alice is 30
        p_30 = algebra.get_or_mint_prime("alice", "age", "30")
        # Branch B says Alice is 31
        p_31 = algebra.get_or_mint_prime("alice", "age", "31")

        merged = math.lcm(p_30, p_31)
        paradoxes = algebra.detect_curvature_paradoxes(merged)

        assert len(paradoxes) == 1
        assert "alice||age" in paradoxes[0]

    def test_entailment_across_branches(self):
        """Entailment verification works correctly on branch states."""
        algebra = GodelStateAlgebra()

        axioms = [("X", "is", "1"), ("Y", "is", "2"), ("Z", "is", "3")]
        main_state = algebra.encode_chunk_state(axioms)

        # Branch only contains a subset
        branch_state = algebra.encode_chunk_state(axioms[:2])

        # Main entails the branch subset
        assert algebra.verify_entailment(main_state, branch_state)
        # Branch does NOT entail the full main state
        assert not algebra.verify_entailment(branch_state, main_state)

    def test_multiple_branches(self):
        """Multiple concurrent branches operate independently."""
        algebra = GodelStateAlgebra()

        p_base = algebra.get_or_mint_prime("base", "is", "truth")
        branches = {"main": p_base}

        # Create 5 branches
        for i in range(5):
            branches[f"branch_{i}"] = branches["main"]
            p_new = algebra.get_or_mint_prime(f"fact_{i}", "is", "novel")
            branches[f"branch_{i}"] = math.lcm(branches[f"branch_{i}"], p_new)

        # Each branch should have its unique axiom
        for i in range(5):
            p_i = algebra.axiom_to_prime[f"fact_{i}||is||novel"]
            assert branches[f"branch_{i}"] % p_i == 0

            # Other branches should NOT have this axiom
            for j in range(5):
                if i != j:
                    assert branches[f"branch_{j}"] % p_i != 0
