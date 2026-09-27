"""
Divisibility witness tests (renamed from test_zk_proofs.py).

``DivisibilityWitness`` publishes ``state // prime`` with a salted SHA-256
commitment. It is NOT zero-knowledge: the published quotient times the prime
IS the state, and verification checks only the hash commitment (it binds
neither the prime nor a state). These tests pin that honest description as
well as the unchanged behaviour.

Covers:
  - Witness generation + commitment verification round-trip
  - Non-entailed prime rejection
  - Tampered commitment / salt / quotient detection
  - The witness reveals the state; verification does not bind the prime
  - Fresh salt per witness (commitments differ; quotients do not)
  - Large state stress test
  - Edge cases (state=prime, state=1)

Author: ototao
License: Apache License 2.0
"""

import math
import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.algorithms.divisibility_witness import DivisibilityWitness


@pytest.fixture
def algebra_with_state():
    """Create an algebra with 5 axioms and return (algebra, state)."""
    alg = GodelStateAlgebra()
    axioms = [
        ("alice", "likes", "cats"),
        ("bob", "knows", "python"),
        ("earth", "orbits", "sun"),
        ("water", "is", "wet"),
        ("mars", "has", "moons"),
    ]
    primes = []
    for s, p, o in axioms:
        primes.append(alg.get_or_mint_prime(s, p, o))
    state = 1
    for p in primes:
        state = math.lcm(state, p)
    return alg, state, primes


class TestWitnessRoundTrip:

    def test_basic_proof_verifies(self, algebra_with_state):
        """Generate a proof for an entailed prime → verification succeeds."""
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        assert DivisibilityWitness.verify_proof(proof) is True

    def test_proof_contains_required_fields(self, algebra_with_state):
        """Proof dict has all required fields."""
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        assert "commitment" in proof
        assert "salt" in proof
        assert "prime" in proof
        assert "quotient" in proof

    def test_quotient_is_correct(self, algebra_with_state):
        """Quotient = state // prime (exact integer division)."""
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        assert int(proof["quotient"]) == state // primes[0]

    def test_commitment_is_sha256_hex(self, algebra_with_state):
        """Commitment is a 64-char hex string (SHA-256)."""
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        assert len(proof["commitment"]) == 64
        int(proof["commitment"], 16)  # Should not raise


class TestWitnessNonEntailment:

    def test_non_entailed_prime_rejected(self, algebra_with_state):
        """Proof generation for a prime NOT in the state raises ValueError."""
        alg, state, _ = algebra_with_state
        foreign_prime = alg.get_or_mint_prime("fake", "not", "here")
        assert state % foreign_prime != 0  # Confirm not entailed
        with pytest.raises(ValueError, match="does not entail"):
            DivisibilityWitness.generate_proof(state, foreign_prime)

    def test_prime_larger_than_state_rejected(self):
        """A prime larger than the state cannot be a factor."""
        from sympy import nextprime
        state = 2 * 3 * 5  # small state
        big_prime = nextprime(1000)
        with pytest.raises(ValueError):
            DivisibilityWitness.generate_proof(state, big_prime)


class TestWitnessTampering:

    def test_tampered_commitment_fails(self, algebra_with_state):
        """Flipping a bit in the commitment invalidates the proof."""
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        proof["commitment"] = "a" * 64  # Replace with wrong hash
        assert DivisibilityWitness.verify_proof(proof) is False

    def test_tampered_salt_fails(self, algebra_with_state):
        """Changing the salt invalidates the proof."""
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        proof["salt"] = "0" * 32  # Replace with zero salt
        assert DivisibilityWitness.verify_proof(proof) is False

    def test_tampered_quotient_fails(self, algebra_with_state):
        """Changing the quotient invalidates the proof."""
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        proof["quotient"] = str(int(proof["quotient"]) + 1)
        assert DivisibilityWitness.verify_proof(proof) is False

    def test_swapped_prime_proof_fails(self, algebra_with_state):
        """Proof for prime A does not verify if quotient is swapped to prime B's."""
        _, state, primes = algebra_with_state
        proof_a = DivisibilityWitness.generate_proof(state, primes[0])
        proof_b = DivisibilityWitness.generate_proof(state, primes[1])
        # Swap quotient
        proof_a["quotient"] = proof_b["quotient"]
        assert DivisibilityWitness.verify_proof(proof_a) is False


class TestWitnessMultiple:

    def test_all_axioms_proveable(self, algebra_with_state):
        """Every axiom in the state can generate a valid proof."""
        _, state, primes = algebra_with_state
        for prime in primes:
            proof = DivisibilityWitness.generate_proof(state, prime)
            assert DivisibilityWitness.verify_proof(proof) is True

    def test_proofs_have_different_salts(self, algebra_with_state):
        """Two proofs for the same prime have different salts (randomness)."""
        _, state, primes = algebra_with_state
        proof1 = DivisibilityWitness.generate_proof(state, primes[0])
        proof2 = DivisibilityWitness.generate_proof(state, primes[0])
        assert proof1["salt"] != proof2["salt"]
        # Both still verify
        assert DivisibilityWitness.verify_proof(proof1) is True
        assert DivisibilityWitness.verify_proof(proof2) is True

    def test_commitments_differ_but_witnesses_are_linkable(self, algebra_with_state):
        """Fresh salts give different commitments, but the published
        quotient is identical, so two witnesses are trivially linkable."""
        _, state, primes = algebra_with_state
        proof1 = DivisibilityWitness.generate_proof(state, primes[0])
        proof2 = DivisibilityWitness.generate_proof(state, primes[0])
        assert proof1["commitment"] != proof2["commitment"]
        assert proof1["quotient"] == proof2["quotient"]


class TestWitnessIsNotZeroKnowledge:
    """Pins what the old name hid: the witness reveals the state, and the
    verifier checks only the hash commitment."""

    def test_witness_reveals_the_state(self, algebra_with_state):
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        assert int(proof["quotient"]) * proof["prime"] == state

    def test_verify_does_not_bind_the_prime(self, algebra_with_state):
        _, state, primes = algebra_with_state
        proof = DivisibilityWitness.generate_proof(state, primes[0])
        proof["prime"] = primes[1]  # a different prime still verifies
        assert DivisibilityWitness.verify_proof(proof) is True

    def test_old_zero_knowledge_module_name_is_gone(self):
        import importlib.util
        assert importlib.util.find_spec("sum_engine_internal.algorithms.zk_semantics") is None


class TestWitnessEdgeCases:

    def test_state_equals_prime(self):
        """When state = prime itself, quotient = 1."""
        proof = DivisibilityWitness.generate_proof(7, 7)
        assert int(proof["quotient"]) == 1
        assert DivisibilityWitness.verify_proof(proof) is True

    def test_state_is_one_rejects_all(self):
        """State=1 (empty) entails no primes."""
        with pytest.raises(ValueError):
            DivisibilityWitness.generate_proof(1, 2)

    def test_large_state_proof(self, algebra_with_state):
        """Stress test: 100 axioms, prove each one."""
        alg = GodelStateAlgebra()
        state = 1
        primes = []
        for i in range(100):
            p = alg.get_or_mint_prime(f"s{i}", f"p{i}", f"o{i}")
            primes.append(p)
            state = math.lcm(state, p)

        for prime in primes:
            proof = DivisibilityWitness.generate_proof(state, prime)
            assert DivisibilityWitness.verify_proof(proof) is True
