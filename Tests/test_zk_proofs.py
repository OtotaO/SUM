"""
Zero-Knowledge Semantic Proof Tests

Comprehensive test suite for ZKSemanticProver — SUM's most powerful
mathematical claim: proving axiom entailment without revealing state.

Covers:
  - Proof generation + verification round-trip
  - Non-entailed prime rejection
  - Tampered proof detection (commitment, salt, quotient)
  - Multiple independent proofs from same state
  - Proof non-linkability (different salts)
  - Large state stress test
  - Edge cases (state=prime, state=1)

Author: ototao
License: Apache License 2.0
"""

import math
import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.algorithms.zk_semantics import ZKSemanticProver


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


class TestZKProofRoundTrip:

    def test_basic_proof_verifies(self, algebra_with_state):
        """Generate a proof for an entailed prime → verification succeeds."""
        _, state, primes = algebra_with_state
        proof = ZKSemanticProver.generate_proof(state, primes[0])
        assert ZKSemanticProver.verify_proof(proof) is True

    def test_proof_contains_required_fields(self, algebra_with_state):
        """Proof dict has all required fields."""
        _, state, primes = algebra_with_state
        proof = ZKSemanticProver.generate_proof(state, primes[0])
        assert "commitment" in proof
        assert "salt" in proof
        assert "prime" in proof
        assert "quotient" in proof

    def test_quotient_is_correct(self, algebra_with_state):
        """Quotient = state // prime (exact integer division)."""
        _, state, primes = algebra_with_state
        proof = ZKSemanticProver.generate_proof(state, primes[0])
        assert int(proof["quotient"]) == state // primes[0]

    def test_commitment_is_sha256_hex(self, algebra_with_state):
        """Commitment is a 64-char hex string (SHA-256)."""
        _, state, primes = algebra_with_state
        proof = ZKSemanticProver.generate_proof(state, primes[0])
        assert len(proof["commitment"]) == 64
        int(proof["commitment"], 16)  # Should not raise


class TestZKNonEntailment:

    def test_non_entailed_prime_rejected(self, algebra_with_state):
        """Proof generation for a prime NOT in the state raises ValueError."""
        alg, state, _ = algebra_with_state
        foreign_prime = alg.get_or_mint_prime("fake", "not", "here")
        assert state % foreign_prime != 0  # Confirm not entailed
        with pytest.raises(ValueError, match="does not entail"):
            ZKSemanticProver.generate_proof(state, foreign_prime)

    def test_prime_larger_than_state_rejected(self):
        """A prime larger than the state cannot be a factor."""
        from sympy import nextprime
        state = 2 * 3 * 5  # small state
        big_prime = nextprime(1000)
        with pytest.raises(ValueError):
            ZKSemanticProver.generate_proof(state, big_prime)


class TestZKTamperedProofs:

    def test_tampered_commitment_fails(self, algebra_with_state):
        """Flipping a bit in the commitment invalidates the proof."""
        _, state, primes = algebra_with_state
        proof = ZKSemanticProver.generate_proof(state, primes[0])
        proof["commitment"] = "a" * 64  # Replace with wrong hash
        assert ZKSemanticProver.verify_proof(proof) is False

    def test_tampered_salt_fails(self, algebra_with_state):
        """Changing the salt invalidates the proof."""
        _, state, primes = algebra_with_state
        proof = ZKSemanticProver.generate_proof(state, primes[0])
        proof["salt"] = "0" * 32  # Replace with zero salt
        assert ZKSemanticProver.verify_proof(proof) is False

    def test_tampered_quotient_fails(self, algebra_with_state):
        """Changing the quotient invalidates the proof."""
        _, state, primes = algebra_with_state
        proof = ZKSemanticProver.generate_proof(state, primes[0])
        proof["quotient"] = str(int(proof["quotient"]) + 1)
        assert ZKSemanticProver.verify_proof(proof) is False

    def test_swapped_prime_proof_fails(self, algebra_with_state):
        """Proof for prime A does not verify if quotient is swapped to prime B's."""
        _, state, primes = algebra_with_state
        proof_a = ZKSemanticProver.generate_proof(state, primes[0])
        proof_b = ZKSemanticProver.generate_proof(state, primes[1])
        # Swap quotient
        proof_a["quotient"] = proof_b["quotient"]
        assert ZKSemanticProver.verify_proof(proof_a) is False


class TestZKMultipleProofs:

    def test_all_axioms_proveable(self, algebra_with_state):
        """Every axiom in the state can generate a valid proof."""
        _, state, primes = algebra_with_state
        for prime in primes:
            proof = ZKSemanticProver.generate_proof(state, prime)
            assert ZKSemanticProver.verify_proof(proof) is True

    def test_proofs_have_different_salts(self, algebra_with_state):
        """Two proofs for the same prime have different salts (randomness)."""
        _, state, primes = algebra_with_state
        proof1 = ZKSemanticProver.generate_proof(state, primes[0])
        proof2 = ZKSemanticProver.generate_proof(state, primes[0])
        assert proof1["salt"] != proof2["salt"]
        # Both still verify
        assert ZKSemanticProver.verify_proof(proof1) is True
        assert ZKSemanticProver.verify_proof(proof2) is True

    def test_proofs_non_linkable(self, algebra_with_state):
        """Two proofs for the same prime have different commitments."""
        _, state, primes = algebra_with_state
        proof1 = ZKSemanticProver.generate_proof(state, primes[0])
        proof2 = ZKSemanticProver.generate_proof(state, primes[0])
        assert proof1["commitment"] != proof2["commitment"]


class TestZKEdgeCases:

    def test_state_equals_prime(self):
        """When state = prime itself, quotient = 1."""
        proof = ZKSemanticProver.generate_proof(7, 7)
        assert int(proof["quotient"]) == 1
        assert ZKSemanticProver.verify_proof(proof) is True

    def test_state_is_one_rejects_all(self):
        """State=1 (empty) entails no primes."""
        with pytest.raises(ValueError):
            ZKSemanticProver.generate_proof(1, 2)

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
            proof = ZKSemanticProver.generate_proof(state, prime)
            assert ZKSemanticProver.verify_proof(proof) is True
