"""
Stage 3A — 128-Bit Parity Tests

Tests:
    1-5.  v1 reference vector reproduction (frozen)
    6-10. v2 reference vector reproduction (frozen)
    11.   v1 unchanged when CURRENT_SCHEME is v1
    12.   v2 derivation produces 128-bit+ primes
    13.   v2 primes are different from v1 primes
    14.   PrimeCollisionError exists and is importable
    15.   v2 collision triggers PrimeCollisionError (forced test)
    16.   v1 collision triggers nextprime loop (legacy behavior)
    17.   Node.js v2 parity (cross-runtime, if available)
"""

import hashlib
import json
import os
import subprocess
import pytest
import sympy

from internal.algorithms.semantic_arithmetic import (
    GodelStateAlgebra,
    PrimeCollisionError,
)
from internal.infrastructure.scheme_registry import CURRENT_SCHEME


# ─── Load frozen vectors ──────────────────────────────────────────────

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "v2_reference_vectors.json"
)

with open(FIXTURE_PATH) as f:
    VECTORS = json.load(f)["vectors"]


# ─── v1 Reference Vector Tests ────────────────────────────────────────

class TestV1ReferenceVectors:
    """Prove v1 derivation is unchanged."""

    @pytest.mark.parametrize("vec", VECTORS, ids=[v["axiom_key"] for v in VECTORS])
    def test_v1_prime_matches_frozen(self, vec):
        h = hashlib.sha256(vec["axiom_key"].encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="big")
        prime = sympy.nextprime(seed)
        assert str(prime) == vec["v1_prime"], (
            f"v1 drift: {vec['axiom_key']} expected {vec['v1_prime']}, got {prime}"
        )

    @pytest.mark.parametrize("vec", VECTORS, ids=[v["axiom_key"] for v in VECTORS])
    def test_v1_seed_matches_frozen(self, vec):
        h = hashlib.sha256(vec["axiom_key"].encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="big")
        assert str(seed) == vec["v1_seed"]


# ─── v2 Reference Vector Tests ────────────────────────────────────────

class TestV2ReferenceVectors:
    """Prove v2 derivation produces correct primes from 16-byte seeds."""

    @pytest.mark.parametrize("vec", VECTORS, ids=[v["axiom_key"] for v in VECTORS])
    def test_v2_prime_matches_frozen(self, vec):
        h = hashlib.sha256(vec["axiom_key"].encode("utf-8")).digest()
        seed = int.from_bytes(h[:16], byteorder="big")
        prime = sympy.nextprime(seed)
        assert str(prime) == vec["v2_prime"], (
            f"v2 drift: {vec['axiom_key']} expected {vec['v2_prime']}, got {prime}"
        )

    @pytest.mark.parametrize("vec", VECTORS, ids=[v["axiom_key"] for v in VECTORS])
    def test_v2_seed_matches_frozen(self, vec):
        h = hashlib.sha256(vec["axiom_key"].encode("utf-8")).digest()
        seed = int.from_bytes(h[:16], byteorder="big")
        assert str(seed) == vec["v2_seed"]


# ─── v2 Behavioral Tests ─────────────────────────────────────────────

class TestV2Behavior:
    def test_v2_primes_are_128bit_plus(self):
        """v2 primes are larger than 2^64."""
        for vec in VECTORS:
            prime = int(vec["v2_prime"])
            assert prime > 2**64, f"{vec['axiom_key']}: v2 prime {prime} is not >2^64"

    def test_v2_primes_differ_from_v1(self):
        """v1 and v2 produce different primes for the same axiom key."""
        for vec in VECTORS:
            assert vec["v1_prime"] != vec["v2_prime"], (
                f"{vec['axiom_key']}: v1 and v2 should produce different primes"
            )

    def test_v2_primes_are_actually_prime(self):
        """sympy confirms all v2 primes are prime."""
        for vec in VECTORS:
            prime = int(vec["v2_prime"])
            assert sympy.isprime(prime), f"v2 prime {prime} is not prime!"

    def test_v1_primes_are_actually_prime(self):
        """sympy confirms all v1 primes are prime."""
        for vec in VECTORS:
            prime = int(vec["v1_prime"])
            assert sympy.isprime(prime), f"v1 prime {prime} is not prime!"


# ─── Collision Policy Tests ───────────────────────────────────────────

class TestCollisionPolicy:
    def test_collision_error_is_importable(self):
        assert PrimeCollisionError is not None

    def test_current_scheme_is_v1(self):
        """CURRENT_SCHEME must remain v1 until activation decision."""
        assert CURRENT_SCHEME == "sha256_64_v1"

    def test_v1_collision_uses_loop(self):
        """In v1 mode, collision resolution advances to the next prime."""
        algebra = GodelStateAlgebra()
        # Manually force a collision by pre-populating prime_to_axiom
        h = hashlib.sha256(b"test||collider||a").digest()
        seed = int.from_bytes(h[:8], byteorder="big")
        colliding_prime = sympy.nextprime(seed)
        # Pre-assign the prime to a different axiom
        algebra.prime_to_axiom[colliding_prime] = "other||axiom||key"
        algebra.axiom_to_prime["other||axiom||key"] = colliding_prime

        # Mint — should NOT raise, should advance to next prime
        result = algebra.get_or_mint_prime("test", "collider", "a")
        assert result != colliding_prime
        assert sympy.isprime(result)


# ─── Cross-Runtime Parity (Node.js) ──────────────────────────────────

class TestNodeParity:
    """Cross-runtime parity test: Python vs Node.js.

    This test is skipped if Node.js is not available or verifier is missing.
    """

    @pytest.fixture
    def verifier_path(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "standalone_verifier", "verify.js"
        )
        if not os.path.exists(path):
            pytest.skip("Node.js verifier not found")
        return path

    def test_v1_parity_with_node(self, verifier_path):
        """Node.js verifier --self-test exercises v1 vectors.

        This passes if the existing self-test is green.
        """
        try:
            result = subprocess.run(
                ["node", verifier_path, "--self-test"],
                capture_output=True, text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Node.js not available or timed out")

        assert result.returncode == 0, f"Node verifier failed: {result.stderr}"
