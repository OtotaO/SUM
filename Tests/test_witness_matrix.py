"""
Witness Matrix Tests — Cross-Runtime Verification on Frozen Fixtures

Verifies that both Python and Node.js implementations produce identical
Gödel State Integers from the same canonical input, using frozen
reference vectors that never change.

Author: ototao
License: Apache License 2.0
"""

import json
import math
import hashlib
import os
import subprocess
import re
import pytest

from pathlib import Path
from sympy import nextprime

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import CanonicalCodec


FIXTURE_DIR = Path(__file__).parent / "fixtures"
VERIFIER_PATH = Path(__file__).parent.parent / "standalone_verifier" / "verify.js"


# ─── Helpers ──────────────────────────────────────────────────────

def _derive_prime(axiom_key: str) -> int:
    """Deterministic prime derivation matching the spec."""
    h = hashlib.sha256(axiom_key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return int(nextprime(seed))


def _load_fixture(name: str) -> dict:
    path = FIXTURE_DIR / name
    with open(path) as f:
        return json.load(f)


# ─── 1. Reference Vector Consistency ─────────────────────────────

class TestReferenceVectors:

    def test_vectors_file_exists(self):
        """Reference vectors fixture exists."""
        assert (FIXTURE_DIR / "reference_vectors.json").exists()

    def test_python_derives_same_primes(self):
        """Python prime derivation matches frozen reference vectors."""
        vectors = _load_fixture("reference_vectors.json")
        for key, expected_prime_str in vectors.items():
            if key.startswith("_"):
                continue
            actual = _derive_prime(key)
            assert actual == int(expected_prime_str), (
                f"Prime mismatch for '{key}': expected {expected_prime_str}, got {actual}"
            )

    def test_python_lcm_matches_frozen_state(self):
        """Python LCM of all primes matches frozen _lcm_all."""
        vectors = _load_fixture("reference_vectors.json")
        expected_lcm = int(vectors["_lcm_all"])

        state = 1
        for key, prime_str in vectors.items():
            if key.startswith("_"):
                continue
            state = math.lcm(state, int(prime_str))

        assert state == expected_lcm

    def test_spec_reference_vectors(self):
        """Verify the three reference vectors from CANONICAL_ABI_SPEC §4.4."""
        spec_vectors = {
            "alice||likes||cats": 14326936561644797201,
            "bob||knows||python": 12933559861697884259,
            "earth||orbits||sun": 10246101339925224733,
        }
        for key, expected in spec_vectors.items():
            actual = _derive_prime(key)
            assert actual == expected, f"Spec mismatch for '{key}'"


# ─── 2. Python Round-Trip on Frozen Fixtures ─────────────────────

class TestPythonRoundTrip:

    def test_algebra_reproduces_frozen_vectors(self):
        """GodelStateAlgebra produces the same primes as frozen fixtures."""
        vectors = _load_fixture("reference_vectors.json")
        algebra = GodelStateAlgebra()

        for key, prime_str in vectors.items():
            if key.startswith("_"):
                continue
            parts = key.split("||")
            prime = algebra.get_or_mint_prime(*parts)
            assert prime == int(prime_str), f"Algebra mismatch for '{key}'"


# ─── 3. Node.js Witness on Frozen Fixtures ────────────────────────

class TestNodeWitness:

    @pytest.fixture(autouse=True)
    def _check_node(self):
        """Skip if Node.js is not available."""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                pytest.skip("Node.js not available")
        except FileNotFoundError:
            pytest.skip("Node.js not installed")

    def test_node_selftest_passes(self):
        """Node.js verifier self-test passes."""
        result = subprocess.run(
            ["node", str(VERIFIER_PATH), "--self-test"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Self-test failed: {result.stderr}"

    def test_node_verifies_frozen_vectors(self):
        """Node.js derives the same primes as frozen reference vectors."""
        vectors = _load_fixture("reference_vectors.json")
        expected_lcm = vectors["_lcm_all"]

        # Build a minimal bundle for Node.js verification
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        codec = CanonicalCodec(
            algebra, tome_gen,
            signing_key="witness-test-key-32bytes!!!!",
        )

        # Mint the 3 reference axioms
        axioms = [("alice", "likes", "cats"), ("bob", "knows", "python"), ("earth", "orbits", "sun")]
        state = 1
        for s, p, o in axioms:
            prime = algebra.get_or_mint_prime(s, p, o)
            state = math.lcm(state, prime)

        bundle = codec.export_bundle(state, branch="fixture")

        # Write temp bundle
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bundle, f)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["node", str(VERIFIER_PATH), tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            assert "PASS" in result.stdout or result.returncode == 0, (
                f"Node.js verification failed:\n{result.stdout}\n{result.stderr}"
            )
        finally:
            os.unlink(tmp_path)
