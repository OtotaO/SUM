"""
Phase 16 Test Vectors: Deterministic Cross-Language Verification

Tests the fundamental primitives that the standalone Node.js witness
relies on: canonical line parsing, SHA-256 hashing, prime derivation,
and mini-state reconstruction.

These exact same values are hardcoded in standalone_verifier/verify.js
as the --self-test suite. If any vector changes here, the JS side
must be updated too.

Author: ototao
License: Apache License 2.0
"""

import hashlib
import math
import re
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator


# ─── Reference Vectors ────────────────────────────────────────────
# These are the ground truth for cross-language agreement.

VECTORS = [
    {
        "axiom_key": "alice||likes||cats",
        "sha256_first8_hex": "c6d380e53c64fca9",
        "seed": 14326936561644797097,
        "prime": 14326936561644797201,
    },
    {
        "axiom_key": "bob||knows||python",
        "sha256_first8_hex": "b37d3c2b55c0b019",
        "seed": 12933559861697884185,
        "prime": 12933559861697884259,
    },
    {
        "axiom_key": "earth||orbits||sun",
        "sha256_first8_hex": "8e3176e1eae59d0a",
        "seed": 10246101339925224714,
        "prime": 10246101339925224733,
    },
]

EXPECTED_LCM_STATE = 1898585074409907150524167558344558620554613878579045806247


# ─── 1. Canonical Line Parsing ───────────────────────────────────

class TestCanonicalLineParsing:

    def test_parse_simple_line(self):
        """'The alice likes cats.' → 'alice||likes||cats'"""
        line = "The alice likes cats."
        match = re.match(r'^The\s+(\S+)\s+(\S+)\s+(\S+)\.$', line.strip())
        assert match is not None
        s, p, o = match.groups()
        assert f"{s}||{p}||{o}" == "alice||likes||cats"

    def test_ignore_header_lines(self):
        """Headers like '@canonical_version' and '# Title' are not fact lines."""
        headers = [
            "@canonical_version: 1.0.0",
            "# My Tome",
            "## Subject Section",
            "",
            "   ",
        ]
        for line in headers:
            match = re.match(r'^The\s+(\S+)\s+(\S+)\s+(\S+)\.$', line.strip())
            assert match is None, f"Should not match: {repr(line)}"

    def test_parse_from_full_tome(self):
        """Parse axioms from a realistic canonical tome."""
        tome = "\n".join([
            "@canonical_version: 1.0.0",
            "# Test",
            "",
            "## Alice",
            "",
            "The alice likes cats.",
            "",
            "## Bob",
            "",
            "The bob knows python.",
            "",
        ])
        axioms = []
        for line in tome.split("\n"):
            match = re.match(r'^The\s+(\S+)\s+(\S+)\s+(\S+)\.$', line.strip())
            if match:
                s, p, o = match.groups()
                axioms.append(f"{s}||{p}||{o}")

        assert axioms == ["alice||likes||cats", "bob||knows||python"]


# ─── 2. SHA-256 Hashing Vectors ──────────────────────────────────

class TestAxiomHashing:

    @pytest.mark.parametrize("vec", VECTORS)
    def test_sha256_first8_bytes(self, vec):
        """SHA-256 of axiom key, first 8 bytes, matches reference."""
        h = hashlib.sha256(vec["axiom_key"].encode("utf-8")).digest()
        hex8 = h[:8].hex()
        assert hex8 == vec["sha256_first8_hex"]

    @pytest.mark.parametrize("vec", VECTORS)
    def test_seed_value(self, vec):
        """First 8 bytes big-endian → 64-bit integer matches reference."""
        h = hashlib.sha256(vec["axiom_key"].encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="big")
        assert seed == vec["seed"]


# ─── 3. Prime Derivation Vectors ─────────────────────────────────

class TestPrimeDerivation:

    @pytest.mark.parametrize("vec", VECTORS)
    def test_deterministic_prime(self, vec):
        """seed → sympy.nextprime(seed) matches reference."""
        algebra = GodelStateAlgebra()
        parts = vec["axiom_key"].split("||")
        prime = algebra.get_or_mint_prime(parts[0], parts[1], parts[2])
        assert prime == vec["prime"]

    def test_primes_are_distinct(self):
        """All three reference primes are distinct."""
        primes = [v["prime"] for v in VECTORS]
        assert len(set(primes)) == 3


# ─── 4. Mini State Reconstruction ────────────────────────────────

class TestMiniStateReconstruction:

    def test_lcm_state_matches_reference(self):
        """LCM of all three primes matches the reference state."""
        state = 1
        for vec in VECTORS:
            state = math.lcm(state, vec["prime"])
        assert state == EXPECTED_LCM_STATE

    def test_state_digit_count(self):
        """Reference state is 58 digits."""
        assert len(str(EXPECTED_LCM_STATE)) == 58

    def test_full_pipeline_matches(self):
        """Full pipeline: algebra → tome → parse → reconstruct matches."""
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)

        # Mint primes the normal way
        for vec in VECTORS:
            parts = vec["axiom_key"].split("||")
            algebra.get_or_mint_prime(parts[0], parts[1], parts[2])

        # Build state
        state = 1
        for vec in VECTORS:
            state = math.lcm(state, vec["prime"])

        # Generate canonical tome
        tome = tome_gen.generate_canonical(state, "Vector Test")

        # Parse it back
        axiom_keys = []
        for line in tome.split("\n"):
            m = re.match(r'^The\s+(\S+)\s+(\S+)\s+(\S+)\.$', line.strip())
            if m:
                s, p, o = m.groups()
                axiom_keys.append(f"{s}||{p}||{o}")

        # Reconstruct
        reconstructed_state = 1
        for key in axiom_keys:
            parts = key.split("||")
            prime = algebra.get_or_mint_prime(parts[0], parts[1], parts[2])
            reconstructed_state = math.lcm(reconstructed_state, prime)

        assert reconstructed_state == state
