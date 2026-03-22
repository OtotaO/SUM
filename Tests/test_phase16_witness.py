"""
Phase 16 Cross-Language Witness Test

End-to-end test that:
1. Builds a small deterministic state in Python
2. Exports it via the real CanonicalCodec.export_bundle()
3. Writes the bundle JSON to a temp file
4. Invokes the Node.js standalone verifier
5. Asserts exit code 0 (cross-runtime state equivalence)

Skips gracefully if Node.js is not installed.

Author: ototao
License: Apache License 2.0
"""

import json
import math
import shutil
import subprocess
import tempfile
import os
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import CanonicalCodec

# Path to the standalone verifier
VERIFIER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "standalone_verifier",
    "verify.js",
)

NODE_AVAILABLE = shutil.which("node") is not None


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def witness_system():
    """Fresh algebra, tome_gen, and codec for witness tests."""
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, tome_gen, signing_key="witness-test-key")
    return algebra, tome_gen, codec


def _build_state(algebra, triplets):
    """Mint primes and build LCM state."""
    state = 1
    for s, p, o in triplets:
        prime = algebra.get_or_mint_prime(s, p, o)
        state = math.lcm(state, prime)
    return state


# ─── 1. Node.js Self-Test ─────────────────────────────────────────

@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js not installed")
class TestNodeSelfTest:

    def test_self_test_passes(self):
        """Node.js --self-test exits 0 with all vectors matching."""
        result = subprocess.run(
            ["node", VERIFIER_PATH, "--self-test"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
        assert result.returncode == 0, f"Self-test failed:\n{result.stdout}\n{result.stderr}"


# ─── 2. Cross-Language Witness ────────────────────────────────────

@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js not installed")
class TestCrossLanguageWitness:

    def test_3_axiom_bundle(self, witness_system):
        """Export 3-axiom state from Python, verify in Node.js."""
        algebra, _, codec = witness_system
        state = _build_state(algebra, [
            ("alice", "likes", "cats"),
            ("bob", "knows", "python"),
            ("earth", "orbits", "sun"),
        ])

        bundle = codec.export_bundle(state, branch="witness-test")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bundle, f)
            bundle_path = f.name

        try:
            result = subprocess.run(
                ["node", VERIFIER_PATH, bundle_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            print(result.stdout)
            assert result.returncode == 0, (
                f"Witness verification failed:\n{result.stdout}\n{result.stderr}"
            )
            assert "WITNESS VERIFICATION PASSED" in result.stdout
        finally:
            os.unlink(bundle_path)

    def test_10_axiom_bundle(self, witness_system):
        """Export 10-axiom state from Python, verify in Node.js."""
        algebra, _, codec = witness_system
        triplets = [
            ("alice", "likes", "cats"),
            ("alice", "age", "30"),
            ("bob", "knows", "python"),
            ("bob", "works_at", "google"),
            ("charlie", "drives", "tesla"),
            ("earth", "orbits", "sun"),
            ("mars", "has", "atmosphere"),
            ("python", "runs_on", "cpython"),
            ("sun", "type", "star"),
            ("moon", "orbits", "earth"),
        ]
        state = _build_state(algebra, triplets)
        bundle = codec.export_bundle(state, branch="witness-10")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bundle, f)
            bundle_path = f.name

        try:
            result = subprocess.run(
                ["node", VERIFIER_PATH, bundle_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            print(result.stdout)
            assert result.returncode == 0, (
                f"Witness verification failed:\n{result.stdout}\n{result.stderr}"
            )
            assert "WITNESS VERIFICATION PASSED" in result.stdout
        finally:
            os.unlink(bundle_path)

    def test_single_axiom_bundle(self, witness_system):
        """Minimal: single-axiom state round-trips through witness."""
        algebra, _, codec = witness_system
        state = _build_state(algebra, [("x", "y", "z")])
        bundle = codec.export_bundle(state, branch="minimal")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bundle, f)
            bundle_path = f.name

        try:
            result = subprocess.run(
                ["node", VERIFIER_PATH, bundle_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            print(result.stdout)
            assert result.returncode == 0
        finally:
            os.unlink(bundle_path)

    def test_delta_bundle(self, witness_system):
        """Delta bundles (novel axioms only) also verify correctly."""
        algebra, _, codec = witness_system
        state_a = _build_state(algebra, [("a", "knows", "b")])
        state_b = _build_state(algebra, [
            ("a", "knows", "b"),
            ("c", "knows", "d"),
        ])

        delta_bundle = codec.compress_delta(state_a, state_b)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(delta_bundle, f)
            bundle_path = f.name

        try:
            result = subprocess.run(
                ["node", VERIFIER_PATH, bundle_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            print(result.stdout)
            assert result.returncode == 0
        finally:
            os.unlink(bundle_path)
