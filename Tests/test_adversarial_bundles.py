"""
Adversarial Bundle Tests

Tests the system's rejection paths for malformed, tampered, and adversarial
bundle inputs. The enemy is not missing functionality — it is hidden ambiguity,
unhandled edge cases, and silent acceptance of invalid data.

Author: ototao
License: Apache License 2.0
"""

import json
import math
import copy
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import (
    CanonicalCodec,
    InvalidSignatureError,
)


# ─── Fixtures ──────────────────────────────────────────────────────

SIGNING_KEY = "adversarial-test-key-32bytes!!!"

@pytest.fixture
def codec():
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, tome_gen, signing_key=SIGNING_KEY)
    # Mint some axioms
    for s, p, o in [("alice", "likes", "cats"), ("bob", "knows", "python")]:
        algebra.get_or_mint_prime(s, p, o)
    return codec, algebra


@pytest.fixture
def valid_bundle(codec):
    codec_inst, algebra = codec
    state = 1
    for prime in algebra.axiom_to_prime.values():
        state = math.lcm(state, prime)
    return codec_inst.export_bundle(state, branch="adversarial-test")


# ─── 1. Tampered Canonical Content ────────────────────────────────

class TestTamperedContent:

    def test_modified_tome_rejected(self, codec, valid_bundle):
        """Modifying canonical_tome invalidates HMAC → rejection."""
        codec_inst, _ = codec
        tampered = copy.deepcopy(valid_bundle)
        tampered["canonical_tome"] += "\nThe evil injected axiom."
        with pytest.raises(InvalidSignatureError):
            codec_inst.import_bundle(tampered)

    def test_modified_state_integer_rejected(self, codec, valid_bundle):
        """Modifying state_integer invalidates HMAC → rejection."""
        codec_inst, _ = codec
        tampered = copy.deepcopy(valid_bundle)
        tampered["state_integer"] = "999999999999999999999"
        with pytest.raises(InvalidSignatureError):
            codec_inst.import_bundle(tampered)

    def test_modified_timestamp_rejected(self, codec, valid_bundle):
        """Modifying timestamp invalidates HMAC → rejection."""
        codec_inst, _ = codec
        tampered = copy.deepcopy(valid_bundle)
        tampered["timestamp"] = "2000-01-01T00:00:00+00:00"
        with pytest.raises(InvalidSignatureError):
            codec_inst.import_bundle(tampered)

    def test_swapped_signature_rejected(self, codec, valid_bundle):
        """Replacing signature with a random hex string → rejection."""
        codec_inst, _ = codec
        tampered = copy.deepcopy(valid_bundle)
        tampered["signature"] = "hmac-sha256:" + "a" * 64
        with pytest.raises(InvalidSignatureError):
            codec_inst.import_bundle(tampered)


# ─── 2. Missing Fields ───────────────────────────────────────────

class TestMissingFields:

    @pytest.mark.parametrize("field", [
        "canonical_tome", "state_integer", "timestamp", "signature"
    ])
    def test_missing_required_field(self, codec, valid_bundle, field):
        """Each required field removal causes ValueError, not silent failure."""
        codec_inst, _ = codec
        incomplete = copy.deepcopy(valid_bundle)
        del incomplete[field]
        with pytest.raises(ValueError, match="missing required fields"):
            codec_inst.import_bundle(incomplete)

    def test_empty_bundle_rejected(self, codec):
        """Completely empty dict is rejected."""
        codec_inst, _ = codec
        with pytest.raises(ValueError):
            codec_inst.import_bundle({})


# ─── 3. Wrong Key ────────────────────────────────────────────────

class TestWrongKey:

    def test_different_key_rejects(self, valid_bundle):
        """Bundle signed with key A, imported with key B → rejection."""
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        wrong_key_codec = CanonicalCodec(algebra, tome_gen, signing_key="wrong-key-wrong-key-32bytes!!")
        with pytest.raises(InvalidSignatureError):
            wrong_key_codec.import_bundle(valid_bundle)


# ─── 4. Version Mismatch ─────────────────────────────────────────

class TestVersionMismatch:

    def test_unsupported_bundle_version_still_importable(self, codec, valid_bundle):
        """Bundle with unknown bundle_version but valid signature still imports.
        (Bundle version is metadata, not a gate in current implementation.)"""
        codec_inst, _ = codec
        modified = copy.deepcopy(valid_bundle)
        modified["bundle_version"] = "99.0.0"
        # Re-sign since we haven't changed signed fields
        # (bundle_version is NOT part of HMAC payload)
        result = codec_inst.import_bundle(modified)
        assert result > 0


# ─── 5. Duplicate and Edge-Case Content ──────────────────────────

class TestEdgeCases:

    def test_duplicate_canonical_lines_idempotent(self, codec):
        """Duplicate fact lines in canonical tome produce same state (LCM idempotency)."""
        codec_inst, algebra = codec
        # Build state with one axiom
        prime = algebra.get_or_mint_prime("test", "is", "test")
        state = prime

        # Generate canonical and duplicate a line
        tome = codec_inst.tome_generator.generate_canonical(state, "Dup Test")
        lines = tome.split("\n")
        # Find the fact line and duplicate it
        for i, line in enumerate(lines):
            if line.startswith("The "):
                lines.insert(i + 1, line)  # duplicate
                break
        dup_tome = "\n".join(lines)

        # Parse both and reconstruct
        import re
        original_keys = []
        dup_keys = []
        for line in tome.split("\n"):
            m = re.match(r'^The\s+(\S+)\s+(\S+)\s+(\S+)\.$', line.strip())
            if m:
                original_keys.append("||".join(m.groups()))
        for line in dup_tome.split("\n"):
            m = re.match(r'^The\s+(\S+)\s+(\S+)\s+(\S+)\.$', line.strip())
            if m:
                dup_keys.append("||".join(m.groups()))

        # Reconstruct states
        import math as m2
        state_1 = 1
        state_2 = 1
        for key in original_keys:
            parts = key.split("||")
            p = algebra.get_or_mint_prime(*parts)
            state_1 = m2.lcm(state_1, p)
        for key in dup_keys:
            parts = key.split("||")
            p = algebra.get_or_mint_prime(*parts)
            state_2 = m2.lcm(state_2, p)

        assert state_1 == state_2, "Duplicate lines must not change state (LCM idempotency)"

    def test_empty_state_bundle(self, codec):
        """State = 1 (empty) produces a valid bundle."""
        codec_inst, _ = codec
        bundle = codec_inst.export_bundle(1, branch="empty")
        assert bundle["state_integer"] == "1"
        assert bundle["axiom_count"] == 0

    def test_single_axiom_state(self, codec):
        """Single axiom produces valid bundle with axiom_count = 1."""
        codec_inst, algebra = codec
        prime = algebra.get_or_mint_prime("sole", "fact", "here")
        bundle = codec_inst.export_bundle(prime, branch="single")
        assert bundle["axiom_count"] == 1
        assert int(bundle["state_integer"]) == prime

    def test_very_long_axiom_key(self, codec):
        """Axiom keys with long tokens still produce valid primes."""
        codec_inst, algebra = codec
        long_s = "a" * 200
        long_p = "b" * 200
        long_o = "c" * 200
        prime = algebra.get_or_mint_prime(long_s, long_p, long_o)
        assert prime > 1
        bundle = codec_inst.export_bundle(prime, branch="long-key")
        # Round-trip through import
        imported_state = codec_inst.import_bundle(bundle)
        assert imported_state == prime


# ─── 6. DoS Defense (Resource Exhaustion) ────────────────────────

class TestDoSDefense:

    def test_oversized_tome_rejected(self, codec, valid_bundle):
        """Bundle with >10MB canonical_tome is rejected."""
        codec_inst, _ = codec
        attack = copy.deepcopy(valid_bundle)
        attack["canonical_tome"] = "X" * (11 * 1024 * 1024)  # 11MB
        with pytest.raises(ValueError, match="size limit"):
            codec_inst.import_bundle(attack)

    def test_oversized_state_integer_rejected(self, codec, valid_bundle):
        """Bundle with >100K digit state_integer is rejected."""
        codec_inst, _ = codec
        attack = copy.deepcopy(valid_bundle)
        attack["state_integer"] = "9" * 200_000  # 200K digits
        with pytest.raises(ValueError, match="digit limit"):
            codec_inst.import_bundle(attack)

    def test_excessive_axiom_count_rejected(self, codec, valid_bundle):
        """Bundle claiming >10K axioms is rejected."""
        codec_inst, _ = codec
        attack = copy.deepcopy(valid_bundle)
        attack["axiom_count"] = 50_000
        with pytest.raises(ValueError, match="axiom count exceeds"):
            codec_inst.import_bundle(attack)
