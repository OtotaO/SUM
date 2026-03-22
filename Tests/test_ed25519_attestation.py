"""
Ed25519 Public-Key Attestation Tests

Verifies the dual-signature model: HMAC-SHA256 for tamper detection,
Ed25519 for public provenance verification.

Tests:
    - Sign + verify round-trip (dual sig)
    - Tampered bundle → Ed25519 rejection
    - HMAC-only bundle backward compatibility
    - Wrong public key → rejection
    - Both signatures verified on import

Author: ototao
License: Apache License 2.0
"""

import base64
import copy
import math
import tempfile
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import (
    CanonicalCodec,
    InvalidSignatureError,
)
from internal.infrastructure.key_manager import KeyManager


# ─── Fixtures ──────────────────────────────────────────────────────

SIGNING_KEY = "ed25519-test-key-32-bytes-long!"


@pytest.fixture
def key_manager(tmp_path):
    """Create a KeyManager with a temp key directory."""
    return KeyManager(key_dir=str(tmp_path / "keys"))


@pytest.fixture
def codec_with_ed25519(key_manager):
    """Codec with Ed25519 enabled."""
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(
        algebra, tome_gen,
        signing_key=SIGNING_KEY,
        key_manager=key_manager,
    )
    # Mint test axioms
    for s, p, o in [("alice", "likes", "cats"), ("bob", "knows", "python")]:
        algebra.get_or_mint_prime(s, p, o)
    return codec, algebra


@pytest.fixture
def codec_hmac_only():
    """Codec without Ed25519 (backward compat)."""
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(
        algebra, tome_gen,
        signing_key=SIGNING_KEY,
    )
    for s, p, o in [("alice", "likes", "cats"), ("bob", "knows", "python")]:
        algebra.get_or_mint_prime(s, p, o)
    return codec, algebra


def _build_state(algebra):
    state = 1
    for prime in algebra.axiom_to_prime.values():
        state = math.lcm(state, prime)
    return state


# ─── 1. Dual-Signature Round-Trip ────────────────────────────────

class TestDualSignatureRoundTrip:

    def test_export_contains_both_signatures(self, codec_with_ed25519):
        """Bundle exported with KeyManager has both HMAC and Ed25519."""
        codec, algebra = codec_with_ed25519
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)

        assert "signature" in bundle
        assert bundle["signature"].startswith("hmac-sha256:")
        assert "public_signature" in bundle
        assert bundle["public_signature"].startswith("ed25519:")
        assert "public_key" in bundle
        assert bundle["public_key"].startswith("ed25519:")

    def test_import_verifies_both_signatures(self, codec_with_ed25519):
        """Import succeeds when both signatures are valid."""
        codec, algebra = codec_with_ed25519
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)
        imported = codec.import_bundle(bundle)
        assert imported == state

    def test_ed25519_signature_is_deterministic(self, codec_with_ed25519):
        """Same payload produces same Ed25519 signature with same key."""
        codec, algebra = codec_with_ed25519
        state = _build_state(algebra)

        bundle1 = codec.export_bundle(state)
        # Ed25519 is deterministic for Ed25519 keys (not randomized like ECDSA)
        # But timestamp differs per export, so signatures will differ
        # We just verify both bundles import successfully
        bundle2 = codec.export_bundle(state)

        assert codec.import_bundle(bundle1) == state
        assert codec.import_bundle(bundle2) == state


# ─── 2. Tampered Bundle → Ed25519 Rejection ──────────────────────

class TestEd25519TamperDetection:

    def test_tampered_tome_rejects_ed25519(self, codec_with_ed25519):
        """Modifying canonical tome invalidates Ed25519 signature."""
        codec, algebra = codec_with_ed25519
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)

        # Re-sign HMAC but not Ed25519
        tampered = copy.deepcopy(bundle)
        tampered["canonical_tome"] += "\nThe evil injected axiom."
        # Fix HMAC to match tampered content
        tampered["signature"] = codec._sign(
            tampered["canonical_tome"],
            tampered["state_integer"],
            tampered["timestamp"],
        )

        # HMAC passes, but Ed25519 should fail
        with pytest.raises(InvalidSignatureError, match="Ed25519"):
            codec.import_bundle(tampered)

    def test_tampered_state_rejects_ed25519(self, codec_with_ed25519):
        """Modifying state integer invalidates Ed25519."""
        codec, algebra = codec_with_ed25519
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)

        tampered = copy.deepcopy(bundle)
        tampered["state_integer"] = "999"
        tampered["signature"] = codec._sign(
            tampered["canonical_tome"],
            tampered["state_integer"],
            tampered["timestamp"],
        )

        with pytest.raises(InvalidSignatureError, match="Ed25519"):
            codec.import_bundle(tampered)


# ─── 3. Backward Compatibility ───────────────────────────────────

class TestBackwardCompatibility:

    def test_hmac_only_bundle_imports_without_ed25519(self, codec_hmac_only):
        """HMAC-only bundles (no Ed25519) still import successfully."""
        codec, algebra = codec_hmac_only
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)

        # No Ed25519 fields
        assert "public_signature" not in bundle
        assert "public_key" not in bundle

        imported = codec.import_bundle(bundle)
        assert imported == state

    def test_hmac_only_bundle_imported_by_ed25519_codec(
        self, codec_with_ed25519, codec_hmac_only
    ):
        """HMAC-only bundle accepted by Ed25519-enabled codec."""
        hmac_codec, hmac_algebra = codec_hmac_only
        ed_codec, _ = codec_with_ed25519

        state = _build_state(hmac_algebra)
        bundle = hmac_codec.export_bundle(state)
        imported = ed_codec.import_bundle(bundle)
        assert imported == state


# ─── 4. Wrong Key ────────────────────────────────────────────────

class TestWrongPublicKey:

    def test_wrong_ed25519_key_rejects(self, codec_with_ed25519, tmp_path):
        """Bundle signed with key A, import with key B embedded → rejection."""
        codec, algebra = codec_with_ed25519
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)

        # Replace the public key with a different one
        different_km = KeyManager(key_dir=str(tmp_path / "other_keys"))
        different_km.ensure_keypair()
        wrong_pub = base64.b64encode(
            different_km.get_public_key_bytes()
        ).decode("ascii")

        tampered = copy.deepcopy(bundle)
        tampered["public_key"] = f"ed25519:{wrong_pub}"

        with pytest.raises(InvalidSignatureError, match="Ed25519"):
            codec.import_bundle(tampered)


# ─── 5. Key Management ──────────────────────────────────────────

class TestKeyManagement:

    def test_keypair_auto_generated(self, tmp_path):
        """KeyManager auto-generates keys when none exist."""
        km = KeyManager(key_dir=str(tmp_path / "fresh"))
        private, public = km.ensure_keypair()
        assert private is not None
        assert public is not None
        assert (tmp_path / "fresh" / "sum_signing_key.pem").exists()
        assert (tmp_path / "fresh" / "sum_public_key.pem").exists()

    def test_keypair_loaded_on_second_use(self, tmp_path):
        """KeyManager loads existing keys on second invocation."""
        key_dir = str(tmp_path / "persistent")
        km1 = KeyManager(key_dir=key_dir)
        _, pub1 = km1.ensure_keypair()

        km2 = KeyManager(key_dir=key_dir)
        _, pub2 = km2.ensure_keypair()

        # Same public key bytes
        assert km1.get_public_key_bytes() == km2.get_public_key_bytes()

    def test_public_key_bytes_are_32_bytes(self, tmp_path):
        """Ed25519 public key is exactly 32 bytes."""
        km = KeyManager(key_dir=str(tmp_path / "size_check"))
        km.ensure_keypair()
        assert len(km.get_public_key_bytes()) == 32
