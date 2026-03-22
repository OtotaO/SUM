"""
Key Rotation Tests

Verifies that key rotation archives old keys and maintains
backward compatibility with bundles signed by rotated keys.

Author: ototao
License: Apache License 2.0
"""

import math
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import CanonicalCodec
from internal.infrastructure.key_manager import KeyManager


SIGNING_KEY = "rotation-test-key-32-bytes!!!!!"


@pytest.fixture
def key_manager(tmp_path):
    return KeyManager(key_dir=str(tmp_path / "keys"))


class TestKeyRotation:

    def test_rotate_archives_old_key(self, key_manager):
        """rotate_keypair() moves old public key to rotated/."""
        key_manager.ensure_keypair()
        original_pub = key_manager.get_public_key_bytes()

        key_manager.rotate_keypair()
        new_pub = key_manager.get_public_key_bytes()

        assert original_pub != new_pub
        assert key_manager.rotated_dir.exists()
        archived = list(key_manager.rotated_dir.glob("*.pem"))
        assert len(archived) == 1

    def test_trusted_keys_includes_current_and_archived(self, key_manager):
        """list_trusted_public_keys() returns both old and new keys."""
        key_manager.ensure_keypair()
        pub1 = key_manager.get_public_key_bytes()

        key_manager.rotate_keypair()
        pub2 = key_manager.get_public_key_bytes()

        trusted = key_manager.list_trusted_public_keys()
        assert pub2 in trusted  # current
        assert pub1 in trusted  # archived
        assert len(trusted) == 2

    def test_multiple_rotations(self, key_manager):
        """Multiple rotations accumulate archived keys."""
        key_manager.ensure_keypair()

        for _ in range(3):
            key_manager.rotate_keypair()

        trusted = key_manager.list_trusted_public_keys()
        assert len(trusted) == 4  # 1 current + 3 archived

    def test_old_bundle_verifiable_after_rotation(self, tmp_path):
        """Bundle signed with old key is still verifiable via trusted keys."""
        km = KeyManager(key_dir=str(tmp_path / "keys"))
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)

        # Sign with original key
        codec1 = CanonicalCodec(algebra, tome_gen, SIGNING_KEY, key_manager=km)
        algebra.get_or_mint_prime("test", "fact", "one")
        state = list(algebra.axiom_to_prime.values())[0]
        bundle = codec1.export_bundle(state)
        old_pub = km.get_public_key_bytes()

        # Rotate key
        km.rotate_keypair()
        new_pub = km.get_public_key_bytes()
        assert old_pub != new_pub

        # Old bundle still imports (HMAC still valid, Ed25519 uses embedded key)
        codec2 = CanonicalCodec(algebra, tome_gen, SIGNING_KEY, key_manager=km)
        imported = codec2.import_bundle(bundle)
        assert imported == state

    def test_rotation_empty_start(self, key_manager):
        """Rotation when no keys exist yet just generates first pair."""
        key_manager.rotate_keypair()
        assert key_manager.private_key_path.exists()
        assert key_manager.public_key_path.exists()
        assert len(key_manager.list_trusted_public_keys()) == 1
