"""
Key Manager — Ed25519 Keypair Management

Generates, loads, and manages Ed25519 signing keypairs for
public-key bundle attestation.

Key storage:
    - Private key: PEM file (Ed25519, PKCS8)
    - Public key:  PEM file (SubjectPublicKeyInfo)

Auto-generates a keypair on first use if none exists.

Author: ototao
License: Apache License 2.0
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

DEFAULT_KEY_DIR = Path.home() / ".sum" / "keys"
PRIVATE_KEY_FILE = "sum_signing_key.pem"
PUBLIC_KEY_FILE = "sum_public_key.pem"


class KeyManager:
    """
    Ed25519 key management for bundle attestation.

    Loads or generates Ed25519 keypairs from PEM files.
    If no keys exist at the configured path, a new keypair
    is generated automatically.
    """

    def __init__(self, key_dir: Optional[str] = None):
        self.key_dir = Path(key_dir) if key_dir else DEFAULT_KEY_DIR
        self._private_key: Optional[Ed25519PrivateKey] = None
        self._public_key: Optional[Ed25519PublicKey] = None

    @property
    def private_key_path(self) -> Path:
        return self.key_dir / PRIVATE_KEY_FILE

    @property
    def public_key_path(self) -> Path:
        return self.key_dir / PUBLIC_KEY_FILE

    def ensure_keypair(self) -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
        """
        Load or generate the Ed25519 keypair.

        Returns:
            (private_key, public_key) tuple.
        """
        if self._private_key and self._public_key:
            return self._private_key, self._public_key

        if self.private_key_path.exists():
            self._private_key, self._public_key = self._load_keypair()
            logger.info("Ed25519 keypair loaded from %s", self.key_dir)
        else:
            self._private_key, self._public_key = self._generate_keypair()
            logger.info("Ed25519 keypair generated at %s", self.key_dir)

        return self._private_key, self._public_key

    def get_public_key(self) -> Ed25519PublicKey:
        """Get the public key, loading or generating if needed."""
        _, pub = self.ensure_keypair()
        return pub

    def get_public_key_bytes(self) -> bytes:
        """Get the raw 32-byte public key for embedding in bundles."""
        pub = self.get_public_key()
        return pub.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )

    def _generate_keypair(
        self,
    ) -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
        """Generate a new Ed25519 keypair and save to disk."""
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Ensure directory exists
        self.key_dir.mkdir(parents=True, exist_ok=True)

        # Save private key
        private_pem = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        self.private_key_path.write_bytes(private_pem)
        os.chmod(self.private_key_path, 0o600)

        # Save public key
        public_pem = public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.public_key_path.write_bytes(public_pem)

        return private_key, public_key

    def _load_keypair(self) -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
        """Load an existing keypair from PEM files."""
        private_pem = self.private_key_path.read_bytes()
        private_key = serialization.load_pem_private_key(private_pem, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(
                f"Expected Ed25519 private key, got {type(private_key).__name__}"
            )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def load_public_key_from_bytes(raw_bytes: bytes) -> Ed25519PublicKey:
        """
        Load a public key from raw 32-byte representation.

        Used by verifiers to reconstruct the public key from
        the `public_key` field in a bundle.
        """
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey as PubKey,
        )
        return PubKey.from_public_bytes(raw_bytes)
