"""
Key Manager — Ed25519 Keypair Management with Key Rotation

Generates, loads, rotates, and manages Ed25519 signing keypairs
for public-key bundle attestation.

Key storage:
    - Private key: PEM file (Ed25519, PKCS8)
    - Public key:  PEM file (SubjectPublicKeyInfo)
    - Archive:     rotated/ subdirectory with timestamped old keys

Key rotation:
    - rotate_keypair() archives the current key and generates a new one
    - list_trusted_public_keys() returns all historical + current keys
    - Old bundles signed with rotated keys can still be verified

Author: ototao
License: Apache License 2.0
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

DEFAULT_KEY_DIR = Path.home() / ".sum" / "keys"
PRIVATE_KEY_FILE = "sum_signing_key.pem"
PUBLIC_KEY_FILE = "sum_public_key.pem"
ROTATED_DIR = "rotated"


class KeyManager:
    """
    Ed25519 key management for bundle attestation.

    Loads or generates Ed25519 keypairs from PEM files.
    If no keys exist at the configured path, a new keypair
    is generated automatically.

    Supports key rotation: old keys are archived with timestamps
    and remain trusted for verifying historical bundles.
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

    @property
    def rotated_dir(self) -> Path:
        return self.key_dir / ROTATED_DIR

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

    # ------------------------------------------------------------------
    # Key Rotation
    # ------------------------------------------------------------------

    def rotate_keypair(self) -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
        """
        Rotate the signing keypair.

        Archives the current key with a timestamp suffix, then
        generates a fresh keypair. Old bundles signed with the
        archived key can still be verified via list_trusted_public_keys().

        Returns:
            (new_private_key, new_public_key) tuple.
        """
        if self.private_key_path.exists():
            self._archive_current_key()

        # Clear cached keys
        self._private_key = None
        self._public_key = None

        # Generate fresh keypair
        self._private_key, self._public_key = self._generate_keypair()
        logger.info("Ed25519 keypair rotated at %s", self.key_dir)
        return self._private_key, self._public_key

    def _archive_current_key(self):
        """Move current public key to the rotated/ archive."""
        self.rotated_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive_name = f"sum_public_key_{ts}.pem"
        archive_path = self.rotated_dir / archive_name

        # Copy public key to archive
        if self.public_key_path.exists():
            archive_path.write_bytes(self.public_key_path.read_bytes())
            logger.info("Archived public key to %s", archive_path)

    def list_trusted_public_keys(self) -> List[bytes]:
        """
        List all trusted public keys (current + rotated).

        Returns:
            List of raw 32-byte public key bytes, current key first.
        """
        keys = []

        # Current key
        if self.public_key_path.exists():
            pub = self._load_public_key_from_pem(self.public_key_path)
            keys.append(
                pub.public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
            )

        # Archived keys
        if self.rotated_dir.exists():
            for pem_file in sorted(self.rotated_dir.glob("*.pem")):
                try:
                    pub = self._load_public_key_from_pem(pem_file)
                    raw = pub.public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw,
                    )
                    if raw not in keys:
                        keys.append(raw)
                except Exception as e:
                    logger.warning("Skipping corrupt archived key %s: %s", pem_file, e)

        return keys

    @staticmethod
    def _load_public_key_from_pem(path: Path) -> Ed25519PublicKey:
        """Load a public key from a PEM file."""
        pem_data = path.read_bytes()
        pub = serialization.load_pem_public_key(pem_data)
        if not isinstance(pub, Ed25519PublicKey):
            raise TypeError(f"Expected Ed25519 public key, got {type(pub).__name__}")
        return pub

    # ------------------------------------------------------------------
    # Core Key Operations
    # ------------------------------------------------------------------

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
