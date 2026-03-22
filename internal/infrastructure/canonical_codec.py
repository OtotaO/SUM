"""
Canonical Codec — Signed Knowledge Transport Bundles

Implements a versioned, signed interchange format for transporting
Gödel-State knowledge between nodes and across time.

Bundle format:
    {
        "bundle_version": "1.0.0",
        "branch": "main",
        "axiom_count": 42,
        "canonical_tome": "...",
        "state_integer": "...",
        "timestamp": "2026-03-22T12:00:00+00:00",
        "signature": "hmac-sha256:..."
    }

The signature covers ``canonical_tome + state_integer + timestamp``
using HMAC-SHA256, ensuring tamper detection during transport.

Delta bundles contain only the novel axioms (the difference between
a source and target state), enabling bandwidth-efficient sync.

Phase 15: Canonical Semantic ABI.

Author: ototao
License: Apache License 2.0
"""

import hashlib
import hmac
import json
import math
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import (
    AutoregressiveTomeGenerator,
    CANONICAL_FORMAT_VERSION,
)

logger = logging.getLogger(__name__)

BUNDLE_VERSION = "1.0.0"


@dataclass
class CanonicalBundle:
    """A self-contained, signed knowledge transport unit."""
    bundle_version: str
    canonical_format_version: str
    branch: str
    axiom_count: int
    canonical_tome: str
    state_integer: str  # String for BigInt JSON safety
    timestamp: str
    signature: str
    is_delta: bool = False


class InvalidSignatureError(Exception):
    """Raised when bundle signature verification fails."""
    pass


class CanonicalCodec:
    """
    Signed knowledge transport codec.

    Exports and imports CanonicalBundles with HMAC-SHA256 signatures
    for tamper-proof knowledge transport between nodes.
    """

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        tome_generator: AutoregressiveTomeGenerator,
        signing_key: str = "sum-default-key",
    ):
        self.algebra = algebra
        self.tome_generator = tome_generator
        self.signing_key = signing_key.encode("utf-8")

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def _sign(self, canonical_tome: str, state_str: str, timestamp: str) -> str:
        """Produce HMAC-SHA256 signature over the proof-critical fields."""
        payload = f"{canonical_tome}|{state_str}|{timestamp}"
        sig = hmac.new(
            self.signing_key,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"hmac-sha256:{sig}"

    def _verify_signature(
        self, canonical_tome: str, state_str: str, timestamp: str, signature: str
    ) -> bool:
        """Verify HMAC-SHA256 signature."""
        expected = self._sign(canonical_tome, state_str, timestamp)
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_bundle(
        self,
        state: int,
        branch: str = "main",
        title: str = "Exported Knowledge Bundle",
    ) -> dict:
        """
        Export a branch's state as a signed, self-contained bundle.

        Args:
            state:  The Gödel integer to export.
            branch: Branch name for metadata.
            title:  Tome title.

        Returns:
            A dict representing the signed CanonicalBundle.
        """
        canonical_tome = self.tome_generator.generate_canonical(state, title)
        active_axioms = self.tome_generator.extract_active_axioms(state)
        state_str = str(state)
        timestamp = datetime.now(timezone.utc).isoformat()

        signature = self._sign(canonical_tome, state_str, timestamp)

        bundle = CanonicalBundle(
            bundle_version=BUNDLE_VERSION,
            canonical_format_version=CANONICAL_FORMAT_VERSION,
            branch=branch,
            axiom_count=len(active_axioms),
            canonical_tome=canonical_tome,
            state_integer=state_str,
            timestamp=timestamp,
            signature=signature,
            is_delta=False,
        )

        return asdict(bundle)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_bundle(self, bundle_dict: dict) -> int:
        """
        Import and verify a signed bundle.

        Validates the HMAC-SHA256 signature, then returns the
        verified state integer for merging into a branch.

        Args:
            bundle_dict: The bundle dict (as received from /export).

        Returns:
            The verified state integer.

        Raises:
            InvalidSignatureError: If the signature doesn't match.
            ValueError: If required fields are missing.
        """
        required = {"canonical_tome", "state_integer", "timestamp", "signature"}
        missing = required - set(bundle_dict.keys())
        if missing:
            raise ValueError(f"Bundle missing required fields: {missing}")

        canonical_tome = bundle_dict["canonical_tome"]
        state_str = bundle_dict["state_integer"]
        timestamp = bundle_dict["timestamp"]
        signature = bundle_dict["signature"]

        if not self._verify_signature(canonical_tome, state_str, timestamp, signature):
            raise InvalidSignatureError(
                "Bundle signature verification failed. "
                "The content may have been tampered with."
            )

        state = int(state_str)
        logger.info(
            "Bundle imported: branch=%s, axioms=%s, verified=True",
            bundle_dict.get("branch", "unknown"),
            bundle_dict.get("axiom_count", "?"),
        )
        return state

    # ------------------------------------------------------------------
    # Delta compression
    # ------------------------------------------------------------------

    def compress_delta(
        self,
        source_state: int,
        target_state: int,
        branch: str = "main",
    ) -> dict:
        """
        Produce a delta bundle containing only the novel axioms.

        The delta is ``target // gcd(target, source)`` — the axioms
        present in the target but not in the source.

        Args:
            source_state: The receiver's current state.
            target_state: The sender's full state.
            branch:       Branch name for metadata.

        Returns:
            A signed delta bundle dict.
        """
        shared = math.gcd(target_state, source_state)
        delta_state = target_state // shared

        if delta_state == 1:
            # No novel axioms
            return self.export_bundle(1, branch, "Empty Delta Bundle")

        return self.export_bundle(
            delta_state,
            branch,
            f"Delta Bundle ({branch})",
        )
