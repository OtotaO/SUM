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
        "signature": "hmac-sha256:...",
        "public_signature": "ed25519:<base64>",  // optional
        "public_key": "ed25519:<base64>"           // optional
    }

Signature layers:
    1. HMAC-SHA256 (shared secret) — tamper detection between trusted peers
    2. Ed25519 (public key) — provenance verification by any third party

Both signatures cover the same payload: ``canonical_tome|state_integer|timestamp``.
Both are optional — pass ``signing_key`` for HMAC, a ``KeyManager`` for Ed25519,
or both. A codec instance with neither configured still validates structure +
Ed25519-if-embedded, which is the right posture for public-domain transport
where the HMAC shared-secret model does not apply (did:web / did:key flows).

Security note: when ``signing_key`` is configured on the importer, any bundle
missing a ``signature`` field is rejected. This is the downgrade-protection
contract — a peer that expects shared-secret tamper-detection must not silently
accept unsigned bundles.

Phase 15: Canonical Semantic ABI.
Phase 17: Ed25519 Public-Key Attestation.

Author: ototao
License: Apache License 2.0
"""

import base64
import hashlib
import hmac
import json
import math
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Optional, TYPE_CHECKING

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.tome_generator import (
    AutoregressiveTomeGenerator,
    CANONICAL_FORMAT_VERSION,
)
from sum_engine_internal.infrastructure.scheme_registry import (
    CURRENT_SCHEME,
    validate_scheme_or_raise,
)
from sum_engine_internal.infrastructure.state_encoding import to_hex

if TYPE_CHECKING:
    from sum_engine_internal.infrastructure.key_manager import KeyManager

logger = logging.getLogger(__name__)


def _zig():
    try:
        from sum_engine_internal.infrastructure.zig_bridge import zig_engine
        return zig_engine
    except ImportError:
        return None

BUNDLE_VERSION = "1.1.0"  # Minor bump: added optional Ed25519 fields


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
    signature: Optional[str] = None  # HMAC-SHA256; omitted when no signing_key
    prime_scheme: str = CURRENT_SCHEME
    state_integer_hex: Optional[str] = None
    is_delta: bool = False
    public_signature: Optional[str] = None  # "ed25519:<base64>"
    public_key: Optional[str] = None        # "ed25519:<base64>"


class InvalidSignatureError(Exception):
    """Raised when bundle signature verification fails."""
    pass


class CanonicalCodec:
    """
    Signed knowledge transport codec.

    Exports and imports CanonicalBundles with HMAC-SHA256 signatures
    for tamper detection, and optional Ed25519 signatures for
    public-key provenance verification.
    """

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        tome_generator: AutoregressiveTomeGenerator,
        signing_key: Optional[str] = None,
        key_manager: Optional["KeyManager"] = None,
    ):
        self.algebra = algebra
        self.tome_generator = tome_generator
        # signing_key=None means HMAC is disabled on this codec: export emits
        # no "signature" field and import tolerates its absence. The old
        # "sum-default-key" placeholder was cryptographic theater (publicly
        # known), so its removal is an honest improvement, not a regression.
        self.signing_key: Optional[bytes] = (
            signing_key.encode("utf-8") if signing_key else None
        )
        self.key_manager = key_manager

    # ------------------------------------------------------------------
    # HMAC Signing
    # ------------------------------------------------------------------

    def _sign(self, canonical_tome: str, state_str: str, timestamp: str) -> Optional[str]:
        """Produce HMAC-SHA256 signature over the proof-critical fields.

        Returns ``None`` when this codec has no signing_key — caller must
        omit the signature field from the emitted bundle.
        """
        if self.signing_key is None:
            return None
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
        """Verify HMAC-SHA256 signature. Caller must gate on signing_key."""
        expected = self._sign(canonical_tome, state_str, timestamp)
        if expected is None:
            return False
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Ed25519 Signing
    # ------------------------------------------------------------------

    def _ed25519_sign(
        self, canonical_tome: str, state_str: str, timestamp: str
    ) -> tuple:
        """
        Produce Ed25519 signature and public key.

        Returns:
            (public_signature_str, public_key_str) or (None, None)
            if no KeyManager is configured.
        """
        if self.key_manager is None:
            return None, None

        private_key, _ = self.key_manager.ensure_keypair()
        payload = f"{canonical_tome}|{state_str}|{timestamp}".encode("utf-8")
        sig_bytes = private_key.sign(payload)
        pub_bytes = self.key_manager.get_public_key_bytes()

        sig_str = f"ed25519:{base64.b64encode(sig_bytes).decode('ascii')}"
        pub_str = f"ed25519:{base64.b64encode(pub_bytes).decode('ascii')}"
        return sig_str, pub_str

    @staticmethod
    def _ed25519_verify(
        canonical_tome: str,
        state_str: str,
        timestamp: str,
        public_signature: str,
        public_key_str: str,
    ) -> bool:
        """
        Verify Ed25519 signature using the embedded public key.

        Args:
            canonical_tome: The canonical tome text.
            state_str:      Decimal state integer string.
            timestamp:      ISO 8601 timestamp.
            public_signature: "ed25519:<base64_sig>"
            public_key_str:   "ed25519:<base64_pubkey>"

        Returns:
            True if valid, False otherwise.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
            from cryptography.exceptions import InvalidSignature

            sig_b64 = public_signature.split(":", 1)[1]
            pub_b64 = public_key_str.split(":", 1)[1]

            sig_bytes = base64.b64decode(sig_b64)
            pub_bytes = base64.b64decode(pub_b64)

            public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            payload = f"{canonical_tome}|{state_str}|{timestamp}".encode(
                "utf-8"
            )
            public_key.verify(sig_bytes, payload)
            return True
        except (InvalidSignature, Exception):
            return False

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
        pub_sig, pub_key = self._ed25519_sign(canonical_tome, state_str, timestamp)

        bundle = CanonicalBundle(
            bundle_version=BUNDLE_VERSION,
            canonical_format_version=CANONICAL_FORMAT_VERSION,
            branch=branch,
            axiom_count=len(active_axioms),
            canonical_tome=canonical_tome,
            state_integer=state_str,
            timestamp=timestamp,
            signature=signature,
            prime_scheme=CURRENT_SCHEME,
            state_integer_hex=to_hex(state),
            is_delta=False,
            public_signature=pub_sig,
            public_key=pub_key,
        )

        # Strip None fields for backward compatibility
        result = asdict(bundle)
        return {k: v for k, v in result.items() if v is not None}

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_bundle(self, bundle_dict: dict) -> int:
        """
        Import and verify a signed bundle.

        Validates the HMAC-SHA256 signature, and optionally the
        Ed25519 public-key signature if present. Returns the
        verified state integer for merging into a branch.

        Args:
            bundle_dict: The bundle dict (as received from /export).

        Returns:
            The verified state integer.

        Raises:
            InvalidSignatureError: If any signature doesn't match.
            ValueError: If required fields are missing.
        """
        # Structural fields are always required. The HMAC "signature" field
        # is only required when this codec was configured with a signing_key
        # — an importer that expects shared-secret tamper-detection refuses
        # to silently accept unsigned bundles (downgrade protection). An
        # importer with no signing_key relies on Ed25519 (if present) or
        # accepts the bundle on structural grounds alone.
        required = {"canonical_tome", "state_integer", "timestamp"}
        if self.signing_key is not None:
            required = required | {"signature"}
        missing = required - set(bundle_dict.keys())
        if missing:
            raise ValueError(f"Bundle missing required fields: {missing}")

        # ── DoS defense: enforce size limits before any crypto ──
        MAX_TOME_BYTES = 10 * 1024 * 1024    # 10 MB
        MAX_STATE_DIGITS = 100_000            # ~330K bits
        MAX_AXIOM_COUNT = 10_000

        tome_size = len(bundle_dict.get("canonical_tome", ""))
        if tome_size > MAX_TOME_BYTES:
            raise ValueError(
                f"Bundle tome exceeds size limit: {tome_size} bytes "
                f"(max {MAX_TOME_BYTES})"
            )

        state_digits = len(bundle_dict.get("state_integer", ""))
        if state_digits > MAX_STATE_DIGITS:
            raise ValueError(
                f"Bundle state integer exceeds digit limit: {state_digits} "
                f"(max {MAX_STATE_DIGITS})"
            )

        axiom_count = bundle_dict.get("axiom_count", 0)
        if isinstance(axiom_count, int) and axiom_count > MAX_AXIOM_COUNT:
            raise ValueError(
                f"Bundle axiom count exceeds limit: {axiom_count} "
                f"(max {MAX_AXIOM_COUNT})"
            )
        # ── Timestamp validation ──
        raw_ts = bundle_dict.get("timestamp", "")
        try:
            from datetime import datetime, timezone, timedelta
            parsed_ts = datetime.fromisoformat(raw_ts)
            # Reject timestamps more than 24h in the future (clock skew tolerance)
            now = datetime.now(timezone.utc)
            if parsed_ts.tzinfo and parsed_ts > now + timedelta(hours=24):
                raise ValueError(
                    f"Bundle timestamp is >24h in the future: {raw_ts}"
                )
        except (ValueError, TypeError) as e:
            if "future" in str(e):
                raise
            raise ValueError(
                f"Bundle timestamp is not valid ISO 8601: {raw_ts!r}"
            ) from e

        canonical_tome = bundle_dict["canonical_tome"]
        state_str = bundle_dict["state_integer"]
        timestamp = bundle_dict["timestamp"]
        signature = bundle_dict.get("signature")

        # Verify HMAC when this codec has a signing_key configured. The
        # "required" check above already guarantees signature is present
        # in that branch. If no signing_key is configured, we ignore the
        # field entirely — no verification, no downgrade attack possible
        # against a codec that explicitly opted out of HMAC.
        if self.signing_key is not None:
            if not self._verify_signature(
                canonical_tome, state_str, timestamp, signature
            ):
                raise InvalidSignatureError(
                    "Bundle HMAC signature verification failed. "
                    "The content may have been tampered with."
                )

        # Verify Ed25519 (if present)
        pub_sig = bundle_dict.get("public_signature")
        pub_key = bundle_dict.get("public_key")
        if pub_sig and pub_key:
            if not self._ed25519_verify(
                canonical_tome, state_str, timestamp, pub_sig, pub_key
            ):
                raise InvalidSignatureError(
                    "Bundle Ed25519 signature verification failed. "
                    "The provenance cannot be verified."
                )
            logger.info("Ed25519 provenance verified for bundle")

        # ── Scheme compatibility check ──
        bundle_scheme = bundle_dict.get("prime_scheme", CURRENT_SCHEME)
        validate_scheme_or_raise(bundle_scheme, context="bundle import")

        state = int(state_str)

        # ── Hex cross-check: if state_integer_hex is present, it must agree ──
        hex_str = bundle_dict.get("state_integer_hex")
        if hex_str is not None:
            try:
                hex_val = int(hex_str, 16)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Bundle state_integer_hex is not valid hex: {hex_str!r}"
                )
            if hex_val != state:
                raise ValueError(
                    f"Bundle state_integer_hex ({hex_str}) does not match "
                    f"state_integer ({state_str}). Possible tampering."
                )
        logger.info(
            "Bundle imported: branch=%s, axioms=%s, scheme=%s, hmac=%s, ed25519=%s",
            bundle_dict.get("branch", "unknown"),
            bundle_dict.get("axiom_count", "?"),
            bundle_scheme,
            "✓" if self.signing_key is not None else "n/a",
            "✓" if pub_sig else "n/a",
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
        z = _zig()
        zg = z.bigint_gcd(target_state, source_state) if z else None
        shared = zg if zg is not None else math.gcd(target_state, source_state)
        delta_state = target_state // shared

        if delta_state == 1:
            # No novel axioms
            return self.export_bundle(1, branch, "Empty Delta Bundle")

        return self.export_bundle(
            delta_state,
            branch,
            f"Delta Bundle ({branch})",
        )
