"""Wire format helpers for ``sum.transform_receipt.v1``.

Companion to ``docs/TRANSFORM_RECEIPT_FORMAT.md``. This module exposes:

    SUPPORTED_SCHEMA           = "sum.transform_receipt.v1"
    TransformReceiptPayload    — dataclass mirroring the payload
                                  fields in §1.1 of the spec.
    build_payload(...)         — convenience constructor that produces
                                  a dict ready for sign_jose_envelope.
    canonical_hash(bytes)      — sha256-hex helper producing
                                  ``"sha256-<hex>"`` strings used in
                                  parameters_hash / input_hash /
                                  output_hash.

All four fields the receipt binds (``parameters_hash``, ``input_hash``,
``output_hash``, ``transform_id``) get their canonical bytes from the
caller — typically the Transform implementation in
``sum_engine_internal.transforms`` — so the hash inputs match across
Python / TypeScript runtimes byte-for-byte.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal


SUPPORTED_SCHEMA = "sum.transform_receipt.v1"


Provider = Literal[
    "anthropic",
    "openai",
    "cf-ai-gateway-anthropic",
    "cf-ai-gateway-openai",
    "canonical-path",
]
DigitalSourceType = Literal["trainedAlgorithmicMedia", "algorithmicMedia"]


def canonical_hash(canonical_bytes: bytes) -> str:
    """Produce ``"sha256-<hex>"`` from canonical bytes. The single
    consistent shape used for parameters_hash / input_hash /
    output_hash across both Python and Worker code."""
    return "sha256-" + hashlib.sha256(canonical_bytes).hexdigest()


@dataclass(frozen=True)
class TransformReceiptPayload:
    """Mirrors §1.1 of TRANSFORM_RECEIPT_FORMAT.md.

    ``transform_id`` is derived (first 16 hex chars of sha256 over
    ``transform ‖ parameters_hash ‖ input_hash ‖ output_hash``). See
    ``build_payload()`` for the constructor that derives it.
    """
    transform_id: str
    transform: str
    parameters_hash: str
    input_hash: str
    output_hash: str
    model: str
    provider: Provider
    signed_at: str
    digital_source_type: DigitalSourceType

    def to_dict(self) -> dict[str, Any]:
        """Plain dict suitable for ``sign_jose_envelope``. Field order
        is preserved by Python 3.7+ dict semantics; JCS canonicalisation
        in the signing layer will reorder alphabetically regardless."""
        return asdict(self)


def _derive_transform_id(
    transform: str,
    parameters_hash: str,
    input_hash: str,
    output_hash: str,
) -> str:
    """First 16 hex chars of ``sha256(transform ‖ parameters_hash ‖
    input_hash ‖ output_hash)``. Stable across runs and runtimes."""
    h = hashlib.sha256()
    h.update(transform.encode("utf-8"))
    h.update(b"|")
    h.update(parameters_hash.encode("utf-8"))
    h.update(b"|")
    h.update(input_hash.encode("utf-8"))
    h.update(b"|")
    h.update(output_hash.encode("utf-8"))
    return h.hexdigest()[:16]


def build_payload(
    *,
    transform: str,
    parameters_hash: str,
    input_hash: str,
    output_hash: str,
    model: str,
    provider: Provider,
    digital_source_type: DigitalSourceType,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Construct a transform-receipt payload dict from the inputs a
    Transform supplies after a successful ``apply()`` call.

    Computes ``transform_id`` and stamps ``signed_at`` (current UTC
    ISO-8601 if not supplied). Returns a plain dict suitable for
    passing to ``sign_jose_envelope(payload=..., ...)``.

    Parameter shape mirrors the spec exactly — adding fields here
    requires bumping the schema version per §1.5 forward-compat
    policy.
    """
    if signed_at is None:
        # ISO-8601 UTC with millisecond precision + trailing Z.
        # Matches the JS Date.prototype.toISOString() output the
        # Worker produces, so signed_at strings are byte-identical
        # across runtimes.
        now = datetime.now(timezone.utc)
        signed_at = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    transform_id = _derive_transform_id(
        transform, parameters_hash, input_hash, output_hash
    )

    payload = TransformReceiptPayload(
        transform_id=transform_id,
        transform=transform,
        parameters_hash=parameters_hash,
        input_hash=input_hash,
        output_hash=output_hash,
        model=model,
        provider=provider,
        signed_at=signed_at,
        digital_source_type=digital_source_type,
    )
    return payload.to_dict()
