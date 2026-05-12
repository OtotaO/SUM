"""Sign a ``sum.transform_receipt.v1`` envelope.

Thin wrapper over ``sum_engine_internal.infrastructure.jose_envelope.
sign_jose_envelope`` that adds the schema identifier. The render-
receipt signer in the Worker's ``worker/src/receipt/sign.ts`` does
the equivalent for TypeScript; the same JCS canonicalisation +
Ed25519 + detached JWS shape produces byte-identical envelopes
across runtimes.
"""
from __future__ import annotations

from typing import Any

from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope
from sum_engine_internal.transform_receipt.format import SUPPORTED_SCHEMA


def sign_transform_receipt(
    payload: dict[str, Any],
    *,
    private_jwk: dict[str, Any],
    kid: str,
) -> dict[str, Any]:
    """Produce a signed ``sum.transform_receipt.v1`` envelope.

    ``payload`` must already be a transform-receipt payload dict
    (typically from ``transform_receipt.format.build_payload``).
    Returns the four-key envelope dict ``{schema, kid, payload, jws}``
    ready to be embedded in a response body or stored as-is.

    Parameters
    ----------
    payload
        Transform-receipt payload — the dict that will be JCS-
        canonicalised and signed. See TRANSFORM_RECEIPT_FORMAT.md
        §1.1 for the field semantics.
    private_jwk
        Ed25519 OKP private JWK (``kty=OKP, crv=Ed25519`` with the
        ``d`` private scalar). Operator-side material; never persist
        in source control.
    kid
        The key ID used both as the top-level envelope.kid and in
        the JWS protected header. Must match an entry in the
        operator's JWKS so verifiers can resolve the public key.
    """
    envelope = sign_jose_envelope(
        payload, private_jwk=private_jwk, kid=kid,
    )
    envelope["schema"] = SUPPORTED_SCHEMA
    return envelope
