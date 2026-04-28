"""Shared JOSE-envelope verifier core (Phase R0.2).

Both the render-receipt verifier (Phase E.1 v0.9.C,
``sum_engine_internal.render_receipt``) and the trust-root manifest
verifier (Phase R0.2, ``sum_engine_internal.trust_root``) implement the
same six-step JWS-over-JCS verification algorithm. The only differences
are the supported ``schema`` string and the surface-specific consumer
of the verified payload. This module factors the shared core out so a
schema-aware verifier is ~15 lines wrapping ``verify_jose_envelope``.

Algorithm — exactly mirrors ``docs/RENDER_RECEIPT_FORMAT.md`` §2.1
and ``docs/TRUST_ROOT_FORMAT.md`` §2.1:

    Step 0    shape gate         — envelope is a dict; required fields present.
    Step 0.5  schema gate        — forward-compat: envelope.schema must equal
                                   the caller-supplied supported_schema.
    Step 1    kid lookup         — envelope.kid must be present in jwks.keys.
    Step 2    JCS canonicalize   — envelope.payload → UTF-8 bytes via
                                   sum_engine_internal.infrastructure.jcs.
    Step 3    detached JWS split — envelope.jws on "."; middle MUST be empty.
    Step 3.5  crit gate          — forward-compat: any unknown name in the
                                   protected header `crit` array → fail
                                   closed (RFC 7515 §4.1.11). Runs BEFORE
                                   signature verification so future crit
                                   extensions surface as
                                   crit_unknown_extension, not
                                   signature_invalid.
    Step 4-5  cryptographic      — joserfc deserialize_compact with
                                   payload=canonical_bytes verifies the
                                   Ed25519 signature.
    Step 6    header invariants  — alg=EdDSA, kid matches, b64=false,
                                   crit contains "b64".

Optional joserfc dependency: callers that import this module MUST be
prepared for ImportError if joserfc isn't installed. The two consumer
modules surface this with a clear hint pointing at
``pip install sum-engine[receipt-verify]``.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from sum_engine_internal.infrastructure.jcs import canonicalize


# crit extensions any v1 SUM-signed envelope may carry. b64=false is
# the unencoded-payload semantics from RFC 7797 — the only critical
# extension v1 envelopes use. Any other name in `crit` triggers
# crit_unknown_extension (RFC 7515 §4.1.11 fail-closed).
DEFAULT_KNOWN_CRIT_EXTENSIONS: frozenset[str] = frozenset({"b64"})


class JoseEnvelopeErrorClass:
    """String constants — kept identical across runtimes (mirrored by
    the JS verifier's ERROR_CLASSES) so cross-runtime fixture
    assertions match by string compare."""
    MALFORMED_ENVELOPE = "malformed_envelope"
    MALFORMED_JWS = "malformed_jws"
    MALFORMED_JWKS = "malformed_jwks"
    UNKNOWN_KID = "unknown_kid"
    KID_MISMATCH = "kid_mismatch"
    SCHEMA_UNKNOWN = "schema_unknown"
    CRIT_UNKNOWN_EXTENSION = "crit_unknown_extension"
    HEADER_INVARIANT_VIOLATED = "header_invariant_violated"
    SIGNATURE_INVALID = "signature_invalid"


class JoseEnvelopeError(Exception):
    """Raised when envelope verification fails. ``error_class`` is one
    of the strings in JoseEnvelopeErrorClass; downstream consumers
    branch on it to distinguish failure modes."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class JoseEnvelopeResult:
    """Successful verification result. The schema-aware caller
    interprets ``payload`` per its own surface (render receipt vs
    trust-root manifest)."""

    verified: bool
    kid: str
    protected_header: dict[str, Any]
    payload: dict[str, Any]


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def verify_jose_envelope(
    envelope: Any,
    jwks: Any,
    *,
    supported_schema: str,
    known_crit_extensions: frozenset[str] = DEFAULT_KNOWN_CRIT_EXTENSIONS,
) -> JoseEnvelopeResult:
    """Verify a SUM-shaped signed JSON envelope (receipt or manifest).

    Parameters
    ----------
    envelope
        The signed envelope: a dict with keys ``schema``, ``kid``,
        ``payload``, ``jws``.
    jwks
        A dict with key ``keys`` containing JWK dicts. Typically the
        parsed body of a JWKS endpoint.
    supported_schema
        The schema string this verifier accepts (e.g.
        ``"sum.render_receipt.v1"``). Envelopes with any other schema
        are rejected closed with ``schema_unknown``.
    known_crit_extensions
        Set of JWS protected-header `crit` extension names this
        verifier understands. Defaults to ``{"b64"}`` — any other name
        in `crit` triggers ``crit_unknown_extension``.

    Raises
    ------
    JoseEnvelopeError
        On any verification failure. ``.error_class`` distinguishes
        between failure modes per JoseEnvelopeErrorClass.
    """
    # Lazy joserfc import; if unavailable, surface with a clear hint.
    try:
        from joserfc.errors import BadSignatureError
        from joserfc.jwk import OKPKey
        from joserfc.jws import deserialize_compact
    except ImportError as e:
        raise ImportError(
            "JOSE envelope verification requires `joserfc`. "
            "Install with: pip install sum-engine[receipt-verify]"
        ) from e

    # ---- Step 0: shape gate ----
    if not isinstance(envelope, dict):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_ENVELOPE,
            "envelope is not a dict",
        )

    # ---- Step 0.5: forward-compat schema ----
    if envelope.get("schema") != supported_schema:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SCHEMA_UNKNOWN,
            f"unsupported envelope schema: {envelope.get('schema')!r} "
            f"(this verifier handles {supported_schema!r})",
        )

    kid = envelope.get("kid")
    payload = envelope.get("payload")
    jws_str = envelope.get("jws")
    if not isinstance(kid, str) or not kid:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_ENVELOPE,
            "envelope.kid missing or empty",
        )
    if not isinstance(payload, dict):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_ENVELOPE,
            "envelope.payload missing or non-dict",
        )
    if not isinstance(jws_str, str) or not jws_str:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_ENVELOPE,
            "envelope.jws missing or empty",
        )

    # ---- Step 1: kid lookup ----
    keys = (jwks or {}).get("keys") or []
    key_jwk = next((k for k in keys if k.get("kid") == kid), None)
    if key_jwk is None:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.UNKNOWN_KID,
            f"no key in JWKS for kid={kid}",
        )
    if key_jwk.get("kty") != "OKP" or key_jwk.get("crv") != "Ed25519":
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWKS,
            f"expected OKP/Ed25519 JWK for kid={kid}, got "
            f"kty={key_jwk.get('kty')} crv={key_jwk.get('crv')}",
        )

    # ---- Step 2: JCS canonicalize ----
    try:
        canonical_bytes = canonicalize(payload)
    except (TypeError, ValueError) as e:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_ENVELOPE,
            f"payload could not be JCS-canonicalized: {e}",
        ) from e

    # ---- Step 3: split detached JWS ----
    parts = jws_str.split(".")
    if len(parts) != 3:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWS,
            f"JWS must have exactly 3 segments, got {len(parts)}",
        )
    proto, middle, signature = parts
    if middle != "":
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWS,
            "detached JWS middle segment must be empty (RFC 7515 §A.5)",
        )

    # ---- Step 3.5: crit-extension fail-closed (RFC 7515 §4.1.11) ----
    try:
        header_raw = _b64url_decode(proto)
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWS,
            f"protected header is not valid JSON: {e}",
        ) from e
    crit = header.get("crit")
    if isinstance(crit, list):
        for ext in crit:
            if ext not in known_crit_extensions:
                raise JoseEnvelopeError(
                    JoseEnvelopeErrorClass.CRIT_UNKNOWN_EXTENSION,
                    f"protected header crit contains unsupported "
                    f"extension: {ext!r}",
                )

    # ---- Step 4: import key ----
    try:
        key = OKPKey.import_key(key_jwk)
    except Exception as e:  # noqa: BLE001
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWKS,
            f"JWKS key for kid={kid} could not be imported: {e}",
        ) from e

    # ---- Step 5: cryptographic verify ----
    compact = f"{proto}..{signature}"
    try:
        verified = deserialize_compact(
            compact, key, payload=canonical_bytes, algorithms=["EdDSA"]
        )
    except BadSignatureError as e:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SIGNATURE_INVALID,
            f"signature verification failed: {e}",
        ) from e
    except Exception as e:  # noqa: BLE001
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SIGNATURE_INVALID,
            f"signature verification failed: {type(e).__name__}: {e}",
        ) from e

    # ---- Step 6: protected header invariants ----
    ph = verified.protected
    if ph.get("alg") != "EdDSA":
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.HEADER_INVARIANT_VIOLATED,
            f"expected alg=EdDSA, got {ph.get('alg')!r}",
        )
    if ph.get("kid") != kid:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.KID_MISMATCH,
            f"protected header kid={ph.get('kid')!r} != "
            f"envelope.kid={kid!r}",
        )
    if ph.get("b64") is not False:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.HEADER_INVARIANT_VIOLATED,
            f"expected b64=False (detached payload encoding), "
            f"got b64={ph.get('b64')!r}",
        )
    crit_h = ph.get("crit")
    if not isinstance(crit_h, list) or "b64" not in crit_h:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.HEADER_INVARIANT_VIOLATED,
            f'expected crit array containing "b64", got {crit_h!r}',
        )

    return JoseEnvelopeResult(
        verified=True,
        kid=kid,
        protected_header=dict(ph),
        payload=payload,
    )


def sign_jose_envelope(
    payload: dict[str, Any],
    *,
    private_jwk: dict[str, Any],
    kid: str,
) -> dict[str, Any]:
    """Produce a signed envelope wrapping a payload.

    Returns the four-key envelope dict (schema is the caller's
    responsibility to add — this helper signs the payload but doesn't
    name the surface). Suitable for both render-receipt signing
    (operator-side; lives in worker/src/receipt/sign.ts in TS today,
    not Python) AND trust-root manifest signing (operator-side; this
    is the primary Python user).

    Parameters
    ----------
    payload
        The dict to sign. JCS-canonicalised; the canonical bytes
        become the detached JWS payload.
    private_jwk
        Ed25519 OKP private JWK (must have `d`, `x`, `kty=OKP`,
        `crv=Ed25519`).
    kid
        The key ID claim — both the top-level envelope.kid and the
        JWS protected-header kid get this value.

    Returns
    -------
    A dict with `kid`, `payload`, `jws` keys (caller adds `schema`).
    """
    try:
        from joserfc.jwk import OKPKey
        from joserfc.jws import serialize_compact, detach_compact_content
    except ImportError as e:
        raise ImportError(
            "JOSE envelope signing requires `joserfc`. "
            "Install with: pip install sum-engine[receipt-verify]"
        ) from e

    if private_jwk.get("kty") != "OKP" or private_jwk.get("crv") != "Ed25519":
        raise ValueError(
            f"signing key must be an Ed25519 OKP JWK, got "
            f"kty={private_jwk.get('kty')} crv={private_jwk.get('crv')}"
        )

    canonical_bytes = canonicalize(payload)

    protected = {
        "alg": "EdDSA",
        "kid": kid,
        "b64": False,
        "crit": ["b64"],
    }
    key = OKPKey.import_key(private_jwk)
    # joserfc's serialize_compact with the b64=False (RFC 7797) path:
    # produces the full compact form with the canonical bytes
    # in-line, then detach_compact_content strips the middle segment
    # to give the detached form expected by the verifier.
    #
    # algorithms=["EdDSA"] is required because joserfc treats EdDSA
    # as "not recommended" in its default registry (per RFC 9864
    # advisory) and refuses to sign with it without an explicit
    # opt-in. SUM uses EdDSA / Ed25519 throughout the receipt + trust-
    # root path; the same `algorithms` argument also features in the
    # verify path above for the same reason.
    compact = serialize_compact(
        protected, canonical_bytes, key, algorithms=["EdDSA"]
    )
    detached = detach_compact_content(compact)

    return {
        "kid": kid,
        "payload": payload,
        "jws": detached,
    }
