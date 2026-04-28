"""Render-receipt verifier (Phase E.1 v0.9.C, Python side).

Mirrors single_file_demo/receipt_verifier.js step-for-step:

    Step 0  shape gate        — receipt is a dict; required fields present.
    Step 0.5 schema gate      — forward-compat: receipt.schema must equal
                                SUPPORTED_SCHEMA. v1 verifiers reject
                                v2+ receipts closed (RENDER_RECEIPT_FORMAT
                                §1.4).
    Step 1  kid lookup        — receipt.kid must be present in jwks.keys.
    Step 2  JCS canonicalize  — receipt.payload → UTF-8 bytes via the
                                pure-Python JCS module. Cross-runtime
                                byte-equivalent with the TS canonicalize
                                library used at signing time (verified
                                empirically by signature verification on
                                the positive-control fixture).
    Step 3  detached JWS split — receipt.jws on "."; middle MUST be empty.
    Step 3.5 crit gate         — forward-compat: any unknown name in the
                                 protected header `crit` array → fail
                                 closed (RFC 7515 §4.1.11). Runs BEFORE
                                 signature verification so future crit
                                 extensions surface as crit_unknown_extension
                                 not signature_invalid.
    Step 4-5 cryptographic    — joserfc deserialize_compact with
                                payload=canonical_bytes verifies the
                                Ed25519 signature.
    Step 6  header invariants — alg=EdDSA, kid matches, b64=false,
                                crit contains "b64".

Error class taxonomy mirrors single_file_demo/receipt_verifier.js's
ERROR_CLASSES. fixtures/render_receipts/*.json names the expected
class for each negative-path case; both runtimes must produce the
same class on the same fixture.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from sum_engine_internal.infrastructure.jcs import canonicalize


SUPPORTED_SCHEMA = "sum.render_receipt.v1"

# crit extensions this verifier knows how to handle. b64=false is the
# unencoded-payload semantics from RFC 7797 — the only critical
# extension v1 receipts use. Any other name in `crit` triggers
# crit_unknown_extension.
KNOWN_CRIT_EXTENSIONS = frozenset({"b64"})


class ErrorClass:
    """String constants — kept identical to the JS verifier's
    ERROR_CLASSES enum so cross-runtime fixture assertions match
    by string compare."""
    MALFORMED_RECEIPT = "malformed_receipt"
    MALFORMED_JWS = "malformed_jws"
    MALFORMED_JWKS = "malformed_jwks"
    UNKNOWN_KID = "unknown_kid"
    KID_MISMATCH = "kid_mismatch"
    SCHEMA_UNKNOWN = "schema_unknown"
    CRIT_UNKNOWN_EXTENSION = "crit_unknown_extension"
    HEADER_INVARIANT_VIOLATED = "header_invariant_violated"
    SIGNATURE_INVALID = "signature_invalid"


class VerifyError(Exception):
    """Raised when receipt verification fails. ``error_class`` is the
    string identifier in ErrorClass; tests + downstream consumers
    branch on it to distinguish failure modes."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class VerifyResult:
    verified: bool
    kid: str
    protected_header: dict[str, Any]
    payload: dict[str, Any]


def _b64url_decode(s: str) -> bytes:
    """RFC 7515 base64url decode with padding restoration."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _import_joserfc() -> Any:
    """Lazy-import joserfc so ImportError surfaces with a clear hint
    on missing optional dep, rather than at module import time when
    receipt verification isn't being used."""
    try:
        from joserfc.jwk import OKPKey  # noqa: F401
        from joserfc.jws import deserialize_compact  # noqa: F401
        from joserfc.errors import BadSignatureError  # noqa: F401
        return None
    except ImportError as e:
        raise ImportError(
            "sum_engine_internal.render_receipt requires `joserfc`. "
            "Install with: pip install sum-engine[receipt-verify]"
        ) from e


def verify_receipt(receipt: Any, jwks: Any) -> VerifyResult:
    """Verify a SUM render receipt against a JWKS.

    Parameters
    ----------
    receipt
        The ``render_receipt`` block from an ``/api/render`` response.
        Must be a dict with keys ``schema``, ``kid``, ``payload``, ``jws``.
    jwks
        A dict with key ``keys`` containing JWK dicts. Typically the
        parsed body of ``/.well-known/jwks.json``.

    Returns
    -------
    VerifyResult on success.

    Raises
    ------
    VerifyError on any failure. ``error_class`` distinguishes between
    failure modes — see ``ErrorClass`` for the taxonomy.
    """
    _import_joserfc()
    from joserfc.jwk import OKPKey
    from joserfc.jws import deserialize_compact
    from joserfc.errors import BadSignatureError

    # ---- Step 0: shape gate ----
    if not isinstance(receipt, dict):
        raise VerifyError(
            ErrorClass.MALFORMED_RECEIPT, "receipt is not a dict"
        )

    # ---- Step 0.5: forward-compat schema ----
    if receipt.get("schema") != SUPPORTED_SCHEMA:
        raise VerifyError(
            ErrorClass.SCHEMA_UNKNOWN,
            f"unsupported receipt schema: {receipt.get('schema')!r} "
            f"(this verifier handles {SUPPORTED_SCHEMA!r})",
        )

    kid = receipt.get("kid")
    payload = receipt.get("payload")
    jws_str = receipt.get("jws")
    if not isinstance(kid, str) or not kid:
        raise VerifyError(
            ErrorClass.MALFORMED_RECEIPT, "receipt.kid missing or empty"
        )
    if not isinstance(payload, dict):
        raise VerifyError(
            ErrorClass.MALFORMED_RECEIPT,
            "receipt.payload missing or non-dict",
        )
    if not isinstance(jws_str, str) or not jws_str:
        raise VerifyError(
            ErrorClass.MALFORMED_RECEIPT, "receipt.jws missing or empty"
        )

    # ---- Step 1: kid lookup ----
    keys = (jwks or {}).get("keys") or []
    key_jwk = next((k for k in keys if k.get("kid") == kid), None)
    if key_jwk is None:
        raise VerifyError(
            ErrorClass.UNKNOWN_KID, f"no key in JWKS for kid={kid}"
        )
    if key_jwk.get("kty") != "OKP" or key_jwk.get("crv") != "Ed25519":
        raise VerifyError(
            ErrorClass.MALFORMED_JWKS,
            f"expected OKP/Ed25519 JWK for kid={kid}, got "
            f"kty={key_jwk.get('kty')} crv={key_jwk.get('crv')}",
        )

    # ---- Step 2: JCS canonicalize ----
    try:
        canonical_bytes = canonicalize(payload)
    except (TypeError, ValueError) as e:
        raise VerifyError(
            ErrorClass.MALFORMED_RECEIPT,
            f"payload could not be JCS-canonicalized: {e}",
        ) from e

    # ---- Step 3: split detached JWS ----
    parts = jws_str.split(".")
    if len(parts) != 3:
        raise VerifyError(
            ErrorClass.MALFORMED_JWS,
            f"JWS must have exactly 3 segments, got {len(parts)}",
        )
    proto, middle, signature = parts
    if middle != "":
        raise VerifyError(
            ErrorClass.MALFORMED_JWS,
            "detached JWS middle segment must be empty (RFC 7515 §A.5)",
        )

    # ---- Step 3.5: crit-extension fail-closed (RFC 7515 §4.1.11) ----
    # The protected header is base64url-encoded JSON; we can read it
    # without verifying the signature, and we MUST reject closed on
    # unknown crit extensions BEFORE attempting signature verification
    # so a future crit extension surfaces as crit_unknown_extension —
    # the spec's intended fail-closed class — not as signature_invalid
    # (which is what would fire if crit-check ran after signature).
    try:
        header_raw = _b64url_decode(proto)
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — surface the real error
        raise VerifyError(
            ErrorClass.MALFORMED_JWS,
            f"protected header is not valid JSON: {e}",
        ) from e
    crit = header.get("crit")
    if isinstance(crit, list):
        for ext in crit:
            if ext not in KNOWN_CRIT_EXTENSIONS:
                raise VerifyError(
                    ErrorClass.CRIT_UNKNOWN_EXTENSION,
                    f"protected header crit contains unsupported "
                    f"extension: {ext!r}",
                )

    # ---- Step 4: import key ----
    try:
        key = OKPKey.import_key(key_jwk)
    except Exception as e:  # noqa: BLE001
        raise VerifyError(
            ErrorClass.MALFORMED_JWKS,
            f"JWKS key for kid={kid} could not be imported: {e}",
        ) from e

    # ---- Step 5: cryptographic verify ----
    # joserfc's deserialize_compact accepts a `payload` kwarg for the
    # detached-payload case (RFC 7515 §A.5 + RFC 7797 b64=false). On
    # signature failure it raises BadSignatureError; other malformed-
    # input cases come through as ValueError / DecodeError. We surface
    # all of them as signature_invalid since the receipt cannot be
    # trusted regardless.
    compact = f"{proto}..{signature}"
    try:
        verified = deserialize_compact(
            compact, key, payload=canonical_bytes, algorithms=["EdDSA"]
        )
    except BadSignatureError as e:
        raise VerifyError(
            ErrorClass.SIGNATURE_INVALID,
            f"signature verification failed: {e}",
        ) from e
    except Exception as e:  # noqa: BLE001
        raise VerifyError(
            ErrorClass.SIGNATURE_INVALID,
            f"signature verification failed: {type(e).__name__}: {e}",
        ) from e

    # ---- Step 6: protected header invariants ----
    ph = verified.protected
    if ph.get("alg") != "EdDSA":
        raise VerifyError(
            ErrorClass.HEADER_INVARIANT_VIOLATED,
            f"expected alg=EdDSA, got {ph.get('alg')!r}",
        )
    if ph.get("kid") != kid:
        raise VerifyError(
            ErrorClass.KID_MISMATCH,
            f"protected header kid={ph.get('kid')!r} != "
            f"receipt.kid={kid!r}",
        )
    if ph.get("b64") is not False:
        raise VerifyError(
            ErrorClass.HEADER_INVARIANT_VIOLATED,
            f"expected b64=False (detached payload encoding), "
            f"got b64={ph.get('b64')!r}",
        )
    crit_h = ph.get("crit")
    if not isinstance(crit_h, list) or "b64" not in crit_h:
        raise VerifyError(
            ErrorClass.HEADER_INVARIANT_VIOLATED,
            f'expected crit array containing "b64", got {crit_h!r}',
        )

    return VerifyResult(
        verified=True,
        kid=kid,
        protected_header=dict(ph),
        payload=payload,
    )
