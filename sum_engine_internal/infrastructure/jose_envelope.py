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
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any

from sum_engine_internal.infrastructure.jcs import canonicalize


# crit extensions any v1 SUM-signed envelope may carry. b64=false is
# the unencoded-payload semantics from RFC 7797 — the only critical
# extension v1 envelopes use. Any other name in `crit` triggers
# crit_unknown_extension (RFC 7515 §4.1.11 fail-closed).
DEFAULT_KNOWN_CRIT_EXTENSIONS: frozenset[str] = frozenset({"b64"})

# G3 crypto-agility: signature algorithms accepted under `current`
# in the in-tree algorithm registry (docs/ALGORITHM_REGISTRY.md).
# A `alg` claim outside this set is rejected with `unsupported_alg`,
# distinct from the `header_invariant_violated` class used for
# missing/malformed headers. Adding an algorithm requires a spec
# PR + cross-runtime K-matrix gate; this is not a config knob.
SUPPORTED_SIGNATURE_ALGORITHMS: frozenset[str] = frozenset({"EdDSA"})


class JoseEnvelopeErrorClass:
    """String constants — kept identical across runtimes (mirrored by
    the JS verifier's ERROR_CLASSES) so cross-runtime fixture
    assertions match by string compare."""
    MALFORMED_RECEIPT = "malformed_receipt"
    MALFORMED_JWS = "malformed_jws"
    MALFORMED_JWKS = "malformed_jwks"
    UNKNOWN_KID = "unknown_kid"
    KID_MISMATCH = "kid_mismatch"
    SCHEMA_UNKNOWN = "schema_unknown"
    CRIT_UNKNOWN_EXTENSION = "crit_unknown_extension"
    HEADER_INVARIANT_VIOLATED = "header_invariant_violated"
    SIGNATURE_INVALID = "signature_invalid"
    REVOKED_KID = "revoked_kid"
    """G3 revocation: receipt's kid is on the revocation list with
    effective_revocation_at <= receipt.payload.signed_at. Distinct
    from signature_invalid so the operator-side distinction between
    'tampered' and 'issued under a now-revoked key' is visible at
    the consumer."""
    UNSUPPORTED_ALG = "unsupported_alg"
    """G3 crypto-agility: protected header `alg` claim is not in the
    in-tree algorithm registry under `current`. Distinct from
    header_invariant_violated so a deprecated/retired/unknown alg
    fails with a precise classification — operator can act on the
    specific failure mode (migrate, revoke, or update verifier).
    See docs/ALGORITHM_REGISTRY.md."""
    SIGNED_AT_OUT_OF_WINDOW = "signed_at_out_of_window"
    """Receipt's payload.signed_at is outside the caller-supplied
    acceptance window. Distinct from signature_invalid: the receipt
    is cryptographically valid; the caller has rejected it by policy
    (replay defense). Raised only when the caller opts in via
    max_age_seconds. Past: signed_at is more than max_age_seconds
    behind now. Future: signed_at is more than max_future_skew_seconds
    ahead of now (clock-skew tolerance). The error message names
    which direction. Default behaviour (max_age_seconds=None) does
    NOT enforce — long-lived archival receipts remain valid."""


class SumVerifyError(Exception):
    """Common base for every error a SUM verification surface raises
    (envelope/signature failure, meaning-receipt replay and disclosure
    failure, unsupported schema). ``except SumVerifyError`` is the
    one-clause way to catch "verification failed for any reason";
    callers who need to branch keep catching the specific subclasses."""


class JoseEnvelopeError(SumVerifyError):
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


_B64URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def _is_canonical_b64url(s: object, n_bytes: int) -> bool:
    """True when ``s`` is the canonical unpadded base64url of ``n_bytes``
    bytes: the exact length, base64url characters only (no ``=``, as RFC
    7515 section 2 requires), and the unused low bits of the last
    character zero. The JS verifiers apply the same rule, so both runtimes
    give the same verdict on padded or respelled signatures and keys."""
    length = -(-n_bytes * 4 // 3)
    spare_bits = length * 6 - n_bytes * 8
    return (
        isinstance(s, str)
        and len(s) == length
        and all(c in _B64URL_ALPHABET for c in s)
        and _B64URL_ALPHABET.index(s[-1]) % (1 << spare_bits) == 0
    )


def _is_canonical_b64url_segment(s: object) -> bool:
    """The protected header follows the same rule at any length: base64url
    characters only, no ``=``, a length that is not 1 mod 4, spare low
    bits zero. Mirrors isCanonicalB64urlSegment in the JS verifiers."""
    if not isinstance(s, str) or not s or len(s) % 4 == 1:
        return False
    if not all(c in _B64URL_ALPHABET for c in s):
        return False
    spare_bits = (0, 0, 4, 2)[len(s) % 4]
    return _B64URL_ALPHABET.index(s[-1]) % (1 << spare_bits) == 0


def _jwk_member_error(jwk: dict) -> str | None:
    """Why this JWK may not verify an EdDSA signature, or None if it may.

    The same rule as jwkMemberError in the three JS verifiers, applied before
    import so both runtimes reject the same keys with the same class.
    WebCrypto refused a private key, key_ops without "verify" and use other
    than "sig" at import, while joserfc accepted a private key and refused
    some of the others (key_ops without "verify", use "enc", another alg)
    only at verify time, as signature_invalid. alg may be
    "EdDSA" or its RFC 9864 name "Ed25519", which WebCrypto writes when it
    exports an Ed25519 key; the protected header must still name EdDSA.
    """
    if "d" in jwk:
        return ("carries the private key member d; a verification key set "
                "holds public keys only")
    if "key_ops" in jwk:
        ops = jwk["key_ops"]
        if (not isinstance(ops, list) or not all(isinstance(o, str) for o in ops)
                or len(set(ops)) != len(ops) or "verify" not in ops):
            return 'key_ops must be an array of distinct strings that includes "verify"'
    if "use" in jwk and jwk["use"] != "sig":
        return 'use must be "sig"'
    if "alg" in jwk and jwk["alg"] not in ("EdDSA", "Ed25519"):
        return 'alg must be "EdDSA" or "Ed25519"'
    return None


def verify_jose_envelope(
    envelope: Any,
    jwks: Any,
    *,
    supported_schema: str,
    known_crit_extensions: frozenset[str] = DEFAULT_KNOWN_CRIT_EXTENSIONS,
    max_age_seconds: int | None = None,
    max_future_skew_seconds: int = 60,
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
    max_age_seconds
        Optional replay-defense window. If provided, the verifier
        rejects the envelope with ``signed_at_out_of_window`` when
        ``now - payload.signed_at > max_age_seconds`` (stale receipt)
        OR when ``payload.signed_at - now > max_future_skew_seconds``
        (future-dated receipt beyond the clock-skew tolerance).
        Default ``None`` does NOT enforce — long-lived archival
        receipts and historical fixtures remain valid. The receiver
        opts in per use-case (e.g., agent-swarm handoff: 60s; fresh
        render: 300s; legal-discovery: no window). See
        docs/RENDER_RECEIPT_FORMAT.md §6.2 and
        docs/TRANSFORM_RECEIPT_FORMAT.md §6.2.
    max_future_skew_seconds
        How far ahead of ``now`` ``signed_at`` is allowed to be
        before rejecting (clock-skew tolerance). Only consulted when
        ``max_age_seconds`` is set. Defaults to 60.

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
            JoseEnvelopeErrorClass.MALFORMED_RECEIPT,
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
            JoseEnvelopeErrorClass.MALFORMED_RECEIPT,
            "envelope.kid missing or empty",
        )
    if not isinstance(payload, dict):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_RECEIPT,
            "envelope.payload missing or non-dict",
        )
    if not isinstance(jws_str, str) or not jws_str:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_RECEIPT,
            "envelope.jws missing or empty",
        )

    # ---- Step 1: kid lookup ----
    # A malformed JWKS is the verifier's trusted input (P1), but it must still
    # fail CLOSED with a clean class rather than crash: a non-dict jwks, a
    # missing/non-list `keys`, or non-dict key entries previously raised an
    # unhandled AttributeError (`(5).get(...)`), breaking verifier totality and
    # diverging from the Node runtime. Validate the shape first.
    if not isinstance(jwks, dict):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWKS,
            f"jwks must be an object with a 'keys' array; got {type(jwks).__name__}",
        )
    keys = jwks.get("keys")
    if not isinstance(keys, list):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWKS,
            f"jwks.keys must be an array; got {type(keys).__name__}",
        )
    key_jwk = next((k for k in keys if isinstance(k, dict) and k.get("kid") == kid), None)
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
            JoseEnvelopeErrorClass.MALFORMED_RECEIPT,
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
    if not _is_canonical_b64url_segment(proto):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWS,
            "protected header is not canonical unpadded base64url",
        )
    try:
        header_raw = _b64url_decode(proto)
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWS,
            f"protected header is not valid JSON: {e}",
        ) from e
    # Valid JSON is not enough: RFC 7515 §4 requires the protected header to
    # be a JSON *object*. `[1,2]`, `5`, `"x"` and `null` all parse cleanly and
    # would reach `.get` below as a raw AttributeError, escaping the
    # SumVerifyError contract the public SDK documents. The three JS verifiers
    # are fixed to match in this same commit (they had no guard at all in
    # receipt_verifier.js, and a bare `typeof` check that arrays passed in the
    # other two), so the error classes stay in parity.
    if not isinstance(header, dict):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWS,
            f"protected header must be a JSON object, got "
            f"{type(header).__name__}",
        )
    crit = header.get("crit")
    if isinstance(crit, list):
        for ext in crit:
            # `known_crit_extensions` is a frozenset, so an UNHASHABLE element
            # (a nested list or object) raises TypeError from the membership
            # test itself, escaping the declared error contract on
            # unauthenticated bytes. A non-str element cannot name a crit
            # extension anyway, so reject it as an unknown one. JS reaches the
            # same verdict because Set.has never throws.
            if not isinstance(ext, str) or ext not in known_crit_extensions:
                raise JoseEnvelopeError(
                    JoseEnvelopeErrorClass.CRIT_UNKNOWN_EXTENSION,
                    f"protected header crit contains unsupported "
                    f"extension: {ext!r}",
                )

    # ---- Step 3.6: G3 alg-registry check (crypto-agility) ----
    # Cross-check the protected header's `alg` claim against the
    # in-tree algorithm registry BEFORE invoking joserfc's verify
    # (which has its own alg list, but ours is the load-bearing
    # one — pinned by spec, not by upstream-library default). Any
    # `alg` not in SUPPORTED_SIGNATURE_ALGORITHMS rejects with
    # `unsupported_alg`. See docs/ALGORITHM_REGISTRY.md for the
    # full list + deprecation policy.
    declared_alg = header.get("alg")
    if not isinstance(declared_alg, str) or declared_alg not in SUPPORTED_SIGNATURE_ALGORITHMS:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.UNSUPPORTED_ALG,
            f"protected header alg={declared_alg!r} is not in the "
            f"supported algorithm registry "
            f"({sorted(SUPPORTED_SIGNATURE_ALGORITHMS)}); see "
            f"docs/ALGORITHM_REGISTRY.md",
        )

    # ---- Step 4: import key ----
    if not _is_canonical_b64url(key_jwk.get("x") if isinstance(key_jwk, dict) else None, 32):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWKS,
            f"JWKS key for kid={kid}: x is not the canonical base64url of a "
            f"32-byte Ed25519 public key",
        )
    member_error = _jwk_member_error(key_jwk)
    if member_error is not None:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWKS,
            f"JWKS key for kid={kid} {member_error}",
        )
    try:
        # Import the public key from x alone, so no library can build the key
        # from another member; the key that verifies is the x a pin names.
        key = OKPKey.import_key({"kty": "OKP", "crv": "Ed25519", "x": key_jwk["x"]})
    except Exception as e:  # noqa: BLE001
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.MALFORMED_JWKS,
            f"JWKS key for kid={kid} could not be imported: {e}",
        ) from e

    # ---- Step 5: cryptographic verify ----
    if not _is_canonical_b64url(signature, 64):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SIGNATURE_INVALID,
            "signature is not the canonical base64url of a 64-byte Ed25519 signature",
        )
    compact = f"{proto}..{signature}"
    try:
        # joserfc emits a SecurityWarning ("EdDSA is deprecated via RFC 9864")
        # on every EdDSA verify. We sign the whole trust loop with EdDSA by
        # design (RFC 8032 Ed25519), so the alias-deprecation note is noise —
        # but under a hardened `-W error` posture it would be raised as an
        # exception and surface as a misleading "signature verification failed"
        # on a VALID receipt. Suppress that one message locally (by text, not a
        # blanket filter) so neither the noise nor the false-negative reaches
        # the caller.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*EdDSA is deprecated.*")
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

    # ---- Step 7: optional replay-window check ----
    # Only enforced when caller opts in via max_age_seconds. The
    # cryptographic verify above is already complete; this is a
    # policy-layer rejection ("the receipt is genuine but I don't
    # accept it at this age"), distinct from signature_invalid.
    if max_age_seconds is not None:
        _enforce_signed_at_window(
            payload, max_age_seconds, max_future_skew_seconds
        )

    return JoseEnvelopeResult(
        verified=True,
        kid=kid,
        protected_header=dict(ph),
        payload=payload,
    )


# The receipt timestamp form (RENDER_RECEIPT_FORMAT.md §6.1), a strict profile
# of RFC 3339: uppercase T and Z, seconds required (00 to 59), an optional
# fraction of 1 to 9 digits truncated to milliseconds, Z or +-HH:MM.
# Used by the replay window here and by revocation in render_receipt.
# The grammar, range checks and arithmetic are shared line for line with
# parseInstantMs in single_file_demo/receipt_verifier.js, so Python and the
# browser give the same revocation verdict for every input (datetime
# parsing differed across runtimes and Python versions, and could raise
# OverflowError). Cases: Tests/fixtures/revocation_instants.json.
_INSTANT = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]{1,9}))?(Z|([+-])([0-9]{2}):([0-9]{2}))"
)


def _days_from_civil(y: int, m: int, d: int) -> int:
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _instant_ms(value: Any) -> int | None:
    """Epoch milliseconds for an instant in the receipt timestamp form, or None."""
    if not isinstance(value, str):
        return None
    m = _INSTANT.fullmatch(value)
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    ms = int((m.group(7) or "").ljust(3, "0")[:3])
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    days_in_month = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if not (1 <= mo <= 12 and 1 <= d <= days_in_month[mo - 1]
            and h <= 23 and mi <= 59 and s <= 59):
        return None
    offset = 0
    if m.group(8) != "Z":
        oh, om = int(m.group(10)), int(m.group(11))
        if oh > 23 or om > 59:
            return None
        offset = (oh * 60 + om) * (1 if m.group(9) == "+" else -1)
    return (((_days_from_civil(y, mo, d) * 24 + h) * 60 + mi) * 60 + s) * 1000 + ms - offset * 60000


def _enforce_signed_at_window(
    payload: dict,
    max_age_seconds: int,
    max_future_skew_seconds: int,
) -> None:
    """Raise ``JoseEnvelopeError(SIGNED_AT_OUT_OF_WINDOW)`` if the
    receipt's ``signed_at`` is outside the acceptance window. The
    payload's ``signed_at`` field is REQUIRED when this check is
    requested; a missing or unparseable value fails closed."""
    signed_at_str = payload.get("signed_at")
    if not isinstance(signed_at_str, str):
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SIGNED_AT_OUT_OF_WINDOW,
            f"max_age_seconds={max_age_seconds} requested but "
            f"payload.signed_at is missing or non-string "
            f"({signed_at_str!r}); failing closed",
        )
    # The same instant grammar and integer millisecond arithmetic as
    # parseInstantMs in the JS verifiers and the revocation check, so every
    # Python version and Node give the same verdict on the same signed_at.
    signed_ms = _instant_ms(signed_at_str)
    if signed_ms is None:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SIGNED_AT_OUT_OF_WINDOW,
            f"max_age_seconds={max_age_seconds} requested but "
            f"payload.signed_at={signed_at_str!r} is not in the receipt "
            f"timestamp form (RENDER_RECEIPT_FORMAT.md §6.1); failing closed",
        )
    age_seconds = (time.time_ns() // 1_000_000 - signed_ms) / 1000
    if age_seconds > max_age_seconds:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SIGNED_AT_OUT_OF_WINDOW,
            f"receipt signed_at={signed_at_str} is {age_seconds:.0f}s "
            f"old; max_age_seconds={max_age_seconds}",
        )
    if -age_seconds > max_future_skew_seconds:
        raise JoseEnvelopeError(
            JoseEnvelopeErrorClass.SIGNED_AT_OUT_OF_WINDOW,
            f"receipt signed_at={signed_at_str} is "
            f"{-age_seconds:.0f}s in the future; "
            f"max_future_skew_seconds={max_future_skew_seconds}",
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
    with warnings.catch_warnings():
        # Same EdDSA-deprecation noise as the verify path (see Step 5) — we
        # sign with EdDSA by design; suppress that one message locally.
        warnings.filterwarnings("ignore", message=r".*EdDSA is deprecated.*")
        compact = serialize_compact(
            protected, canonical_bytes, key, algorithms=["EdDSA"]
        )
    detached = detach_compact_content(compact)

    return {
        "kid": kid,
        "payload": payload,
        "jws": detached,
    }
