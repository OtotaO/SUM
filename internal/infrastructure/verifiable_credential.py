"""W3C Verifiable Credentials 2.0 emission and verification — eddsa-jcs-2022.

Ships the Polytaxis Bucket A capstone: every SUM signed bundle can now also
be rendered as a VC 2.0 JSON-LD document, signed under the Data Integrity 1.0
cryptosuite ``eddsa-jcs-2022`` (Ed25519 over SHA-256(JCS(proofConfig)) ||
SHA-256(JCS(document))). Any compliant VC 2.0 verifier can consume the output
without SUM-specific knowledge.

Public surface:

    make_credential(subject, issuer, credential_type, ...)  -> dict
        Unsigned VC 2.0 document.

    sign_credential(credential, private_key, verification_method, ...) -> dict
        Returns a new dict with a Data Integrity proof block appended.
        Does not mutate the input.

    verify_credential(credential, public_key) -> bool
        Recomputes the hashData and checks the Ed25519 signature. Raises
        VerificationError with a specific reason on failure.

Non-goals (by design):
  - did:key / did:web resolution. ``verificationMethod`` is accepted as an
    opaque URI string and not dereferenced; callers supply the Ed25519
    public key out-of-band when verifying.
  - Credential status, schema, or revocation; these are VC 2.0 optional
    fields and may be added by callers before signing.
  - Full JSON-LD processing (framing, expansion, compaction). eddsa-jcs-2022
    intentionally avoids RDF canonicalization — JCS is applied directly to
    the JSON form, so an external JSON-LD processor is not required.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

from .jcs import canonicalize

__all__ = [
    "CRYPTOSUITE",
    "DATA_INTEGRITY_PROOF_TYPE",
    "VC_V2_CONTEXT",
    "VerificationError",
    "make_credential",
    "sign_credential",
    "verify_credential",
    "multibase_base58btc_encode",
    "multibase_base58btc_decode",
]


VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2"
DATA_INTEGRITY_PROOF_TYPE = "DataIntegrityProof"
CRYPTOSUITE = "eddsa-jcs-2022"
DEFAULT_PROOF_PURPOSE = "assertionMethod"

_B58_ALPHABET = (
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
)
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


class VerificationError(Exception):
    """Raised when a credential fails Data Integrity verification."""


# ─── Multibase (base58btc) ───────────────────────────────────────────
# Data Integrity 1.0 §4.1 requires proofValue be encoded as a multibase
# value. eddsa-jcs-2022 uses prefix 'z' (base58btc, leading-zero-preserving).


def multibase_base58btc_encode(data: bytes) -> str:
    n_leading = 0
    for b in data:
        if b == 0:
            n_leading += 1
        else:
            break
    num = int.from_bytes(data, "big") if data else 0
    out_chars: list[str] = []
    while num > 0:
        num, rem = divmod(num, 58)
        out_chars.append(_B58_ALPHABET[rem])
    body = "".join(reversed(out_chars))
    return "z" + ("1" * n_leading) + body


def multibase_base58btc_decode(s: str) -> bytes:
    if not s or s[0] != "z":
        raise ValueError(
            "multibase prefix must be 'z' for base58btc (Data Integrity "
            "eddsa-jcs-2022), got "
            f"{s[:1]!r}"
        )
    body = s[1:]
    n_leading = 0
    for ch in body:
        if ch == "1":
            n_leading += 1
        else:
            break
    num = 0
    for ch in body[n_leading:]:
        try:
            num = num * 58 + _B58_INDEX[ch]
        except KeyError as e:
            raise ValueError(f"invalid base58btc character: {ch!r}") from e
    if num == 0:
        return b"\x00" * n_leading
    nbytes = (num.bit_length() + 7) // 8
    return (b"\x00" * n_leading) + num.to_bytes(nbytes, "big")


# ─── Credential construction ─────────────────────────────────────────


def make_credential(
    subject: Mapping[str, Any],
    issuer: str,
    credential_type: str | Sequence[str] = "VerifiableCredential",
    credential_id: str | None = None,
    valid_from: str | None = None,
    extra_contexts: Sequence[str] = (),
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an unsigned VC 2.0 document.

    ``subject`` becomes ``credentialSubject``. ``credential_type`` is coerced
    into the list form that VC 2.0 expects — the first entry is always
    ``VerifiableCredential``, followed by any additional subtypes.

    ``valid_from`` defaults to the current UTC time in ISO-8601 with a ``Z``
    suffix (VC 2.0 §4.4 recommends dateTimeStamp). Callers that want
    deterministic output should pass a fixed string.
    """
    if not issuer:
        raise ValueError("issuer is required (IRI)")

    types: list[str] = []
    if isinstance(credential_type, str):
        types = ["VerifiableCredential"]
        if credential_type and credential_type != "VerifiableCredential":
            types.append(credential_type)
    else:
        types = list(credential_type)
        if "VerifiableCredential" not in types:
            types.insert(0, "VerifiableCredential")

    contexts: list[str] = [VC_V2_CONTEXT]
    for c in extra_contexts:
        if c and c not in contexts:
            contexts.append(c)

    cred: dict[str, Any] = {
        "@context": contexts,
        "type": types,
        "issuer": issuer,
        "credentialSubject": dict(subject),
        "validFrom": valid_from or _utcnow_iso(),
    }
    if credential_id:
        cred["id"] = credential_id
    if extra_fields:
        for k, v in extra_fields.items():
            if k in {"proof", "@context", "type", "issuer", "credentialSubject"}:
                raise ValueError(
                    f"extra_fields key {k!r} collides with a reserved field"
                )
            cred[k] = v
    return cred


# ─── Data Integrity (eddsa-jcs-2022) sign / verify ───────────────────


def sign_credential(
    credential: Mapping[str, Any],
    private_key: Ed25519PrivateKey,
    verification_method: str,
    created: str | None = None,
    proof_purpose: str = DEFAULT_PROOF_PURPOSE,
) -> dict[str, Any]:
    """Return a copy of ``credential`` with an eddsa-jcs-2022 proof appended."""
    if "proof" in credential:
        raise ValueError(
            "credential already has a proof block; multi-proof chains are "
            "out of scope — construct a fresh credential instead"
        )
    if not verification_method:
        raise ValueError("verification_method (IRI) is required")

    base_doc = {k: v for k, v in credential.items()}
    proof_config: dict[str, Any] = {
        "type": DATA_INTEGRITY_PROOF_TYPE,
        "cryptosuite": CRYPTOSUITE,
        "created": created or _utcnow_iso(),
        "verificationMethod": verification_method,
        "proofPurpose": proof_purpose,
    }

    hash_data = _hash_data_integrity(base_doc, proof_config)
    signature = private_key.sign(hash_data)

    signed_proof = dict(proof_config)
    signed_proof["proofValue"] = multibase_base58btc_encode(signature)

    out = dict(base_doc)
    out["proof"] = signed_proof
    return out


def verify_credential(
    credential: Mapping[str, Any],
    public_key: Ed25519PublicKey,
) -> bool:
    """Verify an eddsa-jcs-2022 credential. Returns True on success.

    Raises VerificationError with a specific reason on failure — never
    returns False, so callers can inspect failure modes without parsing
    boolean ambiguity.
    """
    proof = credential.get("proof")
    if not isinstance(proof, Mapping):
        raise VerificationError("missing or malformed proof block")
    if proof.get("type") != DATA_INTEGRITY_PROOF_TYPE:
        raise VerificationError(
            f"unexpected proof type: {proof.get('type')!r}"
        )
    if proof.get("cryptosuite") != CRYPTOSUITE:
        raise VerificationError(
            f"unexpected cryptosuite: {proof.get('cryptosuite')!r}"
        )
    proof_value = proof.get("proofValue")
    if not isinstance(proof_value, str):
        raise VerificationError("missing proofValue")
    try:
        signature = multibase_base58btc_decode(proof_value)
    except ValueError as e:
        raise VerificationError(f"proofValue decode failed: {e}") from e

    base_doc = {k: v for k, v in credential.items() if k != "proof"}
    proof_config = {k: v for k, v in proof.items() if k != "proofValue"}

    hash_data = _hash_data_integrity(base_doc, proof_config)
    try:
        public_key.verify(signature, hash_data)
    except InvalidSignature as e:
        raise VerificationError("Ed25519 signature does not verify") from e
    return True


def _hash_data_integrity(
    document: Mapping[str, Any], proof_config: Mapping[str, Any]
) -> bytes:
    """Compute hashData per eddsa-jcs-2022.

    hashData = SHA-256(JCS(proofConfig)) || SHA-256(JCS(document))
    """
    canon_proof = canonicalize(proof_config)
    canon_doc = canonicalize(document)
    return hashlib.sha256(canon_proof).digest() + hashlib.sha256(canon_doc).digest()


def _utcnow_iso() -> str:
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")
