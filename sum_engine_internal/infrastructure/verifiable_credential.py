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

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

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
    # DID helpers (did:web + did:key)
    "DID_CONTEXT",
    "ED25519_MULTICODEC_PREFIX",
    "build_did_web_document",
    "did_web_verification_method",
    "ed25519_to_did_key",
    "ed25519_public_key_multibase",
]


VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2"
DATA_INTEGRITY_PROOF_TYPE = "DataIntegrityProof"
CRYPTOSUITE = "eddsa-jcs-2022"
DEFAULT_PROOF_PURPOSE = "assertionMethod"

# W3C DID core context — load-bearing for did:web resolver compatibility.
DID_CONTEXT = [
    "https://www.w3.org/ns/did/v1",
    "https://w3id.org/security/multikey/v1",
]

# Multicodec prefix for Ed25519 public keys (0xed 0x01 little-endian) per
# https://github.com/multiformats/multicodec. Used by ed25519_to_did_key and
# by the did:web publicKeyMultibase field — resolvers parse this prefix to
# identify the key type without out-of-band metadata.
ED25519_MULTICODEC_PREFIX = b"\xed\x01"

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


# ─── DID helpers (did:web + did:key) ─────────────────────────────────
# These turn a raw Ed25519 public key into the DID-compatible identifiers
# that any VC 2.0 verifier can resolve via the DIF Universal Resolver
# (https://dev.uniresolver.io) or an equivalent. Until now, SUM emitted
# credentials with `verificationMethod="did:example:issuer#key-1"` — a
# placeholder that no real verifier can dereference. The helpers below
# make the issuer identifier resolvable without requiring SUM to run
# additional DID infrastructure.
#
# Two supported schemes:
#   did:key   — self-resolving. The entire public key is encoded in the
#               DID itself. Zero hosting; just hand someone the DID and
#               they can verify any credential it signed. Ideal for
#               ephemeral or air-gapped use.
#   did:web   — domain-resolving. The DID is `did:web:<domain>` and the
#               public key is served at `https://<domain>/.well-known/did.json`.
#               Ideal for a hosted SUM deployment (e.g. Cloudflare Pages).


def ed25519_public_key_multibase(public_key: Ed25519PublicKey) -> str:
    """Encode an Ed25519 public key as a multicodec + base58btc string.

    Result shape: ``"z<base58btc>"`` where the base58btc-encoded payload is
    ``ED25519_MULTICODEC_PREFIX || raw_32_byte_public_key``. This is the
    `publicKeyMultibase` value used in DID Core documents and in the
    `did:key:z6Mk...` format.
    """
    from cryptography.hazmat.primitives import serialization

    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return multibase_base58btc_encode(ED25519_MULTICODEC_PREFIX + raw)


def ed25519_to_did_key(public_key: Ed25519PublicKey) -> str:
    """Derive a `did:key` DID URI for an Ed25519 public key.

    The entire public key is encoded in the DID identifier itself — there
    is nothing to host or resolve over the network. A verifier with only
    the DID can reconstruct the public key and verify any credential the
    corresponding private key signed.

    Returns: ``"did:key:z6Mk..."`` (always begins with ``z6Mk`` for
    Ed25519 because the multicodec prefix 0xed 0x01 encodes to that
    sequence in base58btc).
    """
    return f"did:key:{ed25519_public_key_multibase(public_key)}"


def did_web_verification_method(domain: str, key_id: str = "key-1") -> str:
    """Compose the `verificationMethod` URI for a `did:web` issuer.

    Args:
        domain: The hosted domain (e.g. "sum-demo.pages.dev"). MUST NOT
                include a scheme or trailing slash — the `did:web` method
                specification derives `https://{domain}/.well-known/did.json`
                from the domain alone.
        key_id: The key fragment (default "key-1"). Must match an entry
                in the hosted DID document's `verificationMethod` array.

    Returns: ``"did:web:{domain}#{key_id}"``.
    """
    if "://" in domain or domain.endswith("/"):
        raise ValueError(
            f"did:web domain must not include scheme or trailing slash, got {domain!r}"
        )
    return f"did:web:{domain}#{key_id}"


def build_did_web_document(
    domain: str,
    public_key: Ed25519PublicKey,
    key_id: str = "key-1",
    also_known_as: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the DID document to serve at `https://{domain}/.well-known/did.json`.

    The resulting dict is ready to ``json.dump`` to the file. Any DID
    resolver (DIF Universal Resolver, Veramo, Spruce ssi, digitalbazaar's
    stack) will accept this shape for `did:web` resolution and consume
    the embedded multikey for signature verification.

    Args:
        domain: The hosted domain, no scheme, no trailing slash.
        public_key: Ed25519 public key to publish as the `assertionMethod`
                    + `authentication` key.
        key_id: Fragment used in the verificationMethod id and in the
                `assertionMethod` / `authentication` relation arrays.
        also_known_as: Optional list of equivalent DIDs (e.g. a `did:key`
                       self-resolving URI can be listed here so verifiers
                       can fall back to self-resolution if the domain is
                       temporarily unreachable).

    Returns: a dict conforming to W3C DID Core (Recommendation, 2022) +
    the `security/multikey/v1` context, which is what every current VC
    2.0 verifier looks for.
    """
    did = f"did:web:{domain}"
    method_id = f"{did}#{key_id}"
    doc: dict[str, Any] = {
        "@context": list(DID_CONTEXT),
        "id": did,
        "verificationMethod": [
            {
                "id": method_id,
                "type": "Multikey",
                "controller": did,
                "publicKeyMultibase": ed25519_public_key_multibase(public_key),
            }
        ],
        "assertionMethod": [method_id],
        "authentication": [method_id],
    }
    if also_known_as:
        doc["alsoKnownAs"] = list(also_known_as)
    return doc
