"""Explicit, offline receiver policy layered above the receipt wire verifier.

A caller supplies trust material from its own trusted channel. This module
never downloads keys or revocations, and never infers an organization from a kid.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.infrastructure.jose_envelope import SumVerifyError


class TrustPolicyError(ValueError, SumVerifyError):
    """A valid receipt failed an explicitly requested receiver check."""

    def __init__(self, check: str, message: str) -> None:
        self.check = check
        self.error_class = check
        super().__init__(message)


def _time(value: str | datetime, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(parsed, datetime) or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp") from exc


def _seconds(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")


def key_fingerprint(jwk: Mapping[str, Any]) -> str:
    """SHA-256 hex of JCS {crv, kty, x}; pins Ed25519 material, not a key label.

    Only public key members participate. A key's kid, use, or issuer label
    cannot change its fingerprint. This is not an organization identity check.
    """
    if not isinstance(jwk, Mapping) or jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
        raise ValueError("expected an OKP/Ed25519 public JWK")
    x = jwk.get("x")
    if not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", x):
        raise ValueError("JWK x must encode a 32-byte Ed25519 public key")
    raw = base64.urlsafe_b64decode(x + "=")
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != x:
        raise ValueError("JWK x must use canonical base64url encoding")
    return "sha256-" + hashlib.sha256(canonicalize({"kty": "OKP", "crv": "Ed25519", "x": x})).hexdigest()


@dataclass(frozen=True)
class RevocationSnapshot:
    """Caller-trusted offline snapshot, with a caller-recorded retrieval time.

    Revoked key IDs are rejected even for backdated receipts. A compromised
    signer can forge signed_at. Archival mode does not waive this check.
    Authentication of the snapshot and retrieval clock remain caller duties.
    """

    revoked_kids: frozenset[str]
    retrieved_at: datetime | str

    def __post_init__(self) -> None:
        if isinstance(self.revoked_kids, str):
            raise ValueError("revoked_kids must be a collection of key IDs")
        kids = frozenset(self.revoked_kids)
        if any(not isinstance(k, str) or not k for k in kids):
            raise ValueError("revoked_kids must contain nonempty strings")
        object.__setattr__(self, "revoked_kids", kids)
        object.__setattr__(self, "retrieved_at", _time(self.retrieved_at, "retrieved_at"))

    @classmethod
    def from_document(cls, document: Any, *, retrieved_at: datetime | str) -> RevocationSnapshot:
        """Read a sum.revoked_kids.v1 document; malformed entries fail closed.

        All listed keys are treated as revoked. Effective times in the
        legacy feed do not authenticate a receipt's historical existence.
        """
        if not isinstance(document, dict) or document.get("schema") != "sum.revoked_kids.v1":
            raise ValueError("expected a sum.revoked_kids.v1 snapshot")
        entries = document.get("revoked")
        if not isinstance(entries, list):
            raise ValueError("snapshot.revoked must be a list")
        kids = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("kid"), str) or not entry["kid"]:
                raise ValueError("every revocation entry needs a nonempty kid")
            kids.append(entry["kid"])
        return cls(frozenset(kids), retrieved_at)


@dataclass(frozen=True)
class TrustPolicy:
    """Checks requested by the relying caller; unspecified checks stay unchecked.

    trusted_keys maps caller-trusted key IDs to public-key fingerprints. Keep
    historical keys in the catalog when consuming revocations: unresolved
    revoked IDs fail closed rather than permitting a key to be renamed.
    expected_bindings maps signed top-level *_hash fields to values the
    caller independently computed. required_bindings additionally requires
    those fields to be checked, not merely present in a signed payload.
    This compares commitments; it does not retrieve sources or judge truth.
    """

    trusted_keys: Mapping[str, str] | None = None
    revocations: RevocationSnapshot | None = None
    require_revocations: bool = False
    max_revocation_age_seconds: int | None = None
    max_age_seconds: int | None = None
    max_future_skew_seconds: int = 60
    archival: bool = False
    expected_bindings: Mapping[str, str] = field(default_factory=dict)
    required_bindings: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in ("max_revocation_age_seconds", "max_age_seconds", "max_future_skew_seconds"):
            _seconds(getattr(self, name), name)
        if self.max_future_skew_seconds is None:
            raise ValueError("max_future_skew_seconds must be an integer")
        if type(self.archival) is not bool or type(self.require_revocations) is not bool:
            raise ValueError("archival and require_revocations must be booleans")
        if self.archival and self.max_age_seconds is not None:
            raise ValueError("archival and max_age_seconds are mutually exclusive")
        if self.revocations is not None and not isinstance(self.revocations, RevocationSnapshot):
            raise ValueError("revocations must be a RevocationSnapshot")
        if self.trusted_keys is not None:
            pins = dict(self.trusted_keys)
            if any(not isinstance(k, str) or not k for k in pins):
                raise ValueError("trusted_keys must map nonempty key IDs to fingerprints")
            if any(not isinstance(p, str) or not re.fullmatch(r"sha256-[0-9a-f]{64}", p) for p in pins.values()):
                raise ValueError("trusted_keys must contain SHA-256 fingerprints")
            object.__setattr__(self, "trusted_keys", MappingProxyType(pins))
        bindings = dict(self.expected_bindings)
        required = frozenset(self.required_bindings)
        for name in set(bindings) | required:
            if not isinstance(name, str) or not name.endswith("_hash"):
                raise ValueError("bindings must name signed top-level *_hash fields")
        if any(not isinstance(v, str) or not v for v in bindings.values()):
            raise ValueError("expected bindings must be nonempty hash strings")
        object.__setattr__(self, "expected_bindings", MappingProxyType(bindings))
        object.__setattr__(self, "required_bindings", required)


@dataclass(frozen=True)
class VerificationReport:
    """Explicit check outcomes for one verified envelope, never blanket trust."""

    result: Any
    receipt_schema: str
    kid: str
    policy_applied: bool
    checks: Mapping[str, Mapping[str, str]]
    _payload_snapshot: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        payload = self.result if isinstance(self.result, dict) else self.result.payload
        object.__setattr__(self, "_payload_snapshot", copy.deepcopy(payload))
        object.__setattr__(self, "checks", MappingProxyType({
            name: MappingProxyType(dict(check)) for name, check in self.checks.items()
        }))

    @property
    def payload(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload_snapshot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sum.verification_report.v1",
            "receipt_schema": self.receipt_schema,
            "kid": self.kid,
            "cryptographically_verified": True,
            "policy_applied": self.policy_applied,
            "checks": {key: dict(value) for key, value in self.checks.items()},
            "statistical_scope": self.payload.get("statistical_scope", "not_declared"),
        }


def policy_checks(
    envelope: dict[str, Any], jwks: dict[str, Any], policy: TrustPolicy | None,
    *, now: datetime | None = None,
) -> dict[str, dict[str, str]]:
    """Evaluate policy only after the family verifier authenticated the payload."""
    checks = {name: {"status": "not_checked", "detail": detail} for name, detail in {
        "trusted_key": "Supplied JWKS is not independently authenticated by this verifier.",
        "revocation": "No caller-trusted revocation snapshot was supplied.",
        "revocation_freshness": "No snapshot age limit was requested.",
        "receipt_freshness": "No receipt age limit was requested.",
        "artifact_bindings": "No independently computed artifact hashes were supplied.",
        "organization_identity": "A key ID or supplied JWKS does not establish organization identity.",
        "source_remeasurement": "This verifier does not rerun source extraction or model judgments.",
        "sampling_assumptions": "Receipt arithmetic does not validate sampling or deployment assumptions.",
    }.items()}
    if policy is None:
        return checks
    if not isinstance(policy, TrustPolicy):
        raise TypeError("trust_policy must be a TrustPolicy")
    clock = _time(now or datetime.now(timezone.utc), "now")
    kid, payload = envelope["kid"], envelope["payload"]

    def passed(name: str, detail: str) -> None:
        checks[name] = {"status": "passed", "detail": detail}

    fingerprint = None
    if policy.trusted_keys is not None:
        # The wire verifier uses the FIRST matching kid. Pin that exact key.
        selected = next(k for k in jwks["keys"] if isinstance(k, dict) and k.get("kid") == kid)
        try:
            fingerprint = key_fingerprint(selected)
        except ValueError as exc:
            raise TrustPolicyError("untrusted_key", "Signing key does not have a canonical public-key fingerprint") from exc
        if policy.trusted_keys.get(kid) != fingerprint:
            raise TrustPolicyError("untrusted_key", "Verified signing key is outside the caller's pinned key set")
        passed("trusted_key", "Signing key material and its key ID match the caller's trusted catalog.")
    snapshot = policy.revocations
    if snapshot is None and (policy.require_revocations or policy.max_revocation_age_seconds is not None):
        raise TrustPolicyError("revocation_unavailable", "The requested revocation snapshot is missing")
    if snapshot is not None:
        age = (clock - snapshot.retrieved_at).total_seconds()
        if age < -policy.max_future_skew_seconds:
            raise TrustPolicyError("revocation_snapshot_out_of_window", "Snapshot retrieval time is in the future")
        if policy.max_revocation_age_seconds is not None:
            if age > policy.max_revocation_age_seconds:
                raise TrustPolicyError("revocation_snapshot_out_of_window", "Revocation snapshot is older than the caller's limit")
            passed("revocation_freshness", "Caller-recorded retrieval time is inside the requested age window.")
        revoked_material = set()
        if policy.trusted_keys is not None:
            unknown = snapshot.revoked_kids - policy.trusted_keys.keys()
            if unknown:
                raise TrustPolicyError("revocation_key_unresolved", "Trusted key catalog lacks revoked key IDs; include historical key material to detect aliases")
            revoked_material = {policy.trusted_keys[k] for k in snapshot.revoked_kids}
        if kid in snapshot.revoked_kids or fingerprint in revoked_material:
            raise TrustPolicyError("revoked_kid", "Signing key is revoked in the caller's snapshot; signed_at cannot establish pre-compromise existence")
        passed("revocation", (
            "Supplied key ID and its trusted material aliases are absent from the snapshot; newer revocations are not established."
            if policy.trusted_keys is not None else
            "Supplied key ID is absent from the snapshot. Material aliases were not resolved without a trusted key catalog."
        ))
    if policy.archival:
        checks["receipt_freshness"]["detail"] = "Archival mode: receipt age was deliberately not checked; this is not proof of historical existence."
    elif policy.max_age_seconds is not None:
        try:
            signed_at = _time(payload.get("signed_at"), "signed_at")
        except ValueError as exc:
            raise TrustPolicyError("signed_at_out_of_window", str(exc)) from exc
        age = (clock - signed_at).total_seconds()
        if age > policy.max_age_seconds or age < -policy.max_future_skew_seconds:
            raise TrustPolicyError("signed_at_out_of_window", "Receipt timestamp is outside the caller's acceptance window")
        passed("receipt_freshness", "Signed timestamp is inside the requested age window; signer clock is still an assertion.")
    missing = policy.required_bindings - policy.expected_bindings.keys()
    if missing:
        raise TrustPolicyError("source_binding_missing", f"Required independently computed bindings were not supplied: {sorted(missing)}")
    for name, expected in policy.expected_bindings.items():
        if payload.get(name) != expected:
            raise TrustPolicyError("source_binding_mismatch", f"Signed {name} does not match the caller's computed hash")
    if policy.expected_bindings:
        passed("artifact_bindings", "Matched caller-computed signed hash fields: " + ", ".join(sorted(policy.expected_bindings)))
    return checks
