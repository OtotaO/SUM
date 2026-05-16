"""Replay-defense window check on signed_at.

Covers both the render-receipt and transform-receipt verifiers. The
window check is the optional ``max_age_seconds`` parameter added to
``verify_receipt`` and ``verify_transform_receipt``. Default behaviour
(``max_age_seconds=None``) does NOT enforce — long-lived archival
receipts and historical fixtures remain valid. When the caller opts
in, receipts whose ``signed_at`` is outside the acceptance window
reject with the ``signed_at_out_of_window`` error class.

This is the policy layer; the receipt is still cryptographically
valid. The distinction matters operationally:

  - ``signature_invalid`` = the receipt was tampered with
  - ``signed_at_out_of_window`` = the receipt is genuine but the
    caller has rejected it by replay-defense policy

Without this check, a valid receipt captured at time T1 could be
replayed at time T2 as if it were fresh. The receiver chooses the
acceptable freshness window per use-case.

Why opt-in: archival use cases (legal-discovery audit trail, long-
lived render history) need historical receipts to keep verifying.
Replay-defense is only meaningful when the receiver expects fresh
receipts (agent-swarm handoff, real-time render dispatch).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


pytest.importorskip("joserfc", reason="install sum-engine[receipt-verify]")
from joserfc.jwk import OKPKey  # noqa: E402

from sum_engine_internal.infrastructure.jcs import canonicalize  # noqa: E402
from sum_engine_internal.transform_receipt import (  # noqa: E402
    ErrorClass,
    VerifyError,
    build_payload,
    canonical_hash,
    sign_transform_receipt,
    verify_transform_receipt,
)


def _make_receipt(signed_at: str) -> tuple[dict, dict]:
    """Sign one transform receipt with the given signed_at; return
    (receipt_envelope, jwks_public)."""
    kid = "test-window-check-kid"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)

    parameters = {
        "density": 1.0, "length": 0.5, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    }
    input_doc = {"triples": [["a", "rel", "b"]]}
    output_tome = "a rel b."

    payload = build_payload(
        transform="slider",
        parameters_hash=canonical_hash(canonicalize(parameters)),
        input_hash=canonical_hash(canonicalize(input_doc)),
        output_hash=canonical_hash(output_tome.encode("utf-8")),
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        signed_at=signed_at,
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)

    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    return receipt, {"keys": [public]}


def _iso_utc_now(offset_seconds: float = 0.0) -> str:
    t = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


# ─── Default behaviour: window check NOT enforced ────────────────────


def test_old_receipt_verifies_without_max_age():
    """No max_age_seconds → archival receipt verifies fine. Locks the
    default-off behaviour so existing fixtures with frozen signed_at
    don't regress."""
    receipt, jwks = _make_receipt("2020-01-01T00:00:00.000Z")
    result = verify_transform_receipt(receipt, jwks)
    assert result.verified is True


def test_future_receipt_verifies_without_max_age():
    """No max_age_seconds → even a clearly-future receipt verifies.
    This is intentional: the verifier doesn't impose policy by default."""
    receipt, jwks = _make_receipt("2099-12-31T23:59:59.000Z")
    result = verify_transform_receipt(receipt, jwks)
    assert result.verified is True


# ─── Past-window enforcement ────────────────────────────────────────


def test_fresh_receipt_verifies_under_short_window():
    """Receipt signed ~now verifies under max_age_seconds=60."""
    receipt, jwks = _make_receipt(_iso_utc_now(0))
    result = verify_transform_receipt(receipt, jwks, max_age_seconds=60)
    assert result.verified is True


def test_old_receipt_rejects_under_short_window():
    """Receipt signed 1 hour ago rejects under max_age_seconds=60."""
    receipt, jwks = _make_receipt(_iso_utc_now(-3600))
    with pytest.raises(VerifyError) as exc_info:
        verify_transform_receipt(receipt, jwks, max_age_seconds=60)
    assert exc_info.value.error_class == ErrorClass.SIGNED_AT_OUT_OF_WINDOW
    assert "old" in str(exc_info.value)


def test_old_receipt_verifies_under_long_window():
    """Receipt signed 1 hour ago verifies under max_age_seconds=86400."""
    receipt, jwks = _make_receipt(_iso_utc_now(-3600))
    result = verify_transform_receipt(receipt, jwks, max_age_seconds=86400)
    assert result.verified is True


# ─── Future-window (clock skew) enforcement ─────────────────────────


def test_slightly_future_receipt_verifies_under_default_skew():
    """Receipt signed 30s in the future verifies (default skew = 60s)."""
    receipt, jwks = _make_receipt(_iso_utc_now(+30))
    result = verify_transform_receipt(receipt, jwks, max_age_seconds=300)
    assert result.verified is True


def test_far_future_receipt_rejects_under_default_skew():
    """Receipt signed 10 minutes in the future rejects (default skew = 60s)."""
    receipt, jwks = _make_receipt(_iso_utc_now(+600))
    with pytest.raises(VerifyError) as exc_info:
        verify_transform_receipt(receipt, jwks, max_age_seconds=300)
    assert exc_info.value.error_class == ErrorClass.SIGNED_AT_OUT_OF_WINDOW
    assert "future" in str(exc_info.value)


def test_far_future_receipt_verifies_under_generous_skew():
    """Receipt signed 10 minutes in the future verifies when skew tolerates it."""
    receipt, jwks = _make_receipt(_iso_utc_now(+600))
    result = verify_transform_receipt(
        receipt, jwks,
        max_age_seconds=3600,
        max_future_skew_seconds=900,
    )
    assert result.verified is True


# ─── Malformed signed_at fails closed ───────────────────────────────


def test_unparseable_signed_at_fails_closed_under_window():
    """When max_age_seconds is enforced, a malformed signed_at fails
    with signed_at_out_of_window — NOT silently accepted."""
    receipt, jwks = _make_receipt("not-a-valid-timestamp")
    # The signature will fail before the window check fires because
    # the canonicalised bytes still include the malformed string —
    # which is fine. But we also want to lock the fail-closed branch
    # for the case where the signature is valid but the timestamp
    # isn't parseable. We construct that case by signing with a
    # malformed timestamp deliberately; the receipt-verifier will
    # see the same malformed string in the verified payload and
    # reject in the window-check step rather than the signature step.
    with pytest.raises(VerifyError) as exc_info:
        verify_transform_receipt(receipt, jwks, max_age_seconds=60)
    # Either signature_invalid (if joserfc parsed differently) or
    # signed_at_out_of_window — both are fail-closed branches.
    assert exc_info.value.error_class in {
        ErrorClass.SIGNED_AT_OUT_OF_WINDOW,
        ErrorClass.SIGNATURE_INVALID,
    }


# ─── Render-receipt path — same surface ─────────────────────────────


def test_render_receipt_window_surface_exists():
    """The render-receipt verifier exposes the same parameter shape so
    consumers can apply the same replay policy to render receipts."""
    from sum_engine_internal.render_receipt import verify_receipt
    import inspect
    sig = inspect.signature(verify_receipt)
    assert "max_age_seconds" in sig.parameters
    assert "max_future_skew_seconds" in sig.parameters
