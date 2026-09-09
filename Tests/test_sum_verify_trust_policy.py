"""Relying-party policy must never confuse unchecked facts with trusted ones."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("joserfc")

from sum_verify import (
    RevocationSnapshot,
    SumVerifyError,
    TrustPolicy,
    TrustPolicyError,
    key_fingerprint,
    verify,
    verify_report,
    verify_transform_receipt,
)


def _load(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(params=["render", "transform", "meaning", "chain"])
def case(request):
    if request.param in ("render", "transform"):
        fixture = _load(f"fixtures/{request.param}_receipts/positive_control.json")
        return fixture["receipt"], fixture["jwks"]
    folder, name = {
        "meaning": ("meaning_receipts_billsum", "meaning_risk_receipt.billsum.golden.json"),
        "chain": ("chain_receipts_billsum", "chain_receipt.billsum.golden.json"),
    }[request.param]
    return _load(f"fixtures/{folder}/{name}"), _load(f"fixtures/{folder}/jwks.json")


def _signed_at(receipt):
    return datetime.fromisoformat(receipt["payload"]["signed_at"].replace("Z", "+00:00"))


def _pins(jwks):
    return {k["kid"]: key_fingerprint(k) for k in jwks["keys"]}


def test_default_api_unchanged_and_unchecked_report(case):
    receipt, jwks = case
    result = verify(receipt, jwks)
    report = verify_report(receipt, jwks)
    assert report.result == result
    assert report.payload == receipt["payload"]
    assert report.to_dict()["cryptographically_verified"] is True
    assert report.policy_applied is False
    assert report.checks["signature_and_structure"]["status"] == "passed"
    for name in ("trusted_key", "revocation", "revocation_freshness", "receipt_freshness",
                 "artifact_bindings", "organization_identity", "sampling_assumptions",
                 "source_remeasurement", "loss_arithmetic_replay", "chain_hop_artifacts"):
        assert report.checks[name]["status"] == "not_checked"


def test_pins_key_material_and_checks_fresh_offline_snapshot(case):
    receipt, jwks = case
    now = _signed_at(receipt) + timedelta(seconds=30)
    policy = TrustPolicy(
        trusted_keys=_pins(jwks),
        revocations=RevocationSnapshot(frozenset(), now),
        require_revocations=True, max_revocation_age_seconds=60, max_age_seconds=60,
    )
    report = verify_report(receipt, jwks, trust_policy=policy, now=now)
    for name in ("trusted_key", "revocation", "revocation_freshness", "receipt_freshness"):
        assert report.checks[name]["status"] == "passed"
    assert report.checks["organization_identity"]["status"] == "not_checked"
    assert verify(receipt, jwks, trust_policy=TrustPolicy(trusted_keys=_pins(jwks))) == report.result


def test_valid_signature_from_unpinned_key_rejected(case):
    receipt, jwks = case
    with pytest.raises(TrustPolicyError, match="outside") as exc:
        verify(receipt, jwks, trust_policy=TrustPolicy(trusted_keys={}))
    assert isinstance(exc.value, SumVerifyError)
    assert exc.value.check == "untrusted_key"


@pytest.mark.parametrize("archival", [False, True])
def test_revoked_key_rejected_even_for_historical_signed_time(case, archival):
    receipt, jwks = case
    now = _signed_at(receipt) + timedelta(days=365)
    snapshot = RevocationSnapshot.from_document({
        "schema": "sum.revoked_kids.v1", "revoked": [{
            "kid": receipt["kid"], "effective_revocation_at": now.isoformat(),
        }],
    }, retrieved_at=now)
    with pytest.raises(TrustPolicyError) as exc:
        verify_report(receipt, jwks, trust_policy=TrustPolicy(revocations=snapshot, archival=archival), now=now)
    assert exc.value.check == "revoked_kid"


@pytest.mark.parametrize("limit", [None, 300])
def test_missing_revocation_snapshot_fails_when_required(case, limit):
    receipt, jwks = case
    with pytest.raises(TrustPolicyError) as exc:
        verify(receipt, jwks, trust_policy=TrustPolicy(require_revocations=True, max_revocation_age_seconds=limit))
    assert exc.value.check == "revocation_unavailable"


@pytest.mark.parametrize("offset", [-61, 61])
def test_stale_or_future_snapshot_fails(case, offset):
    receipt, jwks = case
    now = _signed_at(receipt)
    policy = TrustPolicy(revocations=RevocationSnapshot(frozenset(), now + timedelta(seconds=offset)), max_revocation_age_seconds=60)
    with pytest.raises(TrustPolicyError) as exc:
        verify_report(receipt, jwks, trust_policy=policy, now=now)
    assert exc.value.check == "revocation_snapshot_out_of_window"


@pytest.mark.parametrize("offset", [-61, 61])
def test_receipt_freshness_past_and_future(case, offset):
    receipt, jwks = case
    with pytest.raises(TrustPolicyError) as exc:
        verify_report(receipt, jwks, trust_policy=TrustPolicy(max_age_seconds=60), now=_signed_at(receipt) + timedelta(seconds=offset))
    assert exc.value.check == "signed_at_out_of_window"


def test_archival_is_deliberately_unchecked_not_fresh(case):
    receipt, jwks = case
    report = verify_report(receipt, jwks, trust_policy=TrustPolicy(archival=True), now=_signed_at(receipt) + timedelta(days=3650))
    assert report.checks["receipt_freshness"]["status"] == "not_checked"
    assert "historical existence" in report.checks["receipt_freshness"]["detail"]


def test_bound_artifacts_match_and_missing_required_hash_fails(case):
    receipt, jwks = case
    payload = receipt["payload"]
    name = next((k for k in payload if k.endswith("_hash")), "source_chain_hash")
    if name not in payload:  # chain fixtures have no outer source commitment
        with pytest.raises(TrustPolicyError) as exc:
            verify(receipt, jwks, trust_policy=TrustPolicy(expected_bindings={name: "sha256-example"}))
        assert exc.value.check == "source_binding_mismatch"
    else:
        expected = {name: payload[name]}
        policy = TrustPolicy(expected_bindings=expected, required_bindings={name})
        expected[name] = "mutated after policy construction"
        report = verify_report(receipt, jwks, trust_policy=policy)
        assert report.checks["artifact_bindings"]["status"] == "passed"
        assert report.checks["source_remeasurement"]["status"] == "not_checked"
        with pytest.raises(TrustPolicyError) as exc:
            verify(receipt, jwks, trust_policy=TrustPolicy(expected_bindings={name: "wrong"}))
        assert exc.value.check == "source_binding_mismatch"
    with pytest.raises(TrustPolicyError) as exc:
        verify(receipt, jwks, trust_policy=TrustPolicy(required_bindings={name}))
    assert exc.value.check == "source_binding_missing"


def test_fingerprint_ignores_untrusted_labels_but_not_material(case):
    _, jwks = case
    key = jwks["keys"][0]
    other = dict(key, kid="different claimed organization", use="other")
    assert key_fingerprint(key) == key_fingerprint(other)
    other["x"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    assert key_fingerprint(key) != key_fingerprint(other)


def test_wrong_key_reusing_trusted_kid_does_not_pass_pin():
    from joserfc.jwk import OKPKey

    from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope

    fixture = _load("fixtures/render_receipts/positive_control.json")
    key = OKPKey.generate_key("Ed25519")
    kid = fixture["receipt"]["kid"]
    receipt = sign_jose_envelope(fixture["receipt"]["payload"], private_jwk=key.as_dict(private=True), kid=kid)
    receipt["schema"] = fixture["receipt"]["schema"]
    jwks = {"keys": [dict(key.as_dict(private=False), kid=kid)]}
    with pytest.raises(TrustPolicyError) as exc:
        verify(receipt, jwks, trust_policy=TrustPolicy(trusted_keys=_pins(fixture["jwks"])))
    assert exc.value.check == "untrusted_key"


def test_replay_report_does_not_imply_source_or_sampling_validation():
    receipt = _load("fixtures/meaning_receipts_billsum/meaning_risk_receipt.billsum.golden.json")
    jwks = _load("fixtures/meaning_receipts_billsum/jwks.json")
    losses = _load("fixtures/meaning_receipts_billsum/losses_billsum.json")
    report = verify_report(receipt, jwks, losses=losses)
    assert report.checks["loss_arithmetic_replay"]["status"] == "passed"
    assert report.checks["source_remeasurement"]["status"] == "not_checked"
    assert report.checks["sampling_assumptions"]["status"] == "not_checked"


def test_transform_legacy_revocation_parity():
    fixture = _load("fixtures/transform_receipts/positive_control.json")
    receipt, jwks = fixture["receipt"], fixture["jwks"]
    assert verify_transform_receipt(receipt, jwks, revoked_kids=[]).verified
    with pytest.raises(SumVerifyError) as exc:
        verify_transform_receipt(receipt, jwks, revoked_kids=[{
            "kid": receipt["kid"], "effective_revocation_at": receipt["payload"]["signed_at"],
        }])
    assert exc.value.error_class == "revoked_kid"
    assert verify_transform_receipt(receipt, jwks, revoked_kids=[{
        "kid": receipt["kid"], "effective_revocation_at": "2099-01-01T00:00:00Z",
    }]).verified  # legacy path preserves effective-time semantics


@pytest.mark.parametrize("kwargs", [
    {"max_age_seconds": -1}, {"max_age_seconds": True},
    {"max_age_seconds": 1.5}, {"max_revocation_age_seconds": -1},
    {"max_future_skew_seconds": None}, {"archival": "yes"},
    {"archival": True, "max_age_seconds": 1},
    {"required_bindings": {"provider"}}, {"expected_bindings": {"input_hash": None}},
])
def test_policy_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TrustPolicy(**kwargs)


@pytest.mark.parametrize("document", [{}, {"schema": "sum.revoked_kids.v1"},
    {"schema": "sum.revoked_kids.v1", "revoked": [None]},
    {"schema": "sum.revoked_kids.v1", "revoked": [{"kid": ""}]},
])
def test_malformed_snapshot_never_becomes_empty_success(document):
    with pytest.raises(ValueError):
        RevocationSnapshot.from_document(document, retrieved_at=datetime.now(timezone.utc))


def test_timezone_and_conflicting_clock_configuration_rejected(case):
    receipt, jwks = case
    with pytest.raises(ValueError):
        RevocationSnapshot(frozenset(), "2026-09-09T00:00:00")
    with pytest.raises(ValueError):
        verify_report(receipt, jwks, trust_policy=TrustPolicy(max_age_seconds=60), now=datetime(2026, 9, 9))
    with pytest.raises(ValueError):
        verify(receipt, jwks, trust_policy=TrustPolicy(), max_age_seconds=60)


def test_signature_failure_still_rejected_before_policy(case):
    receipt, jwks = case
    bad = copy.deepcopy(receipt)
    bad["payload"]["signed_at"] = "2000-01-01T00:00:00Z"
    with pytest.raises(SumVerifyError) as exc:
        verify(bad, jwks, trust_policy=TrustPolicy(trusted_keys={}))
    assert not isinstance(exc.value, TrustPolicyError)


def test_revoked_material_cannot_be_renamed_or_hidden_behind_trusted_alias():
    from joserfc.jwk import OKPKey

    from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope

    fixture = _load("fixtures/render_receipts/positive_control.json")
    key = OKPKey.generate_key("Ed25519")
    old_kid, new_kid = "revoked-original", "renamed-alias"
    public = dict(key.as_dict(private=False), kid=new_kid)
    receipt = sign_jose_envelope(fixture["receipt"]["payload"], private_jwk=key.as_dict(private=True), kid=new_kid)
    receipt["schema"] = fixture["receipt"]["schema"]
    fingerprint = key_fingerprint(public)
    snapshot = RevocationSnapshot({old_kid}, datetime.now(timezone.utc))
    for catalog, expected_error in [
        ({old_kid: fingerprint}, "untrusted_key"),
        ({old_kid: fingerprint, new_kid: fingerprint}, "revoked_kid"),
        ({new_kid: fingerprint}, "revocation_key_unresolved"),
    ]:
        with pytest.raises(TrustPolicyError) as exc:
            verify(receipt, {"keys": [public]}, trust_policy=TrustPolicy(trusted_keys=catalog, revocations=snapshot))
        assert exc.value.check == expected_error


def test_report_metadata_is_a_snapshot_and_checks_cannot_be_relabelled(case):
    receipt, jwks = case
    report = verify_report(receipt, jwks)
    before = report.to_dict()
    receipt["payload"]["statistical_scope"] = "invented after verification"
    report.payload["statistical_scope"] = "mutated returned copy"
    assert report.to_dict() == before
    assert report.payload.get("statistical_scope") != "invented after verification"
    with pytest.raises(TypeError):
        report.checks["trusted_key"]["status"] = "passed"


@pytest.mark.parametrize("receipt", [None, [], {}, {"payload": [1]}])
def test_transform_revocation_option_preserves_malformed_input_error_taxonomy(receipt):
    with pytest.raises(SumVerifyError):
        verify_transform_receipt(receipt, {"keys": []}, revoked_kids=[])


def test_trusted_key_catalog_is_copied(case):
    receipt, jwks = case
    catalog = _pins(jwks)
    policy = TrustPolicy(trusted_keys=catalog)
    catalog.clear()
    assert verify_report(receipt, jwks, trust_policy=policy).checks["trusted_key"]["status"] == "passed"
