"""Receipt-type confusion: an unsigned discriminator must not pick the validator.

``receipt["schema"]`` sits OUTSIDE the JWS. Only ``payload`` is signed, and the
protected header carries no ``typ``/``cty``. So ``schema`` is attacker-editable
on a receipt whose signature is entirely genuine.

The pre-existing defence only caught mislabels AWAY from a verifier's expected
schema (fixtures/transform_receipts/schema_confusion_render_receipt.json). The
hole was relabelling TOWARD it: set ``schema`` to exactly what the target
verifier expects and its schema check passes, on a payload that verifier has
never validated. A genuine signature over a DIFFERENT family's payload is still
a genuine signature, so the crypto cannot close this.

This is the same bug class as JWT alg-confusion. The fix is a payload-shape
gate per family: a foreign payload cannot carry this family's required fields.

The real fix is to bind the schema INTO the signature, which needs a wire
version bump and would require re-signing every committed receipt. These tests
pin the no-bump defence; they do not make the bump unnecessary.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] extra not installed")

_REPO = Path(__file__).resolve().parents[1]
_MEANING = _REPO / "fixtures" / "meaning_receipts_billsum"
_CHAIN = _REPO / "fixtures" / "chain_receipts_billsum"


def _load(p: Path):
    return json.loads(p.read_text("utf-8"))


@pytest.fixture(scope="module")
def meaning_golden():
    return _load(_MEANING / "meaning_risk_receipt.billsum.golden.json")


@pytest.fixture(scope="module")
def meaning_jwks():
    return _load(_MEANING / "jwks.json")


@pytest.fixture(scope="module")
def chain_golden():
    return _load(_CHAIN / "chain_receipt.billsum.golden.json")


def _relabel(receipt: dict, schema: str) -> dict:
    out = copy.deepcopy(receipt)
    out["schema"] = schema
    return out


def test_render_verifier_rejects_relabelled_meaning_receipt(meaning_golden, meaning_jwks):
    """The exact reproduction: one string edit to a COMMITTED golden."""
    from sum_engine_internal.render_receipt.verifier import (
        SUPPORTED_SCHEMA, VerifyError, verify_receipt,
    )
    evil = _relabel(meaning_golden, SUPPORTED_SCHEMA)
    with pytest.raises(VerifyError) as ei:
        verify_receipt(evil, meaning_jwks)
    assert ei.value.error_class == "malformed_receipt"


def test_render_verifier_rejects_relabelled_chain_receipt(chain_golden, meaning_jwks):
    from sum_engine_internal.render_receipt.verifier import (
        SUPPORTED_SCHEMA, VerifyError, verify_receipt,
    )
    with pytest.raises(VerifyError):
        verify_receipt(_relabel(chain_golden, SUPPORTED_SCHEMA), meaning_jwks)


def test_transform_verifier_rejects_relabelled_meaning_receipt(meaning_golden, meaning_jwks):
    from sum_engine_internal.transform_receipt.format import SUPPORTED_SCHEMA
    from sum_engine_internal.transform_receipt.verifier import (
        VerifyError, verify_transform_receipt,
    )
    with pytest.raises(VerifyError) as ei:
        verify_transform_receipt(_relabel(meaning_golden, SUPPORTED_SCHEMA), meaning_jwks)
    assert ei.value.error_class == "malformed_receipt"


def test_transform_verifier_rejects_relabelled_chain_receipt(chain_golden, meaning_jwks):
    from sum_engine_internal.transform_receipt.format import SUPPORTED_SCHEMA
    from sum_engine_internal.transform_receipt.verifier import (
        VerifyError, verify_transform_receipt,
    )
    with pytest.raises(VerifyError):
        verify_transform_receipt(_relabel(chain_golden, SUPPORTED_SCHEMA), meaning_jwks)


@pytest.mark.parametrize("bad_payload", [None, [], "a string", 5])
def test_verifiers_reject_non_object_payload(bad_payload, meaning_jwks):
    """Totality: a non-object payload yields a verdict, never a crash."""
    from sum_engine_internal.render_receipt.verifier import (
        SUPPORTED_SCHEMA as R, VerifyError as RVE, verify_receipt,
    )
    from sum_engine_internal.transform_receipt.format import SUPPORTED_SCHEMA as T
    from sum_engine_internal.transform_receipt.verifier import (
        VerifyError as TVE, verify_transform_receipt,
    )
    for schema, fn, exc in ((R, verify_receipt, RVE), (T, verify_transform_receipt, TVE)):
        with pytest.raises(exc) as ei:
            fn({"schema": schema, "kid": "k", "payload": bad_payload, "jws": "a..b"},
               meaning_jwks)
        assert ei.value.error_class == "malformed_receipt"


def test_required_field_sets_are_disjoint_enough_to_discriminate():
    """The gate only works if the families' required fields differ.

    If a future family's required set became a subset of another's, this
    defence would silently stop discriminating.
    """
    from sum_engine_internal.render_receipt.verifier import REQUIRED_PAYLOAD_FIELDS as R
    from sum_engine_internal.transform_receipt.verifier import REQUIRED_PAYLOAD_FIELDS as T
    assert not R.issubset(T), "render's required fields no longer discriminate"
    assert not T.issubset(R), "transform's required fields no longer discriminate"


def test_meaning_golden_still_verifies_under_its_own_verifier(meaning_golden, meaning_jwks):
    """No regression: the untouched golden must still verify."""
    sum_verify = pytest.importorskip("sum_verify")
    out = sum_verify.verify(meaning_golden, meaning_jwks)
    assert out is not None
