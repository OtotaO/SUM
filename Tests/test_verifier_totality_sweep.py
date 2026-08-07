"""Totality sweep: every verifier is TOTAL on adversarial input — it returns or
raises one of its DECLARED exceptions, never an undeclared one (a crash).

Extends the meaning-verifier totality property (test_meaning_verifier_robustness_
property.py) to the other trust-boundary verifiers:
  - render  / transform  receipts  — the JOSE-family pure verifies (declared: JoseEnvelopeError, incl. its VerifyError subclass).
  - CanonicalBundle (`sum verify`)  — a DIFFERENT scheme (HMAC + raw Ed25519, not JOSE; declared: InvalidSignatureError, ValueError).

Node-independent (no cross-runtime dependency); the cross-runtime parity of these
same families is in test_differential_cross_runtime_fuzz.py.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis (hardening extra) not installed")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# Adversarial JSON values, including the leaves that historically crashed
# verifiers: None, non-finite floats, coercible strings, oversized ints, arrays.
_NASTY = (
    st.none() | st.booleans()
    | st.integers(min_value=-(10**19), max_value=10**19)
    | st.floats() | st.text(max_size=8)
    | st.sampled_from(["645438", "0.5", "", "​", "0", "-1"])
)
_JSON = st.recursive(
    _NASTY,
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=4), c, max_size=4),
    max_leaves=8,
)
_SETTINGS = settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow], deadline=None)


def _assert_total(fn, declared):
    try:
        fn()
    except declared:
        pass
    # any other exception propagates -> Hypothesis reports the falsifying crash


# ─────────────────────────────  JOSE families (render / transform)  ──────────

joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] extra not installed")
from joserfc.jwk import OKPKey  # noqa: E402
from sum_engine_internal.infrastructure.jose_envelope import (  # noqa: E402
    JoseEnvelopeError, sign_jose_envelope)


def _keypair(kid="totality"):
    k = OKPKey.generate_key("Ed25519")
    pr = k.as_dict(private=True); pr.update(kid=kid, alg="EdDSA", use="sig")
    pu = k.as_dict(private=False); pu.update(kid=kid, alg="EdDSA", use="sig")
    return pr, {"keys": [pu]}, kid


def _jose_family(family):
    pr, jwks, kid = _keypair()
    if family == "render":
        from sum_engine_internal.render_receipt import verify_receipt, SUPPORTED_SCHEMA
        verify = verify_receipt
    else:
        from sum_engine_internal.transform_receipt import verify_transform_receipt, SUPPORTED_SCHEMA
        verify = verify_transform_receipt
    payload = {"render_id": "r", "tome_hash": "sha256-" + "0" * 64,
               "model": "demo", "signed_at": "2026-06-06T12:00:00.000Z"}
    env = sign_jose_envelope(copy.deepcopy(payload), private_jwk=pr, kid=kid)
    env["schema"] = SUPPORTED_SCHEMA
    return env, jwks, verify


_RENDER_ENV, _RENDER_JWKS, _RENDER_VERIFY = _jose_family("render")
_XFORM_ENV, _XFORM_JWKS, _XFORM_VERIFY = _jose_family("transform")


@pytest.mark.parametrize("family", ["render", "transform"])
@_SETTINGS
@given(field=st.sampled_from(["kid", "payload", "jws", "schema", "__jwks__"]), val=_JSON)
def test_jose_family_verifier_is_total(family, field, val):
    env, jwks, verify = (
        (_RENDER_ENV, _RENDER_JWKS, _RENDER_VERIFY) if family == "render"
        else (_XFORM_ENV, _XFORM_JWKS, _XFORM_VERIFY))
    if field == "__jwks__":
        _assert_total(lambda: verify(env, val), JoseEnvelopeError)
        return
    e = copy.deepcopy(env); e[field] = val
    _assert_total(lambda: verify(e, jwks), JoseEnvelopeError)


@pytest.mark.parametrize("family", ["render", "transform"])
@pytest.mark.parametrize(
    "header_json", ["[1,2]", "5", '"x"', "null", "true"],
    ids=["array", "number", "string", "null", "bool"],
)
def test_jose_family_rejects_non_object_protected_header(family, header_json):
    """A protected header that is valid JSON but not an OBJECT must raise the
    declared class, not a raw AttributeError.

    RFC 7515 §4 requires an object. These inputs parse cleanly and then reach
    ``header.get(...)``, which is how an undeclared AttributeError escaped the
    ``SumVerifyError`` contract the public SDK documents. The property test
    above cannot reach this: it replaces ``jws`` wholesale, so it effectively
    never generates a well-formed 3-segment JWS whose first segment decodes to
    valid non-object JSON. Cross-runtime parity for the same inputs is pinned
    in test_differential_cross_runtime_fuzz.py.
    """
    import base64

    env, jwks, verify = (
        (_RENDER_ENV, _RENDER_JWKS, _RENDER_VERIFY) if family == "render"
        else (_XFORM_ENV, _XFORM_JWKS, _XFORM_VERIFY))
    proto = base64.urlsafe_b64encode(header_json.encode()).decode().rstrip("=")
    _, middle, sig = env["jws"].split(".")
    e = copy.deepcopy(env); e["jws"] = f"{proto}.{middle}.{sig}"
    with pytest.raises(JoseEnvelopeError) as ei:
        verify(e, jwks)
    assert ei.value.error_class == "malformed_jws", (
        f"expected malformed_jws for a {header_json} protected header, "
        f"got {ei.value.error_class}"
    )


# ─────────────────────────────  CanonicalBundle (sum verify)  ────────────────

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra  # noqa: E402
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator  # noqa: E402
from sum_engine_internal.infrastructure.canonical_codec import (  # noqa: E402
    CanonicalCodec, InvalidSignatureError)

_BUNDLE_DECLARED = (InvalidSignatureError, ValueError)


def _mint_bundle():
    algebra = GodelStateAlgebra()
    codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra),
                           signing_key="totality-sweep-key-32-bytes-long")
    for s, p, o in [("alice", "likes", "cats"), ("bob", "knows", "python")]:
        algebra.get_or_mint_prime(s, p, o)
    state = 1
    for prime in algebra.axiom_to_prime.values():
        state = math.lcm(state, prime)
    return codec, codec.export_bundle(state, branch="totality")


_BUNDLE_CODEC, _BUNDLE = _mint_bundle()
_BUNDLE_KEYS = list(_BUNDLE.keys())


@_SETTINGS
@given(field=st.sampled_from(_BUNDLE_KEYS), val=_JSON)
def test_bundle_verifier_is_total(field, val):
    """A CanonicalBundle with any field mutated to an arbitrary value must
    raise InvalidSignatureError or ValueError — never an undeclared crash."""
    b = copy.deepcopy(_BUNDLE); b[field] = val
    _assert_total(lambda: _BUNDLE_CODEC.import_bundle(b), _BUNDLE_DECLARED)


@_SETTINGS
@given(b=st.dictionaries(st.text(max_size=6), _JSON, max_size=8))
def test_bundle_verifier_total_on_arbitrary_dict(b):
    """An entirely arbitrary dict (not even a real bundle) must still fail
    closed with a declared exception, not crash."""
    _assert_total(lambda: _BUNDLE_CODEC.import_bundle(b), _BUNDLE_DECLARED)


# ─────────────────────────  meaning family (node-free)  ──────────────────────

@pytest.mark.parametrize(
    "header_json", ["[1,2]", "5", '"x"', "null", "true"],
    ids=["array", "number", "string", "null", "bool"],
)
def test_meaning_verifier_rejects_non_object_protected_header(header_json):
    """Same guard as the JOSE families, exercised through the MEANING schema.

    The meaning family shares the Python envelope core, but it reaches it via a
    different public entry point (``sum_verify.verify``), and without this case
    its behaviour is pinned only by the cross-runtime fuzzer, which is skipped
    on a runner without node. Uses the committed BillSum golden so no signing
    key is needed.
    """
    import base64
    import json

    sum_verify = pytest.importorskip("sum_verify")
    fx = _REPO_ROOT / "fixtures" / "meaning_receipts_billsum"
    env = json.loads((fx / "meaning_risk_receipt.billsum.golden.json").read_text("utf-8"))
    jwks = json.loads((fx / "jwks.json").read_text("utf-8"))

    # Sanity: the pristine golden still verifies, so a failure below is the guard.
    assert sum_verify.verify(env, jwks)

    proto = base64.urlsafe_b64encode(header_json.encode()).decode().rstrip("=")
    _, middle, sig = env["jws"].split(".")
    bad = dict(env); bad["jws"] = f"{proto}.{middle}.{sig}"
    with pytest.raises(sum_verify.SumVerifyError):
        sum_verify.verify(bad, jwks)


@pytest.mark.parametrize(
    "crit", [[["b64"]], [{"b64": 1}], [None], [1]],
    ids=["nested-list", "nested-object", "null-elem", "int-elem"],
)
def test_jose_family_rejects_unhashable_crit_elements(crit):
    """A non-string ``crit`` element must raise the declared class.

    ``known_crit_extensions`` is a frozenset, so an unhashable element made the
    membership test itself raise TypeError on unauthenticated bytes, before any
    signature check. JS never crashed here (``Set.has`` is total), so this was a
    totality break and a cross-runtime class divergence at once.
    """
    import base64
    import json

    header = {"alg": "EdDSA", "kid": "totality", "b64": False, "crit": crit}
    proto = base64.urlsafe_b64encode(
        json.dumps(header, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    _, middle, sig = _RENDER_ENV["jws"].split(".")
    e = copy.deepcopy(_RENDER_ENV); e["jws"] = f"{proto}.{middle}.{sig}"
    with pytest.raises(JoseEnvelopeError) as ei:
        _RENDER_VERIFY(e, _RENDER_JWKS)
    assert ei.value.error_class in {"crit_unknown_extension", "malformed_jws"}
