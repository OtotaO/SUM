"""Guard suite for the ``sum_verify`` dependency-light verify SDK.

Two things this suite must keep true, or the SDK is not trustworthy:

  1. **Dependency isolation.** ``import sum_verify`` — and a full
     meaning-risk replay — must NOT pull numpy / scipy / torch. That
     property is the whole point of the package (demand #2: integrators
     won't install a numeric stack to check a receipt). Tested in a
     clean subprocess so other tests' imports can't mask a regression.

  2. **No silent divergence from the canonical kernels.** The SDK
     re-derives the conformal bounds in pure Python (``_conformal``). This
     suite pins that re-derivation to the canonical numpy/scipy kernels —
     by replaying the committed golden receipts through BOTH paths and by
     a numeric parity grid — so a future edit that makes them disagree
     reds here instead of silently false-rejecting a genuine receipt.

The golden-replay, disclosure, and dispatcher tests need no numeric
stack and run everywhere. The equivalence / parity tests importorskip
numpy + scipy (present in the [research]/[dev] envs).
"""
from __future__ import annotations

import copy
import glob
import json
import subprocess
import sys

import pytest

import sum_verify
from sum_verify import (
    SUPPORTED_SCHEMAS,
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
    UnsupportedSchemaError,
    verify,
    verify_meaning_risk_receipt,
)

joserfc = pytest.importorskip("joserfc")  # the only hard SDK dep beyond stdlib+cryptography

_FIXTURE_DIRS = [
    "fixtures/meaning_receipts_billsum",
    "fixtures/meaning_receipts_translation",
]


def _load_case(d: str):
    receipt = json.load(open(glob.glob(d + "/*golden*.json")[0]))
    jwks = json.load(open(d + "/jwks.json"))
    wrapped = json.load(open(glob.glob(d + "/losses_*.json")[0]))
    losses = wrapped["losses"] if isinstance(wrapped, dict) else wrapped
    return receipt, jwks, losses


# --------------------------------------------------------------------------
# 1. Dependency isolation — the load-bearing property of the SDK.
# --------------------------------------------------------------------------


def test_import_does_not_pull_numeric_stack():
    """A clean interpreter that imports sum_verify and replays a golden
    receipt must never load numpy / scipy / torch / transformers."""
    code = (
        "import sys, json, glob; import sum_verify;"
        "d='fixtures/meaning_receipts_billsum';"
        "r=json.load(open(glob.glob(d+'/*golden*.json')[0]));"
        "j=json.load(open(d+'/jwks.json'));"
        "w=json.load(open(glob.glob(d+'/losses_*.json')[0]));"
        "L=w['losses'] if isinstance(w,dict) else w;"
        "sum_verify.verify(r,j,losses=L);"
        "bad=[m for m in ('numpy','scipy','torch','transformers') if m in sys.modules];"
        "print('LEAKED:'+(','.join(bad) if bad else 'none'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="."
    )
    assert out.returncode == 0, out.stderr
    assert "LEAKED:none" in out.stdout, out.stdout


def test_version_and_schemas_are_stable_surface():
    assert sum_verify.__version__  # SemVer string, present
    assert "sum.meaning_risk_receipt.v1" in SUPPORTED_SCHEMAS
    assert "sum.render_receipt.v1" in SUPPORTED_SCHEMAS
    assert "sum.transform_receipt.v1" in SUPPORTED_SCHEMAS


# --------------------------------------------------------------------------
# 2. Golden receipts verify + replay through the SDK.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d", _FIXTURE_DIRS)
def test_golden_verifies_signature_only(d):
    receipt, jwks, _ = _load_case(d)
    payload = verify(receipt, jwks)  # no losses → crypto + disclosure only
    assert payload["scorer"]
    assert payload["not_covered"]


@pytest.mark.parametrize("d", _FIXTURE_DIRS)
def test_golden_replays_bound_offline(d):
    receipt, jwks, losses = _load_case(d)
    payload = verify(receipt, jwks, losses=losses)
    # Replay reproduced the committed bound by exact integer equality
    # (the verify path raises otherwise), so the payload is returned intact.
    assert payload["risk_upper_bound_micro"] == receipt["payload"]["risk_upper_bound_micro"]
    assert payload["n"] == len(losses)


@pytest.mark.parametrize("d", _FIXTURE_DIRS)
def test_library_accepts_metadata_wrapped_losses(d):
    """The documented library snippet — `verify(receipt, jwks,
    losses=json.load(open("losses.json")))` with the committed metadata-wrapped
    losses file — must replay verbatim, not crash. Regression for the on-ramp
    bug the v2 adoption sim found (the {judge,...,losses:[...]} wrapper was
    unwrapped only in the CLI, so the documented library path raised
    ValueError on the project's own golden)."""
    receipt = json.load(open(glob.glob(d + "/*golden*.json")[0]))
    jwks = json.load(open(d + "/jwks.json"))
    wrapped = json.load(open(glob.glob(d + "/losses_*.json")[0]))
    assert isinstance(wrapped, dict) and "losses" in wrapped  # the committed shape
    payload = verify(receipt, jwks, losses=wrapped)  # the wrapper, not the bare list
    assert payload["risk_upper_bound_micro"] == receipt["payload"]["risk_upper_bound_micro"]


def test_no_eddsa_warning_and_no_false_negative_under_w_error():
    """Verifying a valid receipt must emit no EdDSA SecurityWarning (the single
    most-reported friction in the v2 sim) and must NOT false-negative when
    warnings are errors. We assert both: zero warnings recorded, and that
    promoting warnings to errors still verifies."""
    import warnings as _w

    receipt, jwks, losses = _load_case(_FIXTURE_DIRS[0])
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("error")  # any unsuppressed warning would raise here
        payload = verify(receipt, jwks, losses=losses)
    assert payload["scorer"]
    assert not [w for w in caught if "EdDSA" in str(w.message)]


@pytest.mark.parametrize("d", _FIXTURE_DIRS)
def test_wrong_losses_rejected(d):
    receipt, jwks, losses = _load_case(d)
    tampered = list(losses)
    tampered[0] = min(1.0, tampered[0] + 0.5) if tampered[0] < 0.5 else 0.0
    with pytest.raises(MeaningReceiptReplayError):
        verify(receipt, jwks, losses=tampered)


def _fresh_keypair(kid: str):
    """A throwaway Ed25519 keypair as private/public JWKs (test-only)."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    def b64u(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

    seed = bytes(range(32))  # fixed, throwaway — never a real key
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    x = b64u(
        sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    )
    private = {"kty": "OKP", "crv": "Ed25519", "d": b64u(seed), "x": x,
               "kid": kid, "alg": "EdDSA", "use": "sig"}
    public = {"kty": "OKP", "crv": "Ed25519", "x": x,
              "kid": kid, "alg": "EdDSA", "use": "sig"}
    return private, public


def test_missing_disclosure_rejected_even_with_valid_signature():
    """A receipt whose signature is genuine but which omits the required
    disclosure fields must still be refused — the SDK enforces the honesty
    contract independently, not just the crypto. Built by signing a
    disclosure-stripped payload with a fresh throwaway key."""
    from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope

    kid = "throwaway-disclosure-test"
    private, public = _fresh_keypair(kid)
    jwks = {"keys": [public]}

    # A structurally-plausible payload that deliberately omits not_covered.
    payload = {
        "scorer": "x",
        "scorer_version": "1",
        "loss_definition": "demo",
        "n": 2,
        "method": "hoeffding",
        "delta_micro": 50000,
        "point_estimate_micro": 0,
        "risk_upper_bound_micro": 0,
        "losses_hash": "sha256-deadbeef",
        "corpus_id": "demo",
        "transform": "demo",
        "disclosure": "present",  # but not_covered is missing
        "signed_at": "2026-01-01T00:00:00.000Z",
    }
    envelope = sign_jose_envelope(payload, private_jwk=private, kid=kid)
    envelope["schema"] = "sum.meaning_risk_receipt.v1"

    with pytest.raises(MeaningReceiptDisclosureError):
        verify(envelope, jwks)  # signature is valid; disclosure is not


def test_tampered_signature_rejected():
    receipt, jwks, losses = _load_case(_FIXTURE_DIRS[0])
    bad = copy.deepcopy(receipt)
    # Flip a character in the detached JWS → signature must fail.
    bad["jws"] = bad["jws"][:-4] + ("aaaa" if not bad["jws"].endswith("aaaa") else "bbbb")
    with pytest.raises(Exception):
        verify(bad, jwks, losses=losses)


# --------------------------------------------------------------------------
# 3. Dispatcher behaviour.
# --------------------------------------------------------------------------


def test_unsupported_schema_raises():
    with pytest.raises(UnsupportedSchemaError):
        verify({"schema": "sum.not_a_real_receipt.v9"}, {"keys": []})


def test_direct_meaning_entrypoint_matches_dispatch():
    receipt, jwks, losses = _load_case(_FIXTURE_DIRS[0])
    a = verify(receipt, jwks, losses=losses)
    b = verify_meaning_risk_receipt(receipt, jwks, losses=losses)
    assert a == b


# --------------------------------------------------------------------------
# 4. No silent divergence from the canonical numpy/scipy kernels.
# --------------------------------------------------------------------------


def test_sdk_verifier_matches_canonical_on_goldens():
    """The pure-Python SDK verifier and the canonical numpy verifier must
    return identical payloads on every committed golden — the
    cross-implementation equivalence pin."""
    pytest.importorskip("numpy")
    from sum_engine_internal.research.meaning.receipt import (
        verify_meaning_risk_receipt as canonical_verify,
    )

    for d in _FIXTURE_DIRS:
        receipt, jwks, losses = _load_case(d)
        sdk = verify_meaning_risk_receipt(receipt, jwks, losses=losses)
        can = canonical_verify(receipt, jwks, losses=losses)
        assert sdk == can, f"SDK and canonical verifier disagree on {d}"


def test_bound_kernels_parity_grid():
    """Pure-Python bounds must agree with the canonical numpy/scipy kernels
    to far inside the receipt's 1e-6 micro wire grid — across sample sizes,
    deltas, and loss patterns, for all three methods. This is the guard
    that keeps the two implementations from silently diverging."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from sum_engine_internal.research.conformal import risk_control as canon
    from sum_verify import _conformal as sdk

    def micro(x: float) -> int:
        return int(round(x * 1_000_000))

    rng = np.random.RandomState(0)
    deltas = [0.01, 0.05, 0.1, 1.0 / 30.0]  # incl. an off-grid Bonferroni delta
    sizes = [1, 2, 8, 16, 64, 257]

    for delta in deltas:
        for n in sizes:
            # fractional pattern → hoeffding + empirical_bernstein
            frac = rng.uniform(size=n).tolist()
            for name, sdk_fn, can_fn in [
                ("hoeffding", sdk.hoeffding_lower_bound, canon.hoeffding_lower_bound),
                (
                    "empirical_bernstein",
                    sdk.empirical_bernstein_lower_bound,
                    canon.empirical_bernstein_lower_bound,
                ),
            ]:
                s = sdk_fn(frac, delta)
                c = can_fn(frac, delta)
                assert abs(s - c) < 1e-9, (name, delta, n, s, c)
                assert micro(s) == micro(c), (name, delta, n, s, c)

            # binary pattern → clopper_pearson (the scipy-replacement path)
            binv = (rng.uniform(size=n) < 0.7).astype(float)
            successes = int(binv.sum())
            s_cp = sdk.clopper_pearson_lower_bound(successes, n, delta)
            c_cp = canon.clopper_pearson_lower_bound(successes, n, delta)
            assert abs(s_cp - c_cp) < 1e-9, ("clopper_pearson", delta, n, successes, s_cp, c_cp)
            assert micro(s_cp) == micro(c_cp), ("clopper_pearson", delta, n, successes)


def test_certify_meaning_risk_parity_all_methods():
    """The composed certifier (the function the receipt commits) must
    reproduce the canonical one's bound + point estimate to the micro on
    every method, including the binary/auto Clopper–Pearson branch."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from sum_engine_internal.research.meaning import conformal_meaning as canon
    from sum_verify import _conformal as sdk

    def micro(x: float) -> int:
        return int(round(x * 1_000_000))

    rng = np.random.RandomState(7)
    for method in ("hoeffding", "empirical_bernstein", "auto"):
        for n in (1, 4, 16, 64):
            if method in ("auto",):
                losses = (rng.uniform(size=n) < 0.3).astype(float).tolist()  # binary
            else:
                losses = rng.uniform(size=n).tolist()
            s = sdk.certify_meaning_risk(
                losses, scorer_name="x", scorer_version="1", delta=0.05, method=method
            )
            c = canon.certify_meaning_risk(
                losses, scorer_name="x", scorer_version="1", delta=0.05, method=method
            )
            assert s.method == c.method, (method, n)
            assert micro(s.risk_upper_bound) == micro(c.risk_upper_bound), (method, n)
            assert micro(s.point_estimate) == micro(c.point_estimate), (method, n)
