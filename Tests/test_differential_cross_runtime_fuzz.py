"""Differential cross-runtime fuzzer: Python and Node verifiers must reach the
SAME accept/reject verdict on the SAME adversarial receipt, and neither may
crash. This is the regression guard for the trust triangle's core promise —
"verify anywhere, get the same answer" — under MUTATION, which the fixed-fixture
cross-runtime tests cannot cover (they found, e.g., a non-dict-JWKS crash that no
committed fixture exercised; see the 2026-06-14 differential campaign).

Two invariants are asserted hard:
  1. accept/reject PARITY  — for every mutation, Python ACCEPT iff Node ACCEPT.
  2. TOTALITY              — neither runtime throws an undeclared exception.

A third, weaker property (agree on the error-CLASS, not just accept/reject) is
checked too, but the runtimes have a small set of KNOWN, accept/reject-harmless
class-name divergences (Python `malformed_envelope` vs Node `malformed_receipt`
for structural malformations, + two Node `[]`-is-object edges). Those are
allow-listed below so a NEW class divergence still fails. Reconciling the
taxonomy is an open contract decision; see KNOWN_CLASS_DIVERGENCE.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] extra not installed")
from joserfc.jwk import OKPKey

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "single_file_demo" / "diff_verdict_driver.mjs"

# Cases where the runtimes both REJECT but name a different error class. These
# are accept/reject-harmless (parity still holds); listed so a new divergence is
# not masked. (mutation tag) -> (python_class, node_class).
KNOWN_CLASS_DIVERGENCE = {
    "receipt-is-array": ("malformed_envelope", "schema_unknown"),
    "receipt-is-string": ("malformed_envelope", "malformed_receipt"),
    "receipt-is-number": ("malformed_envelope", "malformed_receipt"),
    "receipt-is-null": ("malformed_envelope", "malformed_receipt"),
    "drop-payload": ("malformed_envelope", "malformed_receipt"),
    "drop-jws": ("malformed_envelope", "malformed_receipt"),
    "drop-kid": ("malformed_envelope", "malformed_receipt"),
    "payload-is-array": ("malformed_envelope", "signature_invalid"),
    "payload-is-string": ("malformed_envelope", "malformed_receipt"),
    "jws-empty": ("malformed_envelope", "malformed_receipt"),
}


def _keypair(kid="diff-kid"):
    k = OKPKey.generate_key("Ed25519")
    pr = k.as_dict(private=True); pr.update(kid=kid, alg="EdDSA", use="sig")
    pu = k.as_dict(private=False); pu.update(kid=kid, alg="EdDSA", use="sig")
    return pr, {"keys": [pu]}, kid


def _setup(family):
    pr, jwks, kid = _keypair()
    if family in ("render", "transform"):
        from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope
        from sum_engine_internal.render_receipt import verify_receipt, SUPPORTED_SCHEMA as RS
        from sum_engine_internal.transform_receipt import (
            verify_transform_receipt, SUPPORTED_SCHEMA as TS)
        schema = RS if family == "render" else TS
        payload = {"render_id": "r", "tome_hash": "sha256-" + "0" * 64,
                   "triples_hash": "sha256-" + "1" * 64,
                   "sliders_quantized": {"density": 50, "audience": 20},
                   "model": "demo", "provider": "canonical-path",
                   "digital_source_type": "trainedAlgorithmicMedia",
                   "signed_at": "2026-06-06T12:00:00.000Z"}
        env = sign_jose_envelope(copy.deepcopy(payload), private_jwk=pr, kid=kid)
        env["schema"] = schema
        return env, jwks, schema, (verify_receipt if family == "render" else verify_transform_receipt)
    from sum_engine_internal.research.meaning import (
        certify_meaning_risk, build_payload, sign_meaning_risk_receipt, verify_meaning_risk_receipt)
    losses = [0.0] * 40 + [0.2] * 40
    g = certify_meaning_risk(losses, scorer_name="lexical-coverage-bidirectional", scorer_version="1")
    pl = build_payload(guarantee=g, losses=losses, corpus_id="diff", transform="t",
                       loss_definition="d", alpha_target=0.5)
    env = sign_meaning_risk_receipt(pl, private_jwk=pr, kid=kid)
    return env, jwks, "sum.meaning_risk_receipt.v1", verify_meaning_risk_receipt


_DECLARED = {"JoseEnvelopeError", "VerifyError", "MeaningReceiptReplayError",
             "MeaningReceiptDisclosureError"}


def _py_verdict(pyverify, receipt, jwks):
    try:
        pyverify(receipt, jwks)
        return ("ACCEPT", None)
    except Exception as e:  # noqa: BLE001 — totality probe: any undeclared type is a crash
        cls = getattr(e, "error_class", None) or type(e).__name__
        kind = "REJECT" if type(e).__name__ in _DECLARED else "CRASH"
        return (kind, cls)


def _mutations(env, jwks):
    P = env.get("payload")

    def E(**kw):
        e = copy.deepcopy(env); e.update(kw); return e

    yield ("pristine", env, jwks)
    yield ("receipt-is-array", [1, 2, 3], jwks)
    yield ("receipt-is-string", "x", jwks)
    yield ("receipt-is-number", 42, jwks)
    yield ("receipt-is-null", None, jwks)
    yield ("jwks-is-number", env, 5)
    yield ("jwks-no-keys-field", env, {})
    yield ("jwks-empty-keys", env, {"keys": []})
    yield ("jwks-is-null", env, None)
    yield ("jwks-is-array", env, [1, 2])
    e = copy.deepcopy(env); e.pop("payload", None); yield ("drop-payload", e, jwks)
    e = copy.deepcopy(env); e.pop("jws", None); yield ("drop-jws", e, jwks)
    e = copy.deepcopy(env); e.pop("kid", None); yield ("drop-kid", e, jwks)
    e = copy.deepcopy(env); e.pop("schema", None); yield ("drop-schema", e, jwks)
    yield ("schema-wrong", E(schema="sum.evil.v1"), jwks)
    yield ("kid-unknown", E(kid="nope"), jwks)
    yield ("payload-empty-dict", E(payload={}), jwks)
    yield ("payload-is-array", E(payload=[1, 2]), jwks)
    yield ("payload-is-string", E(payload="x"), jwks)
    jws = env.get("jws", "")
    parts = jws.split(".") if isinstance(jws, str) else ["", "", ""]
    yield ("jws-empty", E(jws=""), jwks)
    yield ("jws-two-segments", E(jws="a.b"), jwks)
    yield ("jws-nonempty-middle",
           E(jws=".".join([parts[0], "ZZ", parts[2]]) if len(parts) == 3 else "a.b.c"), jwks)
    if len(parts) == 3:
        sig = list(parts[2]); sig[3] = "A" if sig[3] != "A" else "B"
        yield ("jws-sig-corrupt", E(jws=".".join([parts[0], parts[1], "".join(sig)])), jwks)
    if isinstance(P, dict) and P:
        k = sorted(P.keys())[0]
        e = copy.deepcopy(env); e["payload"] = dict(P); e["payload"][k] = "TAMPERED"
        yield (f"payload-tamper[{k}]", e, jwks)


def _node_verdicts(family, cases, schema):
    lines = "\n".join(json.dumps({"receipt": r, "jwks": j, "schema": schema}) for _, r, j in cases)
    res = subprocess.run([NODE, str(DRIVER), family], input=lines,
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"node driver failed: {res.stderr[:300]}"
    verdicts = []
    for ln in res.stdout.split("\n"):
        nj = json.loads(ln)
        if "crash" in nj:
            verdicts.append(("CRASH", nj["crash"]))
        elif nj.get("v") is True:
            verdicts.append(("ACCEPT", None))
        else:
            verdicts.append(("REJECT", nj.get("c")))
    return verdicts


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_cross_runtime_verdict_parity_and_totality(family):
    env, jwks, schema, pyverify = _setup(family)
    cases = list(_mutations(env, jwks))
    py = [_py_verdict(pyverify, r, j) for _, r, j in cases]
    node = _node_verdicts(family, cases, schema)
    assert len(node) == len(cases)

    parity_fail, crashes, new_classdiff = [], [], []
    for (tag, _, _), (pv, pc), (nv, nc) in zip(cases, py, node):
        if pv == "CRASH" or nv == "CRASH":
            crashes.append((tag, f"py={pv}:{pc}", f"node={nv}:{nc}"))
        elif pv != nv:
            parity_fail.append((tag, pv, nv))
        elif pv == "REJECT" and pc != nc and KNOWN_CLASS_DIVERGENCE.get(tag) != (pc, nc):
            new_classdiff.append((tag, pc, nc))

    assert not crashes, f"[{family}] runtime CRASH on adversarial input: {crashes}"
    assert not parity_fail, f"[{family}] accept/reject DIVERGENCE: {parity_fail}"
    assert not new_classdiff, (
        f"[{family}] NEW error-class divergence (not in KNOWN_CLASS_DIVERGENCE): {new_classdiff}")
