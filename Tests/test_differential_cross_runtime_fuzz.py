"""Differential cross-runtime fuzzer: Python and Node verifiers must reach the
SAME accept/reject verdict on the SAME adversarial receipt, and neither may
crash. This is the regression guard for the trust triangle's core promise —
"verify anywhere, get the same answer" — under MUTATION, which the fixed-fixture
cross-runtime tests cannot cover (they found, e.g., a non-dict-JWKS crash that no
committed fixture exercised; see the 2026-06-14 differential campaign).

Three invariants are asserted hard:
  1. accept/reject PARITY  — for every mutation, Python ACCEPT iff Node ACCEPT.
  2. TOTALITY              — neither runtime throws an undeclared exception.
  3. error-CLASS PARITY    — both runtimes name the SAME rejection class.

(3) was reconciled 2026-06-14: the Python enum value malformed_envelope was
renamed to malformed_receipt and the JS verifiers taught to reject array
receipts/payloads structurally, collapsing the last class divergences. The
KNOWN_CLASS_DIVERGENCE allow-list is therefore empty; any new divergence fails.
"""
from __future__ import annotations

import base64
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
#
# RECONCILED 2026-06-14 — this set is now EMPTY: the runtimes agree on the exact
# error CLASS for every mutation, not just accept/reject. The reconciliation
# renamed the Python enum value malformed_envelope -> malformed_receipt (matching
# the existing render/transform MALFORMED_RECEIPT alias intent and the JS string)
# and taught the JS verifiers to reject array receipts/payloads as
# malformed_receipt (they previously slipped past `typeof === "object"`). Keep
# the dict so a FUTURE intentional divergence can be documented here rather than
# silently allow-listed.
KNOWN_CLASS_DIVERGENCE: dict[str, tuple[str, str]] = {}


def _keypair(kid="diff-kid"):
    k = OKPKey.generate_key("Ed25519")
    pr = k.as_dict(private=True); pr.update(kid=kid, alg="EdDSA", use="sig")
    pu = k.as_dict(private=False); pu.update(kid=kid, alg="EdDSA", use="sig")
    return pr, {"keys": [pu]}, kid


_SIGNING_KEYS: dict[str, dict] = {}  # family -> private JWK of the last _setup


def _setup(family):
    pr, jwks, kid = _keypair()
    _SIGNING_KEYS[family] = pr
    if family in ("render", "transform"):
        from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope
        from sum_engine_internal.render_receipt import verify_receipt, SUPPORTED_SCHEMA as RS
        from sum_engine_internal.transform_receipt import (
            verify_transform_receipt, SUPPORTED_SCHEMA as TS)
        schema = RS if family == "render" else TS
        # Family-specific payloads. This USED to sign one render-shaped
        # payload and relabel it for the transform family, which made the
        # "pristine" transform case an instance of the very receipt-type
        # confusion the verifiers now reject. See
        # Tests/test_receipt_type_confusion.py.
        _common = {"model": "demo", "provider": "canonical-path",
                   "digital_source_type": "trainedAlgorithmicMedia",
                   "signed_at": "2026-06-06T12:00:00.000Z"}
        if family == "render":
            payload = {"render_id": "r", "tome_hash": "sha256-" + "0" * 64,
                       "triples_hash": "sha256-" + "1" * 64,
                       "sliders_quantized": {"density": 50, "audience": 20},
                       **_common}
        else:
            payload = {"transform": "slider", "transform_id": "t",
                       "input_hash": "sha256-" + "0" * 64,
                       "output_hash": "sha256-" + "1" * 64,
                       "parameters_hash": "sha256-" + "2" * 64,
                       **_common}
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
        # Same signature bytes, respelled: set an unused low bit of the last
        # character. Lenient decoders (atob, WebCrypto) read the same bytes;
        # both runtimes must still give the same verdict.
        _A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        _s = parts[2]
        yield ("jws-sig-noncanonical-tail",
               E(jws=".".join([parts[0], parts[1], _s[:-1] + _A[_A.index(_s[-1]) ^ 1]])), jwks)
        _h = parts[0]
        if len(_h) % 4 in (2, 3):  # a header with spare bits: respell its tail
            yield ("jws-header-noncanonical-tail",
                   E(jws=".".join([_h[:-1] + _A[_A.index(_h[-1]) ^ 1], parts[1], parts[2]])), jwks)
        _j = copy.deepcopy(jwks)
        if isinstance(_j, dict) and _j.get("keys") and isinstance(_j["keys"][0].get("x"), str):
            _x = _j["keys"][0]["x"]
            _j["keys"][0]["x"] = _x[:-1] + _A[_A.index(_x[-1]) ^ 1]
            yield ("jwks-x-noncanonical-tail", env, _j)
        # Padding: RFC 7515 base64url carries none. Both runtimes refuse it,
        # on all three base64url parts: header, signature and key.
        for _pad in ("=", "=="):
            yield (f"jws-header-padded{len(_pad)}", E(jws=".".join([parts[0] + _pad, parts[1], parts[2]])), jwks)
            yield (f"jws-sig-padded{len(_pad)}", E(jws=".".join([parts[0], parts[1], _s + _pad])), jwks)
            _j = copy.deepcopy(jwks)
            if isinstance(_j, dict) and _j.get("keys") and isinstance(_j["keys"][0].get("x"), str):
                _j["keys"][0]["x"] += _pad
                yield (f"jwks-x-padded{len(_pad)}", env, _j)
        # Protected header that is VALID JSON but not an object. RFC 7515 §4
        # requires an object; these parse cleanly and then reach `.get`/property
        # access, which is how a raw AttributeError once escaped the declared
        # error contract on the Python side. Mutating the header segment (rather
        # than the middle/signature) is the case this fuzzer previously missed.
        for _name, _json in (
            ("array", "[1,2]"), ("number", "5"), ("string", '"x"'),
            ("null", "null"), ("bool", "true"),
            # Object headers whose FIELDS are adversarial. These are what
            # actually differed between runtimes once the non-object case was
            # closed: an unhashable `crit` element crashed Python's frozenset
            # membership test, and a falsy/absent `alg` skipped the
            # algorithm-registry check in two of the three JS verifiers.
            ("crit-nested-list", '{"alg":"EdDSA","b64":false,"crit":[["b64"]]}'),
            ("crit-nested-object", '{"alg":"EdDSA","b64":false,"crit":[{"b64":1}]}'),
            ("crit-null-elem", '{"alg":"EdDSA","b64":false,"crit":[null]}'),
            ("crit-int-elem", '{"alg":"EdDSA","b64":false,"crit":[1]}'),
            ("alg-missing", '{"b64":false,"crit":["b64"]}'),
            ("alg-empty-string", '{"alg":"","b64":false,"crit":["b64"]}'),
            ("alg-null", '{"alg":null,"b64":false,"crit":["b64"]}'),
            ("alg-false", '{"alg":false,"b64":false,"crit":["b64"]}'),
            ("alg-zero", '{"alg":0,"b64":false,"crit":["b64"]}'),
            ("alg-object", '{"alg":{"a":1},"b64":false,"crit":["b64"]}'),
            ("empty-object", "{}"),
        ):
            _proto = base64.urlsafe_b64encode(_json.encode()).decode().rstrip("=")
            yield (f"jws-header-{_name}",
                   E(jws=".".join([_proto, parts[1], parts[2]])), jwks)
    if isinstance(P, dict) and P:
        k = sorted(P.keys())[0]
        e = copy.deepcopy(env); e["payload"] = dict(P); e["payload"][k] = "TAMPERED"
        yield (f"payload-tamper[{k}]", e, jwks)
        # A lone surrogate has no UTF-8 encoding, so JCS cannot canonicalize the
        # payload. Python rejects it as malformed_receipt; the JS verifiers must
        # reject it the same way, not let the vendored canonicalize's plain Error
        # escape. json.dumps sends it to Node as a \ud800 escape.
        e = copy.deepcopy(env); e["payload"] = dict(P); e["payload"][k] = "\ud800" + str(P[k])
        yield (f"payload-lone-surrogate-value[{k}]", e, jwks)
        e = copy.deepcopy(env); e["payload"] = dict(P); e["payload"]["x\udc00"] = "v"
        yield ("payload-lone-surrogate-key", e, jwks)


def _node_verdicts(family, cases, schema, opts=None):
    extra = {} if opts is None else {"opts": opts}
    lines = "\n".join(json.dumps({"receipt": r, "jwks": j, "schema": schema, **extra})
                      for _, r, j in cases)
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


def _resign_with_header(env, family, proto):
    """Sign env's payload again over a chosen protected-header string, as an
    issuer would. b64:false, so the signing input is proto + "." + the JCS
    payload bytes."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from sum_engine_internal.infrastructure.jcs import canonicalize

    d = _SIGNING_KEYS[family]["d"]
    sk = Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(d + "=" * (-len(d) % 4)))
    sig = sk.sign(proto.encode("ascii") + b"." + canonicalize(env["payload"]))
    out = copy.deepcopy(env)
    out["jws"] = proto + ".." + base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return out


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_issuer_signed_noncanonical_header_rejected_by_both(family):
    """A header changed after signing fails the signature anyway; one the
    issuer signed in a padded or respelled spelling is a genuine signature.
    Both runtimes must give the same verdict on it: reject, malformed_jws.
    The canonical header re-signed the same way is the control: accepted."""
    env, jwks, schema, pyverify = _setup(family)
    proto = env["jws"].split(".")[0]
    _A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    assert len(proto) % 4 in (2, 3)
    cases = [("control", _resign_with_header(env, family, proto), jwks)]
    for tag, h in (("padded1", proto + "="), ("padded2", proto + "=="),
                   ("respelled", proto[:-1] + _A[_A.index(proto[-1]) ^ 1])):
        cases.append((tag, _resign_with_header(env, family, h), jwks))
    py = [_py_verdict(pyverify, r, j) for _, r, j in cases]
    node = _node_verdicts(family, cases, schema)
    assert py[0] == node[0] == ("ACCEPT", None), (py[0], node[0])
    for (tag, _, _), pv, nv in zip(cases[1:], py[1:], node[1:]):
        assert pv == nv == ("REJECT", "malformed_jws"), (family, tag, pv, nv)


# signed_at spellings for the opt-in replay window. Each one is signed into the
# receipt, so the signature holds and only the window decides. With
# max_age_seconds set, both runtimes must read signed_at with the same strict
# instant grammar that revocation uses (_instant_ms / parseInstantMs): seconds
# required, uppercase T, Z or +-HH:MM. The window is wide enough (10**12 s) that
# the clock never decides, only the grammar. The JS meaning verifier has no
# window option, so this covers render and transform.
_WINDOW_FORMS = {
    # admitted by the instant grammar: both accept
    "ms3": ("2026-06-06T12:00:00.000Z", True),
    "frac1": ("2026-06-06T12:00:00.5Z", True),
    "frac9": ("2026-06-06T12:00:00.123456789Z", True),
    "offset": ("2026-06-06T17:30:00+05:30", True),
    "calendar-edge": ("0001-01-01T00:00:00+01:00", True),
    # refused by it: both reject as signed_at_out_of_window
    "any-separator": ("2026-06-06X12:00:00Z", False),
    "hour-only": ("2026-06-06T12Z", False),
    "offset-seconds": ("2026-06-06T12:00:00+00:00:00", False),
    "lowercase-t-z": ("2026-06-06t12:00:00z", False),
    "date-only": ("2026-06-06", False),
    "rfc2822": ("Sat, 06 Jun 2026 12:00:00 GMT", False),
}
_WIDE_WINDOW = 10**12


@pytest.mark.parametrize("family", ["render", "transform"])
def test_replay_window_timestamp_grammar_parity(family):
    """Python once read signed_at with datetime.fromisoformat (a grammar that
    changes between Python versions) and the JS verifiers with Date.parse, so
    the same signed receipt got opposite window verdicts."""
    from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope

    env, jwks, schema, pyverify = _setup(family)
    cases = []
    for tag, (signed_at, _) in _WINDOW_FORMS.items():
        payload = copy.deepcopy(env["payload"])
        payload["signed_at"] = signed_at
        signed = sign_jose_envelope(payload, private_jwk=_SIGNING_KEYS[family], kid=env["kid"])
        signed["schema"] = schema
        cases.append((tag, signed, jwks))
    py = [_py_verdict(lambda r, j: pyverify(r, j, max_age_seconds=_WIDE_WINDOW), r, j)
          for _, r, j in cases]
    node = _node_verdicts(family, cases, schema, opts={"maxAgeSeconds": _WIDE_WINDOW})
    wrong = []
    for (tag, _, _), pv, nv in zip(cases, py, node):
        want = ("ACCEPT", None) if _WINDOW_FORMS[tag][1] else ("REJECT", "signed_at_out_of_window")
        if not (pv == nv == want):
            wrong.append((tag, f"py={pv}", f"node={nv}", f"want={want}"))
    assert not wrong, f"[{family}] replay-window verdicts differ: {wrong}"


# JWK members around the public key. WebCrypto (the JS side) refused some of these
# at import and accepted others joserfc refused, and joserfc refused some only at
# verify time as signature_invalid, so the runtimes split on accept/reject as well
# as on class. Both now apply one rule before import: no private member d; key_ops,
# when present, an array of distinct strings that includes "verify"; use, when
# present, "sig"; alg, when present, "EdDSA" or its RFC 9864 name "Ed25519"
# (WebCrypto writes it on export; joserfc refused it against an EdDSA header).
# Anything else is malformed_jwks in both. ext and unknown members stay ignored.
_JWK_MEMBERS = {
    "baseline": ({}, True),
    "key_ops-verify": ({"key_ops": ["verify"]}, True),
    "key_ops-sign-verify": ({"key_ops": ["sign", "verify"]}, True),
    "use-sig": ({"use": "sig"}, True),
    "alg-EdDSA": ({"alg": "EdDSA"}, True),
    "alg-Ed25519": ({"alg": "Ed25519"}, True),
    "webcrypto-export": ({"key_ops": ["verify"], "ext": True, "alg": "Ed25519"}, True),
    "ext-true": ({"ext": True}, True),
    "ext-false": ({"ext": False}, True),
    "unknown-member": ({"foo": 1}, True),
    "d-private-scalar": (None, False),
    "d-garbage": ({"d": "AAAA"}, False),
    "key_ops-sign": ({"key_ops": ["sign"]}, False),
    "key_ops-empty": ({"key_ops": []}, False),
    "key_ops-string": ({"key_ops": "verify"}, False),
    "key_ops-duplicate": ({"key_ops": ["verify", "verify"]}, False),
    "key_ops-non-string": ({"key_ops": ["verify", 5]}, False),
    "use-enc": ({"use": "enc"}, False),
    "use-empty": ({"use": ""}, False),
    "use-number": ({"use": 5}, False),
    "alg-RS256": ({"alg": "RS256"}, False),
    "alg-empty": ({"alg": ""}, False),
    "alg-number": ({"alg": 5}, False),
}


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_jwk_member_rules_parity(family):
    """A JWKS entry that carried the private key verified in Python and failed
    in JS; a non-string key_ops entry did the reverse, and a key labelled
    alg "Ed25519", as WebCrypto exports it, verified in JS and not in Python."""
    env, jwks, schema, pyverify = _setup(family)
    cases = []
    for tag, (members, _) in _JWK_MEMBERS.items():
        j = copy.deepcopy(jwks)
        key = next(k for k in j["keys"] if k.get("kid") == env["kid"])
        key.update({"d": _SIGNING_KEYS[family]["d"]} if members is None else members)
        cases.append((tag, env, j))
    py = [_py_verdict(pyverify, r, j) for _, r, j in cases]
    node = _node_verdicts(family, cases, schema)
    wrong = []
    for (tag, _, _), pv, nv in zip(cases, py, node):
        want = ("ACCEPT", None) if _JWK_MEMBERS[tag][1] else ("REJECT", "malformed_jwks")
        if not (pv == nv == want):
            wrong.append((tag, f"py={pv}", f"node={nv}", f"want={want}"))
    assert not wrong, f"[{family}] JWK member verdicts differ: {wrong}"


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_jwk_member_rule_runs_at_the_same_step_in_both(family):
    """Check order decides the class when a receipt has two faults. The member
    rule runs where the key is imported: after the protected-header checks
    (an unknown crit name wins) and before the signature-encoding check (the
    key fault wins), in both runtimes."""
    env, jwks, schema, pyverify = _setup(family)
    proto, _, sig = env["jws"].split(".")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    respelled = copy.deepcopy(env)
    respelled["jws"] = f"{proto}..{sig[:-1]}{alphabet[(alphabet.index(sig[-1]) & ~3) | 1]}"
    header = json.loads(base64.urlsafe_b64decode(proto + "=" * (-len(proto) % 4)))
    header["crit"] = list(header.get("crit", [])) + ["zip2"]
    header["zip2"] = 1
    crit_edit = copy.deepcopy(env)
    crit_edit["jws"] = (base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode())
                        .rstrip(b"=").decode() + ".." + sig)
    cases, want = [], []
    for member, value in (("d", _SIGNING_KEYS[family]["d"]), ("alg", "RS256")):
        bad = copy.deepcopy(jwks)
        next(k for k in bad["keys"] if k.get("kid") == env["kid"])[member] = value
        cases += [(f"{member}+respelled-signature", respelled, bad), (f"{member}+unknown-crit", crit_edit, bad)]
        want += [("REJECT", "malformed_jwks"), ("REJECT", "crit_unknown_extension")]
    py = [_py_verdict(pyverify, r, j) for _, r, j in cases]
    node = _node_verdicts(family, cases, schema)
    for (tag, _, _), pv, nv, w in zip(cases, py, node, want):
        assert pv == nv == w, (family, tag, pv, nv, w)


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_jwks_entry_with_a_private_member_is_refused_by_both(family):
    """joserfc imported a JWK that carries d from d alone and never compared it
    with x. With one key's x and another key's d, Python verified the other
    key's signatures, so a caller pinning x (sum_verify TrustPolicy) trusted a
    key it never approved. WebCrypto refuses any JWK with d for verification."""
    from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope

    env, jwks, schema, pyverify = _setup(family)
    other, _, _ = _keypair()
    by_other = sign_jose_envelope(copy.deepcopy(env["payload"]), private_jwk=other, kid=env["kid"])
    by_other["schema"] = schema
    foreign_d = copy.deepcopy(jwks)
    next(k for k in foreign_d["keys"] if k.get("kid") == env["kid"])["d"] = other["d"]
    cases = [("foreign-d/genuine-signature", env, foreign_d),
             ("foreign-d/signed-by-the-d-key", by_other, foreign_d)]
    py = [_py_verdict(pyverify, r, j) for _, r, j in cases]
    node = _node_verdicts(family, cases, schema)
    for (tag, _, _), pv, nv in zip(cases, py, node):
        assert pv == nv == ("REJECT", "malformed_jwks"), (family, tag, pv, nv)


# Broad garbage battery for JS-side totality: every Node verifier must throw a
# structured VerifyError (-> {"v":false,"c":...}) or verify, NEVER an undeclared
# exception (-> {"crash":...}). Complements the Python totality properties.
_GARBAGE_RECEIPTS = [
    None, 0, 1, -1, 1.5, "", "x", True, False, [], [1, 2, 3], [[[]]], {},
    {"schema": "sum.render_receipt.v1"},
    {"kid": None, "payload": None, "jws": None, "schema": None},
    {"kid": "k", "payload": {"a": {"b": {"c": [1, [2, [3]]]}}}, "jws": "a.b.c",
     "schema": "sum.render_receipt.v1"},
    {"kid": 5, "payload": [1], "jws": 7, "schema": 9},
    {"kid": "k", "payload": "x" * 5000, "jws": "." * 100, "schema": "s"},
    {"kid": "k", "payload": {"x": 1e308}, "jws": "a.b", "schema": "sum.render_receipt.v1"},
    {"kid": "\u200b", "payload": {"\ufeff": " "}, "jws": "\U0001f642.\U0001f642.\U0001f642",
     "schema": "sum.render_receipt.v1"},
]
_GARBAGE_JWKS = [None, 0, "x", [], [1], {}, {"keys": None}, {"keys": 5},
                 {"keys": [None, 1, "x"]}, {"keys": [{"kid": None}]}]


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_node_verifier_totality_on_garbage(family):
    """JS verifier never crashes (undeclared throw) on broad garbage — it always
    resolves or rejects with a VerifyError."""
    schema = {"render": "sum.render_receipt.v1", "transform": "sum.transform_receipt.v1",
              "meaning": "sum.meaning_risk_receipt.v1"}[family]
    cases = ([{"receipt": r, "jwks": {"keys": []}, "schema": schema} for r in _GARBAGE_RECEIPTS]
             + [{"receipt": {"kid": "k", "payload": {}, "jws": "a.b.c", "schema": schema}, "jwks": j,
                 "schema": schema} for j in _GARBAGE_JWKS])
    lines = "\n".join(json.dumps(c) for c in cases)
    res = subprocess.run([NODE, str(DRIVER), family], input=lines,
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"node driver failed: {res.stderr[:300]}"
    crashes = [(i, ln) for i, ln in enumerate(res.stdout.split("\n")) if '"crash"' in ln]
    assert not crashes, f"[{family}] Node verifier CRASHED (undeclared throw) on garbage: {crashes}"
