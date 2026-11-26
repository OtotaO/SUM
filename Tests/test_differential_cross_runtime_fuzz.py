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
    # Not strings, and not printable by JS without help: the message names the
    # type, so the class stays malformed_jwks instead of an unclassed TypeError.
    "kty-object": ({"kty": {"toString": 1, "valueOf": 1}}, False),
    "crv-object": ({"crv": {"toString": 1, "valueOf": 1}}, False),
    "kty-array": ({"kty": ["OKP"]}, False),
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


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _signed_under(env, family, header=None, raw=None, payload=None):
    """env's payload (or a replacement) signed under a chosen protected header,
    as an issuer would: over proto + "." + the JCS bytes when the header says
    b64 false (or raw header bytes are given), otherwise over proto + "." +
    base64url(JCS), the RFC 7797 default."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from sum_engine_internal.infrastructure.jcs import canonicalize

    out = copy.deepcopy(env)
    if payload is not None:
        out["payload"] = payload
    hb = raw if raw is not None else json.dumps(header, separators=(",", ":")).encode()
    proto, jcs = _b64u(hb), canonicalize(out["payload"])
    unencoded = raw is not None or header.get("b64", True) is False
    d = _SIGNING_KEYS[family]["d"]
    sk = Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(d + "=" * (-len(d) % 4)))
    sig = sk.sign(proto.encode("ascii") + b"." + (jcs if unencoded else _b64u(jcs).encode()))
    out["jws"] = f"{proto}..{_b64u(sig)}"
    return out


def _string_field(payload):
    return next(k for k in sorted(payload) if isinstance(payload[k], str))


def _header_cases(env, family):
    """(tag, receipt, expected verdict) for correctly signed receipts whose
    protected header or payload size is the only question."""
    kid = env["kid"]

    def H(**kw):
        return {"alg": "EdDSA", "kid": kid, "b64": False, "crit": ["b64"], **kw}

    def without(*names):
        h = H()
        for n in names:
            h.pop(n)
        return h

    ok, mj, hiv = ("ACCEPT", None), ("REJECT", "malformed_jws"), ("REJECT", "header_invariant_violated")
    base = json.dumps(H(), separators=(",", ":")).encode()
    cases = [
        ("control", _signed_under(env, family, H()), ok),
        ("b64 missing, no crit", _signed_under(env, family, without("b64", "crit")), hiv),
        ("b64 missing, crit [b64]", _signed_under(env, family, without("b64")), hiv),
        ("b64 true", _signed_under(env, family, H(b64=True)), hiv),
        ("b64 the string false", _signed_under(env, family, H(b64="false")), mj),
        ("crit absent", _signed_under(env, family, without("crit")), hiv),
        ("crit empty", _signed_under(env, family, H(crit=[])), hiv),
        ("crit names b64 twice", _signed_under(env, family, H(crit=["b64", "b64"])), ok),
        ("crit names kid", _signed_under(env, family, H(crit=["b64", "kid"])), ("REJECT", "crit_unknown_extension")),
        ("crit as a string", _signed_under(env, family, H(crit="b64")), mj),
        ("alg a number", _signed_under(env, family, H(alg=5)), ("REJECT", "unsupported_alg")),
        ("jku https", _signed_under(env, family, H(jku="https://example.com/jwks")), ok),
        ("typ a string", _signed_under(env, family, H(typ="JOSE")), ok),
        ("BOM before the header", _signed_under(env, family, raw=b"\xef\xbb\xbf" + base), mj),
        ("invalid UTF-8 in typ", _signed_under(env, family, raw=base[:-1] + b',"typ":"\xff"}'), mj),
        ("NaN in the header", _signed_under(env, family, raw=base[:-1] + b',"typ":NaN}'), mj),
        ("Infinity in the header", _signed_under(env, family, raw=base[:-1] + b',"typ":Infinity}'), mj),
        # Python accepted NaN or Infinity nested in a jwk object (joserfc
        # checks only that jwk is an object) and JS refused it; Python moves
        # from accept to reject on these.
        ("NaN inside a jwk object", _signed_under(env, family, raw=base[:-1] + b',"jwk":{"a":NaN}}'), mj),
        ("Infinity inside a jwk object", _signed_under(env, family, raw=base[:-1] + b',"jwk":{"a":[-Infinity]}}'), mj),
        ("crit holding a number", _signed_under(env, family, H(crit=["b64", 1])), ("REJECT", "crit_unknown_extension")),
        # An object JS cannot print without help: the message names its type,
        # so the class is the same as Python's instead of an unclassed TypeError.
        ("crit holding an object", _signed_under(env, family, H(crit=["b64", {"toString": 1, "valueOf": 1}])),
         ("REJECT", "crit_unknown_extension")),
        ("alg an object", _signed_under(env, family, H(alg={"toString": 1, "valueOf": 1})), ("REJECT", "unsupported_alg")),
    ]
    for name, bad in (("jku", "ftp://x"), ("jku", 5), ("jwk", "x"), ("jwk", [1]), ("kid", 5), ("x5u", "ftp://x"),
                      ("x5c", "x"), ("x5c", [1]), ("x5t", 5), ("x5t#S256", 5), ("typ", 5), ("cty", 5)):
        cases.append((f"{name}={bad!r}", _signed_under(env, family, H(**{name: bad})), mj))
    nested = cur = {}
    for _ in range(40):
        cur["a"] = {}
        cur = cur["a"]
    cases.append(("jwk nested 40 deep", _signed_under(env, family, H(jwk=nested)), ok))
    for want_len, want in ((512, ok), (514, mj)):  # base64url lengths are never 1 mod 4
        n = 0
        while len(_b64u(json.dumps(H(cty="a" * n), separators=(",", ":")).encode())) < want_len:
            n += 1
        cases.append((f"header of {want_len} base64url characters", _signed_under(env, family, H(cty="a" * n)), want))
    # Every family's payload has a string field to grow (meaning: corpus_id).
    from sum_engine_internal.infrastructure.jcs import canonicalize
    k = _string_field(env["payload"])
    for size, want in ((128_000, ok), (128_001, ("REJECT", "malformed_receipt"))):
        p = copy.deepcopy(env["payload"])
        p[k] = ""
        p[k] = "a" * (size - len(canonicalize(p)))
        assert len(canonicalize(p)) == size
        cases.append((f"payload of {size} JCS bytes", _signed_under(env, family, H(), payload=p), want))
    return cases


# A correctly signed receipt whose header has a member outside the 12
# registered JWS names: Python refuses it through joserfc's strict header
# check (as signature_invalid), the JS verifiers accept it. RENDER_RECEIPT_FORMAT
# section 1 says such a verifier "will succeed", so which runtime moves is held
# for the operator; this pins the current split so neither side changes alone.
KNOWN_VERDICT_DIVERGENCE = {
    "header member outside the registered names": (("REJECT", "signature_invalid"), ("ACCEPT", None)),
}


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_correctly_signed_headers_get_one_verdict(family):
    """The protected-header rule, applied by our own code in both runtimes
    before the signature check: header segment at most 512 base64url
    characters, strict UTF-8 without a BOM, JSON without NaN or Infinity,
    the type joserfc requires of each registered name, b64 false and crit
    naming b64; JCS payload at most 128,000 bytes. Before it, Python refused
    some of these through joserfc as signature_invalid and the JS verifiers
    accepted them, and the b64 variants split by class. A genuine signature
    must never be reported as signature_invalid."""
    env, jwks, schema, pyverify = _setup(family)
    cases = _header_cases(env, family)
    kid = env["kid"]
    split = _signed_under(env, family, {"alg": "EdDSA", "kid": kid, "b64": False, "crit": ["b64"], "foo": 1})
    rows = [(tag, r, jwks) for tag, r, _ in cases] + [("header member outside the registered names", split, jwks)]
    py = [_py_verdict(pyverify, r, j) for _, r, j in rows]
    node = _node_verdicts(family, rows, schema)
    for (tag, _, want), pv, nv in zip(cases, py, node):
        assert pv == nv == want, (family, tag, pv, nv, want)
    assert (py[-1], node[-1]) == KNOWN_VERDICT_DIVERGENCE[rows[-1][0]], (family, py[-1], node[-1])


# One fault per verification step, in the order both runtimes check them.
# Paired, the earlier step must decide the class in both runtimes, in every
# family (the transform and meaning JS verifiers checked alg before crit, and
# all three checked the key type at import rather than at the kid lookup).
_STEP_FAULTS = [
    ("unknown-kid", "unknown_kid"),
    ("non-ed25519-key", "malformed_jwks"),
    ("payload-lone-surrogate", "malformed_receipt"),
    ("payload-over-limit", "malformed_receipt"),
    ("two-segments", "malformed_jws"),
    ("header-over-512", "malformed_jws"),
    ("header-bom", "malformed_jws"),
    ("unknown-crit", "crit_unknown_extension"),
    ("alg-rs256", "unsupported_alg"),
    ("typ-number", "malformed_jws"),
    ("b64-true", "header_invariant_violated"),
    ("jwk-with-d", "malformed_jwks"),
    ("signature-flipped", "signature_invalid"),
]


def _with_faults(env, jwks, family, names):
    e, j = copy.deepcopy(env), copy.deepcopy(jwks)
    proto, _, sig = e["jws"].split(".")
    header = json.loads(base64.urlsafe_b64decode(proto + "=" * (-len(proto) % 4)))
    prefix, two_segments = b"", False
    key = next(k for k in j["keys"] if k.get("kid") == e["kid"])
    k = _string_field(e["payload"])
    for name in names:
        if name == "unknown-kid":
            e["kid"] = "nobody"
        elif name == "non-ed25519-key":
            key["kty"] = "RSA"
        elif name == "payload-lone-surrogate":
            e["payload"][k] = "\ud800" + e["payload"][k]
        elif name == "payload-over-limit":
            e["payload"][k] = "a" * 128_100
        elif name == "two-segments":
            two_segments = True
        elif name == "header-over-512":
            header["cty"] = "a" * 600
        elif name == "header-bom":
            prefix = b"\xef\xbb\xbf"
        elif name == "unknown-crit":
            header["crit"] = list(header["crit"]) + ["zzz"]
        elif name == "alg-rs256":
            header["alg"] = "RS256"
        elif name == "typ-number":
            header["typ"] = 5
        elif name == "b64-true":
            header["b64"] = True
        elif name == "jwk-with-d":
            key["d"] = _SIGNING_KEYS[family]["d"]
        elif name == "signature-flipped":
            sig = ("B" if sig[0] == "A" else "A") + sig[1:]
    proto = _b64u(prefix + json.dumps(header, separators=(",", ":")).encode())
    e["jws"] = f"{proto}.{sig}" if two_segments else f"{proto}..{sig}"
    return e, j


@pytest.mark.parametrize("family", ["render", "transform", "meaning"])
def test_the_earlier_step_decides_the_class_in_both(family):
    env, jwks, schema, pyverify = _setup(family)
    rows, want = [], []
    for i, (a, ca) in enumerate(_STEP_FAULTS):
        for b, _ in _STEP_FAULTS[i:]:
            names = (a,) if a == b else (a, b)
            r, j = _with_faults(env, jwks, family, names)
            rows.append(("+".join(names), r, j))
            want.append(("REJECT", ca))
    py = [_py_verdict(pyverify, r, j) for _, r, j in rows]
    node = _node_verdicts(family, rows, schema)
    wrong = [(tag, pv, nv, w) for (tag, _, _), pv, nv, w in zip(rows, py, node, want) if not pv == nv == w]
    assert not wrong, (family, len(wrong), wrong[:8])


def test_the_signer_refuses_a_payload_the_verifiers_refuse():
    from sum_engine_internal.infrastructure.jcs import canonicalize
    from sum_engine_internal.infrastructure.jose_envelope import MAX_PAYLOAD_JCS_BYTES, sign_jose_envelope

    pr, _, kid = _keypair()
    payload = {"model": ""}
    payload["model"] = "a" * (MAX_PAYLOAD_JCS_BYTES - len(canonicalize(payload)))
    sign_jose_envelope(payload, private_jwk=pr, kid=kid)  # at the limit: signed
    payload["model"] += "a"
    with pytest.raises(ValueError, match="128000"):
        sign_jose_envelope(payload, private_jwk=pr, kid=kid)


def test_the_signer_refuses_a_header_the_verifiers_refuse():
    """A long kid makes the protected header longer than the verifiers take;
    the signer refuses it instead of returning a receipt nobody accepts."""
    from sum_engine_internal.infrastructure.jose_envelope import (
        MAX_PROTECTED_HEADER_B64_CHARS, sign_jose_envelope, verify_jose_envelope,
    )

    pr, pub, _ = _keypair()
    lengths = {}
    for n in range(300, 360):
        try:
            env = sign_jose_envelope({"schema": "sum.test.v1", "a": 1}, private_jwk=pr, kid="k" * n)
        except ValueError as exc:
            assert f"the limit is {MAX_PROTECTED_HEADER_B64_CHARS}" in str(exc)
            lengths[n] = None
            continue
        lengths[n] = len(env["jws"].split(".", 1)[0])
        assert lengths[n] <= MAX_PROTECTED_HEADER_B64_CHARS
        env["schema"] = "sum.test.v1"
        verify_jose_envelope(env, {"keys": [dict(pub["keys"][0], kid="k" * n)]}, supported_schema="sum.test.v1")
    assert any(v is None for v in lengths.values()) and any(v for v in lengths.values())


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
