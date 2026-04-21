"""Cross-runtime CanonicalBundle portability harness.

Locks the load-bearing contract for the single-file deployment moonshot:
SUM's CanonicalBundle wire format must round-trip between the Python
signer (``internal.infrastructure.canonical_codec.CanonicalCodec``) and
the JavaScript verifier (``standalone_verifier/verify.js``) at byte level
on the canonical-tome side and at integer level on the Gödel state side.

This is the instrumentation arm of the v0 single-file moonshot. Any
future change that breaks the contract — a new field in CanonicalBundle
that verify.js does not understand, a change to the prime-derivation
scheme on either side, a tome-template alteration, a JSON-shape drift —
trips this script with a non-zero exit, and the Python side cannot ship
until the JS side is updated to match.

The harness is INTENTIONALLY not a pytest case. It depends on Node.js
(>= 16), which is a CI/runtime dep separate from the Python test
toolchain. Pytest runs stay Python-only; this script runs in its own
job (``python -m scripts.verify_cross_runtime``). CI: any non-zero exit
gates the merge.

Two kill-experiments are run, both with named outcomes:

  K1 (positive):  Mint a CanonicalBundle in Python on a known-good axiom
                  set; pipe to verify.js; assert exit 0 and the
                  ``WITNESS VERIFICATION PASSED`` banner is present.
                  This proves the cross-runtime contract holds today.

  K2 (named-failure): Mint a VC 2.0 + eddsa-jcs-2022 credential in
                  Python; pipe to verify.js; assert exit 2 and the
                  ``Missing required field`` rejection is present.
                  This proves verify.js is the legacy CanonicalBundle
                  verifier — NOT a VC 2.0 verifier — and that it
                  rejects unknown formats cleanly rather than panicking
                  or silently producing wrong output.

If K2 ever starts succeeding, verify.js gained VC 2.0 support and the
moonshot's M2 milestone (axiom-level VC attribution) is unblocked on
the JS side. Update this script's expected-exit assertion when that
happens.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIER = REPO_ROOT / "standalone_verifier" / "verify.js"

K1_EXPECTED_BANNER = "WITNESS VERIFICATION PASSED"
K2_EXPECTED_REJECTION = "Missing required field"


def _run_verifier(bundle_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(VERIFIER), str(bundle_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _mint_bundle(triples: list[tuple[str, str, str]], title: str) -> dict:
    """Helper: mint a CanonicalBundle from the given triples via the
    production codec path. Factored so K1 + K1-multiword share the same
    mint logic and we know they're testing the cross-runtime parser
    under identical bundle shape."""
    from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from internal.infrastructure.canonical_codec import CanonicalCodec

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen)
    state = algebra.encode_chunk_state(triples)
    return codec.export_bundle(state, branch="main", title=title)


def _verify_bundle(bundle: dict, label: str, expected_banner: str) -> bool:
    fd, path_str = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    path = Path(path_str)
    try:
        path.write_text(json.dumps(bundle), encoding="utf-8")
        result = _run_verifier(path)
    finally:
        path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[{label} FAIL] verify.js exited {result.returncode}", file=sys.stderr)
        print(f"[{label} stdout] {result.stdout}", file=sys.stderr)
        print(f"[{label} stderr] {result.stderr}", file=sys.stderr)
        return False
    if expected_banner not in result.stdout:
        print(
            f"[{label} FAIL] expected banner {expected_banner!r} not in stdout",
            file=sys.stderr,
        )
        return False
    return True


def k1_canonical_bundle_round_trip() -> bool:
    """Python mints single-token-object CanonicalBundle → JS verifies → exit 0 expected."""
    bundle = _mint_bundle(
        [("alice", "like", "cat"), ("bob", "own", "dog")],
        title="cross-runtime harness K1",
    )
    fd, path_str = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    path = Path(path_str)
    try:
        path.write_text(json.dumps(bundle), encoding="utf-8")
        result = _run_verifier(path)
    finally:
        path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[K1 FAIL] verify.js exited {result.returncode}", file=sys.stderr)
        print(f"[K1 stdout] {result.stdout}", file=sys.stderr)
        print(f"[K1 stderr] {result.stderr}", file=sys.stderr)
        return False
    if K1_EXPECTED_BANNER not in result.stdout:
        print(
            f"[K1 FAIL] expected banner {K1_EXPECTED_BANNER!r} "
            f"not in stdout",
            file=sys.stderr,
        )
        print(f"[K1 stdout] {result.stdout}", file=sys.stderr)
        return False
    print("[K1 PASS] CanonicalBundle round-trips Python → verify.js")
    return True


def k2_vc2_named_rejection() -> bool:
    """Python mints VC 2.0 → JS rejects with named error → exit 2 expected.

    If this assertion ever fails (verify.js returns 0 on a VC 2.0 bundle),
    it means VC 2.0 support was added to verify.js. Update this function
    to re-assert the new expected behaviour rather than treat the change
    as a regression.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from internal.infrastructure.verifiable_credential import (
        make_credential,
        sign_credential,
    )

    sk = Ed25519PrivateKey.generate()
    cred = make_credential(
        subject={"axiom": "alice||like||cat", "stateInteger": "12345"},
        issuer="did:example:cross-runtime-harness",
        credential_type="SumAxiomAttestation",
        valid_from="2026-04-19T00:00:00Z",
    )
    signed = sign_credential(
        cred,
        sk,
        "did:example:cross-runtime-harness#key-1",
        created="2026-04-19T00:00:00Z",
    )

    fd, path_str = tempfile.mkstemp(suffix=".jsonld")
    os.close(fd)
    path = Path(path_str)
    try:
        path.write_text(json.dumps(signed), encoding="utf-8")
        result = _run_verifier(path)
    finally:
        path.unlink(missing_ok=True)

    if result.returncode == 0:
        print(
            "[K2 ALERT] verify.js accepted a VC 2.0 bundle — VC 2.0 support "
            "appears to have landed. Update k2_vc2_named_rejection in this "
            "harness to assert the new expected behaviour.",
            file=sys.stderr,
        )
        return False
    if K2_EXPECTED_REJECTION not in result.stderr:
        print(
            f"[K2 FAIL] expected rejection {K2_EXPECTED_REJECTION!r} "
            f"not in stderr",
            file=sys.stderr,
        )
        print(f"[K2 stderr] {result.stderr}", file=sys.stderr)
        return False
    print(
        "[K2 PASS] verify.js cleanly rejects VC 2.0 with named error "
        "(no panic, no silent acceptance)"
    )
    return True


def k1_multiword_object_round_trip() -> bool:
    """Python mints a bundle with MULTI-WORD OBJECTS → JS verifies → exit 0 expected.

    This is the regression guard for the real bug caught by the `sum`
    CLI's first kill-experiment on 2026-04-21: `verify.js` used
    ``(\\S+)`` for the object capture while the Python parser used
    ``(.+)``, so every tome with objects like "nobel prizes" or
    "printing press" reconstructed to state=1 in Node while Python
    produced the correct state. Fixed in the same session; this fixture
    guarantees the drift cannot resurface.
    """
    bundle = _mint_bundle(
        [
            ("marie_curie", "win", "nobel prizes"),
            ("johannes_gutenberg", "invent", "printing press"),
            ("ada_lovelace", "write", "computer algorithm"),
        ],
        title="cross-runtime harness K1-multiword",
    )
    ok = _verify_bundle(bundle, "K1-multiword", K1_EXPECTED_BANNER)
    if ok:
        print("[K1-multiword PASS] multi-word-object tome round-trips Python → verify.js")
    return ok


def main() -> int:
    if not VERIFIER.exists():
        print(
            f"[harness] verify.js not found at {VERIFIER}; "
            "cross-runtime portability cannot be verified",
            file=sys.stderr,
        )
        return 3

    try:
        node_v = subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        )
        if node_v.returncode != 0:
            print("[harness] node --version failed", file=sys.stderr)
            return 4
        print(f"[harness] node {node_v.stdout.strip()}")
    except FileNotFoundError:
        print(
            "[harness] node not found on PATH; install Node.js >= 16 to "
            "run the cross-runtime portability harness",
            file=sys.stderr,
        )
        return 4

    ok_k1 = k1_canonical_bundle_round_trip()
    ok_k1_mw = k1_multiword_object_round_trip()
    ok_k2 = k2_vc2_named_rejection()

    if ok_k1 and ok_k1_mw and ok_k2:
        print("\nCROSS-RUNTIME PORTABILITY HARNESS: ALL CHECKS PASSED")
        return 0
    print("\nCROSS-RUNTIME PORTABILITY HARNESS: REGRESSION", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
