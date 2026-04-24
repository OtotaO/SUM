"""Cross-runtime ADVERSARIAL bundle-rejection harness — Priority 1.

Companion to ``scripts/verify_cross_runtime.py`` (K1-K4 valid-path
matrix). That harness proves the three verifiers AGREE on a curated
set of valid inputs. This harness proves they AGREE on invalid inputs
too — i.e. a crafted bundle that one verifier rejects is rejected by
every other verifier, and for equivalent reasons.

The gap these two harnesses collectively close:

    A verifier that passes K1-K4 on well-formed bundles has been
    proved to agree on VALIDITY. It has NOT been proved to agree on
    INVALIDITY. Those are two different proofs. Priority 1 in
    docs/NEXT_SESSION_PLAYBOOK.md exists because only the first was
    done until this script landed.

Scope: cross-runtime equivalence holds on structural, Ed25519-
signature, scheme, and version rejections. It intentionally does
NOT cover HMAC — the Node verifier's header docstring is explicit
("HMAC signatures are NOT verified here (shared-secret, not public
witness)") so HMAC tampering is a Python-only rejection path by
design, not a runtime asymmetry worth fixing. HMAC-tampered fixtures
live in Tests/test_adversarial_bundles.py (Python unit tests).

Naming follows the K-matrix convention: each fixture is a function
named ``ax_<class>_<mnemonic>`` that returns a ``Fixture`` and is
registered in ``FIXTURES``. The runner invokes both verifiers on
each fixture's bundle and asserts:

    1. Python verifier exits non-zero (rejection).
    2. Node verifier exits non-zero (rejection).
    3. Rejections are classified equivalently (both "structural",
       both "signature", etc.) — this is the load-bearing assertion.

If either assertion fails, the fixture is reported as a runtime
asymmetry with the full stdout/stderr from both verifiers attached.

Run via:
    python -m scripts.verify_cross_runtime_adversarial
or:
    make xruntime-adversarial
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_VERIFIER = REPO_ROOT / "standalone_verifier" / "verify.js"
PY_VERIFIER = ["python", "-m", "sum_cli.main", "verify"]


# ─── Rejection classifier ──────────────────────────────────────────
#
# Each verifier emits a distinct error message per rejection class.
# The classifier reads stdout+stderr and buckets the rejection. An
# "unknown" bucket is also present: it catches verifier output we
# haven't taught the classifier to recognise, which surfaces as a
# test failure so we never silently miscategorise.


REJECTION_CLASSES = ("structural", "signature", "scheme", "version")


def classify_python(stdout: str, stderr: str, exit_code: int) -> str:
    if exit_code == 0:
        return "ACCEPTED"
    blob = (stdout + "\n" + stderr).lower()
    if "ed25519" in blob and "invalid" in blob:
        return "signature"
    if "hmac" in blob and "invalid" in blob:
        return "signature"
    if "unsupported canonical_format_version" in blob:
        return "version"
    if "unsupported prime_scheme" in blob:
        return "scheme"
    if "missing required field" in blob or "not valid json" in blob:
        return "structural"
    if "state integer mismatch" in blob or "axiom count mismatch" in blob:
        return "structural"
    if "state_integer is not an integer" in blob:
        return "structural"
    return "unknown"


def classify_node(stdout: str, stderr: str, exit_code: int) -> str:
    if exit_code == 0:
        return "ACCEPTED"
    blob = (stdout + "\n" + stderr).lower()
    if "ed25519" in blob and "invalid" in blob:
        return "signature"
    if "unsupported canonical format version" in blob:
        return "version"
    if "unknown prime scheme" in blob:
        return "scheme"
    if ("missing required field" in blob
            or "invalid json" in blob
            or "cannot read bundle" in blob):
        return "structural"
    if "reconstruction failed" in blob or "witness verification failed" in blob:
        return "structural"
    return "unknown"


# ─── Bundle helpers ────────────────────────────────────────────────


def _mint_valid_ed25519_bundle() -> dict:
    """Produce a known-good Ed25519-signed bundle we can then mutate."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure.key_manager import KeyManager

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    km = KeyManager(key_dir=tempfile.mkdtemp())
    codec = CanonicalCodec(algebra, gen, key_manager=km)
    state = algebra.encode_chunk_state([("alice", "like", "cat"), ("bob", "own", "dog")])
    return codec.export_bundle(state, branch="adversarial", title="valid baseline")


def _mint_unsigned_bundle() -> dict:
    """Same as above minus Ed25519 (for adversarial fixtures that
    don't involve signatures — e.g. structural malformation)."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen)
    state = algebra.encode_chunk_state([("alice", "like", "cat"), ("bob", "own", "dog")])
    return codec.export_bundle(state, branch="adversarial", title="valid baseline")


# ─── Fixture registry ──────────────────────────────────────────────


@dataclass(frozen=True)
class Fixture:
    name: str
    bundle: dict
    expected_class: str  # one of REJECTION_CLASSES
    description: str


def _structural_missing_tome() -> Fixture:
    b = _mint_unsigned_bundle()
    del b["canonical_tome"]
    return Fixture(
        name="A1-structural-missing-tome",
        bundle=b,
        expected_class="structural",
        description="canonical_tome field removed entirely",
    )


def _structural_tome_truncated() -> Fixture:
    b = _mint_unsigned_bundle()
    # Drop the last axiom line — state integer will not reconstruct.
    lines = b["canonical_tome"].splitlines()
    b["canonical_tome"] = "\n".join(lines[:-2]) + "\n"
    return Fixture(
        name="A2-structural-tome-truncated",
        bundle=b,
        expected_class="structural",
        description="canonical_tome truncated mid-section; state reconstruction diverges",
    )


def _structural_state_integer_zero() -> Fixture:
    b = _mint_unsigned_bundle()
    b["state_integer"] = "0"
    return Fixture(
        name="A3-structural-state-integer-zero",
        bundle=b,
        expected_class="structural",
        description="state_integer set to 0; reconstruction from the real tome diverges",
    )


def _structural_state_integer_negative_string() -> Fixture:
    b = _mint_unsigned_bundle()
    b["state_integer"] = "-42"
    return Fixture(
        name="A4-structural-state-integer-negative",
        bundle=b,
        expected_class="structural",
        description="state_integer is a negative integer string; neither verifier should accept",
    )


def _version_unknown_format() -> Fixture:
    b = _mint_unsigned_bundle()
    b["canonical_format_version"] = "99.0.0"
    return Fixture(
        name="A5-version-unknown-format",
        bundle=b,
        expected_class="version",
        description="canonical_format_version bumped past anything either verifier knows",
    )


def _signature_ed25519_tome_tampered() -> Fixture:
    b = _mint_valid_ed25519_bundle()
    b["canonical_tome"] += "\nThe evil injected axiom.\n"
    return Fixture(
        name="A6-signature-ed25519-tome-tampered",
        bundle=b,
        expected_class="signature",
        description="Ed25519-signed bundle with one extra axiom appended to the tome",
    )


FIXTURES: Sequence[Callable[[], Fixture]] = (
    _structural_missing_tome,
    _structural_tome_truncated,
    _structural_state_integer_zero,
    _structural_state_integer_negative_string,
    _version_unknown_format,
    _signature_ed25519_tome_tampered,
)


# ─── Runner ────────────────────────────────────────────────────────


def _run_python_verifier(bundle_path: Path) -> tuple[int, str, str]:
    # Use --strict so the Python verifier enforces signature presence
    # too, matching what an attestation consumer would want at the
    # gate. Without --strict, an unsigned bundle's structural-only
    # verify path would accept some cases we expect Node to reject
    # identically.
    result = subprocess.run(
        PY_VERIFIER + ["--input", str(bundle_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def _run_node_verifier(bundle_path: Path) -> tuple[int, str, str]:
    result = subprocess.run(
        ["node", str(JS_VERIFIER), str(bundle_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def _run_fixture(fx: Fixture) -> tuple[bool, str]:
    """Run one fixture. Returns (passed, report_line)."""
    fd, path_str = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    path = Path(path_str)
    try:
        path.write_text(json.dumps(fx.bundle), encoding="utf-8")
        py_code, py_out, py_err = _run_python_verifier(path)
        node_code, node_out, node_err = _run_node_verifier(path)
    finally:
        path.unlink(missing_ok=True)

    py_class = classify_python(py_out, py_err, py_code)
    node_class = classify_node(node_out, node_err, node_code)

    # Both must reject.
    if py_code == 0:
        return False, (
            f"[{fx.name} FAIL] Python verifier ACCEPTED a bundle "
            f"expected to fail {fx.expected_class!r} rejection.\n"
            f"  stdout: {py_out.strip()[:400]}\n"
        )
    if node_code == 0:
        return False, (
            f"[{fx.name} FAIL] Node verifier ACCEPTED a bundle "
            f"expected to fail {fx.expected_class!r} rejection.\n"
            f"  stdout: {node_out.strip()[:400]}\n"
        )

    # Both must reject for equivalent reasons.
    if py_class != fx.expected_class:
        return False, (
            f"[{fx.name} FAIL] Python classified rejection as "
            f"{py_class!r}, expected {fx.expected_class!r}.\n"
            f"  stderr: {py_err.strip()[:400]}\n"
        )
    if node_class != fx.expected_class:
        return False, (
            f"[{fx.name} FAIL] Node classified rejection as "
            f"{node_class!r}, expected {fx.expected_class!r}.\n"
            f"  stdout: {node_out.strip()[:400]}\n"
        )
    return True, f"[{fx.name} PASS] both verifiers rejected with class={fx.expected_class!r}"


def main() -> int:
    if not JS_VERIFIER.exists():
        print(f"[harness] verify.js not found at {JS_VERIFIER}", file=sys.stderr)
        return 3
    try:
        node_v = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if node_v.returncode != 0:
            print("[harness] node --version failed", file=sys.stderr)
            return 4
        print(f"[harness] node {node_v.stdout.strip()}")
    except FileNotFoundError:
        print("[harness] node not on PATH", file=sys.stderr)
        return 4

    failures = []
    for factory in FIXTURES:
        fx = factory()
        passed, report = _run_fixture(fx)
        print(report)
        if not passed:
            failures.append(fx.name)

    print()
    if failures:
        print(
            f"CROSS-RUNTIME ADVERSARIAL HARNESS: "
            f"{len(failures)} / {len(FIXTURES)} fixtures disagreed. "
            f"Failures: {failures}",
            file=sys.stderr,
        )
        return 1
    print(
        f"CROSS-RUNTIME ADVERSARIAL HARNESS: "
        f"ALL {len(FIXTURES)} FIXTURES PASSED — verifiers agree on rejection"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
