"""Signature-verification contract for ``sum verify``.

The CLI's ``verify`` subcommand used to do one thing: reconstruct the
state integer from the canonical tome and compare it to the claimed
``state_integer``. That caught tampering *of the tome* but silently
passed a bundle with a forged ``signature`` or a wrong ``public_key`` —
a verify command that never verified its signatures.

These tests pin the fixed contract. For each bundle shape, we assert:
  * exit code
  * which signatures were reported verified / skipped / absent / invalid

Kept lightweight: we construct bundles in-memory and invoke the command
through argparse namespaces (no subprocess, no stdin plumbing), so each
case runs in ~10 ms and covers every branch of cmd_verify without the
overhead of an integration shell.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math

import pytest

from sum_cli.main import cmd_verify


# ─── Fixtures ──────────────────────────────────────────────────────


def _mint_signed_bundle(signing_key: str | None, with_ed25519: bool) -> dict:
    """Use CanonicalCodec to mint a real, internally consistent bundle."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure.key_manager import KeyManager

    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    km = None
    if with_ed25519:
        import tempfile

        km = KeyManager(key_dir=tempfile.mkdtemp())
    codec = CanonicalCodec(algebra, tome_gen, signing_key=signing_key, key_manager=km)
    for s, p, o in [("alice", "likes", "cats"), ("bob", "knows", "python")]:
        algebra.get_or_mint_prime(s, p, o)
    state = 1
    for prime in algebra.axiom_to_prime.values():
        state = math.lcm(state, prime)
    return codec.export_bundle(state, branch="verify-test")


def _write_bundle(tmp_path, bundle: dict, name: str = "bundle.json"):
    path = tmp_path / name
    path.write_text(json.dumps(bundle))
    return path


def _run_verify(bundle_path, *, signing_key=None, strict=False) -> tuple[int, dict]:
    """Invoke cmd_verify, capture stdout, return (exit_code, parsed_json)."""
    args = argparse.Namespace(
        input=str(bundle_path),
        signing_key=signing_key,
        strict=strict,
        pretty=False,
    )
    buf = io.StringIO()
    import sys

    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_verify(args)
    finally:
        sys.stdout = old
    stdout = buf.getvalue().strip()
    parsed = json.loads(stdout) if (stdout and code == 0) else {}
    return code, parsed


# ─── 1. Unsigned bundle ─────────────────────────────────────────────


class TestUnsignedBundle:

    def test_unsigned_passes_default(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=False)
        path = _write_bundle(tmp_path, bundle)
        code, result = _run_verify(path)
        assert code == 0
        assert result["signatures"] == {"hmac": "absent", "ed25519": "absent"}

    def test_unsigned_fails_strict(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=False)
        path = _write_bundle(tmp_path, bundle)
        code, _ = _run_verify(path, strict=True)
        assert code == 1


# ─── 2. HMAC-signed bundle ─────────────────────────────────────────


class TestHmacSignedBundle:

    def test_hmac_skipped_without_key(self, tmp_path):
        """HMAC is present but no key supplied — truthful 'skipped' report."""
        bundle = _mint_signed_bundle(signing_key="correct-key", with_ed25519=False)
        path = _write_bundle(tmp_path, bundle)
        code, result = _run_verify(path)
        assert code == 0
        assert result["signatures"]["hmac"] == "skipped"

    def test_hmac_strict_skipped_fails(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key="correct-key", with_ed25519=False)
        path = _write_bundle(tmp_path, bundle)
        code, _ = _run_verify(path, strict=True)
        assert code == 1

    def test_hmac_verifies_with_correct_key(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key="correct-key", with_ed25519=False)
        path = _write_bundle(tmp_path, bundle)
        code, result = _run_verify(path, signing_key="correct-key", strict=True)
        assert code == 0
        assert result["signatures"]["hmac"] == "verified"

    def test_hmac_rejects_wrong_key(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key="correct-key", with_ed25519=False)
        path = _write_bundle(tmp_path, bundle)
        code, _ = _run_verify(path, signing_key="wrong-key")
        assert code == 1

    def test_hmac_rejects_tampered_tome(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key="correct-key", with_ed25519=False)
        tampered = copy.deepcopy(bundle)
        tampered["canonical_tome"] += "\nThe evil injected axiom."
        path = _write_bundle(tmp_path, tampered)
        code, _ = _run_verify(path, signing_key="correct-key")
        assert code == 1


# ─── 3. Ed25519-signed bundle ──────────────────────────────────────


class TestEd25519Bundle:

    def test_ed25519_verifies_self_contained(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=True)
        path = _write_bundle(tmp_path, bundle)
        # No signing key needed — Ed25519 is self-contained.
        code, result = _run_verify(path)
        assert code == 0
        assert result["signatures"]["ed25519"] == "verified"
        assert result["signatures"]["hmac"] == "absent"

    def test_ed25519_rejects_tampered_tome(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=True)
        tampered = copy.deepcopy(bundle)
        tampered["canonical_tome"] += "\nThe evil injected axiom."
        path = _write_bundle(tmp_path, tampered)
        code, _ = _run_verify(path)
        assert code == 1

    def test_ed25519_rejects_swapped_pubkey(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=True)
        bundle["public_key"] = "ed25519:" + base64.b64encode(b"\x00" * 32).decode()
        path = _write_bundle(tmp_path, bundle)
        code, _ = _run_verify(path)
        assert code == 1

    def test_ed25519_strict_passes_without_hmac(self, tmp_path):
        """Ed25519-only bundles satisfy --strict (one real signature is enough)."""
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=True)
        path = _write_bundle(tmp_path, bundle)
        code, result = _run_verify(path, strict=True)
        assert code == 0
        assert result["signatures"]["ed25519"] == "verified"


# ─── 4. Prime-scheme gate ──────────────────────────────────────────


class TestPrimeSchemeGate:

    def test_rejects_unknown_scheme(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=False)
        bundle["prime_scheme"] = "sha256_128_v2"  # not yet active
        path = _write_bundle(tmp_path, bundle)
        code, _ = _run_verify(path)
        assert code == 2  # malformed input class, not verification failure

    def test_rejects_unknown_format_version(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=False)
        bundle["canonical_format_version"] = "2.0.0"
        path = _write_bundle(tmp_path, bundle)
        code, _ = _run_verify(path)
        assert code == 2


# ─── 5. Malformed input ────────────────────────────────────────────


class TestMalformedInput:

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "junk.json"
        path.write_text("{not json")
        code, _ = _run_verify(path)
        assert code == 2

    def test_missing_required_field(self, tmp_path):
        bundle = _mint_signed_bundle(signing_key=None, with_ed25519=False)
        del bundle["canonical_tome"]
        path = _write_bundle(tmp_path, bundle)
        code, _ = _run_verify(path)
        assert code == 2
