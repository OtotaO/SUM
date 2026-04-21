"""End-to-end contract for ``sum attest --ed25519-key``.

Until this flag landed, the CLI could mint bundles signed only with
HMAC (--signing-key) or unsigned. That made the Ed25519 verification
path (sum verify, the browser demo) theoretical — nothing in the
shipping toolbelt actually *produced* Ed25519-signed bundles. This
test pins the round-trip: a PEM private key in → a bundle with
public_signature + public_key out → `sum verify --strict` passes.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sum_cli.main import cmd_attest, cmd_verify


def _write_ed25519_pem(path: Path) -> None:
    sk = Ed25519PrivateKey.generate()
    pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)


def _run_attest(
    text: str,
    *,
    ed25519_key: str | None = None,
    signing_key: str | None = None,
    tmp_path: Path,
) -> dict:
    """Invoke cmd_attest in-process, capture the emitted bundle JSON."""
    in_path = tmp_path / "in.txt"
    in_path.write_text(text)
    args = argparse.Namespace(
        input=str(in_path),
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="Attested Tome",
        signing_key=signing_key,
        ed25519_key=ed25519_key,
        pretty=False,
        verbose=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_attest(args)
    finally:
        sys.stdout = old
    assert code == 0, f"attest failed: {buf.getvalue()}"
    return json.loads(buf.getvalue())


def _run_verify(bundle: dict, *, signing_key=None, strict=False, tmp_path: Path) -> tuple[int, dict]:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    args = argparse.Namespace(
        input=str(path),
        signing_key=signing_key,
        strict=strict,
        pretty=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_verify(args)
    finally:
        sys.stdout = old
    out = buf.getvalue().strip()
    parsed = json.loads(out) if (out and code == 0) else {}
    return code, parsed


class TestAttestEd25519:

    def test_ed25519_key_produces_signed_bundle(self, tmp_path):
        pem = tmp_path / "sk.pem"
        _write_ed25519_pem(pem)
        bundle = _run_attest(
            "Alice likes cats. Bob owns a dog.",
            ed25519_key=str(pem),
            tmp_path=tmp_path,
        )
        assert bundle["public_signature"].startswith("ed25519:")
        assert bundle["public_key"].startswith("ed25519:")
        # HMAC field absent when --signing-key not supplied.
        assert "signature" not in bundle

    def test_ed25519_bundle_round_trips_through_verify(self, tmp_path):
        pem = tmp_path / "sk.pem"
        _write_ed25519_pem(pem)
        bundle = _run_attest(
            "Alice likes cats. Bob owns a dog.",
            ed25519_key=str(pem),
            tmp_path=tmp_path,
        )
        code, result = _run_verify(bundle, strict=True, tmp_path=tmp_path)
        assert code == 0
        assert result["signatures"]["ed25519"] == "verified"

    def test_dual_hmac_and_ed25519(self, tmp_path):
        pem = tmp_path / "sk.pem"
        _write_ed25519_pem(pem)
        bundle = _run_attest(
            "Alice likes cats.",
            ed25519_key=str(pem),
            signing_key="dual-key",
            tmp_path=tmp_path,
        )
        assert bundle["signature"].startswith("hmac-sha256:")
        assert bundle["public_signature"].startswith("ed25519:")
        code, result = _run_verify(
            bundle, signing_key="dual-key", strict=True, tmp_path=tmp_path
        )
        assert code == 0
        assert result["signatures"] == {"hmac": "verified", "ed25519": "verified"}

    def test_missing_key_file_raises_with_pointer(self, tmp_path):
        in_path = tmp_path / "in.txt"
        in_path.write_text("Alice likes cats.")
        args = argparse.Namespace(
            input=str(in_path),
            extractor="sieve",
            model=None,
            source=None,
            branch="main",
            title="t",
            signing_key=None,
            ed25519_key=str(tmp_path / "nonexistent.pem"),
            pretty=False,
            verbose=False,
        )
        with pytest.raises(SystemExit, match="generate_did_web"):
            cmd_attest(args)

    def test_rsa_key_rejected(self, tmp_path):
        from cryptography.hazmat.primitives.asymmetric import rsa

        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        rsa_pem = rsa_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pem_path = tmp_path / "rsa.pem"
        pem_path.write_bytes(rsa_pem)
        in_path = tmp_path / "in.txt"
        in_path.write_text("Alice likes cats.")
        args = argparse.Namespace(
            input=str(in_path),
            extractor="sieve",
            model=None,
            source=None,
            branch="main",
            title="t",
            signing_key=None,
            ed25519_key=str(pem_path),
            pretty=False,
            verbose=False,
        )
        with pytest.raises(SystemExit, match="Ed25519"):
            cmd_attest(args)

    def test_tampered_ed25519_bundle_rejected_by_verify(self, tmp_path):
        pem = tmp_path / "sk.pem"
        _write_ed25519_pem(pem)
        bundle = _run_attest(
            "Alice likes cats.",
            ed25519_key=str(pem),
            tmp_path=tmp_path,
        )
        bundle["canonical_tome"] += "\nThe evil injected axiom."
        code, _ = _run_verify(bundle, tmp_path=tmp_path)
        assert code == 1
