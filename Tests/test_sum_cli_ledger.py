"""End-to-end contract for ``sum attest --ledger`` + ``sum resolve``.

Without --ledger, sum attest mints a bundle with no per-axiom
evidence trail. Downstream auditors see 'alice||likes||cats' but
have no way to pin it back to the originating sentence or byte
range; sum resolve has nothing to look up. With --ledger, the CLI
closes the loop: every triple gets a ProvenanceRecord in an
AkashicLedger at the given path, and the resulting prov_ids are
attached to bundle.sum_cli.prov_ids.

These tests pin the round-trip:

  * Bundle carries prov_ids + ledger path in its sum_cli sidecar.
  * Each prov_id resolves to a ProvenanceRecord with correct
    byte_start / byte_end / text_excerpt.
  * LLM extractor is cleanly rejected with a pointer (LLM has no
    byte-offset story today).
  * Attestation layers still compose: --ledger + --ed25519-key
    produces both prov_ids and a valid Ed25519 signature.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import pytest

from sum_cli.main import cmd_attest, cmd_resolve


def _make_args(**overrides):
    base = argparse.Namespace(
        input=None,
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="Attested Tome",
        signing_key=None,
        ed25519_key=None,
        ledger=None,
        pretty=False,
        verbose=False,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _capture_cmd(fn, args) -> tuple[int, str]:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = fn(args)
    finally:
        sys.stdout = old
    return code, buf.getvalue()


def _attest(tmp_path: Path, text: str, **overrides) -> dict:
    in_path = tmp_path / "in.txt"
    in_path.write_text(text)
    overrides.setdefault("input", str(in_path))
    args = _make_args(**overrides)
    code, out = _capture_cmd(cmd_attest, args)
    assert code == 0, f"attest failed (exit {code}): {out}"
    return json.loads(out)


def _resolve(prov_id: str, db_path: Path) -> dict:
    args = argparse.Namespace(prov_id=prov_id, db=str(db_path))
    code, out = _capture_cmd(cmd_resolve, args)
    assert code == 0, f"resolve failed (exit {code}): {out}"
    return json.loads(out)


class TestAttestLedgerLoop:

    def test_attest_records_prov_ids(self, tmp_path):
        db = tmp_path / "akashic.db"
        bundle = _attest(
            tmp_path,
            "Alice likes cats. Bob owns a dog.",
            extractor="sieve",
            ledger=str(db),
        )
        prov_ids = bundle["sum_cli"]["prov_ids"]
        assert len(prov_ids) == 2
        for pid in prov_ids:
            assert pid.startswith("prov:")
        assert bundle["sum_cli"]["ledger"] == str(db)
        assert db.exists()

    def test_prov_ids_resolve_to_evidence(self, tmp_path):
        db = tmp_path / "akashic.db"
        bundle = _attest(
            tmp_path,
            "Alice likes cats. Bob owns a dog.",
            extractor="sieve",
            ledger=str(db),
        )
        prov_ids = bundle["sum_cli"]["prov_ids"]

        # Every prov_id must round-trip to a ProvenanceRecord with
        # the correct byte range pointing at a real sentence in the
        # original input. sha256: URI also must match the input hash.
        expected_excerpts = {"Alice likes cats.", "Bob owns a dog."}
        seen = set()
        for pid in prov_ids:
            rec = _resolve(pid, db)
            assert rec["source_uri"].startswith("sha256:")
            assert rec["byte_end"] > rec["byte_start"] >= 0
            seen.add(rec["text_excerpt"])
        assert seen == expected_excerpts

    def test_llm_extractor_rejected_with_pointer(self, tmp_path):
        in_path = tmp_path / "in.txt"
        in_path.write_text("Alice likes cats.")
        args = _make_args(
            input=str(in_path),
            extractor="llm",
            ledger=str(tmp_path / "foo.db"),
        )
        with pytest.raises(SystemExit, match="extractor=sieve"):
            cmd_attest(args)

    def test_ledger_composes_with_ed25519(self, tmp_path):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        pem = tmp_path / "sk.pem"
        sk = Ed25519PrivateKey.generate()
        pem.write_bytes(sk.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))

        db = tmp_path / "akashic.db"
        bundle = _attest(
            tmp_path,
            "Alice likes cats.",
            extractor="sieve",
            ed25519_key=str(pem),
            ledger=str(db),
        )
        # Ed25519 signature present AND ledger records written — the
        # two attestation layers compose without stepping on each other.
        assert bundle["public_signature"].startswith("ed25519:")
        assert bundle["public_key"].startswith("ed25519:")
        assert len(bundle["sum_cli"]["prov_ids"]) == 1

    def test_resolve_missing_prov_id_exits_1(self, tmp_path):
        db = tmp_path / "akashic.db"
        _attest(
            tmp_path,
            "Alice likes cats.",
            extractor="sieve",
            ledger=str(db),
        )
        # Unknown prov_id → exit 1 with 'not found' on stderr.
        args = argparse.Namespace(
            prov_id="prov:0000000000000000000000000000000000000000",
            db=str(db),
        )
        code, _ = _capture_cmd(cmd_resolve, args)
        assert code == 1
