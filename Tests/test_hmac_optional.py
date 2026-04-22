"""HMAC-optional CanonicalCodec tests.

The codec used to hard-require a shared secret and mandate an HMAC
"signature" field on every import. That coupled two unrelated concerns:
structural integrity and tamper-detection between trusted peers. For
public-domain transport (did:web / did:key bundles hosted on
Cloudflare Pages, verified by arbitrary third-party verifiers), the
shared-secret model is actively misleading — there is no shared
secret, and the old placeholder "sum-default-key" was cryptographic
theater.

These tests pin the new contract:

  * ``signing_key=None`` → codec is HMAC-disabled. Export emits no
    "signature" field. Import accepts bundles with or without one
    (signature ignored either way).
  * ``signing_key`` set → codec is HMAC-required. Import refuses any
    bundle missing the signature field (downgrade-protection).
  * Ed25519 path is independent of HMAC — a codec with no signing_key
    but a KeyManager still signs and verifies Ed25519.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import copy
import math

import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from sum_engine_internal.infrastructure.canonical_codec import (
    CanonicalCodec,
    InvalidSignatureError,
)
from sum_engine_internal.infrastructure.key_manager import KeyManager


# ─── Fixtures ──────────────────────────────────────────────────────

def _build_state(algebra: GodelStateAlgebra) -> int:
    state = 1
    for prime in algebra.axiom_to_prime.values():
        state = math.lcm(state, prime)
    return state


def _mint_axioms(algebra: GodelStateAlgebra) -> None:
    for s, p, o in [("alice", "likes", "cats"), ("bob", "knows", "python")]:
        algebra.get_or_mint_prime(s, p, o)


@pytest.fixture
def unsigned_codec():
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, tome_gen)  # no signing_key
    _mint_axioms(algebra)
    return codec, algebra


@pytest.fixture
def signed_codec():
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(
        algebra, tome_gen, signing_key="hmac-test-key-32bytes!!!!!!!!!!"
    )
    _mint_axioms(algebra)
    return codec, algebra


# ─── 1. Export behavior ───────────────────────────────────────────

class TestExportHmacOmitted:

    def test_no_signing_key_omits_signature_field(self, unsigned_codec):
        codec, algebra = unsigned_codec
        bundle = codec.export_bundle(_build_state(algebra))
        assert "signature" not in bundle, (
            "HMAC-disabled codec must not emit a signature field; emitting "
            "one would create a parsing ambiguity and a false sense of "
            "security (nothing verifies it)."
        )

    def test_signing_key_set_still_emits_signature(self, signed_codec):
        codec, algebra = signed_codec
        bundle = codec.export_bundle(_build_state(algebra))
        assert bundle["signature"].startswith("hmac-sha256:")


# ─── 2. Import round-trip without HMAC ────────────────────────────

class TestImportWithoutHmac:

    def test_unsigned_roundtrip(self, unsigned_codec):
        """Export with no HMAC → import with no HMAC → state preserved."""
        codec, algebra = unsigned_codec
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)
        assert codec.import_bundle(bundle) == state

    def test_unsigned_codec_ignores_stray_signature(self, unsigned_codec, signed_codec):
        """An HMAC-disabled importer does not verify a signature that
        happens to be present — it ignores the field completely.

        This is the inverse of downgrade protection: opting out of HMAC
        means opting out of its failure modes too. A bundle with a
        bogus signature, when imported by an unsigned codec, must not
        cause rejection purely on signature grounds.
        """
        codec_unsigned, alg_unsigned = unsigned_codec
        codec_signed, alg_signed = signed_codec
        state = _build_state(alg_signed)
        bundle = codec_signed.export_bundle(state)
        # Replace with garbage signature — unsigned importer should not care.
        bundle["signature"] = "hmac-sha256:" + "0" * 64
        # Note: alg_unsigned has separate prime assignments; the imported
        # state integer may not factor there, but import_bundle just returns
        # the claimed state. We check return value, not factorization.
        imported = codec_unsigned.import_bundle(bundle)
        assert imported == state


# ─── 3. Downgrade protection ──────────────────────────────────────

class TestDowngradeProtection:

    def test_signed_codec_rejects_bundle_missing_signature(self, signed_codec):
        """An importer that expects HMAC must refuse a bundle that has
        none, even if every other field is present. Silent acceptance
        would let an attacker strip tamper-detection."""
        codec, algebra = signed_codec
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)
        stripped = copy.deepcopy(bundle)
        del stripped["signature"]
        with pytest.raises(ValueError, match="missing required fields"):
            codec.import_bundle(stripped)

    def test_signed_codec_rejects_wrong_signature(self, signed_codec):
        codec, algebra = signed_codec
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)
        bundle["signature"] = "hmac-sha256:" + "0" * 64
        with pytest.raises(InvalidSignatureError):
            codec.import_bundle(bundle)


# ─── 4. Ed25519 remains independent ───────────────────────────────

class TestEd25519WithoutHmac:

    def test_unsigned_codec_with_keymanager_signs_ed25519(self, tmp_path):
        km = KeyManager(key_dir=str(tmp_path / "keys"))
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        codec = CanonicalCodec(algebra, tome_gen, key_manager=km)  # no HMAC
        _mint_axioms(algebra)
        state = _build_state(algebra)
        bundle = codec.export_bundle(state)

        assert "signature" not in bundle
        assert bundle["public_signature"].startswith("ed25519:")
        assert bundle["public_key"].startswith("ed25519:")
        # Round-trip verifies Ed25519 even though HMAC is absent.
        assert codec.import_bundle(bundle) == state

    def test_unsigned_codec_rejects_tampered_ed25519(self, tmp_path):
        km = KeyManager(key_dir=str(tmp_path / "keys"))
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        codec = CanonicalCodec(algebra, tome_gen, key_manager=km)
        _mint_axioms(algebra)
        bundle = codec.export_bundle(_build_state(algebra))

        tampered = copy.deepcopy(bundle)
        tampered["canonical_tome"] += "\nThe evil injected axiom."
        with pytest.raises(InvalidSignatureError):
            codec.import_bundle(tampered)
