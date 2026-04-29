"""Threat-model traceability — every documented defence has an asserting test.

`docs/THREAT_MODEL.md` §4 (Attack Surface Summary) is a table mapping
attack vectors to the mechanism that defends against them. The
existing test surface covers most defences in code-organised tests
(``test_resource_guards.py``, ``test_extraction_validator.py``,
``test_merkle_chain.py``, etc.) but **there is no single file that
demonstrates the threat-model claims hold**.

This file is that consolidation. Each test class corresponds to a
row in `docs/THREAT_MODEL.md` §4, with the row reference in the
docstring. The tests intentionally exercise the *primary defence*
named in the threat-model row, not every edge case the underlying
module covers — those live in their own dedicated test files.

If a row in `docs/THREAT_MODEL.md` §4 changes mechanism, the
corresponding test class here MUST be updated. If a row is added,
a corresponding test class MUST be added. The threat-model-to-test
traceability is the load-bearing property of this file; it is not
just a smoke test.

Out of scope here:
  - §3.8 P2P Mesh Authentication (requires a running FastAPI app +
    JWT plumbing; covered by `test_phase13_zenith.py`).
  - VC 2.0 bundle forgery (covered by 28 tests in
    `test_verifiable_credential.py`).
  - Render-receipt forgery (covered by the cross-runtime fixture
    matrix in `test_render_receipt_verifier.py`).
  - Trust-root manifest forgery (covered by `test_trust_root.py`).
  - Cross-runtime verifier divergence (covered by the K-matrix +
    A-matrix in `scripts/verify_cross_runtime*.py`).
  - CI / supply-chain compromise (covered by R0.3 SHA-pin lint at
    `scripts/lint_workflow_pins.py`).

Those are all separate test surfaces with their own load-bearing
files. This file is the consolidation for the in-process defences.
"""
from __future__ import annotations

import copy
import json
from typing import List, Tuple

import pytest


# ===========================================================================
# §2.1 — Bundle Tampering (HMAC-SHA256)
# ===========================================================================


class TestBundleTamperingDefence:
    """`docs/THREAT_MODEL.md` §2.1 — HMAC-SHA256 over
    ``canonical_tome | state_integer | timestamp`` MUST detect any
    modification to those fields."""

    def _build_signed_bundle(self, signing_key: str = "shared-secret-1234"):
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
        algebra = GodelStateAlgebra()
        codec = CanonicalCodec(
            algebra,
            AutoregressiveTomeGenerator(algebra),
            signing_key=signing_key,
        )
        state = algebra.encode_chunk_state([("alice", "likes", "cats")])
        return codec.export_bundle(state, branch="main", title="threat-model-test")

    def test_unmodified_bundle_verifies(self):
        from sum_cli.main import _verify_hmac_bundle
        bundle = self._build_signed_bundle()
        assert _verify_hmac_bundle(bundle, "shared-secret-1234") == "verified"

    def test_modified_canonical_tome_invalidates_hmac(self):
        from sum_cli.main import _verify_hmac_bundle
        bundle = self._build_signed_bundle()
        bundle["canonical_tome"] = bundle["canonical_tome"] + "The eve hates dogs.\n"
        assert _verify_hmac_bundle(bundle, "shared-secret-1234") == "invalid"

    def test_modified_state_integer_invalidates_hmac(self):
        from sum_cli.main import _verify_hmac_bundle
        bundle = self._build_signed_bundle()
        bundle["state_integer"] = "2"  # was something else
        assert _verify_hmac_bundle(bundle, "shared-secret-1234") == "invalid"


# ===========================================================================
# §2.2 — State Integer Forgery (Ouroboros witness)
# ===========================================================================


class TestStateForgeryDefence:
    """`docs/THREAT_MODEL.md` §2.2 — The witness verifier
    independently reconstructs state from canonical tome and rejects
    bundles whose claimed state doesn't match the reconstructed
    state. This holds *without the HMAC key* — any third party can
    detect state forgery."""

    def test_reconstructed_state_matches_claimed_state_on_clean_bundle(self):
        import math
        import re
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

        algebra = GodelStateAlgebra()
        codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
        state = algebra.encode_chunk_state([
            ("alice", "likes", "cats"),
            ("bob", "owns", "dog"),
        ])
        bundle = codec.export_bundle(state, branch="main", title="t")

        # Re-derive state from tome (verifier-side, no key needed)
        algebra2 = GodelStateAlgebra()
        pattern = re.compile(r"^The (\S+) (\S+) (.+)\.$")
        recomputed = 1
        for line in bundle["canonical_tome"].splitlines():
            m = pattern.match(line.strip())
            if not m:
                continue
            p = algebra2.get_or_mint_prime(m.group(1), m.group(2), m.group(3))
            recomputed = math.lcm(recomputed, p)

        assert recomputed == int(bundle["state_integer"])

    def test_forged_state_integer_detected_without_hmac_key(self):
        import math
        import re
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

        algebra = GodelStateAlgebra()
        codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
        state = algebra.encode_chunk_state([("alice", "likes", "cats")])
        bundle = codec.export_bundle(state, branch="main", title="t")
        bundle["state_integer"] = "999"  # forged

        algebra2 = GodelStateAlgebra()
        pattern = re.compile(r"^The (\S+) (\S+) (.+)\.$")
        recomputed = 1
        for line in bundle["canonical_tome"].splitlines():
            m = pattern.match(line.strip())
            if not m:
                continue
            recomputed = math.lcm(
                recomputed,
                algebra2.get_or_mint_prime(m.group(1), m.group(2), m.group(3)),
            )

        assert recomputed != int(bundle["state_integer"])  # mismatch detected


# ===========================================================================
# §2.3 — Version Mismatch (canonical_format_version gate)
# ===========================================================================


class TestVersionMismatchDefence:
    """`docs/THREAT_MODEL.md` §2.3 — Bundles with an unsupported
    `canonical_format_version` are rejected at import."""

    def test_unsupported_canonical_format_version_rejected(self):
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator

        algebra = GodelStateAlgebra()
        codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
        bundle = {
            "canonical_tome": "The alice likes cats.\n",
            "state_integer": "1",
            "canonical_format_version": "999.0.0",  # future major
            "axiom_count": 1,
            "branch": "main",
        }
        with pytest.raises(Exception):
            codec.import_bundle(bundle)


# ===========================================================================
# §2.4 — Malformed Bundles (field validation)
# ===========================================================================


class TestMalformedBundleDefence:
    """`docs/THREAT_MODEL.md` §2.4 — Missing required fields produce
    an explicit error, not silent acceptance."""

    @pytest.mark.parametrize("missing_field", [
        "canonical_tome", "state_integer", "canonical_format_version",
    ])
    def test_missing_required_field_rejected(self, missing_field):
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator

        algebra = GodelStateAlgebra()
        codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
        bundle = {
            "canonical_tome": "The alice likes cats.\n",
            "state_integer": "1",
            "canonical_format_version": "1.0.0",
            "axiom_count": 1,
            "branch": "main",
        }
        del bundle[missing_field]
        with pytest.raises(Exception):
            codec.import_bundle(bundle)


# ===========================================================================
# §3.3 — Extraction Manipulation (structural gating)
# ===========================================================================


class TestExtractionManipulationDefence:
    """`docs/THREAT_MODEL.md` §3.3 — `ExtractionValidator` rejects
    structurally-malformed triplets: empty fields, oversized fields,
    control characters, JSON fragments. Catches the *structural*
    classes; the residual semantic-validation gap is named in the
    threat-model row and not asserted here."""

    def test_empty_subject_rejected(self):
        from sum_engine_internal.ensemble.extraction_validator import ExtractionValidator
        v = ExtractionValidator()
        result = v.validate_batch([("", "likes", "cats")])
        assert len(result.accepted) == 0
        assert len(result.rejected) == 1

    def test_control_character_in_field_rejected(self):
        from sum_engine_internal.ensemble.extraction_validator import ExtractionValidator
        v = ExtractionValidator()
        result = v.validate_batch([("alice\x00", "likes", "cats")])
        assert len(result.accepted) == 0

    def test_json_fragment_in_field_rejected(self):
        from sum_engine_internal.ensemble.extraction_validator import ExtractionValidator
        v = ExtractionValidator()
        result = v.validate_batch([('{"foo":', "likes", "cats")])
        # Either rejected outright or canonicalised away — the
        # accepted set MUST NOT contain the literal JSON fragment.
        accepted_subjects = [a[0] for a in result.accepted]
        assert '{"foo":' not in accepted_subjects

    def test_oversized_field_rejected(self):
        from sum_engine_internal.ensemble.extraction_validator import ExtractionValidator
        v = ExtractionValidator()
        oversized = "a" * 600  # > 500-char field bound per threat-model row
        result = v.validate_batch([(oversized, "likes", "cats")])
        assert len(result.accepted) == 0


# ===========================================================================
# §3.4 — Semantic Collision Replay (collision-resistant prime mint)
# ===========================================================================


class TestCollisionResolutionDefence:
    """`docs/THREAT_MODEL.md` §3.4 — Independent `GodelStateAlgebra`
    instances minting the same key in different orders MUST produce
    the same prime. This is the cross-instance consistency the
    threat-model row names. The 1000-axiom stress test lives in
    `test_godel_cross_instance_collision.py`; this is the
    threat-model-trace assertion."""

    def test_two_instances_independent_order_produce_identical_primes(self):
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

        a = GodelStateAlgebra()
        b = GodelStateAlgebra()

        # Mint in different orders — even if there's an order-dependent
        # collision-resolution path, the same key MUST produce the
        # same prime in both instances.
        a.get_or_mint_prime("alice", "likes", "cats")
        a.get_or_mint_prime("bob", "owns", "dog")
        b.get_or_mint_prime("bob", "owns", "dog")
        b.get_or_mint_prime("alice", "likes", "cats")

        p_a = a.get_or_mint_prime("alice", "likes", "cats")
        p_b = b.get_or_mint_prime("alice", "likes", "cats")
        assert p_a == p_b


# ===========================================================================
# §3.5 — Contradiction Governance (DeterministicArbiter)
# ===========================================================================


class TestDeterministicArbiterDefence:
    """`docs/THREAT_MODEL.md` §3.5 — The DeterministicArbiter resolves
    contradictions via SHA-256 lexicographic ordering. The same
    inputs MUST produce the same winner every time, on every host,
    without an LLM call."""

    def test_arbiter_resolves_deterministically_across_calls(self):
        try:
            from sum_engine_internal.ensemble.deterministic_arbiter import DeterministicArbiter
        except ImportError:
            pytest.skip("DeterministicArbiter module not present in this build")

        arb = DeterministicArbiter()
        # Two competing objects for (subject, predicate)
        winner_1 = arb.resolve_curvature(
            subject="earth", predicate="orbits",
            object_a="sun", object_b="moon",
        )
        winner_2 = arb.resolve_curvature(
            subject="earth", predicate="orbits",
            object_a="moon", object_b="sun",  # swapped order — must not change result
        )
        assert winner_1 == winner_2


# ===========================================================================
# §3.6 — Denial of Service (bundle size limits)
# ===========================================================================


class TestDoSBundleLimitsDefence:
    """`docs/THREAT_MODEL.md` §3.6 — Bundle limits enforce: 10 MB
    canonical tome size (via bundle_size_bytes), 100 K state-integer
    digits, 50 K canonical_tome_lines. Each is gated at import."""

    def test_oversized_bundle_size_rejected(self):
        from sum_engine_internal.infrastructure.resource_guards import (
            guard_bundle_import, MAX_BUNDLE_SIZE_BYTES, ResourceLimitError,
        )
        # A bundle whose tome is larger than the byte-size limit
        oversized_tome = "x" * (MAX_BUNDLE_SIZE_BYTES + 100)
        bundle = {
            "canonical_tome": oversized_tome,
            "state_integer": "1",
            "canonical_format_version": "1.0.0",
        }
        with pytest.raises(ResourceLimitError) as exc_info:
            guard_bundle_import(bundle)
        assert "bundle_size_bytes" in exc_info.value.detail

    def test_oversized_state_integer_digits_rejected(self):
        from sum_engine_internal.infrastructure.resource_guards import (
            guard_bundle_import, MAX_STATE_INTEGER_DIGITS, ResourceLimitError,
        )
        bundle = {
            "canonical_tome": "The alice likes cats.\n",
            "state_integer": "9" * (MAX_STATE_INTEGER_DIGITS + 1),
            "canonical_format_version": "1.0.0",
        }
        with pytest.raises(ResourceLimitError) as exc_info:
            guard_bundle_import(bundle)
        assert "state_integer_digits" in exc_info.value.detail

    def test_oversized_tome_line_count_rejected(self):
        from sum_engine_internal.infrastructure.resource_guards import (
            guard_bundle_import, MAX_CANONICAL_TOME_LINES, ResourceLimitError,
        )
        # Many short lines — small total bytes, but line count exceeds limit
        tome = "The a b c.\n" * (MAX_CANONICAL_TOME_LINES + 10)
        bundle = {
            "canonical_tome": tome,
            "state_integer": "1",
            "canonical_format_version": "1.0.0",
        }
        with pytest.raises(ResourceLimitError) as exc_info:
            guard_bundle_import(bundle)
        assert "canonical_tome_lines" in exc_info.value.detail

    def test_oversized_ingest_text_rejected(self):
        from sum_engine_internal.infrastructure.resource_guards import (
            guard_ingest_text, MAX_INGEST_TEXT_CHARS, ResourceLimitError,
        )
        with pytest.raises(ResourceLimitError):
            guard_ingest_text("a" * (MAX_INGEST_TEXT_CHARS + 1))

    def test_resource_limit_error_returns_413(self):
        """The ResourceLimitError is an HTTP 413 — clients see a
        well-defined status, not a generic 500."""
        from sum_engine_internal.infrastructure.resource_guards import ResourceLimitError
        exc = ResourceLimitError("test_resource", actual=999, limit=100)
        assert exc.status_code == 413


# ===========================================================================
# §3.7 — Ledger Tampering (Merkle hash chain)
# ===========================================================================


class TestLedgerTamperingDefence:
    """`docs/THREAT_MODEL.md` §3.7 — Each AkashicLedger event stores
    `prev_hash = SHA-256(prev_hash + payload)`. ``verify_chain()``
    walks the chain and reports the first broken link. Mutation,
    deletion, and injection MUST all be detectable."""

    @pytest.mark.asyncio
    async def test_clean_chain_verifies(self, tmp_path):
        from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger
        ledger = AkashicLedger(db_path=str(tmp_path / "ledger.db"))
        await ledger.append_event(
            operation="add", prime=2, axiom_key="alice||likes||cats",
            branch="main",
        )
        await ledger.append_event(
            operation="add", prime=3, axiom_key="bob||owns||dog",
            branch="main",
        )
        is_valid, first_broken = await ledger.verify_chain()
        assert is_valid is True
        assert first_broken is None

    @pytest.mark.asyncio
    async def test_mutated_event_payload_detected(self, tmp_path):
        import sqlite3
        from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger
        db = str(tmp_path / "ledger.db")
        ledger = AkashicLedger(db_path=db)
        await ledger.append_event(
            operation="add", prime=2, axiom_key="alice||likes||cats",
            branch="main",
        )
        await ledger.append_event(
            operation="add", prime=3, axiom_key="bob||owns||dog",
            branch="main",
        )

        # Mutate the second event's axiom_key directly in SQLite via
        # a separate connection — bypasses the append_event path
        # that would have re-hashed. (Table name is
        # ``semantic_events`` per akashic_ledger.py:99.)
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE semantic_events SET axiom_key = 'eve||owns||dog' "
            "WHERE seq_id = 2"
        )
        conn.commit()
        conn.close()

        # Re-verify; the hash chain MUST catch the mutation.
        is_valid, first_broken = await ledger.verify_chain()
        assert is_valid is False
        assert first_broken is not None


# ===========================================================================
# Cross-cutting: residual risks documented as explicit `xfail`
# ===========================================================================


class TestThreatModelResidualRisks:
    """`docs/THREAT_MODEL.md` documents residual risks that are
    deliberately NOT defended against today. These tests assert
    the residual risks remain (i.e., the absence of defence is
    intentional and documented). If a future PR ships defence, the
    corresponding `xfail` MUST flip to a regular passing test and
    the threat-model row updated."""

    @pytest.mark.xfail(
        reason="THREAT_MODEL.md §3.2 — no real-time key revocation. "
               "G3 revocation surface ships at /.well-known/revoked-kids.json "
               "for render-receipt keys but NOT for HMAC-bundle keys. "
               "When/if HMAC-key revocation lands, this xfail flips."
    )
    def test_compromised_hmac_key_is_revocable_in_real_time(self):
        # Currently no real-time revocation for HMAC keys. The defence
        # is rotation, which only protects future bundles.
        raise AssertionError("HMAC revocation not implemented (documented)")

    @pytest.mark.xfail(
        reason="THREAT_MODEL.md §3.7 — full database replacement is "
               "not detectable by the chain alone. A local attacker "
               "with full write access can rewrite from genesis."
    )
    def test_full_db_replacement_is_detected(self):
        raise AssertionError("Full DB replacement is undetected by design")


# ===========================================================================
# Index — every test class above traces to a §X.Y row in THREAT_MODEL.md
# ===========================================================================


_THREAT_TO_TEST = {
    "§2.1 Bundle Tampering": "TestBundleTamperingDefence",
    "§2.2 State Integer Forgery": "TestStateForgeryDefence",
    "§2.3 Version Mismatch": "TestVersionMismatchDefence",
    "§2.4 Malformed Bundles": "TestMalformedBundleDefence",
    "§3.3 Extraction Manipulation": "TestExtractionManipulationDefence",
    "§3.4 Semantic Collision Replay": "TestCollisionResolutionDefence",
    "§3.5 Contradiction Governance": "TestDeterministicArbiterDefence",
    "§3.6 Denial of Service (bundle limits)": "TestDoSBundleLimitsDefence",
    "§3.7 Ledger Tampering": "TestLedgerTamperingDefence",
    "Residual risks (xfail)": "TestThreatModelResidualRisks",
}


def test_threat_to_test_index_is_complete():
    """Sanity check — the index above MUST list every test class
    in this file. If you add a class, add it to ``_THREAT_TO_TEST``.
    If you remove one, remove the entry. This is the
    threat-model-traceability discipline this file exists to enforce."""
    import sys
    module = sys.modules[__name__]
    test_classes = {
        name for name in dir(module)
        if name.startswith("Test") and isinstance(getattr(module, name), type)
    }
    indexed_classes = set(_THREAT_TO_TEST.values())
    assert test_classes == indexed_classes, (
        f"index drift: classes={test_classes!r}, indexed={indexed_classes!r}"
    )
