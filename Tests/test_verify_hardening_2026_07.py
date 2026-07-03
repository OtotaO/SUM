"""Hardening regressions from the 2026-07-02 adversarial audit.

Pins three fixes:

1. `sum verify` cross-checks the unsigned `axioms` convenience mirror
   against canonical_tome. Before this, a bundle whose mirror was edited
   after attest still verified green (state integer + signature cover
   canonical_tome only) while `sum transform apply compose` consumed the
   edited mirror — verify green-lit one thing, compose ate another.
2. `sum verify` exits 2 (malformed/unreadable input), not 1 (tampered),
   on an unreadable input path. The 0/1/2 contract is documented; a CI
   user branching on exit codes must not read a missing file as tamper.
3. `sum_verify.SumVerifyError` is a genuine common base: every failure
   class the SDK raises derives from it, so `except SumVerifyError`
   catches any verification failure in one clause.

License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from sum_cli.main import cmd_attest, cmd_verify


def _run_attest(text: str, tmp_path: Path) -> dict:
    in_path = tmp_path / "in.txt"
    in_path.write_text(text)
    args = argparse.Namespace(
        input=str(in_path),
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="Audit Hardening",
        signing_key=None,
        ed25519_key=None,
        ledger=None,
        pretty=False,
        verbose=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        assert cmd_attest(args) == 0, f"attest failed: {buf.getvalue()}"
    finally:
        sys.stdout = old
    return json.loads(buf.getvalue())


def _run_verify(bundle_or_path, tmp_path: Path) -> int:
    if isinstance(bundle_or_path, dict):
        path = tmp_path / "bundle.json"
        path.write_text(json.dumps(bundle_or_path))
        path = str(path)
    else:
        path = bundle_or_path
    args = argparse.Namespace(
        input=path,
        signing_key=None,
        strict=False,
        pretty=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        return cmd_verify(args)
    finally:
        sys.stdout = old


_TEXT = "The sun is a star. Mars is a planet. Pluto is a dwarf planet."


# ── 1. axioms-mirror cross-check ─────────────────────────────────────


def test_untampered_mirror_verifies(tmp_path: Path) -> None:
    bundle = _run_attest(_TEXT, tmp_path)
    assert _run_verify(bundle, tmp_path) == 0


def test_tampered_mirror_subject_fails_verify(tmp_path: Path) -> None:
    bundle = _run_attest(_TEXT, tmp_path)
    assert bundle["axioms"], "attest must emit a non-empty mirror"
    bundle["axioms"][0]["subject"] += "X"
    assert _run_verify(bundle, tmp_path) == 1


def test_tampered_mirror_object_swap_same_count_fails_verify(tmp_path: Path) -> None:
    # Same axiom COUNT, different content — the exact shape of the
    # audit's repro (count checks alone would pass).
    bundle = _run_attest(_TEXT, tmp_path)
    bundle["axioms"][0]["object"] = "a black hole"
    assert _run_verify(bundle, tmp_path) == 1


def test_malformed_mirror_fails_verify(tmp_path: Path) -> None:
    bundle = _run_attest(_TEXT, tmp_path)
    bundle["axioms"] = ["not-a-dict"]
    assert _run_verify(bundle, tmp_path) == 1
    bundle["axioms"] = [{"subject": "only-a-subject"}]
    assert _run_verify(bundle, tmp_path) == 1


def test_absent_mirror_still_verifies(tmp_path: Path) -> None:
    # Pre-#251 bundles carry no mirror; they must keep verifying.
    bundle = _run_attest(_TEXT, tmp_path)
    del bundle["axioms"]
    assert _run_verify(bundle, tmp_path) == 0


# ── 2. exit-code contract on unreadable input ────────────────────────


def test_unreadable_input_exits_2_not_1(tmp_path: Path) -> None:
    assert _run_verify(str(tmp_path / "does-not-exist.json"), tmp_path) == 2


# ── 3. SumVerifyError common base ────────────────────────────────────


def test_every_sdk_failure_class_derives_from_base() -> None:
    import sum_verify

    for cls in (
        sum_verify.JoseEnvelopeError,
        sum_verify.ReceiptVerifyError,
        sum_verify.MeaningReceiptReplayError,
        sum_verify.MeaningReceiptDisclosureError,
        sum_verify.UnsupportedSchemaError,
    ):
        assert issubclass(cls, sum_verify.SumVerifyError), cls


def test_single_except_clause_catches_dispatch_failure() -> None:
    import sum_verify

    with pytest.raises(sum_verify.SumVerifyError):
        sum_verify.verify({"schema": "not.a.real.schema"}, {})


def test_single_except_clause_catches_crypto_failure() -> None:
    import sum_verify

    golden_dir = (
        Path(__file__).resolve().parents[1] / "fixtures" / "meaning_receipts_billsum"
    )
    receipt = json.loads(
        (golden_dir / "meaning_risk_receipt.billsum.golden.json").read_text()
    )
    jwks = json.loads((golden_dir / "jwks.json").read_text())
    # Tamper a signed field: the failure must surface as SumVerifyError.
    receipt["payload"]["risk_upper_bound_micro"] += 1
    with pytest.raises(sum_verify.SumVerifyError):
        sum_verify.verify(receipt, jwks)
