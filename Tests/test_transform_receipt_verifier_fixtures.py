"""Cross-runtime smoke test for the Python transform-receipt verifier.

Iterates the fixtures under ``fixtures/transform_receipts/`` and asserts
each ``expected_outcome`` + ``expected_error_class`` matches what
``sum_engine_internal.transform_receipt.verify_transform_receipt``
produces.

The same fixture set is consumed by the JS verifier in
``single_file_demo/test_transform_receipt_fixtures.js``. Cross-runtime
byte-identical outcomes is the K-style equivalence already proved for
CanonicalBundle and render receipts, extended to the transform
substrate.

Skipped if joserfc isn't available (the optional dep that ships under
``pip install sum-engine[receipt-verify]``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Skip generator inputs + outputs; only iterate the derived fixtures.
_SKIP_FILES = {
    "source_receipt.json",
    "jwks_at_capture.json",
}


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "transform_receipts"


def _fixture_files() -> list[Path]:
    return sorted(
        p
        for p in _fixtures_dir().iterdir()
        if p.suffix == ".json" and p.name not in _SKIP_FILES
    )


pytest.importorskip(
    "joserfc",
    reason="install sum-engine[receipt-verify] to run the transform-receipt verifier",
)


@pytest.mark.parametrize(
    "fixture_path",
    _fixture_files(),
    ids=lambda p: p.stem,
)
def test_fixture(fixture_path: Path) -> None:
    from sum_engine_internal.transform_receipt import (
        VerifyError,
        verify_transform_receipt,
    )

    fx = json.loads(fixture_path.read_text())
    name = fx["name"]
    receipt = fx["receipt"]
    jwks = fx["jwks"]
    expected_outcome = fx["expected_outcome"]
    expected_error_class = fx["expected_error_class"]

    if expected_outcome == "verify":
        result = verify_transform_receipt(receipt, jwks)
        assert result.verified is True, f"{name}: expected verify, got {result}"
        assert result.kid == receipt["kid"]
    elif expected_outcome == "reject":
        with pytest.raises(VerifyError) as exc_info:
            verify_transform_receipt(receipt, jwks)
        assert exc_info.value.error_class == expected_error_class, (
            f"{name}: expected error_class={expected_error_class!r}, "
            f"got {exc_info.value.error_class!r} ({exc_info.value})"
        )
    else:
        raise AssertionError(
            f"{name}: unknown expected_outcome {expected_outcome!r}"
        )
