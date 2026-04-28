"""Cross-runtime smoke test for the v0.9.C Python receipt verifier.

Iterates the fixtures under ``fixtures/render_receipts/`` and asserts
each ``expected_outcome`` + ``expected_error_class`` matches what
``sum_engine_internal.render_receipt.verify_receipt`` produces.

The exact same fixture set is consumed by the JS verifier in
``single_file_demo/test_render_receipt_verify.js``. Cross-runtime
byte-identical outcomes is the K-style equivalence we already have
for CanonicalBundle, applied to render receipts. Once both runtimes
pass on every push (this test in CI + the JS smoke as a step in
``vendor-byte-equivalence``), PROOF_BOUNDARY §1.8 upgrades from
"negative path exercised in worker-local TS tests but not yet
locked across runtimes" to "proved on adversarial inputs across
runtimes."

Skipped if joserfc isn't available (the optional dep that v0.9.C
adds via ``pip install sum-engine[receipt-verify]``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Skip the inputs the generator consumes; only iterate generated
# fixtures + the positive control.
_SKIP_FILES = {"source_render.json", "jwks_at_capture.json"}


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "render_receipts"


def _fixture_files() -> list[Path]:
    return sorted(
        p
        for p in _fixtures_dir().iterdir()
        if p.suffix == ".json" and p.name not in _SKIP_FILES
    )


# Skip the whole module if joserfc isn't installed.
joserfc = pytest.importorskip(
    "joserfc",
    reason="install sum-engine[receipt-verify] to run the v0.9.C verifier",
)


@pytest.mark.parametrize(
    "fixture_path",
    _fixture_files(),
    ids=lambda p: p.stem,
)
def test_fixture(fixture_path: Path) -> None:
    """Each fixture asserts its own expected outcome + error class.

    The parameterisation runs every fixture as its own test case so
    pytest output names the failing fixture directly.
    """
    from sum_engine_internal.render_receipt import VerifyError, verify_receipt

    fx = json.loads(fixture_path.read_text())
    name = fx["name"]
    receipt = fx["receipt"]
    jwks = fx["jwks"]
    expected_outcome = fx["expected_outcome"]
    expected_error_class = fx["expected_error_class"]

    if expected_outcome == "verify":
        result = verify_receipt(receipt, jwks)
        assert result.verified is True, f"{name}: expected verify, got {result}"
        assert result.kid == receipt["kid"]
    elif expected_outcome == "reject":
        with pytest.raises(VerifyError) as excinfo:
            verify_receipt(receipt, jwks)
        actual = excinfo.value.error_class
        assert actual == expected_error_class, (
            f"{name}: expected error_class={expected_error_class!r}, "
            f"got {actual!r} (message: {excinfo.value})"
        )
    else:  # pragma: no cover — author error in a fixture
        pytest.fail(f"{name}: unknown expected_outcome {expected_outcome!r}")


def test_all_fixtures_iterate() -> None:
    """Sanity check that the parametrize at module load found all
    15 fixtures. If a fixture file is added or removed and this
    count drifts, that's worth knowing — the cross-runtime contract
    is that BOTH JS and Python run the SAME N fixtures."""
    files = _fixture_files()
    assert len(files) == 15, (
        f"expected 15 fixtures, found {len(files)}: "
        f"{[f.name for f in files]}"
    )
