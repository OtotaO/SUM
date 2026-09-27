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
    # Optional G3 revocation list. Present only on revoked_kid_*
    # fixtures; absent fixtures verify without revocation (default
    # behaviour, backwards-compat with v0.9.C).
    revoked_kids = fx.get("revoked_kids")

    if expected_outcome == "verify":
        result = verify_receipt(receipt, jwks, revoked_kids=revoked_kids)
        assert result.verified is True, f"{name}: expected verify, got {result}"
        assert result.kid == receipt["kid"]
    elif expected_outcome == "reject":
        with pytest.raises(VerifyError) as excinfo:
            verify_receipt(receipt, jwks, revoked_kids=revoked_kids)
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
    assert len(files) == 19, (
        f"expected 19 fixtures (15 v0.9.C + 3 G3 revocation + "
        f"1 G3 crypto-agility), found {len(files)}: "
        f"{[f.name for f in files]}"
    )


# ---------------------------------------------------------------------------
# Legacy revoked_kids parameter: accept the served document, fail closed
# on every other shape, and compare instants rather than strings.
# ---------------------------------------------------------------------------


def _active_revocation() -> tuple[dict, dict, list]:
    fx = json.loads((_fixtures_dir() / "revoked_kid_active.json").read_text())
    return fx["receipt"], fx["jwks"], fx["revoked_kids"]


def test_served_revoked_kids_document_is_honoured() -> None:
    """INCIDENT_RESPONSE tells relying parties to pass the fetched
    /.well-known/revoked-kids.json. Passing that document used to iterate
    its keys, skip them all, and VERIFY a revoked kid."""
    from sum_engine_internal.render_receipt.verifier import VerifyError, verify_receipt

    receipt, jwks, entries = _active_revocation()
    document = {"schema": "sum.revoked_kids.v1", "issued_at": "2026-09-26T00:00:00Z",
                "revoked": entries}
    with pytest.raises(VerifyError) as exc:
        verify_receipt(receipt, jwks, revoked_kids=document)
    assert exc.value.error_class == "revoked_kid"


@pytest.mark.parametrize("bad", [
    "sum-render-2026-04-27-1",
    5,
    [{}],
    [{"kid": 5, "effective_revocation_at": "2026-01-01T00:00:00Z"}],
    [{"kid": "", "effective_revocation_at": "2026-01-01T00:00:00Z"}],
    {"entries": []},
    {"schema": "sum.revoked_kids.v1"},
    {"schema": "some.other.list", "revoked": []},
    {"schema": None, "revoked": []},
    [None],
    ["sum-render-2026-04-27-1"],
])
def test_malformed_revoked_kids_fails_closed(bad) -> None:
    """A string, a number, an unrecognised object or a non-object entry
    must reject, never silently skip revocation (and never raise a raw
    TypeError)."""
    from sum_engine_internal.render_receipt.verifier import VerifyError, verify_receipt

    receipt, jwks, _ = _active_revocation()
    with pytest.raises(VerifyError) as exc:
        verify_receipt(receipt, jwks, revoked_kids=bad)
    assert exc.value.error_class == "revoked_kid"


def test_revocation_compares_instants_not_strings() -> None:
    """Signed 0.849 s AFTER a whole-second effective time must be revoked.
    A string compare put '...16.849Z' before '...16Z' and accepted it."""
    from sum_engine_internal.render_receipt.verifier import VerifyError, verify_receipt

    receipt, jwks, entries = _active_revocation()
    assert receipt["payload"]["signed_at"] == "2026-04-27T00:45:16.849Z"
    same_second = [{**entries[0], "effective_revocation_at": "2026-04-27T00:45:16Z"}]
    with pytest.raises(VerifyError) as exc:
        verify_receipt(receipt, jwks, revoked_kids=same_second)
    assert exc.value.error_class == "revoked_kid"
    later = [{**entries[0], "effective_revocation_at": "2026-04-27T00:45:17Z"}]
    assert verify_receipt(receipt, jwks, revoked_kids=later).payload


def test_transform_wrapper_honours_the_served_document() -> None:
    from sum_engine_internal.transform_receipt.verifier import (
        VerifyError as TransformVerifyError,
        verify_transform_receipt as verify_transform,
    )

    root = Path(__file__).resolve().parents[1]
    fx = json.loads((root / "fixtures" / "transform_receipts" / "positive_control.json").read_text())
    receipt, jwks = fx["receipt"], fx["jwks"]
    document = {"schema": "sum.revoked_kids.v1", "revoked": [{
        "kid": receipt["kid"], "effective_revocation_at": "2026-01-01T00:00:00Z",
        "reason": "compromise"}]}
    with pytest.raises(TransformVerifyError) as exc:
        verify_transform(receipt, jwks, revoked_kids=document)
    assert exc.value.error_class == "revoked_kid"


@pytest.mark.parametrize("empty", [[], {"revoked": []}, {"schema": "sum.revoked_kids.v1", "revoked": []}])
def test_empty_revocation_list_verifies(empty) -> None:
    """RENDER_RECEIPT_FORMAT §6.1: callers opt into fail-closed fetching by
    passing an empty {"revoked": []}; that must verify an unrevoked kid."""
    from sum_engine_internal.render_receipt.verifier import verify_receipt

    fx = json.loads((_fixtures_dir() / "positive_control.json").read_text())
    assert verify_receipt(fx["receipt"], fx["jwks"], revoked_kids=empty).payload


def test_instant_grammar_matches_the_shared_table() -> None:
    """Python and the browser verifier parse revocation instants with the
    same grammar and arithmetic; the table is also run by
    single_file_demo/test_render_receipt_verify.js."""
    from sum_engine_internal.render_receipt.verifier import _instant_ms

    root = Path(__file__).resolve().parents[1]
    table = json.loads((root / "Tests" / "fixtures" / "revocation_instants.json").read_text())
    mismatches = [(c["value"], c["expected_ms"], _instant_ms(c["value"]))
                  for c in table["cases"] if _instant_ms(c["value"]) != c["expected_ms"]]
    assert mismatches == []
