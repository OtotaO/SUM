"""`sum verify --explain` — layered per-dimension verification report.

Productizes the proof-boundary discipline as user-visible output.
Each verification dimension carries a status + detail + epistemic-
status tag per docs/PROOF_BOUNDARY.md §5. Design lives in
docs/ZENITH_FRAMING_2026-05-16.md §5.

This is the v1 of the surface the zenith framing flags as the
operational realization of the proof-boundary discipline.
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest


@pytest.fixture
def signed_bundle(tmp_path):
    """Produce a real attested + signed bundle by running sum attest."""
    from sum_cli.main import main

    out = io.StringIO()
    in_text = "Alice graduated in 2012. Bob owns a dog."
    with patch("sys.stdout", out), patch("sys.stdin", io.StringIO(in_text)):
        rc = main(["attest", "--extractor=sieve"])
    assert rc == 0
    bundle = json.loads(out.getvalue())
    path = tmp_path / "b.json"
    path.write_text(json.dumps(bundle))
    return bundle, path


def _run_explain(input_path) -> tuple[int, dict, str]:
    """Run `sum verify --explain --input <path>` and return rc + parsed
    stdout JSON + stderr."""
    from sum_cli.main import main

    out, err = io.StringIO(), io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        rc = main(["verify", "--input", str(input_path), "--explain"])
    if rc == 0:
        return rc, json.loads(out.getvalue()), err.getvalue()
    return rc, {}, err.getvalue()


def test_explain_schema(signed_bundle):
    """The explain output carries schema sum.verify_explained.v1."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    assert payload["schema"] == "sum.verify_explained.v1"


def test_explain_has_seven_checks(signed_bundle):
    """All seven destination-design dimensions are present."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    checks = payload["checks"]
    expected = {
        "cryptographic_integrity",
        "canonical_reconstruction",
        "axiom_consistency",
        "extraction_provenance",
        "source_evidence_coverage",
        "semantic_preservation",
        "truth_of_content",
    }
    assert set(checks.keys()) == expected


def test_explain_each_check_has_three_fields(signed_bundle):
    """Each check has status + detail + epistemic_status. Locks the
    contract so downstream consumers can branch on epistemic_status
    safely (no missing fields)."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    for name, check in payload["checks"].items():
        assert "status" in check, f"{name}: missing status"
        assert "detail" in check, f"{name}: missing detail"
        assert "epistemic_status" in check, f"{name}: missing epistemic_status"
        assert isinstance(check["detail"], str) and check["detail"]


def test_explain_epistemic_status_values(signed_bundle):
    """Epistemic_status values are drawn from the proof-boundary §5
    taxonomy. Locks the alignment with PROOF_BOUNDARY.md."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    allowed = {"provable", "certified", "empirical-benchmark", "not-asserted"}
    for name, check in payload["checks"].items():
        assert check["epistemic_status"] in allowed, (
            f"{name}: epistemic_status={check['epistemic_status']!r} "
            f"not in {allowed}"
        )


def test_explain_truth_is_always_not_asserted(signed_bundle):
    """Truth-of-content MUST always be not-asserted. This is THE point
    of the proof-boundary discipline; verify must never claim truth."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    truth = payload["checks"]["truth_of_content"]
    assert truth["status"] == "not_asserted"
    assert truth["epistemic_status"] == "not-asserted"


def test_explain_cryptographic_check_is_provable(signed_bundle):
    """Cryptographic integrity is provable per PROOF_BOUNDARY §1."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    assert payload["checks"]["cryptographic_integrity"]["epistemic_status"] == "provable"


def test_explain_semantic_preservation_is_empirical_benchmark(signed_bundle):
    """Semantic preservation is empirical-benchmark per
    SLIDER_CONTRACT + the bench-hardening worktrail. Verify does NOT
    measure it; the surface is forward-compat."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    sp = payload["checks"]["semantic_preservation"]
    assert sp["status"] == "not_measured"
    assert sp["epistemic_status"] == "empirical-benchmark"


def test_explain_known_gaps_present(signed_bundle):
    """Known-gaps list surfaces what this verifier specifically does
    NOT prove. Always non-empty (at minimum: truth, source coverage,
    semantic preservation)."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    gaps = payload["known_gaps"]
    assert isinstance(gaps, list)
    assert len(gaps) >= 3
    assert any("truth" in g.lower() for g in gaps)


def test_explain_recommended_action_present(signed_bundle):
    """Recommended action is a non-empty string that names what's
    safe vs not. Consumers branch on this when serving back to humans."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    rec = payload["recommended_action"]
    assert isinstance(rec, str) and rec
    # The recommendation should orient the consumer ("safe to X /
    # NOT safe to Y / treat as advisory / review the per-check
    # details"); we just want non-empty, action-oriented text here.
    lower = rec.lower()
    assert any(k in lower for k in ("safe", "review", "advisory", "treat"))


def test_explain_returns_verified_true_on_valid_bundle(signed_bundle):
    """Top-level `verified` reflects the overall result."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    assert payload["verified"] is True


def test_explain_extraction_verifiable_for_sieve(signed_bundle):
    """Sieve extractor → extraction_provenance.status == 'verifiable'
    (deterministic re-extraction)."""
    _, path = signed_bundle
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    ext = payload["checks"]["extraction_provenance"]
    assert ext["status"] == "verifiable"
    assert "sieve" in ext["detail"]


def test_non_explain_path_unchanged(signed_bundle):
    """Sanity guard: without --explain, the existing summary output is
    unchanged. Locks that --explain is purely additive."""
    _, path = signed_bundle
    from sum_cli.main import main

    out = io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", io.StringIO()):
        rc = main(["verify", "--input", str(path)])
    assert rc == 0
    payload = json.loads(out.getvalue())
    # Old shape: {"ok": True, "axioms": ..., "state_integer_digits": ...}
    assert payload["ok"] is True
    assert "schema" not in payload  # not the explain shape
    assert "checks" not in payload
