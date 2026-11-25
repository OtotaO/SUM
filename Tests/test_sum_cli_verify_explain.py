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


def _attest(tmp_path, *flags) -> dict:
    from sum_cli.main import main

    out = io.StringIO()
    with patch("sys.stdout", out), patch("sys.stdin", io.StringIO("Alice graduated in 2012. Bob owns a dog.")):
        assert main(["attest", "--extractor=sieve", *flags]) == 0
    return json.loads(out.getvalue())


def test_explain_labels_unsigned_fields_as_the_bundles_claims(tmp_path):
    """The signature covers canonical_tome|state_integer|timestamp only.
    axiom_consistency_check and the sum_cli sidecar are unsigned and verify
    does not re-run them, so explain must not call them provable or
    certified, even when an intermediary has rewritten them."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = tmp_path / "k.pem"
    key.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    bundle = _attest(tmp_path, "--ed25519-key", str(key))
    bundle["axiom_consistency_check"] = {"consistent": True, "n_predicates_checked": 4242}
    bundle.setdefault("sum_cli", {})["extractor"] = "sieve"
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(bundle))
    rc, payload, _ = _run_explain(path)
    assert rc == 0 and payload["checks"]["cryptographic_integrity"]["status"] == "pass"
    for name in ("axiom_consistency", "extraction_provenance"):
        check = payload["checks"][name]
        assert check["epistemic_status"] == "not-asserted", name
        assert "verify does not re-run it" in check["detail"], name
    assert payload["checks"]["axiom_consistency"]["status"] == "advisory"
    assert "safe to cite the issuer" not in payload["recommended_action"]
    assert "unsigned" in payload["recommended_action"]


def test_explain_verified_matches_exit_code_for_unchecked_hmac(tmp_path):
    """An intact HMAC bundle verified without the key exits 0; its JSON
    verdict must agree, and must not be worse than the same bundle with
    the signature stripped."""
    bundle = _attest(tmp_path, "--signing-key", "s3cret")
    path = tmp_path / "hm.json"
    path.write_text(json.dumps(bundle))
    rc, payload, _ = _run_explain(path)
    assert rc == 0 and payload["verified"] is True
    assert payload["checks"]["cryptographic_integrity"]["status"] == "advisory"


@pytest.mark.parametrize("cmd,raw", [
    (["verify"], "42"),
    (["verify"], "null"),
    (["verify"], json.dumps("canonical_tome state_integer canonical_format_version")),
    (["verify"], json.dumps({"canonical_tome": 5, "state_integer": "1", "canonical_format_version": "1.0.0"})),
    (["verify"], json.dumps({"canonical_tome": "", "state_integer": 1, "canonical_format_version": "1.0.0"})),
    (["verify"], json.dumps({"canonical_tome": "", "state_integer": "1", "canonical_format_version": "1.0.0", "axiom_count": "1"})),
    (["inspect"], "[]"),
    (["inspect"], json.dumps({"canonical_tome": 5})),
    (["inspect"], json.dumps({"canonical_tome": "The a b c.", "state_integer": 5})),
    (["verify-meaning"], "[]"),
])
def test_wrong_json_types_are_malformed_input(tmp_path, cmd, raw):
    """Valid JSON of the wrong shape is malformed input (exit 2), not a
    verification failure (exit 1) or a crash."""
    from sum_cli.main import main

    path = tmp_path / "in.json"
    path.write_text(raw)
    if cmd == ["verify-meaning"]:
        jwks = tmp_path / "jwks.json"
        jwks.write_text('{"keys": []}')
        argv = [*cmd, str(path), "--jwks", str(jwks)]
    else:
        argv = [*cmd, "--input", str(path)]
    err = io.StringIO()
    with patch("sys.stdout", io.StringIO()), patch("sys.stderr", err):
        rc = main(argv)
    assert rc == 2, err.getvalue()
    assert "Error" not in err.getvalue()  # no Python exception name leaks


def test_explain_survives_a_wrong_type_unsigned_consistency_field(tmp_path):
    """axiom_consistency_check is unsigned. A list or string there must not
    turn --explain into a crash (exit 1, the tampered code) on a bundle plain
    verify accepts; it is ignored like an absent field."""
    for bad in (["x"], "yes"):
        bundle = _attest(tmp_path)
        bundle["axiom_consistency_check"] = bad
        path = tmp_path / "cc.json"
        path.write_text(json.dumps(bundle))
        rc, payload, err = _run_explain(path)
        assert rc == 0, err
        assert payload["checks"]["axiom_consistency"]["status"] == "absent"


def test_explain_always_lists_the_unsigned_extractor_gap(tmp_path):
    """The sidecar's extractor is unsigned, so rewriting llm -> sieve must not
    remove the extraction gap from known_gaps."""
    bundle = _attest(tmp_path)
    bundle.setdefault("sum_cli", {})["extractor"] = "sieve"
    path = tmp_path / "x.json"
    path.write_text(json.dumps(bundle))
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    assert any("unsigned sidecar" in g for g in payload["known_gaps"])


def test_explain_gap_without_a_sidecar_does_not_claim_one(tmp_path):
    bundle = _attest(tmp_path)
    bundle.pop("sum_cli", None)
    path = tmp_path / "nosidecar.json"
    path.write_text(json.dumps(bundle))
    rc, payload, _ = _run_explain(path)
    assert rc == 0
    assert "No extraction provenance in the bundle" in payload["known_gaps"]
    assert not any("unsigned sidecar" in g for g in payload["known_gaps"])


def test_verify_help_offers_no_positional_file(capsys):
    from sum_cli.main import main

    with pytest.raises(SystemExit):
        main(["verify", "--help"])
    out = capsys.readouterr().out
    assert "--input/arg" not in out and "(or --input FILE)" in out
