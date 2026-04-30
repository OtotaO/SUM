"""Fixture-based unit tests for scripts/verify_pypi_attestation.py.

The release script makes assumptions about PyPI's Integrity API JSON
shape and about Sigstore certificate SAN URIs. Each assumption is
worth pinning here so a future PyPI / Sigstore schema drift is
caught at PR time, not at release time.

The tests deliberately exercise only the JSON parsers + SAN logic;
the network paths (fetch_artifact_url, download, fetch_provenance,
run_pypi_attestations_verify) are out of scope — they compose, but
their happy paths are exercised by the live release pipeline. We're
defending here against a quieter failure: parsers that look like
they work but accept the wrong shape.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# Load scripts/verify_pypi_attestation.py without polluting Tests/
# import semantics; the script is not packaged.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_pypi_attestation.py"
_spec = importlib.util.spec_from_file_location("verify_pypi_attestation", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["verify_pypi_attestation"] = _mod
_spec.loader.exec_module(_mod)


# --------------------------------------------------------------------------
# find_publisher
# --------------------------------------------------------------------------


def test_find_publisher_at_top_level():
    """PyPI Integrity API documented shape: publisher at top level."""
    prov = {
        "publisher": {
            "claims": None,
            "environment": "",
            "kind": "GitHub",
            "repository": "OtotaO/SUM",
            "workflow": "publish-pypi.yml",
        }
    }
    pub = _mod.find_publisher(prov)
    assert pub is not None
    assert pub["repository"] == "OtotaO/SUM"
    assert pub["workflow"] == "publish-pypi.yml"


def test_find_publisher_nested_in_attestation_bundle():
    """Defensive: tolerates nested envelopes if PyPI shape evolves."""
    prov = {
        "attestation_bundles": [
            {
                "publisher": {
                    "kind": "GitHub",
                    "repository": "OtotaO/SUM",
                    "workflow": "publish-pypi.yml",
                    "environment": "testpypi",
                }
            }
        ]
    }
    pub = _mod.find_publisher(prov)
    assert pub is not None
    assert pub["environment"] == "testpypi"


def test_find_publisher_returns_none_when_absent():
    assert _mod.find_publisher({}) is None
    assert _mod.find_publisher({"foo": [{"bar": 1}]}) is None


# --------------------------------------------------------------------------
# check_publisher_coarse
# --------------------------------------------------------------------------


def test_publisher_basename_workflow_passes():
    """Documented PyPI shape: publisher.workflow is basename only."""
    publisher = {
        "kind": "GitHub",
        "repository": "OtotaO/SUM",
        "workflow": "publish-pypi.yml",
        "environment": "",
    }
    failures = _mod.check_publisher_coarse(
        publisher,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment=None,
    )
    assert failures == [], failures


def test_publisher_full_path_workflow_also_passes():
    """If a future schema upgrades to full path, the check still accepts."""
    publisher = {
        "kind": "GitHub",
        "repository": "OtotaO/SUM",
        "workflow": ".github/workflows/publish-pypi.yml",
    }
    failures = _mod.check_publisher_coarse(
        publisher,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment=None,
    )
    assert failures == []


def test_publisher_repo_url_form_also_passes():
    publisher = {
        "kind": "GitHub",
        "repository": "https://github.com/OtotaO/SUM",
        "workflow": "publish-pypi.yml",
    }
    failures = _mod.check_publisher_coarse(
        publisher,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment=None,
    )
    assert failures == []


def test_publisher_workflow_mismatch_fails():
    publisher = {
        "kind": "GitHub",
        "repository": "OtotaO/SUM",
        "workflow": "release.yml",  # wrong workflow
    }
    failures = _mod.check_publisher_coarse(
        publisher,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment=None,
    )
    assert any("publisher.workflow" in f for f in failures)


def test_publisher_repo_mismatch_fails():
    publisher = {
        "kind": "GitHub",
        "repository": "evil/SUM",
        "workflow": "publish-pypi.yml",
    }
    failures = _mod.check_publisher_coarse(
        publisher,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment=None,
    )
    assert any("publisher.repository" in f for f in failures)


def test_publisher_environment_mismatch_fails_when_specified():
    publisher = {
        "kind": "GitHub",
        "repository": "OtotaO/SUM",
        "workflow": "publish-pypi.yml",
        "environment": "pypi",  # wrong env for staging
    }
    failures = _mod.check_publisher_coarse(
        publisher,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment="testpypi",
    )
    assert any("publisher.environment" in f for f in failures)


def test_publisher_empty_environment_does_not_fail():
    """Trusted Publishing without environment is valid; empty string OK."""
    publisher = {
        "kind": "GitHub",
        "repository": "OtotaO/SUM",
        "workflow": "publish-pypi.yml",
        "environment": "",  # empty per documented shape
    }
    failures = _mod.check_publisher_coarse(
        publisher,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment="testpypi",
    )
    assert failures == []


def test_publisher_none_returns_advisory():
    failures = _mod.check_publisher_coarse(
        None,
        expected_repo="https://github.com/OtotaO/SUM",
        expected_workflow=".github/workflows/publish-pypi.yml",
        expected_environment=None,
    )
    assert any("no `publisher` object" in f for f in failures)


# --------------------------------------------------------------------------
# extract_raw_byte_strings
# --------------------------------------------------------------------------


def test_extract_raw_byte_strings_snake_case():
    """PyPI Integrity API uses snake_case (raw_bytes)."""
    prov = {
        "verification_material": {
            "certificate": {"raw_bytes": "AAAA"},
        }
    }
    assert _mod.extract_raw_byte_strings(prov) == ["AAAA"]


def test_extract_raw_byte_strings_camel_case():
    """Sigstore Bundle JSON uses camelCase (rawBytes)."""
    prov = {
        "verificationMaterial": {
            "x509CertificateChain": {
                "certificates": [{"rawBytes": "AAAA"}, {"rawBytes": "BBBB"}],
            }
        }
    }
    assert _mod.extract_raw_byte_strings(prov) == ["AAAA", "BBBB"]


def test_extract_raw_byte_strings_walks_lists_and_nested():
    prov = {
        "envelope": [
            {"item": {"verification_material": {"certificate": {"raw_bytes": "X"}}}},
            {"item": {"verificationMaterial": {"certificate": {"rawBytes": "Y"}}}},
        ]
    }
    out = _mod.extract_raw_byte_strings(prov)
    assert sorted(out) == ["X", "Y"]


def test_extract_raw_byte_strings_returns_empty_when_absent():
    assert _mod.extract_raw_byte_strings({}) == []
    assert _mod.extract_raw_byte_strings({"unrelated": "field"}) == []


def test_extract_raw_byte_strings_pypi_flattened_certificate_string():
    """Regression: PyPI Integrity API now serialises the leaf
    certificate as ``verification_material.certificate`` directly as a
    base64 string (not a sub-object with raw_bytes inside).

    Observed at https://test.pypi.org/integrity/sum-engine/0.4.0/.../provenance
    on 2026-04-30. The v0.4.0 publish attempt's pre-promotion gate
    failed because the older walker only matched ``rawBytes``-shaped
    keys, missed this flattened shape, and reported "no Sigstore
    certificates extractable" even though a valid cert chain was
    sitting at this exact path. The fix widens the walker to also
    collect string values at any ``certificate`` key; non-base64-DER
    candidates are skipped harmlessly by ``parse_certificates``.
    """
    prov = {
        "attestation_bundles": [{
            "attestations": [{
                "envelope": {"signature": "MEUCIQ...", "statement": "eyJfdHl..."},
                "verification_material": {
                    "certificate": "MIIGrTCCBjSgAwIBAgIU...",
                },
                "version": 1,
            }],
            "publisher": {"kind": "GitHub", "repository": "OtotaO/SUM"},
        }],
        "version": 1,
    }
    assert _mod.extract_raw_byte_strings(prov) == ["MIIGrTCCBjSgAwIBAgIU..."]


def test_extract_raw_byte_strings_co_existing_shapes_both_collected():
    """If a provenance carries both flattened (``certificate``: string)
    and traditional (``rawBytes``-shaped) certs in the same tree, both
    must be collected so a multi-cert chain still parses end-to-end.
    """
    prov = {
        "attestations": [
            {"verification_material": {"certificate": "FLATTENED"}},
            {"verification_material": {"certificate": {"raw_bytes": "OLD_SHAPE"}}},
        ]
    }
    assert sorted(_mod.extract_raw_byte_strings(prov)) == ["FLATTENED", "OLD_SHAPE"]


# --------------------------------------------------------------------------
# expected_san_prefix
# --------------------------------------------------------------------------


def test_expected_san_prefix_full_form():
    prefix = _mod.expected_san_prefix(
        "https://github.com/OtotaO/SUM",
        ".github/workflows/publish-pypi.yml",
        "refs/tags/v",
    )
    assert prefix == (
        "https://github.com/OtotaO/SUM/.github/workflows/publish-pypi.yml@refs/tags/v"
    )


def test_expected_san_prefix_strips_trailing_slash_on_repo():
    prefix = _mod.expected_san_prefix(
        "https://github.com/OtotaO/SUM/",
        ".github/workflows/publish-pypi.yml",
        "refs/tags/v",
    )
    assert prefix == (
        "https://github.com/OtotaO/SUM/.github/workflows/publish-pypi.yml@refs/tags/v"
    )


def test_expected_san_prefix_handles_owner_repo_form():
    """If --repository is passed as OWNER/REPO instead of full URL."""
    prefix = _mod.expected_san_prefix(
        "OtotaO/SUM",
        ".github/workflows/publish-pypi.yml",
        "refs/tags/v",
    )
    assert prefix == (
        "https://github.com/OtotaO/SUM/.github/workflows/publish-pypi.yml@refs/tags/v"
    )


# --------------------------------------------------------------------------
# assert_san_identity — failure modes (precise messages)
# --------------------------------------------------------------------------


def test_assert_san_identity_no_certs_returns_precise_message():
    failures, observed = _mod.assert_san_identity(
        certs=[],
        repository="https://github.com/OtotaO/SUM",
        workflow=".github/workflows/publish-pypi.yml",
        ref_prefix="refs/tags/v",
    )
    assert observed == []
    assert len(failures) == 1
    msg = failures[0]
    # The error must be self-describing — a verifier-shape limitation,
    # NOT an implication of artifact tampering.
    assert "do not promote" in msg
    assert "Verifier limitation" in msg


# --------------------------------------------------------------------------
# load_dist_hashes
# --------------------------------------------------------------------------


def test_load_dist_hashes_accepts_v1_schema(tmp_path):
    p = tmp_path / "dist_hashes.json"
    p.write_text(
        '{"schema":"sum.dist_hashes.v1","version":"0.3.1",'
        '"artifacts":[{"filename":"sum_engine-0.3.1-py3-none-any.whl",'
        '"sha256":"deadbeef","size_bytes":1}]}'
    )
    out = _mod.load_dist_hashes(p)
    assert out == {"sum_engine-0.3.1-py3-none-any.whl": "deadbeef"}


def test_load_dist_hashes_rejects_unknown_schema(tmp_path):
    p = tmp_path / "dist_hashes.json"
    p.write_text('{"schema":"some.other.schema.v1","artifacts":[]}')
    with pytest.raises(SystemExit) as exc:
        _mod.load_dist_hashes(p)
    assert "sum.dist_hashes.v1" in str(exc.value)
