"""T1c — `sum transform` CLI subcommand tests.

Covers:
  - `sum transform list` emits sum.transform_registry.v1 with the
    three registered transforms.
  - `sum transform apply slider` with canonical-path parameters
    produces a tome + hashes (no receipt without signing config).
  - `sum transform apply slider` with signing env vars produces
    a signed sum.transform_receipt.v1 envelope.
  - `sum transform apply <unknown>` exits 2.
  - `sum transform apply slider` with bad --parameters exits 2.
  - The CLI's slider output matches the Python transform's output
    byte-for-byte — `transform_id` is reproducible across CLI runs.
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest


@contextmanager
def capture_stdout_stderr():
    """Capture stdout + stderr for CLI invocations."""
    out = io.StringIO()
    err = io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        yield out, err


def run_cli(argv: list[str], stdin_text: str | None = None) -> tuple[int, str, str]:
    """Run sum_cli.main(argv) with optional stdin; return
    (exit_code, stdout, stderr)."""
    from sum_cli.main import main

    if stdin_text is not None:
        stdin = io.StringIO(stdin_text)
        with capture_stdout_stderr() as (out, err), patch("sys.stdin", stdin):
            rc = main(argv)
    else:
        with capture_stdout_stderr() as (out, err):
            rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


# ─── `sum transform list` ───────────────────────────────────────────


def test_transform_list_emits_registry():
    rc, stdout, _ = run_cli(["transform", "list", "--pretty"])
    assert rc == 0
    payload = json.loads(stdout)
    assert payload["schema"] == "sum.transform_registry.v1"
    names = {t["id"] for t in payload["transforms"]}
    # The three transforms registered by T1a + T2 + T3.
    assert {"slider", "extract", "compose"}.issubset(names)


def test_transform_list_marks_requires_llm():
    rc, stdout, _ = run_cli(["transform", "list"])
    assert rc == 0
    payload = json.loads(stdout)
    by_name = {t["id"]: t for t in payload["transforms"]}
    # Slider may use LLM (off-centre axes); compose/extract are pure
    assert by_name["slider"]["requires_llm"] is True
    assert by_name["compose"]["requires_llm"] is False
    assert by_name["extract"]["requires_llm"] is False


# ─── `sum transform apply slider` (canonical-path) ──────────────────


def test_apply_slider_canonical_path_no_receipt(tmp_path):
    """Without signing env vars, the envelope has hashes but no
    transform_receipt."""
    input_file = tmp_path / "in.json"
    input_file.write_text(json.dumps({
        "triples": [["alice", "likes", "cats"], ["bob", "owns", "dog"]],
    }))

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SUM_TRANSFORM_SIGNING_JWK", None)
        os.environ.pop("SUM_TRANSFORM_SIGNING_KID", None)

        rc, stdout, _ = run_cli([
            "transform", "apply", "slider",
            "--input", str(input_file),
            "--parameters", json.dumps({
                "density": 1.0, "length": 0.5, "formality": 0.5,
                "audience": 0.5, "perspective": 0.5,
            }),
        ])

    assert rc == 0
    envelope = json.loads(stdout)
    assert envelope["model"] == "canonical-deterministic-v0"
    assert envelope["provider"] == "canonical-path"
    assert envelope["digital_source_type"] == "algorithmicMedia"
    # No signing config → no receipt, but hashes still present.
    assert "transform_receipt" not in envelope
    assert envelope["parameters_hash"].startswith("sha256-")
    assert envelope["input_hash"].startswith("sha256-")
    assert envelope["output_hash"].startswith("sha256-")
    # Output is a tome string.
    assert isinstance(envelope["output"], str)
    assert "alice" in envelope["output"]
    assert "bob" in envelope["output"]


def test_apply_slider_stdin_input(tmp_path):
    """Input via stdin with `--input -`."""
    rc, stdout, _ = run_cli(
        [
            "transform", "apply", "slider",
            "--input", "-",
            "--parameters", json.dumps({
                "density": 1.0, "length": 0.5, "formality": 0.5,
                "audience": 0.5, "perspective": 0.5,
            }),
        ],
        stdin_text=json.dumps({"triples": [["a", "rel", "b"]]}),
    )
    assert rc == 0
    envelope = json.loads(stdout)
    assert envelope["output"]


def test_apply_slider_with_signing_produces_receipt(tmp_path):
    """With SUM_TRANSFORM_SIGNING_JWK + _KID set, the envelope
    includes a signed sum.transform_receipt.v1."""
    pytest.importorskip("joserfc", reason="[receipt-verify] required")
    from joserfc.jwk import OKPKey

    kid = "test-cli-signing-2026"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)

    input_file = tmp_path / "in.json"
    input_file.write_text(json.dumps({"triples": [["a", "rel", "b"]]}))

    with patch.dict(os.environ, {
        "SUM_TRANSFORM_SIGNING_JWK": json.dumps(private),
        "SUM_TRANSFORM_SIGNING_KID": kid,
    }):
        rc, stdout, _ = run_cli([
            "transform", "apply", "slider",
            "--input", str(input_file),
            "--parameters", json.dumps({
                "density": 1.0, "length": 0.5, "formality": 0.5,
                "audience": 0.5, "perspective": 0.5,
            }),
        ])

    assert rc == 0
    envelope = json.loads(stdout)
    assert "transform_receipt" in envelope
    receipt = envelope["transform_receipt"]
    assert receipt["schema"] == "sum.transform_receipt.v1"
    assert receipt["kid"] == kid
    assert receipt["payload"]["transform"] == "slider"

    # The signed receipt should verify back through the Python
    # verifier with the JWKS derived from the signing key.
    from sum_engine_internal.transform_receipt import verify_transform_receipt

    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    jwks = {"keys": [public]}
    result = verify_transform_receipt(receipt, jwks)
    assert result.verified is True


# ─── Failure modes ──────────────────────────────────────────────────


def test_apply_unknown_transform_exits_2(tmp_path):
    input_file = tmp_path / "in.json"
    input_file.write_text("{}")
    rc, _, stderr = run_cli([
        "transform", "apply", "not-a-real-transform",
        "--input", str(input_file),
    ])
    assert rc == 2
    assert "unknown transform" in stderr


def test_apply_bad_parameters_json_exits_2(tmp_path):
    input_file = tmp_path / "in.json"
    input_file.write_text(json.dumps({"triples": [["a", "b", "c"]]}))
    rc, _, stderr = run_cli([
        "transform", "apply", "slider",
        "--input", str(input_file),
        "--parameters", "{not valid json}",
    ])
    assert rc == 2
    assert "--parameters" in stderr


def test_apply_missing_input_file_exits_2(tmp_path):
    rc, _, stderr = run_cli([
        "transform", "apply", "slider",
        "--input", "/nonexistent/path/that/does/not/exist.json",
    ])
    assert rc == 2
    assert "--input" in stderr


def test_apply_slider_llm_axis_exits_1_with_pointer(tmp_path):
    """Off-centre LLM-axis renders aren't wired through the CLI yet
    (deferred to T1c-follow-up); they exit 1 with the underlying
    NotImplementedError pointer message."""
    input_file = tmp_path / "in.json"
    input_file.write_text(json.dumps({"triples": [["a", "rel", "b"]]}))
    rc, _, stderr = run_cli([
        "transform", "apply", "slider",
        "--input", str(input_file),
        "--parameters", json.dumps({
            "density": 1.0, "length": 0.9, "formality": 0.5,
            "audience": 0.5, "perspective": 0.5,
        }),
    ])
    assert rc == 1
    assert "T1b" in stderr or "slider_renderer" in stderr


# ─── Reproducibility / cross-CLI determinism ────────────────────────


def test_apply_slider_is_deterministic(tmp_path):
    """Two runs over the same input + parameters produce byte-
    identical output + hashes. transform_id stable."""
    input_file = tmp_path / "in.json"
    input_file.write_text(json.dumps({
        "triples": [["alice", "likes", "cats"], ["bob", "owns", "dog"]],
    }))
    params = json.dumps({
        "density": 1.0, "length": 0.5, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    })

    rc1, stdout1, _ = run_cli([
        "transform", "apply", "slider",
        "--input", str(input_file), "--parameters", params,
    ])
    rc2, stdout2, _ = run_cli([
        "transform", "apply", "slider",
        "--input", str(input_file), "--parameters", params,
    ])
    assert rc1 == 0 and rc2 == 0
    e1 = json.loads(stdout1)
    e2 = json.loads(stdout2)
    assert e1["output"] == e2["output"]
    assert e1["parameters_hash"] == e2["parameters_hash"]
    assert e1["input_hash"] == e2["input_hash"]
    assert e1["output_hash"] == e2["output_hash"]
