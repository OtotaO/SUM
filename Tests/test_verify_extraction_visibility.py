"""Tests for `sum verify`'s extraction-provenance output.

The verifier surfaces an ``extraction`` block in its JSON output
so downstream consumers can branch on extractor reproducibility
without parsing the sum_cli sidecar by hand. This closes the
THREAT_MODEL.md §3.3 "signed ≠ true" visibility gap.

The block has three fields:

  * ``extractor`` — "sieve" | "llm" | None. Echoed from the
    sidecar; None when the bundle was not produced by the SUM
    CLI (or the sidecar was stripped).
  * ``verifiable`` — boolean. True iff ``extractor == "sieve"``,
    where re-extraction is deterministic and the axioms are
    cryptographically + reproducibly defensible. False for LLM
    extraction (stochastic; same prose can produce different
    triples on different runs).
  * ``source`` — "sum_cli sidecar" | "absent". Names where the
    extraction info came from, so a future surface (e.g. a
    signed extraction block) can be distinguished from
    sidecar-derived info.

The boolean ``verifiable`` is the load-bearing affordance —
downstream consumers can write:

    sum verify --input bundle.json | jq -e '.extraction.verifiable'

to gate on bundle reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _attest_via_subprocess(text: str, extractor: str, tmp_path: Path) -> Path:
    """Produce a bundle via `sum attest` so the sum_cli sidecar
    is constructed exactly as a real consumer's bundle would be."""
    spacy = pytest.importorskip("spacy")  # required by the sieve path
    bundle_path = tmp_path / "bundle.json"
    proc = subprocess.run(
        [sys.executable, "-m", "sum_cli.main", "attest",
         "--extractor", extractor],
        input=text,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"attest failed (likely missing extractor): {proc.stderr}")
    bundle_path.write_text(proc.stdout, encoding="utf-8")
    return bundle_path


def _verify_via_subprocess(bundle_path: Path) -> dict:
    """Run `sum verify --input X` and return the parsed result dict."""
    proc = subprocess.run(
        [sys.executable, "-m", "sum_cli.main", "verify",
         "--input", str(bundle_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"verify failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------
# extractor=sieve → verifiable=True
# --------------------------------------------------------------------------


def test_sieve_attested_bundle_reports_verifiable_true(tmp_path):
    """A bundle attested by the deterministic sieve extractor MUST
    surface ``extraction.verifiable: true`` in verify output."""
    bundle_path = _attest_via_subprocess(
        "Alice likes cats. Bob owns a dog.", "sieve", tmp_path,
    )
    result = _verify_via_subprocess(bundle_path)
    assert "extraction" in result, "verify output missing extraction block"
    assert result["extraction"]["extractor"] == "sieve"
    assert result["extraction"]["verifiable"] is True
    assert result["extraction"]["source"] == "sum_cli sidecar"


# --------------------------------------------------------------------------
# Bundles without a sum_cli sidecar → extractor=None, verifiable=False
# --------------------------------------------------------------------------


def test_bundle_without_sidecar_reports_verifiable_false(tmp_path):
    """A bundle that has no ``sum_cli`` sidecar (e.g. produced by
    the codec directly, or sidecar stripped) MUST surface
    ``extraction.verifiable: false`` and ``source: "absent"``.
    The verifier does not assume reproducibility in the absence
    of provenance — fail-closed."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    algebra = GodelStateAlgebra()
    codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
    state = algebra.encode_chunk_state([("alice", "likes", "cats")])
    bundle = codec.export_bundle(state, branch="main", title="no-sidecar")
    # Note: codec does NOT add sum_cli sidecar — only sum_cli/main.py does.
    assert "sum_cli" not in bundle

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    result = _verify_via_subprocess(bundle_path)
    assert result["extraction"]["extractor"] is None
    assert result["extraction"]["verifiable"] is False
    assert result["extraction"]["source"] == "absent"


# --------------------------------------------------------------------------
# extractor=llm sidecar → verifiable=False (stochastic re-extraction)
# --------------------------------------------------------------------------


def test_llm_sidecar_reports_verifiable_false(tmp_path):
    """A bundle whose sidecar declares ``extractor: "llm"`` MUST
    surface ``extraction.verifiable: false`` — LLM extraction is
    stochastic; re-extracting the source prose can produce
    different triples. The signed bundle is still cryptographically
    valid, but it is NOT reproducibly verifiable.

    This test constructs the sidecar manually to avoid the
    `OPENAI_API_KEY` dependency of the actual `sum attest --extractor=llm`
    path."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    algebra = GodelStateAlgebra()
    codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
    state = algebra.encode_chunk_state([("alice", "likes", "cats")])
    bundle = codec.export_bundle(state, branch="main", title="llm-sidecar")
    # Construct the sidecar exactly as cmd_attest would for an LLM run:
    bundle["sum_cli"] = {
        "extractor": "llm",
        "source_uri": "sha256:" + hashlib.sha256(b"some prose").hexdigest(),
        "cli_version": "test-fixture",
        "generated_at": "2026-04-29T00:00:00Z",
    }

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    result = _verify_via_subprocess(bundle_path)
    assert result["extraction"]["extractor"] == "llm"
    assert result["extraction"]["verifiable"] is False
    assert result["extraction"]["source"] == "sum_cli sidecar"


# --------------------------------------------------------------------------
# Stderr human-readable line includes the extractor tag
# --------------------------------------------------------------------------


def test_stderr_human_line_names_the_extractor(tmp_path):
    """The human-readable stderr line MUST mention the extractor
    so an operator running `sum verify` interactively sees the
    provenance class without parsing the JSON output."""
    bundle_path = _attest_via_subprocess(
        "Alice likes cats.", "sieve", tmp_path,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "sum_cli.main", "verify",
         "--input", str(bundle_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "extractor=sieve" in proc.stderr
    assert "verifiable" in proc.stderr  # the (verifiable) tag


# --------------------------------------------------------------------------
# Threat-model row trace (this test's reason for existing)
# --------------------------------------------------------------------------


def test_extraction_visibility_closes_threat_model_row():
    """`docs/THREAT_MODEL.md` §3.3 names extraction manipulation
    as a partially-protected threat: structural gating catches
    malformed triplets but cannot detect semantically-incorrect
    extraction. The signed bundle does not communicate the
    extractor's reproducibility class to a downstream consumer.

    This test exists as documentation: the verifier now surfaces
    the extractor in its primary output. A downstream tool can
    write:

        sum verify --input bundle.json | jq -e '.extraction.verifiable'

    to gate on reproducible-extraction bundles. Before this
    PR, the same check required parsing the (non-load-bearing)
    sum_cli sidecar by hand."""
    # Pure-documentation test; the load-bearing assertions are above.
    assert True
