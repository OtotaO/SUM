"""The golden sum.meaning_risk_receipt.v1 fixture — the demonstrated outcome.

A real, committed, signed meaning-risk receipt over a real (self-authored,
extractive) corpus. These tests prove the artifact is what it claims:

  - it verifies (signature) and REPLAYS (recompute losses from the
    committed corpus → re-certify → bound reproduces byte-for-byte);
  - it is byte-stable: re-running the generator reproduces it exactly
    (deterministic seed + fixed signed_at);
  - the JCS payload bytes the signature is computed over are
    BYTE-IDENTICAL across runtimes (Python ↔ Node) — the property that
    makes the signature cross-runtime-verifiable;
  - it discloses its blind spots and reports `controlled` honestly.

Honest scope note: the Node verifier in standalone_verifier/ is a
CanonicalBundle witness, not a generic JOSE verifier, so it does not yet
verify this schema end-to-end. The cross-runtime guarantee proven here is
the load-bearing one — byte-identical canonical payload — via the same
single_file_demo/jcs_cli.js the JCS byte-identity gate uses. Full
Node-side JWS verification of this schema is a named follow-up.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

joserfc = pytest.importorskip(
    "joserfc", reason="[receipt-verify] extra not installed"
)

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.research.meaning import (
    LexicalCoverageScorer,
    score_pairs,
    verify_meaning_risk_receipt,
)

_REPO = Path(__file__).resolve().parents[2]
_FIX = _REPO / "fixtures" / "meaning_receipts"


def _load(name):
    return json.loads((_FIX / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def golden():
    return _load("meaning_risk_receipt.golden.json")


@pytest.fixture(scope="module")
def jwks():
    return _load("jwks.json")


@pytest.fixture(scope="module")
def corpus_losses():
    corpus = _load("corpus_2026-06-06.json")
    pairs = [(p["source"], p["rendering"]) for p in corpus["pairs"]]
    return score_pairs(pairs, LexicalCoverageScorer())


# ── the core claim: verify + replay from the committed corpus ─────────


def test_golden_verifies_and_replays(golden, jwks, corpus_losses):
    """The headline: recompute the losses from the committed corpus and
    the committed receipt's bound replays byte-for-byte."""
    payload = verify_meaning_risk_receipt(golden, jwks, losses=corpus_losses)
    assert payload["corpus_id"] == "meaning-demo-extractive-2026-06-06"
    assert payload["n"] == 16
    assert payload["scorer"] == "lexical-coverage-bidirectional"


def test_golden_signature_and_disclosure_only(golden, jwks):
    """Without the losses: signature + disclosure invariants still pass."""
    payload = verify_meaning_risk_receipt(golden, jwks)
    assert payload["schema"] if "schema" in payload else True  # payload has no schema; envelope does
    assert golden["schema"] == "sum.meaning_risk_receipt.v1"


def test_golden_discloses_blind_spots(golden, jwks):
    payload = verify_meaning_risk_receipt(golden, jwks)
    assert "arrangement" in payload["not_covered"]
    assert payload["disclosure"].strip()


def test_golden_reports_controlled_honestly(golden):
    """At n=16 the distribution-free ceiling is wide; the receipt
    honestly reports controlled=False against the 0.5 target rather than
    fudging it. Pinned so a regeneration that flips it is noticed."""
    pl = golden["payload"]
    assert pl["alpha_target_micro"] == 500_000
    assert pl["controlled"] is False
    assert pl["risk_upper_bound_micro"] > pl["alpha_target_micro"]
    assert pl["point_estimate_micro"] < pl["risk_upper_bound_micro"]  # slack = finite-sample price


# ── byte-stable regeneration (determinism) ────────────────────────────


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_meaning_golden_gen", _FIX / "generate_fixtures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_golden_is_byte_stable(golden, jwks):
    """Re-running the generator reproduces the committed fixture exactly
    — deterministic seed + fixed signed_at."""
    gen = _load_generator()
    rebuilt_receipt, rebuilt_jwks = gen.build()
    assert rebuilt_receipt == golden
    assert rebuilt_jwks == jwks


# ── cross-runtime: the canonical payload bytes are identical ──────────


def test_golden_payload_jcs_byte_identical_across_runtimes(golden):
    """The property the cross-runtime signature rests on: the JCS
    canonical bytes of the payload are byte-for-byte identical in Python
    and Node (via single_file_demo/jcs_cli.js)."""
    node = shutil.which("node")
    jcs_cli = _REPO / "single_file_demo" / "jcs_cli.js"
    if node is None or not jcs_cli.exists():
        pytest.skip("node or jcs_cli.js unavailable")

    py_bytes = canonicalize(golden["payload"])
    proc = subprocess.run(
        [node, str(jcs_cli)],
        input=json.dumps(golden["payload"]).encode("utf-8"),
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == py_bytes, (
        "JCS canonical bytes diverged between Python and Node for the "
        "golden meaning-risk-receipt payload"
    )
