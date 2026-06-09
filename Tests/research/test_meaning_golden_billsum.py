"""The BillSum golden ``sum.meaning_risk_receipt.v1`` — the arXiv binding-gate
artifact over a REAL public-domain corpus (meaning-preserving COMPRESSION).

This is the receipt the bench-hardening / product-vision "one real receipt
over a real corpus" gate names: 64 US Congressional bills (BillSum,
**CC0-1.0**) summarized, scored by the local MiniLM-cosine judge, certified
distribution-free, signed.

The load-bearing honesty split these tests pin:
  * The **certificate replays offline** over the committed integer-micro
    loss vector (``losses_billsum.json``) — the pure-Python certifier
    reproduces the bound with NO model, NO GPU, deterministic everywhere.
    That is what CI checks here.
  * Re-deriving the losses from raw text needs the (machine-pinned) MiniLM
    judge — NOT exercised in CI; the receipt discloses this in its
    ``disclosure`` field rather than hiding it.

So these tests use the committed losses, never the judge — they run in CI
with numpy + joserfc only (no torch).
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
from sum_engine_internal.research.meaning import verify_meaning_risk_receipt

_REPO = Path(__file__).resolve().parents[2]
_FIX = _REPO / "fixtures" / "meaning_receipts_billsum"


def _load(name):
    return json.loads((_FIX / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def golden():
    return _load("meaning_risk_receipt.billsum.golden.json")


@pytest.fixture(scope="module")
def jwks():
    return _load("jwks.json")


@pytest.fixture(scope="module")
def committed_losses():
    return _load("losses_billsum.json")["losses"]


# ── the core claim: verify + replay from the committed loss vector ────


def test_billsum_golden_verifies_and_replays(golden, jwks, committed_losses):
    """Headline: the committed receipt's bound replays byte-for-byte over
    the committed integer-micro loss vector — pure-Python, no judge."""
    payload = verify_meaning_risk_receipt(golden, jwks, losses=committed_losses)
    assert payload["corpus_id"] == "billsum-test-first64-cc0"
    assert payload["n"] == 64
    assert payload["scorer"] == "bidirectional-entailment[minilm-cosine-0.5]"
    assert payload["method"] == "hoeffding"  # auto → hoeffding for fractional


def test_billsum_golden_signature_and_disclosure_only(golden, jwks):
    """Without losses: signature + schema + disclosure invariants pass."""
    verify_meaning_risk_receipt(golden, jwks)
    assert golden["schema"] == "sum.meaning_risk_receipt.v1"


def test_billsum_golden_discloses_machine_pinning(golden, jwks):
    payload = verify_meaning_risk_receipt(golden, jwks)
    assert "arrangement" in payload["not_covered"]
    d = payload["disclosure"].lower()
    assert "machine-pinned" in d and "named" in d  # honest judge caveat present


def test_billsum_golden_reports_controlled_honestly(golden):
    """A real, non-vacuous, CONTROLLED bound: certified meaning-loss
    ≤ 0.6454 at 95% over 64 bills, under the 0.7 target. Pinned so a
    regeneration that moves it is noticed."""
    pl = golden["payload"]
    assert pl["alpha_target_micro"] == 700_000
    assert pl["controlled"] is True
    assert pl["risk_upper_bound_micro"] == 645_438
    assert pl["point_estimate_micro"] < pl["risk_upper_bound_micro"]  # finite-sample slack
    assert pl["risk_upper_bound_micro"] < pl["alpha_target_micro"]    # controlled


def test_billsum_corpus_is_real_public_domain(committed_losses):
    """The corpus is the real CC0 BillSum slice (not self-authored), n=64."""
    corpus = _load("corpus_billsum_test_first64.json")
    assert corpus["source_dataset"] == "FiscalNote/billsum"
    assert corpus["license"].startswith("CC0")
    assert len(corpus["pairs"]) == 64 == len(committed_losses)
    # losses are genuine fractional meaning-loss, not 0/1 — abstractive corpus
    assert all(0.0 <= x <= 1.0 for x in committed_losses)
    assert any(0.1 < x < 0.9 for x in committed_losses)


# ── byte-stable regeneration (judge-free: reads the committed losses) ──


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_billsum_gen", _FIX / "generate_billsum_fixture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_billsum_golden_is_byte_stable(golden, jwks):
    """Re-running the generator reproduces the committed fixture exactly.
    Deterministic + judge-free, because build() reads the committed loss
    vector rather than re-running the model."""
    gen = _load_generator()
    rebuilt_receipt, rebuilt_jwks, _losses = gen.build()
    assert rebuilt_receipt == golden
    assert rebuilt_jwks == jwks


# ── cross-runtime: the canonical payload bytes are identical ──────────


def test_billsum_payload_jcs_byte_identical_across_runtimes(golden):
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
    assert proc.stdout == py_bytes
