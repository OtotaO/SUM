"""The cross-lingual translation golden ``sum.meaning_risk_receipt.v1`` — the
paraphrase-robustness half of the binding-gate pair.

Demonstrates the moat: faithful EN→FR translations preserve meaning (~0 loss)
despite **zero lexical overlap** — the dial scores by meaning, robust to the
most extreme rewriting (a different language). Corpus = opus-100 en-fr test
(first 64 length-aligned pairs); judge = the multilingual mDeBERTa NLI judge.

Like the BillSum golden, CI checks only the offline-replayable half (Stage A
+ Stage B over the committed integer-micro loss vector) — no torch, no fetch.
opus-100's raw text is NOT redistributed (mixed licence): the corpus is a
sha256-pinned pointer; re-deriving the losses re-fetches it under its own
terms. The judge loss computation is machine-pinned (F23/F26), disclosed.
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
_FIX = _REPO / "fixtures" / "meaning_receipts_translation"


def _load(name):
    return json.loads((_FIX / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def golden():
    return _load("meaning_risk_receipt.translation.golden.json")


@pytest.fixture(scope="module")
def jwks():
    return _load("jwks.json")


@pytest.fixture(scope="module")
def committed_losses():
    return _load("losses_translation.json")["losses"]


def test_translation_golden_verifies_and_replays(golden, jwks, committed_losses):
    payload = verify_meaning_risk_receipt(golden, jwks, losses=committed_losses)
    assert payload["corpus_id"] == "opus100-en-fr-test-first64-filtered"
    assert payload["n"] == 64
    assert payload["scorer"].startswith("bidirectional-entailment[nli:")
    assert payload["transform"] == "translate:en->fr"


def test_translation_golden_signature_and_disclosure_only(golden, jwks):
    verify_meaning_risk_receipt(golden, jwks)
    assert golden["schema"] == "sum.meaning_risk_receipt.v1"


def test_translation_golden_discloses_machine_pinning(golden, jwks):
    payload = verify_meaning_risk_receipt(golden, jwks)
    assert "arrangement" in payload["not_covered"]
    d = payload["disclosure"].lower()
    assert "machine-pinned" in d and "zero lexical overlap" in d


def test_translation_demonstrates_the_moat(committed_losses):
    """The headline: a large share of faithful EN→FR pairs score EXACTLY 0
    meaning-loss despite zero lexical overlap — the dial credits cross-lingual
    paraphrase. (39/64 at issue time; pin a conservative floor.)"""
    n_zero = sum(1 for x in committed_losses if x == 0.0)
    assert len(committed_losses) == 64
    assert n_zero >= 30, f"only {n_zero}/64 pairs at zero loss — moat regressed?"


def test_translation_golden_reports_controlled_honestly(golden):
    """Real, non-vacuous, CONTROLLED: certified meaning-loss ≤ 0.4124 at 95%
    over 64 pairs, under the 0.5 target. Mean loss < BillSum compression's —
    translation preserves more than aggressive summarization (the dial
    grades). Pinned so a regeneration that moves it is noticed."""
    pl = golden["payload"]
    assert pl["alpha_target_micro"] == 500_000
    assert pl["controlled"] is True
    assert pl["risk_upper_bound_micro"] == 412_359
    assert pl["point_estimate_micro"] == 259_375
    assert pl["point_estimate_micro"] < pl["risk_upper_bound_micro"] < pl["alpha_target_micro"]


def test_translation_corpus_pointer_is_hash_pinned():
    """The corpus is a sha256-pinned pointer (raw text not redistributed —
    mixed-licence opus-100), so the selection is reproducible + verifiable."""
    ptr = _load("corpus_pointer.json")
    assert ptr["source_dataset"] == "Helsinki-NLP/opus-100"
    assert ptr["n"] == 64
    assert ptr["corpus_sha256"].startswith("sha256-")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_translation_gen", _FIX / "generate_translation_fixture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_translation_golden_is_byte_stable(golden, jwks):
    """Re-running the generator reproduces the committed fixture exactly —
    deterministic, judge-free AND fetch-free (reads the committed losses)."""
    gen = _load_generator()
    rebuilt_receipt, rebuilt_jwks, _losses = gen.build()
    assert rebuilt_receipt == golden
    assert rebuilt_jwks == jwks


def test_translation_payload_jcs_byte_identical_across_runtimes(golden):
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
