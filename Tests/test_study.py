"""`sum study` — the verifiable cheatsheet (machine-studying over a corpus).

Two layers:

  - the **expertise** scalar + StudyArtifact serialisation — pure-Python,
    dependency-free, always runs;
  - the **end-to-end pipeline** (extract → compose → render → score →
    optional signed receipt) — needs the [research]/[sieve]/[judge]
    extras, so it ``importorskip``s when they are absent (and runs in the
    full-extras CI image).

The expertise scalar is a MEASUREMENT, an analogy to studying-expertise —
not a guarantee. These tests pin its contract: bounded, cheap-end-weighted,
deterministic; and they pin the honest-boundary note on the artifact.
"""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from sum_engine_internal.research.study import (
    DEFAULT_DECAY,
    StudyArtifact,
    expertise,
)


# ── expertise scalar (pure-Python, always runs) ───────────────────────────


def test_perfect_fidelity_scores_one():
    # A study artifact that stays fully faithful at every compression →
    # expertise 1.0 (the ceiling).
    assert expertise([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_cheap_end_collapse_scores_below_naive_mean():
    # Faithful (expensive) end perfect, compressed (cheap) end worthless.
    # The cheap-budget weighting must pull the score WELL below the naïve
    # mean of 0.5 — the whole point of the metric.
    score = expertise([0.0, 1.0], [1.0, 0.0])
    assert score < 0.2


def test_cheap_end_good_scores_above_naive_mean():
    # The mirror: cheap end faithful, expensive end worthless → ABOVE 0.5.
    score = expertise([0.0, 1.0], [0.0, 1.0])
    assert score > 0.8


def test_single_point_returns_its_fidelity():
    assert expertise([0.3], [0.7]) == pytest.approx(0.7)


def test_bounded_and_deterministic():
    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    fid = [0.9, 0.7, 0.6, 0.55, 0.5]
    a = expertise(pos, fid)
    b = expertise(pos, fid)
    assert a == b
    assert 0.0 <= a <= 1.0


def test_zero_decay_is_plain_mean():
    # With no cheap-end weighting the score is the ordinary mean.
    assert expertise([0.0, 1.0], [0.4, 0.8], decay=0.0) == pytest.approx(0.6)


def test_default_decay_is_ln10():
    import math
    assert DEFAULT_DECAY == pytest.approx(math.log(10.0))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        expertise([], [])
    with pytest.raises(ValueError):
        expertise([0.0, 1.0], [1.0])          # length mismatch
    with pytest.raises(ValueError):
        expertise([0.0], [1.5])               # fidelity out of range
    with pytest.raises(ValueError):
        expertise([0.0], [1.0], decay=-1.0)   # negative decay


# ── StudyArtifact serialisation ───────────────────────────────────────────


def test_artifact_as_dict_shape():
    a = StudyArtifact(
        corpus_id="c", doc_count=2, state_integer=10 ** 40, axiom_count=3,
        cheatsheet="notes", study_density=0.1, expertise=0.9,
        scorer_name="x", scorer_version="1", frontier={"n": 3},
    )
    d = a.as_dict()
    assert d["schema"] == "sum.study_artifact.v1"
    # big state ints are stringified so JSON never lossily truncates them.
    assert d["state_integer"] == str(10 ** 40)
    assert d["certified"] is False
    assert "receipt" not in d
    # the honest boundary travels on the artifact.
    assert "measurement_note" in d and "not" in d["measurement_note"].lower()
    json.dumps(d)  # must be JSON-serialisable


def test_artifact_certified_flag_when_receipt_present():
    a = StudyArtifact(
        corpus_id="c", doc_count=1, state_integer=2, axiom_count=1,
        cheatsheet="n", study_density=0.1, expertise=0.5,
        scorer_name="x", scorer_version="1", receipt={"schema": "sum.meaning_risk_receipt.v1"},
    )
    d = a.as_dict()
    assert d["certified"] is True
    assert d["receipt"]["schema"] == "sum.meaning_risk_receipt.v1"


# ── CLI guard (runs without extras) ───────────────────────────────────────


@contextmanager
def _capture():
    out, err = io.StringIO(), io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        yield out, err


def _run_cli(argv):
    from sum_cli.main import main
    with _capture() as (out, err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_cli_no_documents_is_usage_error():
    # No --corpus and no --doc → clean usage error (does not crash). When the
    # [research] extra is absent the import-guard fires first; either way the
    # exit is the usage code 2 with a helpful message.
    rc, _, err = _run_cli(["study"])
    assert rc == 2
    assert "sum:" in err


# ── end-to-end pipeline (needs extras; skips gracefully otherwise) ─────────


def _have_full_stack():
    import importlib.util as u
    return all(u.find_spec(m) for m in ("spacy", "numpy"))


pytestmark_e2e = pytest.mark.skipif(
    not _have_full_stack(),
    reason="end-to-end study needs the [research]+[sieve] extras (spacy/numpy)",
)

CORPUS = {
    "a.txt": (
        "The treaty was signed in Vienna in 1815. Delegates redrew the map "
        "of Europe. The settlement held for decades."
    ),
    "b.txt": (
        "Metternich chaired the congress. The great powers balanced their "
        "interests. The agreement restrained French ambition."
    ),
    "c.txt": (
        "Diplomats negotiated borders carefully. The new order favoured "
        "monarchies. Peace endured across the continent."
    ),
}


def _write_corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in CORPUS.items():
        (d / name).write_text(text, encoding="utf-8")
    return str(d)


@pytestmark_e2e
def test_e2e_emits_study_artifact(tmp_path):
    corpus = _write_corpus(tmp_path)
    rc, out, _ = _run_cli(["study", "--corpus", corpus, "--scorer", "lexical"])
    assert rc == 0
    d = json.loads(out)
    assert d["schema"] == "sum.study_artifact.v1"
    assert d["doc_count"] == 3
    assert d["axiom_count"] > 0
    assert d["cheatsheet"]
    assert 0.0 <= d["expertise"] <= 1.0
    assert d["certified"] is False
    assert "measurement_note" in d
    # the state integer is the compose of the per-doc bundles — recompute it.
    int(d["state_integer"])  # parses as int


@pytestmark_e2e
def test_e2e_state_matches_compose(tmp_path):
    # Studying the corpus must produce the same SUM-of-SUMs state integer as
    # composing the per-doc bundles directly — composition is load-bearing.
    from sum_engine_internal.transforms import get_transform
    from sum_engine_internal.transforms._base import TransformEnv, run_sync

    corpus = _write_corpus(tmp_path)
    rc, out, _ = _run_cli(["study", "--corpus", corpus, "--scorer", "lexical"])
    assert rc == 0
    studied_state = int(json.loads(out)["state_integer"])

    env = TransformEnv()
    extract = get_transform("extract")
    compose = get_transform("compose")
    bundles = []
    for text in CORPUS.values():
        ext = run_sync(extract.apply({"text": text}, {"extractor": "sieve"}, env))
        if ext.output:
            bundles.append({"triples": ext.output})
    merged = run_sync(
        compose.apply({"bundles": bundles}, {"merge_strategy": "lcm"}, env)
    )
    assert int(merged.output["state_integer"]) == studied_state


@pytestmark_e2e
def test_e2e_certify_round_trips(tmp_path):
    joserfc = pytest.importorskip("joserfc")  # signing needs [research] joserfc
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization
    import base64

    def _b64u(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    sk = Ed25519PrivateKey.generate()
    raw_priv = sk.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    raw_pub = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    )
    private_jwk = {"kty": "OKP", "crv": "Ed25519",
                   "x": _b64u(raw_pub), "d": _b64u(raw_priv), "kid": "test"}
    jwk_path = tmp_path / "key.jwk"
    jwk_path.write_text(json.dumps(private_jwk), encoding="utf-8")

    corpus = _write_corpus(tmp_path)
    rc, out, _ = _run_cli([
        "study", "--corpus", corpus, "--scorer", "lexical", "--certify",
        "--signing-jwk", str(jwk_path), "--kid", "test",
        "--corpus-id", "study-test-v0",
    ])
    assert rc == 0
    d = json.loads(out)
    assert d["certified"] is True
    assert d["receipt"]["schema"] == "sum.meaning_risk_receipt.v1"
