"""Research paths replay sieve extractor v1; product paths run v2.

The 2026-10-04 clause guard made ``DeterministicSieve()`` default to
``sum.sieve:deterministic_v2``. Results recorded before it (committed
research receipts, documented findings, the bundle MMD baseline) were
extracted with v1, so every research, spike and experiment path
constructs ``DeterministicSieve(extractor_id=SIEVE_EXTRACTOR_ID_V1)``.
Before this pin the F3 diagnostic digest, the recursive walk's
deterministic-arm medians, the v2 ROC bench's corpus and the bundle MMD
baseline all moved under v2 while the docs still cited the v1 values.

The value tests below compare against the committed receipts, so a
receipt that stops reproducing fails here instead of passing silently.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPTS = REPO / "fixtures" / "bench_receipts"

# Research, spike and experiment code, plus bench runners whose committed
# receipts depend on sieve output.
RESEARCH_PATHS = sorted(
    list((REPO / "scripts" / "research").rglob("*.py"))
    + list((REPO / "sum_engine_internal" / "research").rglob("*.py"))
    + [REPO / "scripts" / "bench" / "runners" / "s25_iterated_round_trip.py"]
)

# Shipping paths: the current extractor (no extractor_id argument).
PRODUCT_PATHS = [
    REPO / "sum_cli" / "main.py",
    REPO / "sum_engine_internal" / "algorithms" / "chunked_corpus.py",
    REPO / "sum_engine_internal" / "transforms" / "extract.py",
    REPO / "sum_engine_internal" / "agent_surface" / "mcp_bind.py",
    REPO / "sum_engine_internal" / "evidence" / "chain.py",
    REPO / "api" / "quantum_router.py",
    REPO / "scripts" / "bench" / "runners" / "extraction.py",
    REPO / "scripts" / "bench" / "runners" / "roundtrip.py",
    REPO / "scripts" / "bench" / "runners" / "regeneration.py",
    REPO / "scripts" / "bench" / "runners" / "negative_control.py",
]


def _sieve_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DeterministicSieve"
    ]


def test_every_research_sieve_replays_v1() -> None:
    unpinned = []
    seen = 0
    for path in RESEARCH_PATHS:
        for call in _sieve_calls(path):
            seen += 1
            pinned = any(
                kw.arg == "extractor_id"
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "SIEVE_EXTRACTOR_ID_V1"
                for kw in call.keywords
            )
            if not pinned:
                unpinned.append(f"{path.relative_to(REPO)}:{call.lineno}")
    assert seen >= 20, seen
    assert unpinned == []


def test_product_paths_use_the_current_extractor() -> None:
    seen = 0
    for path in PRODUCT_PATHS:
        for call in _sieve_calls(path):
            seen += 1
            assert not any(kw.arg == "extractor_id" for kw in call.keywords), (
                f"{path.relative_to(REPO)}:{call.lineno}"
            )
    assert seen >= 8, seen


spacy = pytest.importorskip("spacy")
np = pytest.importorskip("numpy")


def test_v2_roc_bench_corpus_matches_its_receipt() -> None:
    # Under v2 this was 106 triples and a 210 / 69 vocabulary.
    from scripts.research.sheaf_v2_roc_bench import extract_corpus_triples

    receipt = json.loads(
        (RECEIPTS / "sheaf_v2_roc_seed_long_paragraphs_2026-05-01.json").read_text()
    )
    docs = extract_corpus_triples()
    triples = [t for _, ts in docs for t in ts]
    entities = {s for s, _, _ in triples} | {o for _, _, o in triples}
    relations = {p for _, p, _ in triples}
    assert len(triples) == receipt["n_source_triples"] == 120
    assert len(entities) == receipt["vocab_size_entities"]
    assert len(relations) == receipt["vocab_size_relations"]


def test_recursive_walk_deterministic_arm_matches_its_receipt(capsys) -> None:
    # Under v2 the medians moved to 0.6667 / 0.775 and no news brief
    # collapsed. The committed bench_digest already differed at 0.11.1
    # (news median_fixed_point_step 2.5 against 2.0), so the
    # documented result-level fields are compared.
    import scripts.research.recursive_compression_walk as rcw

    receipt = json.loads(
        (RECEIPTS / "recursive_compression_walk_deterministic_2026-05-08.json").read_text()
    )
    report = rcw.run_recursive_walk(
        corpora=list(receipt["corpora"]), compressor="deterministic",
        model=rcw.DEFAULT_LLM_MODEL, max_steps=receipt["max_steps"],
        thresholds=tuple(receipt["recall_thresholds"]),
    )
    capsys.readouterr()
    for corpus, data in receipt["by_corpus"].items():
        want = data["aggregate"]["summary"]
        got = report["by_corpus"][corpus]["aggregate"]["summary"]
        for key in (
            "median_fixed_point_recall_vs_original",
            "n_docs_collapsed_to_empty",
            "median_n_axioms_original",
            "median_fixed_point_n_axioms",
        ):
            assert got[key] == want[key], (corpus, key)


def test_mmd_baseline_matches_the_documented_calibration() -> None:
    # docs/MMD_WIRE_FINDINGS.md: 314 baseline triples. Under v2 the
    # baseline had 295 and every bundle's MMD fields changed for
    # identical axioms (p for this set moved from 0.035 to 0.065).
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.mmd.baseline import BaselineMMDComputer

    mmd = BaselineMMDComputer()
    assert mmd.calibrate_from_corpora()
    result = mmd.predict_mmd([
        Triple("marie_curie", "discover", "radium"),
        Triple("marie_curie", "win", "nobel prize"),
    ])
    assert result["n_baseline_samples"] == 314
    assert round(result["permutation_p_value"], 3) == 0.035


@pytest.mark.slow
def test_f3_diagnostic_digest_matches_its_receipt() -> None:
    # Under v2 the digest was ae1e716b... (106 corpus triples, not 120).
    receipt = json.loads(
        (RECEIPTS / "v3_1_f3_diagnostic_2026-05-03.json").read_text()
    )
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": str(REPO),
    }
    code = (
        "from scripts.research.sheaf_v3_1_f3_diagnostic import main\n"
        "print('DIGEST=' + main().bench_digest)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    digest = proc.stdout.rsplit("DIGEST=", 1)[1].strip()
    assert digest == receipt["bench_digest"]
