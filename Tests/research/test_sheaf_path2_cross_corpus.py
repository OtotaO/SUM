"""
Pin the §4.7.4 cross-corpus aggregator. Asserts that loading the
three committed compare receipts (seed_long_paragraphs,
seed_paragraphs, seed_news_briefs at n=5 models each) produces:

  - per-corpus joint findings unchanged
  - 15-cell verdict matrix unchanged
  - cross-corpus joint finding `CROSS_CORPUS_VERDICTS_DIVERGE`
  - the load-bearing claim that 1/15 cells produces BEATS

The aggregator is a pure function over the input receipts (no LLM
calls), so this test runs in CI without API keys.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

PER_CORPUS_RECEIPTS = {
    "seed_long_paragraphs": RECEIPTS_DIR
        / "path2_multi_llm_compare_seed_long_paragraphs_2026-05-06.json",
    "seed_paragraphs": RECEIPTS_DIR
        / "path2_multi_llm_compare_seed_paragraphs_2026-05-06.json",
    "seed_news_briefs": RECEIPTS_DIR
        / "path2_multi_llm_compare_seed_news_briefs_2026-05-06.json",
}

_DETERMINISTIC_BLAS_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Pinned per-(corpus, model) bench_digests at n=5. These are the
# Phase 2 outputs against the committed snapshots; drift on any one
# means either the scorer changed or a snapshot was regenerated.
PINNED_DIGESTS = {
    "seed_long_paragraphs": {
        "gpt-4o-mini-2024-07-18":            "7b364fc6ae23ce4ea24c69cf7b299b10402237f6f0c4364b18fbcb1dbcc4b75e",
        "claude-haiku-4-5-20251001":         "d0f9f175662216d50dbfd1ec23d90eb8b4774bb95d220e2f951399e8ed52f6f7",
        "meta-llama/Llama-3.3-70B-Instruct": "f1c17c3e920811b1fdbd376adc168e6f777be781310eedb945c7c9e2aac29b31",
        "Qwen/Qwen3.6-35B-A3B":              "23da3ecb0404d26920bcf0bd4ad519e9a19d2e1a75085df81464fd92461b8ea2",
        "deepseek-ai/DeepSeek-V3-0324":      "619a413f6b62203aefcc7c2d8d01db36935ce2148307ace10a908f112fe22c9f",
        "google/gemma-3-27b-it":             "fe76913e1eabf88e7d351752b20e0eb4cba011c527508a0616f131609318b9b5",
    },
    "seed_paragraphs": {
        "gpt-4o-mini-2024-07-18":            "2d13d41abdc73badd8f60223ae958bbbb7c056d8068d76df12cac378422fd509",
        "meta-llama/Llama-3.3-70B-Instruct": "a079be325705c593d87f922128939d7778e5acbdace6fb7440f008236547d2b2",
        "Qwen/Qwen3.6-35B-A3B":              "6c8a7a67a50bea609fc955b6242eb67d9b0cab99b421034af306e651ed0c54ef",
        "deepseek-ai/DeepSeek-V3-0324":      "394535b1f41b3a9463a19c5bc29db9b137fed63ca3f4341f71ccbd2477262437",
        "google/gemma-3-27b-it":             "33e8f20c001d3024a89971ebd56d0b82133597078704c01d18d0e4ea48ce629b",
    },
    "seed_news_briefs": {
        "gpt-4o-mini-2024-07-18":            "26a42c0f5884bab92154f0b63a581618855f1dc18ffdde3bed9ed037cdf0d8e3",
        "meta-llama/Llama-3.3-70B-Instruct": "6fb8ea1a1daf1074e17b806c59dd4f587765be92cf1f71a91bc9ac94719740d5",
        "Qwen/Qwen3.6-35B-A3B":              "5d8532fee6a9fce8bcba9b4cad16a3ff95d662166c70467193b29b26528f1b3c",
        "deepseek-ai/DeepSeek-V3-0324":      "a0a3422798febe2880b1fa85e649424402120dbf8aac177e6306df71517c980b",
        "google/gemma-3-27b-it":             "3d7ab30af845d69337dd6a022644a438e56329c368426008c306a565a8c9a0ff",
    },
}

PINNED_PER_CORPUS_JOINT = {
    "seed_long_paragraphs": "STRUCTURAL_GAP_NO_MODEL_BEATS",
    "seed_paragraphs":      "MIXED_VERDICTS_MODEL_DEPENDENT",
    "seed_news_briefs":     "STRUCTURAL_GAP_NO_MODEL_BEATS",
}

_ALL_RECEIPTS_PRESENT = all(p.exists() for p in PER_CORPUS_RECEIPTS.values())


@pytest.mark.skipif(
    not _ALL_RECEIPTS_PRESENT,
    reason="cross-corpus receipts missing — run the multi-LLM compare "
           "on each of the three corpora first",
)
def test_cross_corpus_aggregate_diverges():
    """The §4.7.4 load-bearing claim: across three stylistically
    distinct corpora, the §4.7.3 structural-gap result does NOT
    reproduce uniformly.

    Two corpora (seed_long_paragraphs, seed_news_briefs) reproduce
    `STRUCTURAL_GAP_NO_MODEL_BEATS` at n=5 models. The third
    (seed_paragraphs, smaller at 8 docs) flips to
    `MIXED_VERDICTS_MODEL_DEPENDENT` because gpt-4o-mini's Δ on
    that corpus crosses the +0.030 BEATS threshold (Δ=+0.032).
    Aggregator's joint finding: `CROSS_CORPUS_VERDICTS_DIVERGE`.

    This is a falsification of any naive reading of §4.7.3 as
    asserting *all* hybrid runs LOSE on real LLM. The honest
    reading is: the hybrid does not reliably BEAT baseline across
    LLM families × corpora, but isolated cells can produce a
    positive Δ at the threshold.
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.research.sheaf_path2_cross_corpus_aggregate",
            "--receipts",
            *[str(p) for p in PER_CORPUS_RECEIPTS.values()],
        ],
        cwd=str(REPO),
        env=_DETERMINISTIC_BLAS_ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    out_path = None
    for line in proc.stdout.splitlines():
        if line.lstrip().startswith("→ wrote "):
            out_path = REPO / line.split("→ wrote ", 1)[1].strip()
            break
    assert out_path is not None, (
        f"Aggregator did not print '→ wrote' line. stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    report = json.loads(out_path.read_text())

    assert report["schema"] == "sum.sheaf_path2_cross_corpus_compare.v1"
    assert report["n_corpora"] == 3
    assert sorted(report["corpora"]) == sorted(PER_CORPUS_RECEIPTS.keys())

    # Per-corpus joint findings.
    for corpus, expected in PINNED_PER_CORPUS_JOINT.items():
        got = report["per_corpus_joint_finding"][corpus]
        assert got == expected, (
            f"per-corpus joint finding for {corpus} drifted: "
            f"got {got!r}, expected {expected!r}"
        )

    # Per-(corpus, model) bench_digests pinned through the embedded
    # per-corpus receipts.
    for corpus, model_to_digest in PINNED_DIGESTS.items():
        per_model = report["per_corpus_receipts"][corpus]["per_model_reports"]
        for model, expected in model_to_digest.items():
            got = per_model[model]["bench_digest"]
            assert got == expected, (
                f"{corpus} × {model} digest drift: got {got}, expected {expected}"
            )

    # Cross-corpus joint finding.
    assert report["joint_finding"] == "CROSS_CORPUS_VERDICTS_DIVERGE", (
        f"Cross-corpus joint finding drifted: got "
        f"{report['joint_finding']!r}, expected "
        f"CROSS_CORPUS_VERDICTS_DIVERGE. The §4.7.4 narrative depends "
        f"on the per-corpus joint findings disagreeing across the "
        f"three-corpus sample."
    )

    # Cell counts. The matrix is jagged: seed_long_paragraphs has
    # claude-haiku-4.5 (from PR #158, snapshot committed and reused
    # by the n=6 multi-LLM test), while seed_paragraphs and
    # seed_news_briefs do not (the operator's Anthropic key was
    # unavailable during the §4.7.4 capture). So 6 + 5 + 5 = 16 cells
    # total. Counts: 1 BEATS (seed_paragraphs × gpt-4o-mini), 8 TIES,
    # 7 LOSES (the extra LOSES vs an n=5 reading is claude on
    # seed_long_paragraphs at Δ=-0.032).
    assert report["n_cells_total"] == 16
    assert report["n_cells_beats"] == 1, (
        f"BEATS-cell count drifted: {report['n_cells_beats']}. The "
        f"§4.7.4 narrative records exactly one BEATS cell across the "
        f"16-cell matrix (seed_paragraphs × gpt-4o-mini)."
    )
    assert report["n_cells_ties"] == 8
    assert report["n_cells_loses"] == 7
