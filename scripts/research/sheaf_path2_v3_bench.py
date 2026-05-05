"""
Path 2 — real-LLM-rendered adversarial bench for the v3.x +
complementary-hybrid detector composition.

Closes the load-bearing asterisk in §7 bounded claims of the preprint:
"Not generalising to real-LLM-rendered hallucinations." Tests whether
the complementary-hybrid detector's WIN claim (§4.7.1) survives when
adversarial perturbations come from an actual LLM rather than the
synthetic A1 / A2 / A4 triple-set mutations used in §4.4 / §4.7.

## Architecture: capture-once-replay-forever

The bench has two phases. The first is non-deterministic (LLM API
calls); the second is deterministic.

  Phase 1 — Snapshot capture (one-time, requires OPENAI_API_KEY):

    For each doc D in seed_long_paragraphs:
      - Sieve-extract source triples T_src from D.text
      - Render T_src via the LLM at four prompt classes:
          neutral_prompt  → R_neutral  (faithful baseline)
          a1_adversarial  → R_a1       (subtly substitute one entity)
          a2_adversarial  → R_a2       (subtly change one predicate)
          a4_adversarial  → R_a4       (drop one fact)
      - Save the rendered tomes + provenance to
        `fixtures/bench_renders/path2_seed_long_paragraphs.json`.

    Cost: ~64 LLM calls × ~500 tokens = ~32K tokens at gpt-4o-mini
    rates ≈ $0.05 total. Snapshot is committed to the repo.

  Phase 2 — Score from snapshot (deterministic; runs on every CI):

    For each doc:
      - Re-extract triples from each rendered tome via the
        deterministic sieve. Each (clean, adversarial) pair becomes
        a labeled detection trial.
      - Score with all four detectors that the §4.7.1 hybrid
        composition uses: B1 entity-presence-deficit, B2 jaccard,
        v3.2 + per-triple, complementary Borda hybrid.
    Compute per-(class, detector) AUC. Compute bench_digest over
    quantized AUCs. Schema: `sum.sheaf_path2_v3_bench.v1`.

## Why the architecture is this way

The substrate's reproducibility primitive (`bench_digest`) requires
deterministic input. LLM responses are not deterministic even at
temperature=0 (OpenAI doesn't guarantee bit-stability). Capturing
the snapshot ONCE and committing it to the repo makes the bench's
output byte-stable for all subsequent runs — anyone re-running the
score phase against the committed snapshot reproduces the digest.
The snapshot itself is a one-time empirical measurement, marked with
its model snapshot, prompts, and timestamp.

If the snapshot is ever regenerated (operator runs Phase 1 again
against gpt-4o-mini), the digest will change. The bench prints
"snapshot regenerated" in that case so the operator can decide
whether the change is intentional.

## Honest scope

  - Tests against ONE LLM (gpt-4o-mini-2024-07-18). Other models
    may produce different perturbation distributions; v0.4 candidate.
  - Adversarial prompts approximate A1/A2/A4 classes — they're
    instructions to the LLM to perturb, not direct triple mutation.
    The LLM may comply imperfectly (a "subtle entity swap" prompt
    may produce many different kinds of perturbation). The
    re-extraction step measures the actual perturbation, regardless
    of intent. This is honest: we measure what the LLM does, not
    what we asked it to do.
  - Re-extraction uses the deterministic sieve (same as the
    synthetic bench), so re-extraction noise is held constant and
    the comparison is apples-to-apples on the detector side.

Output: fixtures/bench_receipts/path2_v3_bench_<DATE>.json
        schema: sum.sheaf_path2_v3_bench.v1
"""
from __future__ import annotations

# MUST come before any numpy/scipy import — sets BLAS thread vars
# at process startup so bench_digest is byte-stable across fresh
# Python processes. See `_deterministic_blas` for the rationale.
import scripts.research._deterministic_blas  # noqa: F401

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    combined_detector_score,
    train_restriction_maps,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    weights_from_receipts,
)
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

from scripts.research.sheaf_v3_roc_bench import (
    _build_doc_sheaf,
    extract_corpus_triples,
    partition_trust,
    roc_auc,
)
from scripts.research.sheaf_v3_2_validation import (
    compute_bench_digest,
    quantize_for_digest,
    score_v32_combined,
)
from scripts.research.sheaf_baseline_comparison import (
    score_b1_entity_presence_deficit,
    score_b2_jaccard_distance,
)
from scripts.research.sheaf_per_triple_integration_experiment import (
    score_v32_with_per_triple,
)
from scripts.research.sheaf_hybrid_comparison import borda_fuse

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"
RENDERS_DIR = REPO / "fixtures" / "bench_renders"
SNAPSHOT_PATH = RENDERS_DIR / "path2_seed_long_paragraphs.json"
CORPUS_PATH = REPO / "scripts" / "bench" / "corpora" / "seed_long_paragraphs.json"

# Pinned model snapshot — matches PROOF_BOUNDARY §2.4 convention.
PINNED_MODEL = "gpt-4o-mini-2024-07-18"
GAMMA = 0.1


# ─── Phase 1 — Snapshot capture ──────────────────────────────────────


_PROMPT_CLASSES = ("neutral", "a1_adversarial", "a2_adversarial", "a4_adversarial")


def _build_prompt(triples: list[tuple[str, str, str]], prompt_class: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given perturbation
    class. Source triples are encoded as 'subject {p} object' lines so
    the LLM has unambiguous facts to render."""
    facts_block = "\n".join(f"- {s} {p} {o}" for s, p, o in triples)
    base_user = f"FACTS TO INCLUDE:\n{facts_block}"

    if prompt_class == "neutral":
        sys_prompt = (
            "You are a precise technical writer. Extrapolate the "
            "following absolute facts into a cohesive narrative. "
            "Do NOT invent facts. Do NOT change any subject, predicate, "
            "or object. Render every fact faithfully."
        )
        return sys_prompt, base_user

    if prompt_class == "a1_adversarial":
        sys_prompt = (
            "You are an adversarial test writer. Write a cohesive narrative "
            "that includes most of the supplied facts BUT subtly substitutes "
            "the SUBJECT or OBJECT of EXACTLY ONE fact with a plausible-but-"
            "wrong entity (e.g., a different organization, person, or place "
            "of similar type). The substitution should be subtle enough that "
            "a casual reader would not notice. Render the rest faithfully."
        )
        return sys_prompt, base_user

    if prompt_class == "a2_adversarial":
        sys_prompt = (
            "You are an adversarial test writer. Write a cohesive narrative "
            "that includes most of the supplied facts BUT subtly changes "
            "the PREDICATE of EXACTLY ONE fact to a different relation "
            "of similar type (e.g., 'founded' → 'managed', 'invented' → "
            "'used'). The substitution should preserve grammatical "
            "plausibility. Render the rest faithfully."
        )
        return sys_prompt, base_user

    if prompt_class == "a4_adversarial":
        sys_prompt = (
            "You are an adversarial test writer. Write a cohesive narrative "
            "that includes most of the supplied facts BUT silently OMITS "
            "EXACTLY ONE fact (the LLM's choice — pick a fact whose absence "
            "would not be obvious). Render the rest faithfully. The narrative "
            "should read naturally despite the omission."
        )
        return sys_prompt, base_user

    raise ValueError(f"unknown prompt class: {prompt_class!r}")


async def _render_one(adapter, triples: list[tuple[str, str, str]], prompt_class: str) -> str:
    """Single LLM call for one (doc, prompt_class) cell."""
    sys_prompt, user_prompt = _build_prompt(triples, prompt_class)
    response = await adapter.client.chat.completions.create(
        model=adapter.model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Default temperature; we cache the snapshot, so determinism
        # across temperatures is moot — what matters is the snapshot
        # captures real LLM behaviour.
    )
    return response.choices[0].message.content or ""


async def _capture_snapshot() -> dict[str, Any]:
    """Phase 1: render each doc × prompt class via the LLM.
    Requires OPENAI_API_KEY in env."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set — Phase 1 (snapshot capture) requires "
            "the operator's API key. Set OPENAI_API_KEY then re-run with "
            "--regenerate-snapshot."
        )
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    adapter = LiveLLMAdapter(model=PINNED_MODEL)
    print(f"[capture] using model: {PINNED_MODEL}")

    sieve = DeterministicSieve()
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    docs_raw = data["documents"]
    docs: list[tuple[str, str, list[tuple[str, str, str]]]] = []
    for d in docs_raw:
        triples = list(sieve.extract_triplets(d["text"]))
        if triples:
            docs.append((d["id"], d["text"], triples))
    print(f"[capture] {len(docs)} docs with non-empty extractions")

    snapshot: dict[str, Any] = {
        "schema": "sum.sheaf_path2_render_snapshot.v1",
        "corpus": "seed_long_paragraphs",
        "model": PINNED_MODEL,
        "prompt_classes": list(_PROMPT_CLASSES),
        "renders": {},
    }

    n_calls = 0
    for i, (doc_id, _text, source_triples) in enumerate(docs):
        snapshot["renders"][doc_id] = {
            "source_triples": [list(t) for t in source_triples],
            "by_prompt_class": {},
        }
        for prompt_class in _PROMPT_CLASSES:
            tome = await _render_one(adapter, source_triples, prompt_class)
            snapshot["renders"][doc_id]["by_prompt_class"][prompt_class] = tome
            n_calls += 1
            print(f"[capture] [{i+1}/{len(docs)}] {doc_id} {prompt_class:<18} "
                  f"({len(tome)} chars)")

    snapshot["n_llm_calls"] = n_calls
    return snapshot


def _ensure_snapshot(force: bool = False) -> dict[str, Any]:
    """Return the snapshot dict, capturing if needed."""
    if SNAPSHOT_PATH.exists() and not force:
        with open(SNAPSHOT_PATH) as f:
            snap = json.load(f)
        print(f"[capture] using cached snapshot: {SNAPSHOT_PATH.name}")
        return snap
    print(f"[capture] regenerating snapshot at {SNAPSHOT_PATH}")
    snap = asyncio.run(_capture_snapshot())
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    print(f"[capture] wrote {SNAPSHOT_PATH}")
    return snap


# ─── Phase 2 — Deterministic scoring ─────────────────────────────────


def _re_extract_from_snapshot(snapshot: dict[str, Any]) -> dict[str, dict[str, list]]:
    """Apply the deterministic sieve to each rendered tome in the
    snapshot. Returns {doc_id: {prompt_class: re_extracted_triples}}."""
    sieve = DeterministicSieve()
    out: dict[str, dict[str, list]] = {}
    for doc_id, doc_data in snapshot["renders"].items():
        out[doc_id] = {}
        for prompt_class, tome in doc_data["by_prompt_class"].items():
            triples = list(sieve.extract_triplets(tome))
            out[doc_id][prompt_class] = [tuple(t) for t in triples]
    return out


def run_path2_v3_bench(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Phase 2: score each doc's (clean re-extract, adversarial re-extract)
    pairs with all four detectors; aggregate per-cell AUC; compute
    bench_digest. Deterministic given the snapshot."""
    rng = random.Random(0)
    print("=" * 72)
    print("Path 2 v3 bench — real-LLM-rendered adversarial perturbations")
    print("=" * 72)

    if snapshot is None:
        snapshot = _ensure_snapshot()

    print("\n[1] Re-extracting triples from snapshot…")
    re_ex = _re_extract_from_snapshot(snapshot)
    print(f"    {len(re_ex)} docs × {len(_PROMPT_CLASSES)} prompt classes")

    # Reconstruct source-triple lists for sheaf training, matching the
    # synthetic bench's vocabulary.
    docs_with_src: list[tuple[str, list[tuple[str, str, str]]]] = []
    for doc_id, doc_data in snapshot["renders"].items():
        src = [tuple(t) for t in doc_data["source_triples"]]
        if src:
            docs_with_src.append((doc_id, src))
    all_triples = [t for _, ts in docs_with_src for t in ts]

    print("\n[2] Training v2.1 sheaf (matches synthetic-bench hyperparams)…")
    trained, embeddings, _ = train_restriction_maps(
        all_triples, stalk_dim=8, epochs=200, learning_rate=0.005,
        margin=0.5, n_negatives_per_positive=3, seed=0,
    )

    print("\n[3] Auto-calibrating λ…")
    per_edge_means: list[float] = []
    for doc_id, source in docs_with_src:
        try:
            doc_sheaf, doc_emb = _build_doc_sheaf(source, trained, embeddings)
            clean = combined_detector_score(doc_sheaf, doc_emb, source)
            per_edge_means.append(clean["v_laplacian"] / max(len(source), 1))
        except (ValueError, KeyError):
            pass
    lambda_auto = float(np.mean(per_edge_means)) if per_edge_means else 0.05
    print(f"    λ_auto = {lambda_auto:.4f}")

    print("\n[4] Per-doc scoring across four detectors × three perturbation classes…")
    # cells[(detector, perturbation_class)] = list[(score, label)]
    cells: dict[tuple[str, str], list[tuple[float, int]]] = {}
    n_with_partition = 0

    for doc_id, source in docs_with_src:
        if len(source) < 2:
            continue
        re_neutral = re_ex.get(doc_id, {}).get("neutral", [])
        if not re_neutral:
            continue
        try:
            doc_sheaf, doc_emb = _build_doc_sheaf(source, trained, embeddings)
        except (ValueError, KeyError):
            continue
        # Treat all source edges as "trusted" for this bench — the
        # adversarial perturbation comes from the LLM render, not from
        # a partition of the source bundle.
        weights = weights_from_receipts(doc_sheaf, trusted_edges=source)
        n_with_partition += 1

        # Score the neutral re-extracted triples as the "clean" pair.
        try:
            clean_v32 = round(score_v32_combined(
                doc_sheaf, doc_emb, re_neutral, weights, lambda_auto, GAMMA,
            ), 6)
            clean_v32_pt = round(score_v32_with_per_triple(
                doc_sheaf, doc_emb, re_neutral, weights, lambda_auto, GAMMA,
                global_sheaf=trained, global_embeddings=embeddings,
            ), 6)
        except Exception:  # noqa: BLE001
            continue
        clean_b1 = round(score_b1_entity_presence_deficit(source, re_neutral), 6)
        clean_b2 = round(score_b2_jaccard_distance(source, re_neutral), 6)

        for cls in ("a1_adversarial", "a2_adversarial", "a4_adversarial"):
            re_adv = re_ex.get(doc_id, {}).get(cls, [])
            if not re_adv:
                continue
            try:
                adv_v32 = round(score_v32_combined(
                    doc_sheaf, doc_emb, re_adv, weights, lambda_auto, GAMMA,
                ), 6)
                adv_v32_pt = round(score_v32_with_per_triple(
                    doc_sheaf, doc_emb, re_adv, weights, lambda_auto, GAMMA,
                    global_sheaf=trained, global_embeddings=embeddings,
                ), 6)
            except Exception:  # noqa: BLE001
                continue
            adv_b1 = round(score_b1_entity_presence_deficit(source, re_adv), 6)
            adv_b2 = round(score_b2_jaccard_distance(source, re_adv), 6)

            short = cls.replace("_adversarial", "")  # "a1", "a2", "a4"
            cells.setdefault(("v32_g0.1", short), []).extend(
                [(clean_v32, 0), (adv_v32, 1)]
            )
            cells.setdefault(("v32_plus_per_triple", short), []).extend(
                [(clean_v32_pt, 0), (adv_v32_pt, 1)]
            )
            cells.setdefault(("b1_entity_presence_deficit", short), []).extend(
                [(clean_b1, 0), (adv_b1, 1)]
            )
            cells.setdefault(("b2_jaccard", short), []).extend(
                [(clean_b2, 0), (adv_b2, 1)]
            )

    print(f"    docs with re-extracted neutral baseline: {n_with_partition}")

    # Component AUCs.
    print("\n[5] Per-cell AUC:")
    component_per_cell_auc: dict[str, float] = {}
    for (det, cls), pairs in sorted(cells.items()):
        component_per_cell_auc[f"{det}|{cls}"] = roc_auc(
            [s for s, _ in pairs], [l for _, l in pairs],
        )

    # Borda hybrid: fuse v32_plus_per_triple with b2 per cell.
    print("\n[6] Borda fusion of (v3.2 + per-triple, b2) per cell…")
    hybrid_per_cell_auc: dict[str, float] = {}
    for cls in ("a1", "a2", "a4"):
        v_pairs = cells.get(("v32_plus_per_triple", cls), [])
        b_pairs = cells.get(("b2_jaccard", cls), [])
        if not v_pairs or not b_pairs:
            continue
        labels_v = [l for _, l in v_pairs]
        labels_b = [l for _, l in b_pairs]
        if labels_v != labels_b:
            continue
        v_scores = [s for s, _ in v_pairs]
        b_scores = [s for s, _ in b_pairs]
        fused = borda_fuse(v_scores, b_scores)
        hybrid_per_cell_auc[f"borda_v32pt_b2|{cls}"] = roc_auc(fused, labels_v)

    all_per_cell = {**component_per_cell_auc, **hybrid_per_cell_auc}
    for k in sorted(all_per_cell):
        print(f"    {k:50s} = {all_per_cell[k]:.3f}")

    print("\n[7] Mean AUC across A1+A2+A4:")
    means: dict[str, float] = {}
    for det in ("v32_g0.1", "v32_plus_per_triple", "b1_entity_presence_deficit",
                "b2_jaccard", "borda_v32pt_b2"):
        aucs = [auc for k, auc in all_per_cell.items() if k.startswith(f"{det}|")]
        if aucs:
            means[det] = sum(aucs) / len(aucs)
            print(f"    {det:35s} = {means[det]:.3f}")

    # Verdict: does the hybrid still beat B2 on real-LLM perturbations?
    delta_borda_vs_b2 = means.get("borda_v32pt_b2", 0.0) - means.get("b2_jaccard", 0.0)
    if delta_borda_vs_b2 >= 0.03:
        verdict = "HYBRID_BEATS_BASELINE_ON_REAL_LLM"
    elif delta_borda_vs_b2 >= -0.02:
        verdict = "HYBRID_TIES_BASELINE_ON_REAL_LLM"
    else:
        verdict = "HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM"
    print(f"\n[8] Δ(borda − b2) = {delta_borda_vs_b2:+.3f} → {verdict}")

    report: dict[str, Any] = {
        "schema": "sum.sheaf_path2_v3_bench.v1",
        "corpus": "seed_long_paragraphs",
        "model": snapshot.get("model", PINNED_MODEL),
        "n_docs_total": len(docs_with_src),
        "n_docs_with_partition": n_with_partition,
        "lambda_auto": lambda_auto,
        "gamma_used": GAMMA,
        "alpha": 1.0,
        "beta": 1.0,
        "detectors": ["v32_g0.1", "v32_plus_per_triple",
                      "b1_entity_presence_deficit", "b2_jaccard",
                      "borda_v32pt_b2"],
        "perturbation_classes": ["a1", "a2", "a4"],
        "per_cell_auc": all_per_cell,
        "mean_auc_by_detector": means,
        "delta_borda_vs_b2_mean": delta_borda_vs_b2,
        "verdict": verdict,
        "method_notes": (
            "Phase 1 (snapshot capture) calls LLM 4× per doc "
            "(neutral + a1 + a2 + a4 adversarial prompts); snapshot "
            "committed to fixtures/bench_renders/. Phase 2 re-extracts "
            "via DeterministicSieve and scores deterministically. "
            "Borda hybrid uses v3.2 + per-triple-V channel composition "
            "(matches the §4.7.1 published WIN composition)."
        ),
    }
    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f'\n  "bench_digest": "{report["bench_digest"]}"')
    return report


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate-snapshot", action="store_true",
        help="Force re-running Phase 1 (LLM calls). Costs API budget.",
    )
    args = parser.parse_args()
    snapshot = _ensure_snapshot(force=args.regenerate_snapshot)
    report = run_path2_v3_bench(snapshot)
    from scripts.research._receipt_paths import resolve_receipt_path
    out = resolve_receipt_path(RECEIPTS_DIR, "path2_v3_bench")
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
