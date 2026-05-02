"""Path 2 — real LLM-rendered adversarial bench for the v2.x detector.

Stress-tests v2.x against naturalistic LLM hallucinations rather
than synthetic perturbations. For each of the 16 docs in
seed_long_paragraphs:

  1. Sieve-extract source triples T_src from the doc's prose.
  2. Render T_src via the hosted Worker at three slider configurations:
       - neutral baseline (no LLM call; canonical-deterministic path)
       - moderate stress: length=0.7, formality=0.5 (LLM-conditioned)
       - high stress:     length=0.9, formality=0.1 (LLM-conditioned,
         the slider-bench's documented weak cell where catastrophic
         outliers historically appeared pre-v0.7)
  3. Sieve-extract triples T'_n from each rendered tome.
  4. Score with the v2.x detector trained on the union vocabulary.
  5. Compare V_clean (synthetic Path 1 baseline) to V_llm (this bench)
     to see if naturalistic LLM-rendered text drifts more or less
     than synthetic perturbations.

Each Worker call returns a signed render_receipt.v1 — saved to the
receipt directory as durable artifacts that can later seed v3's
harmonic-extension boundary set.

Cost: ~32 LLM calls × ~200 tokens output = ~6K tokens at
Claude-Haiku-4.5 rates (~$1/1M) = ~$0.01-$0.05 total.

Reproducibility note: LLM-conditioned renders are NOT deterministic
across calls (cache hit/miss + Anthropic-side stochasticity).
Pinning expected V values would be brittle; we report distributions
instead.
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    train_restriction_maps,
    score_rendered_triples_v2,
    combined_detector_score,
)
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve


CORPUS_PATH = REPO / "scripts" / "bench" / "corpora" / "seed_long_paragraphs.json"
WORKER_URL = "https://sum-demo.ototao.workers.dev"
RECEIPT_DIR = REPO / "fixtures" / "bench_receipts"

SLIDER_CONFIGS = {
    "neutral":         {"density": 1.0, "length": 0.5, "formality": 0.5, "audience": 0.5, "perspective": 0.5},
    "moderate_stress": {"density": 1.0, "length": 0.7, "formality": 0.5, "audience": 0.5, "perspective": 0.5},
    "high_stress":     {"density": 1.0, "length": 0.9, "formality": 0.1, "audience": 0.5, "perspective": 0.5},
}


def post_render(triples: list[tuple[str, str, str]], slider: dict) -> dict:
    """POST to /api/render and return parsed JSON. Raises RuntimeError on error."""
    payload = json.dumps({
        "triples": [list(t) for t in triples],
        "slider_position": slider,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{WORKER_URL}/api/render",
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            # Cloudflare in front of the Worker rejects the default
            # Python urllib UA with HTTP 403 / error 1010. Set a
            # recognisable identifier so the request is treated as
            # a legitimate scripted client.
            "user-agent": "sum-engine-bench/0.5.0 (+https://github.com/OtotaO/SUM)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} from worker: {body[:200]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e.reason}") from e


def main() -> dict[str, Any]:
    print("=" * 72)
    print("Path 2 — real LLM-rendered adversarial bench (v2.x detector)")
    print("=" * 72)

    print("\n[1] Loading corpus + sieve-extracting…")
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    docs_raw = data["documents"]
    sieve = DeterministicSieve()
    docs: list[tuple[str, str, list[tuple[str, str, str]]]] = []
    for d in docs_raw:
        triples = list(sieve.extract_triplets(d["text"]))
        if triples:
            docs.append((d["id"], d["text"], triples))
    print(f"    {len(docs)} docs with non-empty extractions")
    total_triples = sum(len(t) for _, _, t in docs)
    print(f"    {total_triples} source triples total")

    print("\n[2] Training v2.1 sheaf on union vocabulary…")
    all_triples = [t for _, _, ts in docs for t in ts]
    trained, embeddings, _ = train_restriction_maps(
        all_triples, stalk_dim=8, epochs=200, learning_rate=0.005,
        margin=0.5, n_negatives_per_positive=3, seed=0,
    )
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    print(f"    vocab: {len(all_entities)} entities, {len(trained.relations)} relations")

    print("\n[3] Rendering each doc at 3 slider configs via the live Worker…")
    print(f"    Worker: {WORKER_URL}")
    print(f"    Configs: {list(SLIDER_CONFIGS.keys())}")
    print(f"    Total expected LLM calls: ~{len(docs) * 2} "
          f"(neutral is canonical-deterministic, no LLM)")

    per_config: dict[str, list[dict]] = {k: [] for k in SLIDER_CONFIGS}
    receipts: list[dict] = []
    n_llm_calls = 0
    n_failed = 0

    t_start = time.monotonic()
    for i, (doc_id, _text, source_triples) in enumerate(docs):
        for cfg_name, slider in SLIDER_CONFIGS.items():
            try:
                resp = post_render(source_triples, slider)
            except RuntimeError as e:
                print(f"    [{i+1}/{len(docs)}] {doc_id} {cfg_name}: FAILED ({e})")
                n_failed += 1
                continue

            tome = resp.get("tome", "")
            llm_calls = resp.get("llm_calls_made", 0)
            n_llm_calls += llm_calls
            cache_status = resp.get("cache_status", "?")
            receipt = resp.get("render_receipt")
            if receipt is not None:
                receipts.append({
                    "doc_id": doc_id,
                    "config": cfg_name,
                    "render_id": resp.get("render_id"),
                    "cache_status": cache_status,
                    "model": receipt.get("payload", {}).get("model"),
                    "kid": receipt.get("kid"),
                    "tome_first_120": tome[:120],
                })

            # Re-extract from rendered tome
            re_extracted = list(sieve.extract_triplets(tome))

            # Score against trained sheaf using doc-specific source graph
            try:
                doc_sheaf = KnowledgeSheafV2.from_triples(source_triples, stalk_dim=8)
                doc_emb = np.zeros((len(doc_sheaf.vertices), 8), dtype=np.float64)
                for j, v in enumerate(doc_sheaf.vertices):
                    if v in trained.vertex_index:
                        doc_emb[j] = embeddings[trained.vertex_index[v]]
                # Auto-calibrated lambda for this doc
                from sum_engine_internal.research.sheaf_laplacian_v2 import laplacian_quadratic_form_v2, cochain_one_hot_v2
                clean_x = cochain_one_hot_v2(doc_sheaf, source_triples, embedding=doc_emb)
                clean_lap = laplacian_quadratic_form_v2(doc_sheaf, clean_x)
                lam = clean_lap / max(len(source_triples), 1)
                combined = combined_detector_score(doc_sheaf, doc_emb, re_extracted, presence_weight=lam)
                per_triple = score_rendered_triples_v2(trained, embeddings, re_extracted)
            except (ValueError, KeyError) as e:
                print(f"    [{i+1}/{len(docs)}] {doc_id} {cfg_name}: scoring error ({e})")
                continue

            per_config[cfg_name].append({
                "doc_id": doc_id,
                "n_source_triples": len(source_triples),
                "n_re_extracted": len(re_extracted),
                "tome_len": len(tome),
                "v_combined": combined["v_combined"],
                "v_laplacian": combined["v_laplacian"],
                "v_deficit": combined["v_deficit"],
                "presence_deficit_count": combined["presence_deficit_count"],
                "lambda_used": lam,
                "max_in_vocab_v": per_triple["max_in_vocab_v"],
                "n_oov": per_triple["n_oov"],
                "model": resp.get("render_receipt", {}).get("payload", {}).get("model", "?"),
                "llm_calls": llm_calls,
                "cache_status": cache_status,
            })
            print(f"    [{i+1}/{len(docs)}] {doc_id} {cfg_name:<16} "
                  f"V_lap={combined['v_laplacian']:>7.3f}  "
                  f"V_def={combined['v_deficit']:>6.3f}  "
                  f"V_combined={combined['v_combined']:>7.3f}  "
                  f"n_oov={per_triple['n_oov']}  "
                  f"model={resp.get('render_receipt', {}).get('payload', {}).get('model', '?')[:18]}")

    elapsed = time.monotonic() - t_start
    print(f"\n    completed in {elapsed:.1f}s; {n_llm_calls} LLM calls; {n_failed} failures")

    # ── Aggregate per config ───────────────────────────────────────────
    print("\n[4] Per-config aggregate stats:")
    print(f"    {'config':<18} {'n_docs':>8} {'V_lap_mean':>11} {'V_combined_mean':>16} "
          f"{'oov_mean':>10} {'deficit_mean':>14}")
    print("    " + "-" * 80)
    config_summary: dict[str, dict] = {}
    for cfg_name, rows in per_config.items():
        if not rows:
            print(f"    {cfg_name:<18} 0 docs (worker unavailable?)")
            config_summary[cfg_name] = {"n": 0}
            continue
        n = len(rows)
        v_lap = np.array([r["v_laplacian"] for r in rows])
        v_comb = np.array([r["v_combined"] for r in rows])
        deficit = np.array([r["presence_deficit_count"] for r in rows])
        oov = np.array([r["n_oov"] for r in rows])
        config_summary[cfg_name] = {
            "n": n,
            "v_laplacian_mean": float(v_lap.mean()),
            "v_laplacian_std": float(v_lap.std()),
            "v_combined_mean": float(v_comb.mean()),
            "v_combined_std": float(v_comb.std()),
            "deficit_mean": float(deficit.mean()),
            "oov_mean": float(oov.mean()),
        }
        print(f"    {cfg_name:<18} {n:>8} {v_lap.mean():>11.3f} {v_comb.mean():>16.3f} "
              f"{oov.mean():>10.2f} {deficit.mean():>14.2f}")

    # ── Per-doc deltas: how much does naturalistic LLM rendering drift? ──
    print("\n[5] Per-doc V_combined deltas (high_stress vs neutral):")
    if per_config["neutral"] and per_config["high_stress"]:
        neutral_by_doc = {r["doc_id"]: r for r in per_config["neutral"]}
        stress_by_doc = {r["doc_id"]: r for r in per_config["high_stress"]}
        common = sorted(set(neutral_by_doc) & set(stress_by_doc))
        deltas = []
        print(f"    {'doc_id':<35} {'V_neutral':>10} {'V_stress':>10} {'Δ':>8}")
        print("    " + "-" * 70)
        for doc_id in common:
            n = neutral_by_doc[doc_id]["v_combined"]
            s = stress_by_doc[doc_id]["v_combined"]
            d = s - n
            deltas.append(d)
            print(f"    {doc_id:<35} {n:>10.3f} {s:>10.3f} {d:>+8.3f}")
        if deltas:
            print(f"\n    mean Δ (high_stress − neutral): {np.mean(deltas):+.3f}")
            print(f"    docs where stress increases V: "
                  f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}")

    return {
        "schema": "sum.sheaf_v2_path2_llm_bench.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs": len(docs),
        "n_source_triples": total_triples,
        "vocab_size_entities": len(all_entities),
        "vocab_size_relations": len(trained.relations),
        "stalk_dim": 8,
        "training_epochs": 200,
        "worker_url": WORKER_URL,
        "slider_configs": SLIDER_CONFIGS,
        "n_llm_calls": n_llm_calls,
        "n_failed": n_failed,
        "elapsed_seconds": elapsed,
        "per_config": per_config,
        "config_summary": config_summary,
        "render_receipts_summary": receipts,
    }


if __name__ == "__main__":
    result = main()
    out_path = REPO / "fixtures" / "bench_receipts" / "sheaf_v2_path2_llm_bench_2026-05-01.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\n[6] Receipt saved: {out_path}")
