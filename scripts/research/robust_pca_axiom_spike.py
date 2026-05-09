"""Robust-PCA axiom-corruption spike — Phase A from the deep-research
article §9.1.

Two experiments:

  1. **Synthetic ground-truth recovery** — verifies the PCP/ADMM
     implementation is mathematically correct: feed a known
     low-rank + sparse-corruption matrix; check that PCP recovers
     both components exactly.

  2. **Real-corpus corruption detection** — applies PCP to an
     embedded axiom matrix from real corpora (`seed_long_paragraphs`)
     with controlled corruption injection. Two corruption types:
     (a) off-corpus entities (`junk_X` strings), (b) miswired triples
     using real corpus tokens in random combinations.

Both experiments emit measurements into a `sum.robust_pca_axiom_spike.v1`
receipt. The synthetic case validates the math; the corpus case
honestly reports what the simplest deterministic embedding does and
does not detect — the article's "smallest experiment" framing
applied with PROOF_BOUNDARY discipline.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RECEIPT_DIR = REPO / "fixtures" / "bench_receipts"
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


# -- Experiment 1: synthetic ground-truth -----------------------------


def _experiment_synthetic(sizes: list[tuple[int, int, int]]) -> list[dict]:
    """Build M = L_true + S_true with known rank and sparsity;
    check PCP recovers both exactly."""
    from sum_engine_internal.research.robust_pca import pcp
    out = []
    for n, d, r in sizes:
        rng = np.random.RandomState(42)
        U = rng.randn(n, r); V = rng.randn(r, d)
        L_true = U @ V
        sparsity = 0.05
        mask = rng.rand(n, d) < sparsity
        S_true = np.zeros((n, d))
        S_true[mask] = rng.choice([-5, 5], size=int(mask.sum()))
        M = L_true + S_true

        result = pcp(M, max_iter=500)
        L_err = float(np.linalg.norm(result.L - L_true, "fro") / np.linalg.norm(L_true, "fro"))
        S_err = float(np.linalg.norm(result.S - S_true, "fro") / np.linalg.norm(S_true, "fro"))
        out.append({
            "n": n, "d": d, "r_true": r, "r_recovered": result.rank_estimate,
            "sparsity_true": float(sparsity),
            "sparsity_recovered": float(result.sparsity_estimate),
            "L_relative_error": L_err,
            "S_relative_error": S_err,
            "n_iter": result.n_iter,
            "residual_norm": result.residual_norm,
        })
    return out


# -- Experiment 2: corpus corruption detection ------------------------


def _vocab_embed(triples, subjects, predicates, objects):
    s_idx = {s: i for i, s in enumerate(subjects)}
    p_idx = {p: i + len(s_idx) for i, p in enumerate(predicates)}
    o_idx = {o: i + len(s_idx) + len(p_idx) for i, o in enumerate(objects)}
    d = len(s_idx) + len(p_idx) + len(o_idx)
    M = np.zeros((len(triples), d))
    for i, t in enumerate(triples):
        if t.subject in s_idx: M[i, s_idx[t.subject]] = 1
        if t.predicate in p_idx: M[i, p_idx[t.predicate]] = 1
        if t.object in o_idx: M[i, o_idx[t.object]] = 1
    return M


def _experiment_corpus(corpus_id: str, n_corrupt_per_type: int = 6) -> dict:
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.robust_pca import (
        corruption_score, embed_triples,
    )

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    clean_triples = [
        Triple(*t)
        for doc in corpus["documents"]
        for t in sieve.extract_triplets(doc["text"])
    ]

    rng = np.random.RandomState(42)
    subjects = sorted({t.subject for t in clean_triples})
    predicates = sorted({t.predicate for t in clean_triples})
    objects = sorted({t.object for t in clean_triples})

    # Type (a): off-corpus entities
    off_corpus = [
        Triple(f"junk_{i}", f"glorp_{i % 3}", f"frobozz_{i}")
        for i in range(n_corrupt_per_type)
    ]
    # Type (b): miswired — real corpus tokens in random (s,p,o) combos
    miswired = [
        Triple(
            str(rng.choice(subjects)),
            str(rng.choice(predicates)),
            str(rng.choice(objects)),
        )
        for _ in range(n_corrupt_per_type)
    ]
    all_triples = clean_triples + off_corpus + miswired
    n_clean = len(clean_triples)
    truth_off_corpus = np.zeros(len(all_triples), dtype=bool)
    truth_off_corpus[n_clean:n_clean + n_corrupt_per_type] = True
    truth_miswired = np.zeros(len(all_triples), dtype=bool)
    truth_miswired[n_clean + n_corrupt_per_type:] = True
    truth_any = truth_off_corpus | truth_miswired

    # Try TWO embeddings:
    #   (i) hashed buckets (axiom_embedding.embed_triples)
    #   (ii) corpus-vocab one-hot (built here)
    embeddings = {
        "hashed_buckets_64": embed_triples(all_triples, n_buckets=64),
        "corpus_vocab": _vocab_embed(all_triples, subjects, predicates, objects),
    }

    results = {}
    for emb_name, M in embeddings.items():
        try:
            scores = corruption_score(M, max_iter=300)
        except Exception as e:
            results[emb_name] = {"error": f"PCP failed: {e}"}
            continue

        # Strategy A: top-K by raw score (article's framing)
        order_score = np.argsort(scores)[::-1]
        # Strategy B: top-K by |score - median| (bidirectional anomaly)
        median = float(np.median(scores))
        order_anom = np.argsort(np.abs(scores - median))[::-1]

        K = n_corrupt_per_type * 2  # total injected corrupt
        results[emb_name] = {
            "embedding_shape": list(M.shape),
            "embedding_density_pct": round(float((M != 0).mean()) * 100, 2),
            "score_median": median,
            "clean_score_mean": float(scores[~truth_any].mean()),
            "off_corpus_score_mean": float(scores[truth_off_corpus].mean()),
            "miswired_score_mean": float(scores[truth_miswired].mean()),
            "strategy_top_k_by_score": _eval_strategy(
                order_score[:K], truth_any, K,
            ),
            "strategy_top_k_by_anomaly": _eval_strategy(
                order_anom[:K], truth_any, K,
            ),
        }

    return {
        "corpus_id": corpus_id,
        "n_clean_triples": n_clean,
        "n_off_corpus_injected": n_corrupt_per_type,
        "n_miswired_injected": n_corrupt_per_type,
        "embeddings": results,
    }


def _eval_strategy(top_indices, truth_any, K) -> dict:
    flagged = np.zeros(len(truth_any), dtype=bool)
    flagged[top_indices] = True
    tp = int((flagged & truth_any).sum())
    fp = int((flagged & ~truth_any).sum())
    fn = int((~flagged & truth_any).sum())
    return {
        "k": K,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_at_k": round(tp / max(K, 1), 3),
        "recall": round(tp / max((truth_any.sum()), 1), 3),
    }


# -- Receipt ----------------------------------------------------------


def _emit_receipt(synthetic, corpus_results, out_path: Path) -> dict:
    receipt = {
        "schema": "sum.robust_pca_axiom_spike.v1",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "experiment_synthetic_recovery": synthetic,
        "experiment_corpus_corruption_detection": corpus_results,
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["receipt_digest"] = (
        "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--corpora", default="seed_long_paragraphs,seed_news_briefs",
        help="Comma-separated list of corpus ids under scripts/bench/corpora/",
    )
    args = parser.parse_args()

    print("=== Experiment 1: synthetic recovery ===")
    synthetic = _experiment_synthetic([(50, 80, 3), (200, 200, 5), (500, 500, 10)])
    for r in synthetic:
        print(
            f"  n={r['n']:4d} d={r['d']:4d} r_true={r['r_true']:2d} "
            f"r_recov={r['r_recovered']:2d} L_err={r['L_relative_error']:.4f} "
            f"S_err={r['S_relative_error']:.4f} iter={r['n_iter']}"
        )

    print()
    print("=== Experiment 2: corpus corruption detection ===")
    corpus_results = []
    for corpus_id in args.corpora.split(","):
        corpus_id = corpus_id.strip()
        print(f"  --- {corpus_id} ---")
        r = _experiment_corpus(corpus_id)
        corpus_results.append(r)
        for emb, eb in r["embeddings"].items():
            if "error" in eb:
                print(f"    {emb}: {eb['error']}")
                continue
            score_strat = eb["strategy_top_k_by_score"]
            anom_strat = eb["strategy_top_k_by_anomaly"]
            print(
                f"    {emb:22s}: score-top-K precision={score_strat['precision_at_k']:.2f} "
                f"recall={score_strat['recall']:.2f} | "
                f"anomaly-top-K precision={anom_strat['precision_at_k']:.2f} "
                f"recall={anom_strat['recall']:.2f}"
            )

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"robust_pca_axiom_spike_{ts}.json"
    else:
        out_path = Path(args.out)

    receipt = _emit_receipt(synthetic, corpus_results, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {receipt['receipt_digest']}")


if __name__ == "__main__":
    main()
