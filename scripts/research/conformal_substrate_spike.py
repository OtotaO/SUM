"""Split-conformal substrate spike — calibrated CIs on real
SUM-shaped data.

Two experiments:

  1. **Synthetic coverage sweep** — verifies the
     finite-sample-coverage guarantee at α ∈ {0.05, 0.10, 0.20}
     across multiple seeds. Empirical coverage must hit 1-α.

  2. **Substrate-shaped triple-quality classifier** — uses the
     deterministic sieve to extract triples from a labeled corpus
     (`seed_v2`, where each document carries `gold_triples`).
     Builds a tiny feature-based predictor of "is this triple in
     the gold set?", calibrates split conformal on a held-out
     fold, reports coverage on a third disjoint fold. Concretely
     demonstrates the wrap-any-predictor pattern at substrate
     scale — the same mechanism applies to slider-axis readouts,
     ridge readouts, or any future per-axiom score.

Receipt: ``sum.split_conformal_spike.v1``.
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


# -- Experiment 1: synthetic coverage sweep ---------------------------


def _experiment_synthetic(
    alphas: list[float], n_per_split: int = 2000, n_seeds: int = 5,
) -> list[dict]:
    from sum_engine_internal.research.conformal import (
        SplitConformal, empirical_coverage, average_interval_width,
    )
    out = []
    for alpha in alphas:
        coverages = []
        widths = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            n = n_per_split * 2
            x = rng.uniform(-5, 5, n)
            y = 2 * x + rng.normal(0, 1, n)
            pred = np.full_like(y, 0.0)
            cal_pred, test_pred = pred[:n_per_split], pred[n_per_split:]
            cal_y, test_y = y[:n_per_split], y[n_per_split:]
            sc = SplitConformal(alpha=alpha)
            sc.calibrate(cal_pred, cal_y)
            intervals = sc.predict_batch(test_pred)
            coverages.append(empirical_coverage(intervals, test_y))
            widths.append(average_interval_width(intervals))
        out.append({
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "n_per_split": n_per_split,
            "n_seeds": n_seeds,
            "empirical_coverage_mean": float(np.mean(coverages)),
            "empirical_coverage_std": float(np.std(coverages)),
            "mean_interval_width": float(np.mean(widths)),
        })
    return out


# -- Experiment 2: substrate-shaped triple-quality classifier ---------


def _triple_features(triple, vocab_subjects, vocab_predicates, vocab_objects):
    """Build a small fixed-length feature vector from a triple. The
    point isn't a great classifier — just a deterministic predictor
    whose errors conformal can quantify."""
    s, p, o = triple.subject, triple.predicate, triple.object
    return np.array([
        len(s), len(p), len(o),
        float(s in vocab_subjects),
        float(p in vocab_predicates),
        float(o in vocab_objects),
        float(s == s.lower()),  # all-lowercase indicator
        float(p == p.lower()),
        float(o == o.lower()),
    ], dtype=np.float64)


def _experiment_substrate(corpus_id: str, alpha: float = 0.1) -> dict:
    """Triple-quality regression on a labeled corpus.

    Target y_i ∈ {0, 1}: 1 if the sieve-extracted triple matches a
    gold_triple, 0 otherwise. Predictor: ridge regression on the
    9-feature vector. Conformal wraps the ridge prediction.
    """
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.conformal import (
        SplitConformal, empirical_coverage, average_interval_width,
    )

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()

    # Build feature matrix + target across all docs
    features = []
    targets = []
    for doc in corpus["documents"]:
        gold = {tuple(t) for t in doc.get("gold_triples", [])}
        if not gold:
            continue
        # Vocabulary per document — tighter feature signal
        vocab_s = {t[0] for t in gold}
        vocab_p = {t[1] for t in gold}
        vocab_o = {t[2] for t in gold}
        for t in sieve.extract_triplets(doc["text"]):
            triple = Triple(*t)
            features.append(_triple_features(triple, vocab_s, vocab_p, vocab_o))
            targets.append(1.0 if t in gold else 0.0)

    if len(features) < 30:
        return {
            "corpus_id": corpus_id,
            "skipped": True,
            "reason": f"too few labeled triples ({len(features)} < 30) for a 3-way split",
        }

    X = np.array(features); y = np.array(targets)

    # Random 3-way split: train (40%) / cal (40%) / test (20%)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X))
    n = len(X)
    n_train = int(0.4 * n); n_cal = int(0.4 * n)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]

    # Ridge regression by hand (closed form) — no sklearn dep.
    X_train = X[train_idx]; y_train = y[train_idx]
    X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
    lam = 0.1
    A = X_train_aug.T @ X_train_aug + lam * np.eye(X_train_aug.shape[1])
    b = X_train_aug.T @ y_train
    w = np.linalg.solve(A, b)

    def predict(X_):
        X_aug = np.hstack([X_, np.ones((len(X_), 1))])
        return X_aug @ w

    cal_pred = predict(X[cal_idx])
    test_pred = predict(X[test_idx])
    cal_y = y[cal_idx]; test_y = y[test_idx]

    sc = SplitConformal(alpha=alpha)
    sc.calibrate(cal_pred, cal_y)
    intervals = sc.predict_batch(test_pred)
    coverage = empirical_coverage(intervals, test_y)
    width = average_interval_width(intervals)

    return {
        "corpus_id": corpus_id,
        "n_total_labeled_triples": len(X),
        "n_train": len(train_idx),
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
        "ridge_lambda": lam,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": float(coverage),
        "mean_interval_width": float(width),
        "predictor_train_mse": float(np.mean((y_train - predict(X_train)) ** 2)),
    }


def _emit_receipt(synthetic, substrate, out_path: Path) -> dict:
    receipt = {
        "schema": "sum.split_conformal_spike.v1",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "experiment_synthetic_coverage_sweep": synthetic,
        "experiment_substrate_triple_quality": substrate,
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
    parser.add_argument("--corpora", default="seed_v2,seed_v1")
    args = parser.parse_args()

    print("=== Experiment 1: synthetic coverage sweep ===")
    syn = _experiment_synthetic([0.05, 0.1, 0.2])
    for r in syn:
        gap = r["empirical_coverage_mean"] - r["target_coverage"]
        flag = "✓" if abs(gap) < 0.03 else "✗"
        print(
            f"  {flag} α={r['alpha']:.2f} target={r['target_coverage']:.2f} "
            f"empirical={r['empirical_coverage_mean']:.4f}±{r['empirical_coverage_std']:.4f} "
            f"width={r['mean_interval_width']:.2f}"
        )

    print()
    print("=== Experiment 2: substrate triple-quality classifier ===")
    sub = []
    for cid in args.corpora.split(","):
        cid = cid.strip()
        r = _experiment_substrate(cid)
        sub.append(r)
        if r.get("skipped"):
            print(f"  {cid}: skipped — {r['reason']}")
            continue
        gap = r["empirical_coverage"] - r["target_coverage"]
        flag = "✓" if abs(gap) < 0.10 else "○"  # looser tolerance — finite-sample
        print(
            f"  {flag} {cid:>10s}: n_train={r['n_train']:3d} n_cal={r['n_cal']:3d} "
            f"n_test={r['n_test']:3d} | target={r['target_coverage']:.2f} "
            f"empirical={r['empirical_coverage']:.3f} width={r['mean_interval_width']:.3f}"
        )

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"split_conformal_spike_{ts}.json"
    else:
        out_path = Path(args.out)
    rec = _emit_receipt(syn, sub, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {rec['receipt_digest']}")


if __name__ == "__main__":
    main()
