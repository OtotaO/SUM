"""Multiplier bootstrap substrate spike — research arc PR #1.

Three experiments demonstrating distribution-free CIs on
substrate-relevant scalar / spectral statistics. Compounds with
PRs #183 (split conformal) and #184 (vN entropy) — the bootstrap
gives the same scalars CIs the conformal kernel can wrap.

  1. **Synthetic mean-coverage** — verifies the CCK 2013
     coverage guarantee on a textbook setup. Empirical coverage
     of bootstrap-CI-on-the-mean tracks 1-α.

  2. **Eigenvalue CIs on substrate axiom-graph Laplacian** —
     bootstrap the spectrum of the graph Laplacian built from
     real corpus axioms. Each eigenvalue gets a CI; replaces
     "λ_2 < threshold" decisions with statistically-grounded
     intervals.

  3. **CI on von Neumann entropy** — bootstrap the per-corpus
     entropy from PR #184. Surfaces the natural follow-on:
     "S(corpus) ∈ [4.65, 4.85] @ 95%" instead of bare scalar.

Receipt: ``sum.multiplier_bootstrap_substrate_spike.v1``.
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


# -- Experiment 1: synthetic mean coverage ----------------------------


def _experiment_synthetic_coverage(
    alphas: list[float], n_trials: int = 100, n_per: int = 200, B: int = 300,
) -> list[dict]:
    from sum_engine_internal.research.bootstrap import (
        bootstrap_ci, multiplier_bootstrap,
    )
    out = []
    for alpha in alphas:
        true_mean = 3.0
        rng = np.random.default_rng(42)
        hits = 0
        widths = []
        for _ in range(n_trials):
            x = rng.normal(true_mean, 1.0, size=n_per).reshape(-1, 1)
            point, reps = multiplier_bootstrap(
                x, lambda s: np.array([s.mean()]), B=B,
                rng=np.random.default_rng(),
            )
            iv = bootstrap_ci(point, reps, alpha=alpha)[0]
            if iv.contains(true_mean):
                hits += 1
            widths.append(iv.width)
        out.append({
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "n_trials": n_trials,
            "n_per_trial": n_per,
            "B": B,
            "empirical_coverage": hits / n_trials,
            "mean_interval_width": float(np.mean(widths)),
            "se": float(np.sqrt(alpha * (1 - alpha) / n_trials)),
        })
    return out


# -- Experiment 2: eigenvalue CIs on substrate Laplacian --------------


def _experiment_eigenvalue_cis(corpus_id: str, n_top: int = 5, B: int = 300) -> dict:
    """Bootstrap the top-k eigenvalues of the substrate axiom
    graph's Laplacian. Each eigenvalue gets a CI."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.bootstrap import (
        bootstrap_ci, multiplier_bootstrap,
    )
    from sum_engine_internal.research.spectral_entropy.vn_entropy import (
        build_axiom_graph, normalized_laplacian,
    )

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    triples = [
        Triple(*t)
        for doc in corpus["documents"]
        for t in sieve.extract_triplets(doc["text"])
    ]
    nodes, A = build_axiom_graph(triples)
    if len(nodes) < n_top + 2:
        return {"corpus_id": corpus_id, "skipped": True,
                "reason": f"too few nodes ({len(nodes)}) for top-{n_top} eigvals"}

    L_full = normalized_laplacian(A)
    full_eigs = np.linalg.eigvalsh(L_full)[::-1][:n_top]

    # Bootstrap setup: rows of "samples" = nodes; statistic_fn
    # rebuilds the Laplacian on the row-resampled adjacency.
    # We treat the adjacency rows as the iid observations — each
    # row encodes one node's connectivity vector.
    rows = A  # shape (n_nodes, n_nodes)

    def top_k_eigvals(rows_subset):
        # Row-resampled adjacency may not be symmetric; symmetrize
        A_sym = (rows_subset + rows_subset.T) / 2
        d = A_sym.sum(axis=1)
        L = np.diag(d) - A_sym
        eigs = np.linalg.eigvalsh(L)[::-1]
        # Pad if fewer than n_top eigvals (small bootstrap subsample)
        if len(eigs) >= n_top:
            return eigs[:n_top]
        out = np.zeros(n_top)
        out[: len(eigs)] = eigs
        return out

    point, reps = multiplier_bootstrap(rows, top_k_eigvals, B=B)
    intervals = bootstrap_ci(point, reps, alpha=0.10)

    return {
        "corpus_id": corpus_id,
        "n_nodes": len(nodes),
        "n_edges": int(A.sum() / 2),
        "B": B,
        "alpha": 0.10,
        "full_top_k_eigvals": [float(e) for e in full_eigs],
        "bootstrap_intervals": [
            {"point": iv.point, "lower": iv.lower, "upper": iv.upper, "width": iv.width}
            for iv in intervals
        ],
    }


# -- Experiment 3: CI on von Neumann entropy --------------------------


def _experiment_vn_entropy_ci(corpus_id: str, B: int = 300) -> dict:
    """Bootstrap the per-corpus von Neumann entropy."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.bootstrap import (
        bootstrap_ci, multiplier_bootstrap,
    )
    from sum_engine_internal.research.spectral_entropy import graph_entropy
    from sum_engine_internal.research.spectral_entropy.vn_entropy import (
        build_axiom_graph,
    )

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    triples = [
        Triple(*t)
        for doc in corpus["documents"]
        for t in sieve.extract_triplets(doc["text"])
    ]
    nodes, A = build_axiom_graph(triples)
    if len(nodes) < 4:
        return {"corpus_id": corpus_id, "skipped": True}

    full_S = graph_entropy(triples)

    # Bootstrap: row-resample the adjacency, recompute entropy
    def entropy_from_rows(rows_subset):
        A_sym = (rows_subset + rows_subset.T) / 2
        d = A_sym.sum(axis=1)
        if d.sum() <= 0:
            return np.array([0.0])
        L = np.diag(d) - A_sym
        rho = L / np.trace(L)
        eigs = np.linalg.eigvalsh(rho)
        eigs = np.clip(eigs, 0.0, None)
        nz = eigs[eigs > 1e-12]
        if nz.size == 0:
            return np.array([0.0])
        return np.array([float(-np.sum(nz * np.log(nz)))])

    point, reps = multiplier_bootstrap(A, entropy_from_rows, B=B)
    iv = bootstrap_ci(point, reps, alpha=0.10)[0]

    return {
        "corpus_id": corpus_id,
        "n_nodes": len(nodes),
        "n_triples": len(triples),
        "B": B,
        "alpha": 0.10,
        "full_S": float(full_S),
        "bootstrap_S_mean": float(point[0]),
        "interval_lower": iv.lower,
        "interval_upper": iv.upper,
        "interval_width": iv.width,
    }


# -- Receipt ----------------------------------------------------------


def _emit_receipt(syn, eigvals, vn, out_path: Path) -> dict:
    receipt = {
        "schema": "sum.multiplier_bootstrap_substrate_spike.v1",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "experiment_synthetic_coverage": syn,
        "experiment_eigenvalue_cis": eigvals,
        "experiment_vn_entropy_ci": vn,
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
    )
    args = parser.parse_args()

    print("=== Experiment 1: synthetic mean coverage ===")
    syn = _experiment_synthetic_coverage([0.05, 0.10, 0.20])
    for r in syn:
        gap = r["empirical_coverage"] - r["target_coverage"]
        # ~3σ band per-trial Monte Carlo
        flag = "✓" if abs(gap) < 3 * r["se"] else "✗"
        print(
            f"  {flag} α={r['alpha']:.2f} target={r['target_coverage']:.2f} "
            f"empirical={r['empirical_coverage']:.3f} ±{r['se']:.3f} "
            f"width={r['mean_interval_width']:.3f}"
        )

    print()
    print("=== Experiment 2: eigenvalue CIs on substrate Laplacian ===")
    eigvals = []
    for cid in args.corpora.split(","):
        cid = cid.strip()
        try:
            r = _experiment_eigenvalue_cis(cid)
            eigvals.append(r)
            if r.get("skipped"):
                print(f"  {cid}: {r['reason']}")
                continue
            print(f"  {cid:>22s}: n_nodes={r['n_nodes']} α={r['alpha']}")
            for i, iv in enumerate(r["bootstrap_intervals"]):
                full = r["full_top_k_eigvals"][i]
                print(
                    f"    λ_{i+1}: full={full:.4f}  CI=[{iv['lower']:.4f}, "
                    f"{iv['upper']:.4f}]  width={iv['width']:.4f}"
                )
        except FileNotFoundError:
            print(f"  {cid}: not found")

    print()
    print("=== Experiment 3: CI on von Neumann entropy ===")
    vn = []
    for cid in args.corpora.split(","):
        cid = cid.strip()
        try:
            r = _experiment_vn_entropy_ci(cid)
            vn.append(r)
            if r.get("skipped"):
                print(f"  {cid}: skipped (too few nodes)")
                continue
            print(
                f"  {cid:>22s}: full_S={r['full_S']:.4f} "
                f"bootstrap_mean_S={r['bootstrap_S_mean']:.4f} "
                f"CI=[{r['interval_lower']:.4f}, {r['interval_upper']:.4f}] "
                f"width={r['interval_width']:.4f}"
            )
        except FileNotFoundError:
            print(f"  {cid}: not found")

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"multiplier_bootstrap_substrate_spike_{ts}.json"
    else:
        out_path = Path(args.out)
    rec = _emit_receipt(syn, eigvals, vn, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {rec['receipt_digest']}")


if __name__ == "__main__":
    main()
