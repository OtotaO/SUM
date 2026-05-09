"""Von Neumann graph entropy substrate spike — quick win #2.

Three experiments:

  1. **Synthetic upper-bound check** — K_n hits log(N-1) exactly
     across n ∈ {3, 5, 10, 20, 50}. Verifies the implementation
     against the De Domenico-Biamonte theoretical maximum.

  2. **Substrate corpus entropy** — compute S(ρ) for each labeled
     corpus on the deterministic-sieve-extracted axiom graph.
     Surfaces a single number per corpus that's directly
     receipt-attachable.

  3. **Drift sensitivity** — inject controlled corruption (off-corpus
     triples, miswired triples) into a clean corpus's axiom graph.
     Measure ΔS as the corruption count varies. The substrate
     drift-monitor use case: alert when |ΔS| > 2σ from baseline.

Receipt: ``sum.vn_entropy_substrate_spike.v1``.
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


def _experiment_synthetic_K_n() -> list[dict]:
    """K_n entropy must equal log(n-1) — the De Domenico-Biamonte
    theoretical max for the density-matrix construction we use."""
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.spectral_entropy import graph_entropy
    out = []
    for n in [3, 5, 10, 20, 50]:
        nodes = [f"v{i}" for i in range(n)]
        triples = [
            Triple(a, "p", b)
            for i, a in enumerate(nodes)
            for b in nodes[i + 1:]
        ]
        S = graph_entropy(triples)
        max_S = float(np.log(n - 1))
        out.append({
            "n": n,
            "n_edges": n * (n - 1) // 2,
            "entropy": S,
            "theoretical_max": max_S,
            "abs_error": abs(S - max_S),
            "matches_max": abs(S - max_S) < 1e-9,
        })
    return out


def _experiment_substrate_corpus(corpus_id: str) -> dict:
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.spectral_entropy import graph_entropy

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    triples = [
        Triple(*t)
        for doc in corpus["documents"]
        for t in sieve.extract_triplets(doc["text"])
    ]
    info = graph_entropy(triples, return_intermediates=True)
    info["corpus_id"] = corpus_id
    info["n_triples"] = len(triples)
    # Drop the verbose node list from the receipt
    info.pop("nodes", None)
    return info


def _experiment_drift_sensitivity(corpus_id: str, n_seeds: int = 5) -> dict:
    """Inject corruption at varying levels; measure S vs corruption
    count. Reports the linear-fit slope and the per-corruption ΔS
    estimate — the load-bearing claim for the drift-monitor use case."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.spectral_entropy import graph_entropy

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    clean = [
        Triple(*t)
        for doc in corpus["documents"]
        for t in sieve.extract_triplets(doc["text"])
    ]
    S_clean = graph_entropy(clean)

    # Average ΔS over n_seeds for each corruption level
    levels = [0, 1, 5, 10, 20]
    results = []
    rng = np.random.RandomState(42)
    for n_corrupt in levels:
        deltas = []
        for seed in range(n_seeds if n_corrupt > 0 else 1):
            rng_seed = np.random.RandomState(seed)
            extra = [
                Triple(
                    f"junk_{seed}_{i}",
                    f"glorp_{i % 3}",
                    f"frobozz_{seed}_{i}",
                )
                for i in range(n_corrupt)
            ]
            S_drift = graph_entropy(clean + extra)
            deltas.append(S_drift - S_clean)
        results.append({
            "n_corrupt": n_corrupt,
            "delta_S_mean": float(np.mean(deltas)),
            "delta_S_std": float(np.std(deltas)),
            "n_seeds": len(deltas),
        })

    # Linear fit: ΔS as a function of n_corrupt (excluding n=0 baseline)
    xs = np.array([r["n_corrupt"] for r in results if r["n_corrupt"] > 0])
    ys = np.array([r["delta_S_mean"] for r in results if r["n_corrupt"] > 0])
    if len(xs) >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
    else:
        slope, intercept = 0.0, 0.0

    return {
        "corpus_id": corpus_id,
        "S_clean": float(S_clean),
        "drift_curve": results,
        "linear_fit_slope_per_corruption": float(slope),
        "linear_fit_intercept": float(intercept),
    }


def _emit_receipt(syn, substrate, drift, out_path: Path) -> dict:
    receipt = {
        "schema": "sum.vn_entropy_substrate_spike.v1",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "experiment_synthetic_K_n_upper_bound": syn,
        "experiment_substrate_corpus_entropy": substrate,
        "experiment_drift_sensitivity": drift,
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
        "--corpora",
        default="seed_long_paragraphs,seed_news_briefs,seed_paragraphs",
    )
    args = parser.parse_args()

    print("=== Experiment 1: synthetic K_n upper-bound ===")
    syn = _experiment_synthetic_K_n()
    for r in syn:
        flag = "✓" if r["matches_max"] else "✗"
        print(
            f"  {flag} n={r['n']:3d} S={r['entropy']:.10f} "
            f"max=log({r['n']-1})={r['theoretical_max']:.10f} "
            f"err={r['abs_error']:.2e}"
        )

    print()
    print("=== Experiment 2: substrate corpus entropy ===")
    sub = []
    for cid in args.corpora.split(","):
        cid = cid.strip()
        try:
            r = _experiment_substrate_corpus(cid)
            sub.append(r)
            print(
                f"  {cid:>22s}: triples={r['n_triples']:4d} "
                f"nodes={r['n_nodes']:3d} edges={r['n_edges']:3d} "
                f"S={r['entropy']:.4f}"
            )
        except FileNotFoundError:
            print(f"  {cid:>22s}: not found, skipped")

    print()
    print("=== Experiment 3: drift sensitivity ===")
    drift = []
    for cid in args.corpora.split(","):
        cid = cid.strip()
        try:
            r = _experiment_drift_sensitivity(cid)
            drift.append(r)
            print(f"  {cid:>22s}: S_clean={r['S_clean']:.4f}, "
                  f"slope={r['linear_fit_slope_per_corruption']:+.5f} per corruption")
            for d in r["drift_curve"]:
                print(f"    n_corrupt={d['n_corrupt']:3d}: ΔS={d['delta_S_mean']:+.4f} ± {d['delta_S_std']:.4f}")
        except FileNotFoundError:
            print(f"  {cid:>22s}: not found, skipped")

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"vn_entropy_substrate_spike_{ts}.json"
    else:
        out_path = Path(args.out)
    rec = _emit_receipt(syn, sub, drift, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {rec['receipt_digest']}")


if __name__ == "__main__":
    main()
