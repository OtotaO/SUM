"""Conformal rate-guarantee spike — distribution-free lower bound on a
preservation rate, the certifier shape the slider contract + bench-
hardening T3 want ("fact preservation ≥ X with 1-δ confidence").

Two experiments, both zero-API-cost:

  1. **Synthetic coverage sweep** — verifies the finite-sample
     guarantee: across (true_rate, n, δ, method), the empirical
     coverage of the lower bound must hit ≥ 1-δ. This is the provable
     claim, validated Monte-Carlo with a fixed seed.

  2. **Worked certification on real SUM data** — reuses the
     deterministic sieve on the `seed_v1` corpus (each doc carries
     `gold_triples`); labels each extracted triple preserved (1) /
     spurious (0), then certifies a distribution-free lower bound on
     the per-triple preservation rate. Demonstrates the certifier on
     real SUM-extracted data without any LLM call. Scope-limited by
     corpus size (same caveat as the split-conformal spike).

Receipt: ``sum.conformal_rate_guarantee_spike.v1``.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT_DIR = REPO / "fixtures" / "bench_receipts"
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


def _experiment_synthetic() -> list[dict]:
    from sum_engine_internal.research.conformal import empirical_bound_coverage

    grid = [
        (0.90, 200, 0.05),
        (0.95, 200, 0.10),
        (0.99, 400, 0.05),
        (0.80, 100, 0.20),
    ]
    out = []
    for true_rate, n, delta in grid:
        for method in ("clopper_pearson", "hoeffding"):
            cov = empirical_bound_coverage(
                true_rate, n, delta, method, n_trials=5000, seed=11
            )
            out.append({
                "true_rate": true_rate,
                "n": n,
                "delta": delta,
                "target_coverage": round(1.0 - delta, 4),
                "method": method,
                "empirical_coverage": round(cov, 4),
                "covers": bool(cov >= (1.0 - delta) - 0.01),
            })
    return out


def _experiment_substrate(corpus_id: str = "seed_v1", delta: float = 0.05) -> dict:
    """Certify a lower bound on the per-triple preservation rate of the
    deterministic sieve against gold triples — real SUM data, no LLM."""
    try:
        from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
        from sum_engine_internal.research.conformal import certify_rate

        corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
        with corpus_path.open() as f:
            corpus = json.load(f)
        sieve = DeterministicSieve()

        labels: list[float] = []
        for doc in corpus["documents"]:
            gold = {tuple(t) for t in doc.get("gold_triples", [])}
            if not gold:
                continue
            for t in sieve.extract_triplets(doc["text"]):
                labels.append(1.0 if tuple(t) in gold else 0.0)

        if len(labels) < 10:
            return {"corpus_id": corpus_id, "skipped": True,
                    "reason": f"too few labeled triples ({len(labels)} < 10)"}

        g = certify_rate(labels, delta=delta)
        return {
            "corpus_id": corpus_id,
            "n": g.n,
            "delta": g.delta,
            "confidence": round(g.confidence, 4),
            "method": g.method,
            "point_estimate": round(g.point_estimate, 4),
            "rate_lower_bound": round(g.rate_lower_bound, 4),
            "slack": round(g.slack, 4),
            "reads_as": (
                f"with {g.confidence:.0%} confidence, ≥ {g.rate_lower_bound:.1%} of "
                f"sieve-extracted triples are gold (n={g.n}, within the seed_v1 envelope)"
            ),
        }
    except Exception as exc:  # noqa: BLE001 — spike is best-effort on real data
        return {"corpus_id": corpus_id, "skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


def _emit_receipt(synthetic, substrate, out_path: Path) -> dict:
    receipt = {
        "schema": "sum.conformal_rate_guarantee_spike.v1",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "experiment_synthetic_coverage_sweep": synthetic,
        "experiment_substrate_preservation_rate": substrate,
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
    args = parser.parse_args()

    print("=== Experiment 1: synthetic coverage sweep ===")
    syn = _experiment_synthetic()
    for r in syn:
        flag = "✓" if r["covers"] else "✗"
        print(
            f"  {flag} {r['method']:>16s} p={r['true_rate']:.2f} n={r['n']:>3d} "
            f"δ={r['delta']:.2f} target≥{r['target_coverage']:.2f} "
            f"empirical={r['empirical_coverage']:.4f}"
        )

    print()
    print("=== Experiment 2: real-data preservation-rate certification ===")
    sub = _experiment_substrate()
    if sub.get("skipped"):
        print(f"  skipped — {sub['reason']}")
    else:
        print(f"  {sub['reads_as']}")
        print(
            f"  [{sub['method']}] point={sub['point_estimate']:.4f} "
            f"LCB={sub['rate_lower_bound']:.4f} slack={sub['slack']:.4f}"
        )

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"conformal_rate_guarantee_spike_{ts}.json"
    else:
        out_path = Path(args.out)
    rec = _emit_receipt(syn, sub, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {rec['receipt_digest']}")

    # Non-zero exit if any synthetic coverage check failed — makes the
    # spike a guard, not just a report.
    if not all(r["covers"] for r in syn):
        print("FAIL: a synthetic coverage check fell below 1-δ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
