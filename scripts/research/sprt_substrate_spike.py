"""SPRT substrate spike — adaptive stopping for round-trip iterations.

Two experiments:

  1. **Synthetic Wald error-bound verification** — across a sweep
     of (p_0, p_1) pairs and (α, β) settings, verify Type-I and
     Type-II error rates stay within the Wald bounds.

  2. **Substrate-shaped budget reduction simulation** — model the
     SUM round-trip extraction as a Bernoulli stream where each
     iteration emits 1 if it improved faithfulness above τ and 0
     otherwise. Compare fixed-N=8 (the substrate's current
     baseline) against SPRT-adaptive at the same operator-chosen
     (α, β). Report mean LLM-call reduction.

Receipt: ``sum.sprt_substrate_spike.v1``.
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


# -- Experiment 1: synthetic Wald-bound verification -----------------


def _experiment_synthetic_wald_bounds(
    settings: list[tuple[float, float, float, float]],
    n_trials: int = 1000,
    n_per_trial: int = 200,
) -> list[dict]:
    from sum_engine_internal.research.sequential import (
        BinomialSPRT, SPRTDecision,
    )
    out = []
    rng = np.random.default_rng(42)
    for p0, p1, alpha, beta in settings:
        for true_p, scenario in [(p0, "under_H0"), (p1, "under_H1")]:
            accept = reject = no_dec = 0
            sample_sizes = []
            for _ in range(n_trials):
                sprt = BinomialSPRT(p0=p0, p1=p1, alpha=alpha, beta=beta)
                obs = rng.binomial(1, true_p, size=n_per_trial)
                state = sprt.run_until_decision(obs)
                sample_sizes.append(state.n)
                if state.decision == SPRTDecision.ACCEPT_H0:
                    accept += 1
                elif state.decision == SPRTDecision.REJECT_H0:
                    reject += 1
                else:
                    no_dec += 1

            err_count = reject if scenario == "under_H0" else accept
            err_bound = alpha if scenario == "under_H0" else beta
            err_rate = err_count / n_trials
            out.append({
                "p0": p0, "p1": p1, "alpha": alpha, "beta": beta,
                "scenario": scenario, "true_p": true_p,
                "n_trials": n_trials,
                "error_rate": err_rate,
                "error_bound": err_bound,
                "within_bound_with_3sigma": bool(
                    err_rate <= err_bound + 3 * np.sqrt(
                        err_bound * (1 - err_bound) / n_trials
                    )
                ),
                "mean_sample_size": float(np.mean(sample_sizes)),
                "median_sample_size": float(np.median(sample_sizes)),
                "n_per_trial_cap": n_per_trial,
                "no_decision_rate": no_dec / n_trials,
            })
    return out


# -- Experiment 2: substrate budget reduction ------------------------


def _experiment_substrate_budget(
    fixed_n: int = 8,
    n_trials: int = 1000,
    p0: float = 0.5,
    p1_settings: list[float] = (0.3, 0.7, 0.85),
    alpha: float = 0.05,
    beta: float = 0.05,
) -> list[dict]:
    """Compare fixed-N=8 vs SPRT for a few effect sizes. The
    substrate's current round-trip extraction loop runs a fixed
    number of iterations per document; SPRT could stop earlier
    when the per-iteration faithfulness signal is decisive.

    For each true_p ∈ p1_settings, simulate a stream of binary
    outcomes (each iteration improves vs not) and compare fixed-N
    vs SPRT-adaptive on (a) error rate, (b) mean sample size."""
    from sum_engine_internal.research.sequential import (
        BinomialSPRT, SPRTDecision,
    )
    out = []
    rng = np.random.default_rng(42)
    # The "fair comparison" is at p1 ∈ p1_settings — we want to
    # make a decision about whether the true rate is closer to p0
    # or p1. SPRT's H_0 / H_1 are anchored at p0 and the operator's
    # chosen p1.
    for p1_target in p1_settings:
        if abs(p1_target - p0) < 1e-9:
            continue
        for true_p in (p0, p1_target):
            sprt_decisions = []
            sprt_sizes = []
            fixed_decisions = []
            for _ in range(n_trials):
                sprt = BinomialSPRT(p0=p0, p1=p1_target, alpha=alpha, beta=beta)
                # Generate enough observations for both schemes
                obs = rng.binomial(1, true_p, size=max(fixed_n, 100))

                # SPRT: stop on first decision
                state = sprt.run_until_decision(obs[:100])
                sprt_decisions.append(state.decision)
                sprt_sizes.append(state.n)

                # Fixed-N: take fixed_n samples; classify by
                # majority — if > p_threshold, "favours H_1"
                p_threshold = (p0 + p1_target) / 2
                obs_fixed = obs[:fixed_n]
                emp_p = obs_fixed.mean()
                if emp_p >= p_threshold:
                    fixed_decisions.append(SPRTDecision.REJECT_H0)
                else:
                    fixed_decisions.append(SPRTDecision.ACCEPT_H0)

            scenario = "under_H0" if abs(true_p - p0) < 1e-9 else "under_H1"
            sprt_err = (
                sum(1 for d in sprt_decisions if d == SPRTDecision.REJECT_H0)
                if scenario == "under_H0"
                else sum(1 for d in sprt_decisions if d == SPRTDecision.ACCEPT_H0)
            ) / n_trials
            fixed_err = (
                sum(1 for d in fixed_decisions if d == SPRTDecision.REJECT_H0)
                if scenario == "under_H0"
                else sum(1 for d in fixed_decisions if d == SPRTDecision.ACCEPT_H0)
            ) / n_trials
            sprt_mean = float(np.mean(sprt_sizes))
            sprt_no_decision = sum(
                1 for d in sprt_decisions if d == SPRTDecision.CONTINUE
            ) / n_trials
            savings = 1 - sprt_mean / fixed_n if fixed_n > 0 else 0.0
            out.append({
                "p0": p0, "p1": p1_target, "true_p": true_p,
                "scenario": scenario,
                "alpha": alpha, "beta": beta,
                "fixed_n": fixed_n,
                "sprt_mean_n": sprt_mean,
                "sprt_no_decision_rate": sprt_no_decision,
                "fixed_n_error_rate": fixed_err,
                "sprt_error_rate": sprt_err,
                "sample_size_reduction_pct": savings * 100,
            })
    return out


# -- Receipt ----------------------------------------------------------


def _emit_receipt(syn, sub, out_path: Path) -> dict:
    receipt = {
        "schema": "sum.sprt_substrate_spike.v1",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "experiment_synthetic_wald_bounds": syn,
        "experiment_substrate_budget_reduction": sub,
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

    print("=== Experiment 1: synthetic Wald-bound verification ===")
    settings = [
        (0.5, 0.8, 0.05, 0.05),
        (0.5, 0.7, 0.05, 0.05),
        (0.5, 0.6, 0.05, 0.05),
        (0.7, 0.95, 0.10, 0.10),
    ]
    syn = _experiment_synthetic_wald_bounds(settings, n_trials=500)
    for r in syn:
        flag = "✓" if r["within_bound_with_3sigma"] else "✗"
        print(
            f"  {flag} p0={r['p0']:.2f} p1={r['p1']:.2f} α={r['alpha']:.2f} "
            f"β={r['beta']:.2f} {r['scenario']:>9s}: err={r['error_rate']:.3f} "
            f"≤{r['error_bound']:.2f}  mean_n={r['mean_sample_size']:.1f}"
        )

    print()
    print("=== Experiment 2: substrate budget reduction (vs fixed-N=8) ===")
    sub = _experiment_substrate_budget(fixed_n=8)
    for r in sub:
        flag = "✓" if r["sprt_no_decision_rate"] < 0.05 else "○"
        print(
            f"  {flag} p0={r['p0']:.2f} p1={r['p1']:.2f} true_p={r['true_p']:.2f} "
            f"({r['scenario']:>9s})  fixed_err={r['fixed_n_error_rate']:.3f} "
            f"sprt_err={r['sprt_error_rate']:.3f}  sprt_mean_n={r['sprt_mean_n']:.1f} "
            f"savings={r['sample_size_reduction_pct']:+.0f}%"
        )

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"sprt_substrate_spike_{ts}.json"
    else:
        out_path = Path(args.out)
    rec = _emit_receipt(syn, sub, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {rec['receipt_digest']}")


if __name__ == "__main__":
    main()
