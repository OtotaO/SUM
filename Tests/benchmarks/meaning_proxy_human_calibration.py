"""Calibrate SUM's meaning-loss proxy against HUMAN judgments (SummEval).

WHY THIS EXISTS (fresh-eyes audit, 2026-06-16). PROOF_BOUNDARY.md tagged
"the named proxy tracks human-perceived meaning-loss" as [empirical-benchmark]
(= measured on a fixed corpus) — but no such measurement existed anywhere in the
repo. The only proxy-vs-human artifact was a 4-row single-author anecdote (F18).
This is the missing measurement: correlate each shipped scorer's meaning-loss
against the expert human ratings in SummEval (Fabbri et al., 2021), on a fixed,
public, human-rated corpus, and report the number honestly — whatever it is.

WHAT IT MEASURES. SummEval rates 16 machine summaries of each CNN/DM article on
four 1-5 axes (averaged expert annotations): consistency (faithfulness — no
hallucination), relevance (captures important content), coherence, fluency. SUM's
meaning-loss = 1 - (w_recall*recall + w_fidelity*fidelity): recall ~ source
coverage (relevance-like), fidelity ~ no unsupported content (consistency-like).
So we correlate proxy LOSS against a human "loss" = (5 - score)/4 ∈ [0,1] and
expect a POSITIVE rank correlation. The faithfulness-aligned composite is
0.6*relevance + 0.4*consistency mapped to loss (mirrors the scorer's weights).

HONEST SCOPE. (1) This is summary-level POOLED rank correlation across articles —
the standard meta-evaluation number, disclosed as such. (2) It uses a SUBSAMPLE
(deterministic, sorted-by-id prefix) sized to the judge's CPU cost; n is recorded
in the artifact. (3) It measures the EXACT shipped scorers (LexicalCoverageScorer,
embedding_entailment_scorer, nli_entailment_scorer) — no reimplementation — so the
number is what the certified proxy actually is. (4) A model judge is machine-
pinned; this is a measurement, not a cross-machine guarantee.

Run (downloads SummEval + the judge models on first use):
    python Tests/benchmarks/meaning_proxy_human_calibration.py \\
        --fast-articles 50 --nli-articles 12 --out Tests/benchmarks/meaning_proxy_human_calibration.result.json

License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from statistics import mean


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _human_loss(score_1_5):
    """Map a 1-5 human quality score to a [0,1] 'loss' (higher = worse), so it is
    rank-comparable to the proxy's meaning-loss."""
    s = _to_float(score_1_5)
    if s is None:
        return None
    return max(0.0, min(1.0, (5.0 - s) / 4.0))


def _spearman_kendall(proxy, human):
    from scipy.stats import kendalltau, spearmanr
    pairs = [(p, h) for p, h in zip(proxy, human) if p is not None and h is not None]
    if len(pairs) < 8:
        return {"n": len(pairs), "spearman": None, "kendall": None}
    pv = [p for p, _ in pairs]
    hv = [h for _, h in pairs]
    rho, prho = spearmanr(pv, hv)
    tau, ptau = kendalltau(pv, hv)
    return {
        "n": len(pairs),
        "spearman": round(float(rho), 4), "spearman_p": round(float(prho), 6),
        "kendall": round(float(tau), 4), "kendall_p": round(float(ptau), 6),
    }


def _load_pairs(n_articles):
    """Return [(source, summary, {axis: human_loss})...] over the first n_articles
    (sorted by id) × their 16 machine summaries."""
    from datasets import load_dataset
    ds = load_dataset("mteb/summeval", split="test")
    rows = sorted(ds, key=lambda r: r["id"])[:n_articles]
    pairs = []
    for r in rows:
        src = r["text"]
        for j, summ in enumerate(r["machine_summaries"]):
            axes = {a: _human_loss(r[a][j]) for a in ("consistency", "relevance", "coherence", "fluency")}
            # faithfulness-aligned composite mirroring the scorer's recall/fidelity weights
            cons, rel = axes["consistency"], axes["relevance"]
            axes["meaning_composite"] = (
                0.6 * rel + 0.4 * cons if rel is not None and cons is not None else None
            )
            vals = [v for k, v in axes.items() if k != "meaning_composite" and v is not None]
            axes["mean_all4"] = mean(vals) if vals else None
            pairs.append((src, summ, axes))
    return pairs


def _score(pairs, scorer):
    out = []
    for src, summ, _ in pairs:
        try:
            out.append(float(scorer.loss(src, summ)))
        except Exception:
            out.append(None)
    return out


AXES = ("consistency", "relevance", "coherence", "fluency", "meaning_composite", "mean_all4")


def _correlate_all(proxy_losses, pairs):
    return {ax: _spearman_kendall(proxy_losses, [p[2][ax] for p in pairs]) for ax in AXES}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast-articles", type=int, default=50, help="articles for lexical + embedding (16 summaries each)")
    ap.add_argument("--nli-articles", type=int, default=12, help="articles for the (slow) NLI judge")
    ap.add_argument("--sweep-articles", type=int, default=20, help="articles for the embedding threshold sweep")
    ap.add_argument("--out", default="Tests/benchmarks/meaning_proxy_human_calibration.result.json")
    args = ap.parse_args(argv)

    from sum_engine_internal.research.meaning.local_judge import (
        embedding_entailment_scorer, nli_entailment_scorer)
    from sum_engine_internal.research.meaning.meaning_loss import LexicalCoverageScorer

    print(f"loading SummEval ({args.fast_articles} articles x16 = {args.fast_articles*16} pairs)…", flush=True)
    fast_pairs = _load_pairs(args.fast_articles)
    nli_pairs = fast_pairs[: args.nli_articles * 16]
    sweep_pairs = fast_pairs[: args.sweep_articles * 16]

    result = {
        "dataset": "mteb/summeval (test, CNN/DM, expert-annotated 1-5)",
        "method": "summary-level pooled rank correlation; proxy meaning-loss vs human loss=(5-score)/4; higher=worse for both, so POSITIVE rho = proxy tracks humans",
        "n_pairs_fast": len(fast_pairs), "n_pairs_nli": len(nli_pairs), "n_pairs_sweep": len(sweep_pairs),
        "scorers": {},
    }

    print("scoring lexical (LexicalCoverageScorer)…", flush=True)
    lex = _score(fast_pairs, LexicalCoverageScorer())
    result["scorers"]["lexical-coverage-bidirectional"] = _correlate_all(lex, fast_pairs)

    print("scoring embedding judge @0.5 (minilm-cosine)…", flush=True)
    emb = _score(fast_pairs, embedding_entailment_scorer(threshold=0.5))
    result["scorers"]["embedding-minilm@0.5"] = _correlate_all(emb, fast_pairs)

    print("embedding threshold sweep (vs meaning_composite)…", flush=True)
    sweep = {}
    for t in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
        losses = _score(sweep_pairs, embedding_entailment_scorer(threshold=t))
        sweep[f"{t}"] = _spearman_kendall(losses, [p[2]["meaning_composite"] for p in sweep_pairs])
        print(f"  t={t}: spearman_vs_meaning={sweep[f'{t}']['spearman']}", flush=True)
    best = max((k for k in sweep if sweep[k]["spearman"] is not None),
               key=lambda k: sweep[k]["spearman"], default=None)
    result["embedding_threshold_sweep"] = {"vs": "meaning_composite", "n": len(sweep_pairs), "by_threshold": sweep, "best": best}

    print(f"scoring NLI judge @0.5 on {len(nli_pairs)} pairs (slow)…", flush=True)
    nli = _score(nli_pairs, nli_entailment_scorer(threshold=0.5))
    result["scorers"]["nli-deberta-mnli-fever-anli@0.5"] = _correlate_all(nli, nli_pairs)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 64 + "\nPROXY vs HUMAN — Spearman rho (positive = proxy tracks humans)\n" + "=" * 64)
    for name, corr in result["scorers"].items():
        c = corr["consistency"]["spearman"]; r = corr["relevance"]["spearman"]; m = corr["meaning_composite"]["spearman"]
        print(f"  {name:38} consistency={c}  relevance={r}  meaning={m}")
    print(f"  best embedding threshold (vs meaning) = {best}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
