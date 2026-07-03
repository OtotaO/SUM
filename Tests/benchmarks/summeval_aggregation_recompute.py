"""Recompute the SummEval proxy-vs-human calibration at BOTH aggregation levels.

Replicates Tests/benchmarks/meaning_proxy_human_calibration.py exactly
(same dataset slice, same shipped scorers, same human-loss mapping), but
persists per-pair losses and adds:
  - pooled summary-level Spearman AND Pearson (the harness reported pooled Spearman only)
  - system-level correlation (aggregate per system across articles, n=16 systems)
  - within-article averaged Spearman
  - a system-index consistency sanity check (permutation null)
  - bootstrap 95% CIs (1000 resamples, seed=0)

Reconciliation target: "proxy ~0.27-0.33 (SummEval)" vs "UFAL/RWS NLI 0.67 /
word-overlap 0.63 (example-level Spearman, hotel highlights, n=120)".
"""
import json
import random
import sys
import time
from statistics import mean, pstdev

import numpy as np
from scipy.stats import pearsonr, spearmanr

OUT_DIR = "/private/tmp/claude-501/-Users-ototao-Github-Projects-SUM-SUM/f068abcf-be8c-465b-b8be-00a2305dd0c7/scratchpad/calibration2"
FAST_ARTICLES = 50   # same as committed harness run (n=800 pairs)
NLI_ARTICLES = 12    # same as committed harness run (n=192 pairs)
N_SYSTEMS = 16


def human_loss(s):
    return max(0.0, min(1.0, (5.0 - float(s)) / 4.0))


def load_rows():
    from datasets import load_dataset
    ds = load_dataset("mteb/summeval", split="test")
    return sorted(ds, key=lambda r: r["id"])[:FAST_ARTICLES]


def corr(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4:
        return {"n": len(pairs)}
    xv = [a for a, _ in pairs]
    yv = [b for _, b in pairs]
    rho, prho = spearmanr(xv, yv)
    r, pr = pearsonr(xv, yv)
    return {"n": len(pairs), "spearman": round(float(rho), 4), "spearman_p": round(float(prho), 6),
            "pearson": round(float(r), 4), "pearson_p": round(float(pr), 6)}


def bootstrap_spearman(x, y, n_boot=1000, seed=0):
    rng = random.Random(seed)
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        xv = [pairs[i][0] for i in idx]
        yv = [pairs[i][1] for i in idx]
        rho, _ = spearmanr(xv, yv)
        if not np.isnan(rho):
            vals.append(float(rho))
    vals.sort()
    return {"ci95": [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals))], 4)],
            "n_boot": len(vals)}


def main():
    t_start = time.time()
    from sum_engine_internal.research.meaning.local_judge import (
        embedding_entailment_scorer,
        nli_entailment_scorer,
    )
    from sum_engine_internal.research.meaning.meaning_loss import LexicalCoverageScorer

    rows = load_rows()
    # records: one per (article a, system j)
    recs = []
    for a, r in enumerate(rows):
        for j, summ in enumerate(r["machine_summaries"]):
            recs.append({
                "article": a, "system": j, "src": r["text"], "summ": summ,
                "consistency": human_loss(r["consistency"][j]),
                "relevance": human_loss(r["relevance"][j]),
                "coherence": human_loss(r["coherence"][j]),
                "fluency": human_loss(r["fluency"][j]),
            })
    for rec in recs:
        rec["meaning_composite"] = 0.6 * rec["relevance"] + 0.4 * rec["consistency"]
    assert len(recs) == FAST_ARTICLES * N_SYSTEMS

    scorers = {
        "lexical-coverage-bidirectional": (LexicalCoverageScorer(), len(recs)),
        "embedding-minilm@0.5": (embedding_entailment_scorer(threshold=0.5), len(recs)),
        "nli-deberta-mnli-fever-anli@0.5": (nli_entailment_scorer(threshold=0.5), NLI_ARTICLES * N_SYSTEMS),
    }
    for name, (scorer, n) in scorers.items():
        print(f"scoring {name} on {n} pairs...", flush=True)
        t0 = time.time()
        for i, rec in enumerate(recs):
            if i >= n:
                break
            try:
                rec[name] = float(scorer.loss(rec["src"], rec["summ"]))
            except Exception as e:
                rec[name] = None
                print(f"  pair {i} failed: {e}", flush=True)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # persist per-pair losses (drop text to keep the file small)
    slim = [{k: v for k, v in rec.items() if k not in ("src", "summ")} for rec in recs]
    with open(f"{OUT_DIR}/summeval_perpair.json", "w") as f:
        json.dump(slim, f)

    axes = ("consistency", "relevance", "meaning_composite")
    out = {
        "dataset": "mteb/summeval test, first 50 articles sorted by id x 16 systems",
        "n_pairs": {name: n for name, (_, n) in scorers.items()},
        "pooled_summary_level": {}, "system_level": {}, "within_article_mean_spearman": {},
        "human_score_distribution": {},
        "system_index_sanity": {},
    }

    # human score distribution (the range-restriction question)
    for ax in ("consistency", "relevance"):
        v = [rec[ax] for rec in slim]
        raw = [5.0 - 4.0 * x for x in v]  # back to 1-5
        out["human_score_distribution"][ax] = {
            "mean_1to5": round(mean(raw), 3), "sd": round(pstdev(raw), 3),
            "frac_at_5.0": round(sum(1 for s in raw if s >= 4.999) / len(raw), 3),
            "frac_ge_4.5": round(sum(1 for s in raw if s >= 4.5) / len(raw), 3),
        }

    # system-index sanity: do per-index mean consistency scores vary more than chance?
    rng = random.Random(0)
    cons_by_art = [[slim[a * N_SYSTEMS + j]["consistency"] for j in range(N_SYSTEMS)] for a in range(FAST_ARTICLES)]
    obs_means = [mean(cons_by_art[a][j] for a in range(FAST_ARTICLES)) for j in range(N_SYSTEMS)]
    obs_sd = pstdev(obs_means)
    null_sds = []
    for _ in range(1000):
        shuf = [list(row) for row in cons_by_art]
        for row in shuf:
            rng.shuffle(row)
        m = [mean(shuf[a][j] for a in range(FAST_ARTICLES)) for j in range(N_SYSTEMS)]
        null_sds.append(pstdev(m))
    p = sum(1 for s in null_sds if s >= obs_sd) / len(null_sds)
    out["system_index_sanity"] = {
        "observed_sd_of_per_index_mean_consistency_loss": round(obs_sd, 4),
        "null_sd_mean": round(mean(null_sds), 4), "permutation_p": p,
        "reading": "small p => machine_summaries index is system-consistent across articles",
    }

    for name, (_, n) in scorers.items():
        sub = slim[:n]
        losses = [rec.get(name) for rec in sub]
        out["pooled_summary_level"][name] = {}
        for ax in axes:
            c = corr(losses, [rec[ax] for rec in sub])
            c.update(bootstrap_spearman(losses, [rec[ax] for rec in sub]))
            out["pooled_summary_level"][name][ax] = c

        # system level: mean loss / mean human per system index
        n_articles = n // N_SYSTEMS
        out["system_level"][name] = {}
        for ax in axes:
            sysx, sysy = [], []
            for j in range(N_SYSTEMS):
                lv = [sub[a * N_SYSTEMS + j].get(name) for a in range(n_articles)]
                hv = [sub[a * N_SYSTEMS + j][ax] for a in range(n_articles)]
                lv = [v for v in lv if v is not None]
                sysx.append(mean(lv))
                sysy.append(mean(hv))
            c = corr(sysx, sysy)
            # bootstrap over ARTICLES (resample articles, recompute system means)
            rngb = random.Random(0)
            vals = []
            for _ in range(1000):
                arts = [rngb.randrange(n_articles) for _ in range(n_articles)]
                bx, by = [], []
                for j in range(N_SYSTEMS):
                    lv = [sub[a * N_SYSTEMS + j].get(name) for a in arts]
                    hv = [sub[a * N_SYSTEMS + j][ax] for a in arts]
                    lv = [v for v in lv if v is not None]
                    bx.append(mean(lv))
                    by.append(mean(hv))
                rho, _ = spearmanr(bx, by)
                if not np.isnan(rho):
                    vals.append(float(rho))
            vals.sort()
            c["spearman_ci95_article_bootstrap"] = [round(vals[int(0.025 * len(vals))], 4),
                                                    round(vals[int(0.975 * len(vals))], 4)]
            out["system_level"][name][ax] = c

        # within-article mean spearman
        out["within_article_mean_spearman"][name] = {}
        for ax in axes:
            rhos = []
            for a in range(n_articles):
                lv = [sub[a * N_SYSTEMS + j].get(name) for j in range(N_SYSTEMS)]
                hv = [sub[a * N_SYSTEMS + j][ax] for j in range(N_SYSTEMS)]
                ok = [(x, y) for x, y in zip(lv, hv) if x is not None]
                if len(ok) >= 8:
                    rho, _ = spearmanr([x for x, _ in ok], [y for _, y in ok])
                    if not np.isnan(rho):
                        rhos.append(float(rho))
            out["within_article_mean_spearman"][name][ax] = {
                "mean_rho": round(mean(rhos), 4) if rhos else None, "n_articles": len(rhos)}

    out["wall_seconds"] = round(time.time() - t_start)
    with open(f"{OUT_DIR}/summeval_aggregation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("pooled_summary_level", "system_level")}, indent=1))
    print(f"wrote summeval_aggregation.json in {out['wall_seconds']}s")


if __name__ == "__main__":
    sys.exit(main())
