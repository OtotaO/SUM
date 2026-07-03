"""Second-corpus calibration: SUM's shipped meaning-loss proxies vs FRANK human
faithfulness annotations (Pagnoni et al., NAACL 2021 — github.com/artidoro/frank).

Corpus: FRANK XSum half (BERTS2S / TConvS2S / PtGen / TranS2S) — a genuinely
different distribution from CNN/DM-SummEval (abstractive one-sentence summaries,
high hallucination rate). The CNN/DM half is scored too as an in-scheme control
(same annotation protocol, SummEval-like distribution).

Human target: FRANK `Factuality` in [0,1] = proportion of summary judged factual
(sentence-level judgments aggregated over annotators). human_loss = 1 - Factuality.

Text repair (disclosed): FRANK stores XSum articles with sentences joined WITHOUT
whitespace after '.', which collapses the repo's `_sentences` splitter to one
giant unit and degenerates the entailment scorers (loss == 1.0 everywhere). We
insert a single space after [.!?] when directly followed by an uppercase letter,
quote, or '(' — corpus repair, not scorer modification. The repair is a no-op on
already-spaced text (verified on the CNN/DM half: median sentence count 18 -> 18).
Raw (unrepaired) numbers are reported alongside for the XSum lexical scorer,
which is insensitive to sentence splitting.

Subsample: deterministic seed=0 sample of the (hash, model_name)-sorted pairs.
"""
import json
import random
import re
import sys
import time
from statistics import mean, pstdev

import numpy as np
from scipy.stats import pearsonr, spearmanr

OUT_DIR = "/private/tmp/claude-501/-Users-ototao-Github-Projects-SUM-SUM/f068abcf-be8c-465b-b8be-00a2305dd0c7/scratchpad/calibration2"
XSUM_MODELS = {"BERTS2S", "TConvS2S", "PtGen", "TranS2S"}
N_XSUM = 250   # NLI budget ~1.1 s/pair
N_CNNDM = 150  # NLI budget ~2.5 s/pair

REPAIR = re.compile(r'(?<=[.!?])(?=[A-Z"“(])')


def repair(t):
    return REPAIR.sub(" ", t)


def corr(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    xv = [a for a, _ in pairs]
    yv = [b for _, b in pairs]
    rho, prho = spearmanr(xv, yv)
    r, pr = pearsonr(xv, yv)
    return {"n": len(pairs), "spearman": round(float(rho), 4), "spearman_p": round(float(prho), 6),
            "pearson": round(float(r), 4), "pearson_p": round(float(pr), 6)}


def bootstrap_ci(x, y, stat="spearman", n_boot=1000, seed=0):
    rng = random.Random(seed)
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        xv = [pairs[i][0] for i in idx]
        yv = [pairs[i][1] for i in idx]
        v = spearmanr(xv, yv)[0] if stat == "spearman" else pearsonr(xv, yv)[0]
        if not np.isnan(v):
            vals.append(float(v))
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals))], 4)]


def main():
    t_start = time.time()
    from sum_engine_internal.research.meaning.local_judge import (
        embedding_entailment_scorer,
        nli_entailment_scorer,
    )
    from sum_engine_internal.research.meaning.meaning_loss import LexicalCoverageScorer

    bench = json.load(open(f"{OUT_DIR}/frank_benchmark_data.json"))
    ann = json.load(open(f"{OUT_DIR}/frank_human_annotations.json"))
    ann_by_key = {(a["hash"], a["model_name"]): a for a in ann}

    halves = {}
    for half, pred, n_sub in (
        ("xsum", lambda b: b["model_name"] in XSUM_MODELS, N_XSUM),
        ("cnndm", lambda b: b["model_name"] not in XSUM_MODELS, N_CNNDM),
    ):
        rows = sorted((b for b in bench if pred(b)), key=lambda b: (b["hash"], b["model_name"]))
        rows = [b for b in rows if (b["hash"], b["model_name"]) in ann_by_key]
        rng = random.Random(0)
        sub = rng.sample(rows, n_sub)
        recs = []
        for b in sub:
            a = ann_by_key[(b["hash"], b["model_name"])]
            recs.append({
                "hash": b["hash"], "model": b["model_name"],
                "src": b["article"], "summ": b["summary"],
                "human_loss": 1.0 - float(a["Factuality"]),
            })
        halves[half] = recs
        print(f"{half}: {len(rows)} annotated pairs, subsampled {len(recs)} (seed=0)", flush=True)

    scorers = {
        "lexical-coverage-bidirectional": LexicalCoverageScorer(),
        "embedding-minilm@0.5": embedding_entailment_scorer(threshold=0.5),
        "nli-deberta-mnli-fever-anli@0.5": nli_entailment_scorer(threshold=0.5),
    }

    out = {"corpus": "FRANK (Pagnoni et al. 2021), human Factuality -> loss = 1 - Factuality",
           "repair": "space inserted after [.!?] when followed by uppercase/quote/( (XSum text stored without inter-sentence spaces)",
           "halves": {}}

    for half, recs in halves.items():
        hv = [r["human_loss"] for r in recs]
        res = {"n": len(recs),
               "human_loss_distribution": {
                   "mean": round(mean(hv), 3), "sd": round(pstdev(hv), 3),
                   "frac_zero_loss": round(sum(1 for v in hv if v <= 1e-9) / len(hv), 3)},
               "scorers": {}}
        for name, scorer in scorers.items():
            t0 = time.time()
            losses = []
            for i, r in enumerate(recs):
                try:
                    losses.append(float(scorer.loss(repair(r["src"]), repair(r["summ"]))))
                except Exception as e:
                    losses.append(None)
                    print(f"  {half}/{name} pair {i} failed: {e}", flush=True)
                if (i + 1) % 50 == 0:
                    print(f"  {half}/{name}: {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
            for r, loss_val in zip(recs, losses):
                r[name] = loss_val
            c = corr(losses, hv)
            c["spearman_ci95"] = bootstrap_ci(losses, hv, "spearman")
            c["pearson_ci95"] = bootstrap_ci(losses, hv, "pearson")
            c["proxy_loss_sd"] = round(pstdev([x for x in losses if x is not None]), 3)
            res["scorers"][name] = c
            print(f"{half}/{name}: rho={c['spearman']} CI{c['spearman_ci95']} r={c['pearson']} ({time.time()-t0:.0f}s)", flush=True)

        # raw-text (unrepaired) lexical control on xsum: repair shouldn't matter for lexical
        if half == "xsum":
            lex = scorers["lexical-coverage-bidirectional"]
            raw_lex = [float(lex.loss(r["src"], r["summ"])) for r in recs]
            res["scorers"]["lexical-RAW-unrepaired"] = corr(raw_lex, hv)
            raw_nli_note = "NLI on unrepaired xsum text degenerates (splitter sees 1 sentence; probe: losses all 1.0)"
            res["raw_text_note"] = raw_nli_note
        out["halves"][half] = res

    slim = {h: [{k: v for k, v in r.items() if k not in ("src", "summ")} for r in recs]
            for h, recs in halves.items()}
    with open(f"{OUT_DIR}/frank_perpair.json", "w") as f:
        json.dump(slim, f)
    out["wall_seconds"] = round(time.time() - t_start)
    with open(f"{OUT_DIR}/frank_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["halves"], indent=1, default=str)[:4000])
    print(f"wrote frank_results.json in {out['wall_seconds']}s")


if __name__ == "__main__":
    sys.exit(main())
