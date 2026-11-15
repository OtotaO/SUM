"""Tags against claims: which of a text's atomic claims a list of notes states.

    PYTHONPATH=. python scripts/tags_claims.py TEXT_ID [--out FILE.json] [--repeat 2]

TEXT_ID names a folder under fixtures/tags_roundtrip/ (efso_vaults_1 or
efso_vaults_2). The claims come from fixtures/tags_claims/claims.json: short,
self-contained sentences, one fact each, written once per text from the text
alone and frozen. Each claim carries a "checked" mark that only the text's
author sets; the run records how many claims were checked.

For each note list, the premise is the notes joined one per line (as the
round-trip scripts score notes alone), and a claim is credited when the exact
judge of scripts/tags_roundtrip.py (DeBERTa-v3-large NLI, windows of at most
80 tokens, probability at least 0.9) finds it in some window. Recall is the
credited claims over all claims. Nothing is generated.

Note lists scored: the same lists as scripts/tags_bits.py (every recorded
equal-budget rung, the other EFSO text's notes, the negated, swapped and
unrelated controls, a filler list), plus the source itself (each claim should
be found in its own text).

The floor: the six recorded rewrites written with no tags (three for each EFSO
text) are about something else, yet the judge credits them with a few claims.
A claim any of them is credited with is loose; the rest are firm, and the
firm counts are the measure. The floor texts set which claims are loose, so
they score zero on firm claims by construction; the other controls do not.

The whole measurement runs --repeat times and the largest difference in any
judge probability between runs is recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from scripts import tags_bits as tb

METHOD_ID = "sum.tags_claims.v1"
ROOT = Path(__file__).resolve().parent.parent
CLAIMS = Path("fixtures/tags_claims/claims.json")
MaxProbs = Callable[[str, Sequence[str]], list]


def load(text_id: str) -> dict[str, Any]:
    inp = tb.load_inputs(text_id)
    path = ROOT / CLAIMS
    claims = json.loads(path.read_text(encoding="utf-8"))["texts"][text_id]
    inp["claims"], inp["text"] = claims, text_id
    inp["files"][CLAIMS.as_posix()] = tb.sha256_file(path)
    return inp


def conditions(inp: dict[str, Any]) -> list[dict[str, Any]]:
    out = [{k: c[k] for k in ("condition", "group", "notes")} for c in tb.conditions(inp)]
    out.append({"condition": "source", "group": "self", "notes": [inp["source"]]})
    for tid, res in ((inp["text"], inp["results"]), (inp["other_id"], inp["other"])):
        ctl = [r for r in res["rungs"] if r["condition"] == "control"][0]
        out += [{"condition": f"no_tags_rewrite:{tid}:seed{s['seed']}", "group": "floor", "notes": [s["rewrite"]]}
                for s in ctl["samples"]]
    return out


def credited(p: float | None, threshold: float) -> bool:
    return p is not None and p >= threshold


def measure(max_probs: MaxProbs, claims: Sequence[dict[str, Any]], conds: Sequence[dict[str, Any]],
            threshold: float) -> list[dict[str, Any]]:
    texts = [c["claim"] for c in claims]
    rows = []
    for c in conds:
        probs = max_probs("\n".join(c["notes"]), texts)
        hits = [cl["id"] for cl, p in zip(claims, probs) if credited(p, threshold)]
        rows.append({"condition": c["condition"], "group": c["group"],
                     "note_count": len(c["notes"]), "note_words": sum(len(n.split()) for n in c["notes"]),
                     "credited": len(hits), "recall": len(hits) / len(claims) if claims else 0.0,
                     "credited_ids": hits, "probs": probs})
    return rows


def firm(rows: Sequence[dict[str, Any]], claims: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A claim is loose if any floor text (a recorded rewrite written with no
    tags, about something else) is credited with it; the rest are firm. Adds
    each row's firm counts in place and returns the split."""
    loose = sorted({i for r in rows if r["group"] == "floor" for i in r["credited_ids"]})
    firm_ids = [c["id"] for c in claims if c["id"] not in loose]
    keep = set(firm_ids)
    for r in rows:
        r["firm_credited_ids"] = [i for i in r["credited_ids"] if i in keep]
        r["firm_credited"] = len(r["firm_credited_ids"])
        r["firm_recall"] = r["firm_credited"] / len(firm_ids) if firm_ids else 0.0
    return {"loose_ids": loose, "firm_claims": len(firm_ids)}


def checks(rows: Sequence[dict[str, Any]], split: dict[str, Any], claims: int) -> dict[str, Any]:
    by = {r["condition"]: r for r in rows}
    true = by[tb.FACT_RUNG]["firm_credited"]

    def most(group: str) -> int | None:
        vals = [r["firm_credited"] for r in rows if r["group"] == group]
        return max(vals) if vals else None

    return {
        "claims": claims, "loose_share": len(split["loose_ids"]) / claims if claims else 0.0,
        "firm_claims": split["firm_claims"],
        "source_firm_recall": by["source"]["firm_recall"],
        "fact_firm_credited": true,
        "negated_firm_credited": by[f"negated:{tb.FACT_RUNG}"]["firm_credited"],
        "swapped_firm_credited": by[f"swapped:{tb.FACT_RUNG}"]["firm_credited"],
        "unrelated_firm_credited": by["unrelated"]["firm_credited"],
        "filler_firm_credited": by["filler"]["firm_credited"],
        "other_text_max_firm_credited": most("same_series"),
    }


def max_run_difference(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> float:
    diffs = [abs((x or 0.0) - (y or 0.0)) for ra, rb in zip(a, b) for x, y in zip(ra["probs"], rb["probs"])]
    return max(diffs) if diffs else 0.0


def run(text_id: str, max_probs: MaxProbs, threshold: float, repeat: int = 2) -> dict[str, Any]:
    inp = load(text_id)
    claims = inp["claims"]
    conds = conditions(inp)
    runs = [measure(max_probs, claims, conds, threshold) for _ in range(max(1, repeat))]
    split = firm(runs[0], claims)
    return {
        "method": METHOD_ID, "text": text_id, "inputs": inp["files"],
        "premise": "the notes joined one per line",
        "measures": "which of the text's frozen atomic claims the notes state, for one NLI judge",
        "claims": len(claims), "claims_checked_by_author": sum(1 for c in claims if c.get("checked") is True),
        "repeats": len(runs),
        "max_difference_between_runs": max(max_run_difference(runs[0], r) for r in runs),
        **split, "rows": runs[0], "checks": checks(runs[0], split, len(claims)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.tags_roundtrip import EXACT_THRESHOLD, WindowedNLI

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("text", choices=tb.TEXTS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args(argv)
    nli = WindowedNLI()
    result = run(a.text, nli.max_probs, EXACT_THRESHOLD, a.repeat)
    out = {"judge": nli.manifest(), **result}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["checks"].items()}, indent=1))
    print(f"claims {out['claims']}, checked by the author {out['claims_checked_by_author']}, "
          f"largest difference between runs {out['max_difference_between_runs']}")
    for r in out["rows"]:
        print(f"{r['condition']:36} {r['group']:11} words {r['note_words']:4}  credited {r['credited']:3}  "
              f"firm {r['firm_credited']:3}  firm recall {r['firm_recall']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
