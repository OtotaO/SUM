"""§2.5 LLM round-trip drift attack — canonicalization replay.

Replays the cached llm_roundtrip output from `bench_history.jsonl`
under progressively more aggressive canonicalization regimes, and
records what each regime does to the load-bearing numbers from
PROOF_BOUNDARY §2.5:

  * **drift_pct** — `100 × |A Δ A'| / max(|A|, |A'|)` per doc
  * **exact-match recall** — fraction of source triples that
    appear verbatim (after canonicalization) in the reconstructed
    set

The baseline (cached) numbers are 107.75 % drift / 0.12 recall
on `seed_v1` (50 docs, both legs ``gpt-4o-mini-2024-07-18``).
This runner adds NO new LLM calls — it operates over the cached
``missing_claims`` and ``extra_claims`` arrays per doc and
recomputes the metrics under each regime.

The point is empirical receipts, not improvement-by-claim. Each
regime's transformation rules are explicit and inspectable; if a
regime conflates legitimately-different concepts and the recall
moves up, the user can see exactly which rule did it.

Regimes:

  * **L0 — baseline.** No canonicalization. Sanity-checks that
    the replay reproduces the cached metrics.
  * **L1 — predicate normalisation.** Lowercase; strip auxiliary
    prefixes (``was_``, ``has_``, ``is_``, ``had_``, ``have_``)
    and preposition suffixes (``_in``, ``_to``, ``_of``, ``_with``,
    ``_by``, ``_on``, ``_for``, ``_from``, ``_at``); strip verb
    inflection (``-s``, ``-ed``, ``-ing`` via simple suffix
    rules). Catches `eat`/`eats`/`consumes`-style drift.
  * **L2 — + subject canonicalisation.** Lowercase; replace
    underscores with spaces; for multi-word subjects, take the
    last word as the canonical key. Catches
    `newton`/`isaac_newton`-style drift but conflates anyone
    sharing a surname (acceptable on `seed_v1`'s synthetic
    corpus; would be risky in production).
  * **L3 — + object canonicalisation.** Lowercase; replace
    underscores with spaces; strip articles (`a`, `an`, `the`);
    take first content word. Aggressive — also conflates
    different ``cats provide companionship`` /
    ``cats are companionship`` shapes; reported with the explicit
    caveat that L3 is the ceiling, not a recommendation.

Output (JSON to stdout or ``--out FILE``):

```json
{
  "schema": "sum.s25_canonicalization_replay.v1",
  "issued_at": "<iso>",
  "source_run": {"run_id": "...", "git_sha": "...", "corpus": "seed_v1"},
  "regimes": [
    {
      "name": "L0_baseline",
      "rules": [],
      "metrics": {"drift_pct_mean": ..., "exact_match_recall": ...},
      "per_doc_recall": {"doc_001": 0.0, ...}
    },
    ...
  ]
}
```

Usage:
    python -m scripts.bench.runners.canonicalization_replay \\
        --bench-history bench_history.jsonl \\
        --out /tmp/s25_replay.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

# ----------------------------------------------------------------------
# Canonicalization rules — explicit, inspectable, no learned models.
# ----------------------------------------------------------------------

_PRED_AUX_PREFIXES: tuple[str, ...] = (
    "was_", "has_", "is_", "had_", "have_", "are_",
    "be_", "been_", "being_",
)
_PRED_PREP_SUFFIXES: tuple[str, ...] = (
    "_in", "_to", "_of", "_with", "_by", "_on",
    "_for", "_from", "_at", "_into", "_about",
)
_OBJECT_ARTICLES: tuple[str, ...] = ("a ", "an ", "the ")


def _normalize_predicate_l1(p: str) -> str:
    """L1 predicate normalisation."""
    s = p.lower().strip()
    # Strip aux prefixes, repeatedly until none apply.
    changed = True
    while changed:
        changed = False
        for pre in _PRED_AUX_PREFIXES:
            if s.startswith(pre) and len(s) > len(pre):
                s = s[len(pre):]
                changed = True
                break
    # Strip preposition suffixes (one round; multiple suffixes is rare
    # and conflating "leans_into_doing_for" → "do" would be too aggressive).
    for suf in _PRED_PREP_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            s = s[: -len(suf)]
            break
    # Verb inflection — strip -ing / -ed / -s, in that order. Simple
    # English heuristics; not lemmatisation. Doubled-consonant cases
    # (`cutting`→`cut`, not `cutt`) handled with a small fixup table.
    if s.endswith("ing") and len(s) > 4:
        s = s[:-3]
        # `running`→`runn`→`run`, `cutting`→`cutt`→`cut`
        if len(s) >= 2 and s[-1] == s[-2] and s[-1] in "bdfgklmnprt":
            s = s[:-1]
    elif s.endswith("ed") and len(s) > 3:
        s = s[:-2]
        if len(s) >= 2 and s[-1] == s[-2] and s[-1] in "bdfgklmnprt":
            s = s[:-1]
    elif s.endswith("es") and len(s) > 3:
        s = s[:-2]
    elif s.endswith("s") and len(s) > 2 and not s.endswith("ss"):
        s = s[:-1]
    return s


def _normalize_subject_l2(subj: str) -> str:
    """L2 subject canonicalisation — last word of underscore-separated."""
    s = subj.lower().replace("_", " ").strip()
    parts = s.split()
    return parts[-1] if parts else s


def _normalize_object_l3(obj: str) -> str:
    """L3 object canonicalisation — first content word, articles stripped."""
    s = obj.lower().replace("_", " ").strip()
    for art in _OBJECT_ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
            break
    parts = s.split()
    return parts[0] if parts else s


# ----------------------------------------------------------------------
# Triple transformers per regime.
# ----------------------------------------------------------------------

Triple = tuple[str, str, str]


def _t_l0(t: Triple) -> Triple:
    return t


def _t_l1(t: Triple) -> Triple:
    return (t[0], _normalize_predicate_l1(t[1]), t[2])


def _t_l2(t: Triple) -> Triple:
    return (_normalize_subject_l2(t[0]), _normalize_predicate_l1(t[1]), t[2])


def _t_l3(t: Triple) -> Triple:
    return (
        _normalize_subject_l2(t[0]),
        _normalize_predicate_l1(t[1]),
        _normalize_object_l3(t[2]),
    )


_REGIMES: list[tuple[str, str, Callable[[Triple], Triple]]] = [
    (
        "L0_baseline",
        "no canonicalization — sanity check against cached numbers",
        _t_l0,
    ),
    (
        "L1_predicate",
        "predicate: lowercase, strip aux prefixes, strip prep suffixes, strip verb inflection",
        _t_l1,
    ),
    (
        "L2_subject_predicate",
        "L1 + subject: lowercase, underscore→space, last-word-as-key",
        _t_l2,
    ),
    (
        "L3_all_aggressive",
        "L2 + object: lowercase, strip articles, first-content-word-as-key (CEILING — risky)",
        _t_l3,
    ),
]


# ----------------------------------------------------------------------
# Replay over cached per-doc data.
# ----------------------------------------------------------------------


def _doc_metrics(
    source_axioms: list[Triple],
    reconstructed_axioms: list[Triple],
    transform: Callable[[Triple], Triple],
) -> dict:
    """Recompute drift_pct + exact-match-recall under a transform."""
    src = {transform(t) for t in source_axioms}
    rec = {transform(t) for t in reconstructed_axioms}
    if not src and not rec:
        return {"drift_pct": 0.0, "recall": 1.0, "matched": 0, "src": 0, "rec": 0}
    sym_diff = src.symmetric_difference(rec)
    denom = max(len(src), len(rec))
    drift = (100.0 * len(sym_diff) / denom) if denom else 0.0
    matched = len(src & rec)
    recall = (matched / len(src)) if src else 0.0
    return {
        "drift_pct": drift,
        "recall": recall,
        "matched": matched,
        "src": len(src),
        "rec": len(rec),
    }


def _load_cached_run(history_path: Path) -> dict:
    """Find the most recent llm_roundtrip record in bench_history."""
    latest = None
    with history_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("llm_roundtrip"):
                latest = rec
    if latest is None:
        raise SystemExit(
            f"no llm_roundtrip records found in {history_path}; "
            f"run scripts/bench/run_bench.py with the LLM-gated runner first"
        )
    return latest


def _reconstruct_per_doc_axioms(per_doc: list[dict]) -> list[dict]:
    """Reconstruct full source/reconstructed sets per doc from missing+extra.

    The cached schema stores ``missing_claims`` (in source, not in
    reconstructed) and ``extra_claims`` (in reconstructed, not in
    source). To rebuild the full sets:
      source         = matched ∪ missing
      reconstructed  = matched ∪ extra

    The ``matched`` set is the source∩reconstructed intersection; the
    schema doesn't store it directly but ``n_source_axioms`` and
    ``len(missing_claims)`` together pin |matched| = n_source_axioms -
    |missing_claims|. Since matched triples appear in BOTH sets they
    don't drive the drift metric — but they DO drive the
    exact-match-recall numerator. We can't reconstruct the matched
    triples' actual values from the cached fields, so we tag them as
    opaque match-stubs (``("__matched__", str(i), "")``) that survive
    set operations under any transform that doesn't touch them. This
    is correct for the recall computation: an opaque match in baseline
    stays opaque under canonicalisation, contributing 1 to recall.
    """
    out: list[dict] = []
    for d in per_doc:
        n_src = d["n_source_axioms"]
        missing = [tuple(t) for t in d["missing_claims"]]
        extra = [tuple(t) for t in d["extra_claims"]]
        n_matched = n_src - len(missing)
        match_stubs: list[Triple] = [
            (f"__matched_{d['doc_id']}_{i}__", "match_stub", "")
            for i in range(n_matched)
        ]
        out.append(
            {
                "doc_id": d["doc_id"],
                "source": match_stubs + missing,
                "reconstructed": match_stubs + extra,
                "cached_drift_pct": d["drift_pct"],
            }
        )
    return out


def _aggregate(per_doc_metrics: list[dict]) -> dict:
    drifts = [m["drift_pct"] for m in per_doc_metrics]
    recalls = [m["recall"] for m in per_doc_metrics]
    n_exact_recall_1 = sum(1 for r in recalls if r >= 0.999)
    return {
        "n_docs": len(per_doc_metrics),
        "drift_pct_mean": round(statistics.mean(drifts), 4) if drifts else 0.0,
        "drift_pct_median": round(statistics.median(drifts), 4) if drifts else 0.0,
        "exact_match_recall_mean": round(statistics.mean(recalls), 4) if recalls else 0.0,
        "exact_match_recall_p10": round(_pct(recalls, 10), 4) if recalls else 0.0,
        "n_docs_full_recall": n_exact_recall_1,
        "fraction_full_recall": round(n_exact_recall_1 / len(recalls), 4) if recalls else 0.0,
    }


def _pct(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-history", default="bench_history.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rec = _load_cached_run(Path(args.bench_history))
    rt = rec["llm_roundtrip"][0]

    per_doc = _reconstruct_per_doc_axioms(rt["per_doc"])

    regime_results: list[dict] = []
    for name, rule_doc, transform in _REGIMES:
        per_doc_metrics: list[dict] = []
        per_doc_recall: dict[str, float] = {}
        for d in per_doc:
            m = _doc_metrics(d["source"], d["reconstructed"], transform)
            per_doc_metrics.append(m)
            per_doc_recall[d["doc_id"]] = round(m["recall"], 4)
        regime_results.append(
            {
                "name": name,
                "rules": rule_doc,
                "aggregate": _aggregate(per_doc_metrics),
                "per_doc_recall": per_doc_recall,
            }
        )

    payload = {
        "schema": "sum.s25_canonicalization_replay.v1",
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run": {
            "run_id": rec.get("run_id"),
            "git_sha": rec.get("git_sha"),
            "corpus": rt.get("corpus_id"),
            "n_docs": rt.get("n_roundtrips"),
            "model_snapshots": rec.get("model_snapshots"),
        },
        "regimes": regime_results,
        "interpretation_notes": (
            "Each regime is a measurement, not a recommendation. "
            "L0 sanity-checks the replay against the cached run. "
            "L1 isolates predicate-only canonicalisation. "
            "L2 adds last-name subject canonicalisation. "
            "L3 is the aggressive ceiling and conflates legitimate "
            "object-form differences. The receipt is the curve: each "
            "successive regime's contribution to exact-match recall "
            "tells the user where the gap is and where it is not."
        ),
    }

    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\ns2.5 replay written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
