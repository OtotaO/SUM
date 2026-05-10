"""SMT-backed axiom-consistency substrate spike.

Two experiments:

  1. **Synthetic verification matrix** — five canonical
     contradiction patterns (mutual-antisymmetric, self-loop on
     irreflexive, two-output on functional, transitive cycle,
     mixed clean+contradicting) all detected with minimal UNSAT
     cores.

  2. **Substrate corpus check** — extract axioms from real
     corpora via the deterministic sieve; declare a small
     predicate library matched to common substrate predicates
     (e.g. ``visit`` / ``born_on`` / ``parent_of``); check
     consistency. Surfaces whether real-corpus axioms contain
     latent contradictions and how fast Z3 returns the verdict.

Receipt: ``sum.smt_consistency_substrate_spike.v1``.
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

# A starter predicate library — operator-curated, conservative.
# Add to this as the substrate's vocabulary grows.
SUBSTRATE_PREDICATE_LIBRARY = {
    "parent_of": {"antisymmetric", "irreflexive"},
    "child_of": {"antisymmetric", "irreflexive"},
    "ancestor_of": {"transitive", "irreflexive"},
    "born_on": {"functional"},
    "born_in": {"functional"},
    "married_to": {"irreflexive"},
    # NOTE: these are illustrative — real substrate use needs
    # operator-vetted properties per predicate.
}


def _experiment_synthetic_verification() -> list[dict]:
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.smt_consistency import (
        PredicateProperty as P, check_consistency,
    )
    t = Triple
    cases = [
        ("clean_no_props",
         [t("alice", "knows", "bob")], None, True),
        ("mutual_antisymmetric",
         [t("alice", "parent_of", "bob"), t("bob", "parent_of", "alice")],
         {"parent_of": {P.ANTISYMMETRIC}}, False),
        ("self_loop_irreflexive",
         [t("alice", "parent_of", "alice")],
         {"parent_of": {P.IRREFLEXIVE}}, False),
        ("two_outputs_functional",
         [t("alice", "born_on", "1990-01-01"),
          t("alice", "born_on", "1991-02-02")],
         {"born_on": {P.FUNCTIONAL}}, False),
        ("3_cycle_transitive_irreflexive",
         [t("a", "ancestor_of", "b"),
          t("b", "ancestor_of", "c"),
          t("c", "ancestor_of", "a")],
         {"ancestor_of": {P.TRANSITIVE, P.IRREFLEXIVE}}, False),
        ("needle_in_haystack",
         [t(f"e{i}", "knows", f"e{i+1}") for i in range(50)]
         + [t("alice", "parent_of", "alice")],
         {"parent_of": {P.IRREFLEXIVE}}, False),
    ]
    out = []
    for name, triples, props, expected in cases:
        r = check_consistency(triples, predicate_properties=props)
        out.append({
            "case": name,
            "n_triples": r.n_triples,
            "expected_consistent": expected,
            "got_consistent": r.consistent,
            "passed": r.consistent == expected,
            "unsat_core_size": len(r.unsat_core),
            "z3_check_ms": round(r.z3_check_seconds * 1000, 3),
        })
    return out


def _properties_for(name: str):
    """Map a string name from SUBSTRATE_PREDICATE_LIBRARY to the
    PredicateProperty enum set."""
    from sum_engine_internal.research.smt_consistency import PredicateProperty
    mapping = {p.value: p for p in PredicateProperty}
    return {mapping[n] for n in SUBSTRATE_PREDICATE_LIBRARY.get(name, set())}


def _experiment_substrate_corpus(corpus_id: str) -> dict:
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.smt_consistency import check_consistency

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    triples = [
        Triple(*t)
        for doc in corpus["documents"]
        for t in sieve.extract_triplets(doc["text"])
    ]
    # Filter to predicates the library covers, plus all clean
    # axioms (so the corpus's full structure is checked, not just
    # the constrained subset)
    used_predicates = {t.predicate for t in triples}
    predicates_with_props = used_predicates & SUBSTRATE_PREDICATE_LIBRARY.keys()
    props_payload = {p: _properties_for(p) for p in predicates_with_props}

    r = check_consistency(triples, predicate_properties=props_payload)
    return {
        "corpus_id": corpus_id,
        "n_triples": r.n_triples,
        "n_distinct_predicates": len(used_predicates),
        "predicates_with_properties_declared": sorted(predicates_with_props),
        "consistent": r.consistent,
        "unsat_core_indices": r.unsat_core,
        "unsat_core_triples": [
            list(triples[i].as_tuple()) for i in r.unsat_core
        ] if r.unsat_core else [],
        "z3_check_ms": round(r.z3_check_seconds * 1000, 3),
    }


def _emit_receipt(syn, sub, out_path: Path) -> dict:
    receipt = {
        "schema": "sum.smt_consistency_substrate_spike.v1",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "predicate_library": {
            k: sorted(v) for k, v in SUBSTRATE_PREDICATE_LIBRARY.items()
        },
        "experiment_synthetic_verification": syn,
        "experiment_substrate_corpus_check": sub,
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
        default="seed_v1,seed_v2,seed_long_paragraphs,seed_news_briefs",
    )
    args = parser.parse_args()

    print("=== Experiment 1: synthetic verification matrix ===")
    syn = _experiment_synthetic_verification()
    for r in syn:
        flag = "✓" if r["passed"] else "✗"
        print(
            f"  {flag} {r['case']:>32s}: n={r['n_triples']:3d} "
            f"expected={r['expected_consistent']} got={r['got_consistent']} "
            f"core={r['unsat_core_size']}  ({r['z3_check_ms']:.1f}ms)"
        )

    print()
    print("=== Experiment 2: substrate corpus check ===")
    sub = []
    for cid in args.corpora.split(","):
        cid = cid.strip()
        try:
            r = _experiment_substrate_corpus(cid)
            sub.append(r)
            verdict = "CONSISTENT" if r["consistent"] else "INCONSISTENT"
            preds = ", ".join(r["predicates_with_properties_declared"]) or "(none)"
            print(
                f"  {cid:>22s}: {verdict:12s}  triples={r['n_triples']:3d}  "
                f"preds_with_props=[{preds}]  "
                f"core_size={len(r['unsat_core_indices'])}  "
                f"({r['z3_check_ms']:.1f}ms)"
            )
            if r["unsat_core_triples"]:
                for tr in r["unsat_core_triples"]:
                    print(f"      ▸ {tr}")
        except FileNotFoundError:
            print(f"  {cid}: not found")

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"smt_consistency_substrate_spike_{ts}.json"
    else:
        out_path = Path(args.out)
    rec = _emit_receipt(syn, sub, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {rec['receipt_digest']}")


if __name__ == "__main__":
    main()
