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


# Predicate library imported from the package (single source of
# truth). Curation rationale + iteration-2 evidence:
# docs/SMT_CONSISTENCY_SPIKE_FINDINGS.md.
from sum_engine_internal.research.smt_consistency.predicate_library import (
    SUBSTRATE_PREDICATE_LIBRARY as _LIBRARY,
)
# Stringly-typed shim so the existing receipt schema (which serialises
# property names as strings, not enum members) keeps working.
SUBSTRATE_PREDICATE_LIBRARY = {
    pred: {p.value for p in props}
    for pred, props in _LIBRARY.items()
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


def _experiment_real_corpus_injection(corpus_id: str) -> dict:
    """The needle-in-real-haystack test: inject a curated-library
    contradiction into a real corpus's axioms and confirm Z3
    catches it with a minimal UNSAT core."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.smt_consistency import check_consistency

    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    if not corpus_path.exists():
        return {"corpus_id": corpus_id, "skipped": True}

    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    clean = [
        Triple(*t)
        for doc in corpus["documents"]
        for t in sieve.extract_triplets(doc["text"])
    ]
    if not clean:
        return {"corpus_id": corpus_id, "skipped": True}

    used_predicates = {t.predicate for t in clean}
    props_payload = {
        p: _properties_for(p)
        for p in (used_predicates & SUBSTRATE_PREDICATE_LIBRARY.keys())
    }

    results = []

    # Test 1: irreflexive injection on a curated predicate
    irreflexive_preds = [
        p for p, v in SUBSTRATE_PREDICATE_LIBRARY.items()
        if "irreflexive" in v and p in used_predicates
    ]
    if irreflexive_preds:
        bad_pred = irreflexive_preds[0]
        with_self = clean + [
            Triple("__test_entity__", bad_pred, "__test_entity__"),
        ]
        r = check_consistency(with_self, predicate_properties=props_payload)
        results.append({
            "injection_type": "irreflexive_self_loop",
            "predicate_used": bad_pred,
            "n_after_injection": len(with_self),
            "caught_by_z3": not r.consistent,
            "unsat_core_size": len(r.unsat_core),
            "minimal_core": r.unsat_core == [len(clean)],
        })

    # Test 2: antisymmetric injection on a curated predicate
    antisym_preds = [
        p for p, v in SUBSTRATE_PREDICATE_LIBRARY.items()
        if "antisymmetric" in v and p in used_predicates
    ]
    if antisym_preds:
        bad_pred = antisym_preds[0]
        with_mutual = clean + [
            Triple("__test_a__", bad_pred, "__test_b__"),
            Triple("__test_b__", bad_pred, "__test_a__"),
        ]
        r = check_consistency(with_mutual, predicate_properties=props_payload)
        results.append({
            "injection_type": "antisymmetric_mutual",
            "predicate_used": bad_pred,
            "n_after_injection": len(with_mutual),
            "caught_by_z3": not r.consistent,
            "unsat_core_size": len(r.unsat_core),
            "minimal_core": set(r.unsat_core) == {len(clean), len(clean) + 1},
        })

    return {
        "corpus_id": corpus_id,
        "n_clean": len(clean),
        "n_curated_predicates_present": len(props_payload),
        "results": results,
    }


def _emit_receipt(syn, sub, inj, out_path: Path) -> dict:
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
        "experiment_real_corpus_injection": inj,
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

    print()
    print("=== Experiment 3: real-corpus injection (needle-in-haystack) ===")
    inj = []
    for cid in args.corpora.split(","):
        cid = cid.strip()
        try:
            r = _experiment_real_corpus_injection(cid)
            inj.append(r)
            if r.get("skipped"):
                print(f"  {cid}: skipped"); continue
            print(f"  {cid:>22s}: n_clean={r['n_clean']} curated_preds={r['n_curated_predicates_present']}")
            for tr in r["results"]:
                flag = "✓" if tr["caught_by_z3"] and tr["minimal_core"] else "✗"
                print(
                    f"    {flag} {tr['injection_type']:>22s}  pred={tr['predicate_used']:>10s}  "
                    f"caught={tr['caught_by_z3']}  core_size={tr['unsat_core_size']}  "
                    f"minimal={tr['minimal_core']}"
                )
        except FileNotFoundError:
            print(f"  {cid}: not found")

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"smt_consistency_substrate_spike_{ts}.json"
    else:
        out_path = Path(args.out)
    rec = _emit_receipt(syn, sub, inj, out_path)
    print()
    print(f"Receipt → {out_path}")
    print(f"Digest:  {rec['receipt_digest']}")


if __name__ == "__main__":
    main()
