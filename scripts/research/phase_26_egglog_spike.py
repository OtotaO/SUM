"""Phase 26.0 spike — egglog backing-store measurement.

Re-encodes substrate corpora as graphs in the egglog backend,
measures insert / query / determinism on the same workload that
PR #170's design doc named, and emits a
`sum.phase_26_backing_store_spike.v1` receipt.

This is the egglog candidate only; the design doc calls for
parallel runs against Neo4j and PostgreSQL+AGE. Those are
separate spike PRs (and they belong on a machine with the
backing-store services already provisioned). Egglog ships as
a Python wheel — zero infrastructure beyond `pip install` — so
it's the natural first candidate.

Pass criteria taken from `docs/PHASE_26_DESIGN.md` §4 risks:

  - Determinism: re-encoding the same triple set in a fresh
    process produces the same `content_hash` (sha256 over
    canonical-sorted bytes, independent of egglog internals).
  - Wall time: insert + query on the corpora completes well
    within interactive latency (< 1 s for the seed corpora,
    extrapolatable trend visible across n=120 / 1k / 10k).
  - Substrate parity: egglog-backed `find_*` queries return the
    same answers as a brute-force list scan over the same
    triple set.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT_DIR = REPO / "fixtures" / "bench_receipts"
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

# --- workloads -------------------------------------------------------


def _load_corpus_triples(corpus_id: str) -> list[tuple[str, str, str]]:
    """Extract triples from one of the substrate's seed corpora using
    the same DeterministicSieve the rest of the substrate uses, so the
    spike measures realistic axiom shapes."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_id}.json"
    with corpus_path.open() as f:
        corpus = json.load(f)
    sieve = DeterministicSieve()
    triples: list[tuple[str, str, str]] = []
    for doc in corpus["documents"]:
        for t in sieve.extract_triplets(doc["text"]):
            triples.append(tuple(t))
    return triples


def _synthetic_triples(n: int) -> list[tuple[str, str, str]]:
    """Synthetic workload for scale measurement. Subject /
    predicate / object cardinalities tuned to mimic the substrate's
    typical fan-out (each subject has ~3 predicates; each (s,p) has
    ~2 objects)."""
    out: list[tuple[str, str, str]] = []
    n_subjects = max(1, n // 6)
    for i in range(n):
        s = f"s{i % n_subjects}"
        p = f"p{(i // n_subjects) % 3}"
        o = f"o{i}"
        out.append((s, p, o))
    return out


# --- measurement helpers --------------------------------------------


def _measure_insert(triples, store):
    from sum_engine_internal.graph_store import Triple
    triple_objs = [Triple(*t) for t in triples]
    t0 = time.perf_counter()
    store.add_triples(triple_objs)
    return time.perf_counter() - t0


def _measure_queries(triples, store, sample_count: int = 100):
    """Run a representative mix: half find_objects, half find_subjects.
    Sample query keys from the actual triple set so every query has a
    hit and we measure the populated path."""
    if not triples:
        return {"sample_count": 0, "total_ms": 0.0, "per_query_us": 0.0}
    import random
    rng = random.Random(0xC0DEC0DE)
    sampled = rng.sample(triples, min(sample_count, len(triples)))
    half = len(sampled) // 2
    t0 = time.perf_counter()
    for s, p, o in sampled[:half]:
        store.find_objects(s, p)
    for s, p, o in sampled[half:]:
        store.find_subjects(p, o)
    elapsed = time.perf_counter() - t0
    return {
        "sample_count": len(sampled),
        "total_ms": round(elapsed * 1000, 3),
        "per_query_us": round((elapsed / max(1, len(sampled))) * 1_000_000, 3),
    }


def _verify_query_parity(triples, store, sample_count: int = 50):
    """Compare store's find_* answers to brute-force list scan
    answers. Sample some triples and check that both surfaces
    return the same set."""
    import random
    rng = random.Random(0xBA5E00)
    sampled = rng.sample(triples, min(sample_count, len(triples)))
    mismatches = 0
    for s, p, o in sampled:
        from_store = set(store.find_objects(s, p))
        ground = {oo for (ss, pp, oo) in triples if ss == s and pp == p}
        if from_store != ground:
            mismatches += 1
    return {"sample_count": len(sampled), "mismatches": mismatches}


def _verify_determinism(triples) -> dict:
    """Insert the triples into two fresh stores, in different orders,
    and check that both produce the same content_hash. Also persist
    the canonical hash so the receipt is comparable across machines."""
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    from sum_engine_internal.graph_store import Triple

    store_a = EgglogStore()
    store_a.add_triples([Triple(*t) for t in triples])
    hash_a = store_a.content_hash()

    store_b = EgglogStore()
    reversed_triples = list(reversed(triples))
    store_b.add_triples([Triple(*t) for t in reversed_triples])
    hash_b = store_b.content_hash()

    return {
        "hash_forward": hash_a,
        "hash_reversed": hash_b,
        "matches": hash_a == hash_b,
    }


def _run_workload(label: str, triples) -> dict:
    from sum_engine_internal.graph_store.egglog_store import EgglogStore
    store = EgglogStore()
    insert_s = _measure_insert(triples, store)
    info = store.info()
    return {
        "workload_label": label,
        "n_triples_input": len(triples),
        "n_distinct_after_insert": store.count_triples(),
        "n_egraph_registered": store.egraph_registered(),
        "insert_wall_s": round(insert_s, 4),
        "queries": _measure_queries(triples, store),
        "query_parity": _verify_query_parity(triples, store),
        "determinism": _verify_determinism(triples),
        "content_hash": store.content_hash(),
        "backend_info": {
            "name": info.name, "version": info.version, "notes": info.notes,
        },
    }


def _emit_receipt(results: list[dict], out_path: Path) -> dict:
    receipt = {
        "schema": "sum.phase_26_backing_store_spike.v1",
        "candidate": "egglog",
        "iso_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "workloads": results,
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["receipt_digest"] = (
        "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip the synthetic 10k workload (CI-friendly).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Override receipt output path. Default: timestamped in fixtures/bench_receipts/.",
    )
    args = parser.parse_args()

    workloads: list[tuple[str, list[tuple[str, str, str]]]] = [
        ("seed_long_paragraphs", _load_corpus_triples("seed_long_paragraphs")),
        ("seed_news_briefs", _load_corpus_triples("seed_news_briefs")),
        ("synthetic_1k", _synthetic_triples(1000)),
    ]
    if not args.quick:
        workloads.append(("synthetic_10k", _synthetic_triples(10_000)))

    results = [_run_workload(label, ts) for label, ts in workloads]

    if args.out is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RECEIPT_DIR / f"phase_26_backing_store_spike_egglog_{ts}.json"
    else:
        out_path = Path(args.out)

    receipt = _emit_receipt(results, out_path)

    print()
    print("=" * 72)
    print(f"egglog spike receipt → {out_path}")
    print(f"receipt_digest:       {receipt['receipt_digest']}")
    print()
    print(f"{'workload':25s} {'n_in':>6s} {'n_uniq':>6s} {'insert_s':>10s} {'q_us':>8s} {'det':>5s}")
    for r in results:
        det = "OK" if r["determinism"]["matches"] else "FAIL"
        parity = "✓" if r["query_parity"]["mismatches"] == 0 else "✗"
        print(
            f"{r['workload_label']:25s} "
            f"{r['n_triples_input']:>6d} "
            f"{r['n_distinct_after_insert']:>6d} "
            f"{r['insert_wall_s']:>10.4f} "
            f"{r['queries']['per_query_us']:>8.2f} "
            f"{det:>5s} "
            f"parity={parity}"
        )


if __name__ == "__main__":
    main()
