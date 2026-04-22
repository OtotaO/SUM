"""
Golden Benchmark Scoring Harness — Phase 19B

Measures extraction quality against the golden corpus:
    - Valid-schema rate: % of LLM outputs passing structural validation
    - Precision: correct triplets / total extracted
    - Recall: correct triplets / total gold-standard
    - F1: harmonic mean of precision and recall
    - Contradiction capture rate
    - Dedup quality (for duplicate-phrasing category)

Usage:
    # Offline mode (no API key needed) — validates corpus structure
    python Tests/benchmarks/extraction_benchmark.py --validate

    # Live mode (requires OPENAI_API_KEY)
    python Tests/benchmarks/extraction_benchmark.py --baseline

Author: ototao
License: Apache License 2.0
"""

import json
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sum_engine_internal.ensemble.extraction_validator import ExtractionValidator
from sum_engine_internal.algorithms.predicate_canon import canonicalize


CORPUS_PATH = Path(__file__).parent / "golden_corpus.json"


@dataclass
class BenchmarkResult:
    """Results from a single document extraction."""
    doc_id: str
    category: str
    gold_count: int
    extracted_count: int
    valid_count: int
    rejected_count: int
    true_positives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class CorpusMetrics:
    """Aggregate metrics across the entire corpus."""
    total_docs: int = 0
    total_gold_triplets: int = 0
    total_extracted: int = 0
    total_valid: int = 0
    total_rejected: int = 0
    total_true_positives: int = 0
    valid_schema_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


def normalize_triplet(s: str, p: str, o: str) -> Tuple[str, str, str]:
    """Normalize a triplet for comparison."""
    return (
        s.strip().lower().replace(" ", "_"),
        canonicalize(p.strip().lower().replace(" ", "_")),
        o.strip().lower().replace(" ", "_"),
    )


def triplet_match(extracted: Tuple[str, str, str], gold: Tuple[str, str, str]) -> bool:
    """
    Check if an extracted triplet matches a gold-standard triplet.
    Uses normalized exact match — the strictest possible comparison.
    """
    e_s, e_p, e_o = normalize_triplet(*extracted)
    g_s, g_p, g_o = normalize_triplet(*gold)
    return e_s == g_s and e_p == g_p and e_o == g_o


def compute_metrics(
    extracted: List[Tuple[str, str, str]],
    gold: List[Tuple[str, str, str]],
) -> Tuple[int, float, float, float]:
    """
    Compute precision, recall, F1 using strict triplet matching.
    Returns (true_positives, precision, recall, f1).
    """
    if not extracted and not gold:
        return 0, 1.0, 1.0, 1.0

    gold_normalized = [normalize_triplet(*g) for g in gold]
    extracted_normalized = [normalize_triplet(*e) for e in extracted]

    # True positives: extracted triplets that match any gold triplet
    matched_gold = set()
    tp = 0
    for e in extracted_normalized:
        for i, g in enumerate(gold_normalized):
            if i not in matched_gold and e == g:
                tp += 1
                matched_gold.add(i)
                break

    precision = tp / len(extracted_normalized) if extracted_normalized else 0.0
    recall = tp / len(gold_normalized) if gold_normalized else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return tp, precision, recall, f1


def validate_corpus() -> Dict[str, Any]:
    """
    Validate the golden corpus structure and report statistics.
    No API key needed.
    """
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)

    docs = corpus["documents"]
    validator = ExtractionValidator()

    stats = {
        "total_documents": len(docs),
        "by_category": {},
        "total_gold_triplets": 0,
        "gold_triplets_valid": 0,
        "gold_triplets_invalid": 0,
        "categories": {},
    }

    for doc in docs:
        cat = doc["category"]
        if cat not in stats["categories"]:
            stats["categories"][cat] = {"count": 0, "triplets": 0}
        stats["categories"][cat]["count"] += 1

        gold = [tuple(t) for t in doc["gold_triplets"]]
        stats["total_gold_triplets"] += len(gold)
        stats["categories"][cat]["triplets"] += len(gold)

        # Validate gold triplets themselves
        if gold:
            result = validator.validate_batch(gold, canonicalize_predicates=False)
            stats["gold_triplets_valid"] += result.accepted_count
            stats["gold_triplets_invalid"] += result.rejected_count

    return stats


def print_corpus_report(stats: Dict[str, Any]):
    """Pretty-print corpus validation report."""
    print("\n" + "=" * 60)
    print("  GOLDEN BENCHMARK CORPUS — VALIDATION REPORT")
    print("=" * 60)
    print(f"\n  Total documents: {stats['total_documents']}")
    print(f"  Total gold triplets: {stats['total_gold_triplets']}")
    print(f"  Valid gold triplets: {stats['gold_triplets_valid']}")
    print(f"  Invalid gold triplets: {stats['gold_triplets_invalid']}")
    print(f"\n  By category:")
    for cat, data in sorted(stats["categories"].items()):
        print(f"    {cat:30s}  {data['count']:3d} docs  {data['triplets']:3d} triplets")
    print("\n" + "=" * 60)

    if stats['gold_triplets_invalid'] > 0:
        print("  ⚠️  Some gold triplets fail structural validation!")
        print("     This is expected for negation docs with empty gold sets.")
    else:
        print("  ✅ All gold triplets pass structural validation")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Golden Benchmark Scoring Harness for SUM Extraction"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate corpus structure (no API key needed)"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run baseline extraction benchmark (requires OPENAI_API_KEY)"
    )
    args = parser.parse_args()

    if args.validate or (not args.baseline and not args.validate):
        stats = validate_corpus()
        print_corpus_report(stats)

    if args.baseline:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY required for --baseline mode")
            sys.exit(1)
        print("Baseline extraction benchmark not yet implemented.")
        print("This requires live LLM calls to measure extraction quality.")
        print("Use --validate to check corpus structure offline.")


if __name__ == "__main__":
    main()
