"""Regenerate with `python -m worker.test.generate_unicode_fixture` from repo root."""
import hashlib
import json
from pathlib import Path

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.transform_receipt.format import compute_source_chain_hash

triples = [
    ["\U00010000", "p", "o"], ["\ue000", "p", "o"],
    ["a", "\U00010000", "o"], ["a", "\ue000", "o"],
    ["a", "p", "\U0001f600"], ["a", "p", "\uffff"],
    ["a,", "p", "o"], ["a", "z", "o"],
    ["é", "p", "o"], ["e\u0301", "p", "o"],
]
chain = [
    {"claim": claim, "provenance": {"source_uri": uri, "byte_start": 0, "byte_end": 1}}
    for claim in ["\U00010000", "\ue000"]
    for uri in ["local:\U00010000", "local:\ue000"]
]
fixture = {
    "schema": "sum.unicode_order_cross_runtime.v1",
    "reference": "Python sorted tuples and SUM JCS; no Unicode normalization",
    "triples": triples,
    "source_chain": chain,
    "source_chain_hash": compute_source_chain_hash(chain),
    "sorted_triples": sorted(triples),
    "triples_hash": "sha256-" + hashlib.sha256(canonicalize(sorted(triples))).hexdigest(),
    "density_half": sorted(triples, key=lambda t: "||".join(t))[:len(triples) // 2],
}
Path(__file__).with_name("fixtures").joinpath("unicode_order.json").write_text(
    json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
