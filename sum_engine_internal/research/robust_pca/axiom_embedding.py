"""Deterministic axiom → feature-vector embedding for PCP input.

The substrate's axioms are (subject, predicate, object) string
triples. PCP wants a real-valued matrix M ∈ ℝ^{n × d} where each
row is one axiom. The embedding choice determines what "low-rank
manifold" means:

  - Predicate one-hot ⇒ axioms sharing a predicate cluster on the
    same coordinate. With ~k distinct predicates in a corpus,
    this block has rank ≤ k.
  - Subject / object hash buckets ⇒ axioms sharing entities
    cluster too (within a document, entities recur).
  - Concatenating the three blocks gives a 3·B-dimensional
    embedding where B is the number of buckets per role.

This is the simplest deterministic embedding that produces a
matrix where:
  - The bulk of axioms live on a low-rank manifold (because
    predicates and entities recur within and across documents)
  - Corrupted axioms (mismatched s/p/o, off-corpus entities)
    appear as sparse off-manifold residuals

We deliberately do not use a learned embedding (sentence
encoder, BERT, etc.) for the spike: a deterministic embedding
keeps the experiment reproducible without ML dependencies and
matches the substrate's PROOF_BOUNDARY discipline (no
"learned-but-unverified" components in the load-bearing path).
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from sum_engine_internal.graph_store import Triple


def _hash_to_bucket(value: str, n_buckets: int) -> int:
    """Deterministic string → bucket index. Uses sha256 for
    stability across Python versions (vs Python's built-in
    `hash` which is randomised per process)."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_buckets


def embed_triple(t: Triple, *, n_buckets: int = 64) -> np.ndarray:
    """One-hot embed a single triple into a 3·n_buckets vector.

    Layout: [subject_one_hot | predicate_one_hot | object_one_hot]
    The fixed n_buckets ⇒ collisions when corpus diversity exceeds
    the bucket count, but for our spike workloads (≤ ~150 distinct
    entities per corpus) collisions are negligible at n_buckets=64.
    """
    out = np.zeros(3 * n_buckets, dtype=np.float64)
    out[_hash_to_bucket(t.subject, n_buckets)] = 1.0
    out[n_buckets + _hash_to_bucket(t.predicate, n_buckets)] = 1.0
    out[2 * n_buckets + _hash_to_bucket(t.object, n_buckets)] = 1.0
    return out


def embed_triples(
    triples: Iterable[Triple], *, n_buckets: int = 64,
) -> np.ndarray:
    """Stack many triples into an n × (3·n_buckets) matrix."""
    triples = list(triples)
    if not triples:
        return np.zeros((0, 3 * n_buckets), dtype=np.float64)
    return np.vstack([embed_triple(t, n_buckets=n_buckets) for t in triples])
