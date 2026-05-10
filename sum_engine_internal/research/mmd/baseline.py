"""Per-bundle MMD against the substrate's calibration baseline.

Each new bundle gets a single MMD² scalar measuring its
axiom-distribution distance from a precomputed reference set
of triples drawn from the substrate's seed corpora. Cross-bundle
distribution-shift detection becomes a single scalar on bundle
metadata.

The baseline computer is intentionally lazy + cold-start safe:
if the calibration corpora aren't loadable, ``predict_mmd``
returns ``None`` and attestation continues unaffected (the
defense-in-depth pattern from wires #1, #2, #3).

Embedding choice: reuses the ``embed_triples`` deterministic
sha256-bucket embedding from PR #182's RPCA module. RBF kernel
on those vectors gives a proper kernel-mean MMD without any
new dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from sum_engine_internal.research.mmd.mmd import (
    median_heuristic_bandwidth, mmd_squared, rbf_kernel_matrix,
)


_REPO = Path(__file__).resolve().parents[3]
_BASELINE_CORPORA = (
    "seed_v1", "seed_v2", "seed_long_paragraphs", "seed_news_briefs",
    "seed_paragraphs", "seed_paragraphs_16",
)
_BUCKETS = 64  # same as RPCA axiom_embedding default


@dataclass(frozen=True, slots=True)
class _BaselineState:
    """Frozen baseline embedding matrix + median-heuristic
    bandwidth + within-baseline kernel-row sum (precomputed for
    speed)."""
    embeddings: np.ndarray  # shape (n_baseline, d)
    sigma: float            # bandwidth
    s_yy: float             # (1/m²) Σ K_yy precomputed
    n_samples: int


class BaselineMMDComputer:
    """Builds the baseline once at construction, then answers
    ``mmd_against_baseline(triples)`` in O(n·m) where n = bundle
    triples, m = baseline triples.

    Lazy-loaded: ``calibrate_from_corpora()`` is the entry point;
    silently no-ops if the corpora are unavailable. Failures
    leave ``is_calibrated == False``; ``predict_mmd`` returns None.
    """

    def __init__(self) -> None:
        self._state: Optional[_BaselineState] = None

    @property
    def is_calibrated(self) -> bool:
        return self._state is not None

    def calibrate_from_corpora(
        self,
        corpora: tuple[str, ...] | None = None,
    ) -> bool:
        """Load triples from the listed corpora, embed them, fit
        the median-heuristic bandwidth, precompute the within-
        baseline kernel sum. Returns True on success."""
        try:
            from sum_engine_internal.algorithms.syntactic_sieve import (
                DeterministicSieve,
            )
            from sum_engine_internal.graph_store import Triple
            from sum_engine_internal.research.robust_pca import embed_triples
        except ImportError:
            return False

        sieve = DeterministicSieve()
        triples: list = []
        for cid in (corpora or _BASELINE_CORPORA):
            corpus_path = (
                _REPO / "scripts" / "bench" / "corpora" / f"{cid}.json"
            )
            if not corpus_path.exists():
                continue
            try:
                corpus = json.loads(corpus_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for doc in corpus.get("documents", []):
                for t in sieve.extract_triplets(doc.get("text", "")):
                    triples.append(Triple(*t))
        if len(triples) < 10:
            return False

        embeddings = embed_triples(triples, n_buckets=_BUCKETS)
        # Median heuristic on the baseline alone (a one-sample
        # heuristic is fine when test samples have similar shape;
        # bundles use the same embedding so distance distribution
        # is comparable)
        sigma = median_heuristic_bandwidth(embeddings)
        K_yy = rbf_kernel_matrix(embeddings, embeddings, sigma)
        s_yy = float(K_yy.sum() / (len(embeddings) ** 2))
        self._state = _BaselineState(
            embeddings=embeddings,
            sigma=sigma,
            s_yy=s_yy,
            n_samples=len(triples),
        )
        return True

    def predict_mmd(self, bundle_triples: list) -> Optional[dict]:
        """MMD² between ``bundle_triples`` and the baseline.

        ``bundle_triples`` is a list of ``Triple``. Returns dict:
        ``{mmd_squared, bandwidth, n_baseline_samples,
           n_bundle_samples}``. Returns ``None`` if the computer
        isn't calibrated or the bundle is empty.
        """
        if not self.is_calibrated:
            return None
        if not bundle_triples:
            return None
        try:
            from sum_engine_internal.research.robust_pca import embed_triples
        except ImportError:
            return None
        state = self._state
        assert state is not None
        X = embed_triples(bundle_triples, n_buckets=_BUCKETS)
        K_xx = rbf_kernel_matrix(X, X, state.sigma)
        K_xy = rbf_kernel_matrix(X, state.embeddings, state.sigma)
        # Inline the MMD² since we have s_yy precomputed
        n = X.shape[0]
        m = state.embeddings.shape[0]
        s_xx = K_xx.sum() / (n * n)
        s_xy = K_xy.sum() / (n * m)
        val = max(float(s_xx + state.s_yy - 2.0 * s_xy), 0.0)
        return {
            "mmd_squared": val,
            "bandwidth": float(state.sigma),
            "n_baseline_samples": int(state.n_samples),
            "n_bundle_samples": int(n),
        }

    @property
    def n_baseline_samples(self) -> int:
        return 0 if self._state is None else self._state.n_samples


# Module-level singleton, lazy-initialised so import time stays
# cheap and the canonical_codec hot path doesn't pay re-load cost
# per bundle.
_default_computer: Optional[BaselineMMDComputer] = None


def get_default_mmd_computer() -> BaselineMMDComputer:
    """Lazy singleton accessor. Calibrates once on first call;
    subsequent calls return the same instance. Failures during
    calibration leave the computer uncalibrated — callers must
    check ``is_calibrated``."""
    global _default_computer
    if _default_computer is None:
        c = BaselineMMDComputer()
        c.calibrate_from_corpora()  # silently no-op if unavailable
        _default_computer = c
    return _default_computer
