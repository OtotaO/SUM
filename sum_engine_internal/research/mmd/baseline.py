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
    median_heuristic_bandwidth, mmd_permutation_pvalue, mmd_squared,
    rbf_kernel_matrix,
)


_REPO = Path(__file__).resolve().parents[3]


def _build_size_stratified_calibration(
    embeddings: np.ndarray,
    sigma: float,
    K_yy: np.ndarray,
    *,
    sizes: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50),
    n_subsamples_per_size: int = 60,
    seed: int = 0xCA11C,
) -> dict[int, np.ndarray]:
    """K3: size-stratified empirical MMD² distribution under H_0.

    For each subsample size in ``sizes``, draw ``n_subsamples_per_size``
    random subsamples of the baseline; compute MMD² between each
    subsample and the rest. The result maps size → array of MMD²
    values.

    Why stratified: MMD² between a small sample and a large
    reference is systematically larger than MMD² between two
    same-size samples (a sample-size confounder). Picking the
    threshold from a same-size calibration eliminates that
    confounder — the comparison becomes "is this bundle's MMD²
    larger than what we'd see for a *similar-sized* draw from
    the baseline?" — the actual question the substrate cares
    about.

    Sizes default to (1, 2, 3, 5, 10, 20, 50) covering the
    typical bundle-size range. At predict time we look up the
    closest size's distribution.
    """
    n = len(embeddings)
    out: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    full_indices = np.arange(n)
    for size in sizes:
        if n < size + 5:
            continue
        cal = np.zeros(n_subsamples_per_size, dtype=np.float64)
        for i in range(n_subsamples_per_size):
            idx = rng.choice(n, size=size, replace=False)
            complement = np.setdiff1d(full_indices, idx, assume_unique=True)
            X = embeddings[idx]
            Y = embeddings[complement]
            K_xx = rbf_kernel_matrix(X, X, sigma)
            K_xy = rbf_kernel_matrix(X, Y, sigma)
            K_yy_sub = K_yy[np.ix_(complement, complement)]
            s_xx = K_xx.sum() / (size * size)
            s_xy = K_xy.sum() / (size * len(complement))
            s_yy_sub = K_yy_sub.sum() / (len(complement) * len(complement))
            cal[i] = max(float(s_xx + s_yy_sub - 2.0 * s_xy), 0.0)
        out[size] = cal
    return out


def _closest_calibration_size(
    bundle_size: int, available_sizes: tuple[int, ...],
) -> int:
    """Pick the calibration size closest to the bundle size."""
    return min(available_sizes, key=lambda s: abs(s - bundle_size))
_BASELINE_CORPORA = (
    "seed_v1", "seed_v2", "seed_long_paragraphs", "seed_news_briefs",
    "seed_paragraphs", "seed_paragraphs_16",
)
_BUCKETS = 64  # same as RPCA axiom_embedding default


@dataclass(frozen=True, slots=True)
class _BaselineState:
    """Frozen baseline embedding matrix + median-heuristic
    bandwidth + within-baseline kernel-row sum (precomputed for
    speed) + conformal-style calibration MMD² distribution."""
    embeddings: np.ndarray  # shape (n_baseline, d)
    sigma: float            # bandwidth
    s_yy: float             # (1/m²) Σ K_yy precomputed
    n_samples: int
    # K3: empirical distribution of MMD² under H_0 (baseline
    # subsamples vs the rest of the baseline), STRATIFIED by
    # subsample size. Map size → array of MMD² values. At
    # predict time we look up the closest stratified size to
    # the actual bundle size; this avoids the
    # smaller-sample-larger-MMD² confounder.
    calibration_by_size: dict[int, np.ndarray] | None = None


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

        # K3: build the SIZE-STRATIFIED calibration distribution
        # of MMD² under H_0. For each subsample size in a covering
        # range, compute MMD² between many random subsamples and
        # the rest of the baseline. Eliminates the sample-size
        # confounder: at predict time we compare each bundle's
        # MMD² to the calibration for the closest matching size.
        cal_by_size = _build_size_stratified_calibration(
            embeddings, sigma, K_yy,
        )

        self._state = _BaselineState(
            embeddings=embeddings,
            sigma=sigma,
            s_yy=s_yy,
            n_samples=len(triples),
            calibration_by_size=cal_by_size,
        )
        return True

    def predict_mmd(
        self,
        bundle_triples: list,
        *,
        n_permutations: int = 200,
    ) -> Optional[dict]:
        """MMD² between ``bundle_triples`` and the baseline, plus
        Gretton 2012 permutation-test p-value for significance.

        Returns dict:
        ``{mmd_squared, permutation_p_value, n_permutations,
           bandwidth, n_baseline_samples, n_bundle_samples}``.
        Returns ``None`` if the computer isn't calibrated or the
        bundle is empty.

        The p-value is the substrate's "is this bundle
        significantly different from baseline?" answer at
        operator-chosen α — small p (< 0.05) ⇒ reject H_0 that
        bundle and baseline come from the same distribution.

        ``n_permutations`` defaults to 200 (gives p-values to
        ~0.005 resolution; ~60 ms at substrate scale). Set to 0
        to skip the permutation test entirely.
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
        n = X.shape[0]
        m = state.embeddings.shape[0]
        s_xx = K_xx.sum() / (n * n)
        s_xy = K_xy.sum() / (n * m)
        val = max(float(s_xx + state.s_yy - 2.0 * s_xy), 0.0)

        # Gretton 2012 §3.2 permutation test for significance
        if n_permutations > 0:
            import numpy as np
            _, p_value = mmd_permutation_pvalue(
                X, state.embeddings, state.sigma,
                n_permutations=n_permutations,
                rng=np.random.default_rng(0xB007),
            )
        else:
            p_value = None

        return {
            "mmd_squared": val,
            "permutation_p_value": float(p_value) if p_value is not None else None,
            "n_permutations": int(n_permutations),
            "bandwidth": float(state.sigma),
            "n_baseline_samples": int(state.n_samples),
            "n_bundle_samples": int(n),
        }

    @property
    def n_baseline_samples(self) -> int:
        return 0 if self._state is None else self._state.n_samples

    def predict_threshold(
        self,
        observed_mmd_squared: float,
        bundle_size: int,
        *,
        alpha: float = 0.10,
    ) -> Optional[dict]:
        """K3: conformal-style threshold decision for an observed
        MMD², calibrated against a SAME-SIZE subsample distribution.

        Returns dict:
        ``{threshold_alpha, threshold_value, exceeds_threshold,
           n_calibration_samples, calibration_size_used}``.
        Returns ``None`` if the computer isn't calibrated, the
        calibration table is empty, or alpha is out of (0, 1).

        The threshold is the ⌈(n+1)(1-α)⌉/n quantile (matching
        SplitConformal's finite-sample correction) of the
        calibration MMD² distribution at the size closest to the
        bundle's size. A bundle's MMD² > threshold ⇒ reject H_0
        at level α — the bundle is more atypical than what
        same-size in-distribution draws produce.
        """
        if not (0 < alpha < 1):
            return None
        if self._state is None:
            return None
        cal_by_size = self._state.calibration_by_size
        if not cal_by_size:
            return None
        import numpy as np
        available = tuple(sorted(cal_by_size.keys()))
        size_used = _closest_calibration_size(bundle_size, available)
        cal = cal_by_size[size_used]
        n = len(cal)
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        threshold = float(np.quantile(cal, q_level, method="higher"))
        return {
            "threshold_alpha": float(alpha),
            "threshold_value": threshold,
            "exceeds_threshold": bool(observed_mmd_squared > threshold),
            "n_calibration_samples": int(n),
            "calibration_size_used": int(size_used),
        }


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
