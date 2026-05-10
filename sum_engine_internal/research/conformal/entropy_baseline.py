"""Calibrated CI on the substrate's per-bundle axiom-graph entropy.

Wraps PR #184's `graph_entropy()` output with a finite-sample,
distribution-free interval via PR #183's `SplitConformal`. The
predictor is calibrated once at module load against a precomputed
baseline of (axiom_count, entropy) pairs from the substrate's
labeled seed corpora.

Substrate use: every signed bundle gains an
``axiom_graph_entropy_ci`` field — a CI for "what entropy is
typical at this bundle's axiom_count, given the substrate's
calibration corpus." Cross-bundle anomaly detection becomes a
binary in-or-out check on the bundle's own metadata, without any
multi-bundle history.

The predictor is intentionally simple: ridge regression
``predicted_entropy = a + b · log(1 + axiom_count)`` on a training
fold, then split conformal on residuals on a calibration fold.
The (1-α) CI is centered on the predicted value with the
finite-sample-corrected quantile width.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from sum_engine_internal.research.conformal.split_conformal import (
    ConformalInterval, SplitConformal,
)


_BASELINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "fixtures" / "calibration" / "entropy_baseline.json"
)


@dataclass(frozen=True, slots=True)
class _RidgeModel:
    """Tiny univariate ridge regression: y = intercept + slope * x.

    Stored separately from SplitConformal so the calibration
    process is transparent (we don't hide the predictor inside the
    conformal wrapper)."""
    intercept: float
    slope: float

    def predict(self, x_value: float) -> float:
        return self.intercept + self.slope * x_value

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * x


class BaselineEntropyPredictor:
    """Conformal-CI-wrapped entropy predictor for the substrate.

    Constructed empty; ``calibrate_from_baseline()`` loads the
    precomputed ``fixtures/calibration/entropy_baseline.json``
    and runs ridge + split conformal on it. Subsequent
    ``predict_ci(axiom_count)`` returns a finite-sample,
    distribution-free CI under the calibration corpus.

    Cold-start design: if the baseline file is missing or has
    too few pairs, the predictor reports unavailable rather than
    raising — this keeps the load-bearing
    ``canonical_codec.export_bundle`` path resilient.
    """

    MIN_CALIBRATION_PAIRS = 20

    def __init__(self) -> None:
        self._ridge: Optional[_RidgeModel] = None
        self._conformal: Optional[SplitConformal] = None
        self._n_train: int = 0
        self._n_cal: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._ridge is not None and self._conformal is not None

    def calibrate_from_baseline(
        self,
        baseline_path: Optional[Path] = None,
        *,
        alpha: float = 0.10,
        train_frac: float = 0.5,
    ) -> bool:
        """Load + train + calibrate. Returns True on success,
        False if the baseline is unavailable or too small.
        Never raises."""
        path = baseline_path or _BASELINE_PATH
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        pairs = data.get("pairs", [])
        if len(pairs) < self.MIN_CALIBRATION_PAIRS:
            return False
        return self._fit(pairs, alpha=alpha, train_frac=train_frac)

    def _fit(
        self, pairs: list[dict], *,
        alpha: float = 0.10, train_frac: float = 0.5,
        seed: int = 0xCA11B,
    ) -> bool:
        x = np.array(
            [np.log1p(p["axiom_count"]) for p in pairs], dtype=np.float64,
        )
        y = np.array(
            [p["axiom_graph_entropy"] for p in pairs], dtype=np.float64,
        )

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x))
        n_train = max(int(train_frac * len(x)), 1)
        train_idx = idx[:n_train]
        cal_idx = idx[n_train:]
        if len(cal_idx) < 5:
            return False

        # Ridge with tiny λ — basically OLS at this scale, but
        # numerically stable. Closed form for univariate.
        lam = 1e-6
        x_tr = x[train_idx]; y_tr = y[train_idx]
        # design matrix [1, x]
        X = np.column_stack([np.ones_like(x_tr), x_tr])
        A = X.T @ X + lam * np.eye(2)
        b = X.T @ y_tr
        w = np.linalg.solve(A, b)
        ridge = _RidgeModel(intercept=float(w[0]), slope=float(w[1]))

        # Split conformal on calibration residuals
        cal_pred = ridge.predict_array(x[cal_idx])
        cal_targets = y[cal_idx]
        sc = SplitConformal(alpha=alpha, score_type="absolute")
        sc.calibrate(cal_pred, cal_targets)

        self._ridge = ridge
        self._conformal = sc
        self._n_train = len(train_idx)
        self._n_cal = len(cal_idx)
        return True

    def predict_ci(self, axiom_count: int) -> Optional[ConformalInterval]:
        """Return the calibrated CI for the expected entropy at
        this axiom_count. None if not calibrated or input is
        nonsensical (axiom_count ≤ 0)."""
        if not self.is_calibrated:
            return None
        if axiom_count <= 0:
            return None
        x = float(np.log1p(axiom_count))
        point = self._ridge.predict(x)
        return self._conformal.predict(point)

    @property
    def n_calibration_pairs(self) -> int:
        return self._n_cal

    @property
    def n_training_pairs(self) -> int:
        return self._n_train


# Module-level singleton, lazy-initialised so import time stays cheap
_default_predictor: Optional[BaselineEntropyPredictor] = None


def get_default_predictor() -> BaselineEntropyPredictor:
    """Lazy singleton accessor. Calibrates once on first call;
    subsequent calls return the same instance."""
    global _default_predictor
    if _default_predictor is None:
        p = BaselineEntropyPredictor()
        p.calibrate_from_baseline()  # silently no-op if unavailable
        _default_predictor = p
    return _default_predictor
