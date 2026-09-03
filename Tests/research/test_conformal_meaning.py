"""Contract for the distribution-free meaning-loss upper bound.

Two layers:
  - Contract layer: empirical coverage ≥ 1-δ (the provable dual
    guarantee — a valid (1-δ) upper bound must sit above the true loss
    rate at least 1-δ of the time).
  - Algebra layer: the bound is the exact dual of the rate kernel
    (upper-bound on E[loss] = 1 - lower-bound on E[1-loss]), is
    deterministic, rejects garbage, and is conservative (ub ≥ mean).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from sum_engine_internal.research.conformal.risk_control import (
    empirical_bernstein_lower_bound,
    hoeffding_lower_bound,
)
from sum_engine_internal.research.meaning.conformal_meaning import (
    MeaningRiskGuarantee,
    certify_meaning_risk,
    empirical_risk_coverage,
)


_SCORER = dict(scorer_name="lexical-coverage-bidirectional", scorer_version="1")


# ── Contract: empirical coverage ──────────────────────────────────────


@pytest.mark.parametrize(
    "method", ["hoeffding", "clopper_pearson", "empirical_bernstein"]
)
@pytest.mark.parametrize(
    ("true_loss_rate", "n", "delta"),
    [
        (0.1, 50, 0.05),
        (0.3, 100, 0.05),
        (0.5, 80, 0.10),
        (0.05, 200, 0.05),
    ],
)
def test_upper_bound_covers_true_loss_rate(true_loss_rate, n, delta, method):
    """A valid (1-δ) upper bound must achieve coverage ≥ 1-δ — for EVERY
    method, including the variance-adaptive empirical-Bernstein bound. The
    Monte-Carlo coverage is the receipt that eB's tighter radius stays
    sound on the loss (dual) side too."""
    coverage = empirical_risk_coverage(
        true_loss_rate, n, delta, method=method, n_trials=4000, seed=11
    )
    # ±0.01 flake band (≈4σ at n_trials=4000), matching the rate kernel's
    # coverage-test tolerance.
    assert coverage >= (1 - delta) - 0.01


def test_empirical_bernstein_is_dual_of_rate_kernel():
    """The meaning-risk eB ceiling is exactly 1 − (eB preservation LB) —
    the same duality the Hoeffding path obeys, so the variance-adaptive
    bound composes with the rest of the receipt machinery unchanged."""
    losses = [0.03, 0.0, 0.05, 0.02, 0.04, 0.01, 0.0, 0.03] * 8  # low-variance batch
    g = certify_meaning_risk(
        losses, delta=0.05, method="empirical_bernstein", **_SCORER
    )
    preservations = [1.0 - x for x in losses]
    lb = empirical_bernstein_lower_bound(preservations, 0.05)
    assert g.risk_upper_bound == pytest.approx(1.0 - lb, abs=1e-12)
    assert g.method == "empirical_bernstein"


def test_empirical_bernstein_tighter_meaning_ceiling_at_batch():
    """The product win (F22): on a faithful, low-variance batch eB
    certifies a LOWER meaning-loss ceiling than Hoeffding — a useful
    receipt where Hoeffding's was near-vacuous."""
    losses = [0.03] * 200  # faithful batch, preservation 0.97
    g_ho = certify_meaning_risk(losses, delta=0.05, method="hoeffding", **_SCORER)
    g_eb = certify_meaning_risk(
        losses, delta=0.05, method="empirical_bernstein", **_SCORER
    )
    assert g_eb.risk_upper_bound < g_ho.risk_upper_bound


# ── Algebra: duality with the rate kernel ─────────────────────────────


def test_bound_is_dual_of_rate_kernel():
    losses = [0.0, 0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25]
    g = certify_meaning_risk(losses, delta=0.05, method="hoeffding", **_SCORER)
    preservations = [1.0 - x for x in losses]
    lb = hoeffding_lower_bound(preservations, 0.05)
    assert g.risk_upper_bound == pytest.approx(1.0 - lb, abs=1e-12)


def test_point_estimate_is_mean_loss():
    losses = [0.0, 0.2, 0.4, 0.6]
    g = certify_meaning_risk(losses, **_SCORER)
    assert g.point_estimate == pytest.approx(np.mean(losses))


def test_bound_is_conservative():
    """The certified ceiling sits at or above the observed mean — slack
    is the price of rigour."""
    losses = [0.1, 0.2, 0.15, 0.05, 0.25, 0.1]
    g = certify_meaning_risk(losses, **_SCORER)
    assert g.risk_upper_bound >= g.point_estimate
    assert g.slack >= 0.0


def test_more_data_tightens_bound():
    rng = np.random.RandomState(3)
    small = rng.uniform(0.0, 0.3, size=20).tolist()
    large = rng.uniform(0.0, 0.3, size=500).tolist()
    g_small = certify_meaning_risk(small, **_SCORER)
    g_large = certify_meaning_risk(large, **_SCORER)
    assert g_large.slack < g_small.slack


def test_scorer_identity_is_carried():
    g = certify_meaning_risk([0.1, 0.2], **_SCORER)
    assert g.scorer_name == "lexical-coverage-bidirectional"
    assert g.scorer_version == "1"


def test_confidence_is_one_minus_delta():
    g = certify_meaning_risk([0.1, 0.2], delta=0.05, **_SCORER)
    assert g.confidence == pytest.approx(0.95)


def test_controls_at_level():
    # 400 perfectly-preserved pairs: Hoeffding radius ≈ 0.061, so the
    # certified ceiling clears 0.10. Fewer samples could not — control
    # is a function of sample size, not just observed loss.
    g = certify_meaning_risk([0.0] * 400, **_SCORER)
    assert g.controls(0.10) is True
    g2 = certify_meaning_risk([0.9] * 20, **_SCORER)  # heavy loss
    assert g2.controls(0.10) is False


def test_small_n_cannot_certify_tight_control():
    """The proof-boundary in action: even perfect observed preservation
    over only 4 pairs cannot certify control at 0.5 — the finite-sample
    radius is wider than that. The certificate refuses to overclaim."""
    g = certify_meaning_risk([0.0] * 4, **_SCORER)
    assert g.point_estimate == 0.0       # proxy sees zero loss
    assert g.controls(0.5) is False      # but 4 samples can't prove it


# ── Hardening: garbage in → error, not a fabricated bound ─────────────


def test_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        certify_meaning_risk([0.1, float("nan"), 0.2], **_SCORER)


def test_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        certify_meaning_risk([0.1, 1.5], **_SCORER)


def test_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        certify_meaning_risk([], **_SCORER)


def test_clopper_pearson_requires_binary():
    with pytest.raises(ValueError, match="binary"):
        certify_meaning_risk([0.1, 0.2], method="clopper_pearson", **_SCORER)


def test_auto_picks_clopper_pearson_for_binary():
    g = certify_meaning_risk([0.0, 1.0, 0.0, 0.0], method="auto", **_SCORER)
    assert g.method == "clopper_pearson"


def test_auto_picks_hoeffding_for_fractional():
    g = certify_meaning_risk([0.0, 0.5, 0.3], method="auto", **_SCORER)
    assert g.method == "hoeffding"


def test_is_deterministic():
    losses = [0.1, 0.2, 0.3, 0.15]
    a = certify_meaning_risk(losses, **_SCORER)
    b = certify_meaning_risk(losses, **_SCORER)
    assert a == b


# ── Issuer/verifier kernel parity on the quantised micro grid ─────────
#
# The issuer (this module) and the shipped verifier
# (``sum_verify._conformal``) must agree on ``point_estimate`` to the last
# micro-unit, or a minted receipt fails its own replay check. The verifier
# uses ``math.fsum(values) / len(values)``; ``np.ndarray.mean`` uses pairwise
# summation. On the micro grid the two land either side of a round-half-even
# tie for some even-n inputs, which is exactly the seam these tests pin.

_MICRO = 1_000_000
# Micro values chosen so a small exhaustive product still surfaces ties.
_TIE_POOL = (1, 3, 999_997, 999_999)


def _micro(x: float) -> int:
    return int(round(float(x) * _MICRO))


def _numpy_mean(values):
    """The pre-fix issuer kernel, spelled exactly as it was written."""
    return float(np.asarray(values, dtype=np.float64).mean())


def _fsum_mean(values):
    """The shipped verifier's kernel (``sum_verify._conformal._mean``)."""
    return math.fsum(values) / len(values)


def _divergent_micro_grid_cases():
    """Brute-force the even-n quantised micro grid (values k/1e6) for
    inputs where the numpy mean and the fsum mean disagree *after*
    rounding to micro."""
    cases = []
    for n in (2, 4, 6, 8):
        for combo in itertools.product(_TIE_POOL, repeat=n):
            values = [m / _MICRO for m in combo]
            if _micro(_numpy_mean(values)) != _micro(_fsum_mean(values)):
                cases.append(values)
    return cases


def test_micro_grid_actually_has_numpy_vs_fsum_ties():
    """The seam exists. Without this the parity tests below could pass
    vacuously on a grid that never ties."""
    cases = _divergent_micro_grid_cases()
    assert cases, (
        "no numpy-vs-fsum micro divergence found on the tie pool; the "
        "parity tests below would be vacuous"
    )


def test_point_estimate_uses_the_sdk_fsum_kernel_on_micro_ties():
    """On every input where the two kernels diverge, the issuer must now
    report the fsum value, byte-identical to what the shipped verifier
    recomputes during replay."""
    cases = _divergent_micro_grid_cases()
    assert cases
    for values in cases:
        want = _fsum_mean(values)
        g = certify_meaning_risk(values, delta=0.05, method="hoeffding", **_SCORER)
        assert g.point_estimate == want, values
        assert _micro(g.point_estimate) == _micro(want), values


def test_issuer_and_sdk_agree_on_point_estimate_for_tie_inputs():
    """End to end: the research certifier and ``sum_verify``'s certifier
    return the same ``point_estimate``, exactly and to the micro, on the
    inputs that used to split them."""
    from sum_verify import _conformal as sdk

    cases = _divergent_micro_grid_cases()
    assert cases
    for values in cases:
        c = certify_meaning_risk(values, delta=0.05, method="hoeffding", **_SCORER)
        s = sdk.certify_meaning_risk(
            values, delta=0.05, method="hoeffding", **_SCORER
        )
        assert c.point_estimate == s.point_estimate, values
        assert _micro(c.point_estimate) == _micro(s.point_estimate), values
        # the bound itself never diverged; guard that it stays that way
        assert _micro(c.risk_upper_bound) == _micro(s.risk_upper_bound), values
