"""Exchangeability advisory — does a meaning-risk bound apply to YOUR text?

Every conformal meaning-risk bound (``conformal_meaning.py``) and every
drift-budget (``drift_budget.py``) is valid *only under exchangeability*
between the calibration corpus and deployment. Until now that caveat lived
**only in prose** — the receipt names ``corpus_id`` but offers no signal
that a reader's deployment text is actually exchangeable with it.

This module turns that prose into a *measured advisory*. It embeds the
calibration corpus and a deployment batch with the receipt's **named judge
embedder**, then runs the in-repo MMD two-sample permutation test
(``research.mmd``) between them. A significant result is **evidence against
exchangeability** — the certified bound may not apply to this deployment,
so it should not be quoted out there.

The honest boundary, which is the whole point:

  - **ADVISORY, NEVER GATING.** This does not change, invalidate, or
    re-sign any bound. It is a separate measurement a consumer runs to
    decide whether the bound is *applicable* to their data. A meaning-risk
    receipt's cryptographic verification and bound-replay are unaffected.
  - **Asymmetric evidence.** A *significant* p is evidence the deployment
    distribution differs from calibration (→ do not quote the bound). A
    *non-significant* p is **consistent with** exchangeability but does NOT
    prove it — a two-sample test cannot accept the null. Absence of a
    detected shift is not a certificate of exchangeability.
  - **Judge/hardware-pinned.** The embeddings come from a model forward
    pass; like the loss computation (F23/F26) they are reproducible only on
    a matching stack. So this is a *measurement*, not a same-commit-
    replayable certificate — it is emitted as an unsigned report
    (``sum.exchangeability_advisory.v1``), never folded into a signed
    receipt payload (that would put an unreplayable number in a signed
    field — the discipline this project holds).

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

_MICRO = 1_000_000
# Default permutation seed — fixing it makes the advisory reproducible on a
# matching judge stack (the only non-determinism is the embedder + the
# permutation draw; the embedder is pinned, this pins the draw).
_DEFAULT_SEED = 0xB007


@dataclass(frozen=True, slots=True)
class ExchangeabilityAdvisory:
    """A measured advisory on whether a deployment batch is exchangeable
    with a named calibration corpus, under a named judge embedder.

    ``distinguishable`` is the operative bit: True ⇒ the MMD test found the
    two samples distinguishable at level ``alpha`` ⇒ **evidence against
    exchangeability** ⇒ the certified bound for ``calibration_corpus_id``
    may not apply to this deployment and should not be quoted for it.
    """

    calibration_corpus_id: str
    judge: str
    judge_version: str
    n_calibration: int
    n_deployment: int
    mmd2: float
    p_value: float
    n_permutations: int
    seed: int
    bandwidth: float
    alpha: float

    @property
    def distinguishable(self) -> bool:
        """True iff the two samples are distinguishable at ``alpha`` —
        evidence AGAINST exchangeability."""
        return self.p_value < self.alpha

    @property
    def verdict(self) -> str:
        if self.distinguishable:
            return (
                "shift-detected: deployment is distinguishable from the "
                "calibration corpus — treat the certified bound as possibly "
                "OUT-OF-SCOPE for this deployment; do not quote it"
            )
        return (
            "no-shift-detected: consistent with exchangeability (NOT proof "
            "of it — a two-sample test cannot accept the null)"
        )

    @property
    def scope(self) -> str:
        return (
            "ADVISORY, never gating — a measurement under a named judge "
            "embedder, not a certificate and not a re-signing of any bound. "
            "A significant result is evidence AGAINST exchangeability; a "
            "non-significant result is consistent with but does not prove "
            "exchangeability. Embeddings are judge/hardware-pinned (F23/F26)"
        )


def assess_exchangeability(
    calibration_embeddings: Any,
    deployment_embeddings: Any,
    *,
    calibration_corpus_id: str,
    judge: str,
    judge_version: str = "unspecified",
    alpha: float = 0.05,
    n_permutations: int = 2000,
    seed: int = _DEFAULT_SEED,
) -> ExchangeabilityAdvisory:
    """Run the MMD two-sample permutation test between two embedded batches
    and return an :class:`ExchangeabilityAdvisory`.

    ``calibration_embeddings`` / ``deployment_embeddings`` are ``(n, d)`` /
    ``(m, d)`` arrays from the SAME named judge embedder (use
    :func:`embed_texts`). Bandwidth is the standard median heuristic; the
    permutation draw is seeded for reproducibility on a matching stack.
    """
    import numpy as np

    from sum_engine_internal.research.mmd.mmd import (
        median_heuristic_bandwidth,
        mmd_permutation_pvalue,
    )

    cal = np.asarray(calibration_embeddings, dtype=np.float64)
    dep = np.asarray(deployment_embeddings, dtype=np.float64)
    if cal.ndim != 2 or dep.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d) arrays")
    if cal.shape[1] != dep.shape[1]:
        raise ValueError(
            f"embedding dims disagree: calibration d={cal.shape[1]} vs "
            f"deployment d={dep.shape[1]} — same judge embedder required"
        )
    if cal.shape[0] < 2 or dep.shape[0] < 2:
        raise ValueError("each batch needs at least 2 samples for the test")

    sigma = median_heuristic_bandwidth(cal, dep)
    mmd2, p = mmd_permutation_pvalue(
        cal, dep, sigma=sigma,
        n_permutations=n_permutations,
        rng=np.random.default_rng(seed),
    )
    return ExchangeabilityAdvisory(
        calibration_corpus_id=calibration_corpus_id,
        judge=judge, judge_version=judge_version,
        n_calibration=int(cal.shape[0]), n_deployment=int(dep.shape[0]),
        mmd2=float(mmd2), p_value=float(p),
        n_permutations=int(n_permutations), seed=int(seed),
        bandwidth=float(sigma), alpha=float(alpha),
    )


def embed_texts(texts: Sequence[str], judge: Any = None) -> Any:
    """Embed a list of documents to a ``(n, d)`` float array with the named
    judge embedder (default: the local MiniLM ``EmbeddingJudge``). One
    mean-pooled vector per document — the representation the MMD test
    compares. Needs the ``[judge]`` extra."""
    import numpy as np

    if judge is None:
        from sum_engine_internal.research.meaning.local_judge import EmbeddingJudge

        judge = EmbeddingJudge()
    vecs = judge._embed(list(texts))
    arr = vecs.detach().cpu().numpy() if hasattr(vecs, "detach") else np.asarray(vecs)
    return arr.astype(np.float64)


def advisory_report(advisory: ExchangeabilityAdvisory) -> dict[str, Any]:
    """A float-free ``sum.exchangeability_advisory.v1`` report dict — the
    rate quantities ride as integer micro-units (like the meaning receipt)
    so the report canonicalises identically across runtimes. This is an
    UNSIGNED measurement report (advisory, judge-pinned), deliberately not a
    signed receipt field."""
    return {
        "schema": "sum.exchangeability_advisory.v1",
        "calibration_corpus_id": advisory.calibration_corpus_id,
        "judge": advisory.judge,
        "judge_version": advisory.judge_version,
        "n_calibration": advisory.n_calibration,
        "n_deployment": advisory.n_deployment,
        "mmd2_micro": int(round(advisory.mmd2 * _MICRO)),
        "p_value_micro": int(round(advisory.p_value * _MICRO)),
        "alpha_micro": int(round(advisory.alpha * _MICRO)),
        "n_permutations": advisory.n_permutations,
        "seed": advisory.seed,
        "distinguishable": advisory.distinguishable,
        "verdict": advisory.verdict,
        "scope": advisory.scope,
        "advisory": "This does NOT gate or re-sign any bound; it measures "
                    "whether the bound's exchangeability precondition is "
                    "plausible for this deployment.",
    }
