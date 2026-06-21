"""study — the *verifiable cheatsheet*: machine-studying over a corpus.

Motivation (Li, *Machine Studying*, 2026,
https://jacobxli.com/blog/2026/machine-studying): an agent builds
**expertise** over a corpus *before* the downstream task is known.
"Expertise" is framed there as efficiency at turning inference-compute
into accurate answers — a weighted area-under-the-curve of accuracy vs.
compute, favouring *cheap* budgets — and the one method that empirically
helped was writing a reusable **cheatsheet** (amortized context
management), beating continual-pretraining and synthetic fine-tuning.

SUM already owns the pieces of that loop; this module composes them and
adds one scalar:

  - the **cheatsheet** = ``extract`` each document → ``compose`` the
    bundles (LCM-union of triples → one SUM-of-SUMs) → ``slider``-render
    the merged bundle at a chosen density (the study notes);
  - the **curve** = a :class:`RenderFrontier` over a faithful→compressed
    path, carrying measured meaning-loss per point;
  - the gap the blog leaves open — a cheatsheet is *unverified* ("did my
    notes silently drop something load-bearing?") — is closed by an
    optional ``sum.meaning_risk_receipt.v1`` over the corpus.

The one new piece is :func:`expertise`: SUM's native efficiency scalar,
the weighted AUC of *fidelity* (``1 - meaning_loss``) vs. *consultation
budget* (a small/compressed note is cheap to read), favouring the cheap
end the way the blog's weighting does.

The honest line (proof boundary)
--------------------------------
:func:`expertise` is a **derived MEASUREMENT over a named proxy**, not a
guarantee — and it is an *analogy* to the blog's metric, not the same
quantity: the blog measures downstream **task accuracy**; SUM measures
**meaning-fidelity** under a named scorer whose declared blind spots
(arrangement, sound, connotation, implicature) it inherits. A higher
expertise number means "this study artifact stays faithful even when
small," nothing more. The only *certified* claim a study artifact can
carry is its embedded ``sum.meaning_risk_receipt.v1`` (a marginal,
distribution-free corpus bound); the expertise scalar and the frontier
losses are per-run measurements. A surface must never present the
expertise scalar as a guarantee.

This module is dependency-free and offline (no numpy, no LLM, no
network): it takes positions + fidelities (the caller scores them via
the frontier) and returns a number. It ships behind the ``[research]``
extra, the same home as ``research.frontier`` and ``research.meaning``;
it is NOT a cataloged production feature.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# ``ln(10)`` — the blog's decay constant (``w(x) = ln(10)·10^(-x)``),
# carried over as the default so the weighting's heritage is legible.
# Here it weights an exp-decay over a *normalised* consultation budget,
# not over log-decades of compute — see the analogy boundary in the
# module docstring.
DEFAULT_DECAY: float = math.log(10.0)


def expertise(
    positions: Sequence[float],
    fidelities: Sequence[float],
    *,
    decay: float = DEFAULT_DECAY,
) -> float:
    """Weighted AUC of fidelity vs. consultation budget, in ``[0, 1]``.

    ``positions`` are the frontier points' places on the
    faithful→compressed path (``0.0`` = most faithful / most expensive to
    read, ``1.0`` = most compressed / cheapest), aligned by index with
    ``fidelities`` (``= 1 - meaning_loss``, each in ``[0, 1]``).

    The *consultation budget* of a point is ``1 - position`` (a faithful,
    verbose note costs more to read; a compressed note is cheap), and the
    weight is ``exp(-decay · budget)`` — favouring the cheap end exactly
    as the blog's ``10^(-x)`` favours cheap compute. The score is the
    weight-normalised mean fidelity:

        expertise = Σ wᵢ · fidelityᵢ  /  Σ wᵢ ,   wᵢ = exp(-decay · (1-pᵢ))

    So an artifact that stays faithful *when small* scores near 1.0; one
    that collapses at the cheap end scores well below its naïve mean. A
    single point returns its own fidelity. Deterministic; pure-Python.

    Raises ``ValueError`` on length mismatch, empty input, out-of-range
    fidelity, or negative ``decay``.
    """
    n = len(positions)
    if n == 0:
        raise ValueError("expertise needs at least one point")
    if len(fidelities) != n:
        raise ValueError(
            f"positions has {n} entries but fidelities has {len(fidelities)}"
        )
    if decay < 0.0:
        raise ValueError(f"decay must be non-negative; got {decay}")
    weighted_sum = 0.0
    weight_total = 0.0
    for p, f in zip(positions, fidelities):
        f = float(f)
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"fidelity must be in [0, 1]; got {f}")
        budget = 1.0 - max(0.0, min(1.0, float(p)))
        w = math.exp(-decay * budget)
        weighted_sum += w * f
        weight_total += w
    # weight_total > 0 always (exp is positive), so this is safe.
    return weighted_sum / weight_total


@dataclass(frozen=True, slots=True)
class StudyArtifact:
    """A persistent, verifiable cheatsheet produced by studying a corpus.

    Composes the corpus knowledge base (the ``compose``d ``state_integer``
    + axiom count), the study notes at the chosen density (``cheatsheet``),
    the faithful→compressed ``frontier`` it sits on, the ``expertise``
    scalar, and an OPTIONAL signed ``sum.meaning_risk_receipt.v1``
    (``receipt``) — the only certified element. ``as_dict`` serialises a
    ``sum.study_artifact.v1`` for the CLI / API / persistence surface.
    """
    corpus_id: str
    doc_count: int
    state_integer: int
    axiom_count: int
    cheatsheet: str
    study_density: float
    expertise: float
    scorer_name: str
    scorer_version: str
    frontier: Mapping[str, Any] = field(default_factory=dict)
    receipt: Mapping[str, Any] | None = None

    # The same honest-boundary string the frontier carries, one layer up:
    # the artifact is a MEASUREMENT except for its embedded receipt.
    _MEASUREMENT_NOTE = (
        "expertise and frontier meaning_loss are per-run MEASUREMENTS "
        "under the named scorer (an analogy to studying-expertise, not a "
        "task-accuracy metric), not certified bounds; the only certified "
        "claim is the embedded sum.meaning_risk_receipt.v1 (a marginal, "
        "distribution-free corpus bound) when present"
    )

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": "sum.study_artifact.v1",
            "corpus_id": self.corpus_id,
            "doc_count": self.doc_count,
            "state_integer": str(self.state_integer),  # may exceed JSON int
            "axiom_count": self.axiom_count,
            "study_density": self.study_density,
            "expertise": self.expertise,
            "cheatsheet": self.cheatsheet,
            "scorer": self.scorer_name,
            "scorer_version": self.scorer_version,
            "frontier": dict(self.frontier),
            "certified": self.receipt is not None,
            "measurement_note": self._MEASUREMENT_NOTE,
        }
        if self.receipt is not None:
            d["receipt"] = dict(self.receipt)
        return d
