"""SUM meaning-loss receipt → GEPA reflective-metric adapter.

GEPA (Agrawal et al., *Reflective Prompt Evolution Can Outperform RL*, ICLR
2026 Oral, arXiv:2507.19457) optimises a system against a metric that returns
**both** a scalar score and **textual feedback** — the feedback is what its
reflection LM reads to propose a better prompt. A SUM meaning-loss scorer is
mechanically that shape:

  - the bounded loss in [0, 1] becomes the **score** (preservation = 1 - loss,
    so higher is better — GEPA maximises and defaults ``perfect_score=1.0``);
  - the per-document kept / dropped / added readout
    (``EntailmentScorer.explain`` → :class:`MeaningReadout`) becomes the
    **feedback** — it names the exact source sentences a candidate dropped and
    the exact transform sentences it added without grounding.

So the same proxy SUM *certifies* in a signed receipt also *steers* a prompt
optimiser. This module is the adapter; ``gepa_meaning_demo.py`` runs it.

The honest boundary (load-bearing — do not paper over it)
---------------------------------------------------------
The per-document readout is a **measurement under a named judge**, not a
certified bound. GEPA optimises against the *measurement*; the distribution-free
(1-δ) **guarantee** lives in a ``sum.meaning_risk_receipt.v1`` over a named
calibration corpus (``examples/issue_meaning_receipt.py``). The intended loop:
optimise the prompt with this metric, then *certify the chosen prompt* with a
receipt over held-out pairs. Optimising and certifying on the same data would
be train-on-test — the receipt's exchangeability scope is what keeps the bound
honest, and this metric does not produce one.

The score is only as meaningful as the injected judge. ``LexicalCoverageScorer``
is dependency-free but **misranks paraphrase** (it over-counts a faithful reword
as loss); use it only for extractive compression. For anything paraphrase-bearing
plug ``EntailmentScorer`` over an NLI / embedding / LLM judge
(``--scorer nli|embedding`` in the CLI). When a scorer cannot itemise (the
lexical one has no ``explain``) the feedback degrades to the bare number and says
so, rather than inventing a kept/dropped/added breakdown.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sum_engine_internal.research.meaning.meaning_loss import (
    MeaningReadout,
    MeaningScorer,
    _sentences,  # same splitter explain() uses — to list the *preserved* claims
)

# Feedback design follows current GEPA best practice (DSPy GEPA docs; Arize
# 2025-11 benchmark — "the biggest gains come from richer / more actionable
# feedback, not the search algorithm"; Decagon 2026-03 — "include both positive
# AND negative examples"). So the feedback itemises KEPT claims (positive),
# DROPPED + ungrounded-ADDED claims (negative), and names the judge, rather than
# emitting a bare number. Verified against gepa==0.1.1 / dspy>=3.2.1 (the 3.3.0b1
# beta keeps the same metric contract). An alternative injection point is gepa's
# optimize_anything + oa.log(); the dspy.GEPA metric path here is lower-friction.
_MAX_LISTED = 6  # cap each itemised list so the feedback stays scannable


@dataclass(frozen=True)
class MeaningSignal:
    """The framework-agnostic core: a (score, feedback) pair plus the raw
    numbers it was built from. ``score`` is preservation in [0, 1] (higher is
    better, for a maximiser); ``loss`` is the SUM meaning-loss it complements
    (``score == 1 - loss``); ``readout`` is the itemised breakdown when the
    scorer can produce one, else ``None``."""

    score: float
    feedback: str
    loss: float
    readout: MeaningReadout | None


def meaning_signal(source: str, transform: str, scorer: MeaningScorer) -> MeaningSignal:
    """Score one ``(source, transform)`` pair and build the GEPA signal.

    Uses ``scorer.explain`` (present on ``EntailmentScorer``) for itemised
    feedback when available; otherwise reports the bare loss and names the
    limitation. Pure and deterministic for a deterministic ``scorer`` — no
    framework import, so it is unit-testable without dspy/gepa installed."""
    loss = float(scorer.loss(source, transform))
    readout = scorer.explain(source, transform) if hasattr(scorer, "explain") else None
    feedback = _format_feedback(scorer, loss, readout, source)
    score = max(0.0, min(1.0, 1.0 - loss))
    return MeaningSignal(score=score, feedback=feedback, loss=loss, readout=readout)


def _preserved_claims(source: str, readout: MeaningReadout) -> list[str]:
    """The source sentences the judge DID preserve = source sentences minus the
    dropped ones, using the same splitter ``explain`` used so the two align."""
    dropped = set(readout.dropped_claims)
    return [s for s in _sentences(source) if s not in dropped]


def _format_feedback(
    scorer: MeaningScorer, loss: float, readout: MeaningReadout | None, source: str = ""
) -> str:
    """Turn a score (+ optional readout) into the text GEPA's reflection LM
    reads. The whole point of GEPA over a scalar metric is this channel, so it
    is concrete: it names the KEPT (positive) and DROPPED/ADDED (negative)
    claims and what to do about them — richer feedback is what GEPA rewards."""
    head = (
        f"Meaning-loss proxy = {loss:.3f} (preservation {1.0 - loss:.0%}) "
        f"under judge '{scorer.name}' v{scorer.version}."
    )
    if readout is None:
        return (
            head
            + " This scorer cannot itemise what changed (no per-claim readout); "
            "it reports only the aggregate number. If it is the lexical proxy, "
            "treat the score as reliable only for extractive (sentence-dropping) "
            "compression — it misranks faithful paraphrase. To improve, reduce "
            "omission of source content and avoid introducing unsupported content."
        )
    lines = [
        head,
        f"Kept {readout.preserved_claims}/{readout.source_claims} source claims "
        f"(recall {readout.recall:.0%}, fidelity {readout.fidelity:.0%}).",
    ]
    kept = _preserved_claims(source, readout) if source else []
    if kept:
        lines.append("KEPT from the source (this is working — keep preserving these):")
        lines.extend(f"  + {s}" for s in kept[:_MAX_LISTED])
        if len(kept) > _MAX_LISTED:
            lines.append(f"  + … and {len(kept) - _MAX_LISTED} more")
    if readout.dropped_claims:
        lines.append(
            "DROPPED from the source (preserve these to raise the score):"
        )
        lines.extend(f"  - {s}" for s in readout.dropped_claims[:_MAX_LISTED])
        if len(readout.dropped_claims) > _MAX_LISTED:
            lines.append(f"  - … and {len(readout.dropped_claims) - _MAX_LISTED} more")
    if readout.unsupported_claims:
        lines.append(
            "ADDED but not grounded in the source (remove or ground these):"
        )
        lines.extend(f"  - {t}" for t in readout.unsupported_claims[:_MAX_LISTED])
        if len(readout.unsupported_claims) > _MAX_LISTED:
            lines.append(f"  - … and {len(readout.unsupported_claims) - _MAX_LISTED} more")
    if not readout.dropped_claims and not readout.unsupported_claims:
        lines.append(
            "No dropped or ungrounded claims detected by this judge — the "
            "transform preserved the source's assertions."
        )
    lines.append(
        "Reminder: this is a per-document MEASUREMENT under the named judge, "
        "not a certified bound. Certify the chosen prompt with a meaning_risk "
        "receipt over held-out pairs."
    )
    return "\n".join(lines)


# ── dspy.GEPA path ──────────────────────────────────────────────────────────
# dspy.GEPA expects metric(gold, pred, trace, pred_name, pred_trace) returning
# either a float or a dspy.Prediction(score=..., feedback=...). We return the
# latter so the feedback reaches the reflection LM.

_DEFAULT_PRED_FIELDS = ("summary", "translation", "rendering", "output", "answer", "text")


def make_gepa_metric(
    scorer: MeaningScorer,
    *,
    source_key: str = "source",
    pred_key: str | None = None,
):
    """Build a ``dspy.GEPA``-compatible metric from a SUM meaning scorer.

    ``source_key`` names the field on the gold example holding the original
    text. ``pred_key`` names the field on the prediction holding the transform;
    if ``None``, the first of ``{summary, translation, rendering, output,
    answer, text}`` present is used (or the prediction itself if it is a string).

    The returned callable matches dspy.GEPA's signature
    ``metric(gold, pred, trace=None, pred_name=None, pred_trace=None)`` and
    returns ``dspy.Prediction(score, feedback)`` when dspy is importable, else a
    lightweight dict-and-attr shim (so the metric is exercisable in tests
    without dspy)."""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        source = _extract(gold, source_key)
        transform = _extract_pred(pred, pred_key)
        sig = meaning_signal(source, transform, scorer)
        return _wrap(sig.score, sig.feedback)

    return metric


def _extract(obj: Any, key: str) -> str:
    """Read ``key`` off a dspy.Example / dict / namespace, or fall back to str."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if key in obj:
            return str(obj[key])
    else:
        try:
            if key in obj:  # dspy.Example supports __contains__
                return str(obj[key])
        except TypeError:
            pass
        if hasattr(obj, key):
            return str(getattr(obj, key))
    raise KeyError(
        f"could not find field '{key}' on {type(obj).__name__}; "
        f"pass source_key=/pred_key= to name the right field"
    )


def _extract_pred(pred: Any, pred_key: str | None) -> str:
    """Read the transform text off a prediction. Honours an explicit
    ``pred_key``; otherwise probes the common output fields, then falls back to
    the object's string form."""
    if pred_key is not None:
        return _extract(pred, pred_key)
    if isinstance(pred, str):
        return pred
    for field in _DEFAULT_PRED_FIELDS:
        try:
            if isinstance(pred, dict):
                if field in pred:
                    return str(pred[field])
            elif field in pred or hasattr(pred, field):
                return _extract(pred, field)
        except TypeError:
            if hasattr(pred, field):
                return str(getattr(pred, field))
    return str(pred)


class _ScoreWithFeedback(dict):
    """Offline shim mirroring dspy's ScoreWithFeedback: supports both attribute
    access (``o.score``) and item access (``o["score"]``), which is exactly the
    surface dspy.GEPA touches. Used only when dspy is not installed (tests)."""

    def __init__(self, score: float, feedback: str) -> None:
        super().__init__(score=score, feedback=feedback)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


def _wrap(score: float, feedback: str):
    try:
        import dspy

        return dspy.Prediction(score=score, feedback=feedback)
    except ImportError:
        return _ScoreWithFeedback(score, feedback)
