"""SUM meaning-loss → an Inspect (UK AISI) faithfulness scorer *with feedback*.

Inspect (github.com/UKGovernmentBEIS/inspect_ai) scores a sample with a
``Score(value, answer, explanation, metadata)``. SUM's meaning-loss readout maps
onto it 1:1: the bounded preservation in [0, 1] is the ``value``, and the
per-document kept / dropped / added breakdown (``EntailmentScorer.explain``) is
the ``explanation`` — so an eval author scoring summarisation/RAG faithfulness
gets not just a number but WHICH source claims a model dropped or fabricated,
which is exactly what makes a failing eval debuggable.

This reuses the framework-agnostic ``(score, feedback)`` core already built and
tested for the GEPA adapter (``gepa_meaning_metric.meaning_signal``); the only
Inspect-specific part is the thin ``@scorer`` shim at the bottom.

The honest boundary (load-bearing — do not paper over it)
---------------------------------------------------------
The score is a per-document MEASUREMENT under a NAMED, swappable judge — NOT
certified faithfulness and NOT human judgment. The proxy tracks human
faithfulness only MODESTLY (Spearman rho ~0.27-0.33 on SummEval), so it is a
debuggable signal, never ground truth. Use ``--scorer nli`` / ``embedding`` (the
``[judge]`` extra — a local, offline, zero-$ judge); the lexical judge has no
per-claim readout and misranks paraphrase. The distribution-free (1-delta)
*guarantee* is a SEPARATE signed artifact (``sum.meaning_risk_receipt.v1`` via
``examples/issue_meaning_receipt.py``), not this scorer.

This file targets inspect_ai's documented scorer API. The ``score_meaning`` core
is importable and unit-tested WITHOUT inspect_ai (see
``Tests/test_inspect_scorer_example.py``); the ``@scorer`` wiring is the one path
that cannot self-test without inspect_ai installed — **confirm it against your
installed inspect_ai version before opening a PR** (the same discipline as the
GEPA example's live-run check).

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import os
import sys
from typing import Any

# Reuse the shared (score, feedback) core from the sibling GEPA example.
# examples/ is not an installed package, so add this dir to the path.
sys.path.insert(0, os.path.dirname(__file__))
from gepa_meaning_metric import meaning_signal  # noqa: E402


def score_meaning(source: str, completion: str, scorer: Any) -> dict[str, Any]:
    """Map one ``(source, model-output)`` pair to the fields an Inspect ``Score``
    needs: ``value`` (preservation in [0, 1], higher is better), ``explanation``
    (the kept/dropped/added feedback), and ``metadata`` (raw numbers + the
    honesty caveat). Pure and deterministic for a deterministic ``scorer``, and
    importable without inspect_ai — so the mapping is unit-testable here while the
    ``@scorer`` registration is confirmed against inspect_ai by the user."""
    sig = meaning_signal(source, completion, scorer)
    meta: dict[str, Any] = {
        "meaning_loss": sig.loss,
        "judge": f"{scorer.name} v{scorer.version}",
        "proxy_caveat": (
            "per-document MEASUREMENT under a named proxy (Spearman rho "
            "~0.27-0.33 vs human on SummEval); not certified faithfulness, not "
            "human judgment. The (1-delta) guarantee is a separate signed "
            "sum.meaning_risk_receipt.v1."
        ),
    }
    if sig.readout is not None:
        meta["kept"] = sig.readout.preserved_claims
        meta["source_claims"] = sig.readout.source_claims
        meta["dropped"] = list(sig.readout.dropped_claims)
        meta["added_unsupported"] = list(sig.readout.unsupported_claims)
    return {"value": sig.score, "explanation": sig.feedback, "metadata": meta}


def make_default_scorer(name: str = "nli"):
    """Construct a SUM meaning scorer. ``nli``/``embedding`` need the ``[judge]``
    extra (a local, offline, zero-$ judge); ``lexical`` is dependency-free but
    misranks paraphrase (use only for extractive, sentence-dropping compression)."""
    if name == "lexical":
        from sum_engine_internal.research.meaning.meaning_loss import (
            LexicalCoverageScorer,
        )
        return LexicalCoverageScorer()
    from sum_engine_internal.research.meaning.local_judge import (
        embedding_entailment_scorer,
        nli_entailment_scorer,
    )
    return nli_entailment_scorer() if name == "nli" else embedding_entailment_scorer()


# ── Inspect @scorer shim — CONFIRM this signature against your inspect_ai ──────
# inspect_ai is intentionally NOT a dependency of this repo; an eval author who
# wants this scorer already has it. When absent, the core above still imports +
# tests fine and `meaning_faithfulness` is None.
try:
    from inspect_ai.scorer import Score, Target, mean, scorer, stderr
    from inspect_ai.solver import TaskState

    @scorer(metrics=[mean(), stderr()])
    def meaning_faithfulness(scorer_name: str = "nli"):
        """Reference-free summarisation/RAG faithfulness scorer: scores the
        model output against the SAMPLE INPUT (the source), returning
        preservation in [0, 1] plus a kept/dropped/added explanation. Assumes
        source = ``state.input_text`` and model output = ``state.output.completion``;
        adjust to your Task if those differ, and confirm the ``Score`` signature
        against your inspect_ai version."""
        sum_scorer = make_default_scorer(scorer_name)

        async def score(state: "TaskState", target: "Target") -> "Score":
            r = score_meaning(state.input_text, state.output.completion, sum_scorer)
            return Score(
                value=r["value"],
                answer=state.output.completion,
                explanation=r["explanation"],
                metadata=r["metadata"],
            )

        return score

except ImportError:  # inspect_ai not installed — core above remains usable + testable
    meaning_faithfulness = None  # type: ignore
