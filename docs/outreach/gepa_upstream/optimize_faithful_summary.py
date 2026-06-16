"""Optimise a summariser with GEPA against a signed meaning-loss certificate (SUM).

A worked example for the standalone ``gepa`` package showing how to drive GEPA
with a *semantic-faithfulness* metric instead of an exact-match one. The metric
here is a SUM meaning-loss proxy (``pip install sum-engine[research]``): a
bounded [0, 1] score plus a per-document **kept / dropped / added** readout —
which is exactly GEPA's (score, textual-feedback) shape. The readout tells the
reflection LM *which source claims a candidate summary dropped* and *which
claims it added without grounding*, so reflection is targeted rather than blind.

Why this is interesting beyond the example: the same proxy that scores a summary
here can be turned into a signed, offline-verifiable certificate with a
distribution-free (1-δ) bound over a corpus (a ``sum.meaning_risk_receipt.v1``).
So the loop is: **optimise** the prompt with GEPA against the measurement, then
**certify** the winning prompt's meaning-loss over held-out data. (Optimising and
certifying on the same data would be train-on-test — keep them separate.)

The honest boundary, stated plainly: the per-document readout is a MEASUREMENT
under a *named, swappable* judge, not a certified bound and not "meaning" itself.
The judge used in ``__main__`` below is a deterministic token-overlap stand-in so
this file runs with no key and no model download; for real use plug an
NLI/embedding/LLM judge (``EntailmentScorer`` over your model). The stand-in (like
any purely lexical proxy) is blind to paraphrase and is labelled as such.

Run the offline self-test (no key, no LLM — exercises the SUM scoring core):
    python optimize_faithful_summary.py --selftest

Run real GEPA optimisation (needs a model + the standalone gepa package):
    pip install gepa "sum-engine[research]"
    export OPENAI_API_KEY=...
    python optimize_faithful_summary.py --task-lm openai/gpt-4.1-mini \\
        --reflection-lm openai/gpt-5 --budget 60

License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Mapping, Sequence

# ── SUM meaning-loss scorer (the metric) ─────────────────────────────────────
# EntailmentScorer wraps a caller-supplied entails(premise, hypothesis) -> bool
# (an NLI model, an embedding judge, or an LLM at temperature 0) and computes a
# bidirectional-entailment meaning-loss, plus an itemised explain() readout.
from sum_engine_internal.research.meaning.meaning_loss import (  # type: ignore
    EntailmentScorer,
    MeaningScorer,
    _content_units,
)


def score_and_explain(source: str, summary: str, scorer: MeaningScorer) -> tuple[float, str]:
    """Return (score, feedback) for one (source, summary) pair.

    score    = preservation in [0, 1] (1 - meaning_loss), higher is better, so
               GEPA (a maximiser with perfect_score=1.0) climbs toward faithful
               summaries.
    feedback = the kept/dropped/added readout the reflection LM reads.
    """
    loss = float(scorer.loss(source, summary))
    score = max(0.0, min(1.0, 1.0 - loss))
    readout = scorer.explain(source, summary) if hasattr(scorer, "explain") else None
    if readout is None:
        feedback = (
            f"meaning-loss {loss:.3f} (preservation {score:.0%}) under "
            f"'{scorer.name}' v{scorer.version}; this scorer reports no itemised "
            "breakdown. Reduce omission of source content; avoid adding content "
            "not present in the source."
        )
        return score, feedback
    lines = [
        f"meaning-loss {loss:.3f} (preservation {score:.0%}) under "
        f"'{scorer.name}' v{scorer.version}.",
        f"kept {readout.preserved_claims}/{readout.source_claims} source claims "
        f"(recall {readout.recall:.0%}, fidelity {readout.fidelity:.0%}).",
    ]
    if readout.dropped_claims:
        lines.append("DROPPED from the source (preserve these):")
        lines += [f"  - {s}" for s in readout.dropped_claims]
    if readout.unsupported_claims:
        lines.append("ADDED but not grounded in the source (remove or ground these):")
        lines += [f"  - {t}" for t in readout.unsupported_claims]
    if not readout.dropped_claims and not readout.unsupported_claims:
        lines.append("no dropped or ungrounded claims detected — faithful summary.")
    return score, "\n".join(lines)


# ── GEPA adapter (standalone gepa GEPAAdapter protocol) ──────────────────────
# evaluate() puts numeric scores on EvaluationBatch.scores; make_reflective_dataset()
# routes the textual feedback into each record's "Feedback" key. A candidate is a
# dict {component_name: prompt_text}; here the single component is "summarize".
def build_adapter(scorer: MeaningScorer, task_lm: Callable[[str], str], component: str = "summarize"):
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: F401

    class SumMeaningAdapter(GEPAAdapter):
        """Scores each summary by SUM meaning-loss; feeds the kept/dropped/added
        readout back to GEPA's reflection LM as textual feedback."""

        def evaluate(self, batch: list[str], candidate: dict[str, str], capture_traces: bool = False):
            prompt = candidate[component]
            outputs: list[str] = []
            scores: list[float] = []
            traces: list[dict[str, Any]] = []
            for source in batch:
                summary = task_lm(f"{prompt}\n\nSOURCE:\n{source}\n\nSUMMARY:")
                score, feedback = score_and_explain(source, summary, scorer)
                outputs.append(summary)
                scores.append(score)
                traces.append({"source": source, "summary": summary, "feedback": feedback})
            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=traces if capture_traces else None,
            )

        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_batch,
            components_to_update: list[str],
        ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
            records = [
                {
                    "Inputs": t["source"],
                    "Generated Outputs": t["summary"],
                    "Feedback": t["feedback"],
                }
                for t in (eval_batch.trajectories or [])
            ]
            return {c: records for c in components_to_update}

    return SumMeaningAdapter()


# ── a deterministic, key-free judge so this file runs anywhere ────────────────
def _overlap_entails(premise: str, hypothesis: str, *, threshold: float = 0.6) -> bool:
    """DETERMINISTIC token-overlap stand-in for an NLI judge. Not a real entailment
    model — it shares the lexical blind spot for paraphrase. Lets the example run
    with no key; swap in an NLI/embedding/LLM judge for real use."""
    hyp = set(_content_units(hypothesis))
    if not hyp:
        return True
    return len(hyp & set(_content_units(premise))) / len(hyp) >= threshold


def _demo_scorer() -> EntailmentScorer:
    return EntailmentScorer(entails=_overlap_entails, judge_name="token-overlap-demo", judge_version="0")


SEED_PROMPT = "Summarise the source in one or two sentences."
TRAINSET = [
    "The Q3 audit found three issues. Revenue grew 4% over Q2. "
    "The Berlin office missed its hiring target. No security incidents were reported.",
    "The bridge reopened Monday after repairs. Tolls rose to 3 euros. "
    "Cyclists now have a dedicated lane. Traffic is expected to ease.",
]


def _selftest() -> int:
    """Exercise the SUM scoring core with no LLM and no key: a faithful summary
    must outscore a lossy one, and the feedback must name what changed."""
    scorer = _demo_scorer()
    source = TRAINSET[0]
    faithful = source
    lossy = "Revenue grew 4% over Q2. The company launched a new product line in Asia."
    s_faithful, _ = score_and_explain(source, faithful, scorer)
    s_lossy, fb_lossy = score_and_explain(source, lossy, scorer)
    print(f"faithful score = {s_faithful:.3f}\nlossy score    = {s_lossy:.3f}\n")
    print("feedback for the lossy summary (what GEPA's reflection LM reads):")
    print(fb_lossy)
    ok = s_faithful == 1.0 and s_lossy < s_faithful and "DROPPED" in fb_lossy and "ADDED" in fb_lossy
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _run_gepa(task_lm_name: str, reflection_lm: str, budget: int) -> int:
    try:
        import gepa
    except ImportError:
        print("needs the standalone gepa package: pip install gepa", file=sys.stderr)
        return 2
    try:
        import litellm
    except ImportError:
        print("this example calls the task model via litellm: pip install litellm", file=sys.stderr)
        return 2

    def task_lm(prompt: str) -> str:
        resp = litellm.completion(model=task_lm_name, messages=[{"role": "user", "content": prompt}])
        return resp["choices"][0]["message"]["content"]

    adapter = build_adapter(_demo_scorer(), task_lm)  # swap _demo_scorer() for a real NLI judge
    result = gepa.optimize(
        seed_candidate={"summarize": SEED_PROMPT},
        trainset=TRAINSET,
        valset=TRAINSET,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=budget,
        display_progress_bar=True,
    )
    print("\nOptimised summariser prompt:\n" + result.best_candidate["summarize"])
    print(
        "\nNext: certify this prompt's meaning-loss over HELD-OUT pairs with a "
        "signed receipt:\n  python -m sum_cli ...  (see sum-engine examples/issue_meaning_receipt.py)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="run the key-free SUM-scoring self-test")
    p.add_argument("--task-lm", default="openai/gpt-4.1-mini", help="model that writes the summaries")
    p.add_argument("--reflection-lm", default="openai/gpt-5", help="model that rewrites the prompt")
    p.add_argument("--budget", type=int, default=60, help="max metric calls")
    args = p.parse_args(argv)
    if args.selftest:
        return _selftest()
    return _run_gepa(args.task_lm, args.reflection_lm, args.budget)


if __name__ == "__main__":
    raise SystemExit(main())
