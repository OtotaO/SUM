"""Runnable demo: optimise a summariser against a SUM meaning-loss receipt with GEPA.

Two modes:

  • OFFLINE (default) — no network, no API key, no extra installs. Uses a
    deterministic token-overlap stand-in judge so the kept/dropped/added
    readout is reproducible, and shows the exact (score, feedback) signal that
    GEPA's reflection LM consumes for two candidate summaries — a weak one and
    a strong one. This proves the SUM→GEPA metric contract end-to-end without
    pretending GEPA itself ran.

  • LIVE (``--live``) — actually runs ``dspy.GEPA`` to evolve a summariser
    prompt against the SUM metric. Requires ``pip install dspy`` and a model
    (``--model`` / ``--reflection-model`` + the provider's API key in env).

Honesty note: OFFLINE mode demonstrates the *metric*, not the optimiser — GEPA
needs a reflection LM to propose prompts, which can't be faked offline. The demo
says which mode it is in and never claims an optimisation happened when it did
not. The stand-in judge is a crude lexical proxy labelled as such; production
plugs an NLI/embedding/LLM judge (see ``examples/issue_meaning_receipt.py``).

Run:
    python examples/gepa_meaning_demo.py              # offline contract demo
    python examples/gepa_meaning_demo.py --live --model openai/gpt-4.1-mini \\
        --reflection-model openai/gpt-5               # real GEPA optimisation

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import sys

from sum_engine_internal.research.meaning.meaning_loss import (
    EntailmentScorer,
    _content_units,  # reused only for the demo's stand-in judge
)

from gepa_meaning_metric import make_gepa_metric, meaning_signal


# ── a deterministic, dependency-free stand-in judge for the offline demo ─────
def demo_entails(premise: str, hypothesis: str, *, threshold: float = 0.6) -> bool:
    """A crude DETERMINISTIC entailment stand-in: 'premise entails hypothesis'
    iff at least ``threshold`` of the hypothesis's content words appear in the
    premise. This is NOT an NLI model — it is a reproducible placeholder so the
    offline demo runs anywhere. It shares the lexical proxy's blind spot
    (paraphrase) and exists only to make the (score, feedback) flow visible.
    Production uses EntailmentScorer over a real judge (--scorer nli|embedding)."""
    hyp = set(_content_units(hypothesis))
    if not hyp:
        return True
    prem = set(_content_units(premise))
    return len(hyp & prem) / len(hyp) >= threshold


def _demo_scorer() -> EntailmentScorer:
    return EntailmentScorer(
        entails=demo_entails,
        judge_name="demo-token-overlap",
        judge_version="0",
    )


# ── the toy task: summarise a short report ───────────────────────────────────
SOURCE = (
    "The Q3 audit found three issues. Revenue grew 4% over Q2. "
    "The Berlin office missed its hiring target. No security incidents were reported."
)

# A weak candidate: drops a real finding (Berlin) AND adds an ungrounded claim.
WEAK_SUMMARY = (
    "Revenue grew 4% over Q2. The company launched a new product line in Asia."
)

# A strong candidate: keeps every source claim, adds nothing ungrounded.
STRONG_SUMMARY = (
    "The Q3 audit found three issues. Revenue grew 4% over Q2. "
    "The Berlin office missed its hiring target. No security incidents were reported."
)


def run_offline() -> int:
    scorer = _demo_scorer()
    print("=" * 72)
    print("OFFLINE demo — the SUM→GEPA metric contract (no GEPA run; no API key)")
    print(f"judge: {scorer.name} v{scorer.version}  (deterministic lexical stand-in)")
    print("=" * 72)
    print(f"\nSOURCE:\n  {SOURCE}\n")

    for label, summary in [("WEAK candidate", WEAK_SUMMARY), ("STRONG candidate", STRONG_SUMMARY)]:
        sig = meaning_signal(SOURCE, summary, scorer)
        print("-" * 72)
        print(f"{label}:\n  {summary}\n")
        print(f"  GEPA score (preservation, higher=better): {sig.score:.3f}")
        print("  GEPA feedback (verbatim, this is what the reflection LM reads):")
        for line in sig.feedback.splitlines():
            print(f"    {line}")
        print()

    # Show the metric in the exact dspy.GEPA callable shape too.
    metric = make_gepa_metric(scorer, source_key="source", pred_key="summary")
    gold = {"source": SOURCE}
    weak_out = metric(gold, {"summary": WEAK_SUMMARY})
    strong_out = metric(gold, {"summary": STRONG_SUMMARY})
    print("-" * 72)
    print("As a dspy.GEPA metric(gold, pred, ...) -> {score, feedback}:")
    print(f"  weak  -> score={weak_out['score']:.3f}")
    print(f"  strong-> score={strong_out['score']:.3f}")
    ok = strong_out["score"] > weak_out["score"]
    print(
        f"\nResult: the faithful summary scores higher ({strong_out['score']:.3f} > "
        f"{weak_out['score']:.3f}) — {'PASS' if ok else 'FAIL'}. "
        "GEPA would climb this gradient, using the feedback to keep the dropped\n"
        "Berlin/audit claims and drop the ungrounded 'product line in Asia'."
    )
    print(
        "\nNext: run with --live to have dspy.GEPA actually evolve the prompt, "
        "then certify\nthe winner with examples/issue_meaning_receipt.py."
    )
    return 0 if ok else 1


def run_live(model: str, reflection_model: str, budget: int) -> int:
    try:
        import dspy
    except ImportError:
        print(
            "live mode needs dspy: pip install dspy  (and set your provider's "
            "API key in env, e.g. OPENAI_API_KEY)",
            file=sys.stderr,
        )
        return 2

    # A real (paraphrase-aware) judge belongs here in production. To keep the
    # demo self-contained we reuse the deterministic stand-in; swap in
    # EntailmentScorer over an NLI/embedding judge for real use.
    scorer = _demo_scorer()
    metric = make_gepa_metric(scorer, source_key="source", pred_key="summary")

    dspy.configure(lm=dspy.LM(model=model))

    class Summarise(dspy.Signature):
        """Summarise the source faithfully, preserving every claim."""

        source: str = dspy.InputField()
        summary: str = dspy.OutputField()

    student = dspy.Predict(Summarise)

    trainset = [
        dspy.Example(source=SOURCE).with_inputs("source"),
        dspy.Example(
            source=(
                "The bridge reopened Monday after repairs. Tolls rose to 3 euros. "
                "Cyclists now have a dedicated lane. Traffic is expected to ease."
            )
        ).with_inputs("source"),
    ]

    gepa = dspy.GEPA(
        metric=metric,
        max_metric_calls=budget,
        reflection_lm=dspy.LM(model=reflection_model, temperature=1.0, max_tokens=8000),
        track_stats=True,
    )
    print(f"Running dspy.GEPA (task={model}, reflection={reflection_model}, "
          f"budget={budget} metric calls)…")
    optimised = gepa.compile(student, trainset=trainset, valset=trainset)

    print("\n" + "=" * 72)
    print("OPTIMISED summariser instruction:")
    print("=" * 72)
    pred = optimised.predict if hasattr(optimised, "predict") else optimised
    sig_obj = getattr(pred, "signature", None)
    print(getattr(sig_obj, "instructions", "(see optimised program above)"))
    print(
        "\nNow certify this prompt's meaning-loss over held-out pairs:\n"
        "  python examples/issue_meaning_receipt.py held_out_pairs.json "
        "--scorer nli --corpus-id my-corpus-v0 --transform summarize:gepa-optimised"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="run real dspy.GEPA (needs dspy + API key)")
    p.add_argument("--model", default="openai/gpt-4.1-mini", help="task LM (live mode)")
    p.add_argument("--reflection-model", default="openai/gpt-5", help="reflection LM (live mode)")
    p.add_argument("--budget", type=int, default=30, help="max metric calls (live mode)")
    args = p.parse_args(argv)
    if args.live:
        return run_live(args.model, args.reflection_model, args.budget)
    return run_offline()


if __name__ == "__main__":
    raise SystemExit(main())
