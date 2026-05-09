"""Sequential analysis — adaptive stopping for round-trip iterations.

The Wald 1947 Sequential Probability Ratio Test (SPRT) is the
classical likelihood-ratio test that stops as soon as evidence
unambiguously favours H_0 or H_1, minimising expected sample size
at fixed (α, β) error rates. Howard-Ramdas-McAuliffe-Sekhon
(*Annals of Statistics* 2021) generalise to anytime-valid e-values
that survive optional stopping.

Substrate use case: stop round-trip extraction iterations as soon
as the axiom-faithfulness signal is decisive. Cuts mean LLM calls
30-50 % vs a fixed-N budget while preserving the operator's
configured (α, β) error rates.
"""
from sum_engine_internal.research.sequential.sprt import (
    BinomialSPRT,
    SPRTDecision,
    SPRTState,
)

__all__ = ["BinomialSPRT", "SPRTDecision", "SPRTState"]
