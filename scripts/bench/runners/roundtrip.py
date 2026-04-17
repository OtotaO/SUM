from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..corpus import Corpus
from ..schema import RoundtripMetrics


@dataclass(frozen=True)
class SumRoundtripRunner:
    """
    Measures axiom-set symmetric-difference percent across text -> axioms -> text -> axioms.
    The Ouroboros verifier currently only exercises canonical-template input; this runner
    exercises prose input and reports drift, which is what the conservation claim needs.
    """
    name: str = "sum.roundtrip"

    def run(self, corpus: Corpus) -> Sequence[RoundtripMetrics]:
        raise NotImplementedError(
            "STATE 4: ingest each doc, render via generator, re-ingest, diff entailed "
            "axiom sets via internal.algorithms.semantic_arithmetic"
        )
