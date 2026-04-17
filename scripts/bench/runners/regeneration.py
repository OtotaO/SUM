from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..corpus import Corpus
from ..schema import RegenerationMetrics


@dataclass(frozen=True)
class SumRegenerationRunner:
    name: str = "sum.tome_generator"
    factscore_model_id: str = ""
    minicheck_model_id: str = ""

    def run(self, corpus: Corpus) -> Sequence[RegenerationMetrics]:
        raise NotImplementedError(
            "STATE 4: regenerate via internal.ensemble.tome_generator for both "
            "canonical and freeform paths; score with FActScore + MiniCheck"
        )
