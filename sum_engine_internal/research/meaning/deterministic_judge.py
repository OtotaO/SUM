"""Deterministic judge mode — de-pinning the judge from one machine.

The replayability boundary documented in ``local_judge.py`` has been:
"eval-mode + fixed weights is deterministic on a given machine, but
floating-point ops can differ across hardware / library versions, so a
boolean near ``threshold`` could flip cross-machine — treat a model
judge's reproducibility as machine-pinned." This module is the measured
attack on that boundary.

What it does:

1. ``apply_determinism_settings()`` — pins every same-machine
   nondeterminism source we can (single thread, deterministic
   algorithms where implemented, TF32 off) and RECORDS the environment
   (torch version, quantized engine, platform) so a probe result is
   never quoted without its scope.
2. ``DeterministicNLIJudge`` — the load-bearing NLI judge under those
   settings, with opt-in INT8 dynamic quantization (``quantize="int8"``).
   **Measured negative result (2026-07-10, torch 2.7.1 / qnnpack /
   arm64):** naive dynamic INT8 quantization COLLAPSES this judge — on
   the committed probe set it flips 11 of 22 decisions (every entailment
   goes to False; max margin shift 0.965). The hypothesis "integer
   weights shrink the float-divergence surface" is not wrong, but
   ``quantize_dynamic`` on DeBERTa-v3 destroys validity before
   determinism ever matters. int8 therefore ships as an EXPERIMENT flag
   (excluded from the committed expectations and the CI gate); the
   de-pin research path runs through static/QAT quantization or an
   ONNX INT8 export with calibration — with a probe run like this one
   as its acceptance test.
3. A probe harness: a committed, fixed probe set is scored to
   ``(decision, margin_micro)`` pairs; expected results generated on one
   architecture (arm64, this repo's dev machine) are committed; the
   monthly ``judge-smoke`` CI canary recomputes them on x86_64 and
   compares.

The honest boundary (do not let any surface overclaim past this):

- Same machine, same process: bit-stable (tested by
  ``Tests/research/test_deterministic_judge.py``).
- Cross-architecture: logits are NOT expected to be bit-exact (BLAS
  kernels and quantized backends — fbgemm vs qnnpack — differ). What
  replay of a receipt actually needs is DECISION stability at the
  recorded threshold. Cross-architecture decision agreement is therefore
  MEASURED (monthly, in CI, on the committed probe set), never assumed:
  pairs whose committed margin is decisive (|margin| >= MARGIN_DECISIVE)
  must agree; near-threshold pairs are reported as the known boundary.
- A signed receipt's judge reproducibility remains machine-pinned until
  the verifying side reruns this probe on THEIR machine and sees
  agreement. The probe makes that a one-command check instead of a
  leap of faith.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sum_engine_internal.research.meaning.local_judge import (
    DEFAULT_NLI_MODEL_ID,
    NLIJudge,
)

_MICRO = 1_000_000
# |probability - threshold| at or above this is a "decisive" decision:
# cross-architecture float divergence measured in this family of models is
# orders of magnitude below 0.05, so a decisive decision flipping across
# architectures is a real failure, not noise. Near-threshold pairs are
# reported, not gated.
MARGIN_DECISIVE_MICRO = 50_000

PROBE_SET_PATH = Path("fixtures/deterministic_judge/probe_set.json")
EXPECTED_PATH = Path("fixtures/deterministic_judge/expected_decisions.json")


def apply_determinism_settings() -> dict:
    """Pin same-machine nondeterminism sources and return the recorded
    environment. Call once before loading/using a deterministic judge."""
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    # warn_only: CPU inference for this model class has deterministic
    # implementations; warn_only keeps an exotic op from hard-crashing the
    # probe while the setting still applies everywhere it exists.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():  # pragma: no cover - CPU-only in CI/dev
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {
        "torch_version": torch.__version__,
        "num_threads": 1,
        "quantized_engine": torch.backends.quantized.engine,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
    }


@dataclass
class DeterministicNLIJudge:
    """The NLI judge under pinned determinism settings, with opt-in INT8
    dynamic quantization. Exposes ``probability`` / ``entails`` plus the
    recorded ``environment`` so results always carry their scope."""

    threshold: float = 0.5
    model_id: str = DEFAULT_NLI_MODEL_ID
    quantize: "str | None" = None  # None (float32) or "int8"
    environment: dict = field(default_factory=dict, init=False)
    _judge: Any = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._judge is not None:
            return
        if self.quantize not in (None, "int8"):
            raise ValueError(f"unsupported quantize mode: {self.quantize!r}")
        self.environment = apply_determinism_settings()
        judge = NLIJudge(threshold=self.threshold, model_id=self.model_id)
        judge._ensure_loaded()
        if self.quantize == "int8":
            import torch

            # Some builds ship with no quantized engine SELECTED (engine ==
            # "none") even when one is supported (e.g. qnnpack on arm64,
            # fbgemm/x86 on x86_64) — quantized::linear_prepack then fails
            # with NoQEngine. Select the first real engine; record it, since
            # the engine IS the cross-architecture divergence surface this
            # module exists to measure.
            if torch.backends.quantized.engine == "none":
                engines = [
                    e
                    for e in torch.backends.quantized.supported_engines
                    if e != "none"
                ]
                if not engines:  # pragma: no cover - build-dependent
                    raise RuntimeError(
                        "int8 mode unavailable: this torch build has no "
                        "quantized engine"
                    )
                torch.backends.quantized.engine = engines[0]
            self.environment["quantized_engine"] = (
                torch.backends.quantized.engine
            )
            judge._mdl = torch.ao.quantization.quantize_dynamic(
                judge._mdl, {torch.nn.Linear}, dtype=torch.qint8
            ).eval()
        self._judge = judge

    @property
    def mode(self) -> str:
        return "int8-det" if self.quantize == "int8" else "float32-det"

    @property
    def name(self) -> str:
        return f"nli:{self.model_id}+{self.mode}"

    def probability(self, premise: str, hypothesis: str) -> float:
        self._ensure_loaded()
        return self._judge.entailment_probability(premise, hypothesis)

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.probability(premise, hypothesis) >= self.threshold


# ---------------------------------------------------------------------------
# Probe harness (pure functions below run without torch — CI/report logic
# is testable without a model).
# ---------------------------------------------------------------------------

def run_probe(judge: DeterministicNLIJudge, pairs: "list[dict]") -> "list[dict]":
    """Score every probe pair to ``{decision, margin_micro}``. Margin is
    ``probability - threshold`` in integer micro-units (the same wire grid
    receipts use), so expected files are float-free."""
    out = []
    for pair in pairs:
        p = judge.probability(pair["premise"], pair["hypothesis"])
        out.append(
            {
                "decision": bool(p >= judge.threshold),
                "margin_micro": int(round((p - judge.threshold) * _MICRO)),
            }
        )
    return out


def decisions_digest(results: "list[dict]") -> str:
    """``sha256-<hex>`` over the decision booleans only (the part expected
    to be stable cross-architecture)."""
    blob = json.dumps([bool(r["decision"]) for r in results]).encode()
    return "sha256-" + hashlib.sha256(blob).hexdigest()


def compare_probe_results(
    expected: "list[dict]",
    actual: "list[dict]",
    *,
    decisive_margin_micro: int = MARGIN_DECISIVE_MICRO,
) -> dict:
    """Compare a fresh probe run against committed expectations.

    Gating rule: a pair whose COMMITTED margin is decisive must reproduce
    its decision. Near-threshold pairs (committed |margin| below the bar)
    are reported, never gated — they are the documented boundary of
    cross-architecture decision stability, and hiding them would be the
    quiet version of an overclaim.
    """
    if len(expected) != len(actual):
        raise ValueError(
            f"probe length mismatch: expected {len(expected)} pairs, "
            f"got {len(actual)}"
        )
    disagreements, near_threshold, drifts = [], [], []
    for i, (e, a) in enumerate(zip(expected, actual)):
        decisive = abs(e["margin_micro"]) >= decisive_margin_micro
        flipped = bool(e["decision"]) != bool(a["decision"])
        entry = {
            "index": i,
            "expected_decision": bool(e["decision"]),
            "actual_decision": bool(a["decision"]),
            "expected_margin_micro": e["margin_micro"],
            "actual_margin_micro": a["margin_micro"],
        }
        if not decisive:
            near_threshold.append(entry)
        elif flipped:
            disagreements.append(entry)
        if not flipped and e["margin_micro"] != a["margin_micro"]:
            drifts.append(
                {
                    "index": i,
                    "margin_drift_micro": a["margin_micro"]
                    - e["margin_micro"],
                }
            )
    n_decisive = len(expected) - len(near_threshold)
    return {
        "n_pairs": len(expected),
        "n_decisive": n_decisive,
        "n_near_threshold": len(near_threshold),
        "decisive_agreement": not disagreements,
        "disagreements": disagreements,
        "near_threshold": near_threshold,
        "margin_drifts": drifts,
        "max_abs_margin_drift_micro": max(
            (abs(d["margin_drift_micro"]) for d in drifts), default=0
        ),
    }
