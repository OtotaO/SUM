"""Tests for the §2.5 runner's per-call timeout discipline.

The seed_long capstone surfaced a real failure mode: an OpenAI
structured-output call hung for 14+ minutes with the python process
alive but no CPU progress. The ``_with_call_timeout`` helper +
``S25CallTimeoutError`` + per-doc skip path are the defence-in-depth.

Three layers of coverage:

  1. **Helper behaviour.** ``_with_call_timeout`` wraps a coroutine in
     ``asyncio.wait_for`` and converts the timeout into a tagged
     ``S25CallTimeoutError`` with the call name + timeout duration.
  2. **Per-doc skip.** When any LLM call inside ``run_doc`` raises
     ``S25CallTimeoutError``, the function returns a per-doc record
     tagged ``error_class="timeout"`` instead of propagating.
  3. **Aggregate exclusion.** ``aggregate()`` excludes timed-out docs
     from drift/recall means and counts them in ``n_docs_timed_out``.

The tests use a tiny mock client that returns a coroutine which
``asyncio.sleep``s for longer than the call timeout, simulating a
hang without making any network call.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Add repo root to sys.path so we can import the runner module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.bench.runners.s25_generator_side import (
    DEFAULT_CALL_TIMEOUT_S,
    S25CallTimeoutError,
    _with_call_timeout,
    aggregate,
    run_doc,
)


# --------------------------------------------------------------------------
# _with_call_timeout
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_call_timeout_passes_through_on_success():
    """A coroutine that completes well under the timeout returns
    its value unmodified."""
    async def quick():
        return "result"

    result = await _with_call_timeout(quick(), timeout_s=1.0, what="test_call")
    assert result == "result"


@pytest.mark.asyncio
async def test_with_call_timeout_raises_tagged_error_on_hang():
    """A coroutine that exceeds the timeout raises
    ``S25CallTimeoutError`` tagged with the call name and duration —
    NOT a bare ``asyncio.TimeoutError``. The tag is what the caller
    needs to construct a per-doc skip record."""
    async def hangs():
        await asyncio.sleep(10)  # well above the test timeout below
        return "should not reach"

    with pytest.raises(S25CallTimeoutError) as exc_info:
        await _with_call_timeout(hangs(), timeout_s=0.05, what="hanging_call")

    assert exc_info.value.what == "hanging_call"
    assert exc_info.value.timeout_s == 0.05
    assert "hanging_call" in str(exc_info.value)
    assert "0.1s" in str(exc_info.value) or "0.05s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_with_call_timeout_does_not_swallow_other_exceptions():
    """If the wrapped coroutine raises something other than a
    timeout, the exception propagates unchanged. We do NOT want
    ``S25CallTimeoutError`` to mask real LLM-side errors."""
    class AnomalousError(Exception):
        pass

    async def raises_anomaly():
        raise AnomalousError("not a timeout")

    with pytest.raises(AnomalousError, match="not a timeout"):
        await _with_call_timeout(
            raises_anomaly(), timeout_s=1.0, what="anomalous_call"
        )


# --------------------------------------------------------------------------
# run_doc — per-doc skip on timeout
# --------------------------------------------------------------------------


class _MockHangingAdapter:
    """Mock LLMAdapter whose calls hang past the test's per-call
    timeout. Used to drive ``run_doc`` into the timeout code path
    without making any network call. Mirrors the dispatcher's
    surface (``parse_structured``, ``generate_text``) and surfaces
    ``LLMCallTimeoutError`` on the per-call budget — exactly what
    the real adapters do.
    """

    def __init__(self):
        self.model = "test-model"

    async def parse_structured(self, *, system, user, schema, call_timeout_s):
        from sum_engine_internal.ensemble.llm_dispatch import LLMCallTimeoutError
        try:
            await asyncio.wait_for(asyncio.sleep(60), timeout=call_timeout_s)
        except asyncio.TimeoutError as e:
            raise LLMCallTimeoutError(
                f"mock.parse_structured: exceeded per-call budget of "
                f"{call_timeout_s:.1f}s"
            ) from e
        raise RuntimeError("should never reach this")

    async def generate_text(self, *, system, user, call_timeout_s):
        from sum_engine_internal.ensemble.llm_dispatch import LLMCallTimeoutError
        try:
            await asyncio.wait_for(asyncio.sleep(60), timeout=call_timeout_s)
        except asyncio.TimeoutError as e:
            raise LLMCallTimeoutError(
                f"mock.generate_text: exceeded per-call budget of "
                f"{call_timeout_s:.1f}s"
            ) from e
        raise RuntimeError("should never reach this")


@pytest.mark.asyncio
async def test_run_doc_returns_timeout_record_when_extract_hangs():
    """When the source-extract call times out, run_doc returns a
    per-doc record tagged ``error_class="timeout"`` rather than
    letting the exception propagate."""
    adapter = _MockHangingAdapter()
    doc = {"id": "test_doc", "text": "Some test prose here.", "gold_triples": []}

    result = await run_doc(
        adapter,
        doc=doc,
        ablation="combined",
        call_timeout_s=0.05,
    )

    assert result["doc_id"] == "test_doc"
    assert result["error_class"] == "timeout"
    assert result["error_what"] == "baseline_extract"
    assert result["error_timeout_s"] == 0.05
    assert result["source_axioms"] == []
    assert result["reconstructed_axioms"] == []
    assert "timeout" in result["narrative_excerpt"].lower()


# --------------------------------------------------------------------------
# aggregate — timeout-aware
# --------------------------------------------------------------------------


def test_aggregate_excludes_timed_out_docs_from_means():
    """A timed-out doc has no drift/recall fields. ``aggregate()``
    must NOT KeyError on it, must exclude it from the means, and
    must report it under ``n_docs_timed_out``."""
    per_doc = [
        {
            "doc_id": "doc_a",
            "drift_pct": 0.0,
            "exact_match_recall": 1.0,
            "n_source": 5,
            "n_reconstructed": 5,
        },
        {
            "doc_id": "doc_b",
            "drift_pct": 50.0,
            "exact_match_recall": 0.5,
            "n_source": 4,
            "n_reconstructed": 6,
        },
        {
            "doc_id": "doc_c",
            "error_class": "timeout",
            "error_what": "baseline_extract",
            "error_timeout_s": 60.0,
            "source_axioms": [],
            "reconstructed_axioms": [],
        },
    ]
    agg = aggregate(per_doc)

    # 3 total docs; 2 measured; 1 timed out.
    assert agg["n_docs"] == 3
    assert agg["n_docs_measured"] == 2
    assert agg["n_docs_timed_out"] == 1
    assert agg["timed_out_doc_ids"] == ["doc_c"]
    # Means computed only over the 2 measured docs.
    assert agg["drift_pct_mean"] == 25.0  # mean(0.0, 50.0)
    assert agg["exact_match_recall_mean"] == 0.75  # mean(1.0, 0.5)
    # Full-recall fraction over measured, not total.
    assert agg["n_docs_full_recall"] == 1
    assert agg["fraction_full_recall"] == 0.5  # 1 of 2 measured


def test_aggregate_with_zero_measured_does_not_zero_divide():
    """If every doc timed out, the means are zero (not NaN, not
    KeyError) and ``n_docs_timed_out`` equals ``n_docs``."""
    per_doc = [
        {
            "doc_id": "doc_a",
            "error_class": "timeout",
            "error_what": "baseline_extract",
            "error_timeout_s": 60.0,
            "source_axioms": [],
            "reconstructed_axioms": [],
        },
        {
            "doc_id": "doc_b",
            "error_class": "timeout",
            "error_what": "canonical_first_generate",
            "error_timeout_s": 60.0,
            "source_axioms": [],
            "reconstructed_axioms": [],
        },
    ]
    agg = aggregate(per_doc)
    assert agg["n_docs"] == 2
    assert agg["n_docs_measured"] == 0
    assert agg["n_docs_timed_out"] == 2
    assert agg["drift_pct_mean"] == 0.0
    assert agg["exact_match_recall_mean"] == 0.0
    assert agg["fraction_full_recall"] == 0.0


def test_default_timeout_is_sixty_seconds():
    """The default per-call timeout is 60s — generous enough for
    multi-paragraph LLM calls, tight enough to catch the
    pathological 14+ minute hang the seed_long capstone surfaced."""
    assert DEFAULT_CALL_TIMEOUT_S == 60.0
