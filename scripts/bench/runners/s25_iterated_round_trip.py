"""T1 of bench-hardening worktrail — iterated round-trip drift.

Spec lives in `docs/BENCH_HARDENING_FROM_QCVV.md` §T1. The single most
powerful idea in QCVV experiment design: shot-count scales sensitivity
as 1/√N; *repetition of the noisy operation inside one experiment*
scales it as 1/L. PROOF_BOUNDARY §2.5's "closure" claim is a single-step
measurement on `seed_v1` (drift 0.00%, exact-match recall 1.000). It
tells you almost nothing about whether canonicalisation is closed
under iteration or merely closed at the first fixed-point neighbourhood.

The runner amplifies. For each document, iterate:

    axioms_0 = extract(text)                        # baseline truth
    for k in 1..K:
        prose_k    = generate(axioms_{k-1})         # canonical-first generator
        axioms_k   = extract(prose_k)               # vocab-pinned extractor
        drift_k    = 1 - exact_match_recall(axioms_k, axioms_0)
        record { k, drift_k, |axioms_k|, set_diff(axioms_k, axioms_0) }

Two informative outcomes:

  - Drift stays flat across k (max drift_k ≤ max drift_1 + ε for ε ≤ 1pp):
    §2.5 closure is a genuine fixed point. PROOF_BOUNDARY §2.5 gains
    a sentence: "closure is stable under K-step iteration."

  - Drift accumulates with k (drift_k grows monotonically or super-
    linearly): §2.5 is qualified. The single-step result is a local
    neighbourhood, not a global fixed point. PROOF_BOUNDARY §2.5
    gains an explicit composition caveat. THIS is the claim that
    needs qualification before any release that cites §2.5 as
    load-bearing for multi-stage pipelines.

Output: NDJSON receipt under `sum.iterated_round_trip_drift.v1`,
written to `fixtures/bench_receipts/s25_iterated_K<k>_<corpus>_<YYYY-MM-DD>.json`.

Runs on the three corpora §2.5 closed against: `seed_v1`, `seed_v2`,
`seed_long_paragraphs`.

Pinned-model-snapshot requirement (PROOF_BOUNDARY §2.6): the receipt
records the exact model id. Raises SystemExit on unpinned identifiers.

Dry-run mode (--dry-run) produces a stubbed receipt with synthetic
drift values — verifies schema + runner shape without LLM spend.
Useful for CI tests.

Cost estimate per BENCH_HARDENING T1: ~10× the per-corpus cost of
`s25_generator_side_combined`. With NIM Llama 3.3 70B on the free
tier (1000-credit allowance), seed_v1 × K=10 fits within budget.

Reproducible:

    # Real run — requires NVIDIA_API_KEY + SUM_TRANSFORM_MODEL set
    python -m scripts.bench.runners.s25_iterated_round_trip \\
        --corpus scripts/bench/corpora/seed_v1.json --k 10 \\
        --out fixtures/bench_receipts/s25_iterated_K10_seed_v1_<date>.json

    # Dry-run (no LLM cost; CI-testable)
    python -m scripts.bench.runners.s25_iterated_round_trip \\
        --corpus scripts/bench/corpora/seed_v1.json --k 3 --dry-run \\
        --out /tmp/iter_dry.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "sum.iterated_round_trip_drift.v1"
PINNED_MODEL_ENV_VAR = "SUM_TRANSFORM_MODEL"


def _utc_iso() -> str:
    t = datetime.now(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _normalize_triple(t: Any) -> tuple[str, str, str]:
    """Componentwise lowercase. Mirrors the canonicalisation in
    sum_engine_internal.transforms.compose so set-diffs across runs are
    comparable."""
    if isinstance(t, dict):
        s = str(t.get("subject", "")).lower()
        p = str(t.get("predicate", "")).lower()
        o = str(t.get("object", t.get("object_", ""))).lower()
        return (s, p, o)
    return (str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower())


def _exact_match_recall(observed: set, truth: set) -> float:
    if not truth:
        return 1.0 if not observed else 0.0
    return len(truth & observed) / len(truth)


def _extract_sieve(text: str) -> list[tuple[str, str, str]]:
    """Synchronous sieve extraction. Returns normalised triples."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

    sieve = DeterministicSieve()
    return [_normalize_triple(t) for t in sieve.extract_triplets(text)]


async def _generate_from_triples(
    triples: list[tuple[str, str, str]],
    model: str,
    *,
    max_retries: int = 8,
    throttle_seconds: float = 2.5,
) -> str:
    """Generate prose from a triple set via the cascade adapter. Uses
    the same routing rules as the slider's LLM-axis dispatch (HF / NIM
    / Groq / Cerebras / Ollama / local / OpenAI per
    LiveLLMAdapter.from_model).

    F12 fix (DOGFOOD_FINDINGS_2026-05-21): NIM enforces a 40 req/min/
    model rate cap independent of credits. The pre-fix version fired
    calls as fast as the LLM responded and crashed mid-K=10-batch on
    429. This version:

      - Throttles each call by ``throttle_seconds`` (default 1.7s ≈ 35
        req/min, safely under the 40-req/min cap)
      - Catches transient errors (429 / 5xx) and retries with
        exponential backoff up to ``max_retries`` times

    For non-NIM providers (Groq, Cerebras, OpenAI) the throttle is
    conservative-but-harmless overhead; the retry is universal value.
    """
    from sum_engine_internal.ensemble.live_llm_adapter import (
        LiveLLMAdapter,
        make_chat_client,
    )

    adapter = LiveLLMAdapter.from_model(model)
    client = make_chat_client(adapter)

    # Canonical-first generator prompt per the §2.5 closure pattern.
    # The intent here is to keep the generator from elaborating beyond
    # the source axioms; iteration stability depends on this.
    system_prompt = (
        "You are translating a structured fact list into prose. Your output "
        "must contain ONLY the facts listed below — do not elaborate, do "
        "not add context, do not infer connections that are not explicitly "
        "stated. Write each fact as a separate sentence in subject-verb-"
        "object order, using the exact terms provided."
    )
    triple_lines = "\n".join(
        f"- ({s}, {p}, {o})" for (s, p, o) in triples
    )
    user_prompt = f"Facts to render as prose:\n{triple_lines}"

    # Throttle: stay safely under NIM's 40-req/min cap.
    if throttle_seconds > 0:
        await asyncio.sleep(throttle_seconds)

    # Exponential backoff on 429 / 5xx. Re-raises after max_retries.
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await client.chat_completion(
                system_prompt, user_prompt, max_tokens=1024
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            # 5xx + 429 are transient. F12 v3 (2026-05-21): seed_v1 K=10
            # crashed on 502 Bad Gateway from NIM — earlier detection
            # caught 500/503/504 but not 502. Broaden to any 5xx + any
            # known transient-exception class.
            is_transient = (
                "rate" in msg
                or "429" in msg
                or "500" in msg
                or "502" in msg
                or "503" in msg
                or "504" in msg
                or "bad gateway" in msg
                or "internal server error" in msg
                or type(e).__name__ in {
                    "RateLimitError", "APIStatusError", "APIError",
                    "APIConnectionError", "APITimeoutError",
                    "InternalServerError", "BadGatewayError",
                }
            )
            if not is_transient or attempt == max_retries - 1:
                raise
            # 429 backoff: minimum 60s (one full window reset on NIM).
            # Other transient errors: standard exponential (3, 5, 9, …).
            is_429 = "429" in msg or type(e).__name__ == "RateLimitError"
            if is_429:
                # 65, 90, 120, 180, 240, 300, 360, 420 seconds — total
                # ~30 minutes of patience across 8 retries. NIM's per-
                # minute cap fully resets in 60s; longer backoffs cover
                # the rare-but-real per-hour soft caps.
                backoff = 65 + 30 * attempt
            else:
                backoff = (2 ** attempt) * 2 + 1
            print(
                f"  [retry] attempt {attempt + 1}/{max_retries} hit "
                f"{type(e).__name__}: {str(e)[:80]} — sleeping {backoff}s",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff)
    if last_err:
        raise last_err
    raise RuntimeError("retry loop exited without return or raise")


def _stub_iteration(
    truth_triples: list[tuple[str, str, str]],
    k: int,
    seed: str,
) -> tuple[list[tuple[str, str, str]], str]:
    """Dry-run synthetic iteration. Drops one triple per iteration step
    (deterministic via hash seed) and emits a placeholder prose. Useful
    for CI testability without LLM spend."""
    h = hashlib.sha256(f"{seed}|{k}".encode()).digest()
    drop_idx = int.from_bytes(h[:2], "big") % max(1, len(truth_triples))
    kept = truth_triples[:drop_idx] + truth_triples[drop_idx + 1:]
    prose = " ".join(f"The {s} {p} {o}." for (s, p, o) in kept)
    return kept, prose


async def run_one_document(
    doc: dict,
    *,
    k: int,
    model: str,
    dry_run: bool,
) -> dict:
    """Run K iterations on one document. Returns per-iteration records."""
    text = doc["text"]
    truth_triples = _extract_sieve(text)
    truth_set = set(truth_triples)
    n_truth = len(truth_set)

    records = []
    current_triples = truth_triples

    for k_step in range(1, k + 1):
        if dry_run:
            next_triples, prose = _stub_iteration(
                truth_triples, k_step, seed=doc.get("id", "doc"),
            )
        else:
            prose = await _generate_from_triples(current_triples, model)
            next_triples = _extract_sieve(prose)

        observed_set = set(next_triples)
        recall = _exact_match_recall(observed_set, truth_set)
        drift = 1.0 - recall
        missing = sorted(truth_set - observed_set)
        extra = sorted(observed_set - truth_set)

        records.append({
            "k": k_step,
            "drift_pct": round(drift * 100, 4),
            "exact_match_recall": round(recall, 6),
            "n_observed": len(observed_set),
            "n_missing": len(missing),
            "n_extra": len(extra),
            "missing_sample": [list(t) for t in missing[:5]],
            "extra_sample": [list(t) for t in extra[:5]],
        })
        current_triples = next_triples

    return {
        "doc_id": doc.get("id", "?"),
        "n_truth_axioms": n_truth,
        "iterations": records,
    }


def aggregate(per_doc_results: list[dict], k: int) -> dict:
    """Per-k aggregate: median / p10 / max drift across all documents."""
    per_k = {step: [] for step in range(1, k + 1)}
    for doc in per_doc_results:
        for rec in doc["iterations"]:
            per_k[rec["k"]].append(rec["drift_pct"])

    def _p10(xs):
        if not xs:
            return None
        return statistics.quantiles(sorted(xs), n=10, method="inclusive")[0] if len(xs) >= 10 else min(xs)

    return {
        "by_k": {
            str(step): {
                "n_docs": len(values),
                "median_drift_pct": (
                    round(statistics.median(values), 4) if values else None
                ),
                "p10_drift_pct": (
                    round(_p10(values), 4) if values else None
                ),
                "max_drift_pct": (
                    round(max(values), 4) if values else None
                ),
            }
            for step, values in per_k.items()
        },
    }


def classify_composition(by_k: dict, k: int, epsilon_pp: float = 1.0) -> dict:
    """Classify whether closure is stable under iteration.

    Returns:
      - "stable": max drift across all K steps is within ε of K=1.
      - "accumulating": drift grows monotonically.
      - "saturating": drift grows then plateaus.
      - "noisy": no clear pattern.
    """
    medians = []
    for step in range(1, k + 1):
        entry = by_k.get(str(step))
        if entry and entry.get("median_drift_pct") is not None:
            medians.append(entry["median_drift_pct"])
    if len(medians) < 2:
        return {"verdict": "insufficient_data", "rationale": f"only {len(medians)} k-steps with data"}

    delta_max = max(medians) - medians[0]
    if delta_max <= epsilon_pp:
        return {
            "verdict": "stable",
            "rationale": (
                f"max-vs-K=1 drift delta = {delta_max:.2f}pp ≤ ε={epsilon_pp}pp. "
                "Closure is robust under K-step iteration."
            ),
            "max_minus_first_pp": round(delta_max, 4),
        }

    is_monotone = all(b >= a - 0.5 for a, b in zip(medians, medians[1:]))
    is_growing = medians[-1] - medians[0] > epsilon_pp
    if is_monotone and is_growing:
        return {
            "verdict": "accumulating",
            "rationale": (
                f"drift grows monotonically: K=1 → {medians[0]:.2f}pp, "
                f"K={k} → {medians[-1]:.2f}pp. §2.5 closure is a local "
                f"neighbourhood, not a global fixed point. PROOF_BOUNDARY §2.5 "
                "needs an explicit composition caveat."
            ),
            "k1_median_pp": medians[0],
            "kfinal_median_pp": medians[-1],
        }

    return {
        "verdict": "noisy",
        "rationale": "drift does not show a monotone trend; needs more samples",
        "medians_by_k": medians,
    }


async def run_corpus(corpus_path: Path, *, k: int, model: str, dry_run: bool) -> dict:
    corpus = json.loads(corpus_path.read_text())
    docs = corpus.get("documents", [])
    if not docs:
        raise SystemExit(f"corpus {corpus_path} has no documents")

    per_doc_results = []
    for i, doc in enumerate(docs, 1):
        if not dry_run:
            print(f"  doc {i}/{len(docs)}: {doc.get('id', '?')}", file=sys.stderr)
        result = await run_one_document(doc, k=k, model=model, dry_run=dry_run)
        per_doc_results.append(result)

    agg = aggregate(per_doc_results, k)
    classification = classify_composition(agg["by_k"], k)

    return {
        "schema": SCHEMA,
        "corpus_id": corpus.get("id"),
        "n_documents": len(docs),
        "k_iterations": k,
        "model": model,
        "generated_at": _utc_iso(),
        "dry_run": dry_run,
        "aggregate": agg,
        "composition_verdict": classification,
        "per_document": per_doc_results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus JSON (e.g. scripts/bench/corpora/seed_v1.json).",
    )
    p.add_argument("--k", type=int, default=10, help="Number of iteration steps (default 10).")
    p.add_argument(
        "--model",
        default=None,
        help=(
            f"Model id for the generator. Defaults to ${PINNED_MODEL_ENV_VAR} "
            "env var. Prefix-routes via LiveLLMAdapter.from_model: nim:, groq:, "
            "cerebras:, ollama:, llamacpp:, local:, org/model (HF), or plain "
            "gpt-... for OpenAI."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; emit synthetic drift values. Useful for CI / schema testing.",
    )
    p.add_argument("--out", default=None, help="Output path for the receipt JSON. Default: stdout.")
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    model = args.model or os.environ.get(PINNED_MODEL_ENV_VAR)
    if not args.dry_run and not model:
        print(
            f"sum: --model not supplied and {PINNED_MODEL_ENV_VAR} env var "
            f"is not set. Either pass --model nim:meta/llama-3.3-70b-instruct "
            f"(or your preferred route — see docs/BYOK_AND_FREE_PROVIDERS.md) "
            f"or run with --dry-run to verify shape without LLM spend.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run and not model:
        model = "(dry-run-stub)"  # marker only; no LLM call

    if not args.dry_run and model.startswith(("gpt-3.5", "gpt-4-0613", "claude-3-sonnet-20240")):
        # Pinned-snapshot guard per PROOF_BOUNDARY §2.6. Reject moving-
        # version model ids that would make the receipt non-reproducible.
        print(
            f"sum: model {model!r} appears to be a moving-version id. "
            f"Pin to a specific snapshot per PROOF_BOUNDARY.md §2.6 — "
            f"e.g. gpt-4o-mini-2024-07-18 instead of gpt-4o.",
            file=sys.stderr,
        )
        return 2

    receipt = asyncio.run(run_corpus(
        Path(args.corpus), k=args.k, model=model, dry_run=args.dry_run,
    ))

    if args.out:
        Path(args.out).write_text(
            json.dumps(receipt, indent=2 if args.pretty else None) + "\n"
        )
        print(f"iterated-round-trip receipt: {args.out}", file=sys.stderr)
    else:
        json.dump(receipt, sys.stdout, indent=2 if args.pretty else None)
        sys.stdout.write("\n")

    verdict = receipt["composition_verdict"]
    print(
        f"corpus={receipt['corpus_id']} K={args.k} verdict={verdict['verdict']}: "
        f"{verdict['rationale']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
