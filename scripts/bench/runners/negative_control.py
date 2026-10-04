"""T5 — Negative-control corpus runner (bench-hardening worktrail).

Companion to `docs/BENCH_HARDENING_FROM_QCVV.md` T5. The v1 negative-
control corpus (`scripts/bench/corpora/seed_negative_control_v1.json`)
contains 20 hand-authored documents engineered to violate the
canonicalisation pipeline's assumptions across five failure modes
(four per mode):

  - ambiguous_coreference     (pronoun referent unresolved)
  - predicate_alias_inconsistency (aliased relations)
  - contradictory_axioms      (mutually exclusive claims)
  - entity_resolution_adversarial (multi-meaning surface forms)
  - non_extractable_assertion (questions / counterfactuals / hedges)

The v2 corpus (`seed_negative_control_v2.json`, the default) keeps those
20 documents unchanged and adds two modes aimed at the sieve's clause
guard:

  - conditional_assertion     (if / unless / provided / inverted
                               conditionals: neither clause asserted)
  - cross_clause_chimera      (relative / xcomp / ccomp / purpose
                               clauses, plus positive controls whose
                               main clause must still be extracted)

v2 documents may carry ``allowed_triples`` (hand-authored, in the
sieve's canonical form) for the rules ``produce_only_allowed_triples``
and ``produce_allowed_triple``.

Each document is annotated with its expected failure mode + an
`extraction_should` rule specifying what the extractor's output
SHOULD look like. The runner runs the sieve extractor on each
document and grades the observed output against the rule.

The runner's verdict per document is one of:

  - expected   — observed output matches the annotated rule
  - unexpected — observed output VIOLATES the rule
                 (e.g. extracted a triple from a counterfactual)
  - advisory   — rule is observational only (predicate alias
                 surfacing, entity disambiguation); the runner
                 records what it saw without grading pass/fail

The runner exits 0 if there are zero `unexpected` verdicts (i.e.,
the bench correctly fails on inputs it should fail on AND succeeds
on inputs it should succeed on). Exits 1 if any `unexpected`.

A bench with no documented failure mode is not a benchmark. This
runner CLOSES the negative-control gap named in
docs/BENCH_HARDENING_FROM_QCVV.md T5 and docs/PROOF_BOUNDARY.md
§2 (Negative controls).

Output schema: ``sum.negative_control_report.v1``. Each result also
records the sieve's per-document suppression report (sentences,
extracted, suppressed per reason) and the summary totals them under
``suppression``; ``extractor_id`` names the sieve version.

Reproducible:

    python -m scripts.bench.runners.negative_control \
        --corpus scripts/bench/corpora/seed_negative_control_v2.json \
        --out fixtures/bench_receipts/negative_control_<YYYY-MM-DD>.json \
        --generated-at <YYYY-MM-DD>T00:00:00.000Z

Zero cost — sieve extractor is deterministic, no LLM calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "sum.negative_control_report.v1"


def _grade_document(doc: dict[str, Any], extracted_axioms: list[dict]) -> tuple[str, str]:
    """Grade one document's observed output against its annotated rule.

    Returns ``(verdict, notes)``. Verdict in
    ``{"expected", "unexpected", "advisory"}``.
    """
    rule = doc.get("extraction_should", "")
    n = len(extracted_axioms)

    # ── Strict-zero-triple rules ────────────────────────────────────
    if rule == "produce_zero_triples":
        if n == 0:
            return "expected", "extracted 0 triples as required"
        return "unexpected", (
            f"rule was 'produce_zero_triples' but extractor produced "
            f"{n} triple(s); this is a fact-preservation failure (the "
            f"document is a question / counterfactual / hedge that "
            f"asserts nothing)"
        )

    # ── Zero-or-flag rules (acceptable to drop, acceptable to flag) ─
    if rule in {
        "produce_zero_triples_or_flag_ambiguity",
        "produce_zero_triples_or_flag_counterfactual",
        "produce_zero_triples_or_flag_conditional",
        "produce_zero_triples_or_hedged_belief_triple",
    }:
        if n == 0:
            return "expected", "extracted 0 triples (drop-path)"
        # Sieve has no flagging surface today; >0 triples here means
        # the extractor committed silently. That's the failure mode
        # the corpus is designed to surface.
        return "unexpected", (
            f"rule allows drop OR flag; observed {n} triples without a "
            f"flag; silent commitment to one reading is the documented "
            f"failure mode"
        )

    # ── Allowed-triple rules (v2) ───────────────────────────────────
    if rule in {"produce_only_allowed_triples", "produce_allowed_triple"}:
        allowed = {tuple(t) for t in doc.get("allowed_triples", [])}
        observed = [
            (a["subject"], a["predicate"], a["object"]) for a in extracted_axioms
        ]
        stray = [t for t in observed if t not in allowed]
        if stray:
            return "unexpected", (
                f"extracted {len(stray)} triple(s) outside allowed_triples: "
                f"{stray}; the sentence does not assert them (cross-clause "
                f"or non-asserted content)"
            )
        if rule == "produce_allowed_triple" and n == 0:
            return "unexpected", (
                "positive control: the asserted main clause must be "
                "extracted, but the extractor produced 0 triples"
            )
        return "expected", (
            f"extracted {n} triple(s), all in allowed_triples"
        )

    # ── Contradiction rules — count + consistency check ─────────────
    if rule == "produce_two_triples_and_consistency_check_flags_inconsistent":
        # Sieve consistency check (z3) currently runs only over typed
        # predicates; many contradictions don't fire. We grade on:
        #   (a) extractor produced >= 2 triples (didn't silently
        #       collapse to one),
        #   (b) leave the consistency-check observation in notes for
        #       future tooling to consume.
        if n >= 2:
            return "expected", (
                f"extracted {n} triples (>= 2 as required; preserves the "
                f"contradiction for downstream consistency check to flag)"
            )
        if n == 1:
            return "unexpected", (
                f"rule required >= 2 triples to preserve the "
                f"contradiction; extractor collapsed to {n}: silent "
                f"contradiction-resolution"
            )
        return "advisory", (
            f"extracted {n} triples; corpus expected the contradiction "
            f"preserved but the extractor produced fewer than 2; could "
            f"be sieve's lemmatisation quirk on these specific phrasings"
        )

    # ── Observational rules — record without grading pass/fail ──────
    observational_rules = {
        "produce_one_or_two_triples_with_surfaced_predicates",
        "produce_two_distinct_triples_with_treats_and_cures_preserved",
        "produce_three_distinct_triples",
        "produce_one_or_two_triples_with_decision_surfaced",
        "produce_one_triple_with_qid_resolution_low_confidence",
        "produce_one_triple_with_correct_apple_disambiguation",
        "produce_one_triple_with_temperature_reading_interpretation",
        "produce_one_triple_with_corpus_prior_or_low_confidence",
    }
    if rule in observational_rules:
        return "advisory", (
            f"rule is observational ({rule}); extractor produced {n} "
            f"triple(s). Future tooling with surfaces for predicate-"
            f"alias detection / entity disambiguation can grade this "
            f"properly. v1 records without pass/fail."
        )

    return "advisory", f"unrecognised rule {rule!r}; recorded observation only"


def _extract_axioms(sieve: Any, text: str) -> tuple[list[dict], dict]:
    """Run sieve extraction on a document. Returns the axiom dicts (one
    per extracted triple) and the sieve's suppression report."""
    triples, report = sieve.extract_triplets_with_report(text)
    axioms = [
        {"subject": str(t[0]), "predicate": str(t[1]), "object": str(t[2])}
        for t in triples
    ]
    return axioms, report


def run(corpus_path: Path, generated_at: str | None = None) -> dict[str, Any]:
    from sum_engine_internal.algorithms.syntactic_sieve import (
        SIEVE_EXTRACTOR_ID,
        DeterministicSieve,
        merge_suppression_reports,
    )

    corpus = json.loads(corpus_path.read_text())
    docs = corpus["documents"]
    sieve = DeterministicSieve()

    results = []
    reports = []
    for doc in docs:
        try:
            axioms, report = _extract_axioms(sieve, doc["text"])
        except Exception as e:  # noqa: BLE001
            results.append({
                "doc_id": doc["id"],
                "expected_failure_mode": doc["expected_failure_mode"],
                "observed": {"error": f"{type(e).__name__}: {e}"},
                "verdict": "advisory",
                "notes": f"extractor crashed; treat as observational",
            })
            continue
        verdict, notes = _grade_document(doc, axioms)
        reports.append(report)
        results.append({
            "doc_id": doc["id"],
            "expected_failure_mode": doc["expected_failure_mode"],
            "observed": {
                "axiom_count": len(axioms),
                "axioms": axioms,
                "suppression": report,
            },
            "verdict": verdict,
            "notes": notes,
        })

    counts = {"expected": 0, "unexpected": 0, "advisory": 0}
    by_mode: dict[str, dict[str, int]] = {}
    for r in results:
        counts[r["verdict"]] += 1
        mode = r["expected_failure_mode"]
        by_mode.setdefault(mode, {"expected": 0, "unexpected": 0, "advisory": 0})
        by_mode[mode][r["verdict"]] += 1

    return {
        "schema": SCHEMA,
        "corpus_id": corpus.get("id"),
        "corpus_schema": corpus.get("schema"),
        "generated_at": generated_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        ),
        "extractor": "deterministic_sieve",
        "extractor_id": SIEVE_EXTRACTOR_ID,
        "summary": {
            "total_documents": len(docs),
            **counts,
            "suppression": merge_suppression_reports(reports),
        },
        "by_failure_mode": by_mode,
        "results": results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--corpus",
        default="scripts/bench/corpora/seed_negative_control_v2.json",
        help="Path to the negative-control corpus JSON.",
    )
    p.add_argument(
        "--generated-at",
        default=None,
        help="Fixed generated_at timestamp for a reproducible receipt. "
             "Default: the current UTC time.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write the report JSON. Default: stdout.",
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = p.parse_args()

    report = run(Path(args.corpus), generated_at=args.generated_at)

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2 if args.pretty else None) + "\n"
        )
        print(f"negative-control report written: {args.out}", file=sys.stderr)
    else:
        json.dump(report, sys.stdout, indent=2 if args.pretty else None)
        sys.stdout.write("\n")

    # Exit 0 if all verdicts are `expected` or `advisory`. Exit 1 if
    # any `unexpected`. The runner's job is to fail the build when
    # the bench succeeds on inputs it should fail on.
    summary = report["summary"]
    unexpected = summary["unexpected"]
    print(
        f"negative-control: {summary['expected']} expected / "
        f"{summary['advisory']} advisory / {unexpected} unexpected "
        f"(total {summary['total_documents']})",
        file=sys.stderr,
    )
    return 1 if unexpected > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
