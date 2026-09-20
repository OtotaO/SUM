"""Run or replay an unsigned, descriptive Jev evaluation over source/output pairs.

Run ``python -m sum_engine_internal.research.meaning.jev_evaluation --help``.
Live calls require --allow-network. Replay never constructs a network client.
Labels are caller-supplied and never included in model requests.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .jev_judge import JevConfig, JevDecision, JevError, JevJudge, _digest, _json_bytes
from .meaning_loss import _sentences, explain_meaning_loss

INPUT_SCHEMA = "sum.jev_evaluation_input.v1"
REPORT_SCHEMA = "sum.jev_evaluation.v1"
MAX_DATASET_BYTES = 16_000_000
MAX_REPORT_BYTES = 64_000_000


def _read_json(path: Path, limit: int) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise JevError("evaluation file exceeds byte limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise JevError("evaluation file must be a JSON object")
    return value


def validate_dataset(dataset: dict) -> None:
    if not isinstance(dataset, dict):
        raise JevError("evaluation input must be an object")
    if dataset.get("schema") != INPUT_SCHEMA:
        raise JevError("unsupported evaluation input schema")
    for key in ("dataset_id", "provenance"):
        if not isinstance(dataset.get(key), str) or not dataset[key].strip():
            raise JevError(f"{key} must identify the supplied dataset")
    pairs = dataset.get("pairs")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 256:
        raise JevError("evaluation needs 1 to 256 pairs")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise JevError("each pair must be an object")
        for key in ("id", "source_id", "source", "output"):
            if not isinstance(pair.get(key), str):
                raise JevError(f"pair {key} must be text")
        if not pair["id"] or not pair["source_id"] or pair["id"] in seen:
            raise JevError("pair IDs must be nonempty and unique; source_id must be nonempty")
        seen.add(pair["id"])
        for key in ("source", "output"):
            if (len(pair[key]) > 256_000 or len(pair[key].encode("utf-8")) > 256_000
                    or len(_sentences(pair[key])) > 4096):
                raise JevError("source/output exceeds text or claim limit")
        labels = pair.get("labels")
        if labels is not None:
            if not isinstance(labels, dict) or labels.keys() != {"recall", "fidelity"}:
                raise JevError("labels must contain recall and fidelity arrays")
            for direction, text in (("recall", pair["source"]), ("fidelity", pair["output"])):
                values = labels[direction]
                if (not isinstance(values, list) or len(values) != len(_sentences(text))
                        or any(type(v) is not bool for v in values)):
                    raise JevError("labels must be Booleans aligned to the sentence unitizer")
    # Validate even optional metadata before calls or output reservation.
    if len(_json_bytes(dataset)) > MAX_DATASET_BYTES:
        raise JevError("dataset exceeds byte limit")


def _preflight_report_size(dataset: dict, judge: JevJudge) -> None:
    # Compact JSON is also the output format. Four dataset copies cover retained
    # input, per-claim text and dropped/unsupported text. Per-row/pair allowances
    # cover keys, hashes, probabilities and readouts; request JSON is counted
    # directly and responses reserve their maximum base64 length before spend.
    upper = 4 * len(_json_bytes(dataset)) + 65_536
    for pair in dataset["pairs"]:
        src, out = _sentences(pair["source"]), _sentences(pair["output"])
        upper += 8192 + 512 * (len(src) + len(out))
        if src and out:
            for premise, hypotheses in ((pair["output"], src), (pair["source"], out)):
                for request in judge._batches(premise, hypotheses):
                    upper += (len(_json_bytes(request)) + 2048
                              + 4 * ((judge.config.max_response_bytes + 2) // 3))
        if upper > MAX_REPORT_BYTES:
            raise JevError("report budget exceeded before inference; split the dataset or lower response cap")


class ReplayTransport:
    """Require exact ordered request bytes and recorded response digests.

    Hash checks detect inconsistency, not authorship: an unsigned file can be
    rewritten with new matching hashes. No hosted inference is rerun here.
    """

    def __init__(self, observations: dict):
        if not isinstance(observations, dict) or observations.get("schema") != "sum.jev_observations.v1":
            raise JevError("unsupported observation schema")
        records = observations.get("requests")
        if not isinstance(records, list) or len(records) > 10_000:
            raise JevError("invalid observation request list")
        self.records = []
        self.index = 0
        for record in records:
            if not isinstance(record, dict) or record.get("status") != "answered":
                raise JevError("only completed responses can be replayed")
            try:
                body = _json_bytes(record["request"])
                raw = base64.b64decode(record["response_base64"], validate=True)
                valid = (_digest(body) == record["request_hash"]
                         and _digest(raw) == record["response_hash"])
            except (KeyError, TypeError, ValueError, binascii.Error):
                raise JevError("invalid replay record") from None
            if not valid or len(body) > 256_000 or len(raw) > 4_000_000:
                raise JevError("replay record digest or byte limit mismatch")
            self.records.append((body, raw))

    def __call__(self, body: bytes) -> bytes:
        if self.index >= len(self.records) or self.records[self.index][0] != body:
            raise JevError("replay request differs from recorded request")
        raw = self.records[self.index][1]
        self.index += 1
        return raw

    def finish(self) -> None:
        if self.index != len(self.records):
            raise JevError("replay contains unconsumed requests")


def _readout(source: str, output: str, directions: dict, config: JevConfig) -> dict | None:
    if any(row["verdict"] == "abstain" for rows in directions.values() for row in rows):
        return None
    # Reuse the existing loss kernel, including its empty-input behavior.
    batches = iter([([r["verdict"] == "supported" for r in directions["recall"]]),
                    ([r["verdict"] == "supported" for r in directions["fidelity"]])])
    result = explain_meaning_loss(
        source, output, entails=lambda p, h: False,
        entails_batch=lambda p, hs: next(batches),
        judge_name="typesafe-jev-noul", judge_version=config.model,
    )
    return asdict(result)


def _metrics(results: list[dict]) -> dict:
    rows = [row for result in results for direction in result.get("directions", {}).values()
            for row in direction if "label" in row and row["probability"] is not None]
    decided = [r for r in rows if r["verdict"] != "abstain"]
    accepted = [r for r in decided if r["verdict"] == "supported"]
    return {
        "scope": "descriptive claim-level counts against caller-supplied labels; claims may correlate",
        "labelled_model_decisions": len(rows), "decided": len(decided),
        "abstained": len(rows) - len(decided),
        "false_supports": sum(not r["label"] for r in accepted),
        "false_rejections": sum(r["label"] for r in decided if r["verdict"] == "not_supported"),
        "decision_coverage_among_observed_labels": len(decided) / len(rows) if rows else None,
        "false_support_fraction_of_accepted": (
            sum(not r["label"] for r in accepted) / len(accepted) if accepted else None),
        "brier_score": (sum((r["probability"] - r["label"]) ** 2 for r in rows) / len(rows)
                        if rows else None),
        "population_confidence_bound": None,
    }


def evaluate(dataset: dict, judge: JevJudge) -> dict:
    validate_dataset(dataset)
    if judge.observations()["requests"]:
        raise JevError("evaluation requires a fresh judge session")
    _preflight_report_size(dataset, judge)
    results = []
    for pair in dataset["pairs"]:
        result = {"id": pair["id"], "source_id": pair["source_id"],
                  "source_hash": _digest(pair["source"].encode("utf-8")),
                  "output_hash": _digest(pair["output"].encode("utf-8")),
                  "status": "complete", "directions": {}, "readout": None}
        results.append(result)
        source_units, output_units = _sentences(pair["source"]), _sentences(pair["output"])
        try:
            for direction, premise, units in (("recall", pair["output"], source_units),
                                              ("fidelity", pair["source"], output_units)):
                model_used = bool(source_units and output_units)
                decisions = (judge.assess_batch(premise, units) if model_used else
                             [JevDecision(0.0, "not_supported") for _ in units])
                rows = []
                for i, (text, decision) in enumerate(zip(units, decisions)):
                    row = {"claim_index": i, "text": text, "verdict": decision.verdict,
                           "probability": decision.probability if model_used else None,
                           "basis": "model" if model_used else "empty_input_rule"}
                    if pair.get("labels") is not None:
                        row["label"] = pair["labels"][direction][i]
                    rows.append(row)
                result["directions"][direction] = rows
            result["readout"] = _readout(pair["source"], pair["output"], result["directions"], judge.config)
            if result["readout"] is None:
                result["status"] = "abstained"
        except JevError:
            result["status"] = "failed"
            result["error"] = "Jev request failed; inspect observations and resource limits"
            break  # Stop spend after a provider/configuration/protocol failure.
    complete = len(results) == len(dataset["pairs"]) and all(r["status"] != "failed" for r in results)
    metrics = _metrics(results)
    requested_labels = sum(len(p["labels"]["recall"]) + len(p["labels"]["fidelity"])
                           for p in dataset["pairs"] if p.get("labels") is not None
                           and _sentences(p["source"]) and _sentences(p["output"]))
    metrics.update(
        labelled_model_decisions_requested=requested_labels,
        label_evaluation_coverage=(metrics["labelled_model_decisions"] / requested_labels
                                   if requested_labels else None),
        coverage_note="observed rows include completed directions only; partial responses remain in observations",
    )
    return {
        "schema": REPORT_SCHEMA, "status": "complete" if complete else "incomplete",
        "scope": "unsigned descriptive evaluation; no semantic guarantee or verified human labels",
        "dataset": dataset, "dataset_hash": _digest(_json_bytes(dataset)),
        "source_groups": len({p["source_id"] for p in dataset["pairs"]}),
        "unitizer": "punctuation-whitespace-or-newline-v1",
        "evaluation_implementation_hash": _digest(Path(__file__).read_bytes()),
        "scorer_instrument": judge.as_scorer().instrument,
        "pairs_requested": len(dataset["pairs"]), "pairs_attempted": len(results),
        "pairs_unattempted": len(dataset["pairs"]) - len(results),
        "pairs_failed": sum(r["status"] == "failed" for r in results),
        "pairs_abstained": sum(r["status"] == "abstained" for r in results),
        "results": results, "metrics": metrics, "observations": judge.observations(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source/output dataset JSON")
    parser.add_argument("--out", type=Path, required=True, help="new private report file; never overwritten")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-network", action="store_true", help="explicitly enable paid TypeSafe calls")
    mode.add_argument("--replay", type=Path, help="offline replay of a previously saved report")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-response-bytes", type=int, default=64_000)
    parser.add_argument("--accept-at-or-above", type=float, default=0.9)
    parser.add_argument("--reject-at-or-below", type=float, default=0.1)
    args = parser.parse_args(argv)
    evaluation_started = False
    try:
        dataset = _read_json(args.input, MAX_DATASET_BYTES)
        validate_dataset(dataset)
        replay = None
        config = JevConfig(model=args.model, max_requests=args.max_requests,
                           max_response_bytes=args.max_response_bytes,
                           accept_at_or_above=args.accept_at_or_above,
                           reject_at_or_below=args.reject_at_or_below)
        if args.replay:
            saved = _read_json(args.replay, MAX_REPORT_BYTES)
            if saved.get("schema") != REPORT_SCHEMA or saved.get("status") != "complete":
                raise JevError("replay requires a complete evaluation report")
            if saved.get("dataset_hash") != _digest(_json_bytes(dataset)):
                raise JevError("replay dataset does not match")
            observations = saved.get("observations")
            replay = ReplayTransport(observations)
            judge = JevJudge(config, transport=replay)
            if observations.get("instrument") != judge.manifest():
                raise JevError("replay requires matching model, thresholds, limits, and adapter implementation")
            current = judge.as_scorer().instrument
            previous = saved.get("scorer_instrument", {})
            if not isinstance(previous, dict):
                raise JevError("replay scorer instrument must be an object")
            if (saved.get("evaluation_implementation_hash") != _digest(Path(__file__).read_bytes())
                    or any(previous.get(k) != current[k] for k in ("implementation", "configuration"))):
                raise JevError("replay requires the recorded scoring and evaluation implementation")
        else:
            judge = JevJudge.from_env(allow_network=args.allow_network, config=config)

        _preflight_report_size(dataset, judge)

        # Reserve the output before inference. Exclusive creation rejects symlinks
        # and existing files; mode 0600 avoids exposing retained source text.
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            try:
                evaluation_started = True
                report = evaluate(dataset, judge)
                if replay:
                    replay.finish()
                    if (_json_bytes(report["results"]) != _json_bytes(saved.get("results"))
                            or report["metrics"] != saved.get("metrics")):
                        raise JevError("recomputed results differ from the recorded evaluation")
                report["execution"] = "offline_response_replay" if replay else "live_inference"
                exit_code = 0 if report["status"] == "complete" else 2
            except (Exception, KeyboardInterrupt):
                report = {"schema": REPORT_SCHEMA, "status": "interrupted_or_failed",
                          "dataset": dataset, "observations": judge.observations(),
                          "scope": "partial unsigned observations; no evaluation result"}
                exit_code = 2
            json.dump(report, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        print(f"Jev evaluation {report['status']}; report saved", file=sys.stderr)
        return exit_code
    except (OSError, ValueError, TypeError, RecursionError):
        message = ("Jev evaluation report could not be saved; inference may have run" if evaluation_started else
                   "Jev evaluation could not start: check input, mode, configuration, report budget, and unused output path")
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
