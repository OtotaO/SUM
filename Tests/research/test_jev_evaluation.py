"""Offline workflow tests, not a live accuracy benchmark."""
import copy
import json
import stat
from pathlib import Path

import pytest

from sum_engine_internal.research.meaning.jev_evaluation import (
    INPUT_SCHEMA,
    ReplayTransport,
    evaluate,
    main,
    validate_dataset,
)
from sum_engine_internal.research.meaning.jev_judge import JevConfig, JevError, JevJudge


def dataset():
    return {"schema": INPUT_SCHEMA, "dataset_id": "test", "provenance": "synthetic test",
            "pairs": [{"id": "one", "source_id": "family-1", "source": "Alice arrived.",
                       "output": "Bob left.", "labels": {"recall": [False], "fidelity": [True]}}]}


def judge_with(values, config=None):
    values = iter(values)

    def post(body):
        request = json.loads(body)
        assert "labels" not in body.decode()
        return json.dumps({"model": request["model"], "usage": {"input_tokens": 10, "output_tokens": 3},
                           "answers": {key: {"type": "noul", "noul": next(values)}
                                       for key in request["questions"]}}).encode()

    return JevJudge(config, transport=post)


def test_both_directions_preserve_probabilities_labels_and_kernel_loss():
    report = evaluate(dataset(), judge_with([0.05, 0.95]))
    result = report["results"][0]
    assert result["readout"]["loss"] == 0.6
    assert result["directions"]["recall"][0]["probability"] == 0.05
    assert report["metrics"]["labelled_model_decisions"] == 2
    assert report["metrics"]["brier_score"] == pytest.approx(0.0025)
    assert report["metrics"]["population_confidence_bound"] is None
    assert report["observations"]["requests"][0]["request"]["state"]["premise"] == "Bob left."


def test_abstentions_do_not_hide_the_other_direction_or_produce_a_loss():
    report = evaluate(dataset(), judge_with([0.5, 0.95]))
    assert report["status"] == "complete"
    assert report["results"][0]["status"] == "abstained"
    assert report["results"][0]["readout"] is None
    assert len(report["observations"]["requests"]) == 2
    assert report["metrics"]["abstained"] == 1
    assert report["metrics"]["decision_coverage_among_observed_labels"] == 0.5
    assert report["metrics"]["brier_score"] == pytest.approx(0.12625)


def test_failure_stops_spend_retains_partial_direction_and_marks_incomplete():
    data = dataset()
    other = copy.deepcopy(data["pairs"][0])
    other["id"] = "two"
    data["pairs"].append(other)
    report = evaluate(data, judge_with([0.05], JevConfig(max_requests=1)))
    assert report["status"] == "incomplete"
    assert report["pairs_requested"] == 2
    assert report["pairs_attempted"] == 1
    assert report["results"][0]["status"] == "failed"
    assert "recall" in report["results"][0]["directions"]
    assert report["metrics"]["labelled_model_decisions_requested"] == 4
    assert report["metrics"]["label_evaluation_coverage"] == 0.25
    assert len(report["observations"]["requests"]) == 1


def test_empty_source_does_not_call_model_or_invent_probabilities():
    data = dataset()
    data["pairs"][0].update(source="", labels={"recall": [], "fidelity": [False]})
    report = evaluate(data, JevJudge())
    assert report["results"][0]["readout"]["loss"] == 1
    assert report["results"][0]["directions"]["fidelity"][0]["probability"] is None
    assert report["metrics"]["labelled_model_decisions"] == 0
    assert report["observations"]["requests"] == []


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema="other"),
    lambda d: d.update(pairs=[]),
    lambda d: d["pairs"].append(copy.deepcopy(d["pairs"][0])),
    lambda d: d["pairs"][0].update(source_id=""),
    lambda d: d["pairs"][0]["labels"].update(recall=[]),
    lambda d: d["pairs"][0]["labels"].update(recall=[0]),
    lambda d: d.update(optional_metadata=float("nan")),
])
def test_invalid_dataset_fails_before_any_call(change):
    data = dataset()
    change(data)
    with pytest.raises(JevError):
        evaluate(data, JevJudge())


def test_shipped_synthetic_inputs_are_valid_and_not_claimed_as_live_evidence():
    path = Path(__file__).resolve().parents[2] / "fixtures/jev_evaluation/synthetic_pairs.json"
    data = json.loads(path.read_text())
    validate_dataset(data)
    assert "not human-adjudicated" in data["provenance"]
    assert "No Jev responses" in data["provenance"]


def test_replay_matches_results_and_rejects_mismatched_or_extra_requests():
    data = dataset()
    original = evaluate(data, judge_with([0.05, 0.95]))
    replay = ReplayTransport(original["observations"])
    rerun = evaluate(data, JevJudge(transport=replay))
    replay.finish()
    assert rerun["results"] == original["results"]
    assert rerun["metrics"] == original["metrics"]
    replay = ReplayTransport(original["observations"])
    with pytest.raises(JevError, match="unconsumed"):
        replay.finish()
    with pytest.raises(JevError, match="differs"):
        replay(b"{}")
    corrupted = copy.deepcopy(original["observations"])
    corrupted["requests"][0]["response_hash"] = "tampered"
    with pytest.raises(JevError, match="digest"):
        ReplayTransport(corrupted)


def test_cli_live_and_replay_with_fake_transport_and_private_output(tmp_path, monkeypatch):
    data = dataset()
    source = tmp_path / "input.json"
    source.write_text(json.dumps(data))
    live_path, replay_path = tmp_path / "live.json", tmp_path / "replay.json"

    def fake_from_env(**kwargs):
        assert kwargs["allow_network"] is True
        return judge_with([0.05, 0.95], kwargs["config"])

    monkeypatch.setattr(JevJudge, "from_env", fake_from_env)
    assert main(["--input", str(source), "--out", str(live_path), "--allow-network"]) == 0
    assert stat.S_IMODE(live_path.stat().st_mode) == 0o600
    live = json.loads(live_path.read_text())

    def forbidden(**kwargs):
        pytest.fail("offline replay constructed a network client")

    monkeypatch.setattr(JevJudge, "from_env", forbidden)
    assert main(["--input", str(source), "--out", str(replay_path), "--replay", str(live_path)]) == 0
    rerun = json.loads(replay_path.read_text())
    assert rerun["execution"] == "offline_response_replay"
    assert live["results"] == rerun["results"]


def test_output_collision_prevents_network_spend(tmp_path, monkeypatch):
    source, out = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps(dataset()))
    out.write_text("keep me")
    judge = JevJudge(transport=lambda _: pytest.fail("existing output must stop all calls"))
    monkeypatch.setattr(JevJudge, "from_env", lambda **_: judge)
    assert main(["--input", str(source), "--out", str(out), "--allow-network"]) == 2
    assert out.read_text() == "keep me"


def test_cli_requires_explicit_execution_mode(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--input", str(tmp_path / "in"), "--out", str(tmp_path / "out")])
    assert error.value.code == 2


def test_replay_rejects_changed_threshold_before_output_creation(tmp_path):
    data = dataset()
    source, saved, out = tmp_path / "in.json", tmp_path / "saved.json", tmp_path / "out.json"
    source.write_text(json.dumps(data))
    saved.write_text(json.dumps(evaluate(data, judge_with([0.05, 0.95], JevConfig(max_response_bytes=64_000)))))
    assert main(["--input", str(source), "--out", str(out), "--replay", str(saved),
                 "--accept-at-or-above", "0.95"]) == 2
    assert not out.exists()


def test_report_size_is_rejected_before_model_calls():
    data = dataset()
    data["pairs"] = [{"id": str(i), "source_id": str(i), "source": "a" * 40_000,
                      "output": "b" * 40_000} for i in range(128)]
    judge = JevJudge(JevConfig(max_requests=256), transport=lambda _: pytest.fail("preflight must reject"))
    with pytest.raises(JevError, match="report budget"):
        evaluate(data, judge)
    assert judge.observations()["requests"] == []


def test_replay_checks_recomputed_results(tmp_path):
    data = dataset()
    source, saved, out = tmp_path / "in.json", tmp_path / "saved.json", tmp_path / "out.json"
    source.write_text(json.dumps(data))
    report = evaluate(data, judge_with([0.05, 0.95], JevConfig(max_response_bytes=64_000)))
    report["results"][0]["readout"]["loss"] = 0
    saved.write_text(json.dumps(report))
    assert main(["--input", str(source), "--out", str(out), "--replay", str(saved)]) == 2
    assert json.loads(out.read_text())["status"] == "interrupted_or_failed"


def test_reusing_a_judge_is_rejected_to_avoid_unbound_old_observations():
    judge = judge_with([0.05, 0.95])
    evaluate(dataset(), judge)
    with pytest.raises(JevError, match="fresh judge"):
        evaluate(dataset(), judge)


@pytest.mark.parametrize("instrument", [None, []])
def test_malformed_replay_instrument_is_rejected_before_output(tmp_path, instrument):
    data = dataset()
    source, saved, out = tmp_path / "in.json", tmp_path / "saved.json", tmp_path / "out.json"
    source.write_text(json.dumps(data))
    report = evaluate(data, judge_with([0.05, 0.95], JevConfig(max_response_bytes=64_000)))
    report["scorer_instrument"] = instrument
    saved.write_text(json.dumps(report))
    assert main(["--input", str(source), "--out", str(out), "--replay", str(saved)]) == 2
    assert not out.exists()
