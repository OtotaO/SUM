"""Protocol/budget regressions only. Fake responses do not validate Jev accuracy."""
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from sum_engine_internal.research.meaning.jev_judge import (
    ENDPOINT,
    JevAbstention,
    JevConfig,
    JevError,
    JevJudge,
    _http_transport,
    _NoRedirect,
)


def response(request, values):
    return json.dumps({
        "model": request["model"],
        "answers": {key: {"type": "noul", "noul": value}
                    for key, value in zip(request["questions"], values)},
        "usage": {"input_tokens": 123, "output_tokens": 8},
    }).encode()


def fake(values=(0.95,)):
    requests = []

    def transport(body):
        request = json.loads(body)
        requests.append(request)
        return response(request, values)

    return transport, requests


def test_network_is_opt_in_even_with_a_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-value")
    with pytest.raises(JevError, match="allow_network"):
        JevJudge.from_env()
    with pytest.raises(JevError, match="networking is disabled"):
        JevJudge().entails("premise", "hypothesis")


@pytest.mark.parametrize("key", ["", "bad\nheader", "not ascii \u2603"])
def test_invalid_credentials_never_construct_transport(key, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", key)
    with pytest.raises(JevError, match="credential"):
        JevJudge.from_env(allow_network=True)


@pytest.mark.parametrize("kwargs", [
    {"model": "jev-latest"}, {"model": "jev-1.13"},
    {"accept_at_or_above": float("nan")}, {"reject_at_or_below": True},
    {"accept_at_or_above": 10**400}, {"timeout_seconds": 10**400},
    {"accept_at_or_above": 0.05}, {"reject_at_or_below": -0.1},
    {"max_requests": 0}, {"max_questions": True}, {"max_request_bytes": 1.5},
    {"timeout_seconds": 0}, {"timeout_seconds": float("inf")},
])
def test_config_rejects_ambiguous_or_unbounded_values(kwargs):
    with pytest.raises(JevError):
        JevConfig(**kwargs)


def test_batch_contains_actual_hypotheses_and_preserves_order():
    transport, requests = fake((0.1, 0.9))
    judge = JevJudge(transport=transport)
    assert judge.entails_batch("Full source \u2603", ["First claim", "Second claim"]) == [False, True]
    assert len(requests) == 1
    assert requests[0]["state"] == {"premise": "Full source \u2603"}
    assert requests[0]["questions"]["claim_0"]["instructions"]["hypothesis"] == "First claim"
    records = judge.observations()["requests"]
    assert json.loads(base64.b64decode(records[0]["response_base64"]))["usage"]["input_tokens"] == 123
    records[0]["request"]["state"]["premise"] = "tampered"
    assert judge.observations()["requests"][0]["request"]["state"]["premise"] == "Full source \u2603"


def test_uncertainty_is_retained_and_never_converted_to_false():
    transport, _ = fake((0.5, 0.99))
    judge = JevJudge(transport=transport)
    with pytest.raises(JevAbstention) as error:
        judge.entails_batch("premise", ["uncertain", "supported"])
    assert error.value.indices == (0,)
    assert judge.observations()["requests"][0]["status"] == "answered"
    assert judge.assess_batch("premise", ["uncertain", "supported"])[0].verdict == "abstain"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model="jev-1.14.0"),
    lambda r: r.update(answers={}),
    lambda r: r["answers"].update(extra={"type": "noul", "noul": 1}),
    lambda r: r["answers"]["claim_0"].update(type="choice"),
    lambda r: r["answers"]["claim_0"].update(noul=True),
    lambda r: r["answers"]["claim_0"].update(noul="0.99"),
    lambda r: r["answers"]["claim_0"].update(noul=float("nan")),
    lambda r: r["answers"]["claim_0"].update(noul=float("inf")),
    lambda r: r["answers"]["claim_0"].update(noul=1.01),
    lambda r: r.pop("usage"),
    lambda r: r["usage"].update(input_tokens=True),
    lambda r: r["usage"].update(input_tokens=-1),
])
def test_invalid_responses_fail_without_a_score(mutation):
    def transport(body):
        r = json.loads(response(json.loads(body), [0.99]))
        mutation(r)
        return json.dumps(r).encode()

    judge = JevJudge(transport=transport)
    with pytest.raises(JevError):
        judge.entails("premise", "hypothesis")
    assert judge.observations()["requests"][0]["status"] == "failed"


@pytest.mark.parametrize("raw", [b"not json", b"\xff", b'{"model":1,"model":2}', b"[]"])
def test_bad_json_is_retained_for_inspection(raw):
    judge = JevJudge(transport=lambda body: raw)
    with pytest.raises(JevError):
        judge.entails("a", "b")
    assert base64.b64decode(judge.observations()["requests"][0]["response_base64"]) == raw


def test_batch_budget_preflight_prevents_partial_spend():
    transport, requests = fake()
    judge = JevJudge(JevConfig(max_questions=1, max_requests=1), transport=transport)
    with pytest.raises(JevError, match="budget"):
        judge.entails_batch("premise", ["a", "b"])
    assert requests == []
    assert judge.entails("premise", "a")
    with pytest.raises(JevError, match="budget"):
        judge.entails("premise", "b")
    assert len(requests) == 1


def test_failed_attempt_is_charged_and_error_does_not_leak_secrets():
    def transport(body):
        raise JevError("private-credential")

    judge = JevJudge(JevConfig(max_requests=1), transport=transport)
    with pytest.raises(JevError) as error:
        judge.entails("a", "b")
    assert "private-credential" not in str(error.value)
    assert error.value.__suppress_context__
    assert "private-credential" not in json.dumps(judge.observations())
    with pytest.raises(JevError, match="budget"):
        judge.entails("a", "b")


def test_unicode_byte_limit_rejects_before_network_without_truncation():
    transport, requests = fake()
    judge = JevJudge(JevConfig(max_request_bytes=1000), transport=transport)
    with pytest.raises(JevError, match="not truncated"):
        judge.entails("\U0001f600" * 400, "claim")
    assert requests == []


def test_unpaired_surrogate_is_a_bounded_preflight_failure():
    transport, requests = fake()
    with pytest.raises(JevError, match="UTF-8"):
        JevJudge(transport=transport).entails("\ud800", "claim")
    assert requests == []


def test_empty_batch_and_question_splitting():
    assert JevJudge().entails_batch("anything", []) == []
    transport, requests = fake()
    judge = JevJudge(JevConfig(max_questions=1), transport=transport)
    assert judge.entails_batch("source", ["one", "two", "three"]) == [True] * 3
    assert len(requests) == 3


def test_concurrent_calls_cannot_overrun_attempt_budget():
    transport, requests = fake()
    judge = JevJudge(JevConfig(max_requests=1), transport=transport)

    def call():
        try:
            return judge.entails("source", "claim")
        except JevError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: call(), range(4)))
    assert results.count(True) == 1
    assert len(requests) == 1


def test_existing_bidirectional_scorer_uses_two_batches():
    def transport(body):
        request = json.loads(body)
        return response(request, [0.99] * len(request["questions"]))

    judge = JevJudge(transport=transport)
    scorer = judge.as_scorer()
    result = scorer.explain("Alice arrived. Bob stayed.", "Alice arrived.")
    assert result.loss == 0  # deliberately fake judge, not a semantic assertion
    assert len(judge.observations()["requests"]) == 2
    assert scorer.instrument["configuration"]["judge"]["model_id"] == "jev-1.13.0"
    assert result.inspection["status"] == "not_inspected"


def test_thresholds_change_instrument_and_config_is_frozen():
    first = JevJudge()
    second = JevJudge(replace(first.config, accept_at_or_above=0.95))
    assert first.manifest() != second.manifest()
    with pytest.raises(AttributeError):
        first.config.max_requests = 999


def test_http_transport_fixed_endpoint_no_redirect_and_bounded_read(monkeypatch):
    calls = []

    class Response:
        status = 200
        reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read1(self, amount):
            calls.append(amount)
            self.reads += 1
            return b"ok" if self.reads == 1 else b""

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == ENDPOINT
            assert request.get_header("Authorization") == "Bearer test-value"
            assert timeout == 15
            return Response()

    def build(handler):
        assert isinstance(handler, _NoRedirect)
        assert handler.redirect_request(None, None, 302, "", {}, "https://example.com") is None
        return Opener()

    monkeypatch.setattr("urllib.request.build_opener", build)
    assert _http_transport("test-value", JevConfig(max_response_bytes=20))(b"{}") == b"ok"
    assert calls == [21, 19]


def test_large_text_is_rejected_before_serialization(monkeypatch):
    def forbidden(value):
        pytest.fail("oversized input reached serialization")

    judge = JevJudge(JevConfig(max_request_bytes=1000))
    monkeypatch.setattr("sum_engine_internal.research.meaning.jev_judge._json_bytes", forbidden)
    with pytest.raises(JevError, match="byte limit"):
        judge.entails("x" * 5000, "claim")
    with pytest.raises(JevError, match="byte limit"):
        judge.entails("premise", "x" * 5000)


def test_trickling_body_stops_at_elapsed_deadline(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read1(self, amount):
            return b"x"

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    ticks = iter([0.0, 0.01, 0.04, 0.08])
    monkeypatch.setattr("urllib.request.build_opener", lambda handler: Opener())
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    with pytest.raises(JevError):
        _http_transport("test-value", JevConfig(timeout_seconds=0.06))(b"{}")
