"""Regression checks for inspection, sampling scope and experimental binding."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sum_engine_internal.research.frontier import RenderFrontier
from sum_engine_internal.research.meaning.evidence import (
    build_evaluation_manifest, evidence_digest, verify_evaluation_pairs,
)
from sum_engine_internal.research.meaning.local_judge import (
    NLIJudge, embedding_entailment_scorer, nli_entailment_scorer,
)
from sum_engine_internal.research.meaning.meaning_loss import EntailmentScorer, LexicalCoverageScorer
from sum_engine_internal.research.meaning.conformal_meaning import certify_meaning_risk
from sum_engine_internal.research.meaning.receipt import build_payload


@pytest.mark.parametrize("source", ["", " \n\t", "Alpha one.", "Alpha one. Bravo two."])
@pytest.mark.parametrize("output", ["", " \n\t", "Alpha one.", "Invented assertion."])
@pytest.mark.parametrize("batch", [False, True])
def test_scalar_and_readout_share_edge_case_kernel(source, output, batch):
    entails = lambda premise, hypothesis: hypothesis in premise
    scorer = EntailmentScorer(entails, "substring", entails_batch=(
        lambda premise, hypotheses: [entails(premise, h) for h in hypotheses]
    ) if batch else None)
    assert scorer.loss(source, output) == scorer.explain(source, output).loss
    if not source.strip() and output.strip():
        result = scorer.explain(source, output)
        assert (result.loss, result.preservation, result.recall, result.fidelity) == (1, 0, 0, 0)
        assert result.unsupported_claims == (output,)


def test_empty_source_cannot_be_credited_by_permissive_judge():
    scorer = EntailmentScorer(lambda *_: True, "always-true")
    result = scorer.explain("", "Unsupported.")
    assert result.loss == 1 and result.unsupported_claims == ("Unsupported.",)


def test_short_batch_output_does_not_silently_credit_unjudged_claims():
    scorer = EntailmentScorer(lambda *_: True, "broken-batch", entails_batch=lambda *_: [])
    with pytest.raises(ValueError, match="one decision"):
        scorer.explain("Source.", "Output.")


def test_instrument_distinguishes_threshold_model_revision_weights_without_loading():
    nli = nli_entailment_scorer(threshold=0.1)
    variants = [nli, nli_entailment_scorer(threshold=0.9),
                nli_entailment_scorer(threshold=0.1, revision="other-revision"),
                replace(nli, w_recall=0.8, w_fidelity=0.2)]
    hashes = [evidence_digest(s.instrument) for s in variants]
    assert len(set(hashes)) == len(variants)
    a = embedding_entailment_scorer(model_id="model-A")
    b = embedding_entailment_scorer(model_id="model-B")
    assert evidence_digest(a.instrument) != evidence_digest(b.instrument)
    assert a.instrument["configuration"]["judge"]["revision_status"] == "mutable_unresolved"
    assert nli_entailment_scorer(revision="main").instrument["configuration"]["judge"]["revision_status"] == "symbolic_ref_unresolved"
    assert evidence_digest(LexicalCoverageScorer().instrument) != evidence_digest(
        LexicalCoverageScorer(w_drop=0.8, w_fab=0.2).instrument)


def test_nli_coverage_reports_unseen_tokens_without_a_model():
    judge = NLIJudge()
    judge._mdl = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=5))
    judge._tok = lambda premise, hypothesis, **_: {"input_ids": list(range(len((premise + " " + hypothesis).split()) + 2))}
    result = judge.inspect_pair("one two three four", "five six")
    assert result == {"truncated": True, "input_tokens": 8, "token_limit": 5, "uninspected_tokens": 3}
    scorer = EntailmentScorer(lambda *_: True, "stub", inspect_pair=judge.inspect_pair)
    readout = scorer.explain("one two three four", "five six")
    assert readout.inspection["status"] == "partial"
    assert {r["direction"] for r in readout.inspection["partial_judgments"]} == {"recall", "fidelity"}


def _payload(manifest=None, sampling=None):
    scorer = LexicalCoverageScorer()
    losses = [scorer.loss("Alpha", "Alpha"), scorer.loss("Beta", "Gamma")]
    guarantee = certify_meaning_risk(losses, scorer_name=scorer.name, scorer_version=scorer.version)
    return build_payload(guarantee=guarantee, losses=losses, corpus_id="test",
                         transform="caller supplied", loss_definition="lexical proxy",
                         evaluation_manifest=manifest, sampling_contract=sampling)


def test_default_issuance_is_descriptive_and_does_not_invent_sampling_provenance():
    payload = _payload()
    assert payload["statistical_scope"] == "descriptive_batch"
    assert payload["sampling_status"] == "not_supplied"
    assert payload["evaluation_evidence_status"] == "not_supplied"
    assert "Exchangeability alone is insufficient" in payload["disclosure"]


def test_independent_sampling_is_explicit_assertion_not_verification():
    contract = {"design": "independent_identically_distributed", "experimental_unit": "document",
                "target_population": "test population", "selection_procedure": "random sample",
                "source_clustering": "one document per independent source",
                "fixed_policy_before_sample": True, "used_for_tuning": False}
    payload = _payload(sampling=contract)
    assert payload["statistical_scope"] == "conditional_expected_proxy_loss"
    assert payload["sampling_status"] == "issuer_asserted_not_verified"
    contract["used_for_tuning"] = True
    with pytest.raises(ValueError, match="tuning"):
        _payload(sampling=contract)
    assert payload["sampling_contract"]["used_for_tuning"] is False


def test_evaluation_binds_order_exact_text_and_instrument_without_claiming_rederivation():
    pairs = [("Alpha", "Alpha"), ("Beta", "Gamma")]
    manifest = build_evaluation_manifest(pairs, LexicalCoverageScorer(), transform_configuration={"mode": "supplied"})
    payload = _payload(manifest)
    result = verify_evaluation_pairs(payload, pairs)
    assert result["source_output_binding"] == "verified"
    assert result["score_derivation"] == "not_rederived"
    for altered in (pairs[::-1], [("Alpha ", "Alpha"), pairs[1]]):
        with pytest.raises(ValueError, match="texts or order"):
            verify_evaluation_pairs(payload, altered)
    manifest["scorer_instrument"]["configuration"]["weights"]["drop"] = "changed"
    assert verify_evaluation_pairs(payload, pairs) == result  # detached snapshot
    with pytest.raises(ValueError, match="instrument commitment"):
        _payload(manifest)


def test_candidate_path_preserves_nonmonotone_order_without_claiming_optimality():
    frontier = RenderFrontier.from_renderings("Alpha beta", [
        ("first", {}, "Gamma"), ("second", {}, "Alpha beta")], LexicalCoverageScorer())
    result = frontier.as_dict()
    assert result["path_kind"] == "caller_ordered_candidates"
    assert [p["word_count"] for p in result["points"]] == [1, 2]
    assert result["points"][0]["meaning_loss"] > result["points"][1]["meaning_loss"]


@pytest.mark.parametrize("bad", [[], "iid", 1, True])
def test_invalid_evidence_containers_fail_cleanly(bad):
    with pytest.raises(ValueError, match="mapping"):
        _payload(sampling=bad)
    with pytest.raises(ValueError, match="mapping"):
        _payload(manifest=bad)


def test_new_evidence_receipt_signs_and_replays_in_lightweight_sdk():
    from joserfc.jwk import OKPKey
    from sum_engine_internal.research.meaning.receipt import sign_meaning_risk_receipt
    from sum_verify import verify

    pairs = [("Alpha", "Alpha"), ("Beta", "Gamma")]
    manifest = build_evaluation_manifest(pairs, LexicalCoverageScorer(), transform_configuration={"mode": "supplied"})
    payload = _payload(manifest)
    key = OKPKey.generate_key("Ed25519")
    private, public = key.as_dict(private=True), key.as_dict(private=False)
    for jwk in (private, public):
        jwk.update(kid="test-evaluation-evidence", alg="EdDSA", use="sig")
    envelope = sign_meaning_risk_receipt(payload, private_jwk=private, kid=private["kid"])
    verified = verify(envelope, {"keys": [public]}, losses=[0.0, 1.0])
    assert verified == payload
    assert verified["evaluation_evidence_status"] == "hash_linked_not_rederived"
    assert verify_evaluation_pairs(verified, pairs)["source_output_binding"] == "verified"


def test_nested_evaluation_payload_canonicalizes_identically_in_node():
    import json
    from pathlib import Path
    import shutil
    import subprocess
    from sum_engine_internal.infrastructure.jcs import canonicalize

    if shutil.which("node") is None:
        pytest.skip("Node unavailable")
    manifest = build_evaluation_manifest([("Alpha", "Alpha"), ("Beta", "Gamma")],
                                         LexicalCoverageScorer(), transform_configuration={"mode": "supplied"})
    payload = _payload(manifest)
    script = Path(__file__).resolve().parents[2] / "single_file_demo/jcs_cli.js"
    result = subprocess.run(["node", str(script)], input=json.dumps(payload),
                            capture_output=True, text=True, check=True)
    assert result.stdout.encode("utf-8") == canonicalize(payload)
