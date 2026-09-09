"""Hash-linked instrument and evaluation evidence, without model imports.

These commitments identify supplied evidence. They do not establish that a
sample is independent, that a model is accurate, or that a source is true.
All wire values are float-free; exact float configuration uses hex strings.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

from sum_engine_internal.infrastructure.jcs import canonicalize


def evidence_digest(value: Any) -> str:
    return "sha256-" + hashlib.sha256(canonicalize(value)).hexdigest()


def _text_hash(text: str) -> str:
    return "sha256-" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot(value: Any) -> Any:
    """Reject ambiguous/nonportable wire values and detach mutable input."""
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int and abs(value) <= 2**53 - 1:
        return value
    if isinstance(value, Mapping) and all(type(k) is str for k in value):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(v) for v in value]
    raise ValueError("evidence must contain float-free JSON values with safe integers")


def instrument_manifest(scorer: Any, configuration: Mapping[str, Any]) -> dict[str, Any]:
    packages = {}
    for package in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not_installed"
    # Source hashes work in a wheel as well as a git checkout; unlike a
    # mutable branch name they bind the actual implementation being used.
    implementation = {}
    for filename in ("meaning_loss.py", "local_judge.py", "evidence.py"):
        implementation[filename] = "sha256-" + hashlib.sha256(
            Path(__file__).with_name(filename).read_bytes()
        ).hexdigest()
    return _snapshot({
        "schema": "sum.scorer_instrument.v1",
        "name": scorer.name, "version": scorer.version,
        "configuration": configuration,
        "implementation": implementation,
        "environment": {"python": platform.python_version(),
                        "system": platform.system(), "machine": platform.machine(),
                        "packages": packages},
        "reproducibility": "configuration commitment; model numerical replay remains machine-dependent",
    })


def sampling_metadata(contract: Mapping[str, Any] | None) -> dict[str, Any]:
    if contract is None:
        return {"statistical_scope": "descriptive_batch",
                "sampling_status": "not_supplied"}
    if not isinstance(contract, Mapping):
        raise ValueError("sampling contract must be a mapping")
    snapshot = _snapshot(contract)
    for field in ("experimental_unit", "target_population", "selection_procedure", "source_clustering"):
        if not isinstance(snapshot.get(field), str) or not snapshot[field].strip():
            raise ValueError(f"sampling_contract.{field} must be a nonempty string")
    if snapshot.get("design") != "independent_identically_distributed":
        raise ValueError("confidence interpretation requires independent_identically_distributed design")
    if snapshot.get("fixed_policy_before_sample") is not True or snapshot.get("used_for_tuning") is not False:
        raise ValueError("confidence interpretation requires fixed policy and no calibration tuning reuse")
    return {"statistical_scope": "conditional_expected_proxy_loss",
            "sampling_status": "issuer_asserted_not_verified",
            "sampling_contract": snapshot,
            "sampling_contract_hash": evidence_digest(snapshot)}


def build_evaluation_manifest(
    pairs: Sequence[tuple[str, str]], scorer: Any, *,
    transform_configuration: Mapping[str, Any],
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind ordered, exact UTF-8 source/output pairs and the scoring instrument.

    Generation and selection metadata are supplied assertions. An unknown
    selection is recorded explicitly instead of inventing sampling provenance.
    """
    instrument = getattr(scorer, "instrument", None)
    if instrument is None:
        instrument = instrument_manifest(scorer, {"status": "caller_configuration_not_supplied"})
    records = []
    for source, output in pairs:
        if not isinstance(source, str) or not isinstance(output, str):
            raise ValueError("evaluation pairs must contain source/output strings")
        records.append({"source_hash": _text_hash(source), "output_hash": _text_hash(output)})
    return _snapshot({
        "schema": "sum.evaluation_manifest.v1", "n": len(records),
        "pairs": records, "pair_order_hash": evidence_digest(records),
        "scorer_instrument": instrument, "scorer_instrument_hash": evidence_digest(instrument),
        "transform_configuration": transform_configuration,
        "selection": selection if selection is not None else {"status": "not_supplied"},
        "verification_scope": "hash-linked evidence; score derivation and sampling assumptions are not verified",
    })


def evaluation_fields(manifest: Mapping[str, Any] | None, *, n: int, scorer: str,
                      scorer_version: str) -> dict[str, Any]:
    if manifest is None:
        return {"evaluation_evidence_status": "not_supplied"}
    if not isinstance(manifest, Mapping):
        raise ValueError("evaluation manifest must be a mapping")
    value = _snapshot(manifest)
    instrument = value.get("scorer_instrument", {})
    if value.get("schema") != "sum.evaluation_manifest.v1" or type(value.get("n")) is not int or value.get("n") != n:
        raise ValueError("evaluation manifest schema/sample count does not match receipt")
    if not isinstance(instrument, dict) or instrument.get("schema") != "sum.scorer_instrument.v1":
        raise ValueError("evaluation scorer instrument schema is invalid")
    if instrument.get("name") != scorer or instrument.get("version") != scorer_version:
        raise ValueError("evaluation instrument identity does not match receipt")
    records = value.get("pairs")
    if not isinstance(records, list) or len(records) != n or value.get("pair_order_hash") != evidence_digest(records):
        raise ValueError("evaluation pair order commitment does not match")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"source_hash", "output_hash"}:
            raise ValueError("evaluation pair must declare source_hash and output_hash")
        for digest in record.values():
            if not isinstance(digest, str) or not digest.startswith("sha256-") or len(digest) != 71 or any(c not in "0123456789abcdef" for c in digest[7:]):
                raise ValueError("evaluation text commitment must be a sha256 digest")
    if value.get("scorer_instrument_hash") != evidence_digest(instrument):
        raise ValueError("evaluation scorer instrument commitment does not match")
    return {"evaluation_evidence_status": "hash_linked_not_rederived",
            "evaluation_manifest": value, "evaluation_manifest_hash": evidence_digest(value),
            "scorer_instrument_hash": evidence_digest(instrument)}


def verify_evaluation_pairs(payload: Mapping[str, Any], pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Check supplied text binding, separately from signature/arithmetic replay.

    Call only after authenticating the receipt under the caller's trust policy.
    This check deliberately does not invoke a judge or claim score re-derivation.
    """
    manifest = payload.get("evaluation_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("receipt has no evaluation manifest")
    if evidence_digest(manifest) != payload.get("evaluation_manifest_hash"):
        raise ValueError("evaluation manifest commitment mismatch")
    evaluation_fields(manifest, n=payload["n"], scorer=payload["scorer"],
                      scorer_version=payload["scorer_version"])
    records = [{"source_hash": _text_hash(src), "output_hash": _text_hash(out)} for src, out in pairs]
    if records != manifest["pairs"]:
        raise ValueError("supplied source/output texts or order do not match evaluation evidence")
    return {"source_output_binding": "verified", "score_derivation": "not_rederived",
            "sampling_assumptions": "not_verified"}
