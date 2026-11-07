"""Generate ``tags_roundtrip_panel.json``, the static data behind the demo
page's "Tags and back" card.

It is a compact subset of one recorded run of ``scripts/tags_roundtrip.py``
on the worked example in ``fixtures/tags_roundtrip/efso_vaults_1/`` (the
operator's essay "EFSO Vaults #1: Wholly Mammon Discovered", used with the
author's agreement). Nothing is measured or recomputed here: every number is
copied from ``results.json`` as stored, and the page formats it for display.
The seed-7 rewrite of each condition is shown as recorded; the page calls no
model.

Run:  python single_file_demo/generate_tags_panel.py
      python single_file_demo/generate_tags_panel.py --check

No model, network or optional extra is needed. ``--check`` exits 1 when the
committed JSON differs from what this script would write.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURE = REPO / "fixtures" / "tags_roundtrip" / "efso_vaults_1"
RESULTS = FIXTURE / "results.json"
SOURCE = FIXTURE / "source.md"
OUT = HERE / "tags_roundtrip_panel.json"

SEED = 7

# Not stored in results.json; taken from the fixture's README.md (a test checks
# that the README still says each of these).
SOURCE_META = {
    "title": "EFSO Vaults #1: Wholly Mammon Discovered",
    "url": "https://ototao.substack.com/p/efso-vaults-1",
    "published": "2024-04-30",
    "credit": "by the project's operator, used here with the author's agreement",
}


def _tag_row(t: dict) -> dict:
    return {
        "tag": t["tag"],
        "school": t["school"],
        "kind": t["kind"],
        "rank": t["rank"],
        "evidence_found": t["evidence_found"],
    }


def _rung_tag(t) -> dict:
    # Topic-only rungs store plain strings; mixed rungs store {school, tag}.
    if isinstance(t, str):
        return {"school": "topic", "tag": t}
    return {"school": t["school"], "tag": t["tag"]}


def _summary(s: dict) -> dict:
    return {
        judge: {
            side: {k: s[judge][side][k] for k in ("mean", "min", "max")}
            for side in ("recall", "fidelity")
        }
        for judge in ("gist", "exact")
    }


def _judge_scores(sc: dict) -> dict:
    keys = (
        "recall", "fidelity", "source_claims", "recovered_claims",
        "rewrite_claims", "not_recovered", "not_supported_by_original",
    )
    return {k: sc[k] for k in keys}


def build(results_bytes: bytes | None = None, source_text: str | None = None) -> dict:
    if results_bytes is None:
        results_bytes = RESULTS.read_bytes()
    if source_text is None:
        source_text = SOURCE.read_text("utf-8")
    res = json.loads(results_bytes)
    if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != res["input"]["sha256"]:
        raise ValueError("source.md does not match the sha256 recorded in results.json")

    rows = []
    for rung in res["rungs"]:
        sample = next(s for s in rung["samples"] if s["seed"] == SEED)
        rows.append(
            {
                "condition": rung["condition"],
                "tag_count": rung["tag_count"],
                "tag_words": rung["tag_words"],
                "tags": [_rung_tag(t) for t in rung["tags"]],
                "samples": len(rung["samples"]),
                "summary": _summary(rung["summary"]),
                "shown_sample": {
                    "seed": sample["seed"],
                    "rewrite": sample["rewrite"],
                    "rewrite_words": sample["rewrite_words"],
                    "scores": {
                        judge: _judge_scores(sample["scores"][judge])
                        for judge in ("gist", "exact")
                    },
                },
            }
        )

    gist, exact = res["judges"]["gist"], res["judges"]["exact"]
    return {
        "generated_by": "single_file_demo/generate_tags_panel.py",
        "generated_from": {
            "path": "fixtures/tags_roundtrip/efso_vaults_1/results.json",
            "sha256": hashlib.sha256(results_bytes).hexdigest(),
        },
        "method": res["method"],
        "scope": res["scope"],
        "model": res["model"],
        "sampling": res["sampling"],
        "judges": {
            "gist": {
                "model_id": gist["model_id"],
                "rule": gist["rule"],
                "threshold": float.fromhex(gist["threshold_float_hex"]),
            },
            "exact": {
                "model_id": exact["model_id"],
                "threshold": exact["threshold"],
                "window": exact["window"],
            },
        },
        "source": {
            **SOURCE_META,
            "file": "fixtures/tags_roundtrip/efso_vaults_1/source.md",
            "words": res["input"]["words"],
            "sha256": res["input"]["sha256"],
            "text": source_text,
        },
        "tags": {
            school: [_tag_row(t) for t in res["tags"][school]]
            for school in ("topic", "form", "fact")
        },
        "rows": rows,
    }


def serialize(data: dict) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str]) -> int:
    data = serialize(build())
    if "--check" in argv:
        if OUT.read_bytes() != data:
            print(f"{OUT.relative_to(REPO)} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(REPO)} is current")
        return 0
    OUT.write_bytes(data)
    print(f"wrote {OUT.relative_to(REPO)} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
