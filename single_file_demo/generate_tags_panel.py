"""Generate ``tags_roundtrip_panel.json``, the static data behind the demo
page's "Tags and back" card.

It is a compact subset of the first recorded run of ``scripts/tags_roundtrip.py``
on each worked example in ``fixtures/tags_roundtrip/`` (the operator's essays
"EFSO Vaults #1" and "EFSO Vaults #2", each used with the author's agreement).
Nothing is measured or recomputed here: every number is copied from a stored
``results.json``, and the page formats it for display. Where a run also scored
the cut (``--cut``), the cut summary and the seed-7 cut are copied too. The
"tags alone" figure for the 32 topic tags is copied from the equal-budget run
in ``fixtures/tags_roundtrip/equal_budget/``, whose topic notes at the largest
budget are the same 32 tags (checked here). The seed-7 rewrite of each
condition is shown as recorded; the page calls no model.

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
TAGS_DIR = REPO / "fixtures" / "tags_roundtrip"
OUT = HERE / "tags_roundtrip_panel.json"

SEED = 7

# Not stored in results.json; taken from each fixture's README.md (a test checks
# that the README still says each of these). "notes" are lines shown for that
# text only.
TEXTS = (
    {
        "id": "efso_vaults_1",
        "alone_from": "topic@80",
        "meta": {
            "title": "EFSO Vaults #1: Wholly Mammon Discovered",
            "url": "https://ototao.substack.com/p/efso-vaults-1",
            "published": "2024-04-30",
            "credit": "by the project's operator, used here with the author's agreement",
            "notes": [
                "The judge settings and the cut rule were chosen on this text, so its numbers do not test them.",
                "A later rerun with the same tags and seeds put topic above mixed on exact recall at 32 tags, "
                "the reverse of this run, and a run that dropped the \"Follow\" instruction barely moved the "
                "first-person share; whether the mixed tags, the labels or the missing \"plain prose\" change "
                "the voice, these runs cannot say.",
                "This run did not score the cut.",
            ],
        },
    },
    {
        "id": "efso_vaults_2",
        "alone_from": "topic@60",
        "meta": {
            "title": "EFSO Vaults #2: Oh the Humanitas",
            "url": "https://ototao.substack.com/p/efso-vaults-2",
            "published": "2024-05-10",
            "credit": "by the project's operator, who sent it for this use",
            "notes": [
                "This text played no part in choosing the judge settings or the cut rule.",
            ],
        },
    },
)


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


def _cut(rung: dict, sample: dict) -> dict | None:
    if "cut_summary" not in rung:
        return None
    cs, c = rung["cut_summary"], sample["cut"]
    return {
        "summary": {
            "kept_sentences_mean": cs["kept_sentences_mean"],
            "nonempty": cs["nonempty"],
            **{
                judge: {side: {k: v[k] for k in ("mean", "min", "max")} for side, v in cs[judge].items()}
                for judge in ("gist", "exact")
            },
        },
        "shown": {
            "text": c["text"],
            "kept_sentences": c["kept_sentences"],
            "rewrite_sentences": c["rewrite_sentences"],
            "scores": None if c["scores"] is None else {
                judge: {k: c["scores"][judge][k] for k in ("recall", "fidelity")}
                for judge in ("gist", "exact")
            },
        },
    }


def _alone(text_id: str, condition: str, rows: list[dict]) -> dict:
    path = TAGS_DIR / "equal_budget" / f"results_{text_id}.json"
    raw = path.read_bytes()
    rung = next(r for r in json.loads(raw)["rungs"] if r["condition"] == condition)
    row = next(r for r in rows if r["condition"] == "topic" and r["tag_count"] == 32)
    if [t["tag"] for t in row["tags"]] != rung["tags"]:
        raise ValueError(f"{condition} notes in {path.name} are not the card's 32 topic tags")
    return {
        "from": {"path": str(path.relative_to(REPO)), "sha256": hashlib.sha256(raw).hexdigest(),
                 "condition": condition},
        "row": {"condition": "topic", "tag_count": 32},
        "gist_recall": rung["notes_only_scores"]["gist"]["recall"],
        "exact_recall": rung["notes_only_scores"]["exact"]["recall"],
    }


def build_text(spec: dict) -> dict:
    fixture = TAGS_DIR / spec["id"]
    results_bytes = (fixture / "results.json").read_bytes()
    source_text = (fixture / "source.md").read_text("utf-8")
    res = json.loads(results_bytes)
    if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != res["input"]["sha256"]:
        raise ValueError(f"{spec['id']}/source.md does not match the sha256 recorded in results.json")

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
                "cut": _cut(rung, sample),
            }
        )

    gist, exact = res["judges"]["gist"], res["judges"]["exact"]
    meta = spec["meta"]
    return {
        "id": spec["id"],
        "generated_from": {
            "path": f"fixtures/tags_roundtrip/{spec['id']}/results.json",
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
        "cut": None if "cut" not in res else {
            "rule": res["cut"]["rule"],
            "threshold": res["cut"]["threshold"],
            "empty_cut": res["cut"]["empty_cut"],
            "note": res.get("cut_note", ""),
        },
        "notes": meta["notes"],
        "source": {
            **{k: meta[k] for k in ("title", "url", "published", "credit")},
            "file": f"fixtures/tags_roundtrip/{spec['id']}/source.md",
            "words": res["input"]["words"],
            "sha256": res["input"]["sha256"],
            "text": source_text,
        },
        "tags": {
            school: [_tag_row(t) for t in res["tags"][school]]
            for school in ("topic", "form", "fact")
        },
        "rows": rows,
        "tags_alone": _alone(spec["id"], spec["alone_from"], rows),
    }


def build() -> dict:
    return {
        "generated_by": "single_file_demo/generate_tags_panel.py",
        "texts": [build_text(spec) for spec in TEXTS],
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
