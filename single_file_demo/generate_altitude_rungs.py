"""Generate ``altitude_rungs.json`` — the static data behind the demo page's
T0 altitude panel (the distiller made visible).

One REAL document from the witnessed certified chain's corpus
(``billsum-test-12``, BillSum test split, CC0-1.0; one of the 32 bills bound
by chain ``9a8ab39f08522c50`` in ``fixtures/chain_receipts_billsum/``),
descending the altitude ladder:

    rung 0  the bill itself (source)
    rung 1  the dataset's OWN reference summary (SUM did not perform it;
            same honest framing as chain hop 1)
    rung 2  deterministic lead-N extractive compression, keep 0.5
            (the same ``lead_extractive`` function as chain hop 2, imported
            from the committed fixture generator so semantics are identical)
    rung 3  deterministic lead-N extractive compression, keep 0.25

Per rung the panel shows the text, the measured NLI meaning-loss vs the
source, and the kept / dropped / added claim readout — the exact output of
``RenderFrontier.depth_diff`` (what ``sum depth-diff`` prints), serialized
once, offline, into a static JSON the Worker ships as an asset.

HONESTY (also carried inside the JSON's ``scope`` field): every number here
is a per-document MEASUREMENT under the named NLI judge, not a guarantee.
The (1-delta) corpus-level bounds live in the signed, witnessed chain
receipt the panel links to. The loss computation is machine-pinned (NLI
float drift, F23/F26): regenerating this file on a different stack may move
the numbers slightly; the committed JSON is the artifact the page shows.

Run:  python single_file_demo/generate_altitude_rungs.py
Needs the [research] + [judge] extras (torch + transformers).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
CORPUS_FILE = (
    REPO / "fixtures" / "meaning_receipts_billsum" / "corpus_billsum_test_first64.json"
)
CHAIN_GEN = (
    REPO / "fixtures" / "chain_receipts_billsum" / "generate_a2_chain_fixture.py"
)
OUT = HERE / "altitude_rungs.json"

DOC_ID = "billsum-test-12"
CHAIN_ID = "9a8ab39f08522c50"

SCOPE = (
    "per-document MEASUREMENT under the named NLI judge; not a guarantee. "
    "The proxy is blind to arrangement, sound, connotation, implicature. "
    "Corpus-level (1-delta) bounds live in signed receipts: this document is "
    "one of the 32 bills bound by the witnessed certified chain."
)


def _load_chain_gen():
    """Import the committed chain-fixture generator so rung 2/3 use the
    byte-identical ``lead_extractive`` the witnessed chain's hop 2 used."""
    spec = importlib.util.spec_from_file_location("_chain_gen", CHAIN_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build() -> dict:
    from sum_engine_internal.research.frontier import RenderFrontier
    from sum_engine_internal.research.meaning.local_judge import (
        nli_entailment_scorer,
    )

    chain_gen = _load_chain_gen()
    corpus = json.loads(CORPUS_FILE.read_text("utf-8"))
    pair = next(p for p in corpus["pairs"] if p["id"] == DOC_ID)
    bill, summary = pair["source"], pair["rendering"]

    renderings = [
        ("reference summary", {}, summary),
        ("extract 1/2", {}, chain_gen.lead_extractive(summary, 0.5)),
        ("extract 1/4", {}, chain_gen.lead_extractive(summary, 0.25)),
    ]

    scorer = nli_entailment_scorer()
    frontier = RenderFrontier.from_renderings(bill, renderings, scorer)
    rungs = frontier.depth_diff(scorer)

    rung_notes = {
        "reference summary": (
            "the dataset's own reference summary (BillSum); SUM did not "
            "perform this transform, it measures it"
        ),
        "extract 1/2": (
            "deterministic lead-N extractive compression of the summary "
            "(keep the first half of its sentences); offline, 0 LLM calls; "
            "the same transform as the witnessed chain's hop 2"
        ),
        "extract 1/4": (
            "deterministic lead-N extractive compression of the summary "
            "(keep the first quarter of its sentences); offline, 0 LLM calls"
        ),
    }
    texts = {label: text for (label, _p, text) in renderings}

    out_rungs = [
        {
            "label": "the bill (source)",
            "note": "the full text of the bill; every claim below is judged against it",
            "text": bill,
            "words": len(bill.split()),
            "meaning_loss": None,
            "compression_pct": 0,
        }
    ]
    for r in rungs:
        d = r.as_dict()
        out_rungs.append(
            {
                "label": d["label"],
                "note": rung_notes[d["label"]],
                "text": texts[d["label"]],
                "words": len(texts[d["label"]].split()),
                "meaning_loss": d["meaning_loss"],
                "compression_pct": round((1.0 - d["compression_ratio"]) * 100),
                "recall": d["recall"],
                "fidelity": d["fidelity"],
                "source_claims": d["source_claims"],
                "preserved_claims": d["preserved_claims"],
                "dropped_claims": d["dropped_claims"],
                "added_claims": d["added_claims"],
                "loss_per_compression": d["loss_per_compression"],
            }
        )

    return {
        "generated_by": "single_file_demo/generate_altitude_rungs.py",
        "scope": SCOPE,
        "scorer": frontier.scorer_name,
        "scorer_version": frontier.scorer_version,
        "document": {
            "id": DOC_ID,
            "corpus": "FiscalNote/billsum test split (CC0-1.0, US-government works)",
            "committed_at": "fixtures/meaning_receipts_billsum/corpus_billsum_test_first64.json",
        },
        "chain_receipt": {
            "chain_id": CHAIN_ID,
            "path": "fixtures/chain_receipts_billsum/chain_receipt.billsum.golden.json",
            "witnessed_in": "transparency/log.jsonl",
            "url": "https://github.com/OtotaO/SUM/tree/main/fixtures/chain_receipts_billsum",
            "note": (
                "signed sum.chain_receipt.v1 over all 32 bills: hop-1 bound "
                "0.865768, hop-2 bound 0.488860, Bonferroni budget 1.354628 "
                "at joint confidence 0.90, direct end-to-end 0.874216 (95% "
                "per hop, Hoeffding, n=32)"
            ),
        },
        "rungs": out_rungs,
    }


def main() -> None:
    data = build()
    OUT.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT.relative_to(REPO)}")
    for r in data["rungs"]:
        loss = "  --  " if r["meaning_loss"] is None else f"{r['meaning_loss']:.3f}"
        kda = (
            ""
            if "preserved_claims" not in r
            else (
                f"  kept {r['preserved_claims']}/{r['source_claims']}"
                f"  dropped {len(r['dropped_claims'])}"
                f"  added {len(r['added_claims'])}"
            )
        )
        print(
            f"  {r['label']:<22} {r['words']:>5}w  compress {r['compression_pct']:>3}%"
            f"  loss {loss}{kda}"
        )


if __name__ == "__main__":
    main()
