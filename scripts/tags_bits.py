"""Tags in bits: how much a fixed language model's surprise at a text drops
when it first reads a list of notes about that text.

    PYTHONPATH=. python scripts/tags_bits.py TEXT_ID [--out FILE.json]
        [--device cpu] [--repeat 2]

TEXT_ID names a folder under fixtures/tags_roundtrip/ (efso_vaults_1 or
efso_vaults_2). Nothing is generated: every note list is read from files
already committed, so the only model is the scorer.

For each sentence u of the text (the same sentence split the round-trip
judges use), the scorer reads

    Here are notes about a piece of writing:
    - note 1
    - note 2
    ...

    A sentence from that writing:
    u

and the script sums -log2 p over the tokens of u alone. "(none)" stands in
for the notes in the null context. The gain of a note list is the null's
bits minus the list's bits, summed over sentences; its share is that gain
over the null's total bits. Sentences are scored one at a time, with no
earlier sentence in the context.

A gain is information about the wording of the text, measured by one model.
It is not recall of the text's claims: the negated-tag control below is what
tells how much of a gain survives when the notes say the opposite.

Note lists scored, all at about the same word budget:
  - every rung of fixtures/tags_roundtrip/equal_budget/results_<TEXT_ID>.json
    (topic, fact and mixed tags, spread and lead extracts at 40, 60 and 80
    words), bound by sha256;
  - the same kinds of notes recorded for the other EFSO text (same author and
    series, so not unrelated);
  - from fixtures/tags_bits/controls.json: the fact@60 tags negated with every
    content word kept, the fact@60 tags with one detail swapped, and an
    unrelated list;
  - a filler list of the same length as fact@60 whose notes say nothing;
  - self: each sentence given as the only note before itself (the most one
    note can give).

The whole measurement runs --repeat times and the largest difference between
runs is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

METHOD_ID = "sum.tags_bits.v1"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
ROOT = Path(__file__).resolve().parent.parent
TEXTS = ("efso_vaults_1", "efso_vaults_2")
HEADER = "Here are notes about a piece of writing:\n"
FOOTER = "\n\nA sentence from that writing:\n"
NULL_NOTES = "(none)"
FILLER_NOTE = "a note"
SISTER_KINDS = ("topic@60", "fact@60", "mixed@60")
FACT_RUNG = "fact@60"


def render(notes: Sequence[str]) -> str:
    body = "\n".join(f"- {n}" for n in notes) if notes else NULL_NOTES
    return HEADER + body + FOOTER


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LM:
    """A pinned causal LM that returns the bits of a target after a context."""

    def __init__(self, device: str = "cpu", model_id: str = MODEL_ID,
                 revision: str = MODEL_REVISION) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model_id, self.revision = model_id, revision
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, use_safetensors=True, torch_dtype=torch.float32
        ).to(device).eval()

    def bits(self, context: str, target: str) -> float:
        torch = self.torch
        ctx = self.tok(context, add_special_tokens=False)["input_ids"]
        tgt = self.tok(target, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([ctx + tgt], device=self.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0].double()
        logp = torch.log_softmax(logits[len(ctx) - 1:-1], dim=-1)
        picked = logp[torch.arange(len(tgt)), torch.tensor(tgt, device=self.device)]
        return float(-picked.sum().item() / math.log(2))

    def manifest(self) -> dict[str, Any]:
        import transformers

        return {"model_id": self.model_id, "model_revision": self.revision,
                "weights": "safetensors", "dtype": "float32", "device": self.device,
                "torch": self.torch.__version__, "transformers": transformers.__version__}


def gains(bits: Callable[[str, str], float], units: Sequence[str],
          notes: Sequence[str], null_bits: Sequence[float]) -> list[float]:
    ctx = render(notes)
    return [nb - bits(ctx, u) for u, nb in zip(units, null_bits)]


def summary(per_unit: Sequence[float], null_total: float) -> dict[str, Any]:
    total = sum(per_unit)
    return {"gain_bits": total, "share": total / null_total if null_total else 0.0,
            "units_gaining": sum(1 for g in per_unit if g > 0)}


def load_inputs(text_id: str) -> dict[str, Any]:
    other = [t for t in TEXTS if t != text_id][0]
    src_path = ROOT / "fixtures/tags_roundtrip" / text_id / "source.md"
    res_path = ROOT / "fixtures/tags_roundtrip/equal_budget" / f"results_{text_id}.json"
    oth_path = ROOT / "fixtures/tags_roundtrip/equal_budget" / f"results_{other}.json"
    ctl_path = ROOT / "fixtures/tags_bits/controls.json"
    results = json.loads(res_path.read_text(encoding="utf-8"))
    source = src_path.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != results["input"]["sha256"]:
        raise SystemExit(f"source.md does not match the sha256 recorded in {res_path.name}")
    return {
        "source": source, "results": results, "other_id": other,
        "other": json.loads(oth_path.read_text(encoding="utf-8")),
        "controls": json.loads(ctl_path.read_text(encoding="utf-8"))["texts"][text_id],
        "files": {p.relative_to(ROOT).as_posix(): sha256_file(p)
                  for p in (src_path, res_path, oth_path, ctl_path)},
    }


def conditions(inp: dict[str, Any]) -> list[dict[str, Any]]:
    rungs = {r["condition"]: r for r in inp["results"]["rungs"] if r["condition"] != "control"}
    other = {r["condition"]: r for r in inp["other"]["rungs"]}
    fact = rungs[FACT_RUNG]["tags"]
    out = [{"condition": c, "group": "recorded", "notes": list(r["tags"])} for c, r in rungs.items()]
    out += [{"condition": f"other_text:{c}", "group": "same_series", "notes": list(other[c]["tags"])}
            for c in SISTER_KINDS if c in other]
    out += [{"condition": f"negated:{FACT_RUNG}", "group": "control", "notes": inp["controls"]["negated"]},
            {"condition": f"swapped:{FACT_RUNG}", "group": "control", "notes": inp["controls"]["swapped"]},
            {"condition": "unrelated", "group": "control", "notes": inp["controls"]["unrelated"]},
            {"condition": "filler", "group": "control", "notes": [FILLER_NOTE] * len(fact)}]
    for c in out:
        c["note_count"] = len(c["notes"])
        c["note_words"] = sum(len(n.split()) for n in c["notes"])
    return out


def measure(bits: Callable[[str, str], float], units: Sequence[str],
            conds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    null_ctx = render([])
    null_bits = [bits(null_ctx, u) for u in units]
    null_total = sum(null_bits)
    rows = []
    for c in conds:
        g = gains(bits, units, c["notes"], null_bits)
        rows.append({**{k: c[k] for k in ("condition", "group", "note_count", "note_words", "notes")},
                     **summary(g, null_total), "per_unit_gain_bits": g})
    self_gain = [nb - bits(render([u]), u) for u, nb in zip(units, null_bits)]
    self_ratio = [g / nb for g, nb in zip(self_gain, null_bits) if nb > 0]
    return {"null_bits": null_bits, "null_total_bits": null_total, "rows": rows,
            "self": {"gain_bits": sum(self_gain), "share": sum(self_gain) / null_total,
                     "mean_unit_ratio": sum(self_ratio) / len(self_ratio),
                     "min_unit_ratio": min(self_ratio), "per_unit_gain_bits": self_gain}}


def checks(m: dict[str, Any]) -> dict[str, Any]:
    by = {r["condition"]: r for r in m["rows"]}
    true_gain = by[FACT_RUNG]["gain_bits"]

    def ratio(c: str) -> float | None:
        return by[c]["gain_bits"] / true_gain if true_gain else None

    return {
        "fact_gain_bits": true_gain,
        "negated_over_true": ratio(f"negated:{FACT_RUNG}"),
        "swapped_over_true": ratio(f"swapped:{FACT_RUNG}"),
        "unrelated_share": by["unrelated"]["share"],
        "filler_share": by["filler"]["share"],
        "self_mean_unit_ratio": m["self"]["mean_unit_ratio"],
    }


def max_run_difference(a: dict[str, Any], b: dict[str, Any]) -> float:
    diffs = [abs(x - y) for x, y in zip(a["null_bits"], b["null_bits"])]
    for ra, rb in zip(a["rows"], b["rows"]):
        diffs += [abs(x - y) for x, y in zip(ra["per_unit_gain_bits"], rb["per_unit_gain_bits"])]
    return max(diffs) if diffs else 0.0


def run(text_id: str, bits: Callable[[str, str], float], repeat: int = 2) -> dict[str, Any]:
    from sum_engine_internal.research.meaning.meaning_loss import _sentences

    inp = load_inputs(text_id)
    units = _sentences(inp["source"])
    conds = conditions(inp)
    runs = [measure(bits, units, conds) for _ in range(max(1, repeat))]
    m = runs[0]
    return {
        "method": METHOD_ID, "text": text_id, "inputs": inp["files"],
        "context_template": {"header": HEADER, "note_prefix": "- ", "null_notes": NULL_NOTES,
                             "footer": FOOTER, "filler_note": FILLER_NOTE},
        "unit": "source sentences, split as the round-trip judges split them; one at a time, no earlier sentence in context",
        "measures": "information about the wording of the text for one model; not recall of claims",
        "units": units, "repeats": len(runs),
        "max_difference_between_runs_bits": max(max_run_difference(runs[0], r) for r in runs),
        **m, "checks": checks(m),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("text", choices=TEXTS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args(argv)
    lm = LM(device=a.device)
    out = {"scorer": lm.manifest(), **run(a.text, lm.bits, a.repeat)}
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    c = out["checks"]
    print(json.dumps({"null_total_bits": round(out["null_total_bits"], 1),
                      "max_difference_between_runs_bits": out["max_difference_between_runs_bits"],
                      **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()}}, indent=1))
    for r in out["rows"]:
        print(f"{r['condition']:24} {r['group']:11} words {r['note_words']:3}  gain {r['gain_bits']:8.1f}  share {r['share']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
