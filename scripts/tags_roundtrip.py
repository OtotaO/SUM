"""Tags and back: turn a text into ordered tags, write the text again from the
tags alone, and measure what came back.

    PYTHONPATH=. python scripts/tags_roundtrip.py INPUT [--out FILE.json]
        [--model gpt-4o-mini-2024-07-18] [--counts 0,4,8,16,32] [--samples 3]

INPUT is a text or Markdown file, or any format the omni-format pivot converts
(PDF, DOCX, HTML, EPUB, ...; needs the markitdown extra).

Steps, all recorded in the output JSON:
  1. One model call proposes up to 32 tags. Each tag is 1 to 4 words, marked
     "mentioned" (the text names it) or "implied" (a reader would file the
     text under it), with an importance rank and the shortest verbatim quote
     it rests on. Quotes are checked against the text; a quote that is not
     found is recorded as not found. Tags are kept in replication order: the
     order their evidence appears in the text.
  2. For each tag count k, the k highest-ranked tags (in replication order,
     tag words only, no quotes) go to a second call that writes the text again
     at the original length, several times (--samples, default 3, one seed
     each), because the model's output varies between runs; the mean and
     range are reported. k = 0 is a control: the model writes a text of that
     length from nothing, which shows how much credit a judge gives unrelated
     prose.
  3. Two local judges score each rewrite against the original in both
     directions: recall (share of the original's sentences the rewrite
     carries) and fidelity (share of the rewrite's sentences the original
     carries).
     - gist: a sentence is carried if some sentence on the other side has
       embedding cosine similarity of at least 0.5 (all-MiniLM-L6-v2).
     - exact: a sentence is carried if an NLI model (DeBERTa-v3-large,
       mnli-fever-anli-ling-wanli) gives entailment probability of at least
       0.9 from some window of the other side. A window is a run of whole
       sentences of at most 80 tokens; a longer sentence is split into
       overlapping pieces of at most 80 tokens. On longer windows the model
       credited unrelated text. Support spread across windows is missed.
     These settings were chosen on one development text (the k = 0 control
     there scored at or near zero under both). They are frozen here; results
     on other texts are not an evaluation of the settings.

Every number is a measurement on this one text under the named judge and
model, not a guarantee. The model may know a famous text from training, which
would inflate recall; use texts it has not seen. Nothing is signed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

METHOD_ID = "sum.tags_roundtrip.v0"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_COUNTS = (0, 4, 8, 16, 32)
MAX_TAGS = 32
MAX_TAG_WORDS = 4
MAX_EVIDENCE_WORDS = 12
GIST_THRESHOLD = 0.5
EXACT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EXACT_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
EXACT_THRESHOLD = 0.9
EXACT_WINDOW_TOKENS = 80
SCOPE = (
    "A measurement on this one text under the named judges and model, not a "
    "guarantee. Recall: share of the original's sentences the rewrite carries. "
    "Fidelity: share of the rewrite's sentences the original carries. Gist "
    "counts a sentence as carried when some sentence on the other side is "
    "similar in embedding space; exact counts it only when an NLI model finds "
    "it entailed by one short window of the other side, so support spread "
    "across a text is missed. The k = 0 row is a control written from no tags. "
    "Judge settings were chosen on one development text and then frozen. A "
    "model that has seen a text in training can rebuild more of it than its "
    "tags carry. Nothing here is signed."
)

TAG_SYSTEM = (
    "You make tags for a text. A tag is a short label of 1 to 4 words. Return "
    'JSON {"tags": [...]} with at most 32 tags. Each tag is an object '
    '{"tag": string, "kind": "mentioned" or "implied", "rank": integer, '
    '"evidence": string}. "mentioned": the text names this topic. "implied": '
    "the text does not name it, but a careful reader would file the text under "
    "it. rank: importance, 1 = most important, each rank used once. evidence: "
    "the shortest exact quote from the text, at most 12 words, copied "
    "character for character, that the tag rests on. Cover what the text names "
    "and what it implies, and list the tags in the order their evidence first "
    "appears in the text."
)
REGEN_SYSTEM = (
    "You are given tags taken from a text, in the order their topics appear "
    "in it. Write the text they came from, about {words} words, as plain "
    "prose. Use every tag. Do not add names, numbers, dates or quotations "
    "that the tags do not contain. Return only the text."
)
CONTROL_SYSTEM = "Write a text of about {words} words. Return only the text."


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def validate_tags(raw: Any, source: str) -> list[dict[str, Any]]:
    """Normalise the model's tag list. Drops entries that are not objects or
    whose tag is empty or longer than MAX_TAG_WORDS words; keeps at most
    MAX_TAGS. Locates each evidence quote in the source (whitespace, case and
    curly quotes normalised) and records its offset in that normalised text,
    used only for ordering, or None if not found. A quote longer than
    MAX_EVIDENCE_WORDS is still located and is flagged as over the limit.
    Ranks are re-assigned 1..n in the model's rank order (ties and missing
    ranks fall back to list order). Returns tags in replication order: found
    evidence by position, then tags whose evidence was not found, in list
    order."""
    items = raw.get("tags") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("tag response has no list of tags")
    norm_source = _norm(source)
    tags: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        tag = item.get("tag")
        if not isinstance(tag, str) or not tag.strip():
            continue
        tag = re.sub(r"\s+", " ", tag).strip()
        if len(tag.split()) > MAX_TAG_WORDS or _norm(tag) in seen:
            continue
        seen.add(_norm(tag))
        evidence = item.get("evidence") if isinstance(item.get("evidence"), str) else ""
        evidence = re.sub(r"\s+", " ", evidence).strip()
        pos = None
        if evidence:
            found = norm_source.find(_norm(evidence))
            pos = found if found >= 0 else None
        rank = item.get("rank")
        tags.append({
            "tag": tag,
            "kind": item.get("kind") if item.get("kind") in ("mentioned", "implied") else "unlabelled",
            "evidence": evidence,
            "evidence_found": pos is not None,
            "evidence_over_word_limit": len(evidence.split()) > MAX_EVIDENCE_WORDS,
            "evidence_position": pos,
            "_model_rank": rank if isinstance(rank, int) and not isinstance(rank, bool) else None,
            "_list_index": i,
        })
        if len(tags) == MAX_TAGS:
            break
    by_rank = sorted(tags, key=lambda t: (t["_model_rank"] is None, t["_model_rank"] or 0, t["_list_index"]))
    for r, t in enumerate(by_rank, 1):
        t["rank"] = r
    ordered = sorted(tags, key=lambda t: (t["evidence_position"] is None,
                                          t["evidence_position"] or 0, t["_list_index"]))
    for t in ordered:
        del t["_model_rank"], t["_list_index"]
    return ordered


def tags_at(tags: Sequence[dict[str, Any]], k: int) -> list[str]:
    """The k highest-ranked tags, in replication order."""
    return [t["tag"] for t in tags if t["rank"] <= k]


def windows(sentences: Sequence[str], lengths: Sequence[int], budget: int) -> list[tuple[str, bool]]:
    """Consecutive whole sentences packed into windows of at most ``budget``
    tokens (each sentence costs its length plus one). Consecutive windows
    share one sentence whenever a window holds two or more. A sentence longer
    than the budget is a window of its own, marked True; the caller must split
    it (WindowedNLI splits such sentences into token pieces first)."""
    out: list[tuple[str, bool]] = []
    i, n = 0, len(sentences)
    while i < n:
        j, total = i, 0
        while j < n and total + lengths[j] + 1 <= budget:
            total += lengths[j] + 1
            j += 1
        if j == i:
            out.append((sentences[i], True))
            i += 1
            continue
        out.append((" ".join(sentences[i:j]), False))
        if j >= n:
            break
        i = j - 1 if j - 1 > i else j
    return out


class WindowedNLI:
    """Batch NLI over short premise windows: ``entails_batch(premise,
    hypotheses)`` is, per hypothesis, whether some window of at most
    ``window_tokens`` tokens of the premise entails it with probability at
    least ``threshold``."""

    def __init__(self, model_id: str = EXACT_MODEL, revision: str = EXACT_REVISION,
                 threshold: float = EXACT_THRESHOLD, window_tokens: int = EXACT_WINDOW_TOKENS) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id, self.revision = model_id, revision
        self.threshold, self.window_tokens = threshold, window_tokens
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval().to(self.device)
        labels = {i: str(l).lower() for i, l in self.mdl.config.id2label.items()}
        self.entail_idx = next(i for i, l in labels.items() if l.startswith("entail"))
        self.split_sentences = 0

    def _len(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _split(self, sent: str) -> list[str]:
        """A sentence longer than the window becomes consecutive token pieces
        of at most ``window_tokens`` tokens, overlapping by a quarter of that."""
        ids = self.tok(sent, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.window_tokens - 1:
            return [sent]
        self.split_sentences += 1
        step = max(1, self.window_tokens * 3 // 4)
        size = self.window_tokens - 1
        return [self.tok.decode(ids[i:i + size]) for i in range(0, max(1, len(ids) - size + step), step)]

    def entails_batch(self, premise: str, hypotheses: Sequence[str]) -> list[bool]:
        import torch
        from sum_engine_internal.research.meaning.meaning_loss import _sentences

        pieces: list[str] = []
        for sent in _sentences(premise) or [premise]:
            pieces.extend(self._split(sent))
        wins = windows(pieces, [self._len(x) for x in pieces], self.window_tokens)
        texts = [w for w, _ in wins]
        out = []
        for h in hypotheses:
            if not h.strip():
                out.append(False)
                continue
            best = 0.0
            for b in range(0, len(texts), 32):
                chunk = texts[b:b + 32]
                hyp_len = self._len(h)
                fits = hyp_len + 4 < 512 - self.window_tokens
                enc = self.tok(chunk, [h] * len(chunk), return_tensors="pt", padding=True,
                               truncation="only_first" if fits else "longest_first",
                               max_length=min(512, self.window_tokens + hyp_len + 4)).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(self.mdl(**enc).logits.float(), dim=-1)[:, self.entail_idx]
                best = max(best, float(probs.max()))
            out.append(best >= self.threshold)
        return out

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.entails_batch(premise, [hypothesis])[0]

    def manifest(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "model_revision": self.revision, "threshold": self.threshold,
                "window": f"runs of whole sentences of at most {self.window_tokens} tokens, consecutive runs "
                          "sharing a sentence where a run holds two or more; a longer sentence is split into "
                          "overlapping token pieces; the premise is truncated to the window as a final cap; "
                          "carried if any window entails",
                "device": self.device, "sentences_split": self.split_sentences}


def score(source: str, rewrite: str, entails_batch: Callable[[str, Sequence[str]], list[bool]],
          judge_name: str) -> dict[str, Any]:
    from sum_engine_internal.research.meaning.meaning_loss import explain_meaning_loss

    r = explain_meaning_loss(source, rewrite, entails=lambda p, h: entails_batch(p, [h])[0],
                             entails_batch=entails_batch, judge_name=judge_name, judge_version="1")
    return {
        "recall": r.recall, "fidelity": r.fidelity,
        "source_claims": r.source_claims, "recovered_claims": r.preserved_claims,
        "rewrite_claims": r.transform_claims,
        "not_recovered": list(r.dropped_claims),
        "not_supported_by_original": list(r.unsupported_claims),
    }


def _chat(client: Any, model: str, system: str, user: str, *, json_mode: bool = False, seed: int = 7) -> str:
    kwargs: dict[str, Any] = {"model": model, "temperature": 0, "seed": seed,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}]}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def read_input(path: Path) -> tuple[str, str]:
    if path.suffix.lower() in (".txt", ".md", ".markdown", ""):
        return path.read_text(encoding="utf-8"), "text"
    from sum_engine_internal.adapters.format_pivot import convert_to_markdown

    doc = convert_to_markdown(path)
    return doc.markdown, f"omni-format pivot ({doc.input_format}, {doc.converter})"


def summarize(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean, min and max of recall and fidelity across samples, per judge."""
    out: dict[str, Any] = {}
    for name in samples[0]["scores"]:
        out[name] = {}
        for m in ("recall", "fidelity"):
            vals = [x["scores"][name][m] for x in samples]
            out[name][m] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
    return out


def run(source: str, *, model: str, counts: Sequence[int], client: Any,
        judges: dict[str, Callable[[str, Sequence[str]], list[bool]]], samples: int = 3,
        log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    words = len(source.split())
    raw = json.loads(_chat(client, model, TAG_SYSTEM, source, json_mode=True))
    tags = validate_tags(raw, source)
    log(f"{len(tags)} tags ({sum(t['evidence_found'] for t in tags)} with evidence found in the text)")
    rungs = []
    for k in counts:
        used = tags_at(tags, k)
        if k and len(used) < k and rungs and rungs[-1]["tag_count"] == len(used):
            continue
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(used, 1))
        runs = []
        for j in range(samples):
            if k == 0:
                rewrite = _chat(client, model, CONTROL_SYSTEM.format(words=words), "Write it.", seed=7 + j)
            else:
                rewrite = _chat(client, model, REGEN_SYSTEM.format(words=words), listing, seed=7 + j)
            runs.append({"seed": 7 + j, "rewrite": rewrite, "rewrite_words": len(rewrite.split()),
                         "scores": {n: score(source, rewrite, fn, n) for n, fn in judges.items()}})
        summary = summarize(runs)
        rungs.append({"tag_count": len(used), "tags": used, "summary": summary, "samples": runs})
        log(f"k={len(used):>2}  " + "  ".join(
            f"{n}: recall {v['recall']['mean']:.2f} [{v['recall']['min']:.2f}-{v['recall']['max']:.2f}]"
            f" fidelity {v['fidelity']['mean']:.2f}" for n, v in summary.items()))
    return {"tags": tags, "rungs": rungs}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--counts", default=",".join(map(str, DEFAULT_COUNTS)))
    ap.add_argument("--samples", type=int, default=3, help="rewrites per tag count (seeds 7, 8, ...)")
    args = ap.parse_args(argv)
    counts = sorted({int(c) for c in args.counts.split(",")})
    if any(c < 0 or c > MAX_TAGS for c in counts):
        ap.error(f"counts must be between 0 and {MAX_TAGS}")
    if args.samples < 1:
        ap.error("samples must be at least 1")

    from openai import OpenAI
    from sum_engine_internal.research.meaning.local_judge import EmbeddingJudge

    source, how = read_input(args.input)
    gist = EmbeddingJudge(threshold=GIST_THRESHOLD)
    exact = WindowedNLI()
    result = run(source, model=args.model, counts=counts, client=OpenAI(),
                 judges={"gist": gist.entails_batch, "exact": exact.entails_batch}, samples=args.samples,
                 log=lambda s: print(s, file=sys.stderr))
    out = {
        "method": METHOD_ID,
        "scope": SCOPE,
        "input": {"file": args.input.name, "read_as": how, "words": len(source.split()),
                  "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()},
        "model": args.model,
        "sampling": "temperature 0 with a fixed seed per sample; the provider does not guarantee identical output on a re-run",
        "prompts": {"tags": TAG_SYSTEM, "rewrite": REGEN_SYSTEM, "control": CONTROL_SYSTEM},
        "judges": {"gist": {**gist.manifest(), "rule": "max cosine over sentences on the other side >= threshold"},
                   "exact": exact.manifest()},
        **result,
    }
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
