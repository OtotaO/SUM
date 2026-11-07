"""Tags and back: turn a text into ordered tags, write the text again from the
tags alone, and measure what came back.

    PYTHONPATH=. python scripts/tags_roundtrip.py INPUT [--out FILE.json]
        [--model gpt-4o-mini-2024-07-18] [--counts 0,4,8,16,32] [--samples 3]
        [--schools topic,form,fact] [--mixed-counts 8,16,32]

INPUT is a text or Markdown file, or any format the omni-format pivot converts
(PDF, DOCX, HTML, EPUB, ...; needs the markitdown extra).

Steps, all recorded in the output JSON:
  1. One model call per tag school proposes up to 32 tags, each with an
     importance rank and the shortest verbatim quote it rests on. topic: 1 to
     4 words, marked "mentioned" (the text names it) or "implied" (a reader
     would file the text under it). form: 1 to 4 words on how the text is
     written (voice, genre, structure, tone). fact: a 3 to 8 word
     subject-verb-object phrase. Quotes are checked against the text; a quote
     that is not found is recorded as not found. Tags are kept in replication
     order: the order their evidence appears in the text.
  2. For each tag count k, the k highest-ranked topic tags (in replication
     order, tag words only, no quotes) go to a second call that writes the
     text again at the original length, several times (--samples, default
     3, one seed each), because the model's output varies between runs; the
     mean and range are reported. k = 0 is a control: the model writes a text
     of that length from nothing, which shows how much credit a judge gives
     unrelated prose. With more than one school, a mixed condition (--mixed-counts)
     shares k tags as half topic, a quarter form and a quarter fact, merged in
     replication order, labelled by school, and rewritten with a prompt that
     also tells the model to follow the form tags. --mixed-conditions
     chooses which prompts the mixed tags go through (default: mixed only):
     mixed, mixed_plain (without that instruction) and mixed_grounded (write
     only what the tags state, leave gaps out). --tags-from reuses the tags
     of an earlier results file for the same text instead of making new
     ones.
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
       credited unrelated text. Support spread across windows is missed. A
       sentence too long to sit beside a full window (over about 430 tokens)
       is counted as not carried, and the manifest counts such sentences.
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

METHOD_ID = "sum.tags_roundtrip.v1"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_COUNTS = (0, 4, 8, 16, 32)
DEFAULT_MIXED_COUNTS = (8, 16, 32)
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
FORM_SYSTEM = (
    "You make form tags for a text: labels for how it is written, not what it "
    "is about. A tag is 1 to 4 words and names one feature of its voice (who "
    "speaks, to whom, in which person), genre, structure (for example a list, "
    "a quotation, a dialogue, an aside), tone, or a rhetorical move. Return "
    'JSON {"tags": [...]} with at most 32 tags. Each tag is an object '
    '{"tag": string, "rank": integer, "evidence": string}. rank: how much the '
    "feature shapes the text, 1 = most, each rank used once. evidence: the "
    "shortest exact quote from the text, at most 12 words, copied character "
    "for character, that shows the feature. List the tags in the order their "
    "evidence first appears in the text."
)
FACT_SYSTEM = (
    "You make fact tags for a text: the events, actions and claims it states, "
    "each as a short subject-verb-object phrase of 3 to 8 words (for example "
    '"author tested the app at night"). Return JSON {"tags": [...]} with at '
    'most 32 tags. Each tag is an object {"tag": string, "rank": integer, '
    '"evidence": string}. rank: importance to the text, 1 = most important, '
    "each rank used once. evidence: the shortest exact quote from the text, at "
    "most 12 words, copied character for character, that states it. List the "
    "tags in the order their evidence first appears in the text."
)
SCHOOLS = {"topic": (TAG_SYSTEM, 4), "form": (FORM_SYSTEM, 4), "fact": (FACT_SYSTEM, 8)}
MIXED_SHARES = (("topic", 2), ("form", 1), ("fact", 1))  # k/2 topic, k/4 form, k/4 fact
REGEN_MIXED_SYSTEM = (
    "You are given tags taken from a text, in the order they appear in it. "
    "Each tag is labelled topic (what the text is about), form (how it is "
    "written: voice, genre, structure, tone) or fact (something the text "
    "states). Write the text they came from, about {words} words. Follow the "
    "form tags for voice and structure. Use every tag. Do not add names, "
    "numbers, dates or quotations that the tags do not contain. Return only "
    "the text."
)
REGEN_MIXED_PLAIN = REGEN_MIXED_SYSTEM.replace(" Follow the form tags for voice and structure.", "")
REGEN_GROUNDED = REGEN_MIXED_SYSTEM.replace(
    " Use every tag.",
    " Use every tag. Write only what the tags state or directly imply; where they leave a gap, leave "
    "it out instead of filling it, even if the text comes out shorter.",
).replace("about {words} words", "at most about {words} words")
MIXED_PROMPTS = {"mixed": REGEN_MIXED_SYSTEM, "mixed_plain": REGEN_MIXED_PLAIN, "mixed_grounded": REGEN_GROUNDED}
CONTROL_SYSTEM = "Write a text of about {words} words. Return only the text."
FIRST_PERSON = frozenset("i me my mine myself we us our ours ourselves".split())
FIRST_PERSON_CONTRACTIONS = frozenset("i'm i've i'd i'll we're we've we'd we'll let's".split())


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def validate_tags(raw: Any, source: str, *, school: str = "topic",
                  max_words: int = MAX_TAG_WORDS) -> list[dict[str, Any]]:
    """Normalise the model's tag list. Drops entries that are not objects or
    whose tag is empty or longer than ``max_words`` words; keeps at most
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
        if len(tag.split()) > max_words or _norm(tag) in seen:
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
            "school": school,
            "kind": (item.get("kind") if item.get("kind") in ("mentioned", "implied") else "unlabelled")
                    if school == "topic" else None,
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


def _order_key(t: dict[str, Any]) -> tuple:
    return (t["evidence_position"] is None, t["evidence_position"] or 0)


def mixed_at(by_school: dict[str, Sequence[dict[str, Any]]], k: int) -> list[dict[str, str]]:
    """k tags shared across schools by MIXED_SHARES (k/2 topic, k/4 form, k/4
    fact, rounding down, any remainder going to topic), each school's
    highest-ranked first, then merged into one sequence in replication order
    (tags whose evidence was not found go last, school by school). Returns
    [{"school", "tag"}]."""
    total = sum(w for _, w in MIXED_SHARES)
    quota = {name: k * w // total for name, w in MIXED_SHARES}
    quota[MIXED_SHARES[0][0]] += k - sum(quota.values())
    picked = []
    for name, _ in MIXED_SHARES:
        picked.extend(t for t in by_school.get(name, ()) if t["rank"] <= quota[name])
    picked.sort(key=_order_key)
    return [{"school": t["school"], "tag": t["tag"]} for t in picked]


def form_signals(text: str) -> dict[str, float]:
    """Two descriptive signals of voice: the share of words that are
    first-person pronouns or contractions of them (I'm, we've, let's, ...), and the
    mean number of words per sentence. Curly and straight apostrophes are
    treated alike, and a contraction is one word."""
    from sum_engine_internal.research.meaning.meaning_loss import _sentences

    words = re.findall(r"\w+(?:'\w+)*", text.lower().replace("\u2019", "'"))
    sents = _sentences(text)
    first = sum(w in FIRST_PERSON or w in FIRST_PERSON_CONTRACTIONS for w in words)
    return {"first_person_share": first / max(1, len(words)),
            "words_per_sentence": len(words) / max(1, len(sents))}


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
        self.specials = self.tok.num_special_tokens_to_add(pair=True)
        self.split_sentences: set[str] = set()
        self.long_hypotheses: set[str] = set()

    def _len(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _split(self, sent: str) -> list[str]:
        """A sentence longer than the window becomes consecutive token pieces
        of at most ``window_tokens - 1`` tokens; each piece starts
        ``3 * window_tokens // 4`` tokens after the previous one, so
        neighbours overlap by roughly a quarter of the window."""
        ids = self.tok(sent, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.window_tokens - 1:
            return [sent]
        self.split_sentences.add(hashlib.sha256(sent.encode("utf-8")).hexdigest())
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
            if self.window_tokens + self._len(h) + self.specials > 512:
                # The judge cannot read the whole sentence beside a full
                # window; it is counted as not carried rather than judged on
                # a prefix.
                self.long_hypotheses.add(hashlib.sha256(h.encode("utf-8")).hexdigest())
                out.append(False)
                continue
            best = 0.0
            for b in range(0, len(texts), 32):
                chunk = texts[b:b + 32]
                enc = self.tok(chunk, [h] * len(chunk), return_tensors="pt", padding=True,
                               truncation="only_first",
                               max_length=self.window_tokens + self._len(h) + self.specials).to(self.device)
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
                "device": self.device,
                "distinct_sentences_split": len(self.split_sentences),
                "distinct_sentences_too_long_to_judge": len(self.long_hypotheses),
                "too_long_rule": "a sentence that cannot fit beside a full window is counted as not carried"}


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
        schools: Sequence[str] = ("topic",), mixed_counts: Sequence[int] = (),
        mixed_conditions: Sequence[str] = ("mixed",), tags: dict[str, list[dict[str, Any]]] | None = None,
        log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """With ``tags`` (school -> validated tags, e.g. from an earlier run of
    the same text), no tagging call is made and those tags are reused."""
    words = len(source.split())
    by_school: dict[str, list[dict[str, Any]]] = dict(tags) if tags is not None else {}
    for name in ([] if tags is not None else schools):
        system, max_words = SCHOOLS[name]
        raw = json.loads(_chat(client, model, system, source, json_mode=True))
        by_school[name] = validate_tags(raw, source, school=name, max_words=max_words)
        log(f"{name}: {len(by_school[name])} tags "
            f"({sum(t['evidence_found'] for t in by_school[name])} with evidence found in the text)")

    def rewrites(system: str, user: str) -> list[dict[str, Any]]:
        out = []
        for j in range(samples):
            rewrite = _chat(client, model, system, user, seed=7 + j)
            out.append({"seed": 7 + j, "rewrite": rewrite, "rewrite_words": len(rewrite.split()),
                        "form_signals": form_signals(rewrite),
                        "scores": {n: score(source, rewrite, fn, n) for n, fn in judges.items()}})
        return out

    def record(condition: str, used: list, runs: list) -> dict[str, Any]:
        summary = summarize(runs)
        tag_words = sum(len((u["tag"] if isinstance(u, dict) else u).split()) for u in used)
        log(f"{condition:>6} k={len(used):>2} ({tag_words:>3} tag words)  " + "  ".join(
            f"{n}: recall {v['recall']['mean']:.2f} [{v['recall']['min']:.2f}-{v['recall']['max']:.2f}]"
            f" fidelity {v['fidelity']['mean']:.2f}" for n, v in summary.items()))
        return {"condition": condition, "tag_count": len(used), "tag_words": tag_words, "tags": used,
                "summary": summary, "samples": runs}

    rungs = []
    topic = by_school.get("topic", [])
    for k in counts:
        used = tags_at(topic, k)
        if k and len(used) < k and rungs and rungs[-1]["tag_count"] == len(used):
            continue
        if k == 0:
            runs = rewrites(CONTROL_SYSTEM.format(words=words), "Write it.")
            rungs.append(record("control", [], runs))
            continue
        if not used:
            continue
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(used, 1))
        rungs.append(record("topic", used, rewrites(REGEN_SYSTEM.format(words=words), listing)))
    if len(by_school) > 1:
        for condition in mixed_conditions:
            for k in mixed_counts:
                used = mixed_at(by_school, k)
                listing = "\n".join(f"{i}. [{u['school']}] {u['tag']}" for i, u in enumerate(used, 1))
                rungs.append(record(condition, used, rewrites(MIXED_PROMPTS[condition].format(words=words), listing)))
    return {"tags": by_school, "source_form_signals": form_signals(source), "rungs": rungs}


def load_tags(previous: Any, source_sha: str, schools: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """Tags of an earlier results JSON, checked to belong to the same text
    (input SHA-256) and to hold every requested school. A v0 file holds one
    topic list."""
    if not isinstance(previous, dict) or not isinstance(previous.get("input"), dict):
        raise ValueError("--tags-from: not a tags_roundtrip results file")
    if previous["input"].get("sha256") != source_sha:
        raise ValueError("--tags-from: that run was on a different text (input SHA-256 differs)")
    tags = previous.get("tags")
    if isinstance(tags, list):
        tags = {"topic": tags}
    if not isinstance(tags, dict) or any(not isinstance(tags.get(n), list) for n in schools):
        raise ValueError(f"--tags-from: the file lacks tags for {', '.join(schools)}")
    return {n: tags[n] for n in schools}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--counts", default=",".join(map(str, DEFAULT_COUNTS)))
    ap.add_argument("--samples", type=int, default=3, help="rewrites per tag count (seeds 7, 8, ...)")
    ap.add_argument("--schools", default="topic,form,fact", help="tag schools to build: topic, form, fact")
    ap.add_argument("--mixed-counts", default=",".join(map(str, DEFAULT_MIXED_COUNTS)),
                    help="tag counts for the mixed condition (used when more than one school is built)")
    ap.add_argument("--mixed-conditions", default="mixed",
                    help="rewrite prompts for mixed tags: mixed, mixed_plain (no instruction to follow form "
                         "tags), mixed_grounded (write only what the tags state)")
    ap.add_argument("--tags-from", type=Path,
                    help="reuse the tags of an earlier results JSON for the same text instead of making new ones")
    args = ap.parse_args(argv)
    counts = sorted({int(c) for c in args.counts.split(",")})
    if any(c < 0 or c > MAX_TAGS for c in counts):
        ap.error(f"counts must be between 0 and {MAX_TAGS}")
    if args.samples < 1:
        ap.error("samples must be at least 1")
    schools = [x.strip() for x in args.schools.split(",") if x.strip()]
    if not schools or any(x not in SCHOOLS for x in schools) or len(set(schools)) != len(schools):
        ap.error(f"schools must be distinct names from {', '.join(SCHOOLS)}")
    mixed_counts = sorted({int(c) for c in args.mixed_counts.split(",") if c.strip()})
    if any(c < 1 or c > MAX_TAGS for c in mixed_counts):
        ap.error(f"mixed counts must be between 1 and {MAX_TAGS}")
    mixed_conditions = [x.strip() for x in args.mixed_conditions.split(",") if x.strip()]
    if any(x not in MIXED_PROMPTS for x in mixed_conditions) or len(set(mixed_conditions)) != len(mixed_conditions):
        ap.error(f"mixed conditions must be distinct names from {', '.join(MIXED_PROMPTS)}")

    from openai import OpenAI
    from sum_engine_internal.research.meaning.local_judge import EmbeddingJudge

    source, how = read_input(args.input)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    reused = None
    if args.tags_from:
        try:
            reused = load_tags(json.loads(args.tags_from.read_text(encoding="utf-8")), source_sha, schools)
        except ValueError as e:
            ap.error(str(e))
    gist = EmbeddingJudge(threshold=GIST_THRESHOLD)
    exact = WindowedNLI()
    result = run(source, model=args.model, counts=counts, client=OpenAI(),
                 judges={"gist": gist.entails_batch, "exact": exact.entails_batch}, samples=args.samples,
                 schools=schools, mixed_counts=mixed_counts, mixed_conditions=mixed_conditions, tags=reused,
                 log=lambda s: print(s, file=sys.stderr))
    out = {
        "method": METHOD_ID,
        "scope": SCOPE,
        "input": {"file": args.input.name, "read_as": how, "words": len(source.split()),
                  "sha256": source_sha},
        "tags_reused_from": ({"file": args.tags_from.name,
                              "sha256": hashlib.sha256(args.tags_from.read_bytes()).hexdigest()}
                             if args.tags_from else None),
        "model": args.model,
        "sampling": "temperature 0 with a fixed seed per sample; the provider does not guarantee identical output on a re-run",
        "prompts": {"tags": {n: SCHOOLS[n][0] for n in schools}, "rewrite": REGEN_SYSTEM,
                    **{f"rewrite_{c}": MIXED_PROMPTS[c] for c in mixed_conditions}, "control": CONTROL_SYSTEM},
        "form_signals": "descriptive only: share of words that are first-person pronouns or contractions of them (I'm, we've, let's, ...), with curly and straight apostrophes treated alike, and mean words per sentence",
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
