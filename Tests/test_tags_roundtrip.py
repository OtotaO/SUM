"""Pure parts of scripts/tags_roundtrip.py: tag validation, ordering, tag
selection, judge windows and scoring. No model or network is used."""
from __future__ import annotations

from scripts.tags_roundtrip import score, summarize, tags_at, validate_tags, windows

SOURCE = "The lease ends in May. Pets need written permission. The deposit is $1,800."


def test_tags_follow_evidence_order_and_rank_is_reassigned():
    raw = {"tags": [
        {"tag": "deposit", "kind": "mentioned", "rank": 5, "evidence": "The deposit is $1,800."},
        {"tag": "lease end", "kind": "mentioned", "rank": 9, "evidence": "lease ends in May"},
        {"tag": "pet policy", "kind": "implied", "rank": 2, "evidence": "Pets need written permission"},
    ]}
    tags = validate_tags(raw, SOURCE)
    assert [t["tag"] for t in tags] == ["lease end", "pet policy", "deposit"]
    assert {t["tag"]: t["rank"] for t in tags} == {"pet policy": 1, "deposit": 2, "lease end": 3}
    assert all(t["evidence_found"] for t in tags)


def test_evidence_not_in_text_is_recorded_and_sorted_last():
    raw = {"tags": [
        {"tag": "invented", "kind": "mentioned", "rank": 1, "evidence": "the rent doubles"},
        {"tag": "deposit", "kind": "mentioned", "rank": 2, "evidence": "deposit is $1,800"},
    ]}
    tags = validate_tags(raw, SOURCE)
    assert [t["tag"] for t in tags] == ["deposit", "invented"]
    assert tags[1]["evidence_found"] is False and tags[1]["evidence_position"] is None


def test_evidence_match_ignores_case_whitespace_and_curly_quotes():
    tags = validate_tags({"tags": [{"tag": "x", "rank": 1, "evidence": "it’s  HERE"}]}, "So it's here now.")
    assert tags[0]["evidence_found"] is True


def test_bad_entries_long_tags_and_duplicates_are_dropped():
    raw = {"tags": [
        "not an object",
        {"tag": "", "rank": 1},
        {"tag": "one two three four five", "rank": 2},
        {"tag": "Deposit", "rank": 3, "evidence": "deposit"},
        {"tag": "deposit", "rank": 4, "evidence": "deposit"},
        {"tag": "weird kind", "kind": "both", "rank": True, "evidence": "lease"},
    ]}
    tags = validate_tags(raw, SOURCE)
    assert [t["tag"] for t in tags] == ["weird kind", "Deposit"]
    assert tags[0]["kind"] == "unlabelled"
    assert {t["tag"]: t["rank"] for t in tags} == {"Deposit": 1, "weird kind": 2}


def test_overlong_evidence_is_located_and_flagged_not_treated_as_invented():
    long_quote = " ".join(f"w{i}" for i in range(13))
    source = long_quote + ". Later text."
    raw = {"tags": [{"tag": "later", "rank": 1, "evidence": "Later text"},
                    {"tag": "long", "rank": 2, "evidence": long_quote},
                    {"tag": "made up", "rank": 3, "evidence": "not in the text"}]}
    tags = validate_tags(raw, source)
    assert [t["tag"] for t in tags] == ["long", "later", "made up"]
    assert tags[0]["evidence_found"] is True and tags[0]["evidence_over_word_limit"] is True
    assert tags[1]["evidence_over_word_limit"] is False
    assert tags[2]["evidence_found"] is False and tags[2]["evidence_over_word_limit"] is False


def test_at_most_32_tags_are_kept():
    raw = {"tags": [{"tag": f"tag {i}", "rank": i} for i in range(1, 41)]}
    assert len(validate_tags(raw, SOURCE)) == 32


def test_missing_tag_list_is_an_error():
    import pytest

    with pytest.raises(ValueError):
        validate_tags({"labels": []}, SOURCE)


def test_tags_at_keeps_top_k_in_replication_order():
    tags = [{"tag": "a", "rank": 3}, {"tag": "b", "rank": 1}, {"tag": "c", "rank": 2}]
    assert tags_at(tags, 2) == ["b", "c"]
    assert tags_at(tags, 0) == []
    assert tags_at(tags, 10) == ["a", "b", "c"]


def test_windows_pack_whole_sentences_with_one_sentence_overlap():
    sents = ["s1", "s2", "s3", "s4", "s5"]
    wins = windows(sents, [3, 3, 3, 3, 3], budget=8)  # two sentences fit (3+1)*2
    assert wins == [("s1 s2", False), ("s2 s3", False), ("s3 s4", False), ("s4 s5", False)]


def test_windows_cover_every_sentence_and_flag_oversize_ones():
    sents = ["short", "huge", "tail"]
    wins = windows(sents, [2, 50, 2], budget=10)
    covered = " ".join(w for w, _ in wins).split()
    assert set(covered) == set(sents)
    assert ("huge", True) in wins
    assert all(not over for w, over in wins if w != "huge")


def test_windows_single_window_when_everything_fits():
    assert windows(["a", "b"], [1, 1], budget=100) == [("a b", False)]


def test_score_counts_recovered_and_unsupported_sentences():
    def entails_batch(premise: str, hypotheses):
        return [h.rstrip(".") in premise for h in hypotheses]

    s = score("Cats sleep. Dogs bark.", "Cats sleep. Birds sing.", entails_batch, "substring")
    assert s["source_claims"] == 2 and s["recovered_claims"] == 1
    assert s["not_recovered"] == ["Dogs bark."]
    assert s["not_supported_by_original"] == ["Birds sing."]
    assert s["recall"] == 0.5 and s["fidelity"] == 0.5


def test_summarize_reports_mean_min_max_per_judge():
    runs = [{"scores": {"gist": {"recall": 0.2, "fidelity": 1.0}}},
            {"scores": {"gist": {"recall": 0.4, "fidelity": 0.5}}}]
    s = summarize(runs)
    assert s["gist"]["recall"] == {"mean": 0.30000000000000004, "min": 0.2, "max": 0.4}
    assert s["gist"]["fidelity"]["min"] == 0.5 and s["gist"]["fidelity"]["max"] == 1.0


class _FakeTok:
    """Whitespace tokenizer: one token per word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def _splitter(window_tokens):
    from scripts.tags_roundtrip import WindowedNLI

    w = object.__new__(WindowedNLI)
    w.tok, w.window_tokens, w.split_sentences = _FakeTok(), window_tokens, set()
    return w


def test_long_sentence_is_split_into_pieces_within_the_window_covering_every_token():
    w = _splitter(10)
    sent = " ".join(f"t{i}" for i in range(25))
    pieces = w._split(sent)
    assert len(w.split_sentences) == 1
    w._split(sent)
    assert len(w.split_sentences) == 1  # the same sentence again is not counted twice
    assert all(len(p.split()) <= 9 for p in pieces)
    assert set(" ".join(pieces).split()) == set(sent.split())
    assert pieces[0].split()[-1] in pieces[1].split()  # neighbouring pieces overlap


def test_short_sentence_is_not_split_and_split_pieces_fit_windows():
    w = _splitter(10)
    assert w._split("a b c") == ["a b c"] and not w.split_sentences
    pieces = w._split(" ".join(f"t{i}" for i in range(30)))
    assert not any(over for _, over in windows(pieces, [len(p.split()) for p in pieces], 10))


def _t(tag, school, rank, pos):
    return {"tag": tag, "school": school, "rank": rank, "evidence_position": pos}


def test_mixed_at_shares_k_by_school_and_merges_in_replication_order():
    from scripts.tags_roundtrip import mixed_at

    by_school = {
        "topic": [_t("t1", "topic", 2, 5), _t("t2", "topic", 1, 40), _t("t3", "topic", 3, 60), _t("t4", "topic", 4, 1)],
        "form": [_t("f1", "form", 1, 10), _t("f2", "form", 2, 0)],
        "fact": [_t("x1", "fact", 1, None), _t("x2", "fact", 2, 30)],
    }
    used = mixed_at(by_school, 4)  # 2 topic, 1 form, 1 fact
    assert [u["tag"] for u in used] == ["t1", "f1", "t2", "x1"]
    assert [u["school"] for u in used] == ["topic", "form", "topic", "fact"]
    six = mixed_at(by_school, 6)  # 6*2//4=3 topic, 1 form, 1 fact, remainder 1 to topic
    assert sum(u["school"] == "topic" for u in six) == 4


def test_mixed_at_with_a_missing_school_uses_what_exists():
    from scripts.tags_roundtrip import mixed_at

    used = mixed_at({"topic": [_t("t1", "topic", 1, 0)]}, 4)
    assert used == [{"school": "topic", "tag": "t1"}]


def test_schools_label_tags_and_apply_their_word_limits():
    raw = {"tags": [{"tag": "author used the simulator at night", "rank": 1, "evidence": "lease"}]}
    assert validate_tags(raw, SOURCE)[0:] == []  # 6 words is over the topic limit
    fact = validate_tags(raw, SOURCE, school="fact", max_words=8)
    assert fact[0]["school"] == "fact" and fact[0]["kind"] is None
    topic = validate_tags({"tags": [{"tag": "lease", "kind": "implied", "rank": 1}]}, SOURCE)
    assert topic[0]["school"] == "topic" and topic[0]["kind"] == "implied"


def test_form_signals_count_first_person_words_and_sentence_length():
    from scripts.tags_roundtrip import form_signals

    f = form_signals("I went home. We ate.")
    assert f["first_person_share"] == 0.4  # i, we of 5 words
    assert f["words_per_sentence"] == 2.5


class _FakeClient:
    """Answers tag prompts with fixed JSON and rewrite prompts with a fixed
    text; records every call."""

    def __init__(self):
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        import json as _json
        from types import SimpleNamespace

        system = kw["messages"][0]["content"]
        self.calls.append(system)
        if "form tags" in system:
            body = _json.dumps({"tags": [{"tag": "first-person anecdote", "rank": 1, "evidence": "The lease"}]})
        elif "fact tags" in system:
            body = _json.dumps({"tags": [{"tag": "deposit is 1,800 dollars", "rank": 1, "evidence": "deposit"}]})
        elif "You make tags" in system:
            body = _json.dumps({"tags": [{"tag": "lease", "kind": "mentioned", "rank": 1, "evidence": "lease"},
                                         {"tag": "pets", "kind": "mentioned", "rank": 2, "evidence": "Pets"}]})
        else:
            body = "The lease ends in May."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])


def test_run_end_to_end_with_fake_model_and_judges():
    from scripts.tags_roundtrip import run

    def judge(premise, hyps):
        return [h.rstrip(".") in premise for h in hyps]

    client = _FakeClient()
    out = run(SOURCE, model="fake", counts=[0, 1, 2, 8, 16], client=client, judges={"j": judge},
              samples=2, schools=["topic", "form", "fact"], mixed_counts=[4])
    conds = [(r["condition"], r["tag_count"]) for r in out["rungs"]]
    # k=2 uses both topic tags; k=8 and k=16 add nothing new and are skipped.
    assert conds == [("control", 0), ("topic", 1), ("topic", 2), ("mixed", 4)]
    mixed = out["rungs"][-1]
    assert [u["tag"] for u in mixed["tags"]] == ["first-person anecdote", "lease", "pets", "deposit is 1,800 dollars"]
    assert mixed["tag_words"] == 2 + 1 + 1 + 4
    assert all(len(r["samples"]) == 2 for r in out["rungs"])
    assert out["rungs"][1]["summary"]["j"]["recall"]["mean"] == 1 / 3
    assert set(out["tags"]) == {"topic", "form", "fact"}
    assert "first_person_share" in out["source_form_signals"]


def test_form_signals_treat_apostrophes_alike_and_count_contractions_once():
    from scripts.tags_roundtrip import form_signals

    straight, curly = form_signals("I'm here. We'd go."), form_signals("I\u2019m here. We\u2019d go.")
    assert straight == curly
    assert straight["first_person_share"] == 0.5  # i'm, we'd of 4 words
    assert straight["words_per_sentence"] == 2.0
    assert form_signals("Can\u2019t stop.")["words_per_sentence"] == 2.0


def test_run_without_topic_school_writes_no_empty_topic_rows():
    from scripts.tags_roundtrip import run

    out = run(SOURCE, model="fake", counts=[4], client=_FakeClient(), judges={"j": lambda p, hs: [False] * len(hs)},
              samples=1, schools=["form", "fact"], mixed_counts=[4])
    assert [r["condition"] for r in out["rungs"]] == ["mixed"]


def test_form_signals_count_lets_as_first_person():
    from scripts.tags_roundtrip import form_signals

    assert form_signals("Let\u2019s go now.")["first_person_share"] == 1 / 3


def test_load_tags_checks_text_and_schools_and_accepts_v0():
    import pytest
    from scripts.tags_roundtrip import load_tags

    prev = {"input": {"sha256": "abc"}, "tags": {"topic": [{"tag": "a"}], "form": []}}
    assert load_tags(prev, "abc", ["topic", "form"]) == {"topic": [{"tag": "a"}], "form": []}
    assert load_tags({"input": {"sha256": "abc"}, "tags": [{"tag": "a"}]}, "abc", ["topic"]) == {"topic": [{"tag": "a"}]}
    for bad, schools in [({"input": {"sha256": "other"}, "tags": {"topic": []}}, ["topic"]),
                         (prev, ["topic", "fact"]),
                         ({"tags": {"topic": []}}, ["topic"]),
                         ([], ["topic"])]:
        with pytest.raises(ValueError):
            load_tags(bad, "abc", schools)


def test_run_with_reused_tags_makes_no_tag_calls_and_runs_each_mixed_prompt():
    from scripts.tags_roundtrip import run

    client = _FakeClient()
    tags = {"topic": validate_tags({"tags": [{"tag": "lease", "rank": 1, "evidence": "lease"}]}, SOURCE),
            "form": validate_tags({"tags": [{"tag": "plain notice", "rank": 1, "evidence": "Pets"}]}, SOURCE, school="form"),
            "fact": validate_tags({"tags": [{"tag": "deposit is 1,800 dollars", "rank": 1, "evidence": "deposit"}]},
                                  SOURCE, school="fact", max_words=8)}
    out = run(SOURCE, model="fake", counts=[1], client=client, judges={"j": lambda p, hs: [True] * len(hs)},
              samples=1, schools=["topic", "form", "fact"], mixed_counts=[4],
              mixed_conditions=["mixed", "mixed_plain", "mixed_grounded"], tags=tags)
    assert not any(c.startswith("You make") for c in client.calls)
    assert [r["condition"] for r in out["rungs"]] == ["topic", "mixed", "mixed_plain", "mixed_grounded"]
    assert len(client.calls) == 4 and len(set(client.calls[1:])) == 3  # one distinct prompt per mixed condition
    assert "leave it out" in client.calls[3] and "Follow the form tags" not in client.calls[2]


def test_tag_premise_writes_each_school_as_a_sentence():
    from scripts.tags_roundtrip import tag_premise

    used = ["lease", {"school": "form", "tag": "plain notice"}, {"school": "fact", "tag": "deposit is 1,800 dollars"},
            {"school": "topic", "tag": "pets"}]
    assert tag_premise(used) == ("The text is about lease. The text uses plain notice. Deposit is 1,800 dollars. "
                                 "The text is about pets.")


def test_cut_keeps_only_sentences_the_tags_entail_and_never_reads_the_source():
    from scripts.tags_roundtrip import cut_rewrite

    seen = []

    def max_probs(premise, hyps):
        seen.append(premise)
        return [0.9 if "deposit" in h else (None if "long" in h else 0.1) for h in hyps]

    c = cut_rewrite("The deposit is back. Pets roam free. A long one.", ["deposit"], max_probs)
    assert c == {"text": "The deposit is back.", "kept_sentences": 1, "rewrite_sentences": 3}
    assert seen == ["The text is about deposit."]
    assert cut_rewrite("Anything.", [], max_probs)["kept_sentences"] == 0


def test_summarize_cut_counts_empty_cuts_as_zero_recall_and_leaves_them_out_of_fidelity():
    from scripts.tags_roundtrip import summarize_cut

    runs = [{"scores": {"j": {}}, "cut": {"kept_sentences": 2, "scores": {"j": {"recall": 0.4, "fidelity": 1.0}}}},
            {"scores": {"j": {}}, "cut": {"kept_sentences": 0, "scores": None}}]
    s = summarize_cut(runs)
    assert s["nonempty"] == 1 and s["kept_sentences_mean"] == 1.0
    assert s["j"]["recall"] == {"mean": 0.2, "min": 0.0, "max": 0.4}
    assert s["j"]["fidelity"] == {"mean": 1.0, "min": 1.0, "max": 1.0}
    none = summarize_cut([runs[1]])
    assert none["j"]["fidelity"] is None and none["nonempty"] == 0


def test_run_with_cut_records_cut_per_sample_but_not_for_control():
    from scripts.tags_roundtrip import run

    out = run(SOURCE, model="fake", counts=[0, 1], client=_FakeClient(),
              judges={"j": lambda p, hs: [h.rstrip(".") in p for h in hs]}, samples=2,
              cut=lambda premise, hyps: [0.9] * len(hyps))
    control, topic = out["rungs"]
    assert "cut" not in control["samples"][0] and "cut_summary" not in control
    assert topic["samples"][0]["cut"]["kept_sentences"] == 1
    assert topic["cut_summary"]["j"]["recall"]["mean"] == 1 / 3


def test_cut_keeps_unpunctuated_items_as_separate_units_for_scoring():
    from scripts.tags_roundtrip import cut_rewrite
    from sum_engine_internal.research.meaning.meaning_loss import _sentences

    c = cut_rewrite("Alignment methods\nEthical principles\nIt is fine.", ["x"], lambda p, hs: [0.9] * len(hs))
    assert c["kept_sentences"] == 3
    assert len(_sentences(c["text"])) == c["kept_sentences"]
