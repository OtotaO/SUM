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
    w.tok, w.window_tokens, w.split_sentences = _FakeTok(), window_tokens, 0
    return w


def test_long_sentence_is_split_into_pieces_within_the_window_covering_every_token():
    w = _splitter(10)
    sent = " ".join(f"t{i}" for i in range(25))
    pieces = w._split(sent)
    assert w.split_sentences == 1
    assert all(len(p.split()) <= 9 for p in pieces)
    assert set(" ".join(pieces).split()) == set(sent.split())
    assert pieces[0].split()[-1] in pieces[1].split()  # neighbouring pieces overlap


def test_short_sentence_is_not_split_and_split_pieces_fit_windows():
    w = _splitter(10)
    assert w._split("a b c") == ["a b c"] and w.split_sentences == 0
    pieces = w._split(" ".join(f"t{i}" for i in range(30)))
    assert not any(over for _, over in windows(pieces, [len(p.split()) for p in pieces], 10))
