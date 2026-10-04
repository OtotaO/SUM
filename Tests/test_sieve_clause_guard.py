"""Tests for the sieve's clause guard (extractor v2, playbook item H2).

Extractor v1 walked every ROOT-or-VERB token of a sentence and let the
last subject, predicate and object win independently. On multi-clause
sentences that produced triples the source never asserts:

    "If the tenant pays rent late, the landlord charges a fee."
        -> (landlord, charge, fee)   conditional consequent as fact
    "Bob said that Alice stole the car."
        -> (alice, steal, car)       reported speech as fact
    "Alice owns the house where Bob grew up."
        -> (bob, grow, house)        relative clause stitched to main object

These tests pin the guard: a triple comes from ONE main-clause
predicate; conditional, question, negated and cross-clause sentences
yield nothing and are counted; the asserted main clause of a sentence
with an ordinary subordinate clause is still extracted; every public
extraction method agrees; and the v2 negative-control corpus has zero
unexpected verdicts in the two clause-guard failure modes.
"""
from __future__ import annotations

import functools
import io
import json
import sys
from pathlib import Path

import pytest

from sum_engine_internal.algorithms.syntactic_sieve import (
    SIEVE_EXTRACTOR_ID,
    DeterministicSieve,
)

REPO = Path(__file__).resolve().parent.parent
CORPUS_V1 = REPO / "scripts/bench/corpora/seed_negative_control_v1.json"
CORPUS_V2 = REPO / "scripts/bench/corpora/seed_negative_control_v2.json"
BASELINE_RECEIPT = REPO / "fixtures/bench_receipts/negative_control_2026-05-17.json"


@functools.lru_cache(maxsize=1)
def _sieve() -> DeterministicSieve:
    return DeterministicSieve()  # type: ignore[no-untyped-call]


def _report(text: str) -> dict:
    return _sieve().extract_triplets_with_report(text)[1]


# (text, the triple v1 emitted, the suppression reason v2 records)
SUPPRESSED = [
    ("If the tenant pays rent late, the landlord charges a fee.",
     ("landlord", "charge", "fee"), "conditional"),
    ("The landlord charges a fee if the tenant pays rent late.",
     ("tenant", "pay", "rent"), "conditional"),
    ("The company keeps its assets unless the court approves the merger.",
     ("court", "approve", "merger"), "conditional"),
    ("Provided the buyer pays the deposit, the seller transfers the deed.",
     ("seller", "transfer", "deed"), "conditional"),
    ("The seller transfers the deed provided that the buyer pays the deposit.",
     ("buyer", "pay", "deposit"), "conditional"),
    ("Had the tenant paid rent, the landlord would have returned the deposit.",
     ("landlord", "return", "deposit"), "conditional"),
    ("The landlord would have returned the deposit had the tenant paid rent.",
     ("tenant", "pay", "rent"), "conditional"),
    ("Were the court to approve the merger, the company would sell its assets.",
     ("company", "sell", "asset"), "conditional"),
    ("Should the buyer default, the seller keeps the deposit.",
     ("seller", "keep", "deposit"), "conditional"),
    ("When you cancel the plan, we charge a fee.",
     ("we", "charge", "fee"), "conditional"),
    ("The insurer pays the claim as long as the owner files the report.",
     ("owner", "file", "report"), "conditional"),
    ("Suppose Alice had won the election.",
     ("alice", "win", "election"), "conditional"),
    ("Should we trust AI-generated content?",
     ("ai", "generate", "generated content"), "question"),
    ("Bob said that Alice stole the car.",
     ("alice", "steal", "car"), "cross_clause"),
    ("Alice believes the vaccine causes autism.",
     ("vaccine", "cause", "autism"), "cross_clause"),
    ("Alice wants to buy the car.",
     ("alice", "buy", "car"), "cross_clause"),
    ("The board refused to approve the merger.",
     ("board", "approve", "merger"), "cross_clause"),
]


class TestNonAssertionsAreSuppressed:
    """Every case v1 turned into a false fact yields nothing in v2, and
    the report names the reason."""

    @pytest.mark.parametrize("text,v1_triple,reason", SUPPRESSED)
    def test_no_triple_and_reason_counted(self, text, v1_triple, reason) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert v1_triple not in triples
        assert triples == []
        assert report["suppressed"][reason] == 1
        assert sum(report["suppressed"].values()) == 1
        assert report["extracted"] == 0


class TestSingleClauseRule:
    """The triple comes from one main-clause predicate, never stitched."""

    @pytest.mark.parametrize("text,expected,v1_triple", [
        # relative clause: v1 took subject + predicate from the relcl
        ("Alice owns the house where Bob grew up.",
         ("alice", "own", "house"), ("bob", "grow", "house")),
        ("Alice bought the car that Bob repaired.",
         ("alice", "buy", "car"), ("bob", "repair", "that")),
        # subordinate clause: v1 returned the subordinate predication
        ("The doctor prescribed the drug after the patient reported pain.",
         ("doctor", "prescribe", "drug"), ("patient", "report", "pain")),
        ("The company hired Bob because the founder trusted him.",
         ("company", "hire", "bob"), ("founder", "trust", "he")),
        # purpose clause: v1 returned the intended effect
        ("The city built a bridge in order to reduce traffic.",
         ("city", "build", "bridge"), ("city", "reduce", "traffic")),
    ])
    def test_main_clause_wins(self, text, expected, v1_triple) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert triples == [expected]
        assert v1_triple not in triples
        assert report["extracted"] == 1
        assert sum(report["suppressed"].values()) == 0

    @pytest.mark.parametrize("text,expected", [
        ("Although the market fell, the fund gained value.",
         ("fund", "gain", "value")),
        ("The scientist who discovered the virus won the prize.",
         ("scientist", "win", "prize")),
        # coordinated main clauses keep v1's tie-break (last complete one)
        ("Alice wrote the novel, and Bob painted the portrait.",
         ("bob", "paint", "portrait")),
        # shared-subject coordination inherits the subject
        ("Alice wrote and published the report.",
         ("alice", "publish", "report")),
        ("The editor reviewed and approved the manuscript.",
         ("editor", "approve", "manuscript")),
        # look-alikes that are not conditionals
        ("Alice remembers the day when Bob bought the house.",
         ("alice", "remember", "day")),
        ("The rope is as long as the table.", ("rope", "be", "long")),
        ("The company provided the data to the regulator.",
         ("company", "provide", "datum")),
    ])
    def test_positive_controls(self, text, expected) -> None:
        assert _sieve().extract_triplets(text) == [expected]

    def test_subject_not_inherited_into_passive_conj(self) -> None:
        # v1: (alice, give, award), Alice did not give the award.
        assert _sieve().extract_triplets(
            "Alice won the race and was given the award."
        ) == [("alice", "win", "race")]

    def test_subject_not_inherited_past_misattached_noun(self) -> None:
        # spaCy attaches "Bob" to "company" and leaves "runs" subjectless;
        # v1 emitted (alice, run, company).
        assert _sieve().extract_triplets(
            "Alice founded the company and Bob runs the company."
        ) == [("alice", "found", "company")]


MIXED = (
    "Alice likes cats. "
    "If the tenant pays rent late, the landlord charges a fee. "
    "Should we trust AI-generated content? "
    "Diamonds cannot cut through steel. "
    "Bob said that Alice stole the car. "
    "Bob owns a dog."
)


class TestAllSitesAgree:
    """extract_triplets, extract_with_provenance and
    extract_annotated_triplets share one per-sentence path. Before v2 the
    annotated path had its own copy of the v1 loop without passive
    handling or the POS fallback, so the first two cases returned []."""

    @pytest.mark.parametrize("text", [
        "Hamlet was written by Shakespeare.",
        "Dogs chase cats.",
        MIXED,
    ] + [case[0] for case in SUPPRESSED])
    def test_three_sites_same_triples(self, text) -> None:
        sieve = _sieve()
        plain = sieve.extract_triplets(text)
        prov = sorted({t for t, _ in sieve.extract_with_provenance(
            text, timestamp="2026-10-04T00:00:00+00:00")})
        annotated = sorted({
            (r["subject"], r["predicate"], r["object"])
            for r in sieve.extract_annotated_triplets(text)
        })
        assert plain == prov == annotated

    def test_annotated_handles_passive_and_fallback(self) -> None:
        sieve = _sieve()
        for text, expected in [
            ("Hamlet was written by Shakespeare.", ("shakespeare", "write", "hamlet")),
            ("Dogs chase cats.", ("dog", "chase", "cat")),
        ]:
            out = sieve.extract_annotated_triplets(text)
            assert [(r["subject"], r["predicate"], r["object"]) for r in out] == [expected]

    def test_mixed_document(self) -> None:
        assert _sieve().extract_triplets(MIXED) == [
            ("alice", "like", "cat"), ("bob", "own", "dog"),
        ]

    def test_annotated_keeps_certainty(self) -> None:
        out = _sieve().extract_annotated_triplets(MIXED)
        assert [(r["subject"], r["linguistic_certainty"]) for r in out] == [
            ("alice", 1.0), ("bob", 1.0),
        ]

    def test_provenance_records_name_v2(self) -> None:
        assert SIEVE_EXTRACTOR_ID == "sum.sieve:deterministic_v2"
        pairs = _sieve().extract_with_provenance(MIXED)
        assert len(pairs) == 2
        assert {rec.extractor_id for _, rec in pairs} == {SIEVE_EXTRACTOR_ID}
        assert [rec.text_excerpt for _, rec in pairs] == [
            "Alice likes cats.", "Bob owns a dog.",
        ]


class TestSuppressionReport:
    def test_counts_per_reason(self) -> None:
        assert _report(MIXED) == {
            "sentences": 6,
            "extracted": 2,
            "suppressed": {
                "negation": 1, "conditional": 1,
                "question": 1, "cross_clause": 1,
            },
        }

    def test_clean_text_reports_nothing_suppressed(self) -> None:
        assert _report("Alice likes cats. Bob owns a dog.") == {
            "sentences": 2,
            "extracted": 2,
            "suppressed": {
                "negation": 0, "conditional": 0,
                "question": 0, "cross_clause": 0,
            },
        }

    def test_reports_are_independent_per_call(self) -> None:
        first = _report(MIXED)
        second = _report("Alice likes cats.")
        assert first["sentences"] == 6
        assert second["sentences"] == 1


class TestSuppressedNotice:
    # No "The bundle omits ..." clause: on a zero-triple exit no bundle
    # is written, so the notice only says what the guard withheld.
    NOTICE = (
        "sum: 4 of 6 sentences were withheld by the clause guard (negation 1, "
        "conditional 1, question 1, cross-clause 1); see "
        "docs/PROOF_BOUNDARY.md section 2.1.\n"
    )

    def test_extract_triplets_writes_one_line(self) -> None:
        stream = io.StringIO()
        _sieve().extract_triplets(MIXED, suppressed_notice=stream)
        assert stream.getvalue() == self.NOTICE

    def test_extract_with_provenance_writes_one_line(self) -> None:
        stream = io.StringIO()
        _sieve().extract_with_provenance(MIXED, suppressed_notice=stream)
        assert stream.getvalue() == self.NOTICE

    def test_singular_wording(self) -> None:
        stream = io.StringIO()
        _sieve().extract_triplets(
            "Bob said that Alice stole the car.", suppressed_notice=stream
        )
        assert stream.getvalue() == (
            "sum: 1 of 1 sentence was withheld by the clause guard (cross-clause 1); "
            "see docs/PROOF_BOUNDARY.md section 2.1.\n"
        )

    def test_label_names_the_source(self) -> None:
        # attest-batch passes file=<path> so each line names its file,
        # in the same format as the batch error lines.
        from sum_engine_internal.algorithms.syntactic_sieve import (
            format_suppression_notice,
        )

        _, report = _sieve().extract_triplets_with_report(
            "Alice likes cats. If it rains, the match stops."
        )
        assert format_suppression_notice(report, label="file=a.txt") == (
            "sum: file=a.txt 1 of 2 sentences was withheld by the clause guard "
            "(conditional 1); see docs/PROOF_BOUNDARY.md section 2.1."
        )

    def test_state_for_corpus_passes_the_label(self) -> None:
        from sum_engine_internal.algorithms.chunked_corpus import state_for_corpus
        from sum_engine_internal.algorithms.semantic_arithmetic import (
            GodelStateAlgebra,
        )

        stream = io.StringIO()
        state_for_corpus(
            "Alice likes cats. If it rains, the match stops.",
            GodelStateAlgebra(),  # type: ignore[no-untyped-call]
            sieve=_sieve(), suppressed_notice=stream, notice_label="file=b.txt",
        )
        assert stream.getvalue().startswith("sum: file=b.txt 1 of 2 sentences")

    def test_nothing_written_when_nothing_suppressed(self) -> None:
        stream = io.StringIO()
        _sieve().extract_triplets("Alice likes cats.", suppressed_notice=stream)
        _sieve().extract_with_provenance(
            "Alice likes cats.", suppressed_notice=stream
        )
        assert stream.getvalue() == ""

    def test_default_is_silent(self, capsys) -> None:
        _sieve().extract_triplets(MIXED)
        _sieve().extract_with_provenance(MIXED)
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""


class TestNegativeControlCorpus:
    """T5 regression: the v2 corpus keeps v1 intact and the sieve has no
    unexpected verdict in the clause-guard modes."""

    def test_v2_contains_v1_unchanged_and_new_modes(self) -> None:
        v1 = json.loads(CORPUS_V1.read_text())
        v2 = json.loads(CORPUS_V2.read_text())
        assert v2["id"] == "seed_negative_control_v2"
        assert v2["schema"] == v1["schema"]
        assert v2["documents"][: len(v1["documents"])] == v1["documents"]
        modes = [d["expected_failure_mode"] for d in v2["documents"]]
        assert modes.count("conditional_assertion") >= 12
        assert modes.count("cross_clause_chimera") >= 12
        for doc in v2["documents"]:
            for triple in doc.get("allowed_triples", []):
                assert len(triple) == 3 and all(isinstance(x, str) for x in triple)

    def test_v2_clause_guard_modes_have_no_unexpected(self) -> None:
        from scripts.bench.runners.negative_control import run

        report = run(CORPUS_V2)
        by_mode = report["by_failure_mode"]
        assert by_mode["conditional_assertion"]["unexpected"] == 0
        assert by_mode["cross_clause_chimera"]["unexpected"] == 0
        assert by_mode["conditional_assertion"]["expected"] >= 12
        assert by_mode["cross_clause_chimera"]["expected"] >= 12

        baseline = json.loads(BASELINE_RECEIPT.read_text())["by_failure_mode"]
        for mode, counts in baseline.items():
            assert by_mode[mode]["unexpected"] <= counts["unexpected"], mode

        totals = report["summary"]["suppression"]
        per_doc = [r["observed"]["suppression"] for r in report["results"]]
        assert totals["sentences"] == sum(r["sentences"] for r in per_doc)
        assert totals["suppressed"]["conditional"] >= 12

    def test_runner_still_runs_v1(self) -> None:
        from scripts.bench.runners.negative_control import run

        report = run(CORPUS_V1, generated_at="2026-10-04T00:00:00.000Z")
        assert report["corpus_id"] == "seed_negative_control_v1"
        assert report["summary"]["total_documents"] == 20
        assert report["generated_at"] == "2026-10-04T00:00:00.000Z"
        assert report["extractor_id"] == SIEVE_EXTRACTOR_ID


# ─── Review fixes (2026-10-04) ────────────────────────────────────────
# Each case below produced a false triple before the fix; the comment
# gives what the first v2 draft emitted.


class TestSubjectInheritanceGuard:
    @pytest.mark.parametrize("text,expected", [
        # spaCy attaches "Bob" / "the tenant" to the first clause's object
        # and the appositive's closing comma sits right before the verb;
        # the draft inherited across it: (alice, buy, it), (landlord,
        # break, it).
        ("Alice founded the company and Bob, her brother, bought it.",
         [("alice", "found", "company")]),
        ("The landlord signed the lease and the tenant, Mr. Smith, broke it.",
         [("landlord", "sign", "lease")]),
    ])
    def test_no_inheritance_past_a_foreign_noun(self, text, expected) -> None:
        assert _sieve().extract_triplets(text) == expected

    @pytest.mark.parametrize("text,expected", [
        # the npadvmod ("last week", "the next day") was taken as the
        # subject: (last_week, fire, bob), (next_day, buy, car)
        ("The company hired Alice and fired Bob last week.",
         [("company", "fire", "bob")]),
        ("Alice met Bob, and the next day bought a car.",
         [("alice", "buy", "car")]),
        # and on a ROOT predicate: (last_year, buy, car)
        ("Alice bought a car last year.", [("alice", "buy", "car")]),
    ])
    def test_time_phrase_is_not_the_subject(self, text, expected) -> None:
        assert _sieve().extract_triplets(text) == expected


DISJUNCTIONS = [
    # draft: (alice, own, house), (tenant, pay, rent), (landlord, evict,
    # tenant), (alice, publish, report), (bank, keep, house)
    "Alice owns the house or Bob owns the car.",
    "The tenant pays rent or the landlord evicts the tenant.",
    "Either the tenant pays rent or the landlord evicts the tenant.",
    "Alice wrote or published the report.",
    "Pay the fee or the bank keeps the house.",
]

CONDITIONAL_MARKERS = [
    # draft: (seller, keep, deposit), (alice, buy, car), (seller, ship, good)
    "In the event the buyer defaults, the seller keeps the deposit.",
    "Alice will buy the car, assuming Bob agrees.",
    "Given that the buyer pays, the seller ships the goods.",
    # Fixed after the held-out review on the final draft (v1 and draft
    # both emitted the consequent, or a corrupted object for the last):
    # the hypothesis verb is tagged as a noun object, or the marker is a
    # legal subordinator the guard did not list.
    "Assuming the market recovers, the fund doubles its returns.",
    "The fund doubles its returns, assuming the market recovers.",
    "Supposing the buyer defaults, the bank seizes the house.",
    "Where the tenant fails to pay rent, the landlord terminates the lease.",
    "Once the buyer pays the deposit, the seller transfers the title.",
    "Subject to board approval, the company pays a dividend.",
    "The company pays a dividend, subject to board approval.",
]


class TestNonAssertedClauses:
    @pytest.mark.parametrize("text", DISJUNCTIONS + CONDITIONAL_MARKERS)
    def test_counted_as_conditional(self, text) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert triples == []
        assert report["suppressed"]["conditional"] == 1
        assert sum(report["suppressed"].values()) == 1

    def test_neither_nor_is_negation(self) -> None:
        # draft: (alice, own, car), the inverse of what the sentence says
        triples, report = _sieve().extract_triplets_with_report(
            "Neither Alice nor Bob owns the car."
        )
        assert triples == []
        assert report["suppressed"]["negation"] == 1

    @pytest.mark.parametrize("text,expected", [
        # "or" inside a relative clause or between numbers: main clause
        # still asserted
        ("The tenant who pays late or skips rent loses the deposit.",
         [("tenant", "lose", "deposit")]),
        ("Alice owns two or three cars.", [("alice", "own", "car")]),
        # look-alikes of the new markers
        ("Given the risk, the board approved the plan.",
         [("board", "approve", "plan")]),
        ("Alice is assuming the role of chief executive.",
         [("alice", "assume", "role")]),
        ("Alice owns the house where Bob grew up.",
         [("alice", "own", "house")]),
        ("Once a farmer, Bob now runs the bank.", [("bob", "run", "bank")]),
        ("The subject of the report is the merger.",
         [("subject", "be", "merger")]),
    ])
    def test_look_alikes_still_extracted(self, text, expected) -> None:
        assert _sieve().extract_triplets(text) == expected

    def test_participle_opener_is_a_recall_cost(self) -> None:
        # Precision over recall: spaCy gives "Assuming control of the
        # company" and "Assuming the market recovers" the same advcl
        # shape, so the asserted participle is withheld too.
        triples, report = _sieve().extract_triplets_with_report(
            "Assuming control of the company, Bob fired the board."
        )
        assert triples == []
        assert report["suppressed"]["conditional"] == 1


class TestPosFallbackSingleClause:
    @pytest.mark.parametrize("text", [
        # draft: (think, bob, lie), (see, bob, leave), (dogs, bark, bite)
        "I think Bob lies.",
        "She saw Bob leave.",
        "Dogs that bark bite.",
    ])
    def test_multi_clause_fallback_withheld(self, text) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert triples == []
        assert report["suppressed"]["cross_clause"] == 1

    def test_single_clause_fallback_kept(self) -> None:
        assert _sieve().extract_triplets("Dogs chase cats.") == [
            ("dog", "chase", "cat"),
        ]


TOME_TRIPLES = [
    ("alice", "own", "house"), ("bob", "sell", "car"),
    ("company", "hire", "bob"), ("court", "approve", "merger"),
    ("dog", "chase", "cat"), ("marie_curie", "discover", "polonium"),
    ("shakespeare", "write", "hamlet"),
]


class TestMarkdownHeadings:
    def test_canonical_tome_round_trips(self) -> None:
        # The draft recovered 2 of these 7 from their canonical tome
        # ("## Company\n\nThe company hire bob." parsed as one sentence).
        from sum_engine_internal.algorithms.semantic_arithmetic import (
            GodelStateAlgebra,
        )
        from sum_engine_internal.ensemble.tome_generator import (
            AutoregressiveTomeGenerator,
        )

        algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
        state = algebra.encode_chunk_state(TOME_TRIPLES)
        tome = AutoregressiveTomeGenerator(algebra).generate_canonical(state)
        triples, report = _sieve().extract_triplets_with_report(tome)
        assert triples == sorted(TOME_TRIPLES)
        assert sum(report["suppressed"].values()) == 0

    def test_heading_is_its_own_sentence_and_yields_nothing(self) -> None:
        text = "## Bench harness substrate\n\nAlice owns a house.\n# Future work"
        triples, report = _sieve().extract_triplets_with_report(text)
        # a three-word heading fed the POS fallback once it stood alone
        assert triples == [("alice", "own", "house")]
        assert report["sentences"] == 1

    def test_hash_without_space_is_not_a_heading(self) -> None:
        # CommonMark headings need "# "; "#1 ..." is ordinary text
        assert _sieve().extract_triplets("#1 Alice owns a house.") == [
            ("alice", "own", "house"),
        ]


class TestFrozenV1Extractor:
    """``extractor_id=SIEVE_EXTRACTOR_ID_V1`` replays v1 so results
    recorded under it (the research bench receipts) reproduce."""

    def test_replays_v1_outputs(self) -> None:
        from sum_engine_internal.algorithms.syntactic_sieve import (
            SIEVE_EXTRACTOR_ID_V1,
        )

        v1 = DeterministicSieve(  # type: ignore[no-untyped-call]
            extractor_id=SIEVE_EXTRACTOR_ID_V1,
        )
        for text, v1_triple, _ in SUPPRESSED:
            assert v1.extract_triplets(text) == [v1_triple], text
        assert v1.extract_triplets("Alice bought a car last year.") == [
            ("last_year", "buy", "car"),
        ]
        pairs = v1.extract_with_provenance(MIXED)
        assert {rec.extractor_id for _, rec in pairs} == {SIEVE_EXTRACTOR_ID_V1}
        _, report = v1.extract_triplets_with_report(MIXED)
        assert report["suppressed"] == {
            "negation": 1, "conditional": 0, "question": 0, "cross_clause": 0,
        }

    def test_unknown_extractor_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown sieve extractor_id"):
            DeterministicSieve(  # type: ignore[no-untyped-call]
                extractor_id="sum.sieve:deterministic_v9",
            )


# ─── Second review (2026-10-04) ───────────────────────────────────────
# Each case below produced a false or silently dropped result in the
# committed v2 (9dc0351); the comment gives what it emitted.


DETERMINER_NEGATION = [
    # 9dc0351 (and v1): the triple asserts the opposite of the sentence
    ("No student passed the exam.", ("student", "pass", "exam")),
    ("Nobody stole the car.", ("nobody", "steal", "car")),
    ("No one signed the contract.", ("one", "sign", "contract")),
    # spaCy splits "No-one" into No / - / one, none tagged det
    ("No-one signed the contract.", ("one", "sign", "contract")),
    ("None of the tenants paid rent.", ("none", "pay", "rent")),
    ("Nothing caused the outage.", ("nothing", "cause", "outage")),
    ("The company paid no dividends.", ("company", "pay", "dividend")),
    ("Noone saw the thief.", ("noone", "see", "thief")),
    ("The thief went nowhere near the vault.", ("thief", "go", "vault")),
]


class TestDeterminerAndPronounNegation:
    @pytest.mark.parametrize("text,old_triple", DETERMINER_NEGATION)
    def test_counted_as_negation(self, text, old_triple) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert old_triple not in triples
        assert triples == []
        assert report["suppressed"]["negation"] == 1
        assert sum(report["suppressed"].values()) == 1

    def test_interjection_no_is_not_negation(self) -> None:
        # "No," answers a question; the clause after it is asserted
        assert _sieve().extract_triplets("No, Alice owns the car.") == [
            ("alice", "own", "car"),
        ]

    @pytest.mark.parametrize("text", [
        # en_core_web_sm tags these "no" as dep neg, so v1's own rule
        # already withheld them; "no longer" inverts the clause.
        "Alice no longer owns the car.",
        "No doubt Alice owns the car.",
    ])
    def test_fixed_phrases_unchanged_from_v1(self, text) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert triples == []
        assert report["suppressed"]["negation"] == 1


ATTRIBUTED = [
    # 9dc0351: the reported content as fact
    ("According to the indictment, the CEO stole the pension fund.",
     ("ceo", "steal", "pension fund")),
    ("The CEO stole the pension fund, according to the indictment.",
     ("ceo", "steal", "pension fund")),
    ("The CEO, according to the indictment, stole the pension fund.",
     ("ceo", "steal", "pension fund")),
    ("According to police, the car was stolen by the suspect.",
     ("suspect", "steal", "car")),
    ("The suspect, police said, stole the car.", ("suspect", "steal", "car")),
    ("The CEO, the indictment alleges, stole the fund.",
     ("ceo", "steal", "fund")),
    ("The vaccine, she believes, causes autism.",
     ("vaccine", "cause", "autism")),
    ("As the company claims, the drug cures migraines.",
     ("drug", "cure", "migraine")),
    ("The suspect fled the scene, as police said.",
     ("suspect", "flee", "scene")),
    # a "by" agent is a reporter too
    ("The national interest is served by fewer warheads, as stated by the "
     "Department of Defense.", ("fewer_warhead", "serve", "national interest")),
]


class TestAttribution:
    """Reported content is withheld and counted as cross_clause, the same
    as "Bob said that Alice stole the car."."""

    @pytest.mark.parametrize("text,old_triple", ATTRIBUTED)
    def test_reported_content_withheld(self, text, old_triple) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert old_triple not in triples
        assert triples == []
        assert report["suppressed"]["cross_clause"] == 1
        assert sum(report["suppressed"].values()) == 1

    def test_parenthetical_reporting_already_withheld_stays_so(self) -> None:
        triples, report = _sieve().extract_triplets_with_report(
            "The drug, the company claims, cures migraines."
        )
        assert triples == []
        assert report["suppressed"]["cross_clause"] == 1

    @pytest.mark.parametrize("text,expected", [
        # a reporting verb as the main predicate with a plain object
        ("Alice said goodbye.", [("alice", "say", "goodbye")]),
        ("The witness told the jury the truth.", [("witness", "tell", "truth")]),
        ("Bob claims the prize.", [("bob", "claim", "prize")]),
        ("The company reported record profits.",
         [("company", "report", "record profit")]),
        ("The guide noted the time.", [("guide", "note", "time")]),
        # a parenthetical reporting verb inside a relative clause leaves
        # the main clause asserted
        ("The man who, police said, stole the car fled the city.",
         [("man", "flee", "city")]),
        # no reporter: a cross-reference, not a report (BillSum has
        # "as added by section ..." throughout)
        ("Section 5, as added by section 2, takes effect in May.",
         [("section", "take", "effect")]),
        ("As noted above, the sieve withholds the triple.",
         [("sieve", "withhold", "triple")]),
    ])
    def test_plain_reporting_verbs_still_extracted(self, text, expected) -> None:
        assert _sieve().extract_triplets(text) == expected


class TestDisjunctionDocstring:
    """``_is_clause_disjunction``'s docstring claimed that an "or" inside
    a relative clause never counts; the code counts it when a coordinated
    main clause follows. Checking the attachment was measured to admit
    stitched triples on legal text, so the docstring now states the
    code's behaviour and these cases pin each documented example."""

    @pytest.mark.parametrize("text", [
        "The firm, which sells tea or coffee, opened a shop and hired staff.",
        "Alice reads books or magazines and Bob writes poems.",
    ])
    def test_or_before_a_coordinated_main_clause_is_withheld(self, text) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert triples == []
        assert report["suppressed"]["conditional"] == 1

    @pytest.mark.parametrize("text,expected", [
        ("The tenant who pays late or skips rent loses the deposit.",
         [("tenant", "lose", "deposit")]),
        ("Alice reads books or magazines.", [("alice", "read", "book")]),
    ])
    def test_or_without_a_later_main_clause_does_not_count(self, text, expected) -> None:
        assert _sieve().extract_triplets(text) == expected


class TestHeadingLines:
    def test_indented_heading_keeps_the_prose(self) -> None:
        # CommonMark allows up to three spaces before "#". 9dc0351 glued
        # the heading to the prose and skipped both, uncounted.
        triples, report = _sieve().extract_triplets_with_report(
            " ## Background\nThe company hired Bob."
        )
        assert triples == [("company", "hire", "bob")]
        assert report["sentences"] == 1

    def test_hash_inside_a_line_is_not_a_heading(self) -> None:
        # 9dc0351 dropped the second sentence uncounted (sentences == 1)
        triples, report = _sieve().extract_triplets_with_report(
            "Alice owns the car. # Bob sells houses."
        )
        assert ("alice", "own", "car") in triples
        assert report["sentences"] == 2

    def test_four_space_indent_is_not_a_heading(self) -> None:
        # four spaces make a code block in CommonMark; the line is not
        # skipped as a heading, so it is counted
        _, report = _sieve().extract_triplets_with_report(
            "Alice owns the car.\n    # Bob sells houses."
        )
        assert report["sentences"] == 2

    @pytest.mark.parametrize("text,expected", [
        # a code block line glued to the prose stitched (bob, sell,
        # space) and lost Carol's sentence
        ("Intro text here.\n\n   ### Indented three\nBob sold the house.\n\n"
         "    #### four spaces\nCarol wrote the book.",
         [("bob", "sell", "house"), ("carol", "write", "book")]),
        ("Alice wrote the report.\n\n    # Bob reviewed the draft\n\n"
         "Carol approved the budget.",
         [("alice", "write", "report"), ("carol", "approve", "budget")]),
        # non-heading "#" lines glued to the prose gave (alice, own,
        # hashtag line) and (alice, write, hash)
        ("Alice owns the car.\n#hashtag line here\nBob sells houses.",
         [("bob", "sell", "house")]),
        ("Alice wrote the report.\n####### seven hashes\nBob sold the house.",
         [("bob", "sell", "house")]),
    ])
    def test_code_block_hash_line_does_not_join_the_prose(self, text, expected) -> None:
        triples = _sieve().extract_triplets(text)
        for triple in expected:
            assert triple in triples
        assert ("bob", "sell", "space") not in triples
        assert ("alice", "own", "hashtag line") not in triples
        assert ("alice", "write", "hash") not in triples

    def test_heading_component_is_linear(self) -> None:
        # 9dc0351 rebuilt Doc.text for every line: 40 s on this input.
        import time

        from sum_engine_internal.algorithms.syntactic_sieve import (
            _mark_heading_breaks,
        )

        text = "".join(
            f"## Part {i}\nThe firm hired {i} staff.\n" for i in range(2500)
        )
        doc = _sieve().nlp.make_doc(text)
        t0 = time.perf_counter()
        _mark_heading_breaks(doc)
        assert time.perf_counter() - t0 < 2.0
        assert sum(1 for t in doc if t.is_sent_start) == 5000

    def test_sieve_docs_serialise(self) -> None:
        # The first linear rewrite stored a frozenset in Doc.user_data,
        # which msgpack rejects: Doc.to_bytes and DocBin raised TypeError
        # on every v2 Doc (9dc0351 left user_data empty).
        from spacy.tokens import Doc, DocBin

        from sum_engine_internal.algorithms.syntactic_sieve import _is_heading

        nlp = _sieve().nlp
        docs = [nlp("## Title\nAlice owns the car."), nlp("Bob sells houses.")]
        for doc in docs:
            restored = Doc(nlp.vocab).from_bytes(doc.to_bytes())
            assert restored.user_data == doc.user_data
        # no heading, no footprint
        assert docs[1].user_data == {}
        data = DocBin(docs=docs, store_user_data=True).to_bytes()
        first = next(iter(DocBin().from_bytes(data).get_docs(nlp.vocab)))
        assert [_is_heading(s) for s in first.sents] == [True, False]

    def test_sieve_pipe_runs_in_worker_processes(self) -> None:
        # nlp.pipe(n_process>1) ships each Doc back with Doc.to_bytes;
        # the frozenset made it fail with spaCy error E871.
        import multiprocessing

        if multiprocessing.get_start_method() != "fork":
            pytest.skip("the sieve's spaCy pipeline is shared with workers by fork only")
        texts = ["## T\nAlice owns the car.", "Bob sells houses."]
        nlp = _sieve().nlp
        assert [d.to_json() for d in nlp.pipe(texts, n_process=2)] == [
            d.to_json() for d in nlp.pipe(texts)
        ]


ANNOTATED_V1_ROMAN = [
    ("who", "become", "first king", 0.85),
    ("roman_republic", "follow", "overthrow", 1.0),
    ("julius_caesar", "end", "dictatorship", 1.0),
    ("octavian", "become", "first roman emperor", 1.0),
    ("roman_empire", "reach", "territorial peak", 1.0),
    ("eastern_roman_empire", "continue", "last emperor", 1.0),
]


class TestFrozenV1Annotated:
    """``extract_annotated_triplets`` under the v1 id replays 0.11.1's own
    annotated slot loop (no passive handling, POS fallback or noise
    filter); 9dc0351 ran the shared v1 path instead."""

    def _v1(self) -> DeterministicSieve:
        from sum_engine_internal.algorithms.syntactic_sieve import (
            SIEVE_EXTRACTOR_ID_V1,
        )

        return DeterministicSieve(  # type: ignore[no-untyped-call]
            extractor_id=SIEVE_EXTRACTOR_ID_V1,
        )

    def test_replays_0_11_1_annotated_rows(self) -> None:
        corpus = json.loads(
            (REPO / "scripts/bench/corpora/seed_long_paragraphs.json").read_text()
        )
        text = next(
            d["text"] for d in corpus["documents"]
            if d["id"] == "doc_long_roman_empire"
        )
        rows = self._v1().extract_annotated_triplets(text)
        assert [
            (r["subject"], r["predicate"], r["object"], r["linguistic_certainty"])
            for r in rows
        ] == ANNOTATED_V1_ROMAN

    def test_passive_and_fallback_not_in_0_11_1_annotated(self) -> None:
        rows = self._v1().extract_annotated_triplets(
            "Alice owns the house where Bob grew up. Hamlet was written by "
            "Shakespeare. Dogs chase cats. Alice does not own a car."
        )
        assert [(r["subject"], r["predicate"], r["object"]) for r in rows] == [
            ("bob", "grow", "house"),
        ]


class TestNegativeControlSecondReview:
    def test_v2_corpus_has_negation_and_attribution_documents(self) -> None:
        v2 = json.loads(CORPUS_V2.read_text())
        by_id = {d["id"]: d for d in v2["documents"]}
        det_neg = [i for i in by_id if i.startswith("nonext_detneg_")]
        attrib = [i for i in by_id if i.startswith("chim_attrib_")]
        assert len(det_neg) >= 2 and len(attrib) >= 3
        for i in det_neg:
            assert by_id[i]["expected_failure_mode"] == "non_extractable_assertion"
            assert by_id[i]["extraction_should"] == "produce_zero_triples"
        for i in attrib:
            assert by_id[i]["expected_failure_mode"] == "cross_clause_chimera"
            assert by_id[i]["extraction_should"] == "produce_only_allowed_triples"
            assert by_id[i]["allowed_triples"] == []

    def test_new_documents_are_expected_under_v2(self) -> None:
        from scripts.bench.runners.negative_control import run

        report = run(CORPUS_V2)
        verdicts = {r["doc_id"]: r["verdict"] for r in report["results"]}
        new = [i for i in verdicts if i.startswith(("nonext_detneg_", "chim_attrib_"))]
        assert new and all(verdicts[i] == "expected" for i in new)

    def test_documented_recipe_reproduces_the_committed_receipt(
        self, tmp_path, monkeypatch,
    ) -> None:
        # The runner docstring's "Reproducible" command, with the date of
        # the committed receipt, must give its exact bytes (it lacked
        # --pretty, so it did not).
        import shlex

        import scripts.bench.runners.negative_control as nc

        doc = nc.__doc__ or ""
        recipe = doc.split("Reproducible", 1)[1].split("\n\n", 2)[1]
        argv = shlex.split(recipe.replace("\\\n", " "))
        assert argv[:3] == ["python", "-m", "scripts.bench.runners.negative_control"]
        receipt = REPO / "fixtures/bench_receipts/negative_control_2026-10-04.json"
        out = tmp_path / "receipt.json"
        args = [
            a.replace("<YYYY-MM-DD>", "2026-10-04") for a in argv[3:]
        ]
        args[args.index("--out") + 1] = str(out)
        monkeypatch.chdir(REPO)
        monkeypatch.setattr(sys, "argv", ["negative_control", *args])
        assert nc.main() == 1  # amb_coref_01/02/04 stay unexpected
        assert out.read_bytes() == receipt.read_bytes()


class TestInCaseConditional:
    """ "in case" before a subject noun or proper noun is a conditional.
    58b3636 exempted every "in case" followed by a noun (meant for "in
    case studies"), so these emitted their consequent or a stitched
    object. "case" is the object of "in" in a conditional and a compound
    modifier of the next noun otherwise."""

    @pytest.mark.parametrize("text,old_triple", [
        ("In case Bob defaults, the bank seizes the house.",
         ("bank", "seize", "house")),
        ("In case Acme defaults, the bank seizes the house.",
         ("bank", "seize", "house")),
        ("In case payment fails, the bank cancels the order.",
         ("bank", "cancel", "order")),
        ("In case borrowers default, the bank seizes the house.",
         ("bank", "seize", "house")),
        ("In case Microsoft wins the case, Google pays the damages.",
         ("google", "pay", "damage")),
        ("The guarantor pays the debt in case Bob defaults.",
         ("guarantor", "pay", "bob default")),
        ("The bank seizes the house in case Bob defaults.",
         ("bank", "seize", "house")),
    ])
    def test_withheld(self, text, old_triple) -> None:
        triples, report = _sieve().extract_triplets_with_report(text)
        assert old_triple not in triples
        assert triples == []
        assert report["suppressed"]["conditional"] == 1

    @pytest.mark.parametrize("text,expected", [
        ("In case studies, researchers found strong effects.",
         [("researcher", "find", "strong effect")]),
        ("In case law, courts follow precedent.",
         [("court", "follow", "precedent")]),
        ("As held in Case C-131/12, the court annulled the decision.",
         [("court", "annul", "decision")]),
    ])
    def test_compound_case_still_extracted(self, text, expected) -> None:
        assert _sieve().extract_triplets(text) == expected


class TestClauseGuardSpeed:
    """Input shapes on which the clause guard was super-linear while v1
    was not: 58b3636 built ``token.subtree`` for every coordinated
    predicate (36 KB took 102 s), rescanned the sentence for every "or"
    (95 KB: 24 s) and re-walked heads for every "provided" (a chain was
    cubic); a later draft walked heads from every noun before a
    coordinated verb. The parse is not timed here; the guard over the
    parsed sentence is."""

    @staticmethod
    def _one_sentence(text: str):
        doc = _sieve().nlp(text)
        sents = list(doc.sents)
        assert len(sents) == 1
        return sents[0]

    def test_coordinated_predicate_chain(self) -> None:
        import time

        from sum_engine_internal.algorithms.syntactic_sieve import (
            _guard_reason,
            _single_clause_triple,
        )

        sent = self._one_sentence(
            "Alice owns the house and " + "sells the car and " * 1500
            + "rents it."
        )
        t0 = time.perf_counter()
        _guard_reason(sent)
        triple, _ = _single_clause_triple(sent)
        assert time.perf_counter() - t0 < 2.0
        assert triple == ("Alice", "rent", "it")  # raw, before lowering

    def test_long_or_list(self) -> None:
        import time

        from sum_engine_internal.algorithms.syntactic_sieve import (
            _is_clause_disjunction,
        )

        sent = self._one_sentence(
            "Alice reads " + " or ".join(f"book{i}" for i in range(6000)) + "."
        )
        t0 = time.perf_counter()
        _is_clause_disjunction(sent)
        assert time.perf_counter() - t0 < 2.0

    def test_provided_chain(self) -> None:
        import time

        from sum_engine_internal.algorithms.syntactic_sieve import (
            _extract_sentence,
        )

        sent = self._one_sentence(
            "Alice owns the house and "
            + "sells the goods provided free and " * 1000 + "rents it."
        )
        t0 = time.perf_counter()
        _extract_sentence(sent)
        assert time.perf_counter() - t0 < 2.0

    def test_nested_phrase_before_coordinated_verb(self) -> None:
        import time

        from sum_engine_internal.algorithms.syntactic_sieve import (
            _extract_sentence,
        )

        sent = self._one_sentence(
            "Alice sold the car and, in the house "
            + "of the friend " * 4000 + "of Bob, bought it."
        )
        t0 = time.perf_counter()
        triple, _ = _extract_sentence(sent)
        assert time.perf_counter() - t0 < 2.0
        assert triple == ("alice", "buy", "it")
