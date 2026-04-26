"""Phase E.1 v0.8 — unit tests for salvage_partial_triplets.

The salvage helper is the third layer of the extractor's
LengthFinishReasonError defense: when the LLM's structured-output
response gets truncated mid-token by the 16384-token completion
ceiling, this helper walks the partial JSON and returns whatever
complete triplet objects appear before the truncation point.

Pure function. No I/O. Tests cover happy path, edge cases, and the
adversarial inputs that the helper must not mis-parse.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

from sum_engine_internal.ensemble.live_llm_adapter import (
    SemanticTriplet,
    salvage_partial_triplets,
)


COMPLETE_TRIPLET = (
    '{"subject":"alice","predicate":"likes","object":"cats",'
    '"source_span":"Alice likes cats.","certainty":"definite",'
    '"extraction_notes":""}'
)


class TestSalvagePartialTriplets:

    def test_recovers_complete_objects_before_truncation(self):
        # Two complete triplets, then a third truncated mid-string.
        partial = (
            '{"triplets":['
            + COMPLETE_TRIPLET
            + ","
            + COMPLETE_TRIPLET
            + ',{"subject":"bob","predicate":"owns","object":"do'
        )
        result = salvage_partial_triplets(partial)
        assert len(result) == 2
        assert all(isinstance(t, SemanticTriplet) for t in result)
        assert result[0].subject == "alice"
        assert result[1].subject == "alice"

    def test_returns_empty_when_no_complete_object_present(self):
        partial = '{"triplets":[{"subject":"truncate'
        assert salvage_partial_triplets(partial) == []

    def test_returns_empty_when_no_triplets_key(self):
        partial = '{"other_field":[{"subject":"alice"}]'
        assert salvage_partial_triplets(partial) == []

    def test_returns_empty_when_no_array_after_key(self):
        # Malformed input: triplets key with no array start.
        assert salvage_partial_triplets('{"triplets":') == []

    def test_handles_escaped_quotes_in_string_field(self):
        # source_span contains escaped quotes; salvage must not
        # break the quoted-string boundary detection.
        triplet_with_escapes = (
            '{"subject":"alice","predicate":"says","object":"hello",'
            '"source_span":"Alice said \\"hello\\" loudly.",'
            '"certainty":"definite","extraction_notes":""}'
        )
        partial = '{"triplets":[' + triplet_with_escapes + ',{"subj'
        result = salvage_partial_triplets(partial)
        assert len(result) == 1
        assert result[0].object_ == "hello"

    def test_handles_braces_inside_strings(self):
        # A `{` or `}` inside a string field must not affect
        # depth tracking — these are part of the string, not
        # structural braces.
        triplet_with_braces = (
            '{"subject":"compiler","predicate":"emits","object":"{warning}",'
            '"source_span":"emits {warning}","certainty":"definite",'
            '"extraction_notes":""}'
        )
        partial = '{"triplets":[' + triplet_with_braces + ',{"subj'
        result = salvage_partial_triplets(partial)
        assert len(result) == 1
        assert result[0].object_ == "{warning}"

    def test_skips_malformed_objects_without_failing(self):
        # First triplet is missing required fields; the helper
        # should drop it (json parses but Pydantic rejects) and
        # recover the second valid triplet.
        bad_triplet = '{"subject":"only_one_field"}'
        partial = (
            '{"triplets":['
            + bad_triplet
            + ","
            + COMPLETE_TRIPLET
            + ',{"trunc'
        )
        result = salvage_partial_triplets(partial)
        # Bad object dropped; one valid triplet recovered.
        assert len(result) == 1
        assert result[0].subject == "alice"

    def test_empty_input(self):
        assert salvage_partial_triplets("") == []

    def test_array_closes_cleanly(self):
        # Full valid response (not truncated) — helper still works
        # and recovers all objects up to the closing bracket.
        partial = (
            '{"triplets":[' + COMPLETE_TRIPLET + "," + COMPLETE_TRIPLET + "]}"
        )
        result = salvage_partial_triplets(partial)
        assert len(result) == 2
