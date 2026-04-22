"""End-to-end tests for M1: ProvenanceRecord flows from sieve through the
AkashicLedger and round-trips losslessly.

These tests are the kill-experiment for the M1 moonshot milestone. If the
full chain (text → sieve.extract_with_provenance → ledger.record_provenance
→ ledger.get_structured_provenance_for_axiom → excerpt validates against
source byte range) does not close here, M1 is not credible and the
provenance moonshot is deferred.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile

import pytest

from sum_engine_internal.algorithms.syntactic_sieve import (
    SIEVE_EXTRACTOR_ID,
    DeterministicSieve,
)
from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger
from sum_engine_internal.infrastructure.provenance import (
    EXCERPT_MAX_CHARS,
    PROVENANCE_SCHEMA_VERSION,
    InvalidProvenanceError,
    ProvenanceRecord,
    compute_prov_id,
    sha256_uri_for_text,
    validate_source_uri,
)


# ─── ProvenanceRecord invariants ─────────────────────────────────────


class TestProvenanceRecordInvariants:
    def test_happy_path(self) -> None:
        rec = ProvenanceRecord(
            source_uri="sha256:" + "a" * 64,
            byte_start=0,
            byte_end=17,
            extractor_id="sum.test",
            timestamp="2026-04-19T00:00:00+00:00",
            text_excerpt="Alice likes cats.",
        )
        assert rec.schema_version == PROVENANCE_SCHEMA_VERSION

    def test_byte_range_reversed_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="byte_end"):
            ProvenanceRecord(
                source_uri="sha256:" + "a" * 64,
                byte_start=10, byte_end=5,
                extractor_id="sum.test",
                timestamp="2026-04-19T00:00:00+00:00",
                text_excerpt="x",
            )

    def test_empty_range_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="byte_end"):
            ProvenanceRecord(
                source_uri="sha256:" + "a" * 64,
                byte_start=0, byte_end=0,
                extractor_id="sum.test",
                timestamp="2026-04-19T00:00:00+00:00",
                text_excerpt="",
            )

    def test_empty_extractor_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="extractor_id"):
            ProvenanceRecord(
                source_uri="sha256:" + "a" * 64,
                byte_start=0, byte_end=1,
                extractor_id="",
                timestamp="2026-04-19T00:00:00+00:00",
                text_excerpt="x",
            )

    def test_excerpt_truncation_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="text_excerpt"):
            ProvenanceRecord(
                source_uri="sha256:" + "a" * 64,
                byte_start=0, byte_end=1,
                extractor_id="sum.test",
                timestamp="2026-04-19T00:00:00+00:00",
                text_excerpt="x" * (EXCERPT_MAX_CHARS + 1),
            )


class TestSourceUriValidation:
    def test_sha256_accepted(self) -> None:
        validate_source_uri("sha256:" + "0" * 64)

    def test_sha256_short_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="64 lowercase hex"):
            validate_source_uri("sha256:abc")

    def test_sha256_uppercase_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="64 lowercase hex"):
            validate_source_uri("sha256:" + "A" * 64)

    def test_doi_accepted(self) -> None:
        validate_source_uri("doi:10.1234/xyz")

    def test_https_accepted(self) -> None:
        validate_source_uri("https://example.org/paper")

    def test_urn_sum_source_accepted(self) -> None:
        validate_source_uri("urn:sum:source:seed_v1")

    def test_bare_http_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="not supported"):
            validate_source_uri("http://example.com/page")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(InvalidProvenanceError, match="not supported"):
            validate_source_uri("file:///tmp/x.txt")


class TestProvIdContentAddressability:
    def _rec(self, **overrides: object) -> ProvenanceRecord:
        kwargs: dict[str, object] = {
            "source_uri": "sha256:" + "b" * 64,
            "byte_start": 0,
            "byte_end": 17,
            "extractor_id": "sum.test",
            "timestamp": "2026-04-19T00:00:00+00:00",
            "text_excerpt": "Alice likes cats.",
        }
        kwargs.update(overrides)
        return ProvenanceRecord(**kwargs)  # type: ignore[arg-type]

    def test_identical_records_same_prov_id(self) -> None:
        assert compute_prov_id(self._rec()) == compute_prov_id(self._rec())

    def test_different_byte_range_different_id(self) -> None:
        assert compute_prov_id(self._rec()) != compute_prov_id(
            self._rec(byte_end=18)
        )

    def test_different_extractor_different_id(self) -> None:
        assert compute_prov_id(self._rec()) != compute_prov_id(
            self._rec(extractor_id="sum.other")
        )

    def test_prov_id_format(self) -> None:
        pid = compute_prov_id(self._rec())
        assert pid.startswith("prov:")
        assert len(pid) == len("prov:") + 32


class TestSha256UriHelper:
    def test_same_text_same_uri(self) -> None:
        assert sha256_uri_for_text("hello") == sha256_uri_for_text("hello")

    def test_different_text_different_uri(self) -> None:
        assert sha256_uri_for_text("hello") != sha256_uri_for_text("world")

    def test_uri_matches_known_hash(self) -> None:
        expected = hashlib.sha256(b"hello").hexdigest()
        assert sha256_uri_for_text("hello") == f"sha256:{expected}"


# ─── Sieve provenance path ───────────────────────────────────────────


@pytest.fixture(scope="module")
def sieve() -> DeterministicSieve:
    return DeterministicSieve()  # type: ignore[no-untyped-call]


class TestSieveExtractWithProvenance:
    def test_single_sentence(self, sieve: DeterministicSieve) -> None:
        pairs = sieve.extract_with_provenance("Alice likes cats.")
        assert len(pairs) == 1
        triple, rec = pairs[0]
        assert triple == ("alice", "like", "cat")
        assert rec.extractor_id == SIEVE_EXTRACTOR_ID
        assert rec.source_uri == sha256_uri_for_text("Alice likes cats.")
        assert rec.byte_start == 0
        assert rec.byte_end == len("Alice likes cats.".encode("utf-8"))
        assert rec.text_excerpt == "Alice likes cats."

    def test_multi_sentence_byte_ranges_cover_their_sentences(
        self, sieve: DeterministicSieve
    ) -> None:
        text = "Alice likes cats. Bob owns a dog."
        pairs = sieve.extract_with_provenance(text)
        assert len(pairs) == 2
        text_bytes = text.encode("utf-8")
        for triple, rec in pairs:
            slice_ = text_bytes[rec.byte_start:rec.byte_end].decode("utf-8")
            assert slice_.strip() == rec.text_excerpt.strip()
            assert len(triple) == 3

    def test_negation_suppressed_in_provenance_path(
        self, sieve: DeterministicSieve
    ) -> None:
        pairs = sieve.extract_with_provenance(
            "Diamonds cannot cut through steel."
        )
        assert pairs == []

    def test_mixed_doc_only_positive_sentences_get_provenance(
        self, sieve: DeterministicSieve
    ) -> None:
        text = (
            "Alice likes cats. "
            "Diamonds cannot cut through steel. "
            "Bob owns a dog."
        )
        pairs = sieve.extract_with_provenance(text)
        assert len(pairs) == 2
        triples = {p[0] for p in pairs}
        assert triples == {
            ("alice", "like", "cat"),
            ("bob", "own", "dog"),
        }

    def test_custom_source_uri_used(
        self, sieve: DeterministicSieve
    ) -> None:
        pairs = sieve.extract_with_provenance(
            "Alice likes cats.",
            source_uri="urn:sum:source:test_fixture",
            timestamp="2026-04-19T00:00:00+00:00",
        )
        _, rec = pairs[0]
        assert rec.source_uri == "urn:sum:source:test_fixture"
        assert rec.timestamp == "2026-04-19T00:00:00+00:00"

    def test_utf8_byte_offsets_handle_multibyte(
        self, sieve: DeterministicSieve
    ) -> None:
        # 'café' is 5 bytes (é = 2 bytes in UTF-8) but 4 chars. If our byte
        # offset logic naively used char offsets, the range would be wrong.
        text = "Café likes cats."
        pairs = sieve.extract_with_provenance(text)
        # Regardless of whether the sieve extracts anything on this string,
        # the byte offsets on any returned record must round-trip via UTF-8.
        for _triple, rec in pairs:
            slice_ = text.encode("utf-8")[rec.byte_start:rec.byte_end]
            # Decoding must succeed — if byte_end lands mid-codepoint, this
            # raises UnicodeDecodeError and the test fails.
            slice_.decode("utf-8")

    def test_extract_triplets_unchanged_after_refactor(
        self, sieve: DeterministicSieve
    ) -> None:
        # Regression guard: the _extract_from_sent refactor must preserve
        # extract_triplets output bit-for-bit on every previously-green input.
        assert sieve.extract_triplets("Alice likes cats.") == [
            ("alice", "like", "cat")
        ]
        assert sieve.extract_triplets(
            "Marie Curie, a physicist, won Nobel Prizes."
        ) == [("marie_curie", "win", "nobel prizes")]
        assert sieve.extract_triplets(
            "Diamonds cannot cut through steel."
        ) == []


# ─── AkashicLedger structured-provenance persistence ─────────────────


@pytest.fixture
def ledger() -> AkashicLedger:  # type: ignore[misc]
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    ledger = AkashicLedger(db_path=path)
    yield ledger
    os.unlink(path)


def _rec(text_excerpt: str = "Alice likes cats.") -> ProvenanceRecord:
    return ProvenanceRecord(
        source_uri="sha256:" + "c" * 64,
        byte_start=0, byte_end=len(text_excerpt.encode("utf-8")),
        extractor_id="sum.test",
        timestamp="2026-04-19T00:00:00+00:00",
        text_excerpt=text_excerpt,
    )


class TestLedgerStructuredProvenance:
    def test_record_then_get(self, ledger: AkashicLedger) -> None:
        async def run() -> None:
            rec = _rec()
            prov_id = await ledger.record_provenance(rec, "alice||like||cat")
            assert prov_id.startswith("prov:")
            fetched = await ledger.get_provenance_record(prov_id)
            assert fetched == rec

        asyncio.run(run())

    def test_unknown_prov_id_returns_none(self, ledger: AkashicLedger) -> None:
        async def run() -> None:
            got = await ledger.get_provenance_record("prov:deadbeef")
            assert got is None

        asyncio.run(run())

    def test_dedup_on_duplicate_record(self, ledger: AkashicLedger) -> None:
        async def run() -> None:
            rec = _rec()
            id1 = await ledger.record_provenance(rec, "alice||like||cat")
            id2 = await ledger.record_provenance(rec, "alice||like||cat")
            assert id1 == id2
            # Only one row in the linking table too
            records = await ledger.get_structured_provenance_for_axiom(
                "alice||like||cat"
            )
            assert len(records) == 1

        asyncio.run(run())

    def test_multi_source_same_axiom(self, ledger: AkashicLedger) -> None:
        async def run() -> None:
            rec_a = _rec(text_excerpt="Alice likes cats.")
            rec_b = ProvenanceRecord(
                source_uri="sha256:" + "d" * 64,
                byte_start=0, byte_end=23,
                extractor_id="sum.test",
                timestamp="2026-04-19T01:00:00+00:00",
                text_excerpt="She really likes cats.",
            )
            await ledger.record_provenance(rec_a, "alice||like||cat")
            await ledger.record_provenance(rec_b, "alice||like||cat")
            records = await ledger.get_structured_provenance_for_axiom(
                "alice||like||cat"
            )
            assert len(records) == 2
            assert {r.source_uri for r in records} == {
                rec_a.source_uri, rec_b.source_uri,
            }

        asyncio.run(run())

    def test_unrelated_axiom_returns_empty(
        self, ledger: AkashicLedger
    ) -> None:
        async def run() -> None:
            await ledger.record_provenance(_rec(), "alice||like||cat")
            records = await ledger.get_structured_provenance_for_axiom(
                "bob||own||dog"
            )
            assert records == []

        asyncio.run(run())


# ─── End-to-end kill experiment ──────────────────────────────────────
# Text → sieve.extract_with_provenance → ledger.record_provenance →
# ledger.get_structured_provenance_for_axiom → excerpt matches source
# byte range. If this chain closes, M1 is credible.


class TestBatchRecordProvenance:
    """record_provenance_batch: N pairs in one transaction, semantically
    identical to N single calls but much faster under load."""

    def test_empty_batch_returns_empty(self, ledger: AkashicLedger) -> None:
        async def run() -> None:
            out = await ledger.record_provenance_batch([])
            assert out == []

        asyncio.run(run())

    def test_batch_persists_all(self, ledger: AkashicLedger) -> None:
        async def run() -> None:
            recs = [
                ProvenanceRecord(
                    source_uri="sha256:" + f"{i:064x}"[:64],
                    byte_start=0, byte_end=17,
                    extractor_id="sum.test.batch",
                    timestamp="2026-04-19T00:00:00+00:00",
                    text_excerpt=f"span {i}",
                )
                for i in range(10)
            ]
            keys = [f"s{i}||p||o{i}" for i in range(10)]
            pairs = list(zip(recs, keys))

            prov_ids = await ledger.record_provenance_batch(pairs)
            assert len(prov_ids) == 10
            assert len(set(prov_ids)) == 10

            # Every record retrievable by axiom_key
            for rec, key in pairs:
                back = await ledger.get_structured_provenance_for_axiom(key)
                assert back == [rec]

        asyncio.run(run())

    def test_batch_dedup_on_duplicate_pairs(
        self, ledger: AkashicLedger
    ) -> None:
        async def run() -> None:
            rec = ProvenanceRecord(
                source_uri="sha256:" + "e" * 64,
                byte_start=0, byte_end=17,
                extractor_id="sum.test.batch",
                timestamp="2026-04-19T00:00:00+00:00",
                text_excerpt="span",
            )
            key = "dup||p||o"

            # Same pair three times in the same batch.
            ids = await ledger.record_provenance_batch(
                [(rec, key), (rec, key), (rec, key)]
            )
            assert ids[0] == ids[1] == ids[2]

            recs = await ledger.get_structured_provenance_for_axiom(key)
            assert len(recs) == 1  # INSERT OR IGNORE collapsed

        asyncio.run(run())

    def test_batch_matches_single_call_semantics(
        self, ledger: AkashicLedger
    ) -> None:
        async def run() -> None:
            rec = ProvenanceRecord(
                source_uri="sha256:" + "f" * 64,
                byte_start=0, byte_end=17,
                extractor_id="sum.test.batch",
                timestamp="2026-04-19T00:00:00+00:00",
                text_excerpt="span",
            )
            key = "equiv||p||o"
            single_id = await ledger.record_provenance(rec, key)
            [batch_id] = await ledger.record_provenance_batch([(rec, key)])
            assert single_id == batch_id

        asyncio.run(run())


class TestEndToEndKillExperiment:
    def test_full_chain_closes(
        self, sieve: DeterministicSieve, ledger: AkashicLedger
    ) -> None:
        text = (
            "Alice likes cats. "
            "Diamonds cannot cut through steel. "
            "Bob owns a dog."
        )

        async def run() -> None:
            pairs = sieve.extract_with_provenance(text)
            # Two positive sentences, one negated (suppressed).
            assert len(pairs) == 2

            # Persist every pair.
            prov_ids_by_axiom: dict[str, str] = {}
            for (s, p, o), rec in pairs:
                axiom_key = f"{s}||{p}||{o}"
                pid = await ledger.record_provenance(rec, axiom_key)
                prov_ids_by_axiom[axiom_key] = pid

            # Retrieve by axiom; verify each round-trips.
            for axiom_key, expected_pid in prov_ids_by_axiom.items():
                recs = await ledger.get_structured_provenance_for_axiom(
                    axiom_key
                )
                assert len(recs) == 1
                rec = recs[0]

                # The excerpt MUST match the bytes in the source at the
                # recorded range — this is the third-party-verifiable
                # property the moonshot claims.
                slice_ = text.encode("utf-8")[rec.byte_start:rec.byte_end]
                assert slice_.decode("utf-8") == rec.text_excerpt
                assert rec.extractor_id == SIEVE_EXTRACTOR_ID
                assert compute_prov_id(rec) == expected_pid

        asyncio.run(run())
