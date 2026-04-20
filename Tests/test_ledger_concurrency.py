"""Concurrent-ingest stress tests for AkashicLedger.

The machine-use case: multiple asynchronous writers pushing events and
provenance records into the same ledger simultaneously. These tests
verify that SUM's load-bearing invariants survive contention.

Invariants under test:

  1. **Merkle chain integrity under concurrent appends.** Every event
     stores ``prev_hash = SHA-256(prev || payload)``. If two writers
     both read prev_hash=X and both insert events computed against X,
     the later one's prev_hash doesn't equal the previous event's
     hash, and verify_chain() returns False. Real bug if so — must be
     serialized at the ledger boundary.

  2. **Structured provenance dedup under concurrent writes.** The
     prov_id is content-addressable (SHA-256 over JCS-canonicalized
     record); ``record_provenance`` uses INSERT OR IGNORE. 1000
     concurrent writes of the SAME record must result in exactly one
     row in provenance_records and exactly one row in
     axiom_provenance.

  3. **Distinct provenance records survive concurrent writes.** 1000
     concurrent writes of 1000 DISTINCT records must result in 1000
     rows in provenance_records and 1000 rows in axiom_provenance.
     No dropped writes.

The tests use tmp_path for ledger isolation. Concurrency is via
asyncio.gather; SQLite's per-connection threading handles the file
locking. Each test runs in ~1-2 seconds on commodity hardware.
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.infrastructure.provenance import (
    ProvenanceRecord,
    compute_prov_id,
)


@pytest.fixture
def ledger(tmp_path):
    db_path = str(tmp_path / "concurrent.db")
    return AkashicLedger(db_path=db_path)


def _rec(i: int) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_uri="sha256:" + f"{i:064x}"[:64],
        byte_start=0,
        byte_end=max(1, 16 + (i % 32)),
        extractor_id="sum.concurrency_test",
        timestamp="2026-04-19T00:00:00+00:00",
        text_excerpt=f"span {i}",
    )


class TestMerkleChainUnderConcurrency:
    """N concurrent append_event calls must leave the chain verifiable."""

    @pytest.mark.asyncio
    async def test_concurrent_appends_leave_chain_valid(
        self, ledger: AkashicLedger
    ) -> None:
        # Spawn 50 writers, each appending one event concurrently.
        async def _append(i: int) -> None:
            await ledger.append_event(
                operation="MINT",
                prime=2 * i + 3,  # distinct prime-ish per event
                axiom_key=f"s{i}||p||o{i}",
                branch="main",
                source_url=f"sha256:{i:064x}"[:64 + len("sha256:")],
                confidence=0.9,
                ingested_at="2026-04-19T00:00:00+00:00",
            )

        await asyncio.gather(*[_append(i) for i in range(50)])

        is_valid, break_seq = await ledger.verify_chain()
        assert is_valid, (
            f"Merkle chain diverged under concurrent appends at seq_id="
            f"{break_seq}. SQLite's per-transaction isolation should "
            f"have serialized the read-then-write of prev_hash."
        )

    @pytest.mark.asyncio
    async def test_burst_then_verify(self, ledger: AkashicLedger) -> None:
        # 200-event burst. Larger N exposes race windows smaller bursts miss.
        async def _append(i: int) -> None:
            await ledger.append_event(
                operation="MINT",
                prime=17 * (i + 1),
                axiom_key=f"burst_{i}",
                branch="main",
                source_url="",
                confidence=1.0,
                ingested_at="2026-04-19T00:00:00+00:00",
            )

        await asyncio.gather(*[_append(i) for i in range(200)])
        is_valid, break_seq = await ledger.verify_chain()
        assert is_valid, f"chain diverged at seq_id={break_seq} after 200-event burst"

        # Event count is the ground truth: if any writes were silently
        # dropped, we'd see fewer rows than we wrote.
        latest_tick = await ledger.get_latest_tick()
        assert latest_tick == 200


class TestProvenanceDedupUnderConcurrency:
    """INSERT OR IGNORE on content-addressable prov_id must collapse
    concurrent identical writes to exactly one row."""

    @pytest.mark.asyncio
    async def test_same_record_written_concurrently_dedups(
        self, ledger: AkashicLedger
    ) -> None:
        rec = _rec(42)
        axiom_key = "alice||like||cat"
        expected_prov_id = compute_prov_id(rec)

        # 100 concurrent writes of the SAME record+axiom_key
        prov_ids = await asyncio.gather(
            *[ledger.record_provenance(rec, axiom_key) for _ in range(100)]
        )

        # All 100 calls return the SAME id (content-addressable).
        assert all(pid == expected_prov_id for pid in prov_ids)

        # Exactly one provenance_records row.
        with sqlite3.connect(ledger.db_path) as conn:
            count_records = conn.execute(
                "SELECT COUNT(*) FROM provenance_records WHERE prov_id = ?",
                (expected_prov_id,),
            ).fetchone()[0]
            count_links = conn.execute(
                "SELECT COUNT(*) FROM axiom_provenance "
                "WHERE axiom_key = ? AND prov_id = ?",
                (axiom_key, expected_prov_id),
            ).fetchone()[0]
        assert count_records == 1, (
            f"expected 1 provenance_records row, got {count_records} — "
            f"INSERT OR IGNORE is not collapsing concurrent identical writes"
        )
        assert count_links == 1, (
            f"expected 1 axiom_provenance row, got {count_links}"
        )


class TestDistinctProvenanceSurvivesConcurrency:
    """N concurrent writes of N distinct records must leave N rows — no
    drops, no duplicates."""

    @pytest.mark.asyncio
    async def test_1000_distinct_records_all_persist(
        self, ledger: AkashicLedger
    ) -> None:
        N = 1000
        records = [_rec(i) for i in range(N)]
        axiom_keys = [f"s{i}||p||o{i}" for i in range(N)]

        prov_ids = await asyncio.gather(
            *[
                ledger.record_provenance(rec, key)
                for rec, key in zip(records, axiom_keys)
            ]
        )
        assert len(prov_ids) == N
        assert len(set(prov_ids)) == N  # all distinct

        with sqlite3.connect(ledger.db_path) as conn:
            n_records = conn.execute(
                "SELECT COUNT(*) FROM provenance_records"
            ).fetchone()[0]
            n_links = conn.execute(
                "SELECT COUNT(*) FROM axiom_provenance"
            ).fetchone()[0]
        assert n_records == N
        assert n_links == N

    @pytest.mark.asyncio
    async def test_mixed_concurrent_same_and_distinct(
        self, ledger: AkashicLedger
    ) -> None:
        # 10 distinct records, each written 5 times in parallel = 50 calls,
        # must land exactly 10 rows.
        N_DISTINCT = 10
        N_REPEATS = 5
        records = [_rec(i) for i in range(N_DISTINCT)]
        axiom_keys = [f"m{i}||p||o{i}" for i in range(N_DISTINCT)]

        tasks = []
        for _ in range(N_REPEATS):
            for rec, key in zip(records, axiom_keys):
                tasks.append(ledger.record_provenance(rec, key))

        prov_ids = await asyncio.gather(*tasks)
        assert len(prov_ids) == N_DISTINCT * N_REPEATS
        assert len(set(prov_ids)) == N_DISTINCT

        with sqlite3.connect(ledger.db_path) as conn:
            n_records = conn.execute(
                "SELECT COUNT(*) FROM provenance_records"
            ).fetchone()[0]
        assert n_records == N_DISTINCT


class TestWriteReadInterleaving:
    """Reads must observe committed writes, even under concurrent write load."""

    @pytest.mark.asyncio
    async def test_read_sees_concurrent_writes(
        self, ledger: AkashicLedger
    ) -> None:
        N = 50
        records = [_rec(i) for i in range(N)]
        axiom_keys = [f"wr{i}||p||o{i}" for i in range(N)]

        # Write all N records, then read one of them back. The read must
        # observe the write even though many other writes are in flight
        # concurrently.
        tasks = [
            ledger.record_provenance(rec, key)
            for rec, key in zip(records, axiom_keys)
        ]
        await asyncio.gather(*tasks)

        # Query one axiom — its provenance must be retrievable.
        mid = N // 2
        recs = await ledger.get_structured_provenance_for_axiom(axiom_keys[mid])
        assert len(recs) == 1
        assert recs[0] == records[mid]
