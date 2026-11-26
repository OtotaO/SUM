"""Agentic-introspection contract for v0.3.0 CLI additions.

Three new subcommand clusters pinning the agent-friendly surface:

  sum ledger list|stats|head    introspect an AkashicLedger without
                                needing a prov_id in hand
  sum inspect                   structural read of a bundle, no crypto,
                                no state reconstruction
  sum schema {bundle|           JSON Schema (Draft 2020-12) for each
   provenance|credential}       output shape SUM emits, so agents can
                                validate programmatically

These tests assert both happy-path semantics and error paths, so a
future memory-less session has concrete examples of what each
subcommand should emit under which conditions.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import io
import json
import sys

import pytest


def _capture(fn, args) -> tuple[int, str]:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = fn(args)
    finally:
        sys.stdout = old
    return code, buf.getvalue()


def _attest_with_ledger(tmp_path, text: str) -> tuple[str, dict]:
    """Helper — mint a bundle WITH --ledger so subsequent ledger tests
    have actual records to introspect. Returns (db_path, bundle_dict)."""
    from sum_cli.main import cmd_attest

    in_path = tmp_path / "in.txt"
    in_path.write_text(text)
    db_path = tmp_path / "akashic.db"
    args = argparse.Namespace(
        input=str(in_path),
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="agentic test",
        signing_key=None,
        ed25519_key=None,
        ledger=str(db_path),
        pretty=False,
        verbose=False,
    )
    code, out = _capture(cmd_attest, args)
    assert code == 0
    return str(db_path), json.loads(out)


# ─── 1. sum ledger list ─────────────────────────────────────────────


class TestLedgerList:

    def test_list_emits_ndjson_rows(self, tmp_path):
        from sum_cli.main import cmd_ledger_list

        db, _ = _attest_with_ledger(tmp_path, "Alice likes cats. Bob owns a dog.")
        args = argparse.Namespace(
            db=db, axiom=None, since=None, limit=0, verbose=False
        )
        code, out = _capture(cmd_ledger_list, args)
        assert code == 0

        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) == 2
        for line in lines:
            rec = json.loads(line)
            # Contract: each NDJSON row has at least these fields.
            assert {"prov_id", "axiom_key", "source_uri",
                    "byte_start", "byte_end", "timestamp"} <= set(rec)
            assert rec["prov_id"].startswith("prov:")
            assert rec["byte_end"] > rec["byte_start"] >= 0

    def test_list_filter_by_axiom(self, tmp_path):
        from sum_cli.main import cmd_ledger_list

        db, _ = _attest_with_ledger(tmp_path, "Alice likes cats. Bob owns a dog.")
        args = argparse.Namespace(
            db=db, axiom="alice||like||cat", since=None, limit=0, verbose=False
        )
        code, out = _capture(cmd_ledger_list, args)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert code == 0 and len(lines) == 1
        assert json.loads(lines[0])["axiom_key"] == "alice||like||cat"

    def test_list_limit_caps_output(self, tmp_path):
        from sum_cli.main import cmd_ledger_list

        db, _ = _attest_with_ledger(tmp_path, "Alice likes cats. Bob owns a dog.")
        args = argparse.Namespace(
            db=db, axiom=None, since=None, limit=1, verbose=False
        )
        _, out = _capture(cmd_ledger_list, args)
        assert len([ln for ln in out.splitlines() if ln.strip()]) == 1

    def test_list_nonexistent_ledger_exits_2(self, tmp_path):
        from sum_cli.main import cmd_ledger_list

        args = argparse.Namespace(
            db=str(tmp_path / "no.db"), axiom=None, since=None, limit=0, verbose=False
        )
        with pytest.raises(SystemExit) as exc:
            cmd_ledger_list(args)
        assert exc.value.code == 2


# Seven records with fixed timestamps (four older, three at _NEW), so the
# prov_ids, which set the listing order, are the same on every run.
_OLD_1 = "2026-11-25T18:19:23.332787+00:00"
_OLD_2 = "2026-11-25T18:19:27.896308+00:00"
_NEW = "2026-11-25T18:19:55.844239+00:00"


def _fixed_ledger(tmp_path, stamps=(_OLD_1, _OLD_1, _OLD_2, _OLD_2, _NEW, _NEW, _NEW)):
    """A ledger holding one record per stamp; returns (db, [(prov_id, stamp)])."""
    import asyncio

    from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger
    from sum_engine_internal.infrastructure.provenance import ProvenanceRecord

    db = str(tmp_path / "fixed.db")
    ledger = AkashicLedger(db_path=db)

    async def _fill():
        out = []
        for i, stamp in enumerate(stamps):
            rec = ProvenanceRecord(
                source_uri="sha256:" + "ab" * 32, byte_start=10 * i,
                byte_end=10 * i + 9, extractor_id="sum.sieve:deterministic_v1",
                timestamp=stamp, text_excerpt=f"sentence {i}",
            )
            out.append((await ledger.record_provenance(rec, axiom_key=f"s{i}||p||o"), stamp))
        return out

    return db, asyncio.run(_fill())


def _ledger_list_cli(db, *extra) -> tuple[int, list, str]:
    """`sum ledger list --db DB ...` through main(): (rc, rows, stderr)."""
    from contextlib import redirect_stderr, redirect_stdout

    from sum_cli.main import main

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(["ledger", "list", "--db", db, *extra])
    rows = [json.loads(ln) for ln in out.getvalue().splitlines() if ln.strip()]
    return rc, rows, err.getvalue()


class TestLedgerListSince:
    """--since is compared as an instant, not as text, and --limit counts the
    rows that pass it (it was an SQL LIMIT applied before the --since filter)."""

    def test_limit_counts_rows_that_pass_since(self, tmp_path):
        db, recs = _fixed_ledger(tmp_path)
        # The first row in listing order is older than _NEW, so a LIMIT taken
        # before the --since filter fetches a row the filter then drops.
        assert min(recs)[1] != _NEW
        for limit, want in (("1", 1), ("2", 2), ("3", 3), ("0", 3), ("-1", 3)):
            rc, rows, _ = _ledger_list_cli(db, "--since", _NEW, f"--limit={limit}")
            assert rc == 0 and len(rows) == want, f"--limit {limit}: {len(rows)} rows"
            assert {r["timestamp"] for r in rows} == {_NEW}
        rc, rows, _ = _ledger_list_cli(db, "--limit=-1")  # <= 0 stays unlimited
        assert rc == 0 and len(rows) == 7

    @pytest.mark.parametrize("since", [
        "2026-11-25T18:19:50-0500",
        "2026-11-25T18:19:50+00",
        "20261125T181950Z",
        "2026-11-25T18:19:50,5Z",
        "2026-W48-3",
        "2026-11-25T24:00:00Z",
        "2026-02-30",
    ])
    def test_since_reads_the_same_forms_on_every_python(self, tmp_path, since):
        """Forms outside --since's grammar exit 2 on every Python. Some are
        read by datetime.fromisoformat from 3.11 on, which is why the parser
        does not use it; main compared them as text and exited 0."""
        db, _ = _fixed_ledger(tmp_path)
        rc, rows, err = _ledger_list_cli(db, "--since", since)
        assert rc == 2 and rows == [], (since, rc, rows)
        assert "is not a date or date-time it reads (for example" in err

    @pytest.mark.parametrize("since", [
        "\uff12\uff10\uff12\uff16-11-25",  # fullwidth digits
        "\u0662\u0660\u0662\u0666-11-25",  # Arabic-Indic digits
        "202\U00016ac6-11-25",  # a Tangsa digit: Unicode 14, so Python 3.11 on
    ])
    def test_since_takes_ascii_digits_only(self, tmp_path, since):
        """\\d matches any Unicode decimal digit, and that set grows with the
        Python version, so the grammar names 0 to 9."""
        db, _ = _fixed_ledger(tmp_path)
        rc, rows, err = _ledger_list_cli(db, "--since", since)
        assert rc == 2 and rows == [], (since, rc, rows)
        assert "is not a date or date-time it reads (for example" in err

    def test_since_is_compared_as_an_instant(self, tmp_path):
        db, _ = _fixed_ledger(tmp_path)
        for since in (
            "2026-11-25T18:19:50+00:00",
            "2026-11-25T13:19:50-05:00",  # the same instant, another offset
            "2026-11-25T18:19:50Z",
            "2026-11-25T18:19:50.5Z",  # one fraction digit: 3.10 fromisoformat refuses it as is
            "2026-11-25T18:19:55Z",  # 0.84 s before the _NEW records
            "2026-11-25T18:19:50",  # no offset: UTC
        ):
            rc, rows, _ = _ledger_list_cli(db, "--since", since)
            assert rc == 0, since
            assert sorted(r["timestamp"] for r in rows) == [_NEW] * 3, since
        rc, rows, _ = _ledger_list_cli(db, "--since", "2026-11-25")  # midnight UTC
        assert rc == 0 and len(rows) == 7

    def test_unreadable_since_is_a_usage_error(self, tmp_path):
        db, _ = _fixed_ledger(tmp_path)
        rc, rows, err = _ledger_list_cli(db, "--since", "yesterday")
        assert rc == 2 and rows == []
        assert "sum: --since 'yesterday' is not a date or date-time it reads" in err
        rc, rows, _ = _ledger_list_cli(db, "--since", "")  # empty is not "no --since"
        assert rc == 2 and rows == []

    def test_unreadable_stored_timestamp_is_left_out_under_since(self, tmp_path):
        db, _ = _fixed_ledger(tmp_path, stamps=(_OLD_1, _NEW, "not-a-time"))
        rc, rows, _ = _ledger_list_cli(db)
        assert rc == 0 and len(rows) == 3  # listed when there is no --since
        rc, rows, _ = _ledger_list_cli(db, "--since", "2026-11-25T18:19:50Z")
        assert rc == 0 and [r["timestamp"] for r in rows] == [_NEW]


# ─── 2. sum ledger stats ────────────────────────────────────────────


class TestLedgerStats:

    def test_stats_shape(self, tmp_path):
        from sum_cli.main import cmd_ledger_stats

        db, _ = _attest_with_ledger(tmp_path, "Alice likes cats. Bob owns a dog.")
        args = argparse.Namespace(db=db, pretty=False)
        code, out = _capture(cmd_ledger_stats, args)
        assert code == 0
        stats = json.loads(out)
        # Contract: agent-consumable summary with these keys present.
        expected = {
            "db_path", "provenance_records_total", "distinct_axiom_keys",
            "earliest_timestamp", "latest_timestamp", "chain_tip_hash", "branches",
        }
        assert expected <= set(stats)
        assert stats["provenance_records_total"] == 2
        assert stats["distinct_axiom_keys"] == 2
        assert stats["earliest_timestamp"] is not None

    def test_stats_orders_timestamps_as_instants(self, tmp_path):
        """13:00-05:00 is 18:00 UTC, later than 17:00Z; the text minimum named
        it the earliest. An unreadable timestamp is left out, as under --since."""
        from sum_cli.main import cmd_ledger_stats

        late, early = "2026-11-25T13:00:00-05:00", "2026-11-25T17:00:00+00:00"
        db, _ = _fixed_ledger(tmp_path, stamps=(late, early, "not a time"))
        code, out = _capture(cmd_ledger_stats, argparse.Namespace(db=db, pretty=False))
        stats = json.loads(out)
        assert code == 0 and stats["provenance_records_total"] == 3
        assert (stats["earliest_timestamp"], stats["latest_timestamp"]) == (early, late)


# ─── 3. sum ledger head ─────────────────────────────────────────────


class TestLedgerHead:

    def test_head_all_branches_shape(self, tmp_path):
        from sum_cli.main import cmd_ledger_head

        db, _ = _attest_with_ledger(tmp_path, "Alice likes cats.")
        args = argparse.Namespace(db=db, branch=None, pretty=False)
        code, out = _capture(cmd_ledger_head, args)
        assert code == 0
        payload = json.loads(out)
        assert "branches" in payload
        # Attest didn't write a branch head (that's a separate code path),
        # so the dict is empty — contract is that the KEY is present even
        # when no heads exist.

    def test_head_unknown_branch_exits_1(self, tmp_path):
        from sum_cli.main import cmd_ledger_head

        db, _ = _attest_with_ledger(tmp_path, "Alice likes cats.")
        args = argparse.Namespace(db=db, branch="nonexistent", pretty=False)
        code, _ = _capture(cmd_ledger_head, args)
        assert code == 1


# ─── 4. sum inspect ─────────────────────────────────────────────────


class TestInspect:

    def test_inspect_unsigned_bundle(self, tmp_path):
        from sum_cli.main import cmd_inspect

        _, bundle = _attest_with_ledger(tmp_path, "Alice likes cats.")
        path = tmp_path / "b.json"
        path.write_text(json.dumps(bundle))
        args = argparse.Namespace(input=str(path), pretty=False)
        code, out = _capture(cmd_inspect, args)
        assert code == 0
        info = json.loads(out)
        assert info["bundle_version"] == bundle["bundle_version"]
        assert info["axiom_count_claimed"] == 1
        assert info["axiom_count_parsed"] == 1
        assert info["signatures"] == {"hmac_present": False, "ed25519_present": False}
        assert info["state_integer_digits"] > 0
        # sum_cli sidecar surfaces prov_ids for agents.
        assert "prov_ids" in info["sum_cli"]

    def test_inspect_does_not_verify(self, tmp_path):
        """inspect should succeed on a bundle with a tampered tome —
        it's a structural read, not a signature-or-state check."""
        from sum_cli.main import cmd_inspect

        _, bundle = _attest_with_ledger(tmp_path, "Alice likes cats.")
        # Tamper the tome; inspect must still return 0 and report
        # the parsed count (which will diverge from claimed).
        bundle["canonical_tome"] += "\nThe evil injected axiom."
        path = tmp_path / "b.json"
        path.write_text(json.dumps(bundle))
        args = argparse.Namespace(input=str(path), pretty=False)
        code, out = _capture(cmd_inspect, args)
        assert code == 0
        info = json.loads(out)
        # One claimed, two parsed — inspect surfaces the divergence for
        # the agent to act on, rather than rejecting.
        assert info["axiom_count_claimed"] == 1
        assert info["axiom_count_parsed"] == 2

    def test_inspect_invalid_json_exits_2(self, tmp_path):
        from sum_cli.main import cmd_inspect

        path = tmp_path / "bad.json"
        path.write_text("{not json")
        args = argparse.Namespace(input=str(path), pretty=False)
        code, _ = _capture(cmd_inspect, args)
        assert code == 2


# ─── 5. sum schema ──────────────────────────────────────────────────


class TestSchema:

    @pytest.mark.parametrize("shape,required_title", [
        ("bundle", "CanonicalBundle"),
        ("provenance", "ProvenanceRecord"),
        ("credential", "VerifiableCredential 2.0 (eddsa-jcs-2022)"),
    ])
    def test_schema_emits_valid_json_schema(self, shape, required_title):
        from sum_cli.main import cmd_schema

        args = argparse.Namespace(shape=shape)
        code, out = _capture(cmd_schema, args)
        assert code == 0
        schema = json.loads(out)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == required_title
        assert schema["type"] == "object"
        assert "required" in schema
        assert "properties" in schema

    def test_schema_bundle_covers_actual_emitted_fields(self, tmp_path):
        """The bundle schema's required fields must be a subset of what
        sum attest actually emits — otherwise validating a real bundle
        against the schema would falsely fail."""
        from sum_cli.main import cmd_schema

        _, bundle = _attest_with_ledger(tmp_path, "Alice likes cats.")
        code, out = _capture(cmd_schema, argparse.Namespace(shape="bundle"))
        schema = json.loads(out)
        for req in schema["required"]:
            assert req in bundle, (
                f"Schema claims {req!r} is required, but sum attest "
                f"does not emit it. Update the schema or the emitter."
            )
