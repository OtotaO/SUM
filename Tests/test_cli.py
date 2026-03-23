"""
Phase 23 Tests — CLI Tool (sum_cli.py)

Tests all 6 CLI commands using Click's CliRunner:
    1. status — shows state summary
    2. ingest — inline triplets
    3. ingest — from JSON file
    4. ask — keyword query
    5. export — JSON format
    6. export — TSV format
    7. diff — compare two ticks
    8. provenance — show provenance chain
    9. status — on empty DB
    10. ask — no matches
    11. provenance — unknown axiom
"""

import os
import sys
import json
import math
import asyncio
import tempfile
import pytest

from click.testing import CliRunner

# Ensure project root on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts.sum_cli import cli


@pytest.fixture(autouse=True)
def _restore_event_loop():
    """Ensure a fresh event loop exists after each CLI test.

    CLI commands call asyncio.run() which closes the event loop.
    Without this, subsequent async tests fail with 'no current event loop'.
    """
    yield
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class TestCLIStatus:
    """Tests for the 'status' command."""

    def test_status_empty_db(self, tmp_path, monkeypatch):
        """Status on a fresh DB shows zero axioms."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "empty.db"))
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Active Axioms:" in result.output
        assert "Ledger Tick:" in result.output

    def test_status_with_data(self, tmp_path, monkeypatch):
        """Status after ingestion shows non-zero counts."""
        db = str(tmp_path / "status.db")
        monkeypatch.setenv("AKASHIC_DB", db)
        runner = CliRunner()
        # Ingest first
        runner.invoke(cli, ["ingest", "earth;orbits;sun"])
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Active Axioms:" in result.output


class TestCLIIngest:
    """Tests for the 'ingest' command."""

    def test_ingest_inline(self, tmp_path, monkeypatch):
        """Ingest inline triplets with semicolons and pipes."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "ingest.db"))
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "earth;orbits;sun | mars;has;moons"])
        assert result.exit_code == 0
        assert "Ingested 2 axioms" in result.output

    def test_ingest_json_file(self, tmp_path, monkeypatch):
        """Ingest triplets from a JSON file."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "ingest_json.db"))
        # Create a temp JSON file
        triplets = [["python", "is_a", "language"], ["rust", "enables", "safety"]]
        json_file = str(tmp_path / "input.json")
        with open(json_file, "w") as f:
            json.dump(triplets, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", json_file])
        assert result.exit_code == 0
        assert "Ingested 2 axioms" in result.output

    def test_ingest_with_source_url(self, tmp_path, monkeypatch):
        """Ingest with --source-url stores provenance."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "prov.db"))
        runner = CliRunner()
        result = runner.invoke(
            cli, ["ingest", "earth;orbits;sun", "--source-url", "https://nasa.gov"]
        )
        assert result.exit_code == 0
        assert "Ingested 1 axioms" in result.output

    def test_ingest_invalid(self, tmp_path, monkeypatch):
        """Invalid input produces error."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "invalid.db"))
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "not-a-valid-triplet"])
        assert result.exit_code != 0

    def test_ingest_dedup(self, tmp_path, monkeypatch):
        """Duplicate triplets are counted as duplicates."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "dedup.db"))
        runner = CliRunner()
        runner.invoke(cli, ["ingest", "earth;orbits;sun"])
        result = runner.invoke(cli, ["ingest", "earth;orbits;sun"])
        assert result.exit_code == 0
        assert "1 duplicates" in result.output


class TestCLIAsk:
    """Tests for the 'ask' command."""

    def test_ask_matches(self, tmp_path, monkeypatch):
        """Ask returns matching axioms."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "ask.db"))
        runner = CliRunner()
        runner.invoke(cli, ["ingest", "python;is_a;language | python;enables;scripting"])
        result = runner.invoke(cli, ["ask", "What is python?"])
        assert result.exit_code == 0
        assert "python" in result.output.lower()

    def test_ask_no_matches(self, tmp_path, monkeypatch):
        """Ask with no matching keywords."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "ask_none.db"))
        runner = CliRunner()
        runner.invoke(cli, ["ingest", "earth;orbits;sun"])
        result = runner.invoke(cli, ["ask", "What is quantum gravity?"])
        assert result.exit_code == 0
        assert "No matching" in result.output

    def test_ask_empty_db(self, tmp_path, monkeypatch):
        """Ask on empty DB."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "ask_empty.db"))
        runner = CliRunner()
        result = runner.invoke(cli, ["ask", "anything"])
        assert result.exit_code == 0
        assert "No knowledge" in result.output


class TestCLIExport:
    """Tests for the 'export' command."""

    def test_export_json(self, tmp_path, monkeypatch):
        """Export as JSON contains axioms."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "export.db"))
        runner = CliRunner()
        runner.invoke(cli, ["ingest", "earth;orbits;sun"])
        result = runner.invoke(cli, ["export", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["axiom_count"] >= 1
        assert len(data["axioms"]) >= 1

    def test_export_tsv(self, tmp_path, monkeypatch):
        """Export as TSV has header and data rows."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "export_tsv.db"))
        runner = CliRunner()
        runner.invoke(cli, ["ingest", "earth;orbits;sun"])
        result = runner.invoke(cli, ["export", "--format", "tsv"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert lines[0] == "subject\tpredicate\tobject"
        assert len(lines) >= 2

    def test_export_to_file(self, tmp_path, monkeypatch):
        """Export to file writes correctly."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "export_file.db"))
        out_file = str(tmp_path / "out.json")
        runner = CliRunner()
        runner.invoke(cli, ["ingest", "earth;orbits;sun"])
        result = runner.invoke(cli, ["export", "--format", "json", "-o", out_file])
        assert result.exit_code == 0
        assert os.path.exists(out_file)
        with open(out_file) as f:
            data = json.load(f)
        assert data["axiom_count"] >= 1


class TestCLIDiff:
    """Tests for the 'diff' command."""

    def test_diff_shows_additions(self, tmp_path, monkeypatch):
        """Diff between tick 0 and after ingest shows additions."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "diff.db"))
        runner = CliRunner()
        runner.invoke(cli, ["ingest", "earth;orbits;sun | mars;has;moons"])
        result = runner.invoke(cli, ["diff", "--tick-a", "0", "--tick-b", "999999"])
        assert result.exit_code == 0
        assert "Added" in result.output


class TestCLIProvenance:
    """Tests for the 'provenance' command."""

    def test_provenance_known(self, tmp_path, monkeypatch):
        """Provenance for a known axiom shows chain."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "prov.db"))
        runner = CliRunner()
        runner.invoke(
            cli, ["ingest", "earth;orbits;sun", "--source-url", "https://nasa.gov"]
        )
        # The axiom key will be canonicalized
        result = runner.invoke(cli, ["provenance", "earth||orbits||sun"])
        assert result.exit_code == 0
        assert "Provenance for:" in result.output

    def test_provenance_unknown(self, tmp_path, monkeypatch):
        """Provenance for unknown axiom shows not found."""
        monkeypatch.setenv("AKASHIC_DB", str(tmp_path / "prov_unknown.db"))
        runner = CliRunner()
        result = runner.invoke(cli, ["provenance", "unknown||pred||obj"])
        assert result.exit_code == 0
        assert "not found" in result.output
