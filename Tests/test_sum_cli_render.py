"""Contract tests for ``sum render`` — the inverse of ``sum attest``.

The CLI render verb closes the "tags ↔ tomes" symmetry promised by
the README. Local path is deterministic and actions only the density
slider; non-neutral length / formality / audience / perspective
require the LLM extrapolator and are routed through --use-worker.

These tests cover both paths without firing any real LLM calls — the
worker-path tests stub urlopen so the CLI's HTTP shape is verified
against a known-good fixture response without touching the network.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys

import pytest

from sum_cli.main import cmd_render


# ─── Fixtures ─────────────────────────────────────────────────────────


def _mint_bundle(triples: list[tuple[str, str, str]]) -> dict:
    """Construct a real CanonicalBundle from a triple set, using the
    same path ``sum attest`` uses internally. No CLI subprocess, no
    spaCy, no network — just the codec."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    algebra = GodelStateAlgebra()
    state = 1
    for s, p, o in triples:
        prime = algebra.get_or_mint_prime(s, p, o)
        state = math.lcm(state, prime)
    codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
    return codec.export_bundle(state, branch="render-test", title="Render Test Tome")


def _write_bundle(tmp_path, bundle: dict, name: str = "bundle.json"):
    path = tmp_path / name
    path.write_text(json.dumps(bundle))
    return path


def _run_render(
    bundle_path,
    *,
    density: float = 1.0,
    length: float = 0.5,
    formality: float = 0.5,
    audience: float = 0.5,
    perspective: float = 0.5,
    title: str = "Rendered Tome",
    output: str | None = None,
    use_worker: str | None = None,
    json_out: bool = False,
    pretty: bool = False,
    verbose: bool = False,
) -> tuple[int, str]:
    """Invoke cmd_render with a captured stdout. Returns (exit_code, stdout_text)."""
    args = argparse.Namespace(
        input=str(bundle_path),
        density=density,
        length=length,
        formality=formality,
        audience=audience,
        perspective=perspective,
        title=title,
        output=output,
        use_worker=use_worker,
        json=json_out,
        pretty=pretty,
        verbose=verbose,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = cmd_render(args)
    finally:
        sys.stdout = old
    return code, buf.getvalue()


# ─── 1. Round-trip integrity at full density ──────────────────────────


class TestRoundTripFullDensity:
    """The load-bearing claim: render → re-extract → state matches.
    At density=1.0 with all LLM axes neutral, the rendered tome must
    contain exactly the same axioms as the source bundle (in canonical
    order) so re-attesting reproduces the same state integer."""

    def test_render_preserves_state_integer(self, tmp_path):
        triples = [
            ("alice", "likes", "cats"),
            ("bob", "owns", "dog"),
            ("carol", "writes", "code"),
        ]
        bundle = _mint_bundle(triples)
        bundle_path = _write_bundle(tmp_path, bundle)

        code, tome_text = _run_render(bundle_path)
        assert code == 0
        assert tome_text  # non-empty

        # Parse the rendered tome's axiom lines and re-mint primes.
        # The state we recompute must equal the bundle's claimed state.
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

        algebra = GodelStateAlgebra()
        line_re = re.compile(r"^The (\S+) (\S+) (.+)\.$")
        state = 1
        axioms = 0
        for line in tome_text.splitlines():
            m = line_re.match(line.strip())
            if not m:
                continue
            state = math.lcm(state, algebra.get_or_mint_prime(*m.groups()))
            axioms += 1

        assert axioms == len(triples)
        assert str(state) == bundle["state_integer"]

    def test_render_emits_slider_header(self, tmp_path):
        bundle = _mint_bundle([("alice", "likes", "cats")])
        bundle_path = _write_bundle(tmp_path, bundle)

        code, tome_text = _run_render(bundle_path, density=0.7)
        assert code == 0
        # generate_controlled stamps the slider position into the tome
        # so a downstream consumer can see what was requested. We do
        # NOT assert on density-axis numeric equality at non-1.0 because
        # the density formatter rounds to 3 decimals.
        assert "@sliders:" in tome_text
        assert "density=0.700" in tome_text


# ─── 2. Density slider ────────────────────────────────────────────────


class TestDensity:

    def test_density_zero_emits_no_axiom_lines(self, tmp_path):
        bundle = _mint_bundle([
            ("a", "p", "o"),
            ("b", "p", "o"),
            ("c", "p", "o"),
        ])
        bundle_path = _write_bundle(tmp_path, bundle)

        code, tome_text = _run_render(bundle_path, density=0.0)
        assert code == 0
        # No `The S P O.` lines should appear; the controlled-tome
        # path emits a "no axioms survive" notice instead.
        line_re = re.compile(r"^The (\S+) (\S+) (.+)\.$")
        kept = [ln for ln in tome_text.splitlines() if line_re.match(ln.strip())]
        assert kept == []

    def test_density_keeps_lex_prefix(self, tmp_path):
        # Three axioms, lex-sorted as: a||p||o, b||p||o, c||p||o.
        # density=0.5 keeps floor(3 * 0.5) = 1 axiom — the
        # lexicographically-first one.
        bundle = _mint_bundle([
            ("c", "p", "o"),
            ("a", "p", "o"),
            ("b", "p", "o"),
        ])
        bundle_path = _write_bundle(tmp_path, bundle)

        code, tome_text = _run_render(bundle_path, density=0.5)
        assert code == 0
        line_re = re.compile(r"^The (\S+) (\S+) (.+)\.$")
        kept_subjects = [
            line_re.match(ln.strip()).group(1)
            for ln in tome_text.splitlines()
            if line_re.match(ln.strip())
        ]
        assert kept_subjects == ["a"]


# ─── 3. Slider validation ─────────────────────────────────────────────


class TestSliderValidation:

    def test_rejects_density_above_one(self, tmp_path):
        bundle = _mint_bundle([("a", "p", "o")])
        bundle_path = _write_bundle(tmp_path, bundle)
        code, _ = _run_render(bundle_path, density=1.5)
        assert code == 2

    def test_rejects_negative_length(self, tmp_path):
        bundle = _mint_bundle([("a", "p", "o")])
        bundle_path = _write_bundle(tmp_path, bundle)
        code, _ = _run_render(bundle_path, length=-0.1)
        assert code == 2

    def test_non_neutral_length_without_worker_errors(self, tmp_path, capsys):
        bundle = _mint_bundle([("a", "p", "o")])
        bundle_path = _write_bundle(tmp_path, bundle)
        code, _ = _run_render(bundle_path, length=0.9)
        assert code == 2
        err = capsys.readouterr().err
        assert "--use-worker" in err
        assert "length=0.9" in err

    def test_non_neutral_formality_without_worker_errors(self, tmp_path, capsys):
        bundle = _mint_bundle([("a", "p", "o")])
        bundle_path = _write_bundle(tmp_path, bundle)
        code, _ = _run_render(bundle_path, formality=0.1)
        assert code == 2
        err = capsys.readouterr().err
        assert "formality=0.1" in err


# ─── 4. Bundle malformed inputs ───────────────────────────────────────


class TestMalformedBundle:

    def test_invalid_json_returns_2(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {")
        code, _ = _run_render(path)
        assert code == 2

    def test_missing_canonical_tome_returns_2(self, tmp_path):
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps({"canonical_format_version": "1.0.0"}))
        code, _ = _run_render(path)
        assert code == 2

    def test_unsupported_canonical_format_returns_2(self, tmp_path):
        path = tmp_path / "future.json"
        path.write_text(json.dumps({
            "canonical_tome": "@canonical_version: 9.9.9\n# X\n\nThe a p o.",
            "canonical_format_version": "9.9.9",
        }))
        code, _ = _run_render(path)
        assert code == 2

    def test_zero_axioms_returns_3(self, tmp_path):
        # Valid bundle shape but the canonical tome has only header
        # lines — no parseable `The S P O.` lines. Render must refuse
        # rather than silently emit an empty tome.
        path = tmp_path / "empty_tome.json"
        path.write_text(json.dumps({
            "canonical_tome": "@canonical_version: 1.0.0\n# Title\n\n",
            "canonical_format_version": "1.0.0",
        }))
        code, _ = _run_render(path)
        assert code == 3


# ─── 5. Output file mode ──────────────────────────────────────────────


class TestOutputFile:

    def test_output_writes_file_and_suppresses_stdout(self, tmp_path):
        bundle = _mint_bundle([("a", "p", "o")])
        bundle_path = _write_bundle(tmp_path, bundle)
        out_path = tmp_path / "tome.md"

        code, stdout = _run_render(bundle_path, output=str(out_path))
        assert code == 0
        # When --output is set without --json, stdout is empty (the
        # tome was written to the file).
        assert stdout == ""
        assert out_path.exists()
        content = out_path.read_text()
        assert "The a p o." in content
        # Trailing newline is enforced by _emit_render_output for
        # convenient piping into shell tools.
        assert content.endswith("\n")

    def test_output_with_json_writes_both(self, tmp_path):
        bundle = _mint_bundle([("a", "p", "o")])
        bundle_path = _write_bundle(tmp_path, bundle)
        out_path = tmp_path / "tome.md"

        code, stdout = _run_render(
            bundle_path, output=str(out_path), json_out=True,
        )
        assert code == 0
        # File got the prose; stdout got the envelope.
        assert "The a p o." in out_path.read_text()
        envelope = json.loads(stdout)
        assert envelope["mode"] == "local-deterministic"
        assert "tome" in envelope
        assert envelope["sliders"]["density"] == 1.0


# ─── 6. JSON envelope on stdout ───────────────────────────────────────


class TestJsonEnvelope:

    def test_json_envelope_local_path(self, tmp_path):
        bundle = _mint_bundle([("a", "p", "o"), ("b", "q", "r")])
        bundle_path = _write_bundle(tmp_path, bundle)

        code, stdout = _run_render(bundle_path, json_out=True)
        assert code == 0
        envelope = json.loads(stdout)
        assert envelope["mode"] == "local-deterministic"
        assert envelope["axiom_count_input"] == 2
        assert envelope["sliders"] == {
            "density": 1.0,
            "length": 0.5,
            "formality": 0.5,
            "audience": 0.5,
            "perspective": 0.5,
        }
        assert "tome" in envelope and envelope["tome"]


# ─── 7. Worker path (stubbed urlopen) ─────────────────────────────────


class TestWorkerPath:
    """The worker path POSTs JSON and propagates the parsed response.
    We stub urllib.request.urlopen so no real HTTP call is made — the
    test fixture echoes the request shape and a known-good response so
    the CLI's wire contract is verified end-to-end."""

    def test_worker_response_propagates_receipt(self, tmp_path, monkeypatch):
        bundle = _mint_bundle([("alice", "graduated", "2012")])
        bundle_path = _write_bundle(tmp_path, bundle)

        captured: dict = {}

        class _FakeResponse:
            def __init__(self, body: bytes):
                self._body = body
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return self._body

        def fake_urlopen(req, timeout=60):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["body"] = json.loads(req.data.decode("utf-8"))
            response_body = json.dumps({
                "tome": "Alice graduated in 2012.",
                "render_id": "deadbeef00000000",
                "cache_status": "miss",
                "llm_calls_made": 1,
                "wall_clock_ms": 412,
                "quantized_sliders": {
                    "density": 1.0, "length": 0.7, "formality": 0.5,
                    "audience": 0.5, "perspective": 0.5,
                },
                "render_receipt": {
                    "schema": "sum.render_receipt.v1",
                    "kid": "sum-render-2026-04-27-1",
                    "payload": {
                        "render_id": "deadbeef00000000",
                        "model": "claude-haiku-4-5-20251001",
                    },
                    "jws": "eyJ...fake.jws...sig",
                },
            }).encode("utf-8")
            return _FakeResponse(response_body)

        import urllib.request as _ur
        monkeypatch.setattr(_ur, "urlopen", fake_urlopen)

        code, stdout = _run_render(
            bundle_path,
            length=0.7,
            use_worker="https://sum.ototao.com",
            json_out=True,
        )
        assert code == 0
        envelope = json.loads(stdout)
        assert envelope["mode"] == "worker"
        assert envelope["worker_url"] == "https://sum.ototao.com"
        assert envelope["tome"] == "Alice graduated in 2012."
        assert envelope["render_receipt"]["schema"] == "sum.render_receipt.v1"
        assert envelope["render_receipt"]["kid"] == "sum-render-2026-04-27-1"

        # Wire-shape contract: the CLI must POST to /api/render with
        # {triples: [[s,p,o]...], slider_position: {...}}.
        assert captured["method"] == "POST"
        assert captured["url"] == "https://sum.ototao.com/api/render"
        assert captured["body"]["triples"] == [["alice", "graduated", "2012"]]
        assert captured["body"]["slider_position"]["length"] == 0.7

    def test_worker_http_error_returns_1(self, tmp_path, monkeypatch):
        bundle = _mint_bundle([("alice", "graduated", "2012")])
        bundle_path = _write_bundle(tmp_path, bundle)

        import urllib.error
        import urllib.request as _ur

        def fake_urlopen(req, timeout=60):  # noqa: ARG001
            raise urllib.error.HTTPError(
                req.full_url, 502, "Bad Gateway", {},
                io.BytesIO(json.dumps({"error": "upstream LLM unavailable"}).encode()),
            )

        monkeypatch.setattr(_ur, "urlopen", fake_urlopen)

        code, _ = _run_render(
            bundle_path,
            length=0.7,
            use_worker="https://sum.ototao.com",
        )
        assert code == 1

    def test_worker_unreachable_returns_1(self, tmp_path, monkeypatch):
        bundle = _mint_bundle([("alice", "graduated", "2012")])
        bundle_path = _write_bundle(tmp_path, bundle)

        import urllib.error
        import urllib.request as _ur

        def fake_urlopen(req, timeout=60):  # noqa: ARG001
            raise urllib.error.URLError("network down")

        monkeypatch.setattr(_ur, "urlopen", fake_urlopen)

        code, _ = _run_render(
            bundle_path,
            length=0.7,
            use_worker="https://sum.ototao.com",
        )
        assert code == 1


# ─── 8. Stdin input ───────────────────────────────────────────────────


class TestStdinInput:

    def test_render_reads_bundle_from_stdin(self, tmp_path, monkeypatch):
        bundle = _mint_bundle([("a", "p", "o")])
        bundle_json = json.dumps(bundle)

        # Stdin redirect — mirrors how a shell pipeline would invoke us.
        monkeypatch.setattr("sys.stdin", io.StringIO(bundle_json))

        args = argparse.Namespace(
            input=None,  # default: read stdin
            density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5,
            title="Rendered Tome",
            output=None, use_worker=None,
            json=False, pretty=False, verbose=False,
        )
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            code = cmd_render(args)
        finally:
            sys.stdout = old
        assert code == 0
        assert "The a p o." in buf.getvalue()
