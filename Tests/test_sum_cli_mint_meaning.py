"""`sum mint-meaning` — the ISSUER on-ramp (twin of `sum verify-meaning`).

Until this verb landed, anyone could VERIFY a meaning-risk receipt
(`python -m sum_verify --demo`) but issuance meant orchestrating the
research primitives by hand via `examples/issue_meaning_receipt.py` —
the "everyone verifies, nobody mints" gap two grounded adoption audits
named as the #1 blocker. This test pins the productized round-trip:
losses (or scored pairs) in → a signed `sum.meaning_risk_receipt.v1` +
public JWKS out → the mint SELF-VERIFIES through the same verifier
`sum verify-meaning` dispatches to → `cmd_verify_meaning` accepts the
written files as a third party would.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import io
import json
import stat
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("joserfc", reason="[receipt-verify] not installed")
pytest.importorskip("numpy", reason="[research] not installed")

from sum_cli.main import cmd_mint_meaning, cmd_verify_meaning  # noqa: E402


@contextmanager
def _cap():
    out, err = io.StringIO(), io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        yield out, err


def _mint_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """Namespace with the parser's defaults; override per test."""
    base = dict(
        source=[],
        rendering=[],
        pairs=None,
        losses=None,
        scorer="embedding",
        scorer_name=None,
        scorer_version="unversioned",
        loss_definition=None,
        corpus_id="test-corpus-v0",
        transform="summarize:test",
        delta=0.05,
        method="empirical_bernstein",
        alpha_target=None,
        ed25519_key=None,
        gen_key=str(tmp_path / "keys"),
        kid="test-issuer-key-1",
        out=str(tmp_path / "receipt.json"),
        jwks_out=None,
        losses_out=None,
        pretty=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _verify_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = dict(
        receipt=str(tmp_path / "receipt.json"),
        jwks=str(tmp_path / "keys" / "jwks.json"),
        losses=None,
        group_ids=None,
        pretty=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_losses(tmp_path: Path, losses) -> str:
    p = tmp_path / "losses.json"
    p.write_text(json.dumps(losses))
    return str(p)


_BYO = dict(  # bring-your-own-proxy identity, required in --losses mode
    scorer_name="my-external-judge",
    scorer_version="1.0",
    loss_definition="entailment loss in [0,1]; 0 = judge detects no loss",
)

# 40 in-range fractional losses — large enough n that the empirical-
# Bernstein bound is far from vacuous (no warning expected).
_GOOD_LOSSES = [round(0.05 + 0.005 * (i % 7), 4) for i in range(40)]


class TestMintLossesMode:

    def test_mint_self_verifies_and_round_trips_through_verify_meaning(
        self, tmp_path
    ):
        args = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, _GOOD_LOSSES),
            alpha_target=0.5,
            **_BYO,
        )
        with _cap() as (out, err):
            rc = cmd_mint_meaning(args)
        assert rc == 0, err.getvalue()

        # verdict JSON on stdout, self-verified with replay
        verdict = json.loads(out.getvalue())
        assert verdict["verified"] is True
        assert verdict["replayed"] is True
        assert verdict["schema"] == "sum.meaning_risk_receipt.v1"
        assert verdict["scorer"] == "my-external-judge"
        assert verdict["controlled"] is True
        assert 0.0 < verdict["risk_upper_bound"] < 0.5

        # honesty surfaces on stderr
        narration = err.getvalue()
        assert "does NOT prove" in narration
        assert "NAMED PROXY" in narration
        assert "exchangeability" in narration
        assert "near-vacuous" not in narration  # n=40, tight bound

        # files written: receipt + public JWKS + 0600 private key
        assert (tmp_path / "receipt.json").exists()
        jwks = json.loads((tmp_path / "keys" / "jwks.json").read_text())
        assert jwks["keys"][0]["kty"] == "OKP"
        assert "d" not in jwks["keys"][0]

        # third-party round-trip: the CLI verify twin, with replay
        vargs = _verify_args(tmp_path, losses=args.losses)
        with _cap() as (vout, _):
            vrc = cmd_verify_meaning(vargs)
        assert vrc == 0
        v = json.loads(vout.getvalue())
        assert v["verified"] is True and v["replayed"] is True
        assert round(v["risk_upper_bound"] * 1e6) == round(
            verdict["risk_upper_bound"] * 1e6
        )

    def test_small_n_prints_vacuity_warning(self, tmp_path):
        args = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, [0.1, 0.2, 0.1]),
            **_BYO,
        )
        with _cap() as (out, err):
            rc = cmd_mint_meaning(args)
        assert rc == 0  # a vacuous bound is still VALID — warn, don't fail
        assert "near-vacuous" in err.getvalue()
        assert "n ≥ ~32" in err.getvalue()
        assert json.loads(out.getvalue())["verified"] is True

    def test_byo_losses_must_name_the_proxy(self, tmp_path):
        args = _mint_args(
            tmp_path, losses=_write_losses(tmp_path, _GOOD_LOSSES)
        )  # no scorer_name / loss_definition
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "--scorer-name" in err.getvalue()


class TestMintMalformedLosses:

    @pytest.mark.parametrize("bad", [
        [0.1, 1.5, 0.2],            # out of range
        [0.1, float("nan"), 0.2],   # non-finite (json emits a NaN literal)
        [0.1, True, 0.2],           # bool is not a loss
        {"foo": 1},                 # dict without a losses key
        "not-an-array",             # not a list at all
        [],                         # empty
    ])
    def test_malformed_losses_exit_2(self, tmp_path, bad):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        args = _mint_args(tmp_path, losses=str(p), **_BYO)
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "--losses" in err.getvalue()
        assert not (tmp_path / "receipt.json").exists()

    def test_unreadable_losses_file_exit_2(self, tmp_path):
        args = _mint_args(
            tmp_path, losses=str(tmp_path / "nope.json"), **_BYO
        )
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2

    def test_losses_and_pairs_mutually_exclusive(self, tmp_path):
        args = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, _GOOD_LOSSES),
            source=["a.txt"], rendering=["b.txt"],
            **_BYO,
        )
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "mutually exclusive" in err.getvalue()

    def test_no_input_at_all_exit_2(self, tmp_path):
        args = _mint_args(tmp_path)
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "nothing to mint" in err.getvalue()


class TestTamperedReceipt:

    def test_tampered_mint_fails_verify_meaning_rc1(self, tmp_path):
        largs = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, _GOOD_LOSSES),
            **_BYO,
        )
        with _cap():
            assert cmd_mint_meaning(largs) == 0
        rpath = tmp_path / "receipt.json"
        receipt = json.loads(rpath.read_text())
        receipt["payload"]["risk_upper_bound_micro"] = 1  # inflate the claim
        rpath.write_text(json.dumps(receipt))
        with _cap() as (vout, _):
            vrc = cmd_verify_meaning(_verify_args(tmp_path))
        assert vrc == 1
        assert json.loads(vout.getvalue())["verified"] is False


class TestKeys:

    def test_gen_key_writes_keypair_and_never_echoes_private_key(
        self, tmp_path
    ):
        args = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, _GOOD_LOSSES),
            **_BYO,
        )
        with _cap() as (out, err):
            rc = cmd_mint_meaning(args)
        assert rc == 0

        priv_path = tmp_path / "keys" / "private_jwk.json"
        private = json.loads(priv_path.read_text())
        assert private["kty"] == "OKP" and "d" in private
        # private key is owner-only on disk …
        mode = stat.S_IMODE(priv_path.stat().st_mode)
        assert mode == 0o600
        # … and its secret NEVER appears on stdout or stderr
        combined = out.getvalue() + err.getvalue()
        assert private["d"] not in combined
        # public half matches and carries no secret
        jwks = json.loads((tmp_path / "keys" / "jwks.json").read_text())
        assert jwks["keys"][0]["x"] == private["x"]
        assert "d" not in jwks["keys"][0]

    def test_gen_key_refuses_to_overwrite_existing_private_key(
        self, tmp_path
    ):
        keys = tmp_path / "keys"
        keys.mkdir()
        (keys / "private_jwk.json").write_text("{}")
        args = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, _GOOD_LOSSES),
            **_BYO,
        )
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "refusing to overwrite" in err.getvalue()

    def test_existing_pem_key_mints_and_verifies(self, tmp_path):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        pem_path = tmp_path / "sk.pem"
        pem_path.write_bytes(
            Ed25519PrivateKey.generate().private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        args = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, _GOOD_LOSSES),
            ed25519_key=str(pem_path),
            gen_key=None,
            jwks_out=str(tmp_path / "jwks.json"),
            **_BYO,
        )
        with _cap() as (out, err):
            rc = cmd_mint_meaning(args)
        assert rc == 0, err.getvalue()
        assert json.loads(out.getvalue())["verified"] is True
        # secret seed never echoed
        assert (
            json.loads((tmp_path / "jwks.json").read_text())["keys"][0].get("d")
            is None
        )
        vargs = _verify_args(tmp_path, jwks=str(tmp_path / "jwks.json"))
        with _cap():
            assert cmd_verify_meaning(vargs) == 0

    def test_key_flags_mutually_exclusive(self, tmp_path):
        args = _mint_args(
            tmp_path,
            losses=_write_losses(tmp_path, _GOOD_LOSSES),
            ed25519_key="whatever.pem",  # gen_key also set by default
            **_BYO,
        )
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "exactly one" in err.getvalue()


class _StubScorer:
    """Deterministic stand-in for the [judge] entailment scorers (which
    need torch); `sum mint-meaning` only calls .loss/.name/.version."""

    name = "stub-judge"
    version = "0"

    def loss(self, source: str, transform: str) -> float:
        # deterministic in the pair, always in [0, 1]
        return 0.0 if source == transform else 0.25


class TestMintPairsMode:

    def _stub_loader(self, name):
        assert name == "embedding"
        return _StubScorer(), None

    def test_pairs_jsonl_scored_minted_and_round_tripped(
        self, tmp_path, monkeypatch
    ):
        import sum_cli.main as cli

        monkeypatch.setattr(cli, "_load_meaning_scorer", self._stub_loader)
        pairs_path = tmp_path / "pairs.jsonl"
        pairs_path.write_text(
            "\n".join(
                json.dumps(
                    {"source": f"sentence {i}.", "rendering": f"sentence {i}."}
                    if i % 2 == 0
                    else {"source": f"sentence {i}.", "rendering": "rewritten."}
                )
                for i in range(40)
            )
        )
        args = _mint_args(tmp_path, pairs=str(pairs_path))
        with _cap() as (out, err):
            rc = cmd_mint_meaning(args)
        assert rc == 0, err.getvalue()
        verdict = json.loads(out.getvalue())
        assert verdict["verified"] is True and verdict["replayed"] is True
        assert verdict["scorer"] == "stub-judge"

        # pairs mode defaults the side-band losses next to the receipt,
        # so a third party can actually replay the bound
        losses_file = tmp_path / "receipt.json.losses.json"
        side = json.loads(losses_file.read_text())
        assert side["judge"] == "stub-judge"
        assert sorted(set(side["losses"])) == [0.0, 0.25]

        vargs = _verify_args(tmp_path, losses=str(losses_file))
        with _cap() as (vout, _):
            assert cmd_verify_meaning(vargs) == 0
        assert json.loads(vout.getvalue())["replayed"] is True

    def test_source_rendering_file_pairs(self, tmp_path, monkeypatch):
        import sum_cli.main as cli

        monkeypatch.setattr(cli, "_load_meaning_scorer", self._stub_loader)
        sources, renderings = [], []
        for i in range(4):
            s = tmp_path / f"s{i}.txt"
            r = tmp_path / f"r{i}.txt"
            s.write_text(f"source text {i}")
            r.write_text(f"source text {i}" if i else "changed")
            sources.append(str(s))
            renderings.append(str(r))
        args = _mint_args(tmp_path, source=sources, rendering=renderings)
        with _cap() as (out, err):
            rc = cmd_mint_meaning(args)
        assert rc == 0, err.getvalue()
        assert json.loads(out.getvalue())["verified"] is True
        assert "near-vacuous" in err.getvalue()  # n=4 is honestly loose

    def test_unbalanced_source_rendering_exit_2(self, tmp_path, monkeypatch):
        import sum_cli.main as cli

        monkeypatch.setattr(cli, "_load_meaning_scorer", self._stub_loader)
        s = tmp_path / "s.txt"
        s.write_text("text")
        args = _mint_args(tmp_path, source=[str(s), str(s)], rendering=[str(s)])
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "pairs" in err.getvalue()

    def test_malformed_pairs_jsonl_exit_2(self, tmp_path, monkeypatch):
        import sum_cli.main as cli

        monkeypatch.setattr(cli, "_load_meaning_scorer", self._stub_loader)
        pairs_path = tmp_path / "pairs.jsonl"
        pairs_path.write_text('{"source": "a"}\n')  # missing "rendering"
        args = _mint_args(tmp_path, pairs=str(pairs_path))
        with _cap() as (_, err):
            rc = cmd_mint_meaning(args)
        assert rc == 2
        assert "rendering" in err.getvalue()


class TestParserWiring:

    def test_subparser_registered_with_research_gated_defaults(self):
        from sum_cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "mint-meaning",
            "--losses", "l.json",
            "--corpus-id", "c", "--transform", "t",
            "--gen-key", "k", "--out", "r.json",
        ])
        assert args.func is cmd_mint_meaning
        assert args.method == "empirical_bernstein"
        assert args.delta == 0.05
        assert args.scorer == "embedding"
