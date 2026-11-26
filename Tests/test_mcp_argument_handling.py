"""MCP tool arguments through FastMCP's own handling (server.call_tool, which
parses and validates the arguments the way a real client call does), bounded
bundle reconstruction, and the offline judge cache miss.

The other MCP tests call a tool's Python function directly, which skips
FastMCP's argument parsing, so none of them could see a string argument
changed on the way in.
"""
from __future__ import annotations

import asyncio
import json
import math
import random

import pytest

mcp = pytest.importorskip("mcp")

# attest runs the sieve extractor, which needs spaCy; the other tests do not.
try:
    import spacy  # noqa: F401
    _HAVE_SPACY = True
except ImportError:
    _HAVE_SPACY = False
needs_spacy = pytest.mark.skipif(not _HAVE_SPACY, reason="the sieve extractor needs spaCy")

_TEXT = "The printing press was invented by Johannes Gutenberg."


@pytest.fixture(scope="module")
def server():
    from sum_engine_internal.mcp_server import build_server
    return build_server()


def _call(server, name, args):
    """Call a tool through FastMCP, argument parsing and validation included."""
    res = asyncio.run(server.call_tool(name, args))
    if isinstance(res, tuple):  # (content, structured) on newer mcp
        res = res[1] if res[1] is not None else res[0]
    if isinstance(res, dict):
        return res.get("result", res)
    return json.loads(res[0].text)


@pytest.fixture(scope="module")
def unsigned_bundle(server):
    if not _HAVE_SPACY:
        pytest.skip("the sieve extractor needs spaCy")
    out = _call(server, "attest", {"text": _TEXT})
    if "error_class" in out:
        pytest.skip(f"attest unavailable: {out['errors']}")
    return out["bundle"]


# ---------------------------------------------------------------------------
# Optional text arguments reach the tool as sent
# ---------------------------------------------------------------------------


def test_a_signing_key_of_null_is_a_key_not_an_absent_key(server, unsigned_bundle):
    """FastMCP json.loads a string argument not annotated exactly `str`, so
    with `str | None` the key "null" arrived as None and the call ran keyless:
    an unsigned bundle came back ok with hmac "absent"."""
    for tool in ("verify", "verify_bind"):
        out = _call(server, tool, {"bundle": unsigned_bundle, "signing_key": "null"})
        if tool == "verify_bind":
            assert out.get("error_class") == "signature", out
            continue
        assert out["ok"] is False, out
        assert out["error_class"] == "signature"
        assert out["signatures"]["hmac"] == "missing"


@needs_spacy
def test_a_bundle_signed_with_the_key_null_verifies_with_it(server):
    signed = _call(server, "attest", {"text": _TEXT, "signing_key": "null"})
    assert "error_class" not in signed, signed
    assert "signature" in signed["bundle"], "attest with the key 'null' returned an unsigned bundle"
    out = _call(server, "verify", {"bundle": signed["bundle"], "signing_key": "null"})
    assert out["ok"] is True, out
    assert out["signatures"]["hmac"] == "verified"


@needs_spacy
def test_attest_refuses_an_empty_signing_key(server):
    """verify already refused an empty key (typically an unset secret);
    attest returned an unsigned bundle for it, with no warning."""
    for tool in ("attest", "attest_bind"):
        out = _call(server, tool, {"text": _TEXT, "signing_key": ""})
        assert out.get("error_class") == "schema", (tool, out)


def test_an_explicit_json_null_still_means_no_value(server, unsigned_bundle):
    out = _call(server, "verify", {"bundle": unsigned_bundle, "signing_key": None})
    assert out["ok"] is True, out
    assert out["signatures"]["hmac"] == "absent"


@needs_spacy
def test_text_that_looks_like_json_stays_text(server):
    """A title of "[1]" failed argument validation (FastMCP turned it into a
    list); it is now the title."""
    out = _call(server, "attest", {"text": _TEXT, "title": "[1]"})
    assert "error_class" not in out, out
    assert "[1]" in out["bundle"]["canonical_tome"]


def test_a_scorer_name_of_null_is_a_name(server):
    pytest.importorskip("scipy")
    from joserfc.jwk import OKPKey

    key = OKPKey.generate_key("Ed25519").as_dict(private=True)
    out = _call(server, "mint_meaning_receipt", {
        "private_jwk": key, "kid": "test-kid", "losses": [0.0] * 40 + [0.2] * 40,
        "scorer_name": "null", "corpus_id": "c", "transform": "t", "loss_definition": "d",
    })
    assert "error_class" not in out, out
    assert out["receipt"]["payload"]["scorer"].startswith("null"), out["receipt"]["payload"]["scorer"]


_OPTIONAL_TEXT = {
    "attest": ("title", "signing_key"), "attest_bind": ("title", "signing_key"),
    "verify": ("signing_key",), "verify_bind": ("signing_key",),
    "mint_meaning_receipt": ("scorer_name",), "mint_chain_receipt": ("scorer_name", "loss_definition"),
}


def test_optional_text_arguments_keep_their_published_schema(server):
    """The published input schema of each argument is still "string or null",
    as `str | None` produced, and FastMCP no longer pre-parses it."""
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    manager = server._tool_manager
    for tool, params in _OPTIONAL_TEXT.items():
        for param in params:
            schema = dict(tools[tool].inputSchema["properties"][param])
            schema.pop("title", None)
            assert schema == {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}, (tool, param, schema)
            field = manager.get_tool(tool).fn_metadata.arg_model.model_fields[param]
            assert field.annotation is str, (tool, param, field.annotation)


# ---------------------------------------------------------------------------
# Bundle reconstruction
# ---------------------------------------------------------------------------


def test_verify_counts_axioms_before_minting_any_prime(server, unsigned_bundle, monkeypatch):
    """The axiom cap was checked inside the minting loop, so an oversized tome
    minted and combined MAX_AXIOM_COUNT primes before it was refused."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.mcp_server import server as srv

    minted = []
    real = GodelStateAlgebra.get_or_mint_prime

    def counting(self, *args):
        minted.append(args)
        return real(self, *args)

    monkeypatch.setattr(GodelStateAlgebra, "get_or_mint_prime", counting)
    monkeypatch.setattr(srv, "MAX_AXIOM_COUNT", 50)
    bundle = dict(unsigned_bundle)
    bundle["canonical_tome"] = "\n".join(f"The s{i} p o{i}." for i in range(51))
    out = _call(server, "verify", {"bundle": bundle})
    assert out.get("error_class") == "input_too_large", out
    assert minted == [], f"{len(minted)} primes minted before the cap was applied"


def test_the_rebuilt_state_is_the_lcm_fold_it_replaces():
    """The state is the LCM of the axioms' primes; _state_of multiplies the
    distinct primes instead of folding math.lcm, which was quadratic."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.mcp_server.server import _state_of

    rng = random.Random(1126)
    for size in (0, 1, 2, 3, 17, 300):
        triples = [(f"s{rng.randrange(size + 1)}", f"p{rng.randrange(7)}", f"o{rng.randrange(size + 1)}")
                   for _ in range(size)]
        triples += triples[: size // 4]  # repeated axioms, as a tome can repeat a line
        algebra = GodelStateAlgebra()
        folded = 1
        for s, p, o in triples:
            folded = math.lcm(folded, algebra.get_or_mint_prime(s, p, o))
        state, _ = _state_of(triples)
        assert state == folded, (size, triples[:5])


def test_the_claimed_state_converts_as_int_does():
    """verify converts a long run of digits in halves (int() is quadratic in
    one GIL-holding call before Python 3.12); the value must be int()'s."""
    from sum_engine_internal.mcp_server.server import _decimal_to_int

    rng = random.Random(26)
    for length in (1, 4096, 4097, 9000, 50_001):
        text = "".join(rng.choice("0123456789") for _ in range(length))
        assert _decimal_to_int(text) == int(text), length
    for text in ("000123", "0" * 5000 + "7", " 12", "+12", "1_000", "-5"):
        assert _decimal_to_int(text) == int(text), text
    with pytest.raises(ValueError):
        _decimal_to_int("12a")


def test_a_state_mismatch_is_reported_without_writing_the_rebuilt_state_in_decimal(server, unsigned_bundle):
    bundle = dict(unsigned_bundle)
    bundle["state_integer"] = "7" + bundle["state_integer"][1:]
    out = _call(server, "verify", {"bundle": bundle})
    assert out.get("error_class") == "structural", out
    message = " ".join(out["errors"])
    assert "state integer mismatch: claimed 7" in message, message
    assert "-bit integer" in message, message


# ---------------------------------------------------------------------------
# The judge model missing from the offline cache
# ---------------------------------------------------------------------------


class _OfflineScorer:
    name = "nli:stub"
    version = "0"

    def explain(self, source, rendering):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files, "
                      "and couldn't find them in the cached files.")


@pytest.fixture
def offline_judge(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from sum_engine_internal.mcp_server import meaning_tools as mt

    monkeypatch.setattr(mt, "_need_research_extra", lambda: None)
    monkeypatch.setattr(mt, "_load_scorer", lambda scorer: (_OfflineScorer(), None))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return mt


def test_an_offline_cache_miss_names_the_missing_judge(server, offline_judge, monkeypatch):
    """The judge loads on first score; offline and uncached, transformers
    raises OSError, which reached callers as internal with no message."""
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    out = _call(server, "meaning_diff", {"source": "A cat sat.", "rendering": "A cat sat.", "scorer": "nli"})
    assert out.get("error_class") == "extractor_unavailable", out
    message = " ".join(out["errors"])
    assert "not in the local Hugging Face cache" in message, message
    assert "SUM_MCP_ALLOW_NETWORK=1" in message and "sum meaning-diff --scorer nli" in message, message


def test_an_oserror_with_the_judge_cached_stays_internal(server, offline_judge, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/cache/config.json")
    out = _call(server, "meaning_diff", {"source": "A cat sat.", "rendering": "A cat sat.", "scorer": "nli"})
    assert out.get("error_class") == "internal", out
