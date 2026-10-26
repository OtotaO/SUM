"""The review page's meaning-receipt box must not claim more than the CLI.

The box verifies a receipt against a JWKS the visitor pastes, so a valid
signature identifies a key the visitor supplied, not an issuer. The bundled
sample is signed with the all-zero-seed test key, whose private key is
public; `python -m sum_verify --demo` warns about it, and the box must too.
The box also never replays the bound, so it must not show the payload's
`controlled` flag (sum_verify/_verdict.py shows it only after a replay).

The box is an ES module script, which the jsdom flow tests do not execute,
so these checks are static over the source text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DEMO = Path(__file__).resolve().parents[1] / "single_file_demo"


def _meaning_module() -> str:
    html = (_DEMO / "index.html").read_text(encoding="utf-8")
    start = html.index('from "./meaning_receipt_verifier.js"')
    return html[start:html.index("</script>", start)]


def test_box_does_not_call_the_signer_an_issuer_or_show_controlled():
    src = _meaning_module()
    assert "Issuer signed" not in src
    assert "p.controlled" not in src
    assert "does not identify the issuer" in src


def test_box_warns_on_published_test_keys():
    src = _meaning_module()
    assert 'import { keyWarning } from "./review_packet.js";' in src
    assert "keyWarning(jwks, res.kid)" in src


def test_bundled_sample_is_signed_with_a_listed_test_key():
    js = (_DEMO / "review_packet.js").read_text(encoding="utf-8")
    zero_seed = re.search(r"PUBLIC_TEST_KEY_X = '([^']+)'", js).group(1)
    listed = js[js.index("PUBLIC_TEST_KEYS_X"):js.index("]);", js.index("PUBLIC_TEST_KEYS_X"))]
    assert "[PUBLIC_TEST_KEY_X," in listed
    jwks = json.loads((_DEMO / "sample_meaning_jwks.json").read_text(encoding="utf-8"))
    assert [k["x"] for k in jwks["keys"]] == [zero_seed]
