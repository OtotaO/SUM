"""Guard for the Inspect faithfulness-scorer example's framework-agnostic core.

Tests ``score_meaning`` WITHOUT inspect_ai (the @scorer shim is gated behind its
import) and WITHOUT the [judge] extra (uses the offline lexical scorer) — so it
runs everywhere. The @scorer wiring itself is confirmed against a real inspect_ai
by the operator before any PR (per the example file's note), the same discipline
as the GEPA live-run check.
"""
from __future__ import annotations

import importlib.util
import os

# examples/ is not an installed package; load the example module by path.
_EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "examples", "inspect_meaning_scorer.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("inspect_meaning_scorer", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_meaning_core_offline_lexical():
    """The (value, explanation, metadata) mapping holds with the dependency-free
    lexical scorer — no inspect_ai, no torch."""
    from sum_engine_internal.research.meaning.meaning_loss import LexicalCoverageScorer

    mod = _load()
    source = "Alice manages the research team. The library closes at nine."
    completion = "Alice manages the research team."
    r = mod.score_meaning(source, completion, LexicalCoverageScorer())

    assert 0.0 <= r["value"] <= 1.0
    assert isinstance(r["explanation"], str) and r["explanation"].strip()
    md = r["metadata"]
    # The honesty caveat + the judge id + the raw loss always travel.
    assert "proxy_caveat" in md and "Spearman" in md["proxy_caveat"]
    assert "judge" in md
    assert abs(md["meaning_loss"] - (1.0 - r["value"])) < 1e-9


def test_module_imports_without_inspect_ai():
    """Importing the example must never hard-require inspect_ai; when it is
    absent, the @scorer symbol is None and the core is still usable."""
    mod = _load()
    # Either inspect_ai is installed (callable) or it is not (None) — but import
    # must succeed and `score_meaning` must exist either way.
    assert hasattr(mod, "score_meaning")
    assert mod.meaning_faithfulness is None or callable(mod.meaning_faithfulness)
