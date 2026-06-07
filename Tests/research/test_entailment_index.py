"""Adversarial-review E1: the ENTAILMENT-label selector must not invert.

`_entailment_index` is pure (no model, no torch), so this runs in the main
CI job — pinning the bug a naive `"entail" in label` substring match
caused: a binary RTE head `{0: "not_entailment", 1: "entailment"}` would
pick index 0 ("not_entailment") and silently invert every judgment.
"""
from __future__ import annotations

import pytest

from sum_engine_internal.research.meaning.local_judge import _entailment_index


def test_binary_rte_not_entailment_label_does_not_invert():
    # the bug: "entail" is a substring of "not_entailment"
    assert _entailment_index({0: "not_entailment", 1: "entailment"}) == 1


def test_three_way_entailment_first():
    assert _entailment_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0


def test_three_way_entailment_last():
    assert _entailment_index({0: "contradiction", 1: "neutral", 2: "entailment"}) == 2


def test_uppercase_labels():
    assert _entailment_index({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}) == 0


def test_non_entailment_negation_variants_excluded():
    # 'non-entailment' must also not be chosen
    assert _entailment_index({0: "non-entailment", 1: "entailment"}) == 1


def test_no_entailment_label_raises():
    with pytest.raises(ValueError, match="not an NLI model"):
        _entailment_index({0: "positive", 1: "negative"})
