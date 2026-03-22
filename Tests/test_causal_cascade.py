"""
Causal Trigger Cascade Verification Tests

Verifies that the CausalTriggerMap correctly fires deductive inference
rules, cascades multi-hop inferences, and maintains idempotency.

Author: ototao
License: Apache License 2.0
"""

import math
import os
import tempfile
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.ensemble.causal_triggers import CausalTriggerMap


@pytest.fixture
def cascade_env():
    """Provide a fresh CausalTriggerMap with algebra and ledger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "cascade_test.db")
        algebra = GodelStateAlgebra()
        ledger = AkashicLedger(db_path=db_path)
        ctm = CausalTriggerMap(algebra, ledger)
        yield ctm, algebra, ledger


class TestCausalRuleFiring:

    @pytest.mark.asyncio
    async def test_simple_rule_fires(self, cascade_env):
        """A rule that matches fires and mints the inferred axiom."""
        ctm, algebra, _ = cascade_env

        # Rule: if X likes Y → X knows Y
        def condition(s, p, o, state, alg):
            return p == "likes"

        def inference(s, p, o, state, alg):
            return [(s, "knows", o)]

        ctm.register_rule(condition, inference)

        # Mint the triggering axiom
        p_likes = algebra.get_or_mint_prime("alice", "likes", "cats")
        state = p_likes

        result = await ctm.apply_cascade(state, ["alice||likes||cats"])

        # The inferred "alice knows cats" should now be in state
        p_knows = algebra.axiom_to_prime.get("alice||knows||cats")
        assert p_knows is not None
        assert result % p_knows == 0

    @pytest.mark.asyncio
    async def test_non_matching_rule_does_not_fire(self, cascade_env):
        """A rule that doesn't match leaves state unchanged."""
        ctm, algebra, _ = cascade_env

        def condition(s, p, o, state, alg):
            return p == "hates"  # Will not match "likes"

        def inference(s, p, o, state, alg):
            return [(s, "avoids", o)]

        ctm.register_rule(condition, inference)

        p = algebra.get_or_mint_prime("alice", "likes", "cats")
        state = p

        result = await ctm.apply_cascade(state, ["alice||likes||cats"])
        assert result == state  # No change

    @pytest.mark.asyncio
    async def test_no_rules_registered(self, cascade_env):
        """No rules → state passes through unchanged."""
        ctm, algebra, _ = cascade_env

        p = algebra.get_or_mint_prime("solo", "fact", "here")
        result = await ctm.apply_cascade(p, ["solo||fact||here"])
        assert result == p


class TestCascadeChaining:

    @pytest.mark.asyncio
    async def test_multi_hop_cascade(self, cascade_env):
        """A→B→C: two chained rules fire in sequence."""
        ctm, algebra, _ = cascade_env

        # Rule 1: likes → knows
        def cond1(s, p, o, state, alg):
            return p == "likes"

        def inf1(s, p, o, state, alg):
            return [(s, "knows", o)]

        # Rule 2: knows → trusts
        def cond2(s, p, o, state, alg):
            return p == "knows"

        def inf2(s, p, o, state, alg):
            return [(s, "trusts", o)]

        ctm.register_rule(cond1, inf1)
        ctm.register_rule(cond2, inf2)

        p_likes = algebra.get_or_mint_prime("alice", "likes", "bob")
        state = p_likes

        result = await ctm.apply_cascade(state, ["alice||likes||bob"])

        # Both inferred axioms should exist
        p_knows = algebra.axiom_to_prime.get("alice||knows||bob")
        p_trusts = algebra.axiom_to_prime.get("alice||trusts||bob")
        assert p_knows is not None
        assert p_trusts is not None
        assert result % p_knows == 0
        assert result % p_trusts == 0


class TestCascadeIdempotency:

    @pytest.mark.asyncio
    async def test_duplicate_cascade_idempotent(self, cascade_env):
        """Running the same cascade twice produces the same state."""
        ctm, algebra, _ = cascade_env

        def condition(s, p, o, state, alg):
            return p == "is"

        def inference(s, p, o, state, alg):
            return [(o, "includes", s)]

        ctm.register_rule(condition, inference)

        p = algebra.get_or_mint_prime("cat", "is", "animal")
        state = p

        result1 = await ctm.apply_cascade(state, ["cat||is||animal"])
        result2 = await ctm.apply_cascade(result1, ["cat||is||animal"])

        assert result1 == result2, "LCM idempotency guarantees same state"

    @pytest.mark.asyncio
    async def test_self_referential_rule_terminates(self, cascade_env):
        """A rule that could produce a cycle terminates (visited set)."""
        ctm, algebra, _ = cascade_env

        # Rule: X is Y → Y is X (could cycle infinitely)
        def condition(s, p, o, state, alg):
            return p == "is"

        def inference(s, p, o, state, alg):
            return [(o, "is", s)]

        ctm.register_rule(condition, inference)

        p = algebra.get_or_mint_prime("a", "is", "b")
        state = p

        # This should terminate, not loop forever
        result = await ctm.apply_cascade(state, ["a||is||b"])

        # Both a→b and b→a should exist
        p_reverse = algebra.axiom_to_prime.get("b||is||a")
        assert p_reverse is not None
        assert result % p_reverse == 0
