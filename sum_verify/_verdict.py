"""Scope and replay fields shared by every meaning-family verdict surface.

``sum verify-meaning``, ``python -m sum_verify`` and the MCP
``verify_receipt`` tool print a small JSON verdict after a receipt verifies.
Each used to assemble it separately, and each showed the signed bound and the
``controlled`` flag whether or not the bound had been replayed. This module is
the single place that decides what a verdict may say:

- the payload's declared ``statistical_scope`` and ``sampling_status`` are
  echoed verbatim, or ``"not_declared"`` for receipts minted before those
  fields existed. A receipt without a sampling contract is a descriptive batch
  measurement; replay checks arithmetic, never sampling assumptions;
- ``n``, ``method`` and ``delta`` (``delta_micro / 1e6``) are echoed so the
  reader sees the arithmetic parameters next to the number;
- a bound is reported as ``risk_upper_bound`` (and ``controlled`` /
  ``controls_all`` as recorded) ONLY when it was replayed from supplied
  losses. Otherwise the value is ``issuer_asserted_risk_upper_bound`` and no
  pass/fail decision is shown, because nothing on this machine checked it;
- a chain's budget is always ``issuer_asserted_budget``: it is the sum of the
  per-hop bound values, and no verify surface replays per-hop losses (hop
  envelopes confirm signatures and mirrored fields, not bound arithmetic).
  Chains carry ``joint_delta`` (signed) and never a derived confidence; a
  receipt signed with the earlier unconditional ``budget_scope`` wording gets
  an unsigned ``budget_scope_note``.

Dependency-light: no imports beyond the standard library.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from typing import Any

MEANING_RISK_SCHEMA = "sum.meaning_risk_receipt.v1"
PERSPECTIVE_SCHEMA = "sum.perspective_risk_receipt.v1"
CHAIN_SCHEMA = "sum.chain_receipt.v1"
NOT_DECLARED = "not_declared"

# The budget_scope text every chain receipt minted before PR #531 signs
# (including the committed BillSum chain golden and builds of the v0.11.0 tag). It states
# the Bonferroni reading unconditionally. Kept verbatim so historical fixtures
# regenerate byte-for-byte (scripts/fixture_history.py re-exports it) and so a
# verdict can flag it; new receipts sign the conditional
# research.meaning.chain_receipt.BUDGET_SCOPE_STATEMENT instead.
HISTORICAL_BUDGET_SCOPE_STATEMENT = (
    "budget_micro bounds the SUM of per-hop expected proxy losses "
    "(Bonferroni union bound: joint confidence >= 1 - joint_delta). It "
    "does NOT bound the end-to-end loss: the proxy is a directed loss, "
    "not a metric, and no triangle inequality holds in either direction. "
    "The end_to_end leg, when present, is a separate DIRECT measurement "
    "over source-to-final pairs with its own replay anchor."
)

# Unsigned note added to a chain verdict whose signed budget_scope is the
# historical wording above.
HISTORICAL_BUDGET_SCOPE_NOTE = (
    "The signed budget_scope uses the earlier unconditional wording. "
    "The Bonferroni reading it states (the sum of per-hop "
    "expected proxy losses is at most the budget with probability >= "
    "1 - joint_delta) holds only if each hop's bound holds under that hop's "
    "own sampling assumptions (independent draws from its target "
    "distribution, a fixed policy, no calibration reuse); without them the "
    "budget is a descriptive sum of per-hop upper-bound values."
)

_CORRELATION = (
    "The bound is over a named proxy; vs human judgments the proxy "
    "correlated only modestly at summary level (Spearman rho = 0.267-0.291, "
    "pooled summary-level, on SummEval; NLI ~0.29 replicates on FRANK; the "
    "embedding judge is corpus-dependent, near zero on abstractive "
    "FRANK-XSum). Not a substitute for human review."
)


def proxy_caveat(replayed: bool) -> str:
    """The unsigned honesty line printed on every verified meaning-risk
    verdict. It names what actually ran: a signature check always, and the
    bound replay only when losses were supplied."""
    if replayed:
        what = (
            "verified=true is a cryptographic fact (signature + replayed "
            "bound arithmetic), not evidence meaning was preserved. "
        )
    else:
        what = (
            "verified=true is a cryptographic fact (signature only; the bound "
            "was NOT replayed, so it is the issuer's assertion), not evidence "
            "meaning was preserved. "
        )
    return what + _CORRELATION


def _micro(obj: Any, key: str) -> float | None:
    value = obj.get(key) if isinstance(obj, dict) else None
    if type(value) is not int:
        return None
    return value / 1_000_000


def _bound_key(replayed: bool) -> str:
    return "risk_upper_bound" if replayed else "issuer_asserted_risk_upper_bound"


def scope_fields(schema: Any, payload: Any, *, replayed: bool) -> dict[str, Any]:
    """Verdict fields for a verified ``payload`` of ``schema``.

    ``replayed`` must be True only when the bound (or every cohort bound, for a
    perspective receipt) was recomputed from supplied losses on this call.
    Render and transform receipts carry no statistical claim and get ``{}``.
    """
    if not isinstance(payload, dict) or schema not in (
        MEANING_RISK_SCHEMA, PERSPECTIVE_SCHEMA, CHAIN_SCHEMA,
    ):
        return {}
    out: dict[str, Any] = {
        "statistical_scope": payload.get("statistical_scope", NOT_DECLARED),
        "sampling_status": payload.get("sampling_status", NOT_DECLARED),
    }
    if schema == CHAIN_SCHEMA:
        out["issuer_asserted_budget"] = _micro(payload, "budget_micro")
        out["joint_delta"] = _micro(payload, "joint_delta_micro")
        out["budget_scope"] = payload.get("budget_scope")
        if payload.get("budget_scope") == HISTORICAL_BUDGET_SCOPE_STATEMENT:
            out["budget_scope_note"] = HISTORICAL_BUDGET_SCOPE_NOTE
        return out
    out["n"] = payload.get("n")
    out["method"] = payload.get("method")
    out["delta"] = _micro(payload, "delta_micro")
    key = _bound_key(replayed)
    if schema == MEANING_RISK_SCHEMA:
        out[key] = _micro(payload, "risk_upper_bound_micro")
        if replayed and "controlled" in payload:
            out["controlled"] = payload["controlled"]
        return out
    out["simultaneous"] = payload.get("simultaneous")
    out["marginal_" + key] = _micro(payload, "marginal_risk_upper_bound_micro")
    groups = payload.get("groups")
    out["cohorts"] = [
        {"group_id": g.get("group_id"), "n": g.get("n"),
         key: _micro(g, "risk_upper_bound_micro")}
        for g in (groups if isinstance(groups, list) else [])
        if isinstance(g, dict)
    ]
    if replayed and "controls_all" in payload:
        out["controls_all"] = payload["controls_all"]
    return out
