"""Bundled demo fixture for ``python -m sum_verify --demo``.

Ships a real binding-gate golden — a ``sum.meaning_risk_receipt.v1`` over
public-domain BillSum text (CC0) plus its JWKS and committed per-pair
losses — INSIDE the ``[verify]`` wheel, so a ``pip install``-only user can
replay a meaning-loss bound fully offline with zero git clone. These are
copies of ``fixtures/meaning_receipts_billsum/`` in the source tree; the
fixtures there remain the source of truth.
"""
