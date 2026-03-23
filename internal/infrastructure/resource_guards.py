"""
Resource Guards — Payload Size and Complexity Limits

Stage 5: Prevents resource exhaustion from oversized payloads,
pathological sync requests, and adversarial ingestion bursts.

All limits are metadata-layer behavior — no algebra semantics change.

Author: ototao
License: Apache License 2.0
"""

import logging
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# ─── Configurable Limits ──────────────────────────────────────────────

MAX_INGEST_TEXT_BYTES = 500_000          # 500 KB
MAX_INGEST_TEXT_CHARS = 200_000          # ~200K characters
MAX_BUNDLE_SIZE_BYTES = 10_000_000       # 10 MB
MAX_STATE_INTEGER_DIGITS = 100_000       # 100K decimal digits
MAX_CANONICAL_TOME_LINES = 50_000        # 50K axiom lines
MAX_SYNC_AXIOMS = 10_000                 # 10K axioms per sync request (unused, kept for compat)
MAX_SYNC_STATE_DIGITS = 100_000          # peer state integer digit count
MAX_ASK_QUERY_LENGTH = 5_000             # 5K characters for /ask
MAX_BRANCH_NAME_LENGTH = 128             # branch name
MAX_AXIOM_KEY_LENGTH = 1_000             # single axiom key


class ResourceLimitError(HTTPException):
    """Raised when a request exceeds resource limits."""

    def __init__(self, resource: str, actual, limit, detail: str = ""):
        msg = f"Resource limit exceeded: {resource} ({actual} > {limit})"
        if detail:
            msg += f". {detail}"
        logger.warning(msg)
        super().__init__(status_code=413, detail=msg)


# ─── Guard Functions ──────────────────────────────────────────────────

def guard_ingest_text(text: str) -> None:
    """Reject oversized ingestion payloads."""
    if len(text) > MAX_INGEST_TEXT_CHARS:
        raise ResourceLimitError(
            "ingest_text_chars", len(text), MAX_INGEST_TEXT_CHARS,
            "Split large documents into smaller chunks."
        )
    byte_len = len(text.encode("utf-8"))
    if byte_len > MAX_INGEST_TEXT_BYTES:
        raise ResourceLimitError(
            "ingest_text_bytes", byte_len, MAX_INGEST_TEXT_BYTES
        )


def guard_bundle_import(bundle: dict) -> None:
    """Reject oversized bundle imports."""
    import json
    bundle_str = json.dumps(bundle)
    if len(bundle_str) > MAX_BUNDLE_SIZE_BYTES:
        raise ResourceLimitError(
            "bundle_size_bytes", len(bundle_str), MAX_BUNDLE_SIZE_BYTES
        )
    # Check state integer size
    state_str = str(bundle.get("state_integer", ""))
    if len(state_str) > MAX_STATE_INTEGER_DIGITS:
        raise ResourceLimitError(
            "state_integer_digits", len(state_str), MAX_STATE_INTEGER_DIGITS,
            "State integer too large for this node."
        )
    # Check tome line count
    tome = bundle.get("canonical_tome", "")
    if isinstance(tome, str):
        line_count = tome.count("\n")
        if line_count > MAX_CANONICAL_TOME_LINES:
            raise ResourceLimitError(
                "canonical_tome_lines", line_count, MAX_CANONICAL_TOME_LINES
            )


def guard_sync_state_digits(digit_count: int) -> None:
    """Reject sync requests with oversized state integers (digit count)."""
    if digit_count > MAX_SYNC_STATE_DIGITS:
        raise ResourceLimitError(
            "sync_state_digits", digit_count, MAX_SYNC_STATE_DIGITS,
            "Peer state integer is too large. Paginate or shard state."
        )


def guard_ask_query(query: str) -> None:
    """Reject oversized /ask queries."""
    if len(query) > MAX_ASK_QUERY_LENGTH:
        raise ResourceLimitError(
            "ask_query_length", len(query), MAX_ASK_QUERY_LENGTH
        )


def guard_branch_name(name: str) -> None:
    """Reject pathological branch names."""
    if len(name) > MAX_BRANCH_NAME_LENGTH:
        raise ResourceLimitError(
            "branch_name_length", len(name), MAX_BRANCH_NAME_LENGTH
        )


def guard_axiom_key(key: str) -> None:
    """Reject pathological axiom keys."""
    if len(key) > MAX_AXIOM_KEY_LENGTH:
        raise ResourceLimitError(
            "axiom_key_length", len(key), MAX_AXIOM_KEY_LENGTH
        )
