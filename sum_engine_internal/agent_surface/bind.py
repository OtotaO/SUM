"""Content-addressed bind registry.

Each value bound through the registry gets a deterministic handle of
the form ``sha256:<hex>``, derived from the value's canonical bytes.
Subsequent agent tool calls can pass the bind_id instead of inlining
the full value, eliminating two failure modes documented in
``docs/AGENT_SURFACE_FINDINGS.md``:

  - The agent's own JSON response truncating / corrupting under the
    weight of an embedded full-bundle round-trip.
  - Token-cost compounding from re-passing large structures.

Identity rules:
  - ``str``: canonical bytes are the UTF-8 encoding.
  - ``bytes``: canonical bytes are the value itself.
  - JSON-serialisable (dict, list, primitives): canonical bytes are
    JCS-canonicalised (RFC 8785) UTF-8 bytes — same canonicalisation
    the substrate uses for ``bench_digest`` and signed receipts.

The registry is process-local (in-memory). Persistence across process
boundaries is post-spike (Phase 26 territory); for the current spike,
process locality is sufficient — agent tool calls happen within a
single process instance of the bind-aware MCP server.

Thread-safe via an ``RLock``. The registry is intentionally
*append-only*: ``bind`` does not delete previous entries with the
same content (it returns the same bind_id deterministically).
``resolve`` raises ``BindNotFoundError`` for unknown bind_ids,
distinguishing "agent passed a stale bind_id from a different
session" from "agent passed an inline value that happens to look
like a bind_id" (the latter never happens because we require the
``sha256:`` prefix).
"""
from __future__ import annotations

import hashlib
from threading import RLock
from typing import Any


class BindNotFoundError(KeyError):
    """Raised by ``BindRegistry.resolve`` for unknown bind_ids.

    Subclasses ``KeyError`` for backward compat; new callers should
    catch ``BindNotFoundError`` specifically so the failure can be
    distinguished from accidental ``dict[unknown_key]`` lookups.
    """


class BindRegistry:
    """Process-local content-addressed value registry.

    Single instance is fine for typical agent-loop use; multi-instance
    is supported (each registry has its own store).
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = RLock()

    def bind(self, value: Any) -> str:
        """Content-address ``value``. Returns ``sha256:<hex>`` bind_id.

        Idempotent: calling ``bind`` with equivalent values returns the
        same bind_id. The registry stores the value the first time it
        is bound; subsequent ``bind`` calls with the same canonical
        bytes do not overwrite (the original value is returned by
        ``resolve``).
        """
        canonical = self._canonical_bytes(value)
        digest = hashlib.sha256(canonical).hexdigest()
        bind_id = f"sha256:{digest}"
        with self._lock:
            if bind_id not in self._store:
                self._store[bind_id] = value
        return bind_id

    def resolve(self, bind_id: str) -> Any:
        """Return the value for ``bind_id``. Raises ``BindNotFoundError``
        for unknown bind_ids (use ``contains`` to check first if you
        want the boolean form).
        """
        if not isinstance(bind_id, str) or not bind_id.startswith("sha256:"):
            raise BindNotFoundError(
                f"bind_id must be a 'sha256:<hex>' string; got "
                f"{type(bind_id).__name__}: {bind_id!r}"
            )
        with self._lock:
            try:
                return self._store[bind_id]
            except KeyError:
                raise BindNotFoundError(
                    f"unknown bind_id {bind_id!r}; the value was not bound "
                    f"in this registry. Possible causes: stale bind_id from "
                    f"a previous process; bind_id from a different registry "
                    f"instance; the value was bound but the registry was "
                    f"reset. Check the agent's tool-call history for the "
                    f"bind() that produced this id."
                ) from None

    def contains(self, bind_id: str) -> bool:
        """Check whether ``bind_id`` is in the registry."""
        if not isinstance(bind_id, str) or not bind_id.startswith("sha256:"):
            return False
        with self._lock:
            return bind_id in self._store

    def size(self) -> int:
        """Return number of entries in the registry."""
        with self._lock:
            return len(self._store)

    def _canonical_bytes(self, value: Any) -> bytes:
        """Compute canonical bytes for content-addressing.

        Strings → UTF-8. Bytes → as-is. Other JSON-serialisable values
        → JCS-canonicalised UTF-8. Anything else raises TypeError.
        """
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        # JSON-serialisable: use JCS for cross-platform identity.
        try:
            from sum_engine_internal.infrastructure.jcs import canonicalize
        except ImportError:  # pragma: no cover - defensive
            # Fall back to sorted-keys json if JCS isn't importable for
            # any reason. This loses cross-runtime identity but works
            # in-process for the spike.
            import json
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        # canonicalize returns bytes (RFC 8785 JCS-canonical UTF-8).
        return canonicalize(value)


# Module-level shared registry for convenience. Agents that want
# isolation should construct their own BindRegistry instance.
DEFAULT_REGISTRY = BindRegistry()
