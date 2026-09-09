"""Content-addressed bind registry.

Each value bound through the registry gets a deterministic handle of
the form ``sha256:<hex>`` for text or ``sha256:v2:<kind>:<hex>``
for bytes and JSON, derived from the value's canonical bytes.
Subsequent agent tool calls can pass the bind_id instead of inlining
the full value, eliminating two failure modes documented in
``docs/AGENT_SURFACE_FINDINGS.md``:

  - The agent's own JSON response truncating / corrupting under the
    weight of an embedded full-bundle round-trip.
  - Token-cost compounding from re-passing large structures.

Identity rules:
  - ``str``: canonical bytes are the UTF-8 encoding; legacy handles stay stable.
  - ``bytes``: canonical bytes are the value itself, in the v2 bytes namespace.
  - JSON-serialisable (dict, list, primitives): canonical bytes are
    JCS-canonicalised (RFC 8785) UTF-8 bytes — same canonicalisation
    the substrate uses for ``bench_digest`` and signed receipts, in the v2
    json namespace. Resolution returns JSON-normalized values (tuples become
    lists; equivalent JSON numbers can share a representation).

Stored values are immutable byte snapshots; resolving JSON decodes a fresh
copy. Text, bytes and JSON cannot alias one another. Registries are process-
local, so legacy non-text handles cannot survive a server restart anyway:
clients must bind again to obtain v2 handles. There is no ambiguous legacy
non-text alias or change to signed receipt formats.

The registry is process-local (in-memory). Persistence across process
boundaries is post-spike (Phase 26 territory); for the current spike,
process locality is sufficient — agent tool calls happen within a
single process instance of the bind-aware MCP server.

Thread-safe via an ``RLock``. The registry is content-addressed and
idempotent (``bind`` of equivalent values returns the same bind_id and
does not overwrite), but it is NOT unbounded: it is a bounded LRU capped
by entry count AND total canonical bytes. On a long-running server a
client issuing many distinct large binds would otherwise grow memory
without limit — the one DoS axis the per-call size caps do not cover
(2026-07-31 review #20). When a cap is exceeded the least-recently-used
entries are evicted; a subsequently-``resolve``d evicted id raises
``BindNotFoundError`` (a declared, graceful failure), so eviction is safe
for the agent loop. ``resolve`` also distinguishes "agent passed a stale
bind_id from a different session" from "agent passed an inline value that
happens to look like a bind_id" (the latter never happens because we
require the ``sha256:`` prefix).
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import RLock
from typing import Any

# Default caps. A bind entry is at most MAX_TOME_CHARS (~10 MB) of canonical
# bytes; these bound the aggregate the per-call caps leave open. Both are
# constructor-overridable so tests (and tight-memory deployments) can shrink
# them.
DEFAULT_MAX_ENTRIES = 2048
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024  # 256 MB


class BindNotFoundError(KeyError):
    """Raised by ``BindRegistry.resolve`` for unknown bind_ids.

    Subclasses ``KeyError`` for backward compat; new callers should
    catch ``BindNotFoundError`` specifically so the failure can be
    distinguished from accidental ``dict[unknown_key]`` lookups.
    """


class BindTooLargeError(ValueError):
    """A single canonical value cannot fit the registry byte budget."""


class BindRegistry:
    """Process-local content-addressed value registry.

    Single instance is fine for typical agent-loop use; multi-instance
    is supported (each registry has its own store).
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        # OrderedDict keyed by bind_id -> (kind, immutable bytes); insertion/most-recent
        # order at the end. move_to_end on access makes eviction LRU.
        self._store: "OrderedDict[str, tuple[str, bytes]]" = OrderedDict()
        self._total_bytes = 0
        self._max_entries = max(1, int(max_entries))
        self._max_total_bytes = max(1, int(max_total_bytes))
        self._lock = RLock()

    def _evict_to_fit(self, incoming_bytes: int) -> None:
        """Evict least-recently-used entries until the store can admit one more
        entry of ``incoming_bytes`` without breaching either cap. Caller holds
        the lock. Oversized values are rejected before this method is called."""
        while self._store and (
            len(self._store) >= self._max_entries
            or self._total_bytes + incoming_bytes > self._max_total_bytes
        ):
            _evicted_id, (_kind, canonical) = self._store.popitem(last=False)
            self._total_bytes -= len(canonical)

    def bind(self, value: Any) -> str:
        """Content-address a snapshot of ``value`` and return its typed bind_id.

        Idempotent: calling ``bind`` with equivalent values returns the
        same bind_id and refreshes its recency. The registry stores the
        canonical bytes the first time it is bound; subsequent ``bind`` calls with
        the same canonical bytes do not overwrite. Bounded: admitting a new
        entry may evict least-recently-used ones to stay within the entry /
        byte caps. An oversized value raises ``BindTooLargeError`` before
        any eviction.
        """
        canonical = self._canonical_bytes(value)
        nbytes = len(canonical)
        if nbytes > self._max_total_bytes:
            raise BindTooLargeError(
                f"canonical value is {nbytes} bytes; registry limit is "
                f"{self._max_total_bytes} bytes"
            )
        kind = "text" if isinstance(value, str) else "bytes" if isinstance(value, bytes) else "json"
        digest = hashlib.sha256(canonical).hexdigest()
        bind_id = f"sha256:{digest}" if kind == "text" else f"sha256:v2:{kind}:{digest}"
        with self._lock:
            if bind_id in self._store:
                self._store.move_to_end(bind_id)  # LRU touch
                return bind_id
            self._evict_to_fit(nbytes)
            self._store[bind_id] = (kind, canonical)
            self._total_bytes += nbytes
        return bind_id

    def resolve(self, bind_id: str) -> Any:
        """Return an independent, JSON-normalized value for ``bind_id``. Raises ``BindNotFoundError``
        for unknown bind_ids (use ``contains`` to check first if you
        want the boolean form).
        """
        if not isinstance(bind_id, str) or not bind_id.startswith("sha256:"):
            raise BindNotFoundError(
                f"bind_id must be a sha256-prefixed string; got "
                f"{type(bind_id).__name__}: {bind_id!r}"
            )
        with self._lock:
            try:
                kind, canonical = self._store[bind_id]
                self._store.move_to_end(bind_id)  # LRU touch
                if kind == "json":
                    return json.loads(canonical)
                if kind == "text":
                    return canonical.decode("utf-8")
                return canonical
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
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        # Fail closed if canonicalization is unavailable; a fallback would
        # change content identities for the same input.
        from sum_engine_internal.infrastructure.jcs import canonicalize
        return canonicalize(value)


# Module-level shared registry for convenience. Agents that want
# isolation should construct their own BindRegistry instance.
DEFAULT_REGISTRY = BindRegistry()
