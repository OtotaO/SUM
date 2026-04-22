"""
Telemetry — Zig FFI Observability

Lightweight decorator for tracking Zig vs Python fallback path selection
and per-call latency.  Emits structured log lines consumable by the
SSE telemetry stream.

Usage:
    @trace_zig_ffi("lcm")
    def merge_parallel_states(self, ...):
        ...

Author: ototao
License: Apache License 2.0
"""

import time
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger("sum.telemetry")

# ── Counters (module-level, thread-safe via GIL) ──

_zig_calls = 0
_python_calls = 0
_total_ns = 0


def get_stats() -> dict:
    """Return telemetry counters."""
    return {
        "zig_calls": _zig_calls,
        "python_calls": _python_calls,
        "total_ns": _total_ns,
        "zig_ratio": _zig_calls / max(_zig_calls + _python_calls, 1),
    }


def trace_zig_ffi(operation_name: str) -> Callable:
    """
    Decorator that wraps a Strangler Fig call site to log:
      - Which path was taken (Zig vs Python)
      - Wall-clock latency in µs
      - Input magnitude (for BigInt args)

    The decorated function must follow the pattern:
        zig = _get_zig_engine()
        if zig is not None:
            result = zig.method(...)
            if result is not None:
                return result
        # fallback
        return python_result

    This decorator wraps the entire function, measuring total time,
    then infers the path from whether Zig was available.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            global _zig_calls, _python_calls, _total_ns

            start = time.perf_counter_ns()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter_ns() - start

            _total_ns += elapsed

            # Detect path by checking if zig_engine is loaded
            try:
                from sum_engine_internal.infrastructure.zig_bridge import zig_engine
                if zig_engine is not None and zig_engine.available:
                    _zig_calls += 1
                    path = "⚡zig"
                else:
                    _python_calls += 1
                    path = "🐍py"
            except ImportError:
                _python_calls += 1
                path = "🐍py"

            logger.debug(
                "[%s] %s %s — %.1f µs",
                path, operation_name, type(result).__name__,
                elapsed / 1000
            )
            return result
        return wrapper
    return decorator
