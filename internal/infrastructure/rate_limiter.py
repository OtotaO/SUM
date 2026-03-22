"""
Rate Limiter — In-Memory Sliding Window

Provides application-level rate limiting for the Quantum API to
mitigate volumetric DoS/DDoS at the request layer.

Uses a sliding window counter per client IP with configurable
limits and window sizes.

Author: ototao
License: Apache License 2.0
"""

import time
import logging
from collections import defaultdict
from typing import Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks request counts per key (typically client IP) within
    a configurable time window. Returns (allowed, remaining, reset_at)
    for each check.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum requests per window per key.
            window_seconds: Window size in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, key: str, now: float) -> None:
        """Remove expired timestamps from the key's window."""
        cutoff = now - self.window_seconds
        self._hits[key] = [t for t in self._hits[key] if t > cutoff]

    def check(self, key: str) -> Tuple[bool, int, float]:
        """
        Check and record a request for the given key.

        Args:
            key: Identifier (typically client IP).

        Returns:
            (allowed, remaining, reset_at):
                allowed — True if under limit.
                remaining — Requests remaining in window.
                reset_at — Unix timestamp when the oldest hit expires.
        """
        now = time.time()
        self._cleanup(key, now)

        if len(self._hits[key]) >= self.max_requests:
            oldest = self._hits[key][0] if self._hits[key] else now
            reset_at = oldest + self.window_seconds
            logger.warning("Rate limit exceeded for %s", key)
            return False, 0, reset_at

        self._hits[key].append(now)
        remaining = self.max_requests - len(self._hits[key])
        reset_at = self._hits[key][0] + self.window_seconds

        return True, remaining, reset_at

    def reset(self, key: str) -> None:
        """Clear all hits for a key (e.g., after authentication upgrade)."""
        self._hits.pop(key, None)

    @property
    def active_keys(self) -> int:
        """Number of keys currently being tracked."""
        return len(self._hits)
