"""
Rate Limiter Tests

Verifies sliding window rate limiting: allow/deny behavior,
window expiry, reset, and concurrent keys.

Author: ototao
License: Apache License 2.0
"""

import time
import pytest

from internal.infrastructure.rate_limiter import RateLimiter


class TestRateLimiterBasics:

    def test_allows_under_limit(self):
        """Requests under the limit are allowed."""
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, remaining, _ = rl.check("192.168.1.1")
            assert allowed is True
        assert remaining == 0

    def test_denies_over_limit(self):
        """Request exceeding the limit is denied."""
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check("192.168.1.1")
        allowed, remaining, _ = rl.check("192.168.1.1")
        assert allowed is False
        assert remaining == 0

    def test_remaining_decrements(self):
        """Remaining count decreases with each request."""
        rl = RateLimiter(max_requests=5, window_seconds=60)
        _, r1, _ = rl.check("ip1")
        _, r2, _ = rl.check("ip1")
        assert r1 == 4
        assert r2 == 3

    def test_different_keys_independent(self):
        """Different keys have independent limits."""
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("ip_a")
        rl.check("ip_a")
        allowed, _, _ = rl.check("ip_b")
        assert allowed is True  # ip_b is fresh

    def test_reset_clears_key(self):
        """Reset removes all hits for a key."""
        rl = RateLimiter(max_requests=1, window_seconds=60)
        rl.check("ip1")
        rl.check("ip1")  # now over limit
        rl.reset("ip1")
        allowed, _, _ = rl.check("ip1")
        assert allowed is True


class TestRateLimiterWindow:

    def test_window_expiry(self):
        """Hits expire after the window passes."""
        rl = RateLimiter(max_requests=2, window_seconds=1)
        rl.check("ip1")
        rl.check("ip1")
        # Manually age the hits
        rl._hits["ip1"] = [time.time() - 2.0, time.time() - 2.0]
        allowed, _, _ = rl.check("ip1")
        assert allowed is True  # Old hits expired

    def test_active_keys_count(self):
        """Active keys tracks how many IPs are being rate-limited."""
        rl = RateLimiter()
        rl.check("ip1")
        rl.check("ip2")
        rl.check("ip3")
        assert rl.active_keys == 3

    def test_reset_at_is_future(self):
        """Reset time is in the future."""
        rl = RateLimiter(max_requests=1, window_seconds=60)
        rl.check("ip1")
        _, _, reset_at = rl.check("ip1")
        assert reset_at > time.time()
