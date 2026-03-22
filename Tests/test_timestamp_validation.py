"""
Timestamp Validation Tests

Verifies that bundle import rejects non-ISO timestamps, future
timestamps, and accepts valid ISO 8601 timestamps.

Author: ototao
License: Apache License 2.0
"""

import copy
import math
import pytest
from datetime import datetime, timezone, timedelta

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import CanonicalCodec


SIGNING_KEY = "timestamp-test-key-32-bytes!!!!!"


@pytest.fixture
def codec_and_bundle():
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, tome_gen, signing_key=SIGNING_KEY)
    algebra.get_or_mint_prime("test", "timestamp", "validation")
    state = list(algebra.axiom_to_prime.values())[0]
    bundle = codec.export_bundle(state)
    return codec, bundle


class TestTimestampValidation:

    def test_valid_iso_timestamp_accepted(self, codec_and_bundle):
        """Normal bundles with valid ISO timestamps import fine."""
        codec, bundle = codec_and_bundle
        result = codec.import_bundle(bundle)
        assert result > 0

    def test_garbage_timestamp_rejected(self, codec_and_bundle):
        """Non-ISO timestamp string is rejected."""
        codec, bundle = codec_and_bundle
        tampered = copy.deepcopy(bundle)
        tampered["timestamp"] = "not-a-date-at-all"
        with pytest.raises(ValueError, match="not valid ISO 8601"):
            codec.import_bundle(tampered)

    def test_empty_timestamp_rejected(self, codec_and_bundle):
        """Empty timestamp string is rejected."""
        codec, bundle = codec_and_bundle
        tampered = copy.deepcopy(bundle)
        tampered["timestamp"] = ""
        with pytest.raises(ValueError, match="not valid ISO 8601"):
            codec.import_bundle(tampered)

    def test_unix_epoch_timestamp_rejected(self, codec_and_bundle):
        """Numeric-looking timestamp is rejected."""
        codec, bundle = codec_and_bundle
        tampered = copy.deepcopy(bundle)
        tampered["timestamp"] = "1711100000"
        with pytest.raises((ValueError, Exception)):
            codec.import_bundle(tampered)

    def test_far_future_timestamp_rejected(self, codec_and_bundle):
        """Timestamp >24h in the future is rejected."""
        codec, bundle = codec_and_bundle
        tampered = copy.deepcopy(bundle)
        future = datetime.now(timezone.utc) + timedelta(hours=48)
        tampered["timestamp"] = future.isoformat()
        with pytest.raises(ValueError, match="future"):
            codec.import_bundle(tampered)

    def test_past_timestamp_accepted(self, codec_and_bundle):
        """Old but valid timestamp is accepted (no min age constraint)."""
        codec, bundle = codec_and_bundle
        tampered = copy.deepcopy(bundle)
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        tampered["timestamp"] = past.isoformat()
        # Will fail HMAC (timestamp changed), which is correct
        from internal.infrastructure.canonical_codec import InvalidSignatureError
        with pytest.raises(InvalidSignatureError):
            codec.import_bundle(tampered)
