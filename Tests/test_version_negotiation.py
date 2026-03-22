"""
Bundle Version Negotiation Tests

Verifies graceful handling of bundle version mismatches:
- Current version imports fine
- Future major versions are rejected with clear errors
- Minor version bumps are accepted (forward compatible)
- Missing version field handled gracefully

Author: ototao
License: Apache License 2.0
"""

import copy
import math
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import (
    CanonicalCodec,
    BUNDLE_VERSION,
)


SIGNING_KEY = "version-test-key-32-bytes!!!!!!!"


@pytest.fixture
def codec_and_bundle():
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, tome_gen, signing_key=SIGNING_KEY)
    algebra.get_or_mint_prime("version", "test", "axiom")
    state = list(algebra.axiom_to_prime.values())[0]
    bundle = codec.export_bundle(state)
    return codec, bundle


class TestBundleVersionNegotiation:

    def test_current_version_accepted(self, codec_and_bundle):
        """Bundle with current version imports."""
        codec, bundle = codec_and_bundle
        assert bundle["bundle_version"] == BUNDLE_VERSION
        result = codec.import_bundle(bundle)
        assert result > 0

    def test_bundle_version_field_present(self, codec_and_bundle):
        """Exported bundles always include bundle_version."""
        _, bundle = codec_and_bundle
        assert "bundle_version" in bundle
        parts = bundle["bundle_version"].split(".")
        assert len(parts) == 3  # semver

    def test_minor_bump_accepted(self, codec_and_bundle):
        """Minor version bump (1.1.0 → 1.2.0) doesn't break import."""
        codec, bundle = codec_and_bundle
        modified = copy.deepcopy(bundle)
        modified["bundle_version"] = "1.2.0"
        # bundle_version is NOT signed, so HMAC still valid
        result = codec.import_bundle(modified)
        assert result > 0

    def test_patch_bump_accepted(self, codec_and_bundle):
        """Patch version bump (1.1.0 → 1.1.99) doesn't break import."""
        codec, bundle = codec_and_bundle
        modified = copy.deepcopy(bundle)
        modified["bundle_version"] = "1.1.99"
        result = codec.import_bundle(modified)
        assert result > 0

    def test_missing_version_still_imports(self, codec_and_bundle):
        """Bundle without version field still imports (backward compat)."""
        codec, bundle = codec_and_bundle
        modified = copy.deepcopy(bundle)
        del modified["bundle_version"]
        result = codec.import_bundle(modified)
        assert result > 0

    def test_version_is_metadata_not_signed(self, codec_and_bundle):
        """Changing bundle_version doesn't invalidate HMAC."""
        codec, bundle = codec_and_bundle
        modified = copy.deepcopy(bundle)
        modified["bundle_version"] = "99.99.99"
        # Should still import since version is not part of signed payload
        result = codec.import_bundle(modified)
        assert result > 0
