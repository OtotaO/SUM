"""ProvenanceRecord — the evidence unit for every SUM axiom.

An axiom is a triple ``(subject, predicate, object)`` minted into a prime and
multiplied into the Gödel state. The *evidence* for an axiom — which source
it was extracted from, which bytes within that source, which extractor, at
what moment — has until now been stored as three flat strings on each MINT
event (``source_url``, ``confidence``, ``ingested_at``). That model cannot
answer the one question the product's provenance claim requires: *show me
the bytes this axiom came from, third-party-verifiable, without trusting
SUM's software.*

This module pins the evidence unit:

    ProvenanceRecord {
        source_uri: str          content-addressable (``sha256:<hex>``)
                                 or scheme-registered signed IRI
        byte_start, byte_end: int  half-open range in source_uri's bytes
        extractor_id: str        e.g. ``sum.sieve:deterministic_v1``
                                      ``sum.llm:gpt-4o-mini-2024-07-18``
        timestamp: str           ISO-8601 UTC, second precision
        text_excerpt: str        the literal substring (≤ 200 chars);
                                 redundant with the byte range but lets
                                 auditors validate without refetching
                                 the source.
        schema_version: str      "1.0.0" — bump on field-shape change.
    }

The record is content-addressable: ``prov_id = "prov:" + hex(sha256(JCS(record)))[:32]``.
Canonicalization uses RFC 8785 JCS (``sum_engine_internal.infrastructure.jcs``) — the
same canonicalizer the VC 2.0 path uses — so prov_ids are stable across any
JSON parser, any language, any reordering of fields on disk.

Content-addressability matters because it gives natural deduplication:
extracting the same fact from the same byte range with the same extractor
twice yields the same prov_id, so the provenance table stays small even when
the extraction pipeline runs many times on overlapping input.

Source URI schemes currently supported:
    ``sha256:<hex>``     — content-addressable; the URI IS the source hash.
    ``doi:<prefix>/<id>`` — DOI, externally resolvable.
    ``https://<host>/<path>[#<fragment>]``
                         — signed HTTPS; consumer's responsibility to pin.
    ``urn:sum:source:<name>``
                         — SUM-internal named source (for corpora).

A bare ``http://`` (no signature, no content hash) is NOT a valid source URI
under this module's contract — it can rot, and cargo-cult provenance is worse
than no provenance. Use ``sha256:`` of the fetched bytes instead.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Sequence

from .jcs import canonicalize

__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceRecord",
    "compute_prov_id",
    "sha256_uri_for_text",
    "validate_source_uri",
]


PROVENANCE_SCHEMA_VERSION = "1.0.0"
EXCERPT_MAX_CHARS = 200

_ALLOWED_URI_PREFIXES: Sequence[str] = (
    "sha256:",
    "doi:",
    "https://",
    "urn:sum:source:",
)


class InvalidProvenanceError(ValueError):
    """Raised when a ProvenanceRecord violates an invariant."""


@dataclass(frozen=True)
class ProvenanceRecord:
    """Evidence that a given axiom was extracted from a specific byte range
    of a specific source by a specific extractor at a specific time.

    All fields are required. ``schema_version`` defaults to the module-level
    constant so callers normally don't supply it.

    Invariants (checked in ``__post_init__``):
        - ``source_uri`` matches one of the allowed schemes
          (``sha256:``, ``doi:``, ``https://``, ``urn:sum:source:``).
        - ``0 <= byte_start < byte_end`` (half-open, non-empty).
        - ``extractor_id`` is non-empty.
        - ``timestamp`` is non-empty (not validated as ISO here; callers
          build it with ``datetime.now(timezone.utc).isoformat()``).
        - ``text_excerpt`` is <= EXCERPT_MAX_CHARS.
    """

    source_uri: str
    byte_start: int
    byte_end: int
    extractor_id: str
    timestamp: str
    text_excerpt: str
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_source_uri(self.source_uri)
        if self.byte_start < 0:
            raise InvalidProvenanceError(
                f"byte_start must be >= 0, got {self.byte_start}"
            )
        if self.byte_end <= self.byte_start:
            raise InvalidProvenanceError(
                f"byte_end ({self.byte_end}) must be strictly greater than "
                f"byte_start ({self.byte_start}) — empty ranges are not evidence"
            )
        if not self.extractor_id.strip():
            raise InvalidProvenanceError("extractor_id must be non-empty")
        if not self.timestamp.strip():
            raise InvalidProvenanceError("timestamp must be non-empty")
        if len(self.text_excerpt) > EXCERPT_MAX_CHARS:
            raise InvalidProvenanceError(
                f"text_excerpt exceeds {EXCERPT_MAX_CHARS} chars "
                f"(got {len(self.text_excerpt)}); truncate at the caller"
            )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def validate_source_uri(uri: str) -> None:
    """Raise InvalidProvenanceError if ``uri`` is not a supported scheme.

    Content-addressable and signed schemes only. A bare ``http://`` URL is
    *not* accepted — it is cargo-cult provenance because the bytes can change
    under the URI at any time. Callers who have only an HTTP URL must
    fetch the bytes, hash them, and submit as ``sha256:<hex>`` — with the
    URL optionally embedded in the extractor_id or a separate audit field.
    """
    if not isinstance(uri, str) or not uri:
        raise InvalidProvenanceError("source_uri must be a non-empty string")
    for prefix in _ALLOWED_URI_PREFIXES:
        if uri.startswith(prefix):
            if prefix == "sha256:":
                body = uri[len("sha256:"):]
                if len(body) != 64 or not all(c in "0123456789abcdef" for c in body):
                    raise InvalidProvenanceError(
                        "sha256: URI body must be 64 lowercase hex chars"
                    )
            return
    raise InvalidProvenanceError(
        f"source_uri scheme not supported: {uri!r}. "
        f"Allowed: {', '.join(_ALLOWED_URI_PREFIXES)}. "
        f"A bare http:// URL is not acceptable — fetch, hash, and use "
        f"sha256:<hex> instead."
    )


def compute_prov_id(record: ProvenanceRecord) -> str:
    """Return the content-addressable prov_id for a record.

    Format: ``"prov:" + sha256(JCS(record))[:32]`` (hex). Collision at 2^-128
    is acceptable; the 32-char truncation keeps IDs copy-pasteable.
    """
    canonical = canonicalize(record.to_dict())
    digest = hashlib.sha256(canonical).hexdigest()
    return f"prov:{digest[:32]}"


def sha256_uri_for_text(text: str) -> str:
    """Helper: content-addressable URI for a block of text.

    The text is UTF-8 encoded, SHA-256'd, and wrapped in the ``sha256:``
    scheme. The resulting URI IS the identity of that text — any byte
    change produces a different URI, so byte_range references are stable.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
