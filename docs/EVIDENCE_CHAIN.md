# Evidence chain over CanonicalBundle

**Status: feature documentation, v0.** This document is the user-facing spec for the evidence-chain layer that ships in `sum_engine_internal/evidence/chain.py` (PR #200, 2026-05-10). It is referenced by `docs/TRANSFORM_RECEIPT_FORMAT.md` §5 + §6 as the path consumers take when they need provenance beyond what `input_hash` alone gives.

## 1. What evidence chains add

A CanonicalBundle's HMAC or Ed25519 signature attests that the issuer signed the (canonical_tome, state_integer, timestamp) tuple. That is the *cryptographic* boundary. It does NOT attest that any specific axiom in the bundle traces back to any specific byte range in any specific source document. The signature can't reach upstream of the issuer.

An evidence chain closes that gap. Every axiom in the bundle is paired with an `EvidenceLink(claim, provenance)`, where the provenance records:
- `source_uri` — content-addressed (typically `sha256:...`) or external (e.g., `https://...`) reference to the source document
- `byte_start`, `byte_end` — exact slice of the source document the claim was extracted from
- `extractor_id` — which extractor produced the claim
- `timestamp` — when extraction happened
- `text_excerpt` — the actual text slice (so a verifier can confirm `source_uri[byte_start:byte_end]` matches without re-fetching when they have the bytes)
- `schema_version` — `1.0.0` at the v0 surface

Combined with the bundle signature, this gives a reproducible end-to-end provenance: any third party can re-fetch `source_uri`, slice `[byte_start:byte_end]`, confirm the excerpt matches, and verify the bundle signature still holds. The chain itself does not need to be signed separately — when a transform receipt is involved, `payload.source_chain_hash` binds the canonicalised chain into the signature.

## 2. Public surface (v0)

The module is `sum_engine_internal.evidence.chain`. Public symbols:

| Symbol | Kind | Signature |
|---|---|---|
| `EvidenceLink` | dataclass | `EvidenceLink(claim: str, provenance: ProvenanceRecord)` |
| `ProvenanceRecord` | dataclass | `ProvenanceRecord(source_uri: str, byte_start: int, byte_end: int, extractor_id: str, timestamp: str, text_excerpt: str, schema_version: str = "1.0.0")` |
| `EvidenceChainError` | exception | raised on malformed chains |
| `verify_chain_well_formed(links: Iterable[EvidenceLink]) -> None` | function | structural check: no malformed records, no duplicate `(claim, source_uri, byte_range)` 4-tuples |
| `verify_chain_covers_axioms(links, axioms) -> None` | function | coverage check: every axiom in the bundle has at least one link |
| `compose_bundle_with_evidence(codec, text, *, branch="main", source_uri=None, timestamp=None) -> dict` | function | convenience builder: runs the sieve, encodes state, exports the bundle, attaches the chain, verifies coverage before returning |

`ProvenanceRecord` is defined in `sum_engine_internal/infrastructure/provenance.py`; the evidence-chain module composes with the AkashicLedger (`sum_engine_internal/infrastructure/akashic_ledger.py`) when one is present — see the `--ledger` flag on `sum attest`.

## 3. Two intended usage modes

### 3a. Build a fresh bundle with chain attached

```python
from sum_engine_internal.evidence.chain import compose_bundle_with_evidence
from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator

algebra = GodelStateAlgebra()
codec = CanonicalCodec(algebra, AutoregressiveTomeGenerator(algebra))
bundle = compose_bundle_with_evidence(
    codec,
    text="Mars is a planet. The sun is a star.",
    branch="main",
    source_uri="sha256:abc123...",  # or None → defaults to hash of the bytes
    timestamp=None,                   # defaults to now
)
# bundle["axioms"] — list of {subject, predicate, object} dicts
# bundle["evidence_chain"] — list of EvidenceLink-shaped dicts
# bundle["state_integer"], bundle["canonical_tome"], etc — usual CanonicalBundle keys
```

The convenience builder runs `verify_chain_covers_axioms` before returning. If any axiom is uncovered, `EvidenceChainError` raises with the uncovered axioms named.

### 3b. Bind an existing chain to a transform receipt

When invoking the CLI's `sum transform apply <name> --source-chain <path>`, the JSON at `<path>` is a list of `EvidenceLink` records. The CLI canonicalises that list (JCS), computes its sha256, and writes the result to `payload.source_chain_hash` of the emitted `sum.transform_receipt.v1` envelope. Tampering with the chain after the fact breaks the receipt signature, because the hash is signed.

See `docs/TRANSFORM_RECEIPT_FORMAT.md` §1.1 for the exact field semantics and the verification algorithm consumers use to confirm the chain matches what was signed.

## 4. What this layer does NOT do (proof boundary)

- It does not validate that `source_uri` is *truthful*. A producer who lies about the source URI can construct a self-consistent chain with bogus URIs; the third-party verifier catches that only when they fetch the URI and find the bytes don't match. The chain attests structure, not truth of the upstream identifier.
- It does not validate that `text_excerpt` matches `source_uri[byte_start:byte_end]`. That match is the verifier's responsibility when they fetch the source bytes. The chain *records* the excerpt; matching is downstream.
- It does not handle non-leaf evidence. v0 supports only first-order claims (an axiom is extracted directly from a byte range). Higher-order evidence (an axiom is *derived* from other axioms under a named rule — e.g., a Lean-4 entailment certificate) is reserved slot in the dataclass (`derived_from`, `derivation_rule`) but unimplemented at v0.
- It does not address chain-of-custody between the source URI and the extractor. If the source bytes were tampered before extraction, the chain happily attests the tampered bytes. C2PA-style content credentials are the orthogonal solution for that layer.

## 5. Forward compatibility

The schema is `schema_version: "1.0.0"` on `ProvenanceRecord`. Future versions may add:
- `derived_from: list[str]` — list of claim keys this claim was derived from (for non-leaf entailments)
- `derivation_rule: str` — named rule under which `derived_from` implies `claim` (e.g., a Lean-4 lemma id, a sieve composition rule)

When those land they will be additive fields. Existing chains remain valid. Verifiers built to schema `1.0.0` will simply ignore the new fields; verifiers built to a future schema will check `derived_from` consistency.

## 6. References

- Module: `sum_engine_internal/evidence/chain.py` (PR #200, 2026-05-10)
- Adjacent tests: `Tests/test_evidence_chain.py`, `Tests/test_evidence_enrichment.py` (26 / 26 passing on main, verified 2026-05-25)
- Consumers: `docs/TRANSFORM_RECEIPT_FORMAT.md` §5 (input-hash trust boundary), §6 (source_chain_hash field semantics)
- AkashicLedger composition: `sum_engine_internal/infrastructure/akashic_ledger.py` is the on-disk store for ProvenanceRecord; the evidence chain layer is the in-memory / on-wire surface over the same record shape.
