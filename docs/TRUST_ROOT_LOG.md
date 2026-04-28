# Trust-root anchor log (public append-only)

This file records the daily transparency anchors for SUM's trust-root state. Each entry is a signed `sum.transparency_anchor.v1` record published to a third-party transparency log (Sigstore Rekor, OpenTimestamps, or both); the entry below names the log and its inclusion-proof location.

The format is documented in [`docs/TRANSPARENCY_ANCHOR.md`](TRANSPARENCY_ANCHOR.md). The verifier algorithm (cross-checking SUM's live trust-root state against the anchor history) is in §"Verifier algorithm" of that doc.

This file is **append-only**. Edits to existing entries are an integrity violation — every commit on this file must add new entries, never modify old ones. CI will enforce this once anchoring is implemented.

## Status

**No anchor entries yet.** Anchoring implementation is gated on:

1. R0.2 trust-root manifest in production (the manifest hash anchored is the load-bearing artifact).
2. A release-signing keypair for trust-root signing (named in [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) §3, not yet generated for v0.3.x).
3. A scheduled GitHub Actions workflow running the daily anchor loop.
4. Rekor / OpenTimestamps integration wired through the workflow.

The first entry will be appended when these gates land. See [`docs/TRANSPARENCY_ANCHOR.md`](TRANSPARENCY_ANCHOR.md) §"Implementation gating" for the full sequence.

## Entry format (when implementation lands)

Each entry is a fenced JSON block:

````
## Anchor entry — <YYYY-MM-DD>

```json
{
  "schema": "sum.transparency_anchor.v1",
  "issued_at": "<ISO-8601 UTC>",
  "snapshots": {
    "jwks_sha256": "<hex>",
    "trust_root_sha256": "<hex>",
    "release_manifest_sha256": "<hex>",
    "receipt_root_sha256": "<hex|null>"
  },
  "log": {
    "kind": "rekor",
    "entry_uuid": "<rekor entry UUID>",
    "verify_url": "https://rekor.sigstore.dev/api/v1/log/entries/<uuid>"
  },
  "jws": "<protected-b64>..<signature-b64>"
}
```
````

Verifiers fetch `verify_url` to obtain the inclusion proof; cross-check the proof against Rekor's signed tree head; cross-check the `snapshots[*]_sha256` fields against the live endpoints' bytes for the date range covered.

## Cross-references

- [`docs/TRANSPARENCY_ANCHOR.md`](TRANSPARENCY_ANCHOR.md) — anchor design + verifier algorithm.
- [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) — manifest format whose hash is anchored.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) — operator runbook; the anchor history is the recovery surface for cases 2, 3, and 5 (JWKS / PyPI / Worker compromise) once implementation lands.
