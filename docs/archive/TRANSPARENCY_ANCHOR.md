# Transparency anchor design (R0.5)

**Status:** design-now / prototype-later. Specifies the daily anchoring loop, the input schema, the verifier semantics, and the rollout sequence — implementation lands as its own cycle once the R0.2 trust-root manifest shape has been stable for a release cycle (anchoring an unstable shape just locks the instability).

A transparency anchor publishes a daily, externally-witnessed snapshot of SUM's trust-relevant byte hashes — JWKS, trust-root manifest, release manifest, and (optionally) a Merkle root over the day's render receipts — to a third-party transparency log (Sigstore Rekor or OpenTimestamps). The anchor gives downstream consumers a witness independent of GitHub, Cloudflare, and the repo author: "as of date X, these were the published trust-root bytes."

Without anchoring, every trust-relevant byte SUM serves is implicitly trusted-because-served-by-the-issuer. With anchoring, a third-party log says "yes, that hash was published at that time" without depending on SUM's infrastructure. Nation-state-level recovery: if both GitHub AND the Worker's deploy pipeline were compromised retroactively, the anchor history still records what was published when, so downstream consumers can detect post-hoc tampering.

## What gets anchored

A `sum.transparency_anchor.v1` record:

```json
{
  "schema": "sum.transparency_anchor.v1",
  "issued_at": "2026-04-28T06:00:00.000Z",
  "snapshots": {
    "jwks_sha256": "<hex>",
    "trust_root_sha256": "<hex>",
    "release_manifest_sha256": "<hex>",
    "receipt_root_sha256": "<hex|null>"
  },
  "log": {
    "kind": "rekor" | "opentimestamps",
    "entry_uuid": "<log-specific identifier>",
    "verify_url": "<URL the anchor can be verified at>"
  }
}
```

`snapshots` is a flat dict of the four trust-relevant hashes captured at anchor time:

- `jwks_sha256` — SHA-256 of the JCS-canonicalized JWKS document at `/.well-known/jwks.json`. The anchor records "this is the JWKS the issuer claimed to be serving on this day."
- `trust_root_sha256` — SHA-256 of the most recent signed trust-root manifest's bytes. Pins the release-state-of-record on the day of anchor.
- `release_manifest_sha256` — SHA-256 of the latest GitHub release's `sum.trust_root.v1.json` asset (canonical fetchable form per `docs/TRUST_ROOT_FORMAT.md` §3). Distinct from `trust_root_sha256` only when the deployed Worker's manifest differs from the GitHub release asset's manifest — divergence between these is itself a finding worth surfacing.
- `receipt_root_sha256` — Optional. If implementation includes daily receipt-batching, this is the Merkle root over `sha256(receipt_bytes)` for every receipt issued that day. **Hashes only, never raw receipt content** — privacy is non-negotiable. `null` means receipt anchoring is not active for this anchor.

`log` carries the anchor's location in the chosen transparency log so a verifier can fetch the inclusion proof.

## Backend choice — Rekor vs OpenTimestamps

Two candidates. Both should be evaluated; SUM might end up using both for defense in depth.

### Sigstore Rekor

Rekor is purpose-built for software supply-chain transparency — it logs signed metadata with cryptographic inclusion + consistency proofs against a Merkle tree of all public log entries. Sigstore's hosted instance at `rekor.sigstore.dev` is operated by the Linux Foundation and is the same log that backs `cosign` attestations.

**For SUM:** authenticate the anchor record with the trust-root signing key (the same key R0.2 introduced for trust-root manifests), submit to Rekor, record the returned `entry_uuid` + verify URL. Verifiers fetch the inclusion proof, verify against Rekor's signed tree head, and assert the entry exists at the claimed time.

**Strengths:** signed metadata is the load-bearing primitive; Rekor's API is well-supported; cosign tooling already exists for verification; the log's tree-head signing keys are openly published (Sigstore TUF root).

**Weaknesses:** Rekor's hosted instance has occasional read-side outages; the Sigstore TUF root requires its own keep-up-to-date discipline.

### OpenTimestamps

OpenTimestamps anchors arbitrary bytes' SHA-256 to the Bitcoin blockchain via aggregating calendar servers. The result is a `.ots` proof file that any Bitcoin node can independently verify decades later — Bitcoin's blockchain is the trust anchor, no SUM-specific infrastructure required.

**For SUM:** hash the `sum.transparency_anchor.v1` record bytes, submit to `https://alice.btc.calendar.opentimestamps.org` (or any OpenTimestamps calendar), receive a `.ots` proof file, publish it.

**Strengths:** Bitcoin-anchored proof outlives any single transparency log operator; decade-scale durability; verification is trivially decentralized.

**Weaknesses:** anchoring latency (Bitcoin block time, ~10 min - 1 hour for a confirmed proof); only proves "X existed before time T," not "X is the only thing published" (no consistency surface).

### Recommendation

**Default to Rekor; OpenTimestamps as a secondary anchor for archival use cases.** Rekor's signed-metadata + inclusion-proof model is the better fit for SUM's "is this hash exactly what the issuer claimed to publish" question. OpenTimestamps as a defense-in-depth layer for archival use cases (research replication, regulatory audit) where the durability case beats the freshness case.

## The daily anchor loop

```
00:00 UTC daily:
  1. Fetch /.well-known/jwks.json bytes; sha256 → jwks_sha256.
  2. Fetch the deployed Worker's /.well-known/sum-trust-root.json
     bytes; sha256 → trust_root_sha256.
  3. Fetch the latest GitHub release's sum.trust_root.v1.json asset
     bytes; sha256 → release_manifest_sha256.
  4. (Optional) Fetch the day's receipt-hash batch from the Worker;
     compute Merkle root → receipt_root_sha256.
  5. Build the sum.transparency_anchor.v1 record above.
  6. Sign with the trust-root signing key (same key as R0.2 manifests).
  7. Submit to Rekor; record entry_uuid + verify_url.
  8. Append the signed record + log location to docs/TRUST_ROOT_LOG.md
     in a new commit.
```

Step 8 — the commit — is the load-bearing public surface. Anyone reading `docs/TRUST_ROOT_LOG.md` sees the day-by-day chain of what SUM claimed to be serving on each day, with each entry independently verifiable against Rekor.

## Verifier algorithm

A verifier presented with a SUM artifact (release wheel, receipt, etc.) and a date range:

```
1. Locate the daily anchor record(s) covering the date range in
   docs/TRUST_ROOT_LOG.md.
2. For each anchor record:
   a. Verify the record's signature with the trust-root JWKS at
      its issued_at time.
   b. Verify the record's inclusion in Rekor by fetching
      `verify_url` and checking the inclusion proof against Rekor's
      signed tree head.
   c. Cross-check the snapshots[*]_sha256 fields against the bytes
      the verifier independently fetched from the live endpoints.
3. If any cross-check fails, the artifact's published-state-of-record
   has drifted between then and now — the issuer's infrastructure is
   not internally consistent over time. Surface for incident response.
```

The verifier's contract: not "this artifact is good" but "this artifact's published trust state has been consistent since the anchor." Bad artifacts published from a clean state still slip through; tampering with the published state after the anchor is what gets caught.

## Implementation gating

This design lands as `docs/TRANSPARENCY_ANCHOR.md` (this file). Implementation does NOT land in R0.5 — it requires:

1. **R0.2 trust-root manifest in production.** The anchor records a manifest hash; without a real signed manifest, the anchor is empty.
2. **A release-signing keypair.** Same key R0.2 documents; not yet generated for v0.3.x.
3. **A scheduled GitHub Actions workflow** that runs the daily anchor loop on a cron schedule.
4. **Rekor account / Cosign integration** wired through the workflow.
5. **`docs/TRUST_ROOT_LOG.md` populated** with at least one anchor entry.

When all five are in place, the implementation cycle is one PR: ~200 LOC of GHA workflow + a Python script that fetches the inputs and submits to Rekor. The design above is the contract that PR implements.

## What this defends against

- **Post-hoc Worker compromise.** An attacker who pwns the Worker deploy pipeline and rewrites historical responses can't rewrite Rekor history; the anchor records what was served at anchor time, period.
- **GitHub release tampering.** Same as above for the GitHub release artifacts: a force-pushed release branch doesn't rewrite the anchor.
- **Issuer disappearance.** If SUM's repo is dormant for a year, downstream consumers can still verify "this artifact was published before timestamp X" from Rekor alone, no SUM endpoints required.

## What this does NOT defend against

- **Pre-anchor compromise.** If the Worker was already serving malicious bytes when the anchor was written, the anchor records the malicious hash as "what was published." The anchor is integrity over time, not integrity at issuance.
- **Trust-root key compromise.** The anchor record is signed; if the signing key is compromised, an attacker can publish anchors for any hash. Mitigation: kid rotation on the signing key + revocation surface (G3) propagates to anchor verification too.
- **Rekor compromise.** If Sigstore's Rekor itself is compromised, the inclusion proofs are forgeable. Mitigation: dual-anchor to OpenTimestamps for decade-scale defense in depth.

## Cross-references

- [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) — the manifest whose hash is anchored.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) cases 2, 3, 5 — JWKS / PyPI / Worker compromise; transparency anchors give an external timestamped record of what was served when.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) R0.5 — playbook entry that scoped this design.
- [`docs/TRUST_ROOT_LOG.md`](TRUST_ROOT_LOG.md) — the public append-only record that implementation populates.
