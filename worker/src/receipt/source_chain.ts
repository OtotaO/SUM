// T4: source-chain canonicalisation for the receipt's
// `source_chain_hash` field. Mirrors
// sum_engine_internal/transform_receipt/format.py::compute_source_chain_hash
// byte-for-byte so a Python-computed hash equals a Worker-computed
// hash for the same input chain. Cross-runtime check pinned in the
// test fixture at fixtures/transform_receipts/source_chain_*.

import canonicalize from "canonicalize";

export interface EvidenceLink {
  claim: string;
  provenance: {
    source_uri: string;
    byte_start: number;
    byte_end: number;
  };
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", bytes);
  return (
    "sha256-" +
    Array.from(new Uint8Array(buf))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")
  );
}

/**
 * Returns `"sha256-<hex>"` of JCS-canonical bytes of the chain
 * (sorted by (claim, source_uri, byte_start, byte_end) for stable
 * ordering), or `null` when the input is null/empty.
 *
 * MUST produce byte-identical output to the Python helper. The
 * sort key + the per-link normalisation (string coercion of
 * source_uri, int coercion of byte_start/byte_end) is what makes
 * this cross-runtime stable.
 */
export async function computeSourceChainHash(
  chain: EvidenceLink[] | null | undefined,
): Promise<string | null> {
  if (!chain || chain.length === 0) return null;

  const normalised = chain.map((link) => ({
    claim: String(link.claim ?? ""),
    provenance: {
      source_uri: String(link.provenance?.source_uri ?? ""),
      byte_start: Math.trunc(Number(link.provenance?.byte_start ?? 0)),
      byte_end: Math.trunc(Number(link.provenance?.byte_end ?? 0)),
    },
  }));

  normalised.sort((a, b) => {
    if (a.claim !== b.claim) return a.claim < b.claim ? -1 : 1;
    if (a.provenance.source_uri !== b.provenance.source_uri) {
      return a.provenance.source_uri < b.provenance.source_uri ? -1 : 1;
    }
    if (a.provenance.byte_start !== b.provenance.byte_start) {
      return a.provenance.byte_start - b.provenance.byte_start;
    }
    return a.provenance.byte_end - b.provenance.byte_end;
  });

  const canonical = canonicalize(normalised);
  if (typeof canonical !== "string") {
    throw new Error("canonicalize returned undefined for evidence chain");
  }
  return sha256Hex(new TextEncoder().encode(canonical));
}
