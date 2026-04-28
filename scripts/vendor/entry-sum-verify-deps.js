// Vendor entry point: re-exports exactly what the v0.9.B browser
// receipt verifier consumes from `jose` and `canonicalize`.
// Re-exporting a narrow surface (rather than the full library) keeps
// the bundled output small + makes the verifier's dep contract
// auditable from this file alone.

export {
  flattenedVerify,
  createRemoteJWKSet,
} from "jose";

export { default as canonicalize } from "canonicalize";
