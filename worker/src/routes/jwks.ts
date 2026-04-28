// /.well-known/jwks.json — Phase E.1 v0.9.A.
//
// Publishes the Worker's render-receipt signing public key(s) as a
// JWKS (RFC 7517). Consumers fetch this once, cache per
// Cache-Control, and use the matching `kid` to verify any
// render_receipt issued by /api/render.
//
// Rotation: add a new entry to `keys[]` with a new `kid`; signing
// switches to the new kid; old kid stays in the JWKS until every
// receipt that referenced it has aged out of any consumer cache
// the operator cares about. Standard JWKS pattern.
//
// CORS (Phase E.1 v0.9.A.3): JWKS is open `Access-Control-Allow-
// Origin: *` because the keys it publishes are public-key material
// by design — there is no privacy or auth surface here, only
// signature-verification material that anyone may use. Open CORS
// is what lets the v0.9.B browser receipt verifier work from a
// `file://` origin (downloaded single-file demo, offline verify),
// from third-party verifier UIs hosted elsewhere, and from native
// browser fetch from any other origin. NOT `Access-Control-Allow-
// Credentials: true` — JWKS reads no cookies / credentials, and
// pinning the negative protects against a future regression that
// flips a cookie-bearing endpoint to inherit these headers.
// `Access-Control-Max-Age: 86400` caches the preflight for a day
// so repeated verifications don't pay an OPTIONS round-trip.

import type { Env } from "../index";

const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, HEAD, OPTIONS",
  "access-control-max-age": "86400",
  // Deliberately NO `access-control-allow-credentials: true`. JWKS
  // reads no auth context; pinning the negative is intentional. See
  // header comment above.
  //
  // Cross-Origin-Resource-Policy MUST override the Worker's baseline
  // `same-origin` for this resource specifically. CORP is enforced
  // on the publish side independently of CORS (`Access-Control-
  // Allow-Origin: *` alone is not enough — a `same-origin` CORP
  // header still blocks cross-origin reads). JWKS is public-key
  // material; `cross-origin` is the correct CORP for it. The
  // baseline-headers helper falls through when a header is already
  // set, so this value wins over the default.
  "cross-origin-resource-policy": "cross-origin",
} as const;

export async function handleJwks(request: Request, env: Env): Promise<Response> {
  // CORS preflight. Some browser fetches (especially with custom
  // headers) issue an OPTIONS request first; we answer it without
  // touching the JWKS data.
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: CORS_HEADERS,
    });
  }

  const raw = env.RENDER_RECEIPT_PUBLIC_JWKS;
  if (!raw) {
    return new Response(
      JSON.stringify({ error: "RENDER_RECEIPT_PUBLIC_JWKS not configured" }),
      {
        status: 503,
        headers: {
          "content-type": "application/json",
          ...CORS_HEADERS,
        },
      },
    );
  }
  // Parsed-and-reserialised so a misconfigured var fails loudly here
  // rather than serving invalid JSON to verifiers.
  let jwks: unknown;
  try {
    jwks = JSON.parse(raw);
  } catch {
    return new Response(
      JSON.stringify({ error: "RENDER_RECEIPT_PUBLIC_JWKS is not valid JSON" }),
      {
        status: 500,
        headers: {
          "content-type": "application/json",
          ...CORS_HEADERS,
        },
      },
    );
  }
  return new Response(JSON.stringify(jwks), {
    status: 200,
    headers: {
      "content-type": "application/jwk-set+json",
      "cache-control": "public, max-age=3600",
      ...CORS_HEADERS,
    },
  });
}
