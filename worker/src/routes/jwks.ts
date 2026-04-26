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

import type { Env } from "../index";

export async function handleJwks(_request: Request, env: Env): Promise<Response> {
  const raw = env.RENDER_RECEIPT_PUBLIC_JWKS;
  if (!raw) {
    return new Response(
      JSON.stringify({ error: "RENDER_RECEIPT_PUBLIC_JWKS not configured" }),
      { status: 503, headers: { "content-type": "application/json" } },
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
      { status: 500, headers: { "content-type": "application/json" } },
    );
  }
  return new Response(JSON.stringify(jwks), {
    status: 200,
    headers: {
      "content-type": "application/jwk-set+json",
      "cache-control": "public, max-age=3600",
    },
  });
}
