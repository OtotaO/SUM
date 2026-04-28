// /.well-known/revoked-kids.json — Phase E.1 G3 (revocation MVP).
//
// Publishes the operator's render-receipt revocation list. Receipts
// whose kid appears here with signed_at >= effective_revocation_at
// MUST be rejected by verifiers per docs/RENDER_RECEIPT_FORMAT.md
// §6.1.
//
// The list itself is operator-managed via the
// RENDER_RECEIPT_REVOKED_KIDS env var (parsed JSON). Default is an
// empty list; populating it is a deliberate operator action when a
// kid is suspected compromised. The shape mirrors the JWKS endpoint
// pattern: simple JSON read at request time, no signing on the list
// itself (the list is itself a public-trust artifact — anyone with
// the secret can mint a list, but a list claiming a kid is revoked
// is only weight-bearing when the operator is trusted; in
// adversarial scenarios the list signature would be a follow-on
// hardening, not a v1 requirement).
//
// CORS open per the same logic as JWKS: revocation list is public-
// state material. v0.9.B in-page verifier needs to fetch this
// cross-origin alongside JWKS to apply the revocation check.

import type { Env } from "../index";

const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, HEAD, OPTIONS",
  "access-control-max-age": "86400",
  "cross-origin-resource-policy": "cross-origin",
} as const;

const EMPTY_LIST = {
  schema: "sum.revoked_kids.v1",
  issued_at: new Date().toISOString(),
  revoked: [],
} as const;

export async function handleRevokedKids(
  request: Request,
  env: Env,
): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: CORS_HEADERS,
    });
  }

  const raw = env.RENDER_RECEIPT_REVOKED_KIDS;
  let body: unknown;
  if (!raw) {
    // No env var set → empty revocation list. Verifiers receive a
    // valid sum.revoked_kids.v1 envelope with revoked: [], NOT a
    // 404. 404 would be hard to distinguish from "endpoint not
    // configured" / "deploy regression."
    body = {
      ...EMPTY_LIST,
      issued_at: new Date().toISOString(),
    };
  } else {
    // Parsed-and-reserialised so a misconfigured var fails loudly
    // here rather than serving invalid JSON.
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object" || parsed.schema !== "sum.revoked_kids.v1") {
        return new Response(
          JSON.stringify({
            error:
              "RENDER_RECEIPT_REVOKED_KIDS does not match sum.revoked_kids.v1 schema",
          }),
          {
            status: 500,
            headers: {
              "content-type": "application/json",
              ...CORS_HEADERS,
            },
          },
        );
      }
      body = parsed;
    } catch {
      return new Response(
        JSON.stringify({
          error: "RENDER_RECEIPT_REVOKED_KIDS is not valid JSON",
        }),
        {
          status: 500,
          headers: {
            "content-type": "application/json",
            ...CORS_HEADERS,
          },
        },
      );
    }
  }

  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      "content-type": "application/json",
      "cache-control": "public, max-age=3600",
      ...CORS_HEADERS,
    },
  });
}
