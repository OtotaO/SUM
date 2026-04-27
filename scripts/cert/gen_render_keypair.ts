// One-shot keygen helper for Phase E.1 v0.9.A render receipts.
//
// Generates an Ed25519 keypair as JWKs, writes:
//   /tmp/render_receipt_private.jwk  — single-line JSON, the secret half
//   /tmp/render_receipt_public.jwks  — single-line JSON of the {keys: [...]} JWKS
//
// Usage (from repo root, run from inside `worker/` so tsx resolves):
//   cd worker
//   npx tsx ../scripts/cert/gen_render_keypair.ts [kid]
//   wrangler secret put RENDER_RECEIPT_SIGNING_JWK < /tmp/render_receipt_private.jwk
//   wrangler secret put RENDER_RECEIPT_SIGNING_KID    # paste the kid printed below
//   # then publish the public JWKS — preferred:
//   #   CF dashboard → Worker → Settings → Variables → add
//   #   RENDER_RECEIPT_PUBLIC_JWKS as plaintext, paste contents of
//   #   /tmp/render_receipt_public.jwks. (Avoid putting JSON in
//   #   wrangler.toml [vars] — escaping nightmare on merge.)
//   # alternative one-shot at deploy time:
//   #   wrangler deploy --var RENDER_RECEIPT_PUBLIC_JWKS:"$(cat /tmp/render_receipt_public.jwks)"
//
// Reversibility: deleting the secret invalidates already-issued receipts
// that reference this kid. To rotate without breaking old receipts, run
// this script again with a new kid, ADD the new JWK to the existing JWKS
// (don't replace), update RENDER_RECEIPT_SIGNING_KID to the new kid, and
// keep the old entry in JWKS for the cache TTL window of any consumer.
//
// Author: ototao
// License: Apache License 2.0

import { generateKeyPair, exportJWK } from "jose";
import { writeFileSync } from "node:fs";

async function main(): Promise<void> {
  const kid = process.argv[2] ?? `sum-render-${new Date().toISOString().slice(0, 10)}-1`;
  const { publicKey, privateKey } = await generateKeyPair("EdDSA", {
    crv: "Ed25519",
    extractable: true,
  });
  const privateJwk = await exportJWK(privateKey);
  const publicJwk = await exportJWK(publicKey);

  // Annotate per RFC 7517: alg, use, kid live in the JWK itself so the
  // verifier can pick by kid + algorithm without external metadata.
  const annotate = (jwk: Record<string, unknown>) => ({
    ...jwk,
    alg: "EdDSA",
    use: "sig",
    kid,
  });
  const priv = annotate(privateJwk as Record<string, unknown>);
  const pub = annotate(publicJwk as Record<string, unknown>);
  const jwks = { keys: [pub] };

  const privPath = "/tmp/render_receipt_private.jwk";
  const pubPath = "/tmp/render_receipt_public.jwks";
  writeFileSync(privPath, JSON.stringify(priv));
  writeFileSync(pubPath, JSON.stringify(jwks));

  console.log(`kid:                 ${kid}`);
  console.log(`private JWK written: ${privPath}`);
  console.log(`public JWKS written: ${pubPath}`);
  console.log();
  console.log("Next steps:");
  console.log(`  wrangler secret put RENDER_RECEIPT_SIGNING_JWK < ${privPath}`);
  console.log(`  wrangler secret put RENDER_RECEIPT_SIGNING_KID  # paste: ${kid}`);
  console.log(`  # then publish the public JWKS as RENDER_RECEIPT_PUBLIC_JWKS:`);
  console.log(`  cat ${pubPath}`);
  console.log();
  console.log("AFTER the secret upload succeeds — wipe the private-key tempfile:");
  console.log(`  rm -P ${privPath}      # macOS / BSD: overwrite then unlink`);
  console.log(`  shred -u ${privPath}   # GNU/Linux equivalent`);
  console.log("Leaving uncovered key material on local disk is a pd_8 violation.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
