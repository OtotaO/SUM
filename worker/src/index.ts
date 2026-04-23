// SUM hosted demo — Cloudflare Worker entry point.
//
// Routing shape:
//   /api/complete   → routes/complete.ts (LLM proxy, migrated from
//                     the Pages Function at single_file_demo/functions/
//                     api/complete.ts)
//   /api/qid        → routes/qid.ts (Phase 4a — Wikidata QID resolver)
//   everything else → static assets from ../single_file_demo/ via the
//                     ASSETS binding (see wrangler.toml [assets]).
//
// Security headers are applied at the Response level via applyBaseline-
// Headers() so they hold for both the API routes and the static assets.
// The old Pages `_headers` file is kept in single_file_demo/ as the
// spec-of-record for the semantics we port here; any future edit to one
// must touch the other (CI gate in follow-up).

import { handleComplete } from "./routes/complete";
import { handleQid } from "./routes/qid";

export interface Env {
  // Static-asset binding — resolves to the ../single_file_demo/
  // directory at deploy time.
  ASSETS: Fetcher;

  // Secrets (optional — absence is signalled to /api/complete callers
  // with a 503; the demo's JS then falls back to the naive tokeniser).
  ANTHROPIC_API_KEY?: string;
  OPENAI_API_KEY?: string;
  CF_AI_GATEWAY_BASE?: string;

  // Plaintext vars (wrangler.toml [vars]).
  SUM_DEFAULT_MODEL_ANTHROPIC?: string;
  SUM_DEFAULT_MODEL_OPENAI?: string;

  // KV — optional until Phase 4a goes live.
  QID_CACHE?: KVNamespace;
}

// Security-baseline Response headers — ported from the Pages `_headers`
// file. Keep in sync. The demo's zero-external-resource property lets
// us run a very tight CSP; if we ever add a CDN-hosted asset, widen
// script-src / style-src accordingly.
const BASELINE_HEADERS: Record<string, string> = {
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Referrer-Policy": "no-referrer",
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
  "Content-Security-Policy": [
    "default-src 'none'",
    "script-src 'unsafe-inline'",
    "style-src 'unsafe-inline'",
    "connect-src 'self'",
    "img-src 'self' data:",
    "font-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join("; "),
  "Permissions-Policy": [
    "accelerometer=()", "autoplay=()", "camera=()",
    "cross-origin-isolated=()", "display-capture=()", "encrypted-media=()",
    "fullscreen=(self)", "geolocation=()", "gyroscope=()",
    "keyboard-map=()", "magnetometer=()", "microphone=()",
    "midi=()", "payment=()", "picture-in-picture=()",
    "publickey-credentials-get=()", "screen-wake-lock=()", "sync-xhr=()",
    "usb=()", "web-share=()", "xr-spatial-tracking=()",
  ].join(", "),
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

function withBaselineHeaders(res: Response): Response {
  // Response objects returned from fetch() or new Response() have
  // immutable headers on the static-asset path; clone to mutate.
  const h = new Headers(res.headers);
  for (const [k, v] of Object.entries(BASELINE_HEADERS)) {
    if (!h.has(k)) h.set(k, v);
  }
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    try {
      if (url.pathname === "/api/complete") {
        return withBaselineHeaders(await handleComplete(request, env));
      }
      if (url.pathname === "/api/qid") {
        return withBaselineHeaders(await handleQid(request, env, ctx));
      }

      // Fall through to static assets. `run_worker_first` in
      // wrangler.toml is set to /api/*, so any non-API path here
      // was NOT routed through this Worker first — the ASSETS
      // binding handles it directly. But we still want baseline
      // headers applied, so we fetch via the binding and wrap.
      const assetRes = await env.ASSETS.fetch(request);
      return withBaselineHeaders(assetRes);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return withBaselineHeaders(
        new Response(JSON.stringify({ error: `worker: ${msg}` }), {
          status: 500,
          headers: { "content-type": "application/json; charset=utf-8" },
        }),
      );
    }
  },
} satisfies ExportedHandler<Env>;
