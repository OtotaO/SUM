// One SQLite Durable Object coordinates all public provider work. Transactions
// admit against the IP quota, shared operator budget, and leases together.
// Cached results and deterministic rendering never reserve a paid attempt.
import type { Env } from "./index";
import { POLICY, rateLimitedResponse, type RateLimitScope } from "./rate_limit";
import { PROVIDER_TIMEOUT_MS } from "./request_limits";

type LLMScope = Extract<RateLimitScope, "llm-axis-demo" | "llm-axis-byok">;
export const OPERATOR_DAILY_LIMIT = 100;
export const MAX_ACTIVE_CALLS = 8;
export const MAX_ACTIVE_PER_IP = 2;
export const LEASE_MS = PROVIDER_TIMEOUT_MS + 5_000;
const MAX_BUCKETS = 4096;

function unavailable(): Response {
  return Response.json({ error: "generation admission unavailable; retry later or use canonical rendering" }, { status: 503, headers: { "retry-after": "30" } });
}

export class LLMBudget {
  private state: DurableObjectState;
  constructor(state: DurableObjectState) {
    this.state = state;
    state.storage.sql.exec("CREATE TABLE IF NOT EXISTS buckets (key TEXT PRIMARY KEY, count INTEGER NOT NULL, reset INTEGER NOT NULL)");
    state.storage.sql.exec("CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, ip TEXT NOT NULL, expires INTEGER NOT NULL)");
  }

  async fetch(request: Request): Promise<Response> {
    const body = await request.json() as { scope?: LLMScope; ip?: string; lease?: string };
    if (new URL(request.url).pathname === "/release" && typeof body.lease === "string") {
      this.state.storage.sql.exec("DELETE FROM leases WHERE id = ?", body.lease);
      return Response.json({ released: true });
    }
    if (!(body.scope === "llm-axis-demo" || body.scope === "llm-axis-byok") || !/^[a-f0-9]{64}$/.test(body.ip ?? "")) return Response.json({ error: "invalid admission" }, { status: 400 });
    const scope = body.scope;
    const ip = body.ip!;
    return this.state.storage.transactionSync(() => {
      const sql = this.state.storage.sql;
      const now = Date.now();
      sql.exec("DELETE FROM buckets WHERE reset <= ?", now);
      sql.exec("DELETE FROM leases WHERE expires <= ?", now);
      const policy = POLICY[scope];
      const reset = (Math.floor(now / (policy.window_seconds * 1000)) + 1) * policy.window_seconds * 1000;
      const key = `${scope}:${ip}`;
      const count = [...sql.exec<{ count: number }>("SELECT count FROM buckets WHERE key = ?", key)][0]?.count ?? 0;
      if (count >= policy.limit) return rateLimitedResponse({ allowed: false, scope, limit: policy.limit, remaining: 0, reset_seconds: Math.ceil((reset - now) / 1000) });
      const globalCount = [...sql.exec<{ count: number }>("SELECT count FROM buckets WHERE key = 'operator'")][0]?.count ?? 0;
      if (scope === "llm-axis-demo" && globalCount >= OPERATOR_DAILY_LIMIT) {
        return Response.json({ error: "shared operator daily budget exhausted", scope, limit: OPERATOR_DAILY_LIMIT }, { status: 429, headers: { "retry-after": String(Math.ceil(((Math.floor(now / 86400000) + 1) * 86400000 - now) / 1000)) } });
      }
      const active = [...sql.exec<{ count: number }>("SELECT count(*) AS count FROM leases")][0].count;
      const ipActive = [...sql.exec<{ count: number }>("SELECT count(*) AS count FROM leases WHERE ip = ?", ip)][0].count;
      if (active >= MAX_ACTIVE_CALLS || ipActive >= MAX_ACTIVE_PER_IP) return Response.json({ error: "generation capacity busy; retry shortly" }, { status: 429, headers: { "retry-after": "5" } });
      const buckets = [...sql.exec<{ count: number }>("SELECT count(*) AS count FROM buckets")][0].count;
      if (count === 0 && buckets >= MAX_BUCKETS) return unavailable();
      sql.exec("INSERT INTO buckets (key, count, reset) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET count = excluded.count, reset = excluded.reset", key, count + 1, reset);
      if (scope === "llm-axis-demo") {
        const dailyReset = (Math.floor(now / 86400000) + 1) * 86400000;
        sql.exec("INSERT INTO buckets (key, count, reset) VALUES ('operator', ?, ?) ON CONFLICT(key) DO UPDATE SET count = excluded.count, reset = excluded.reset", globalCount + 1, dailyReset);
      }
      const lease = crypto.randomUUID();
      sql.exec("INSERT INTO leases (id, ip, expires) VALUES (?, ?, ?)", lease, ip, now + LEASE_MS);
      return Response.json({ lease });
    });
  }
}

export async function admitLLM(request: Request, env: Env, scope: LLMScope): Promise<{ release: () => Promise<void> } | Response> {
  if (!env.LLM_BUDGET) return unavailable();
  try {
    // Store a daily digest, never the IP or a provider credential. Cloudflare
    // supplies CF-Connecting-IP; local requests without it share one bucket.
    const day = Math.floor(Date.now() / 86400000);
    const bytes = new TextEncoder().encode(`${day}:${request.headers.get("cf-connecting-ip") || "unknown"}`);
    const ip = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), b => b.toString(16).padStart(2, "0")).join("");
    const stub = env.LLM_BUDGET.get(env.LLM_BUDGET.idFromName("public-llm-v1"));
    const response = await stub.fetch("https://budget.internal/admit", { method: "POST", body: JSON.stringify({ scope, ip }), signal: AbortSignal.timeout(5000) });
    if (!response.ok) return response.status === 429 ? response : unavailable();
    const result = await response.json() as { lease?: string };
    if (typeof result.lease !== "string") return unavailable();
    return { release: async () => {
      try { await stub.fetch("https://budget.internal/release", { method: "POST", body: JSON.stringify({ lease: result.lease }), signal: AbortSignal.timeout(2000) }); }
      catch { /* Admission is never refunded; abandoned concurrency leases expire. */ }
    } };
  } catch { return unavailable(); }
}
