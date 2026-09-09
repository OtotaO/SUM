// Provider selection regressions with synthetic credentials and mocked admission.
// hardening.test.mjs separately exercises SQLite admission inside workerd.
import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./ts_resolve.mjs", import.meta.url);
const { classifyScope, checkRateLimit } = await import("../src/rate_limit.ts");
const { handleRender } = await import("../src/routes/render.ts");
const { handleComplete } = await import("../src/routes/complete.ts");
const { handleTransform } = await import("../src/routes/transform.ts");

function budgetMock(initialCount = 0) {
  const calls = [];
  return { calls, idFromName: (name) => name, get: () => ({ fetch: async (url, init) => {
    const body = JSON.parse(init.body);
    if (url.endsWith("/release")) return Response.json({ released: true });
    calls.push(body);
    const limit = body.scope === "llm-axis-demo" ? 5 : 100;
    if (initialCount >= limit) return Response.json({ scope: body.scope, limit }, { status: 429, headers: { "x-ratelimit-remaining": "0", "retry-after": "30" } });
    return Response.json({ lease: "test-lease" });
  } }) };
}

const DEMO = "llm-axis-demo";
const BYO = "llm-axis-byok";
const CENTER = { density: 1, length: 0.5, formality: 0.5, audience: 0.5, perspective: 0.5 };
const RENDER_BODY = {
  triples: [["SUM", "renders", "text"]],
  slider_position: { ...CENTER, length: 0.9 },
  force_render: true,
};
const OPERATOR = { ANTHROPIC_API_KEY: "operator-anthropic", OPENAI_API_KEY: "operator-openai" };
const GATEWAY = "https://gateway.example.invalid/sum";
const HEADERS = {
  anthropic: "x-render-llm-key-anthropic",
  openai: "x-render-llm-key-openai",
};

function request(body, keys = {}, endpoint = "render") {
  const headers = { "content-type": "application/json", "cf-connecting-ip": "192.0.2.1" };
  for (const [provider, value] of Object.entries(keys)) headers[HEADERS[provider]] = value;
  return new Request(`https://sum.example.invalid/api/${endpoint}`, {
    method: "POST", headers, body: JSON.stringify(body),
  });
}

function memoryKV(initialCount = 0) {
  const values = new Map();
  const reads = [];
  const writes = [];
  return {
    reads, writes,
    async get(key) {
      reads.push(key);
      return values.get(key) ?? (key.startsWith("rl:") ? String(initialCount) : null);
    },
    async put(key, value, options) {
      writes.push({ key, value, options });
      values.set(key, value);
    },
  };
}

function mockProvider(t) {
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, init) => {
    const provider = String(url).includes("anthropic") ? "anthropic" : "openai";
    const direct = provider === "anthropic"
      ? "https://api.anthropic.com/v1/messages"
      : "https://api.openai.com/v1/chat/completions";
    const gateway = `${GATEWAY}/${provider}/${provider === "anthropic" ? "v1/messages" : "chat/completions"}`;
    assert.ok(url === direct || url === gateway, "unexpected provider URL; no network is permitted");
    const headers = new Headers(init.headers);
    const credential = provider === "anthropic"
      ? headers.get("x-api-key")
      : headers.get("authorization")?.replace(/^Bearer /, "");
    calls.push({ provider, credential, url });
    const body = provider === "anthropic"
      ? { model: "mock-anthropic", content: [{ type: "text", text: "SUM renders text." }] }
      : { model: "mock-openai", choices: [{ message: { content: "SUM renders text." } }] };
    return Response.json(body);
  });
  return calls;
}

test("scope uses only a selected render credential; other endpoints stay conservative", () => {
  for (const key of [undefined, "", " \t "]) assert.equal(classifyScope("render", key), DEMO);
  assert.equal(classifyScope("render", "  selected-key  "), BYO);
  for (const endpoint of ["complete"]) {
    assert.equal(classifyScope(endpoint), DEMO);
    assert.equal(classifyScope(endpoint, "selected-key"), DEMO);
  }
  assert.equal(classifyScope("transform", "selected-key"), "canonical");
  assert.equal(classifyScope("qid", "selected-key"), "qid");
});

const cases = [
  { name: "default Anthropic operator", expected: "anthropic", scope: DEMO },
  { name: "default Anthropic ignores unrelated OpenAI key", keys: { openai: "user-openai" }, expected: "anthropic", scope: DEMO },
  { name: "default Anthropic BYO", keys: { anthropic: "user-anthropic" }, expected: "anthropic", scope: BYO },
  { name: "default both BYO keys retains Anthropic preference", keys: { anthropic: "user-anthropic", openai: "user-openai" }, expected: "anthropic", scope: BYO },
  { name: "explicit Anthropic ignores OpenAI key", provider: "anthropic", keys: { openai: "user-openai" }, expected: "anthropic", scope: DEMO },
  { name: "explicit OpenAI ignores Anthropic key", provider: "openai", keys: { anthropic: "user-anthropic" }, expected: "openai", scope: DEMO },
  { name: "explicit OpenAI operator", provider: "openai", expected: "openai", scope: DEMO },
  { name: "explicit OpenAI BYO", provider: "openai", keys: { openai: "user-openai" }, expected: "openai", scope: BYO },
  { name: "explicit Anthropic BYO", provider: "anthropic", keys: { anthropic: "user-anthropic" }, expected: "anthropic", scope: BYO },
  { name: "explicit OpenAI with both BYO keys", provider: "openai", keys: { anthropic: "user-anthropic", openai: "user-openai" }, expected: "openai", scope: BYO },
  { name: "whitespace Anthropic key and unrelated OpenAI key use operator gateway", keys: { anthropic: "   ", openai: "user-openai" }, gateway: true, expected: "anthropic", scope: DEMO },
  { name: "whitespace OpenAI key and unrelated Anthropic key use operator gateway", provider: "openai", keys: { openai: "   ", anthropic: "user-anthropic" }, gateway: true, expected: "openai", scope: DEMO },
  { name: "trimmed Anthropic BYO bypasses operator gateway", keys: { anthropic: "  user-anthropic  " }, gateway: true, expected: "anthropic", scope: BYO },
  { name: "trimmed OpenAI BYO bypasses operator gateway", provider: "openai", keys: { openai: "  user-openai  " }, gateway: true, expected: "openai", scope: BYO },
  { name: "OpenAI operator fallback", env: { OPENAI_API_KEY: "operator-openai" }, expected: "openai", scope: DEMO },
  { name: "OpenAI BYO without operator keys", env: {}, keys: { openai: "user-openai" }, expected: "openai", scope: BYO },
  { name: "Anthropic BYO without operator keys", env: {}, keys: { anthropic: "user-anthropic" }, expected: "anthropic", scope: BYO },
  { name: "whitespace operator Anthropic key permits configured OpenAI fallback", env: { ANTHROPIC_API_KEY: "  ", OPENAI_API_KEY: "  operator-openai  " }, expected: "openai", scope: DEMO },
];

for (const scenario of cases) {
  test(`render: ${scenario.name}`, async (t) => {
    const calls = mockProvider(t);
    const kv = memoryKV();
    const budget = budgetMock();
    const env = { ...(scenario.env ?? OPERATOR), RENDER_CACHE: kv, LLM_BUDGET: budget };
    if (scenario.gateway) env.CF_AI_GATEWAY_BASE = GATEWAY;
    const response = await handleRender(request({ ...RENDER_BODY, provider: scenario.provider }, scenario.keys), env, {});
    assert.equal(response.status, 200);
    assert.equal((await response.json()).llm_calls_made, 1);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].provider, scenario.expected);
    assert.equal(calls[0].credential, `${scenario.scope === BYO ? "user" : "operator"}-${scenario.expected}`);
    assert.equal(calls[0].url.startsWith(GATEWAY), Boolean(scenario.gateway && scenario.scope === DEMO));
    assert.equal(budget.calls[0].scope, scenario.scope);
    assert.equal(budget.calls.length, 1);
  });
}

for (const provider of ["anthropic", "openai"]) {
  test(`render: rejected ${provider} BYO credential never retries on the operator key`, async (t) => {
    let calls = 0;
    t.mock.method(globalThis, "fetch", async (_url, init) => {
      calls += 1;
      const headers = new Headers(init.headers);
      assert.equal(
        provider === "anthropic" ? headers.get("x-api-key") : headers.get("authorization"),
        provider === "anthropic" ? "rejected-user-key" : "Bearer rejected-user-key",
      );
      return Response.json({ error: "mock credential rejected" }, { status: 401 });
    });
    const kv = memoryKV();
    const budget = budgetMock();
    const response = await handleRender(request({ ...RENDER_BODY, provider }, { [provider]: "rejected-user-key" }), { ...OPERATOR, RENDER_CACHE: kv, LLM_BUDGET: budget }, {});
    assert.equal(response.status, 502);
    assert.equal(calls, 1);
    assert.equal(budget.calls[0].scope, BYO);
  });

  test(`render: explicit missing ${provider} credential does not switch providers`, async (t) => {
    const calls = mockProvider(t);
    const other = provider === "anthropic" ? "openai" : "anthropic";
    const kv = memoryKV();
    const response = await handleRender(request({ ...RENDER_BODY, provider }, { [other]: `user-${other}` }), { RENDER_CACHE: kv }, {});
    assert.equal(response.status, 503);
    assert.equal(calls.length, 0);
    assert.equal(kv.reads.length, 0);
  });

  test(`render: exhausted demo bucket blocks ${provider} dispatch despite other-provider key`, async (t) => {
    const calls = mockProvider(t);
    const other = provider === "anthropic" ? "openai" : "anthropic";
    const kv = memoryKV(5);
    const budget = budgetMock(5);
    const response = await handleRender(request({ ...RENDER_BODY, provider }, { [other]: `user-${other}` }), { ...OPERATOR, RENDER_CACHE: kv, LLM_BUDGET: budget }, {});
    assert.equal(response.status, 429);
    const body = await response.json();
    assert.equal(body.scope, DEMO);
    assert.equal(body.limit, 5);
    assert.equal(response.headers.get("x-ratelimit-remaining"), "0");
    assert.ok(Number(response.headers.get("retry-after")) > 0);
    assert.equal(calls.length, 0);
    assert.equal(kv.writes.length, 0);
  });
}

test("render: exhausted BYO bucket blocks dispatch", async (t) => {
  const calls = mockProvider(t);
  const response = await handleRender(request(RENDER_BODY, { anthropic: "user-anthropic" }), { ...OPERATOR, RENDER_CACHE: memoryKV(100), LLM_BUDGET: budgetMock(100) }, {});
  assert.equal(response.status, 429);
  assert.equal((await response.json()).scope, BYO);
  assert.equal(calls.length, 0);
});

for (const [scope, limit, window] of [["canonical", 100, 3600], ["qid", 60, 3600]]) {
  test(`best-effort KV sequential boundary: ${scope}`, async (t) => {
    t.mock.method(Date, "now", () => 1_800_000_000_000);
    const kv = memoryKV(limit - 1);
    const req = request(RENDER_BODY);
    const allowed = await checkRateLimit(req, kv, scope);
    assert.equal(allowed.allowed, true);
    assert.equal(allowed.remaining, 0);
    assert.equal(allowed.limit, limit);
    assert.ok(allowed.reset_seconds > 0 && allowed.reset_seconds <= window);
    const blocked = await checkRateLimit(req, kv, scope);
    assert.equal(blocked.allowed, false);
    assert.equal(blocked.remaining, 0);
    assert.equal(kv.writes.length, 1);
    assert.equal(kv.writes[0].options.expirationTtl, window + 60);
  });
}

test("complete ignores BYO headers and dispatches on the operator's demo allowance", async (t) => {
  const calls = mockProvider(t);
  const kv = memoryKV();
  const budget = budgetMock();
  const response = await handleComplete(request({ prompt: "Extract triples." }, { anthropic: "user-anthropic", openai: "user-openai" }, "complete"), { ...OPERATOR, RENDER_CACHE: kv, LLM_BUDGET: budget });
  assert.equal(response.status, 200);
  assert.equal(calls[0].credential, "operator-anthropic");
  assert.equal(budget.calls[0].scope, DEMO);
});

test("transform uses the canonical allowance even when both BYO keys are present", async (t) => {
  const calls = mockProvider(t);
  const kv = memoryKV();
  const response = await handleTransform(request({ transform: "slider", input: { triples: RENDER_BODY.triples }, parameters: CENTER }, { anthropic: "user-anthropic", openai: "user-openai" }, "transform"), { ...OPERATOR, RENDER_CACHE: kv });
  assert.equal(response.status, 200);
  assert.equal((await response.json()).llm_calls_made, 0);
  assert.equal(calls.length, 0);
  assert.match(kv.reads[0], /^rl:canonical:/);
});
