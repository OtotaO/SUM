// Real workerd + SQLite tests. Outbound provider calls are intercepted locally.
import assert from "node:assert/strict";
import { before, after, test } from "node:test";
import { register } from "node:module";
import { build } from "esbuild";
import { Miniflare, convertV4MiniflareOptions } from "miniflare";
register("./ts_resolve.mjs", import.meta.url);
const { readBoundedJSON, providerJSON, MAX_REQUEST_BYTES } = await import("../src/request_limits.ts");
const { handleRender } = await import("../src/routes/render.ts");
const { handleComplete } = await import("../src/routes/complete.ts");
let mf, namespace;
const providerCalls = [];
const CENTER = { density: 1, length: .5, formality: .5, audience: .5, perspective: .5 };
const RENDER = { triples: [["SUM", "renders", "text"]], slider_position: { ...CENTER, length: .9 }, force_render: true };
const OPERATOR = { ANTHROPIC_API_KEY: "synthetic-operator" };
const request = (body, endpoint = "render") => new Request(`https://sum.invalid/api/${endpoint}`, { method: "POST", headers: { "content-type": "application/json", "cf-connecting-ip": "192.0.2.10" }, body: JSON.stringify(body) });
before(async () => {
  const built = await build({ entryPoints: ["src/index.ts"], bundle: true, write: false, format: "esm", platform: "browser", target: "es2022" });
  mf = new Miniflare(convertV4MiniflareOptions({ name: "test-sum", modules: true, script: built.outputFiles[0].text, compatibilityDate: "2026-04-23",
    durableObjects: { LLM_BUDGET: { className: "LLMBudget", useSQLite: true } }, kvNamespaces: ["RENDER_CACHE"], bindings: OPERATOR,
    outboundService: async (req) => {
      assert.equal(new URL(req.url).hostname, "api.anthropic.com", "unexpected outbound request");
      const body = await req.json(); providerCalls.push(body); assert.equal(body.max_tokens, 2048);
      return Response.json({ model: "synthetic-model", content: [{ type: "text", text: "SUM renders text." }] });
    },
  }));
  namespace = await mf.getDurableObjectNamespace("LLM_BUDGET");
});
after(async () => { await mf?.dispose(); });
const dispatch = async (req) => mf.dispatchFetch(req.url, { method: req.method, headers: Object.fromEntries(req.headers), body: await req.text() });
const budget = (name) => namespace.get(namespace.idFromName(name));
const admit = (stub, ip = "a".repeat(64), scope = "llm-axis-demo") => stub.fetch("https://internal/admit", { method: "POST", body: JSON.stringify({ ip, scope }) });
const release = (stub, lease) => stub.fetch("https://internal/release", { method: "POST", body: JSON.stringify({ lease }) });
for (const [scope, limit] of [["llm-axis-demo", 5], ["llm-axis-byok", 100]]) {
  test(`SQLite concurrent admission respects the last ${scope} quota slot`, async () => {
    const stub = budget(scope);
    for (let i = 0; i < limit - 1; i++) { const res = await admit(stub, "a".repeat(64), scope); assert.equal(res.status, 200); await release(stub, (await res.json()).lease); }
    const results = await Promise.all(Array.from({ length: 20 }, () => admit(stub, "a".repeat(64), scope)));
    assert.equal(results.filter(r => r.status === 200).length, 1); assert.equal(results.filter(r => r.status === 429).length, 19);
  });
}
test("SQLite shared operator daily budget holds across IPs", async () => {
  const stub = budget("daily");
  for (let i = 0; i < 100; i++) { const res = await admit(stub, i.toString(16).padStart(64, "0")); assert.equal(res.status, 200); await release(stub, (await res.json()).lease); }
  const denied = await admit(stub, "f".repeat(64)); assert.equal(denied.status, 429); assert.equal((await denied.json()).limit, 100);
  assert.equal((await admit(stub, "f".repeat(64), "llm-axis-byok")).status, 200);
});
test("SQLite global concurrent lease limit holds across IPs", async () => {
  const stub = budget("concurrency");
  const results = await Promise.all(Array.from({ length: 30 }, (_, i) => admit(stub, i.toString(16).padStart(64, "0"))));
  assert.equal(results.filter(r => r.status === 200).length, 8); assert.equal(results.filter(r => r.status === 429).length, 22);
  await release(stub, (await results.find(r => r.status === 200).json()).lease);
  assert.equal((await admit(stub, "e".repeat(64))).status, 200);
});
test("SQLite per-IP concurrency rejects without spending denied slots", async () => {
  const stub = budget("per-ip"); const results = await Promise.all(Array.from({ length: 10 }, () => admit(stub)));
  assert.equal(results.filter(r => r.status === 200).length, 2);
  for (const result of results.filter(r => r.status === 200)) await release(stub, (await result.json()).lease);
  for (let i = 0; i < 3; i++) { const res = await admit(stub); assert.equal(res.status, 200); await release(stub, (await res.json()).lease); }
  assert.equal((await admit(stub)).status, 429);
});
test("real Worker dispatches once then serves cache without paid work", async () => {
  const body = { ...RENDER, force_render: false }; let res = await dispatch(request(body));
  assert.equal(res.status, 200); assert.equal((await res.json()).cache_status, "miss"); assert.equal(providerCalls.length, 1);
  res = await dispatch(request(body)); assert.equal(res.status, 200); const cached = await res.json(); assert.equal(cached.cache_status, "hit"); assert.equal(cached.llm_calls_made, 0); assert.equal(providerCalls.length, 1);
});
for (const [name, body, endpoint, expected] of [
  ["null body", null, "render", 400],
  ["triple shape", { ...RENDER, triples: [["a", "b"]] }, "render", 400],
  ["non-string triple", { ...RENDER, triples: [["a", {}, "b"]] }, "render", 400],
  ["triple count", { ...RENDER, triples: Array.from({ length: 257 }, () => ["a", "b", "c"]) }, "render", 400],
  ["triple part length", { ...RENDER, triples: [["a".repeat(513), "b", "c"]] }, "render", 400],
  ["aggregate triple text", { ...RENDER, triples: Array.from({ length: 100 }, () => ["a".repeat(500), "b", "c"]) }, "render", 413],
  ["missing axis", { ...RENDER, slider_position: { density: 1 } }, "render", 400],
  ["null axis", { ...RENDER, slider_position: { ...CENTER, length: null } }, "render", 400],
  ["string axis", { ...RENDER, slider_position: { ...CENTER, length: "0.5" } }, "render", 400],
  ["invalid provider", { ...RENDER, provider: "other" }, "render", 400],
  ["invalid force flag", { ...RENDER, force_render: "false" }, "render", 400],
  ["invalid TTL", { ...RENDER, cache_ttl_seconds: 9999999 }, "render", 400],
  ["arbitrary operator model", { prompt: "test", model: "expensive-unapproved" }, "complete", 400],
  ["long prompt", { prompt: "a".repeat(40001) }, "complete", 413],
  ["transform triple shape", { transform: "slider", input: { triples: [["a", 2, "b"]] } }, "transform", 400],
  ["transform axis", { transform: "slider", input: { triples: RENDER.triples }, parameters: { density: null } }, "transform", 400],
  ["source offsets", { transform: "slider", input: { triples: RENDER.triples }, source_chain: [{ claim: "a", provenance: { source_uri: "local", byte_start: 2, byte_end: 1 } }] }, "transform", 400],
]) {
  test(`${name} rejected before provider dispatch`, async () => {
    const beforeCalls = providerCalls.length; const response = await dispatch(request(body, endpoint));
    assert.equal(response.status, expected, await response.text()); assert.equal(providerCalls.length, beforeCalls);
  });
}
for (const endpoint of ["render", "complete"]) {
  test(`${endpoint} fails closed with absent or unavailable admission`, async () => {
    const handler = endpoint === "render" ? handleRender : handleComplete; const body = endpoint === "render" ? RENDER : { prompt: "test" };
    for (const binding of [undefined, { idFromName: () => "id", get: () => ({ fetch: async () => { throw new Error("offline"); } }) }]) {
      const response = await handler(request(body, endpoint), { ...OPERATOR, LLM_BUDGET: binding }, {}); assert.equal(response.status, 503);
    }
  });
}
test("canonical render works without credentials, cache, or admission", async () => {
  const res = await handleRender(request({ ...RENDER, slider_position: CENTER }), {}, {}); assert.equal(res.status, 200);
  const body = await res.json(); assert.equal(body.llm_calls_made, 0); assert.equal(body.tome, "The SUM renders text.");
});
test("chunked body is bounded without Content-Length", async () => {
  let cancelled = false;
  const body = new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(MAX_REQUEST_BYTES + 1)); }, cancel() { cancelled = true; } });
  await assert.rejects(readBoundedJSON(new Response(body)), err => err.status === 413); assert.equal(cancelled, true);
});
test("stalled body read cancels at its deadline", async () => {
  let cancelled = false;
  await assert.rejects(readBoundedJSON(new Response(new ReadableStream({ cancel() { cancelled = true; } })), 100, 10), err => err.status === 408); assert.equal(cancelled, true);
});
test("provider deadline aborts pending fetch", async (t) => {
  let aborted = false;
  t.mock.method(globalThis, "fetch", async (_url, init) => new Promise((_, reject) => { init.signal.addEventListener("abort", () => { aborted = true; reject(new Error("aborted")); }); }));
  await assert.rejects(providerJSON("https://provider.invalid", {}, 10), /timed out/); assert.equal(aborted, true);
});
test("provider deadline includes stalled response body", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(new ReadableStream()));
  await assert.rejects(providerJSON("https://provider.invalid", {}, 10), /timed out/);
});


test("SQLite quotas survive object eviction and expired leases recover capacity", async () => {
  const name = "restart";
  const stub = budget(name);
  for (let i = 0; i < 2; i++) assert.equal((await admit(stub)).status, 200);
  // Evict while active, or accept that workerd has already made it inactive.
  try { await mf.unsafeEvictDurableObject("test-sum", "LLMBudget", { name }); }
  catch (error) { assert.match(error.message, /not currently running/); }
  // Exercise actual expiry without adding a production clock override.
  await new Promise(resolve => setTimeout(resolve, 35_050));
  const restarted = budget(name);
  for (let i = 0; i < 3; i++) {
    const res = await admit(restarted); assert.equal(res.status, 200);
    await release(restarted, (await res.json()).lease);
  }
  assert.equal((await admit(restarted)).status, 429);
});

test("tiny body chunks preserve bounded JSON parsing", async () => {
  const encoded = new TextEncoder().encode(JSON.stringify({ text: "a".repeat(5000) }));
  let at = 0;
  const body = new ReadableStream({ pull(controller) { if (at === encoded.length) controller.close(); else controller.enqueue(encoded.subarray(at, ++at)); } });
  assert.equal((await readBoundedJSON(new Response(body))).text.length, 5000);
});


test("Python-derived Unicode fixture matches signer, density and transform input", async () => {
  const { readFile } = await import("node:fs/promises");
  const fixture = JSON.parse(await readFile(new URL("./fixtures/unicode_order.json", import.meta.url), "utf8"));
  const { hashTriples } = await import("../src/receipt/sign.ts");
  const { computeSourceChainHash } = await import("../src/receipt/source_chain.ts");
  const { compareTriples } = await import("../src/unicode_order.ts");
  const { applyDensity } = await import("../src/render/axis_prompts.ts");
  const { SLIDER_TRANSFORM } = await import("../src/transforms/slider.ts");
  assert.deepEqual([...fixture.triples].sort(compareTriples), fixture.sorted_triples);
  assert.equal(await hashTriples(fixture.triples), fixture.triples_hash);
  assert.equal(await computeSourceChainHash(fixture.source_chain), fixture.source_chain_hash);
  assert.deepEqual(applyDensity(fixture.triples, .5), fixture.density_half);
  assert.deepEqual(JSON.parse(new TextDecoder().decode(SLIDER_TRANSFORM.canonicalizeInput({ triples: fixture.triples }))), fixture.sorted_triples);
});
