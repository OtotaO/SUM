// sum_core_wasm.js — browser-side WASM loader for the core-zig/ module.
//
// Exposes a single async factory `loadSumCoreWasm(url)` that returns
// { derivePrime(axiomKey) → BigInt, isReady: true } on success, or null
// if WebAssembly is unavailable / the fetch fails / instantiation errors.
// The demo's extraction hot path calls this first and falls back to the
// pure-JS derivePrimeAsync when the factory returns null — so the page
// works identically everywhere (standalone HTML file, Claude artifact,
// hosted Worker deploy), just faster on the last one.
//
// Why a loader instead of top-level await: the single-file demo has one
// script block; top-level await would block the entire UI. Async factory
// lets the page render first and WASM arrive when it's ready.
//
// Load source: sibling-path fetch of `sum_core.wasm`. On a hosted
// deploy (Cloudflare Worker, CF Pages, any static host) this is a
// same-origin fetch and falls under CSP connect-src 'self'. On a
// local file:// open or inside a Claude artifact the fetch fails
// cleanly and the caller falls back.

'use strict';

/**
 * Attempt to load and instantiate the sum_core WASM module.
 *
 * @param {string} [url='sum_core.wasm'] — relative URL to the .wasm file.
 * @returns {Promise<{derivePrime:(key:string)=>bigint, isReady:true}|null>}
 *          Instantiated interface on success, null on any failure
 *          (WebAssembly unavailable, fetch/compile/instantiate error).
 *          Never throws — fallback logic in the caller stays simple.
 */
async function loadSumCoreWasm(url = 'sum_core.wasm') {
  if (typeof WebAssembly === 'undefined') return null;
  if (typeof fetch === 'undefined') return null;

  let instance;
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    const result = await WebAssembly.instantiateStreaming(
      response,
      {}  // sum_core has no imports — pure freestanding WASM.
    );
    instance = result.instance;
  } catch (e) {
    // Streaming not supported on some older browsers (particularly
    // older Safari on file://), or CSP blocked the fetch, or the
    // module wouldn't compile. Fall back silently.
    try {
      const res = await fetch(url);
      if (!res.ok) return null;
      const bytes = await res.arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, {});
      instance = result.instance;
    } catch {
      return null;
    }
  }

  const {
    memory,
    wasm_alloc_bytes,
    wasm_free_bytes,
    sum_get_deterministic_prime,
  } = instance.exports;

  if (
    !memory || !wasm_alloc_bytes ||
    !wasm_free_bytes || !sum_get_deterministic_prime
  ) {
    return null;  // unexpected export shape — never trust a half-loaded module.
  }

  /**
   * Derive the sha256_64_v1 prime for an axiom key via the Zig core.
   *
   * The Zig function returns a u64 directly (no out-buf marshaling
   * needed for v1 — only v2 writes to a buffer). We allocate a
   * small WASM-heap buffer for the UTF-8 input bytes, call, free.
   *
   * Identity with the JS path is enforced by the cross-runtime harness
   * (scripts/verify_godel_cross_runtime.py); WASM must produce the
   * same BigInt as derivePrimeAsync for every axiom key.
   */
  function derivePrime(axiomKey) {
    const encoded = new TextEncoder().encode(axiomKey);
    const ptr = wasm_alloc_bytes(encoded.length);
    if (!ptr) throw new Error('wasm_alloc_bytes returned null');
    try {
      const heap = new Uint8Array(memory.buffer, ptr, encoded.length);
      heap.set(encoded);
      const result = sum_get_deterministic_prime(ptr, encoded.length);
      // WebAssembly i64 returns surface as BigInt in JS, but they come
      // back SIGNED (the runtime has no notion of u64 vs i64 at the
      // boundary — zig's u64 return type is just a bit pattern). Mask
      // with 2^64 - 1 to reinterpret as unsigned. Verified against the
      // cross-runtime fixture set (alice||likes||cats →
      // 14326936561644797201) post-mask; without the mask, primes
      // whose top bit happens to be 1 surface as negative BigInts.
      const big = typeof result === 'bigint' ? result : BigInt(result);
      return big & 0xffffffffffffffffn;
    } finally {
      wasm_free_bytes(ptr, encoded.length);
    }
  }

  return { derivePrime, isReady: true };
}

// Surface for module consumers (bundlers, tests) and the inlined
// demo script block. Non-module browsers see `window.loadSumCoreWasm`.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { loadSumCoreWasm };
} else if (typeof window !== 'undefined') {
  window.loadSumCoreWasm = loadSumCoreWasm;
}
