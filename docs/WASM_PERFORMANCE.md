# WASM performance — what is measured, what is claimed

*Priority 2 of [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md).
The single-file demo ships a Zig → WASM core (`core-zig/`, compiled to
`single_file_demo/sum_core.wasm`) alongside a pure-JS fallback. No
performance claim has ever been made about this path; this file exists
so that when a claim IS made, it points at measurements rather than
at expectations. Every number below is labelled `measured` per
[`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md).*

## The claim this document discharges

None — yet. The repo's guarantee about the WASM path is **correctness**,
proved by `scripts/verify_godel_cross_runtime.py` and the K1/K1-mw/K2/
K3/K4/A1–A6 matrix: the WASM core derives the same prime as the Python
+ Node reference implementations for every axiom key in the fixture
set. That's a proof of equivalence, not of speed.

Once the methodology here has been executed across Chrome 113+, Firefox
129+, and Safari 17+ (the three engines the standalone verifier
supports), the aggregated numbers can cross from **designed** ("WASM
should be faster") to **measured** ("WASM is X× faster on Y, Z× on Y',
equivalent on Y''"). Until that happens, README and commit messages
must remain silent on WASM speed.

## What is measured

### Surface: `derivePrime(axiomKey) → BigInt`

The hot path of axiom attestation. Input: a UTF-8 axiom key of the form
`{subject}||{predicate}||{object}`. Output: the `sha256_64_v1`
deterministic prime for that key, namely
`nextPrime(SHA-256(axiom_key)[:8] as big-endian u64)`.

All four surfaces are measured on this single function:

1. **Browser / WASM** — `loadSumCoreWasm()` → `wasm.derivePrime(key)`
   (Zig SHA-256 + Zig deterministic Miller-Rabin, `core-zig/src/main.zig`
   `sum_get_deterministic_prime`).
2. **Browser / JS** — inlined copy of `derivePrimeAsync` matching
   `single_file_demo/index.html` (WebCrypto `crypto.subtle.digest` +
   BigInt 12-witness Miller-Rabin `nextPrime`).
3. **Node / JS** — `standalone_verifier/math.js`'s `derivePrime`
   (node's `crypto.createHash` + the same 12-witness `nextPrime`).
4. **Python / Zig-or-sympy** —
   `sum_engine_internal.algorithms.semantic_arithmetic.GodelStateAlgebra.get_or_mint_prime`
   (the dispatch resolves to the Zig shared library when
   `libsum_core` is compiled for the host, pure `sympy.nextprime`
   otherwise — the benchmark records which path actually served).

### Input sizes

`N ∈ {10, 100, 1000, 10000}` axioms per trial. Same deterministic key
sequence in every surface so all four benchmarks hash identical bytes:

```
axiom_key(i) = f"sum-bench-v1_subject_{i}||relates_to||sum-bench-v1_object_{i}"
```

Literal seed `"sum-bench-v1"` is versioned into the harness as
`AXIOM_KEY_SEED`; bumping it ends comparability with prior runs, so
do not bump it casually.

### Trial protocol

Per surface, per N: three untimed warm-up calls, then five timed
trials. The reported number is the median of the five; min/max are
reported alongside so the variance is visible.

Trials alternate between WASM and JS (in the browser harness) so that a
transient GC or background task biases both paths equally.

For correctness, every trial also composes the N primes into a
state integer via LCM. The browser harness asserts that the WASM and
JS state integers are bit-identical at the end of each N; a
disagreement is flagged as a benchmark failure, not a speed datum.

### What is NOT measured

- **Bundle verification time.** Ed25519 signature validation dominates
  end-to-end verify; it lives in a different surface (Python's
  `cryptography`, Node's `@noble/ed25519`, browser's WebCrypto) and
  would need its own benchmark. This document is scoped to prime
  derivation.
- **Extraction.** Sieve + LLM extraction is bounded by IO and API
  latency, not CPU on primes. Irrelevant to the WASM claim.
- **Canonical-tome parsing.** Dominates when N is small and axiom
  keys are long; irrelevant when N is large.
- **First-paint / cold-load latency.** Matters for UX but not for
  the prime-derivation kernel. WASM's `instantiateStreaming` cost is
  amortised into the first call and then irrelevant.

## How to reproduce

### Browser surfaces (WASM + JS)

From a clean checkout:

```bash
# Build and stage the WASM artifact (regenerate if core-zig/ has changed)
make wasm

# Serve the repo root over HTTP. file:// blocks WebAssembly fetch and
# crypto.subtle on most engines.
python -m http.server 8000
```

Open each of these in turn, click **Run benchmark**, and copy the JSON
result:

- Chrome 113+:  `http://localhost:8000/Tests/benchmarks/browser_wasm_bench.html`
- Firefox 129+: same URL
- Safari 17+:   same URL (macOS)

Paste the JSON block into the per-browser section below. One block per
browser, dated with the timestamp the harness emits.

### Python surface

```bash
python scripts/bench_python_derive.py --json > /tmp/python_bench.json
cat /tmp/python_bench.json
```

Paste the JSON block into the "Python CLI" section below. If the Zig
shared library is not compiled on your host, the `derivation_path`
field will read `python_fallback`; that's a legitimate measurement and
should be recorded as such (with a note that it's the sympy path, not
the Zig fast path).

### Node surface

```bash
# Inline node one-liner covering the same input generation contract.
node -e '
  const { derivePrime, lcm } = require("./standalone_verifier/math");
  const SEED = "sum-bench-v1";
  const sizes = [10, 100, 1000, 10000];
  const trials = 5;
  const out = { schema: "sum.wasm_bench.v1", surface: "node_js", results: [] };
  for (const N of sizes) {
    const keys = Array.from({length: N}, (_, i) =>
      `${SEED}_subject_${i}||relates_to||${SEED}_object_${i}`);
    // Warmup
    for (let i = 0; i < 3; i++) derivePrime(keys[0]);
    const times = [];
    let state = 1n;
    for (let t = 0; t < trials; t++) {
      state = 1n;
      const t0 = process.hrtime.bigint();
      for (const k of keys) state = lcm(state, derivePrime(k));
      const t1 = process.hrtime.bigint();
      times.push(Number(t1 - t0) / 1e6);
    }
    const sorted = times.slice().sort((a,b) => a-b);
    const median = sorted[Math.floor(sorted.length / 2)];
    out.results.push({
      N, trials,
      ms: { median, min: Math.min(...times), max: Math.max(...times), all: times },
      per_op_us: median * 1000 / N,
      state_integer_first8_hex: state.toString(16).padStart(16, "0").slice(0,16)
    });
  }
  out.node = process.version;
  out.timestamp = new Date().toISOString();
  console.log(JSON.stringify(out, null, 2));
' > /tmp/node_bench.json
```

## Results — to be populated

Every block below is a direct paste of the harness's JSON output.
Blocks are grouped by surface. Dates and engine versions come from the
payload; do not hand-edit the numbers.

The skeletons have the correct `sum.wasm_bench.v1` shape with `null`
numeric fields. To populate: **overwrite the entire fenced block** with
the JSON emitted by the harness. Do not merge key-by-key; the harness's
output is already the complete block.

### Python CLI

Source: `python scripts/bench_python_derive.py --json`

```json
{
  "schema": "sum.wasm_bench.v1",
  "surface": "python_cli",
  "timestamp": null,
  "python_version": null,
  "platform": null,
  "machine": null,
  "derivation_path": null,
  "trials_per_n": 5,
  "axiom_key_seed": "sum-bench-v1",
  "results": [
    { "N": 10,    "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null },
    { "N": 100,   "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null },
    { "N": 1000,  "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null },
    { "N": 10000, "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null }
  ]
}
```

### Node verifier (`standalone_verifier/math.js`)

Source: the inline `node -e '…'` one-liner above → `/tmp/node_bench.json`.

```json
{
  "schema": "sum.wasm_bench.v1",
  "surface": "node_js",
  "node": null,
  "timestamp": null,
  "results": [
    { "N": 10,    "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null },
    { "N": 100,   "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null },
    { "N": 1000,  "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null },
    { "N": 10000, "trials": 5, "ms": { "median": null, "min": null, "max": null, "all": [] }, "per_op_us": null, "state_integer_first8_hex": null }
  ]
}
```

### Browser — Chrome 113+

Source: `http://localhost:8000/Tests/benchmarks/browser_wasm_bench.html` → **Copy JSON result**.

```json
{
  "schema": "sum.wasm_bench.v1",
  "timestamp": null,
  "ua": null,
  "platform": null,
  "hw_concurrency": null,
  "wasm_available": null,
  "trials_per_n": 5,
  "axiom_key_seed": "sum-bench-v1",
  "results": [
    {
      "N": 10,
      "trials": 5,
      "wasm_ms": { "median": null, "min": null, "max": null, "all": [] },
      "js_ms":   { "median": null, "min": null, "max": null, "all": [] },
      "ratio_js_over_wasm": null,
      "state_agree": null,
      "state_integer_first8_hex": null
    },
    { "N": 100,   "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 1000,  "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 10000, "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null }
  ]
}
```

### Browser — Firefox 129+

Source: same URL, Firefox.

```json
{
  "schema": "sum.wasm_bench.v1",
  "timestamp": null,
  "ua": null,
  "platform": null,
  "hw_concurrency": null,
  "wasm_available": null,
  "trials_per_n": 5,
  "axiom_key_seed": "sum-bench-v1",
  "results": [
    { "N": 10,    "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 100,   "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 1000,  "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 10000, "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null }
  ]
}
```

### Browser — Safari 17+ (macOS)

Source: same URL, Safari.

```json
{
  "schema": "sum.wasm_bench.v1",
  "timestamp": null,
  "ua": null,
  "platform": null,
  "hw_concurrency": null,
  "wasm_available": null,
  "trials_per_n": 5,
  "axiom_key_seed": "sum-bench-v1",
  "results": [
    { "N": 10,    "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 100,   "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 1000,  "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null },
    { "N": 10000, "trials": 5, "wasm_ms": null, "js_ms": null, "ratio_js_over_wasm": null, "state_agree": null, "state_integer_first8_hex": null }
  ]
}
```

## Interpretation

Populate the table below once all five blocks above are filled in.
**Do not populate it speculatively.** Every cell is copied from the
matching JSON block; cells without a source stay as `—`.

### Per-op microseconds (median, N=1000)

| Surface                   | median µs/op  | per-op ratio (vs Python) |
|---------------------------|---------------|--------------------------|
| Python (Zig lib)          | —             | 1.00× (baseline)         |
| Python (sympy fallback)   | —             | —                        |
| Node verifier (JS)        | —             | —                        |
| Browser WASM — Chrome     | —             | —                        |
| Browser JS   — Chrome     | —             | —                        |
| Browser WASM — Firefox    | —             | —                        |
| Browser JS   — Firefox    | —             | —                        |
| Browser WASM — Safari     | —             | —                        |
| Browser JS   — Safari     | —             | —                        |

### What to conclude, truthfully

Read the numbers, then pick the honest sentence:

- *"WASM is measurably faster than JS in all three browsers."* — only
  if every Browser-WASM cell is faster than its matching Browser-JS
  cell, beyond the min/max spread. If Safari's WASM is only 1.1× of JS
  and the spread overlaps, do not claim "faster" for Safari.
- *"WASM is faster on Chrome and Firefox, equivalent on Safari."* —
  if that is the shape of the data, write exactly that.
- *"WASM is slower than JS on browser X."* — if that happens, say so.
  Do not bury the finding. Update `single_file_demo/index.html`'s
  dispatcher (`ensureSumWasm` / `derivePrime`) to prefer JS on that
  engine; the API surface is unchanged.
- *"WASM and JS are within noise; the claim is unsupportable."* — if
  the median difference is smaller than the min-max spread on every
  row, the measured answer is "no reliable difference" and that is a
  finding.

## Fallback statement (ships regardless of results)

The demo and verifier are designed to run correctly without the WASM
path. If the WASM path is unavailable (WebAssembly disabled, fetch
blocked by CSP, `instantiateStreaming` unsupported on an older
engine) or measurably slower on a given browser, the pure-JS path
serves. Consumers do not need to know which path ran: `derivePrime(key)`
returns the identical BigInt either way, a property proved by the
cross-runtime harness on every commit.

To force the JS path in-browser (e.g., for debugging or for benchmarks
that want to isolate the JS cost): open the single-file demo with
`?noWasm=1` [TBD if implemented] or simply host without
`sum_core.wasm` present in the same directory — the loader returns
null and the fallback serves.

## Change control

- **Never** add performance language to `README.md` or a commit
  message until the corresponding row of the table above has data.
- **Never** bump `AXIOM_KEY_SEED` in the harness without a CHANGELOG
  entry explaining the break in comparability with prior runs.
- **Never** change the trial count or warm-up count without the same.
- **Never** lose the correctness gate: every benchmark trial still
  confirms WASM-state == JS-state. If that gate fails, that is
  Priority 1 ground, not a benchmark matter.

## Related

- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) — the
  forward playbook that this document discharges Priority 2 of.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — the discipline that
  makes "measured" a first-class label.
- [`Tests/benchmarks/browser_wasm_bench.html`](../Tests/benchmarks/browser_wasm_bench.html)
  — the harness that produces the Browser-WASM and Browser-JS blocks.
- [`scripts/bench_python_derive.py`](../scripts/bench_python_derive.py)
  — the companion that produces the Python-CLI block.
