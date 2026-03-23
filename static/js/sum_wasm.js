/**
 * SUM WASM Loader — Browser-Native Gödel-State Algebra
 *
 * Loads the Zig-compiled WASM module and provides JS wrappers
 * mirroring the Python zig_bridge.py API.
 *
 * Usage:
 *   const wasm = await SumWasm.init('/static/sum_core.wasm');
 *   const prime = wasm.getDeterministicPrime('alice||likes||cats');
 *   const lcm = wasm.bigintLcm(3n * 5n, 5n * 7n); // 105n
 *   const entailed = wasm.isDivisibleBy(105n, 5n);  // true
 *
 * All values are native BigInt. Zero server round-trips.
 */

class SumWasm {
  constructor(instance) {
    this._instance = instance;
    this._exports = instance.exports;
    this._ready = true;
  }

  /**
   * Initialize the WASM module from a .wasm URL.
   * @param {string} wasmUrl - Path to sum_core.wasm
   * @returns {Promise<SumWasm>}
   */
  static async init(wasmUrl = '/static/sum_core.wasm') {
    try {
      const response = await fetch(wasmUrl);
      if (!response.ok) throw new Error(`Failed to fetch WASM: ${response.status}`);
      const bytes = await response.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(bytes, {
        env: { memory: new WebAssembly.Memory({ initial: 256, maximum: 65536 }) }
      });
      console.log('⚡ SUM WASM Core loaded — browser-native math engaged');
      return new SumWasm(instance);
    } catch (e) {
      console.warn('WASM load failed, falling back to server:', e.message);
      return null;
    }
  }

  // ── Memory helpers ──────────────────────────────────────────

  /** Allocate bytes in WASM linear memory. */
  _alloc(len) {
    const ptr = this._exports.wasm_alloc_bytes(len);
    if (ptr === 0) throw new Error('WASM allocation failed');
    return ptr;
  }

  /** Free bytes in WASM linear memory. */
  _free(ptr, len) {
    this._exports.wasm_free_bytes(ptr, len);
  }

  /** Write a Uint8Array into WASM memory, returning the pointer. */
  _writeBytes(bytes) {
    const ptr = this._alloc(bytes.length);
    const mem = new Uint8Array(this._exports.memory.buffer);
    mem.set(bytes, ptr);
    return ptr;
  }

  /** Read bytes from WASM memory. */
  _readBytes(ptr, len) {
    const mem = new Uint8Array(this._exports.memory.buffer);
    return new Uint8Array(mem.buffer, ptr, len).slice();
  }

  // ── BigInt ↔ Bytes conversion ──────────────────────────────

  /** Convert a JS BigInt to big-endian byte array. */
  static bigintToBytes(n) {
    if (n === 0n) return new Uint8Array([0]);
    const hex = n.toString(16);
    const paddedHex = hex.length % 2 === 0 ? hex : '0' + hex;
    const bytes = new Uint8Array(paddedHex.length / 2);
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = parseInt(paddedHex.substring(i * 2, i * 2 + 2), 16);
    }
    return bytes;
  }

  /** Convert big-endian byte array to JS BigInt. */
  static bytesToBigint(bytes) {
    if (bytes.length === 0) return 0n;
    let result = 0n;
    for (const byte of bytes) {
      result = (result << 8n) | BigInt(byte);
    }
    return result;
  }

  // ── Public API ─────────────────────────────────────────────

  /**
   * Deterministic prime from axiom string.
   * @param {string} axiom - e.g. "alice||likes||cats"
   * @returns {BigInt} The deterministic prime
   */
  getDeterministicPrime(axiom) {
    const encoder = new TextEncoder();
    const axiomBytes = encoder.encode(axiom);
    const ptr = this._writeBytes(axiomBytes);
    const result = this._exports.sum_get_deterministic_prime(ptr, axiomBytes.length);
    this._free(ptr, axiomBytes.length);
    return BigInt(result);
  }

  /**
   * Call a BigInt binary operation (LCM, GCD, mod).
   * @private
   */
  _callBigintBinary(fnName, a, b) {
    const aBytes = SumWasm.bigintToBytes(a);
    const bBytes = SumWasm.bigintToBytes(b);
    const aPtr = this._writeBytes(aBytes);
    const bPtr = this._writeBytes(bBytes);

    // Allocate output buffer (generous: max input size * 2 + 64)
    const outCap = Math.max(aBytes.length, bBytes.length) * 2 + 64;
    const outPtr = this._alloc(outCap);
    const outLenPtr = this._alloc(8); // usize

    const fn = this._exports[fnName];
    const rc = fn(aPtr, aBytes.length, bPtr, bBytes.length, outPtr, outCap, outLenPtr);

    if (rc !== 0) {
      this._free(aPtr, aBytes.length);
      this._free(bPtr, bBytes.length);
      this._free(outPtr, outCap);
      this._free(outLenPtr, 8);
      throw new Error(`${fnName} failed with code ${rc}`);
    }

    // Read output length
    const lenView = new DataView(this._exports.memory.buffer);
    const outLen = Number(lenView.getBigUint64(outLenPtr, true)); // little-endian on WASM

    const resultBytes = this._readBytes(outPtr, outLen);

    // Cleanup
    this._free(aPtr, aBytes.length);
    this._free(bPtr, bBytes.length);
    this._free(outPtr, outCap);
    this._free(outLenPtr, 8);

    return SumWasm.bytesToBigint(resultBytes);
  }

  /**
   * Compute LCM of two BigInts.
   * @param {BigInt} a
   * @param {BigInt} b
   * @returns {BigInt}
   */
  bigintLcm(a, b) {
    return this._callBigintBinary('sum_bigint_lcm', a, b);
  }

  /**
   * Compute GCD of two BigInts.
   * @param {BigInt} a
   * @param {BigInt} b
   * @returns {BigInt}
   */
  bigintGcd(a, b) {
    return this._callBigintBinary('sum_bigint_gcd', a, b);
  }

  /**
   * Compute a % b (BigInt modulo).
   * @param {BigInt} a
   * @param {BigInt} b
   * @returns {BigInt}
   */
  bigintMod(a, b) {
    return this._callBigintBinary('sum_bigint_mod', a, b);
  }

  /**
   * Check if state is divisible by a prime (u64).
   * @param {BigInt} state - The Gödel state
   * @param {number|BigInt} prime - The prime to check (must fit in u64)
   * @returns {boolean}
   */
  isDivisibleBy(state, prime) {
    const stateBytes = SumWasm.bigintToBytes(state);
    const statePtr = this._writeBytes(stateBytes);
    const rc = this._exports.sum_bigint_divisible_by_u64(
      statePtr, stateBytes.length, Number(prime)
    );
    this._free(statePtr, stateBytes.length);
    return rc === 1;
  }

  /**
   * Verify entailment: does the global state contain the hypothesis?
   * @param {BigInt} globalState
   * @param {BigInt} hypothesis
   * @returns {boolean}
   */
  verifyEntailment(globalState, hypothesis) {
    if (hypothesis === 0n) return false;

    // For u64 primes, use optimized streaming check
    if (hypothesis < (1n << 64n)) {
      return this.isDivisibleBy(globalState, hypothesis);
    }

    // For composite hypotheses, use full BigInt modulo
    return this.bigintMod(globalState, hypothesis) === 0n;
  }

  /**
   * Merge parallel states via LCM.
   * @param {BigInt[]} states
   * @returns {BigInt}
   */
  mergeParallelStates(states) {
    if (states.length === 0) return 1n;
    let result = states[0];
    for (let i = 1; i < states.length; i++) {
      result = this.bigintLcm(result, states[i]);
    }
    return result;
  }
}

// Export for both module and global scope
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SumWasm;
} else {
  window.SumWasm = SumWasm;
}
