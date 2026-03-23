const std = @import("std");
const builtin = @import("builtin");

// ─── Comptime Allocator Selection ────────────────────────────────────
//
// On native targets (Linux, macOS, Windows), we use the page allocator
// for zero-heap-leak operation.  On WASM (freestanding), we use the
// WASM-specific allocator backed by memory.grow.

const wasm_alloc = if (builtin.target.os.tag == .freestanding)
    std.heap.wasm_allocator
else
    std.heap.page_allocator;

// ─── WASM Memory Exports ─────────────────────────────────────────────
//
// JavaScript needs to allocate/free buffers in WASM linear memory.

export fn wasm_alloc_bytes(len: usize) ?[*]u8 {
    const slice = wasm_alloc.alloc(u8, len) catch return null;
    return slice.ptr;
}

export fn wasm_free_bytes(ptr: [*]u8, len: usize) void {
    wasm_alloc.free(ptr[0..len]);
}

// ─── Modular Exponentiation ──────────────────────────────────────────
//
// Computes (base^exponent) % modulus using binary exponentiation with
// u128 intermediates to prevent overflow for 64-bit operands.

fn modPow(base: u64, exponent: u64, modulus: u64) u64 {
    if (modulus == 1) return 0;
    var res: u128 = 1;
    var b: u128 = @as(u128, base) % @as(u128, modulus);
    var e: u64 = exponent;

    while (e > 0) {
        if (e % 2 == 1) {
            res = (res * b) % @as(u128, modulus);
        }
        e >>= 1;
        b = (b * b) % @as(u128, modulus);
    }
    return @as(u64, @intCast(res));
}

// ─── Deterministic Miller-Rabin ──────────────────────────────────────
//
// For 64-bit integers, using the first 12 primes as bases is provably
// deterministic (no false positives).  This is NOT probabilistic.

fn isProbablePrime(n: u64) bool {
    if (n == 2 or n == 3) return true;
    if (n <= 1 or n % 2 == 0) return false;

    // Write n-1 as 2^s * d
    var d: u64 = n - 1;
    var s: u64 = 0;
    while (d % 2 == 0) {
        d >>= 1;
        s += 1;
    }

    // Deterministic witness set for all 64-bit integers
    const bases = [_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    for (bases) |a| {
        if (n <= a) break;
        var x = modPow(a, d, n);
        if (x == 1 or x == n - 1) continue;

        var composite = true;
        var r: u64 = 1;
        while (r < s) : (r += 1) {
            x = modPow(x, 2, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

// ─── Next Prime ──────────────────────────────────────────────────────

fn nextPrime(n: u64) u64 {
    var p = n + 1;
    if (p % 2 == 0 and p > 2) p += 1;
    while (!isProbablePrime(p)) {
        p += 2;
    }
    return p;
}

// ─── C-ABI Exports ──────────────────────────────────────────────────
//
// These functions are callable from Python via ctypes, or from any
// language that supports the C calling convention.

/// Deterministic prime derivation: SHA-256(axiom) → 8-byte seed → nextPrime.
/// Produces the exact same prime as the Python implementation for any input.
export fn sum_get_deterministic_prime(axiom_ptr: [*c]const u8, axiom_len: usize) u64 {
    const axiom_str = axiom_ptr[0..axiom_len];

    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(axiom_str, &hash, .{});

    // Extract first 8 bytes as big-endian u64 seed
    var seed: u64 = 0;
    for (hash[0..8]) |byte| {
        seed = (seed << 8) | byte;
    }

    return nextPrime(seed);
}

// ═══════════════════════════════════════════════════════════════════════
// STAGE 3 — sha256_128_v2: BPSW Primality for u128
//
// Baillie-PSW = strong base-2 Miller–Rabin + Strong Lucas (Selfridge A)
// No known BPSW pseudoprime exists. This is an engineering trust
// assumption, not a formal proof of primality correctness.
// ═══════════════════════════════════════════════════════════════════════

/// Modular exponentiation for u128 using u256 intermediates.
fn modPow128(base: u128, exponent: u128, modulus: u128) u128 {
    if (modulus == 1) return 0;
    var res: u256 = 1;
    var b: u256 = @as(u256, base) % @as(u256, modulus);
    var e: u128 = exponent;
    while (e > 0) {
        if (e % 2 == 1) {
            res = (res * b) % @as(u256, modulus);
        }
        e >>= 1;
        b = (b * b) % @as(u256, modulus);
    }
    return @as(u128, @intCast(res));
}

/// Signed modular reduction for i512: returns value in [0, n-1].
fn signedMod512(val: i512, n: u128) u128 {
    const n_wide: i512 = @as(i512, n);
    const r = @mod(val, n_wide);
    return @as(u128, @intCast(r));
}

/// Integer square root of u128 via Newton's method.
fn isqrt128(n: u128) u128 {
    if (n < 2) return n;
    var x: u128 = n;
    var y: u128 = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

/// Jacobi symbol (a/n) for u128. n must be odd > 0.
/// Returns -1, 0, or 1.
fn jacobi128(a_in: u128, n_in: u128) i32 {
    var a: u128 = a_in % n_in;
    var n: u128 = n_in;
    var result: i32 = 1;
    while (a != 0) {
        while (a % 2 == 0) {
            a /= 2;
            const n_mod_8 = n % 8;
            if (n_mod_8 == 3 or n_mod_8 == 5) result = -result;
        }
        const tmp = a;
        a = n;
        n = tmp;
        if (a % 4 == 3 and n % 4 == 3) result = -result;
        a = a % n;
    }
    return if (n == 1) result else 0;
}

/// Jacobi symbol for signed D values. Handles negative D by
/// computing Jacobi(D mod n, n).
fn jacobiSigned(d: i32, n: u128) i32 {
    if (d >= 0) {
        return jacobi128(@as(u128, @intCast(d)), n);
    } else {
        // d mod n = n - (|d| mod n), unless |d| mod n == 0
        const abs_d: u128 = @as(u128, @intCast(-d));
        const r = abs_d % n;
        if (r == 0) return 0;
        return jacobi128(n - r, n);
    }
}

/// Selfridge Method A: find first D in {5, -7, 9, -11, 13, ...}
/// where Jacobi(D|n) = -1. Returns D value or 0 if not found.
fn selfridgeD(n: u128) i32 {
    var d: i32 = 5;
    var sign: i32 = 1;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const j = jacobiSigned(d, n);
        if (j == -1) return d;
        if (j == 0) {
            const abs_d: u128 = if (d >= 0) @as(u128, @intCast(d)) else @as(u128, @intCast(-d));
            if (abs_d < n) return 0; // n has a factor
        }
        sign = -sign;
        const abs_d_next = (if (d >= 0) d else -d) + 2;
        d = sign * abs_d_next;
    }
    return 0;
}

/// Strong Lucas probable prime test for u128.
/// Uses Selfridge Method A parameters.
/// All intermediates use i512 to prevent overflow on u128-scale products.
fn strongLucasTest128(n: u128) bool {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    // Perfect square check
    const s_root = isqrt128(n);
    if (s_root * s_root == n) return false;

    const d_val = selfridgeD(n);
    if (d_val == 0) return false;

    // P = 1, Q = (1 - D) / 4
    const d_wide: i512 = @as(i512, d_val);
    const q_signed: i512 = @divExact(1 - d_wide, 4);

    // n + 1 = 2^s * d_odd
    var d_odd: u128 = n + 1;
    if (d_odd == 0) return false; // overflow check (n = max u128)
    var s: u32 = 0;
    while (d_odd % 2 == 0) {
        d_odd /= 2;
        s += 1;
    }

    // Lucas sequence U_d, V_d (mod n) using doubling method
    // i512 intermediates prevent overflow: max product is ~(2^128)^2 = 2^256,
    // which fits comfortably in i512 (max 2^511-1).
    var u_val: i512 = 1;
    var v_val: i512 = 1; // P = 1
    var qk: i512 = q_signed;
    const n_wide: i512 = @as(i512, n);

    // Binary expansion of d_odd, process from second-most-significant bit
    var bit_pos: u7 = 127;
    // Find the MSB of d_odd
    while (bit_pos > 0 and (d_odd >> @as(u7, bit_pos)) == 0) {
        bit_pos -= 1;
    }
    // Skip MSB, process remaining bits
    if (bit_pos > 0) {
        bit_pos -= 1;
        var remaining: u32 = bit_pos + 1;
        while (remaining > 0) {
            remaining -= 1;
            const bp: u7 = @as(u7, @intCast(remaining));

            // Double step
            u_val = @mod(u_val * v_val, n_wide);
            v_val = @mod(v_val * v_val - 2 * qk, n_wide);
            qk = @mod(qk * qk, n_wide);

            if ((d_odd >> bp) & 1 == 1) {
                // Add step: U' = (P*U + V)/2, V' = (D*U + P*V)/2
                const new_u = u_val + v_val; // P=1, so P*U + V = U + V
                const new_v = d_wide * u_val + v_val; // P=1, so D*U + P*V = D*U + V
                // Divide by 2 mod n: if odd, add n first
                u_val = if (@mod(new_u, 2) == 0)
                    @mod(@divExact(new_u, 2), n_wide)
                else
                    @mod(@divExact(new_u + n_wide, 2), n_wide);
                v_val = if (@mod(new_v, 2) == 0)
                    @mod(@divExact(new_v, 2), n_wide)
                else
                    @mod(@divExact(new_v + n_wide, 2), n_wide);
                qk = @mod(qk * q_signed, n_wide);
            }
        }
    }

    // Normalize
    const u_final = signedMod512(u_val, n);
    const v_final = signedMod512(v_val, n);

    // Strong Lucas: U_d ≡ 0 OR V_{d*2^r} ≡ 0 for some 0 <= r < s
    if (u_final == 0 or v_final == 0) return true;

    // Check V_{d*2^r} for r = 1..s-1
    var v_cur: i512 = v_val;
    var qk_cur: i512 = qk;
    var r: u32 = 1;
    while (r < s) : (r += 1) {
        v_cur = @mod(v_cur * v_cur - 2 * qk_cur, n_wide);
        qk_cur = @mod(qk_cur * qk_cur, n_wide);
        if (signedMod512(v_cur, n) == 0) return true;
    }

    return false;
}

/// BPSW primality test for u128.
/// Phase 1: strong base-2 Miller-Rabin
/// Phase 2: Strong Lucas with Selfridge Method A
fn isPrimeBPSW128(n: u128) bool {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 or n % 3 == 0) return false;

    // Small primes fast-path
    const small_primes = [_]u128{ 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43 };
    for (small_primes) |sp| {
        if (n == sp) return true;
        if (n % sp == 0) return false;
    }

    // Phase 1: Strong base-2 Miller-Rabin
    var d: u128 = n - 1;
    var s: u32 = 0;
    while (d % 2 == 0) {
        d /= 2;
        s += 1;
    }

    var x = modPow128(2, d, n);
    if (x != 1 and x != n - 1) {
        var composite = true;
        var r: u32 = 1;
        while (r < s) : (r += 1) {
            x = modPow128(x, 2, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }

    // Phase 2: Strong Lucas
    return strongLucasTest128(n);
}

/// Next prime after n, using BPSW for u128 inputs.
fn nextPrime128(n: u128) u128 {
    var p = n + 1;
    if (p % 2 == 0 and p > 2) p += 1;
    while (!isPrimeBPSW128(p)) {
        p += 2;
    }
    return p;
}

/// Stage 3 v2 C-ABI export: SHA-256(axiom) → 16-byte seed → nextPrime128.
/// Result is written as 16-byte big-endian into caller-provided buffer.
/// Returns 0 on success, -1 on error.
export fn sum_get_deterministic_prime_v2(
    axiom_ptr: [*c]const u8,
    axiom_len: usize,
    out_buf: [*c]u8,
) i32 {
    const axiom_str = axiom_ptr[0..axiom_len];

    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(axiom_str, &hash, .{});

    // Extract first 16 bytes as big-endian u128 seed
    var seed: u128 = 0;
    for (hash[0..16]) |byte| {
        seed = (seed << 8) | byte;
    }

    const prime = nextPrime128(seed);

    // Write as 16-byte big-endian
    var i: u5 = 0;
    while (i < 16) : (i += 1) {
        const shift: u7 = @as(u7, 15 - @as(u7, i)) * 8;
        out_buf[i] = @as(u8, @intCast((prime >> shift) & 0xFF));
    }

    return 0;
}

// ─── Tests (Phase 17 — Prime Derivation) ────────────────────────────

test "modPow basic" {
    try std.testing.expectEqual(@as(u64, 1), modPow(2, 10, 1023));
    try std.testing.expectEqual(@as(u64, 0), modPow(5, 3, 1));
    try std.testing.expectEqual(@as(u64, 8), modPow(2, 3, 1000));
}

test "isProbablePrime known primes" {
    try std.testing.expect(isProbablePrime(2));
    try std.testing.expect(isProbablePrime(3));
    try std.testing.expect(isProbablePrime(7));
    try std.testing.expect(isProbablePrime(104729));
    try std.testing.expect(!isProbablePrime(0));
    try std.testing.expect(!isProbablePrime(1));
    try std.testing.expect(!isProbablePrime(4));
    try std.testing.expect(!isProbablePrime(100));
}

test "nextPrime basic" {
    try std.testing.expectEqual(@as(u64, 5), nextPrime(3));
    try std.testing.expectEqual(@as(u64, 11), nextPrime(7));
    try std.testing.expectEqual(@as(u64, 2), nextPrime(1));
}

test "deterministic prime for 'alice||likes||cats'" {
    const axiom = "alice||likes||cats";
    const prime = sum_get_deterministic_prime(axiom.ptr, axiom.len);
    try std.testing.expectEqual(@as(u64, 14326936561644797201), prime);
}

// ─── Tests (Stage 3 — v2 BPSW 128-bit) ─────────────────────────────

test "modPow128 basic" {
    try std.testing.expectEqual(@as(u128, 1), modPow128(2, 10, 1023));
    try std.testing.expectEqual(@as(u128, 0), modPow128(5, 3, 1));
    try std.testing.expectEqual(@as(u128, 8), modPow128(2, 3, 1000));
}

test "isPrimeBPSW128 known primes" {
    try std.testing.expect(isPrimeBPSW128(2));
    try std.testing.expect(isPrimeBPSW128(3));
    try std.testing.expect(isPrimeBPSW128(7));
    try std.testing.expect(isPrimeBPSW128(104729));
    try std.testing.expect(!isPrimeBPSW128(0));
    try std.testing.expect(!isPrimeBPSW128(1));
    try std.testing.expect(!isPrimeBPSW128(4));
    try std.testing.expect(!isPrimeBPSW128(100));
}

test "isPrimeBPSW128 large prime" {
    // Test with the v2 reference prime for alice||likes||cats
    try std.testing.expect(isPrimeBPSW128(264285332112933860981052902103273947671));
}

test "nextPrime128 basic" {
    try std.testing.expectEqual(@as(u128, 5), nextPrime128(3));
    try std.testing.expectEqual(@as(u128, 11), nextPrime128(7));
    try std.testing.expectEqual(@as(u128, 2), nextPrime128(1));
}

test "v2 deterministic prime for 'alice||likes||cats'" {
    const axiom = "alice||likes||cats";
    var out: [16]u8 = undefined;
    const rc = sum_get_deterministic_prime_v2(axiom.ptr, axiom.len, &out);
    try std.testing.expectEqual(@as(i32, 0), rc);

    // Reconstruct u128 from big-endian bytes
    var prime: u128 = 0;
    for (out[0..16]) |byte| {
        prime = (prime << 8) | byte;
    }
    try std.testing.expectEqual(@as(u128, 264285332112933860981052902103273947671), prime);
}

// ═══════════════════════════════════════════════════════════════════════
// PHASE 17b — BigInt Arithmetic C-ABI Exports
//
// Arbitrary-precision LCM, GCD, modulo, and divisibility operations
// using Zig's std.math.big.int.Managed. These are the mathematical
// heartbeat of the Gödel-State algebra.
//
// Protocol:
//   - Inputs: big-endian byte arrays representing unsigned integers
//   - Output: big-endian byte array written to caller-provided buffer
//   - Return: 0 = success, -1 = buffer too small, -2 = internal error
//   - out_len: set to actual byte count of result
// ═══════════════════════════════════════════════════════════════════════

const Managed = std.math.big.int.Managed;

/// Import a big-endian byte slice into a Managed big integer.
fn bigintFromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Managed {
    var result = try Managed.init(allocator);
    errdefer result.deinit();

    if (bytes.len == 0) {
        try result.set(0);
        return result;
    }

    // Build the integer from big-endian bytes
    try result.set(0);
    for (bytes) |byte| {
        // result = result * 256 + byte
        var temp = try Managed.init(allocator);
        defer temp.deinit();
        try temp.set(256);
        try result.mul(&result, &temp);

        var byte_val = try Managed.init(allocator);
        defer byte_val.deinit();
        try byte_val.set(@as(u64, byte));
        try result.add(&result, &byte_val);
    }

    return result;
}

/// Export a Managed big integer to big-endian bytes in a caller buffer.
/// Returns the number of bytes written, or error.
fn bigintToBytes(val: *const Managed, out: []u8) !usize {
    // Handle zero
    if (val.eqlZero()) {
        if (out.len < 1) return error.BufferTooSmall;
        out[0] = 0;
        return 1;
    }

    const allocator = val.allocator;

    // Extract bytes by repeated divmod 256
    var temp = try val.clone();
    defer temp.deinit();

    var divisor = try Managed.initSet(allocator, 256);
    defer divisor.deinit();

    var bytes_buf: [65536]u8 = undefined;
    var byte_count: usize = 0;

    while (!temp.eqlZero()) {
        if (byte_count >= bytes_buf.len) return error.BufferTooSmall;

        var quotient = try Managed.init(allocator);
        defer quotient.deinit();
        var remainder = try Managed.init(allocator);
        defer remainder.deinit();

        try quotient.divFloor(&remainder, &temp, &divisor);

        // Get the remainder as u8
        const rem_limbs = remainder.limbs;
        const rem_byte: u8 = if (remainder.eqlZero()) 0 else @as(u8, @intCast(rem_limbs[0] & 0xFF));
        bytes_buf[byte_count] = rem_byte;
        byte_count += 1;

        try temp.copy(quotient.toConst());
    }

    if (byte_count > out.len) return error.BufferTooSmall;

    // Reverse to big-endian
    var i: usize = 0;
    while (i < byte_count) : (i += 1) {
        out[i] = bytes_buf[byte_count - 1 - i];
    }

    return byte_count;
}

/// BigInt GCD: gcd(a, b) → result bytes.
/// Returns 0 on success, -1 if buffer too small, -2 on error.
export fn sum_bigint_gcd(
    a_ptr: [*c]const u8,
    a_len: usize,
    b_ptr: [*c]const u8,
    b_len: usize,
    out_ptr: [*c]u8,
    out_cap: usize,
    out_len: *usize,
) i32 {
    const allocator = wasm_alloc;

    var a = bigintFromBytes(allocator, a_ptr[0..a_len]) catch return -2;
    defer a.deinit();
    var b = bigintFromBytes(allocator, b_ptr[0..b_len]) catch return -2;
    defer b.deinit();

    // GCD via Euclidean algorithm
    var result = Managed.init(allocator) catch return -2;
    defer result.deinit();
    result.gcd(&a, &b) catch return -2;

    const written = bigintToBytes(&result, out_ptr[0..out_cap]) catch return -1;
    out_len.* = written;
    return 0;
}

/// BigInt LCM: lcm(a, b) = a * b / gcd(a, b) → result bytes.
/// Returns 0 on success, -1 if buffer too small, -2 on error.
export fn sum_bigint_lcm(
    a_ptr: [*c]const u8,
    a_len: usize,
    b_ptr: [*c]const u8,
    b_len: usize,
    out_ptr: [*c]u8,
    out_cap: usize,
    out_len: *usize,
) i32 {
    const allocator = wasm_alloc;

    var a = bigintFromBytes(allocator, a_ptr[0..a_len]) catch return -2;
    defer a.deinit();
    var b = bigintFromBytes(allocator, b_ptr[0..b_len]) catch return -2;
    defer b.deinit();

    // gcd(a, b)
    var g = Managed.init(allocator) catch return -2;
    defer g.deinit();
    g.gcd(&a, &b) catch return -2;

    if (g.eqlZero()) {
        // LCM(0, x) = 0
        out_ptr[0] = 0;
        out_len.* = 1;
        return 0;
    }

    // product = a * b
    var product = Managed.init(allocator) catch return -2;
    defer product.deinit();
    product.mul(&a, &b) catch return -2;

    // lcm = product / gcd
    var result = Managed.init(allocator) catch return -2;
    defer result.deinit();
    var rem = Managed.init(allocator) catch return -2;
    defer rem.deinit();
    result.divFloor(&rem, &product, &g) catch return -2;

    // Ensure positive
    result.setSign(true);

    const written = bigintToBytes(&result, out_ptr[0..out_cap]) catch return -1;
    out_len.* = written;
    return 0;
}

/// BigInt modulo: a % b → result bytes.
/// Returns 0 on success, -1 if buffer too small, -2 on error.
export fn sum_bigint_mod(
    a_ptr: [*c]const u8,
    a_len: usize,
    b_ptr: [*c]const u8,
    b_len: usize,
    out_ptr: [*c]u8,
    out_cap: usize,
    out_len: *usize,
) i32 {
    const allocator = wasm_alloc;

    var a = bigintFromBytes(allocator, a_ptr[0..a_len]) catch return -2;
    defer a.deinit();
    var b = bigintFromBytes(allocator, b_ptr[0..b_len]) catch return -2;
    defer b.deinit();

    var quotient = Managed.init(allocator) catch return -2;
    defer quotient.deinit();
    var remainder = Managed.init(allocator) catch return -2;
    defer remainder.deinit();

    quotient.divFloor(&remainder, &a, &b) catch return -2;

    const written = bigintToBytes(&remainder, out_ptr[0..out_cap]) catch return -1;
    out_len.* = written;
    return 0;
}

/// Optimized: is state divisible by a 64-bit prime?
/// Returns 1 if divisible, 0 if not, -2 on error.
/// This avoids constructing a BigInt for the divisor.
export fn sum_bigint_divisible_by_u64(
    a_ptr: [*c]const u8,
    a_len: usize,
    prime: u64,
) i32 {
    if (prime == 0) return -2;

    // Compute a mod prime using streaming modular arithmetic
    // No BigInt needed for the divisor — just accumulate mod
    var remainder: u128 = 0;
    const bytes = a_ptr[0..a_len];
    for (bytes) |byte| {
        remainder = (remainder * 256 + byte) % prime;
    }

    return if (remainder == 0) 1 else 0;
}

// ═══════════════════════════════════════════════════════════════════════
// PHASE 18 — Batch Prime Minting
//
// Amortizes FFI overhead by processing N axioms in a single C-ABI call.
// Input:  flat buffer of null-terminated axiom strings
// Output: array of u64 primes written to caller buffer
// ═══════════════════════════════════════════════════════════════════════

/// Batch-mint deterministic primes from null-terminated axiom strings.
/// Input: concatenated null-terminated strings (e.g. "alice||a||b\x00bob||c||d\x00")
/// Output: u64 primes written to out_primes buffer.
/// Returns number of primes minted, or -1 on error.
export fn sum_batch_mint_primes(
    axioms_ptr: [*c]const u8,
    axioms_len: usize,
    out_primes: [*c]u64,
    out_cap: usize,
) i32 {
    const axioms = axioms_ptr[0..axioms_len];
    var count: usize = 0;
    var start: usize = 0;

    for (axioms, 0..) |byte, i| {
        if (byte == 0) {
            if (i > start and count < out_cap) {
                const axiom_slice = axioms[start..i];
                out_primes[count] = sum_get_deterministic_prime(axiom_slice.ptr, axiom_slice.len);
                count += 1;
            }
            start = i + 1;
        }
    }

    // Handle last axiom if no trailing null
    if (start < axioms_len and count < out_cap) {
        const axiom_slice = axioms[start..axioms_len];
        out_primes[count] = sum_get_deterministic_prime(axiom_slice.ptr, axiom_slice.len);
        count += 1;
    }

    return @as(i32, @intCast(count));
}

