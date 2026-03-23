const std = @import("std");

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
    const allocator = std.heap.page_allocator;

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
    const allocator = std.heap.page_allocator;

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
    const allocator = std.heap.page_allocator;

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
