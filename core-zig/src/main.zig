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

// ─── Tests ──────────────────────────────────────────────────────────

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
    // Must match Python: sympy.nextprime(seed) where seed = big-endian first 8 bytes of SHA-256
    try std.testing.expectEqual(@as(u64, 14326936561644797201), prime);
}
