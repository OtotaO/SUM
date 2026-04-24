const std = @import("std");

pub fn build(b: *std.Build) void {
    // ── Native shared library (for Python ctypes FFI) ──
    const native_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "sum_core",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = native_target,
            .optimize = optimize,
            // zig 0.16 moved libc linkage onto the module (it was a
            // method on Compile in 0.15). The field exists on both
            // 0.15.x (added late-cycle) and 0.16, so this form builds
            // against either — mlugg/setup-zig@v2 pins our CI to
            // 0.15.2 which supports the field-form.
            .link_libc = true,
        }),
        .linkage = .dynamic,
    });
    b.installArtifact(lib);

    // ── WASM module (for browser-native math) ──
    const wasm = b.addExecutable(.{
        .name = "sum_core",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .wasm32,
                .os_tag = .freestanding,
            }),
            .optimize = .ReleaseSmall,
        }),
    });
    wasm.root_module.export_symbol_names = &.{
        "wasm_alloc_bytes",
        "wasm_free_bytes",
        "sum_get_deterministic_prime",
        "sum_get_deterministic_prime_v2",
        "sum_bigint_gcd",
        "sum_bigint_lcm",
        "sum_bigint_mod",
        "sum_bigint_divisible_by_u64",
        "sum_batch_mint_primes",
    };
    wasm.entry = .disabled;

    const wasm_install = b.addInstallArtifact(wasm, .{
        .dest_sub_path = "sum_core.wasm",
    });

    const wasm_step = b.step("wasm", "Build the WASM module for browser-native math");
    wasm_step.dependOn(&wasm_install.step);

    // ── Tests ──
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = native_target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
