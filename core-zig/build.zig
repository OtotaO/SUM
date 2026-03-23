const std = @import("std");

pub fn build(b: *std.Build) void {
    // ── Native shared library (for Python ctypes FFI) ──
    const native_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "sum_core",
        .root_source_file = b.path("src/main.zig"),
        .target = native_target,
        .optimize = optimize,
    });
    lib.linkLibC();
    b.installArtifact(lib);

    // ── WASM module (for browser-native math) ──
    const wasm = b.addExecutable(.{
        .name = "sum_core",
        .root_source_file = b.path("src/main.zig"),
        .target = b.resolveTargetQuery(.{
            .cpu_arch = .wasm32,
            .os_tag = .freestanding,
        }),
        .optimize = .ReleaseSmall,
    });
    wasm.rdynamic = true;
    wasm.entry = .disabled;

    const wasm_install = b.addInstallArtifact(wasm, .{
        .dest_sub_path = "sum_core.wasm",
    });

    const wasm_step = b.step("wasm", "Build the WASM module for browser-native math");
    wasm_step.dependOn(&wasm_install.step);

    // ── Tests ──
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = native_target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
