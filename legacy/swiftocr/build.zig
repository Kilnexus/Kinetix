const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const shared_imports = addSharedImports(b, target, optimize);

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    addSharedImportsToModule(exe_mod, shared_imports);

    const exe = b.addExecutable(.{
        .name = "swiftocr",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run SwiftOCR CLI");
    run_step.dependOn(&run_cmd.step);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    addSharedImportsToModule(test_mod, shared_imports);

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run SwiftOCR unit tests");
    test_step.dependOn(&run_unit_tests.step);
}

const SharedImports = struct {
    memory_pool: *std.Build.Module,
    ocr_model: *std.Build.Module,
    ocr_image: *std.Build.Module,
};

fn addSharedImports(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) SharedImports {
    return .{
        .memory_pool = b.createModule(.{
            .root_source_file = b.path("../../engine/core/memory/arena_pool.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .ocr_model = b.createModule(.{
            .root_source_file = b.path("../../engine/artifacts/ocr/model.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .ocr_image = b.createModule(.{
            .root_source_file = b.path("../../engine/artifacts/ocr/image.zig"),
            .target = target,
            .optimize = optimize,
        }),
    };
}

fn addSharedImportsToModule(module: *std.Build.Module, imports: SharedImports) void {
    module.addImport("engine_arena_pool", imports.memory_pool);
    module.addImport("engine_ocr_model", imports.ocr_model);
    module.addImport("engine_ocr_image", imports.ocr_image);
}
