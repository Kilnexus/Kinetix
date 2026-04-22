const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const local_pixio = b.option(
        bool,
        "local_pixio",
        "Use local lib/Pixio checkout for development instead of the pinned remote Pixio dependency",
    ) orelse true;
    const runtime_imports = addRuntimeImports(b, target, optimize, local_pixio);

    const kinetix_module = createRootModule(b, target, optimize, runtime_imports, "sdk/kinetix.zig");

    const kinetix_lib = b.addLibrary(.{
        .name = "kinetix",
        .linkage = .static,
        .root_module = kinetix_module,
    });
    b.installArtifact(kinetix_lib);

    const cli_exe = b.addExecutable(.{
        .name = "kinetix",
        .root_module = createRootModule(b, target, optimize, runtime_imports, "apps/cli/entry.zig"),
    });
    b.installArtifact(cli_exe);

    const run_cmd = b.addRunArtifact(cli_exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the Kinetix CLI");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{ .root_module = createRootModule(b, target, optimize, runtime_imports, "sdk/kinetix.zig") });

    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run Kinetix engine tests");
    test_step.dependOn(&run_tests.step);
}

const RuntimeImports = struct {
    engine_root: *std.Build.Module,
    sdk_execution: *std.Build.Module,
    kinetix_sdk: *std.Build.Module,
    pixio: *std.Build.Module,
    graph: *std.Build.Module,
    engine_vision_inspect: *std.Build.Module,
    engine_vision_base: *std.Build.Module,
    engine_vision_reuse_allocator: *std.Build.Module,
    tensor: *std.Build.Module,
    ops: *std.Build.Module,
    weights: *std.Build.Module,
    env: *std.Build.Module,
    stopwatch: *std.Build.Module,
    runtime: *std.Build.Module,
    vision_preprocess: *std.Build.Module,
};

fn createRootModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    imports: RuntimeImports,
    root_source_file: []const u8,
) *std.Build.Module {
    const module = b.createModule(.{
        .root_source_file = b.path(root_source_file),
        .target = target,
        .optimize = optimize,
    });
    addImportsToRoot(module, imports);
    return module;
}

fn addRuntimeImports(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    local_pixio: bool,
) RuntimeImports {
    const engine_root = b.createModule(.{
        .root_source_file = b.path("engine/kinetix.zig"),
        .target = target,
        .optimize = optimize,
    });
    const sdk_execution = b.createModule(.{
        .root_source_file = b.path("sdk/execution/session.zig"),
        .target = target,
        .optimize = optimize,
    });
    sdk_execution.addImport("engine_root", engine_root);
    const kinetix_sdk = b.createModule(.{
        .root_source_file = b.path("sdk/kinetix.zig"),
        .target = target,
        .optimize = optimize,
    });
    kinetix_sdk.addImport("engine_root", engine_root);
    kinetix_sdk.addImport("sdk_execution", sdk_execution);

    const pixio = resolvePixioModule(b, target, optimize, local_pixio);
    engine_root.addImport("Pixio", pixio);

    const graph = b.createModule(.{
        .root_source_file = b.path("engine/artifacts/vision_graph.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tensor = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/nn/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });
    const engine_vision_inspect = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/analysis/inspect.zig"),
        .target = target,
        .optimize = optimize,
    });
    engine_vision_inspect.addImport("graph", graph);
    const engine_vision_base = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/base.zig"),
        .target = target,
        .optimize = optimize,
    });
    engine_vision_base.addImport("graph", graph);
    engine_vision_base.addImport("tensor", tensor);
    const ops = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/nn/ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops.addImport("tensor", tensor);
    engine_vision_base.addImport("ops", ops);

    const weights = b.createModule(.{
        .root_source_file = b.path("engine/artifacts/vision/weights_blob.zig"),
        .target = target,
        .optimize = optimize,
    });
    weights.addImport("graph", graph);
    const env = b.createModule(.{
        .root_source_file = b.path("engine/core/env.zig"),
        .target = target,
        .optimize = optimize,
    });
    const stopwatch = b.createModule(.{
        .root_source_file = b.path("engine/core/stopwatch.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops.addImport("engine_env", env);
    const engine_vision_reuse_allocator = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/memory/reuse_allocator.zig"),
        .target = target,
        .optimize = optimize,
    });

    const runtime = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    runtime.addImport("graph", graph);
    runtime.addImport("tensor", tensor);
    runtime.addImport("ops", ops);
    runtime.addImport("weights", weights);
    runtime.addImport("engine_vision_inspect", engine_vision_inspect);
    runtime.addImport("engine_vision_base", engine_vision_base);
    runtime.addImport("engine_vision_reuse_allocator", engine_vision_reuse_allocator);
    runtime.addImport("engine_env", env);
    runtime.addImport("engine_stopwatch", stopwatch);

    const vision_preprocess = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/io/preprocess.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_preprocess.addImport("Pixio", pixio);
    vision_preprocess.addImport("runtime", runtime);

    engine_root.addImport("graph", graph);
    engine_root.addImport("engine_vision_inspect", engine_vision_inspect);
    engine_root.addImport("engine_vision_base", engine_vision_base);
    engine_root.addImport("engine_vision_reuse_allocator", engine_vision_reuse_allocator);
    engine_root.addImport("tensor", tensor);
    engine_root.addImport("ops", ops);
    engine_root.addImport("weights", weights);
    engine_root.addImport("engine_env", env);
    engine_root.addImport("engine_stopwatch", stopwatch);
    engine_root.addImport("runtime", runtime);
    engine_root.addImport("vision", vision_preprocess);

    return .{
        .engine_root = engine_root,
        .sdk_execution = sdk_execution,
        .kinetix_sdk = kinetix_sdk,
        .pixio = pixio,
        .graph = graph,
        .engine_vision_inspect = engine_vision_inspect,
        .engine_vision_base = engine_vision_base,
        .engine_vision_reuse_allocator = engine_vision_reuse_allocator,
        .tensor = tensor,
        .ops = ops,
        .weights = weights,
        .env = env,
        .stopwatch = stopwatch,
        .runtime = runtime,
        .vision_preprocess = vision_preprocess,
    };
}

fn resolvePixioModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    local_pixio: bool,
) *std.Build.Module {
    if (local_pixio) {
        return b.createModule(.{
            .root_source_file = b.path("lib/Pixio/src/Pixio.zig"),
            .target = target,
            .optimize = optimize,
        });
    }

    const pixio_dep = b.dependency("Pixio", .{
        .target = target,
        .optimize = optimize,
    });
    return pixio_dep.module("Pixio");
}

fn addImportsToRoot(root: *std.Build.Module, imports: RuntimeImports) void {
    root.addImport("engine_root", imports.engine_root);
    root.addImport("sdk_execution", imports.sdk_execution);
    root.addImport("kinetix_sdk", imports.kinetix_sdk);
    root.addImport("Pixio", imports.pixio);
    root.addImport("graph", imports.graph);
    root.addImport("engine_vision_inspect", imports.engine_vision_inspect);
    root.addImport("engine_vision_base", imports.engine_vision_base);
    root.addImport("engine_vision_reuse_allocator", imports.engine_vision_reuse_allocator);
    root.addImport("tensor", imports.tensor);
    root.addImport("ops", imports.ops);
    root.addImport("weights", imports.weights);
    root.addImport("engine_env", imports.env);
    root.addImport("engine_stopwatch", imports.stopwatch);
    root.addImport("runtime", imports.runtime);
    root.addImport("vision", imports.vision_preprocess);
}
