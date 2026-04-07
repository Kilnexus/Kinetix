const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const legacy_imports = addLegacyImports(b, target, optimize);

    const kinetix_module = createRootModule(b, target, optimize, legacy_imports, "kinetix.zig");

    const kinetix_lib = b.addLibrary(.{
        .name = "kinetix",
        .linkage = .static,
        .root_module = kinetix_module,
    });
    b.installArtifact(kinetix_lib);

    const cli_exe = b.addExecutable(.{
        .name = "kinetix",
        .root_module = createRootModule(b, target, optimize, legacy_imports, "kinetix_cli.zig"),
    });
    b.installArtifact(cli_exe);

    const run_cmd = b.addRunArtifact(cli_exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the Kinetix CLI");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{ .root_module = createRootModule(b, target, optimize, legacy_imports, "kinetix.zig") });

    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run Kinetix engine tests");
    test_step.dependOn(&run_tests.step);
}

const LegacyImports = struct {
    pixio: *std.Build.Module,
    graph: *std.Build.Module,
    engine_vision_graph: *std.Build.Module,
    engine_vision_inspect: *std.Build.Module,
    engine_vision_base: *std.Build.Module,
    engine_vision_modules: *std.Build.Module,
    engine_vision_reuse_allocator: *std.Build.Module,
    engine_vision_engine: *std.Build.Module,
    tensor: *std.Build.Module,
    ops: *std.Build.Module,
    weights: *std.Build.Module,
    global_thread_pool: *std.Build.Module,
    runtime: *std.Build.Module,
    legacy_vision: *std.Build.Module,
};

fn createRootModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    imports: LegacyImports,
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

fn addLegacyImports(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) LegacyImports {
    const pixio = b.createModule(.{
        .root_source_file = b.path("../Pixio/src/Pixio.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = if (target.result.os.tag == .windows) true else null,
    });
    if (target.result.os.tag == .windows) {
        pixio.linkSystemLibrary("ole32", .{});
        pixio.linkSystemLibrary("windowscodecs", .{});
    }

    const graph = b.createModule(.{
        .root_source_file = b.path("legacy/axionyx/src/format/graph.zig"),
        .target = target,
        .optimize = optimize,
    });
    const engine_vision_graph = b.createModule(.{
        .root_source_file = b.path("engine/artifacts/vision_graph.zig"),
        .target = target,
        .optimize = optimize,
    });
    graph.addImport("engine_vision_graph", engine_vision_graph);
    const tensor = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/nn/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });
    const engine_vision_inspect = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/inspect.zig"),
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
    const global_thread_pool = b.createModule(.{
        .root_source_file = b.path("engine/core/threading/global_thread_pool_module.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops.addImport("engine_global_thread_pool", global_thread_pool);
    const engine_vision_modules = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/modules/modules.zig"),
        .target = target,
        .optimize = optimize,
    });
    engine_vision_modules.addImport("graph", graph);
    engine_vision_modules.addImport("weights", weights);
    engine_vision_modules.addImport("ops", ops);
    engine_vision_modules.addImport("tensor", tensor);
    engine_vision_modules.addImport("engine_global_thread_pool", global_thread_pool);
    engine_vision_modules.addImport("engine_vision_base", engine_vision_base);
    const engine_vision_reuse_allocator = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/reuse_allocator.zig"),
        .target = target,
        .optimize = optimize,
    });
    const engine_vision_engine = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/engine.zig"),
        .target = target,
        .optimize = optimize,
    });
    engine_vision_engine.addImport("graph", graph);
    engine_vision_engine.addImport("weights", weights);
    engine_vision_engine.addImport("ops", ops);
    engine_vision_engine.addImport("tensor", tensor);
    engine_vision_engine.addImport("engine_vision_base", engine_vision_base);
    engine_vision_engine.addImport("engine_vision_modules", engine_vision_modules);
    engine_vision_engine.addImport("engine_vision_reuse_allocator", engine_vision_reuse_allocator);

    const runtime = b.createModule(.{
        .root_source_file = b.path("legacy/axionyx/src/runtime/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    runtime.addImport("graph", graph);
    runtime.addImport("tensor", tensor);
    runtime.addImport("ops", ops);
    runtime.addImport("weights", weights);
    runtime.addImport("engine_global_thread_pool", global_thread_pool);
    runtime.addImport("engine_vision_inspect", engine_vision_inspect);
    runtime.addImport("engine_vision_base", engine_vision_base);
    runtime.addImport("engine_vision_modules", engine_vision_modules);
    runtime.addImport("engine_vision_engine", engine_vision_engine);

    const legacy_vision = b.createModule(.{
        .root_source_file = b.path("engine/runtime/vision/preprocess.zig"),
        .target = target,
        .optimize = optimize,
    });
    legacy_vision.addImport("Pixio", pixio);
    legacy_vision.addImport("runtime", runtime);

    return .{
        .pixio = pixio,
        .graph = graph,
        .engine_vision_graph = engine_vision_graph,
        .engine_vision_inspect = engine_vision_inspect,
        .engine_vision_base = engine_vision_base,
        .engine_vision_modules = engine_vision_modules,
        .engine_vision_reuse_allocator = engine_vision_reuse_allocator,
        .engine_vision_engine = engine_vision_engine,
        .tensor = tensor,
        .ops = ops,
        .weights = weights,
        .global_thread_pool = global_thread_pool,
        .runtime = runtime,
        .legacy_vision = legacy_vision,
    };
}

fn addImportsToRoot(root: *std.Build.Module, imports: LegacyImports) void {
    root.addImport("Pixio", imports.pixio);
    root.addImport("graph", imports.graph);
    root.addImport("engine_vision_graph", imports.engine_vision_graph);
    root.addImport("engine_vision_inspect", imports.engine_vision_inspect);
    root.addImport("engine_vision_base", imports.engine_vision_base);
    root.addImport("engine_vision_modules", imports.engine_vision_modules);
    root.addImport("engine_vision_reuse_allocator", imports.engine_vision_reuse_allocator);
    root.addImport("engine_vision_engine", imports.engine_vision_engine);
    root.addImport("tensor", imports.tensor);
    root.addImport("ops", imports.ops);
    root.addImport("weights", imports.weights);
    root.addImport("engine_global_thread_pool", imports.global_thread_pool);
    root.addImport("runtime", imports.runtime);
    root.addImport("vision", imports.legacy_vision);
}
