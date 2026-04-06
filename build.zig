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
    tensor: *std.Build.Module,
    ops: *std.Build.Module,
    weights: *std.Build.Module,
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
    const tensor = b.createModule(.{
        .root_source_file = b.path("legacy/axionyx/src/nn/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ops = b.createModule(.{
        .root_source_file = b.path("legacy/axionyx/src/nn/ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops.addImport("tensor", tensor);

    const weights = b.createModule(.{
        .root_source_file = b.path("engine/artifacts/vision/weights_blob.zig"),
        .target = target,
        .optimize = optimize,
    });
    weights.addImport("graph", graph);

    const runtime = b.createModule(.{
        .root_source_file = b.path("legacy/axionyx/src/runtime/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    runtime.addImport("graph", graph);
    runtime.addImport("tensor", tensor);
    runtime.addImport("ops", ops);
    runtime.addImport("weights", weights);

    const legacy_vision = b.createModule(.{
        .root_source_file = b.path("legacy/axionyx/src/vision/preprocess.zig"),
        .target = target,
        .optimize = optimize,
    });
    legacy_vision.addImport("Pixio", pixio);
    legacy_vision.addImport("runtime", runtime);

    return .{
        .pixio = pixio,
        .graph = graph,
        .tensor = tensor,
        .ops = ops,
        .weights = weights,
        .runtime = runtime,
        .legacy_vision = legacy_vision,
    };
}

fn addImportsToRoot(root: *std.Build.Module, imports: LegacyImports) void {
    root.addImport("Pixio", imports.pixio);
    root.addImport("graph", imports.graph);
    root.addImport("tensor", imports.tensor);
    root.addImport("ops", imports.ops);
    root.addImport("weights", imports.weights);
    root.addImport("runtime", imports.runtime);
    root.addImport("vision", imports.legacy_vision);
}
