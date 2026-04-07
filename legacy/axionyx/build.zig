const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const graph_mod = b.createModule(.{
        .root_source_file = b.path("src/format/graph.zig"),
        .target = target,
        .optimize = optimize,
    });
    const engine_vision_graph_mod = b.createModule(.{
        .root_source_file = b.path("../../engine/artifacts/vision_graph.zig"),
        .target = target,
        .optimize = optimize,
    });
    graph_mod.addImport("engine_vision_graph", engine_vision_graph_mod);
    const tensor_mod = b.createModule(.{
        .root_source_file = b.path("src/nn/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ops_mod = b.createModule(.{
        .root_source_file = b.path("src/nn/ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops_mod.addImport("tensor", tensor_mod);

    const weights_mod = b.createModule(.{
        .root_source_file = b.path("../../engine/artifacts/vision/weights_blob.zig"),
        .target = target,
        .optimize = optimize,
    });
    weights_mod.addImport("graph", graph_mod);
    const global_thread_pool_mod = b.createModule(.{
        .root_source_file = b.path("../../engine/core/threading/global_thread_pool_module.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops_mod.addImport("engine_global_thread_pool", global_thread_pool_mod);
    const pixio_dep = b.dependency("Pixio", .{
        .target = target,
        .optimize = optimize,
    });
    const pixio_mod = pixio_dep.module("Pixio");

    const runtime_mod = b.createModule(.{
        .root_source_file = b.path("src/runtime/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    runtime_mod.addImport("graph", graph_mod);
    runtime_mod.addImport("tensor", tensor_mod);
    runtime_mod.addImport("ops", ops_mod);
    runtime_mod.addImport("weights", weights_mod);
    runtime_mod.addImport("engine_global_thread_pool", global_thread_pool_mod);
    const vision_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/preprocess.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_mod.addImport("Pixio", pixio_mod);
    vision_mod.addImport("runtime", runtime_mod);

    const exe = b.addExecutable(.{
        .name = "zig_yolo_inspect",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("graph", graph_mod);
    exe.root_module.addImport("Pixio", pixio_mod);
    exe.root_module.addImport("runtime", runtime_mod);
    exe.root_module.addImport("vision", vision_mod);
    exe.root_module.addImport("weights", weights_mod);
    exe.root_module.addImport("engine_global_thread_pool", global_thread_pool_mod);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the Zig YOLO full-runtime inspector");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.root_module.addImport("graph", graph_mod);
    unit_tests.root_module.addImport("tensor", tensor_mod);
    unit_tests.root_module.addImport("ops", ops_mod);
    unit_tests.root_module.addImport("weights", weights_mod);
    unit_tests.root_module.addImport("Pixio", pixio_mod);
    unit_tests.root_module.addImport("runtime", runtime_mod);
    unit_tests.root_module.addImport("vision", vision_mod);
    unit_tests.root_module.addImport("engine_global_thread_pool", global_thread_pool_mod);

    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_tests.step);
}
