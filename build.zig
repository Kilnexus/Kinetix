const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const graph_mod = b.createModule(.{
        .root_source_file = b.path("src/format/graph.zig"),
        .target = target,
        .optimize = optimize,
    });
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
        .root_source_file = b.path("src/io/weights.zig"),
        .target = target,
        .optimize = optimize,
    });
    weights_mod.addImport("graph", graph_mod);
    const pixio_mod = b.createModule(.{
        .root_source_file = b.path("../pixio/src/pixio.zig"),
        .target = target,
        .optimize = optimize,
    });

    const runtime_mod = b.createModule(.{
        .root_source_file = b.path("src/runtime/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    runtime_mod.addImport("graph", graph_mod);
    runtime_mod.addImport("tensor", tensor_mod);
    runtime_mod.addImport("ops", ops_mod);
    runtime_mod.addImport("weights", weights_mod);
    const vision_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/preprocess.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_mod.addImport("pixio", pixio_mod);
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
    exe.root_module.addImport("pixio", pixio_mod);
    exe.root_module.addImport("imaging", pixio_mod);
    exe.root_module.addImport("runtime", runtime_mod);
    exe.root_module.addImport("vision", vision_mod);
    exe.root_module.addImport("weights", weights_mod);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the Zig YOLO full-runtime inspector");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/testing/tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.root_module.addImport("graph", graph_mod);
    unit_tests.root_module.addImport("tensor", tensor_mod);
    unit_tests.root_module.addImport("ops", ops_mod);
    unit_tests.root_module.addImport("weights", weights_mod);
    unit_tests.root_module.addImport("pixio", pixio_mod);
    unit_tests.root_module.addImport("imaging", pixio_mod);
    unit_tests.root_module.addImport("runtime", runtime_mod);
    unit_tests.root_module.addImport("vision", vision_mod);

    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_tests.step);
}
