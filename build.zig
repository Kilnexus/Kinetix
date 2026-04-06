const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const kinetix_module = b.createModule(.{
        .root_source_file = b.path("kinetix.zig"),
        .target = target,
        .optimize = optimize,
    });

    const kinetix_lib = b.addLibrary(.{
        .name = "kinetix",
        .linkage = .static,
        .root_module = kinetix_module,
    });
    b.installArtifact(kinetix_lib);

    const cli_exe = b.addExecutable(.{
        .name = "kinetix",
        .root_module = b.createModule(.{
            .root_source_file = b.path("kinetix_cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(cli_exe);

    const run_cmd = b.addRunArtifact(cli_exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the Kinetix CLI");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("kinetix.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run Kinetix engine tests");
    test_step.dependOn(&run_tests.step);
}
