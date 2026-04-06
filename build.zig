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
