const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");
const cli = @import("app/cli.zig");
const modes_bench = @import("app/modes_bench.zig");
const modes_fast = @import("app/modes_fast.zig");
const modes_image = @import("app/modes_image.zig");
const modes_profile = @import("app/modes_profile.zig");
const modes_zero = @import("app/modes_zero.zig");
const app_print = @import("app/print.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);
    const parsed = cli.parseArgs(argv);

    var model_graph = try graph.load(allocator, parsed.graph_path);
    defer model_graph.deinit();

    var weights_blob = try weights.WeightsBlob.load(allocator, parsed.weights_path);
    defer weights_blob.deinit();

    var support = try runtime.inspectModel(allocator, &model_graph);
    defer support.deinit();

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const first_tensor_data = weights_blob.slice(&model_graph.tensors[0]);
    try app_print.printModelSummary(
        stdout,
        parsed.graph_path,
        parsed.weights_path,
        &model_graph,
        &support,
        first_tensor_data,
    );

    switch (parsed.command) {
        .roadmap => {},
        .bench => |args| try modes_bench.runBenchmarkMode(
            allocator,
            &model_graph,
            &weights_blob,
            args.image_path,
            args.warmup,
            args.iterations,
        ),
        .fastbench => |args| try modes_fast.runFastBenchmarkMode(
            allocator,
            &model_graph,
            &weights_blob,
            args.image_path,
            args.warmup,
            args.iterations,
            args.image_size,
            args.score_threshold,
        ),
        .fast => |args| try modes_fast.runFastImageMode(
            allocator,
            &model_graph,
            &weights_blob,
            args.image_path,
            args.image_size,
            args.score_threshold,
        ),
        .profile => |args| try modes_profile.runProfileMode(
            allocator,
            &model_graph,
            &weights_blob,
            args.image_path,
            args.image_size,
        ),
        .zero => |args| try modes_zero.runZeroMode(
            allocator,
            &model_graph,
            &weights_blob,
            args.size,
            args.json_out_path,
            args.trace_json_out_path,
        ),
        .image => |args| try modes_image.runImageMode(
            allocator,
            &model_graph,
            &weights_blob,
            args.image_path,
            args.image_size,
            args.json_out_path,
            args.trace_json_out_path,
        ),
    }

    try runtime.printRoadmap(stdout);
}
