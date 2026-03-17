const std = @import("std");
const graph = @import("graph.zig");
const runtime = @import("runtime.zig");
const weights = @import("weights.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next();
    const graph_path = args.next() orelse "artifacts/graph.json";
    const weights_path = args.next() orelse "artifacts/weights.bin";

    var model_graph = try graph.load(allocator, graph_path);
    defer model_graph.deinit();

    var weights_blob = try weights.WeightsBlob.load(allocator, weights_path);
    defer weights_blob.deinit();

    const first_tensor = &model_graph.tensors[0];
    const first_tensor_data = weights_blob.slice(first_tensor);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("graph: {s}\n", .{graph_path});
    try stdout.print("weights: {s}\n", .{weights_path});
    try stdout.print("format_version: {d}\n", .{model_graph.format_version});
    try stdout.print("model_name: {s}\n", .{model_graph.model_name});
    try stdout.print("execution_nodes: {d}\n", .{model_graph.execution_nodes.len});
    try stdout.print("tensor_count: {d}\n", .{model_graph.tensors.len});
    try stdout.print("class_count: {d}\n", .{model_graph.class_count});
    try stdout.print(
        "first_tensor: {s} len={d} first_value={d:.6}\n",
        .{ first_tensor.name, first_tensor_data.len, first_tensor_data[0] },
    );
    try runtime.printRoadmap(stdout);
}
