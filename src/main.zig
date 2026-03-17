const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next();
    const graph_path = args.next() orelse "artifacts/graph.json";
    const weights_path = args.next() orelse "artifacts/weights.bin";
    const zero_infer_size = if (args.next()) |value| try std.fmt.parseInt(usize, value, 10) else null;

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

    if (zero_infer_size) |size| {
        var input = try runtime.Tensor.init(allocator, 1, 3, size, size);
        defer input.deinit();
        input.fill(0.0);

        var detections = try runtime.runGraph(allocator, &model_graph, &weights_blob, &input, .{
            .score_threshold = 0.0,
            .iou_threshold = 0.7,
            .max_det = 300,
        });
        defer detections.deinit();

        try stdout.print("zero_infer_size: {d}\n", .{size});
        try stdout.print("detect_candidates: {d}\n", .{detections.candidate_count});
        try stdout.print("detect_kept: {d}\n", .{detections.detections.len});
        if (detections.detections.len > 0) {
            const det = detections.detections[0];
            try stdout.print(
                "top_detection: cls={d} score={d:.6} box=[{d:.3}, {d:.3}, {d:.3}, {d:.3}]\n",
                .{ det.class_id, det.score, det.x1, det.y1, det.x2, det.y2 },
            );
        }
    }

    try runtime.printRoadmap(stdout);
}
