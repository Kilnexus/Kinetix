const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");
const output_json = @import("output_json.zig");

pub fn runZeroMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    size: usize,
    json_out_path: ?[]const u8,
    trace_json_out_path: ?[]const u8,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    var input = try runtime.Tensor.init(allocator, 1, 3, size, size);
    defer input.deinit();
    input.fill(0.0);

    var detections = try runtime.runGraph(allocator, model_graph, weights_blob, &input, .{
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

    if (json_out_path) |path| {
        try output_json.writeDetectionsJson(path, &detections);
        try stdout.print("json_out: {s}\n", .{path});
    }

    if (trace_json_out_path) |path| {
        var trace = try runtime.traceGraph(allocator, model_graph, weights_blob, &input);
        defer trace.deinit();
        try output_json.writeTraceJson(path, &trace);
        try stdout.print("trace_json_out: {s}\n", .{path});
    }
}
