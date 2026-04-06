const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");
const output_json = @import("output_json.zig");
const app_print = @import("print.zig");
const vision = @import("../vision/preprocess.zig");
const vision_image = @import("../vision/image.zig");

pub const ImageTimings = struct {
    decode_ns: u64,
    preprocess_ns: u64,
    infer_ns: u64,
    postprocess_ns: u64,

    pub fn totalNs(self: ImageTimings) u64 {
        return self.decode_ns + self.preprocess_ns + self.infer_ns + self.postprocess_ns;
    }
};

pub const TimedImageInference = struct {
    prepared: vision.PreparedInput,
    detections: runtime.DetectOutput,
    timings: ImageTimings,
};

pub fn runTimedImageInference(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    image_size: usize,
    detect_options: runtime.DetectOptions,
) !TimedImageInference {
    var timer = try std.time.Timer.start();
    var src = try vision_image.loadRgb8(allocator, image_path);
    defer src.deinit();
    const decode_ns = timer.read();
    timer.reset();

    var prepared = try vision.prepareImageAsTensor(allocator, &src, image_size);
    const preprocess_ns = timer.read();
    timer.reset();

    const detections = try runtime.runGraph(allocator, model_graph, weights_blob, &prepared.tensor, .{
        .score_threshold = detect_options.score_threshold,
        .iou_threshold = detect_options.iou_threshold,
        .max_det = detect_options.max_det,
    });
    const infer_ns = timer.read();
    timer.reset();

    vision.remapDetectionsToSource(detections.detections, prepared.info);
    const postprocess_ns = timer.read();
    return .{
        .prepared = prepared,
        .detections = detections,
        .timings = .{
            .decode_ns = decode_ns,
            .preprocess_ns = preprocess_ns,
            .infer_ns = infer_ns,
            .postprocess_ns = postprocess_ns,
        },
    };
}

pub fn runImageMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    image_size: usize,
    json_out_path: ?[]const u8,
    trace_json_out_path: ?[]const u8,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    var tracker = runtime.TrackingAllocator.init(allocator);
    defer tracker.deinit();

    var prepared_result = try runTimedImageInference(tracker.allocator(), model_graph, weights_blob, image_path, image_size, .{
        .score_threshold = 0.25,
        .iou_threshold = 0.7,
        .max_det = 300,
    });
    var prepared = prepared_result.prepared;
    var detections = prepared_result.detections;

    try stdout.print("image_infer_path: {s}\n", .{image_path});
    try stdout.print("image_infer_size: {d}\n", .{image_size});
    try stdout.print("image_source_size: {d}x{d}\n", .{ prepared.info.src_width, prepared.info.src_height });
    try stdout.print("image_resized_size: {d}x{d}\n", .{ prepared.info.resized_width, prepared.info.resized_height });
    try stdout.print("image_padding: left={d} top={d}\n", .{ prepared.info.pad_left, prepared.info.pad_top });
    try stdout.print(
        "timing_ms: decode={d:.3} preprocess={d:.3} infer={d:.3} postprocess={d:.3} total={d:.3}\n",
        .{
            app_print.nsToMs(prepared_result.timings.decode_ns),
            app_print.nsToMs(prepared_result.timings.preprocess_ns),
            app_print.nsToMs(prepared_result.timings.infer_ns),
            app_print.nsToMs(prepared_result.timings.postprocess_ns),
            app_print.nsToMs(prepared_result.timings.totalNs()),
        },
    );
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
        var trace = try runtime.traceGraph(allocator, model_graph, weights_blob, &prepared.tensor);
        defer trace.deinit();
        try output_json.writeTraceJson(path, &trace);
        try stdout.print("trace_json_out: {s}\n", .{path});
    }

    detections.deinit();
    prepared.deinit();
    try app_print.printMemoryStats(stdout, tracker.snapshot());
}
