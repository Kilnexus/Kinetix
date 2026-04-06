const std = @import("std");
const graph = @import("graph");
const weights = @import("weights");
const app_print = @import("print.zig");
const modes_image = @import("modes_image.zig");

pub fn runFastImageMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    image_size: usize,
    score_threshold: f32,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    var prepared_result = try modes_image.runTimedImageInference(allocator, model_graph, weights_blob, image_path, image_size, .{
        .score_threshold = score_threshold,
        .iou_threshold = 0.7,
        .max_det = 300,
    });
    defer prepared_result.detections.deinit();
    defer prepared_result.prepared.deinit();

    try stdout.print("fast_image_infer_path: {s}\n", .{image_path});
    try stdout.print("fast_image_infer_size: {d}\n", .{image_size});
    try stdout.print("fast_score_threshold: {d:.3}\n", .{score_threshold});
    try stdout.print(
        "fast_timing_ms: decode={d:.3} preprocess={d:.3} infer={d:.3} postprocess={d:.3} total={d:.3}\n",
        .{
            app_print.nsToMs(prepared_result.timings.decode_ns),
            app_print.nsToMs(prepared_result.timings.preprocess_ns),
            app_print.nsToMs(prepared_result.timings.infer_ns),
            app_print.nsToMs(prepared_result.timings.postprocess_ns),
            app_print.nsToMs(prepared_result.timings.totalNs()),
        },
    );
    try stdout.print("fast_detect_candidates: {d}\n", .{prepared_result.detections.candidate_count});
    try stdout.print("fast_detect_kept: {d}\n", .{prepared_result.detections.detections.len});
    if (prepared_result.detections.detections.len > 0) {
        const det = prepared_result.detections.detections[0];
        try stdout.print(
            "fast_top_detection: cls={d} score={d:.6} box=[{d:.3}, {d:.3}, {d:.3}, {d:.3}]\n",
            .{ det.class_id, det.score, det.x1, det.y1, det.x2, det.y2 },
        );
    }
}

pub fn runFastBenchmarkMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    warmup: usize,
    iterations: usize,
    image_size: usize,
    score_threshold: f32,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();

    for (0..warmup) |_| {
        var warm = try modes_image.runTimedImageInference(allocator, model_graph, weights_blob, image_path, image_size, .{
            .score_threshold = score_threshold,
            .iou_threshold = 0.7,
            .max_det = 300,
        });
        warm.detections.deinit();
        warm.prepared.deinit();
    }

    var decode_sum: u64 = 0;
    var preprocess_sum: u64 = 0;
    var infer_sum: u64 = 0;
    var postprocess_sum: u64 = 0;
    var kept_sum: usize = 0;
    var candidate_sum: usize = 0;

    for (0..iterations) |_| {
        var sample = try modes_image.runTimedImageInference(allocator, model_graph, weights_blob, image_path, image_size, .{
            .score_threshold = score_threshold,
            .iou_threshold = 0.7,
            .max_det = 300,
        });
        decode_sum += sample.timings.decode_ns;
        preprocess_sum += sample.timings.preprocess_ns;
        infer_sum += sample.timings.infer_ns;
        postprocess_sum += sample.timings.postprocess_ns;
        kept_sum += sample.detections.detections.len;
        candidate_sum += sample.detections.candidate_count;
        sample.detections.deinit();
        sample.prepared.deinit();
    }

    const denom = @as(f64, @floatFromInt(iterations));
    const decode_avg = @as(f64, @floatFromInt(decode_sum)) / denom;
    const preprocess_avg = @as(f64, @floatFromInt(preprocess_sum)) / denom;
    const infer_avg = @as(f64, @floatFromInt(infer_sum)) / denom;
    const postprocess_avg = @as(f64, @floatFromInt(postprocess_sum)) / denom;
    const total_avg = decode_avg + preprocess_avg + infer_avg + postprocess_avg;

    try stdout.print("fastbench_image: {s}\n", .{image_path});
    try stdout.print("fastbench_size: {d}\n", .{image_size});
    try stdout.print("fastbench_score_threshold: {d:.3}\n", .{score_threshold});
    try stdout.print("fastbench_warmup: {d}\n", .{warmup});
    try stdout.print("fastbench_iterations: {d}\n", .{iterations});
    try stdout.print(
        "fastbench_timing_ms: decode={d:.3} preprocess={d:.3} infer={d:.3} postprocess={d:.3} total={d:.3}\n",
        .{
            decode_avg / 1_000_000.0,
            preprocess_avg / 1_000_000.0,
            infer_avg / 1_000_000.0,
            postprocess_avg / 1_000_000.0,
            total_avg / 1_000_000.0,
        },
    );
    try stdout.print(
        "fastbench_detect_avg: candidates={d:.3} kept={d:.3}\n",
        .{
            @as(f64, @floatFromInt(candidate_sum)) / denom,
            @as(f64, @floatFromInt(kept_sum)) / denom,
        },
    );
}
