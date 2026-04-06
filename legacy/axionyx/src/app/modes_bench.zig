const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");
const modes_image = @import("modes_image.zig");

pub fn runBenchmarkMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    warmup: usize,
    iterations: usize,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    const sizes = [_]usize{ 96, 128, 160, 320, 640 };

    try stdout.print("benchmark_image: {s}\n", .{image_path});
    try stdout.print("benchmark_warmup: {d}\n", .{warmup});
    try stdout.print("benchmark_iterations: {d}\n", .{iterations});
    try stdout.writeAll("benchmark_table_ms:\n");
    try stdout.writeAll("size decode preprocess infer postprocess total alloc_mb peak_mb allocs\n");

    for (sizes) |image_size| {
        for (0..warmup) |_| {
            var warm_tracker = runtime.TrackingAllocator.init(allocator);
            defer warm_tracker.deinit();
            var warm = try modes_image.runTimedImageInference(warm_tracker.allocator(), model_graph, weights_blob, image_path, image_size, .{
                .score_threshold = 0.25,
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
        var total_alloc_sum: usize = 0;
        var peak_live_max: usize = 0;
        var alloc_count_sum: usize = 0;
        for (0..iterations) |_| {
            var sample_tracker = runtime.TrackingAllocator.init(allocator);
            defer sample_tracker.deinit();
            var sample = try modes_image.runTimedImageInference(sample_tracker.allocator(), model_graph, weights_blob, image_path, image_size, .{
                .score_threshold = 0.25,
                .iou_threshold = 0.7,
                .max_det = 300,
            });

            decode_sum += sample.timings.decode_ns;
            preprocess_sum += sample.timings.preprocess_ns;
            infer_sum += sample.timings.infer_ns;
            postprocess_sum += sample.timings.postprocess_ns;
            sample.detections.deinit();
            sample.prepared.deinit();

            const mem = sample_tracker.snapshot();
            total_alloc_sum += mem.total_allocated_bytes;
            peak_live_max = @max(peak_live_max, mem.peak_live_bytes);
            alloc_count_sum += mem.alloc_count;
        }

        const denom = @as(f64, @floatFromInt(iterations));
        const decode_avg = @as(f64, @floatFromInt(decode_sum)) / denom;
        const preprocess_avg = @as(f64, @floatFromInt(preprocess_sum)) / denom;
        const infer_avg = @as(f64, @floatFromInt(infer_sum)) / denom;
        const postprocess_avg = @as(f64, @floatFromInt(postprocess_sum)) / denom;
        const total_avg = decode_avg + preprocess_avg + infer_avg + postprocess_avg;

        try stdout.print(
            "{d} {d:.3} {d:.3} {d:.3} {d:.3} {d:.3} {d:.3} {d:.3} {d}\n",
            .{
                image_size,
                decode_avg / 1_000_000.0,
                preprocess_avg / 1_000_000.0,
                infer_avg / 1_000_000.0,
                postprocess_avg / 1_000_000.0,
                total_avg / 1_000_000.0,
                @as(f64, @floatFromInt(total_alloc_sum)) / denom / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(peak_live_max)) / (1024.0 * 1024.0),
                @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(alloc_count_sum)) / denom))),
            },
        );
    }
}
