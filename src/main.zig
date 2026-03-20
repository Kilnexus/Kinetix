const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");
const vision = @import("vision/preprocess.zig");
const vision_image = @import("vision/image.zig");

const ImageTimings = struct {
    decode_ns: u64,
    preprocess_ns: u64,
    infer_ns: u64,
    postprocess_ns: u64,

    fn totalNs(self: ImageTimings) u64 {
        return self.decode_ns + self.preprocess_ns + self.infer_ns + self.postprocess_ns;
    }
};

const MemoryStats = runtime.AllocationStats;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next();
    const graph_path = args.next() orelse "artifacts/graph.json";
    const weights_path = args.next() orelse "artifacts/weights.bin";
    const mode_arg = args.next();

    var model_graph = try graph.load(allocator, graph_path);
    defer model_graph.deinit();

    var weights_blob = try weights.WeightsBlob.load(allocator, weights_path);
    defer weights_blob.deinit();
    var support = try runtime.inspectModel(allocator, &model_graph);
    defer support.deinit();

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
    try stdout.print("detect_nodes: {d}\n", .{support.detect_nodes});
    try stdout.print("runtime_compatible: {s}\n", .{if (support.supportsEndToEnd()) "true" else "false"});
    try stdout.print(
        "supported_execution_nodes: {d}/{d}\n",
        .{ support.supported_execution_nodes, support.execution_nodes },
    );
    try stdout.print(
        "supported_module_nodes: {d}/{d}\n",
        .{ support.supported_module_nodes, support.module_nodes },
    );
    try stdout.print(
        "first_tensor: {s} len={d} first_value={d:.6}\n",
        .{ first_tensor.name, first_tensor_data.len, first_tensor_data[0] },
    );
    if (support.unsupported_execution_kinds.len > 0) {
        try stdout.writeAll("unsupported_execution_kinds:");
        for (support.unsupported_execution_kinds) |entry| {
            try stdout.print(" {s}({d})", .{ entry.kind, entry.count });
        }
        try stdout.writeByte('\n');
    }
    if (support.unsupported_module_kinds.len > 0) {
        try stdout.writeAll("unsupported_module_kinds:");
        for (support.unsupported_module_kinds) |entry| {
            try stdout.print(" {s}({d})", .{ entry.kind, entry.count });
        }
        try stdout.writeByte('\n');
    }

    if (mode_arg) |value| {
        if (std.mem.eql(u8, value, "bench")) {
            const image_path = args.next() orelse "data/archive/images/000_0001.png";
            const iterations = if (args.next()) |arg4|
                std.fmt.parseInt(usize, arg4, 10) catch 5
            else
                5;
            const warmup = if (args.next()) |arg5|
                std.fmt.parseInt(usize, arg5, 10) catch 1
            else
                1;
            try runBenchmarkMode(allocator, &model_graph, &weights_blob, image_path, warmup, iterations);
        } else if (std.mem.eql(u8, value, "profile")) {
            const image_path = args.next() orelse "data/archive/images/000_0001.png";
            const image_size = if (args.next()) |arg4|
                std.fmt.parseInt(usize, arg4, 10) catch 160
            else
                160;
            try runProfileMode(allocator, &model_graph, &weights_blob, image_path, image_size);
        } else if (std.fmt.parseInt(usize, value, 10)) |size| {
            const json_out_path = args.next();
            const trace_json_out_path = args.next();
            try runZeroMode(allocator, &model_graph, &weights_blob, size, json_out_path, trace_json_out_path);
        } else |_| {
            const image_path = value;
            const maybe_size_or_json = args.next();
            var image_size: usize = 640;
            var json_out_path: ?[]const u8 = null;
            var trace_json_out_path: ?[]const u8 = null;

            if (maybe_size_or_json) |arg4| {
                if (std.fmt.parseInt(usize, arg4, 10)) |parsed| {
                    image_size = parsed;
                    json_out_path = args.next();
                    trace_json_out_path = args.next();
                } else |_| {
                    json_out_path = arg4;
                    trace_json_out_path = args.next();
                }
            }

            try runImageMode(
                allocator,
                &model_graph,
                &weights_blob,
                image_path,
                image_size,
                json_out_path,
                trace_json_out_path,
            );
        }
    }

    try runtime.printRoadmap(stdout);
}

fn runZeroMode(
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
        try writeDetectionsJson(path, &detections);
        try stdout.print("json_out: {s}\n", .{path});
    }

    if (trace_json_out_path) |path| {
        var trace = try runtime.traceGraph(allocator, model_graph, weights_blob, &input);
        defer trace.deinit();
        try writeTraceJson(path, &trace);
        try stdout.print("trace_json_out: {s}\n", .{path});
    }
}

fn runImageMode(
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
            nsToMs(prepared_result.timings.decode_ns),
            nsToMs(prepared_result.timings.preprocess_ns),
            nsToMs(prepared_result.timings.infer_ns),
            nsToMs(prepared_result.timings.postprocess_ns),
            nsToMs(prepared_result.timings.totalNs()),
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
        try writeDetectionsJson(path, &detections);
        try stdout.print("json_out: {s}\n", .{path});
    }

    if (trace_json_out_path) |path| {
        var trace = try runtime.traceGraph(allocator, model_graph, weights_blob, &prepared.tensor);
        defer trace.deinit();
        try writeTraceJson(path, &trace);
        try stdout.print("trace_json_out: {s}\n", .{path});
    }

    detections.deinit();
    prepared.deinit();
    try printMemoryStats(stdout, tracker.snapshot());
}

const TimedImageInference = struct {
    prepared: vision.PreparedInput,
    detections: runtime.DetectOutput,
    timings: ImageTimings,
};

fn runTimedImageInference(
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

fn runBenchmarkMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    warmup: usize,
    iterations: usize,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    const sizes = [_]usize{ 160, 320, 640 };

    try stdout.print("benchmark_image: {s}\n", .{image_path});
    try stdout.print("benchmark_warmup: {d}\n", .{warmup});
    try stdout.print("benchmark_iterations: {d}\n", .{iterations});
    try stdout.writeAll("benchmark_table_ms:\n");
    try stdout.writeAll("size decode preprocess infer postprocess total alloc_mb peak_mb allocs\n");

    for (sizes) |image_size| {
        for (0..warmup) |_| {
            var warm_tracker = runtime.TrackingAllocator.init(allocator);
            defer warm_tracker.deinit();
            var warm = try runTimedImageInference(warm_tracker.allocator(), model_graph, weights_blob, image_path, image_size, .{
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
            var sample = try runTimedImageInference(sample_tracker.allocator(), model_graph, weights_blob, image_path, image_size, .{
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

fn runProfileMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    image_size: usize,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();

    var src = try vision_image.loadRgb8(allocator, image_path);
    defer src.deinit();

    var prepared = try vision.prepareImageAsTensor(allocator, &src, image_size);
    defer prepared.deinit();

    var profile = try runtime.profileGraph(allocator, model_graph, weights_blob, &prepared.tensor, .{
        .score_threshold = 0.25,
        .iou_threshold = 0.7,
        .max_det = 300,
    });
    defer profile.deinit();

    var sorted = try allocator.dupe(runtime.NodeProfile, profile.nodes);
    defer allocator.free(sorted);
    std.mem.sort(runtime.NodeProfile, sorted, {}, struct {
        fn lessThan(_: void, lhs: runtime.NodeProfile, rhs: runtime.NodeProfile) bool {
            return lhs.elapsed_ns > rhs.elapsed_ns;
        }
    }.lessThan);

    var total_ns: u64 = 0;
    for (profile.nodes) |node| total_ns += node.elapsed_ns;

    try stdout.print("profile_image: {s}\n", .{image_path});
    try stdout.print("profile_size: {d}\n", .{image_size});
    try stdout.print("profile_total_ms: {d:.3}\n", .{nsToMs(total_ns)});
    try stdout.writeAll("profile_top_nodes_ms:\n");
    try stdout.writeAll("rank ms kind path\n");

    const top_n = @min(sorted.len, 8);
    for (sorted[0..top_n], 0..) |node, index| {
        try stdout.print(
            "{d} {d:.3} {s} {s}\n",
            .{ index + 1, nsToMs(node.elapsed_ns), node.kind, node.path },
        );
        if (node.detect_profile) |detect_profile| {
            try stdout.print(
                "detect_profile_ms: branch={d:.3} decode={d:.3} nms={d:.3} candidates={d} kept={d}\n",
                .{
                    nsToMs(detect_profile.branch_ns),
                    nsToMs(detect_profile.decode_ns),
                    nsToMs(detect_profile.nms_ns),
                    detect_profile.candidate_count,
                    detect_profile.kept_count,
                },
            );
        }
        if (node.c3k2_profile) |c3k2_profile| {
            try stdout.print(
                "c3k2_profile_ms: cv1={d:.3} child={d:.3} concat={d:.3} cv2={d:.3} child_kind={s}\n",
                .{
                    nsToMs(c3k2_profile.cv1_ns),
                    nsToMs(c3k2_profile.child_ns),
                    nsToMs(c3k2_profile.concat_ns),
                    nsToMs(c3k2_profile.cv2_ns),
                    c3k2_profile.child_kind,
                },
            );
            if (c3k2_profile.child_c3k) |c3k_profile| {
                try stdout.print(
                    "c3k_profile_ms: cv1={d:.3} seq={d:.3} cv2={d:.3} concat={d:.3} cv3={d:.3} seq_kind={s}\n",
                    .{
                        nsToMs(c3k_profile.cv1_ns),
                        nsToMs(c3k_profile.seq_ns),
                        nsToMs(c3k_profile.cv2_ns),
                        nsToMs(c3k_profile.concat_ns),
                        nsToMs(c3k_profile.cv3_ns),
                        c3k_profile.seq_kind,
                    },
                );
            }
            if (c3k2_profile.child_bottleneck) |bottleneck_profile| {
                try stdout.print(
                    "bottleneck_profile_ms: cv1={d:.3} cv2={d:.3} add={d:.3} has_add={}\n",
                    .{
                        nsToMs(bottleneck_profile.cv1_ns),
                        nsToMs(bottleneck_profile.cv2_ns),
                        nsToMs(bottleneck_profile.add_ns),
                        bottleneck_profile.has_add,
                    },
                );
            }
        }
    }
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn printMemoryStats(writer: anytype, stats: MemoryStats) !void {
    try writer.print(
        "memory: allocs={d} frees={d} resize={d} remap={d} failed={d} total_alloc_mb={d:.3} total_free_mb={d:.3} peak_live_mb={d:.3} live_end_mb={d:.3} outstanding={d}\n",
        .{
            stats.alloc_count,
            stats.free_count,
            stats.resize_count,
            stats.remap_count,
            stats.failed_alloc_count,
            bytesToMiB(stats.total_allocated_bytes),
            bytesToMiB(stats.total_freed_bytes),
            bytesToMiB(stats.peak_live_bytes),
            bytesToMiB(stats.live_bytes),
            stats.outstanding_allocations,
        },
    );
}

fn bytesToMiB(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

fn writeDetectionsJson(path: []const u8, detections: *const runtime.DetectOutput) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
    const out = &writer.interface;

    try out.writeAll("{\n");
    try out.print("  \"candidate_count\": {d},\n", .{detections.candidate_count});
    try out.writeAll("  \"detections\": [\n");
    for (detections.detections, 0..) |det, index| {
        try out.print(
            "    {{\"x1\": {d:.9}, \"y1\": {d:.9}, \"x2\": {d:.9}, \"y2\": {d:.9}, \"score\": {d:.9}, \"class_id\": {d}}}",
            .{ det.x1, det.y1, det.x2, det.y2, det.score, det.class_id },
        );
        if (index + 1 != detections.detections.len) {
            try out.writeAll(",\n");
        } else {
            try out.writeAll("\n");
        }
    }
    try out.writeAll("  ]\n}\n");
    try out.flush();
}

fn writeTraceJson(path: []const u8, trace: *const runtime.GraphTrace) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
    const out = &writer.interface;

    try out.writeAll("{\n  \"nodes\": [\n");
    for (trace.nodes, 0..) |node, index| {
        try out.print(
            "    {{\"index\": {d}, \"path\": \"{s}\", \"kind\": \"{s}\", \"shape\": [{d}, {d}, {d}, {d}], \"min\": {d:.9}, \"max\": {d:.9}, \"mean\": {d:.9}, \"l2\": {d:.9}, \"first\": {d:.9}}}",
            .{
                node.index,
                node.path,
                node.kind,
                node.shape[0],
                node.shape[1],
                node.shape[2],
                node.shape[3],
                node.min,
                node.max,
                node.mean,
                node.l2,
                node.first,
            },
        );
        if (index + 1 != trace.nodes.len) {
            try out.writeAll(",\n");
        } else {
            try out.writeAll("\n");
        }
    }
    try out.writeAll("  ]\n}\n");
    try out.flush();
}
