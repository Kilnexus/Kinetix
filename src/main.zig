const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");
const vision = @import("vision/preprocess.zig");

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
        if (std.fmt.parseInt(usize, value, 10)) |size| {
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
    var prepared = try vision.loadImageAsTensor(allocator, image_path, image_size);
    defer prepared.deinit();

    var detections = try runtime.runGraph(allocator, model_graph, weights_blob, &prepared.tensor, .{
        .score_threshold = 0.25,
        .iou_threshold = 0.7,
        .max_det = 300,
    });
    defer detections.deinit();
    vision.remapDetectionsToSource(detections.detections, prepared.info);

    try stdout.print("image_infer_path: {s}\n", .{image_path});
    try stdout.print("image_infer_size: {d}\n", .{image_size});
    try stdout.print("image_source_size: {d}x{d}\n", .{ prepared.info.src_width, prepared.info.src_height });
    try stdout.print("image_resized_size: {d}x{d}\n", .{ prepared.info.resized_width, prepared.info.resized_height });
    try stdout.print("image_padding: left={d} top={d}\n", .{ prepared.info.pad_left, prepared.info.pad_top });
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
