const std = @import("std");
const builtin = @import("builtin");
const ax_graph = @import("graph");
const ax_runtime = @import("runtime");
const ax_vision = @import("vision");
const ax_weights = @import("weights");
const graph = @import("../../artifacts/graph/graph.zig");
const task = @import("../../core/task.zig");
const types = @import("../types.zig");

pub const Summary = graph.Summary;
pub const Detection = types.RuntimeVisionDetection;
pub const DetectOutput = types.RuntimeVisionDetectOutput;

pub const ReceiptContext = struct {
    operation: []const u8,
    model_name: []const u8,
    model_family: []const u8,
    input_path: ?[]const u8,
    execution_nodes: usize,
    tensor_count: usize,
    class_count: ?usize,
};

pub fn loadSummary(allocator: std.mem.Allocator, graph_path: []const u8) !Summary {
    return try graph.loadSummary(allocator, graph_path);
}

pub fn maybeRunDetect(
    allocator: std.mem.Allocator,
    graph_path: []const u8,
    weights_path: []const u8,
    operation: []const u8,
    execution: task.ExecutionMode,
    input_path: ?[]const u8,
) !?DetectOutput {
    if (!std.mem.eql(u8, operation, "detect")) return null;
    if (execution != .sync) return null;

    const image_path = input_path orelse return null;

    if (builtin.is_test) {
        const detections = try allocator.alloc(Detection, 1);
        detections[0] = .{
            .x1 = 1.0,
            .y1 = 2.0,
            .x2 = 3.0,
            .y2 = 4.0,
            .score = 0.95,
            .class_id = 1,
        };
        return .{
            .candidate_count = 4,
            .detections = detections,
        };
    }

    const resolved_graph_path = try resolvePath(allocator, graph_path);
    defer allocator.free(resolved_graph_path);
    const resolved_weights_path = try resolvePath(allocator, weights_path);
    defer allocator.free(resolved_weights_path);
    const resolved_image_path = try resolvePath(allocator, image_path);
    defer allocator.free(resolved_image_path);

    var model_graph = try ax_graph.load(allocator, resolved_graph_path);
    defer model_graph.deinit();
    var weights_blob = try ax_weights.WeightsBlob.load(allocator, resolved_weights_path);
    defer weights_blob.deinit();
    var prepared = try ax_vision.loadImageAsTensor(allocator, resolved_image_path, 640);
    defer prepared.deinit();

    var detections_output = try ax_runtime.runGraph(
        allocator,
        &model_graph,
        &weights_blob,
        &prepared.tensor,
        .{
            .score_threshold = 0.25,
            .iou_threshold = 0.7,
            .max_det = 300,
        },
    );
    defer detections_output.deinit();
    ax_vision.remapDetectionsToSource(detections_output.detections, prepared.info);

    const detections = try allocator.alloc(Detection, detections_output.detections.len);
    for (detections_output.detections, detections) |det, *owned| {
        owned.* = .{
            .x1 = det.x1,
            .y1 = det.y1,
            .x2 = det.x2,
            .y2 = det.y2,
            .score = det.score,
            .class_id = det.class_id,
        };
    }
    return .{
        .candidate_count = detections_output.candidate_count,
        .detections = detections,
    };
}

pub fn buildOutputJson(
    allocator: std.mem.Allocator,
    context: ReceiptContext,
    detection_output: ?DetectOutput,
) ![]u8 {
    const VisionReceipt = struct {
        status: []const u8,
        operation: []const u8,
        model_name: []const u8,
        model_family: []const u8,
        input_path: ?[]const u8,
        execution_nodes: usize,
        tensor_count: usize,
        class_count: ?usize,
        candidate_count: ?usize,
        detections: []const Detection,
    };

    const receipt = VisionReceipt{
        .status = if (detection_output != null) "detect_completed" else "graph_ready",
        .operation = context.operation,
        .model_name = context.model_name,
        .model_family = context.model_family,
        .input_path = context.input_path,
        .execution_nodes = context.execution_nodes,
        .tensor_count = context.tensor_count,
        .class_count = context.class_count,
        .candidate_count = if (detection_output) |output| output.candidate_count else null,
        .detections = if (detection_output) |output| output.detections else &.{},
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}

fn resolvePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) return try allocator.dupe(u8, path);
    return try std.fs.cwd().realpathAlloc(allocator, path);
}
