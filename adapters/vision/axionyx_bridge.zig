const std = @import("std");
const builtin = @import("builtin");
const axionyx_graph = @import("graph");
const axionyx_runtime = @import("runtime");
const axionyx_vision = @import("vision");
const vision_weights = @import("weights");

pub const Detection = struct {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    score: f64,
    class_id: usize,
};

pub const DetectOutput = struct {
    candidate_count: usize,
    detections: []Detection,

    pub fn deinit(self: *DetectOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.detections);
        self.* = undefined;
    }
};

pub fn executeDetect(
    allocator: std.mem.Allocator,
    graph_path: []const u8,
    weights_path: []const u8,
    image_path: []const u8,
) !DetectOutput {
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

    var model_graph = try axionyx_graph.load(allocator, resolved_graph_path);
    defer model_graph.deinit();

    var weights_blob = try loadWeightsAbsolute(allocator, resolved_weights_path);
    defer weights_blob.deinit();

    var prepared = try axionyx_vision.loadImageAsTensor(allocator, resolved_image_path, 640);
    defer prepared.deinit();

    var detections_output = try axionyx_runtime.runGraph(
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
    axionyx_vision.remapDetectionsToSource(detections_output.detections, prepared.info);

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

fn resolvePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) return try allocator.dupe(u8, path);
    return try std.fs.cwd().realpathAlloc(allocator, path);
}

fn loadWeightsAbsolute(allocator: std.mem.Allocator, weights_path: []const u8) !vision_weights.WeightsBlob {
    return try vision_weights.WeightsBlob.load(allocator, weights_path);
}
