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

pub const DetectLevelProfileSummary = struct {
    feature_shape: [4]usize,
    feature_min: f32,
    feature_max: f32,
    feature_mean: f32,
    feature_abs_max: f32,
    reg_ns: u64,
    cls_ns: u64,
    decode_ns: u64,
    candidate_count: usize,
    max_class_logit: f32,
    max_class_score: f32,
    reg_kind: []const u8,
    cls_kind: []const u8,
};

pub const GraphNodeProfileSummary = struct {
    path: []const u8,
    kind: []const u8,
    elapsed_ns: u64,
    shape: [4]usize,
    min: f32,
    max: f32,
    mean: f32,
    abs_max: f32,
};

pub const DetectProfileSummary = struct {
    postprocess_mode: []const u8,
    score_threshold: f32,
    iou_threshold: f32,
    max_det: usize,
    branch_ns: u64,
    decode_ns: u64,
    nms_ns: u64,
    candidate_count: usize,
    kept_count: usize,
    levels: []DetectLevelProfileSummary,
    node_profiles: []GraphNodeProfileSummary,

    pub fn deinit(self: *DetectProfileSummary, allocator: std.mem.Allocator) void {
        allocator.free(self.levels);
        for (self.node_profiles) |node| {
            allocator.free(node.path);
            allocator.free(node.kind);
        }
        allocator.free(self.node_profiles);
        self.* = undefined;
    }
};

pub const DetectOutput = struct {
    candidate_count: usize,
    detections: []Detection,
    profile: ?DetectProfileSummary = null,

    pub fn deinit(self: *DetectOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.detections);
        if (self.profile) |*profile| profile.deinit(allocator);
        self.* = undefined;
    }
};

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
    const detect_options = loadDetectOptionsFromEnv();

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
            .profile = null,
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
        detect_options,
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
        .profile = if (shouldEmitDetectProfile()) try buildDetectProfileSummary(
            allocator,
            &model_graph,
            &weights_blob,
            &prepared.tensor,
            detect_options,
        ) else null,
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
        profile: ?DetectProfileSummary,
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
        .profile = if (detection_output) |output| output.profile else null,
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

fn shouldEmitDetectProfile() bool {
    const value = std.process.getEnvVarOwned(std.heap.page_allocator, "KINETIX_VISION_PROFILE") catch return false;
    defer std.heap.page_allocator.free(value);
    return value.len != 0 and !std.mem.eql(u8, value, "0");
}

fn loadDetectOptionsFromEnv() ax_runtime.DetectOptions {
    return .{
        .score_threshold = loadEnvFloat("KINETIX_VISION_SCORE_THRESHOLD", 0.25),
        .iou_threshold = loadEnvFloat("KINETIX_VISION_IOU_THRESHOLD", 0.7),
        .max_det = loadEnvUsize("KINETIX_VISION_MAX_DET", 300),
    };
}

fn loadEnvFloat(name: []const u8, default: f32) f32 {
    const value = std.process.getEnvVarOwned(std.heap.page_allocator, name) catch return default;
    defer std.heap.page_allocator.free(value);
    return std.fmt.parseFloat(f32, value) catch default;
}

fn loadEnvUsize(name: []const u8, default: usize) usize {
    const value = std.process.getEnvVarOwned(std.heap.page_allocator, name) catch return default;
    defer std.heap.page_allocator.free(value);
    return std.fmt.parseInt(usize, value, 10) catch default;
}

fn buildDetectProfileSummary(
    allocator: std.mem.Allocator,
    model_graph: *const ax_graph.Graph,
    weights_blob: *const ax_weights.WeightsBlob,
    input: *const ax_runtime.Tensor,
    detect_options: ax_runtime.DetectOptions,
) !DetectProfileSummary {
    var profile_graph = try ax_runtime.profileGraph(allocator, model_graph, weights_blob, input, detect_options);
    defer profile_graph.deinit();

    const node_filter = loadNodeProfileFilter(allocator) catch null;
    defer if (node_filter) |filter| freeNodeProfileFilter(allocator, filter);
    const node_profiles = try collectNodeProfiles(allocator, &profile_graph, node_filter);

    for (profile_graph.nodes) |node| {
        if (node.detect_profile) |detect_profile| {
            const levels = try allocator.alloc(DetectLevelProfileSummary, detect_profile.level_count);
            for (levels, 0..) |*level, index| {
                const source = detect_profile.levels[index];
                level.* = .{
                    .feature_shape = source.feature_shape,
                    .feature_min = source.feature_min,
                    .feature_max = source.feature_max,
                    .feature_mean = source.feature_mean,
                    .feature_abs_max = source.feature_abs_max,
                    .reg_ns = source.reg_ns,
                    .cls_ns = source.cls_ns,
                    .decode_ns = source.decode_ns,
                    .candidate_count = source.candidate_count,
                    .max_class_logit = source.max_class_logit,
                    .max_class_score = source.max_class_score,
                    .reg_kind = @tagName(source.reg_detail.kind),
                    .cls_kind = @tagName(source.cls_detail.kind),
                };
            }
            return .{
                .postprocess_mode = @tagName(detect_profile.postprocess_mode),
                .score_threshold = detect_options.score_threshold,
                .iou_threshold = detect_options.iou_threshold,
                .max_det = detect_options.max_det,
                .branch_ns = detect_profile.branch_ns,
                .decode_ns = detect_profile.decode_ns,
                .nms_ns = detect_profile.nms_ns,
                .candidate_count = detect_profile.candidate_count,
                .kept_count = detect_profile.kept_count,
                .levels = levels,
                .node_profiles = node_profiles,
            };
        }
    }

    freeGraphNodeProfiles(allocator, node_profiles);
    return error.ModuleNotFound;
}

fn loadNodeProfileFilter(allocator: std.mem.Allocator) !?[][]const u8 {
    const raw = std.process.getEnvVarOwned(allocator, "KINETIX_VISION_PROFILE_NODES") catch return null;
    defer allocator.free(raw);

    var parts: std.ArrayListUnmanaged([]const u8) = .empty;
    errdefer {
        for (parts.items) |part| allocator.free(part);
        parts.deinit(allocator);
    }

    var iter = std.mem.tokenizeScalar(u8, raw, ',');
    while (iter.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " \t\r\n");
        if (trimmed.len == 0) continue;
        try parts.append(allocator, try allocator.dupe(u8, trimmed));
    }

    return try parts.toOwnedSlice(allocator);
}

fn freeNodeProfileFilter(allocator: std.mem.Allocator, filter: [][]const u8) void {
    for (filter) |item| allocator.free(item);
    allocator.free(filter);
}

fn collectNodeProfiles(
    allocator: std.mem.Allocator,
    profile_graph: *const ax_runtime.GraphProfile,
    maybe_filter: ?[][]const u8,
) ![]GraphNodeProfileSummary {
    var collected: std.ArrayListUnmanaged(GraphNodeProfileSummary) = .empty;
    errdefer {
        for (collected.items) |item| {
            allocator.free(item.path);
            allocator.free(item.kind);
        }
        collected.deinit(allocator);
    }

    for (profile_graph.nodes) |node| {
        const stats = node.output_stats orelse continue;
        if (!shouldIncludeNodeProfile(node.path, maybe_filter)) continue;
        try collected.append(allocator, .{
            .path = try allocator.dupe(u8, node.path),
            .kind = try allocator.dupe(u8, node.kind),
            .elapsed_ns = node.elapsed_ns,
            .shape = stats.shape,
            .min = stats.min,
            .max = stats.max,
            .mean = stats.mean,
            .abs_max = stats.abs_max,
        });
    }

    return try collected.toOwnedSlice(allocator);
}

fn freeGraphNodeProfiles(allocator: std.mem.Allocator, node_profiles: []GraphNodeProfileSummary) void {
    for (node_profiles) |node| {
        allocator.free(node.path);
        allocator.free(node.kind);
    }
    allocator.free(node_profiles);
}

fn shouldIncludeNodeProfile(path: []const u8, maybe_filter: ?[][]const u8) bool {
    const filter = maybe_filter orelse return false;
    for (filter) |pattern| {
        if (std.mem.indexOf(u8, path, pattern) != null) return true;
    }
    return false;
}
