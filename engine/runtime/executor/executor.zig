const std = @import("std");
const builtin = @import("builtin");
const ax_graph = @import("graph");
const ax_runtime = @import("runtime");
const ax_vision = @import("vision");
const ax_weights = @import("weights");
const adapter_mod = @import("../../adapter/adapter.zig");
const graph = @import("../../artifacts/graph/graph.zig");
const task = @import("../../core/task.zig");
const handle_mod = @import("../model/handle.zig");
const ocr_pipeline = @import("../ocr_pipeline.zig");
const text_runtime = @import("../text/text.zig");
const types = @import("../types.zig");

const native_bridge = if (builtin.is_test) struct {
    pub fn executeQwenSingle(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: types.WeightScheme,
        request: task.TaskRequest,
    ) ![]u8 {
        _ = model_dir;
        _ = preferred_weights;
        _ = request;
        return try allocator.dupe(u8, "stub-native-single");
    }

    pub const NativeBatchOutput = struct {
        texts: [][]u8,
        total_decoded_tokens: usize,
        finished_requests: usize,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            for (self.texts) |text| allocator.free(text);
            allocator.free(self.texts);
            self.* = undefined;
        }
    };

    pub fn executeQwenBatch(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: types.WeightScheme,
        requests: []const task.TaskRequest,
    ) !NativeBatchOutput {
        _ = model_dir;
        _ = preferred_weights;

        const texts = try allocator.alloc([]u8, requests.len);
        errdefer allocator.free(texts);
        var initialized: usize = 0;
        errdefer {
            for (texts[0..initialized]) |text| allocator.free(text);
        }
        for (texts) |*text| {
            text.* = try allocator.dupe(u8, "stub-native-batch");
            initialized += 1;
        }
        return .{
            .texts = texts,
            .total_decoded_tokens = requests.len,
            .finished_requests = requests.len,
        };
    }
} else text_runtime.native_dispatch.NativeBatchBridge;

pub const Executor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Executor {
        return .{ .allocator = allocator };
    }

    pub fn execute(self: Executor, handle: *const handle_mod.ModelHandle, plan: *const types.ExecutionPlan) !types.RuntimeResult {
        if (plan.request_count != 1 or plan.requests.len != 1) return error.InvalidExecutionPlan;

        return switch (handle.normalized.provider_key) {
            .qwen3_text => try executeQwen3(self.allocator, handle, plan.requests[0]),
            .yolo_vision => try executeYoloVision(self.allocator, handle, plan.requests[0]),
            .swiftocr_ocr => try executeSwiftOCR(self.allocator, handle, plan.requests[0]),
            else => error.RuntimeExecutionNotImplemented,
        };
    }

    pub fn executeBatch(self: Executor, handle: *const handle_mod.ModelHandle, plan: *const types.ExecutionPlan) !types.RuntimeBatchResults {
        if (plan.request_count == 0 or plan.requests.len == 0) return error.InvalidExecutionPlan;

        return switch (handle.normalized.provider_key) {
            .qwen3_text => try executeQwen3Batch(self.allocator, handle, plan.requests),
            .yolo_vision, .swiftocr_ocr => try executeBatchSequential(self.allocator, handle, plan.requests),
            else => error.RuntimeExecutionNotImplemented,
        };
    }
};

fn executeQwen3(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    var legacy_result = try text_runtime.native_dispatch.executeSingle(
        allocator,
        native_bridge,
        handle.normalized.artifacts.model_dir,
        .auto,
        .{
            .adapter_id = handle.normalized.descriptor.id,
            .accepted = true,
            .execution = request.execution,
        },
        buildTaskRequest(handle, request),
    );
    errdefer legacy_result.deinit(allocator);

    const output = legacy_result.output;
    legacy_result.output = .none;
    return .{
        .origin = legacy_result.origin,
        .note = @tagName(legacy_result.note),
        .output = output,
    };
}

fn executeQwen3Batch(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    requests: []const types.RuntimeRequest,
) !types.RuntimeBatchResults {
    const task_requests = try allocator.alloc(task.TaskRequest, requests.len);
    defer allocator.free(task_requests);

    for (requests, task_requests) |request, *slot| {
        slot.* = buildTaskRequest(handle, request);
    }

    const legacy_results = try text_runtime.native_dispatch.executeBatch(
        allocator,
        native_bridge,
        handle.normalized.artifacts.model_dir,
        .auto,
        handle.normalized.descriptor.id,
        task_requests,
    );
    defer allocator.free(legacy_results);

    const results = try allocator.alloc(types.RuntimeResult, legacy_results.len);
    errdefer allocator.free(results);

    for (legacy_results, results) |*legacy_result, *result| {
        const output = legacy_result.output;
        legacy_result.output = .none;
        result.* = .{
            .origin = legacy_result.origin,
            .note = @tagName(legacy_result.note),
            .output = output,
        };
    }

    return .{
        .allocator = allocator,
        .items = results,
    };
}

fn executeYoloVision(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    switch (request.input) {
        .none, .image_path => {},
        else => return error.InvalidInputPayload,
    }

    const graph_path = handle.normalized.artifacts.graph_path orelse return error.MissingGraphArtifact;
    const weights_path = handle.normalized.artifacts.binary_weights_path orelse return error.MissingBinaryWeightsArtifact;
    const summary = try graph.loadSummary(allocator, graph_path);
    defer allocator.free(summary.model_name);

    var detection_output = try maybeRunVisionDetect(allocator, graph_path, weights_path, request);
    defer if (detection_output) |*output| output.deinit(allocator);

    const output = try buildVisionOutputJson(allocator, request, summary, handle.normalized.descriptor.family, detection_output);
    return .{
        .origin = .shared_adapter,
        .note = if (detection_output != null) "vision_shared_detect" else "vision_graph_ready",
        .output = .{ .json = output },
    };
}

fn executeSwiftOCR(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    const model_path = handle.normalized.artifacts.ocr_model_path orelse return error.MissingOCRModelArtifact;
    const infer_output = try maybeRunOCRInfer(allocator, model_path, request);
    const output = try buildOCROutputJson(allocator, request, handle.normalized.descriptor.family, model_path, infer_output);
    return .{
        .origin = .shared_adapter,
        .note = if (infer_output != null) "ocr_shared_infer" else "ocr_model_ready",
        .output = .{ .json = output },
    };
}

fn executeBatchSequential(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    requests: []const types.RuntimeRequest,
) !types.RuntimeBatchResults {
    const results = try allocator.alloc(types.RuntimeResult, requests.len);
    errdefer allocator.free(results);

    var initialized: usize = 0;
    errdefer {
        for (results[0..initialized]) |*result| result.deinit(allocator);
        allocator.free(results);
    }

    for (requests, results) |request, *result| {
        result.* = switch (handle.normalized.provider_key) {
            .yolo_vision => try executeYoloVision(allocator, handle, request),
            .swiftocr_ocr => try executeSwiftOCR(allocator, handle, request),
            else => return error.RuntimeExecutionNotImplemented,
        };
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .items = results,
    };
}

fn maybeRunVisionDetect(
    allocator: std.mem.Allocator,
    graph_path: []const u8,
    weights_path: []const u8,
    request: types.RuntimeRequest,
) !?types.RuntimeVisionDetectOutput {
    if (!std.mem.eql(u8, request.operation, "detect")) return null;
    if (request.execution != .sync) return null;

    const image_path = switch (request.input) {
        .image_path => |value| value,
        else => return null,
    };

    if (builtin.is_test) {
        const detections = try allocator.alloc(types.RuntimeVisionDetection, 1);
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

    const detections = try allocator.alloc(types.RuntimeVisionDetection, detections_output.detections.len);
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

fn buildVisionOutputJson(
    allocator: std.mem.Allocator,
    request: types.RuntimeRequest,
    summary: graph.Summary,
    model_family: []const u8,
    detection_output: ?types.RuntimeVisionDetectOutput,
) ![]u8 {
    const Detection = types.RuntimeVisionDetection;
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
        .operation = request.operation,
        .model_name = summary.model_name,
        .model_family = model_family,
        .input_path = request.input.asString(),
        .execution_nodes = summary.execution_nodes,
        .tensor_count = summary.tensor_count,
        .class_count = summary.class_count,
        .candidate_count = if (detection_output) |output| output.candidate_count else null,
        .detections = if (detection_output) |output| output.detections else &.{},
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}

fn maybeRunOCRInfer(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    request: types.RuntimeRequest,
) !?ocr_pipeline.InferResult {
    if (!std.mem.eql(u8, request.operation, "infer-ocr")) return null;
    if (request.execution != .sync) return null;

    const image_path = switch (request.input) {
        .image_path => |value| value,
        else => return null,
    };

    var pipeline = ocr_pipeline.OCRPipeline.init(allocator);
    defer pipeline.deinit();
    return try pipeline.infer(.{
        .model_path = model_path,
        .image_path = image_path,
    });
}

fn buildOCROutputJson(
    allocator: std.mem.Allocator,
    request: types.RuntimeRequest,
    model_family: []const u8,
    model_path: []const u8,
    infer_output: ?ocr_pipeline.InferResult,
) ![]u8 {
    const OCRReceipt = struct {
        status: []const u8,
        operation: []const u8,
        model_family: []const u8,
        model_path: []const u8,
        input_path: ?[]const u8,
        loaded_tensors: ?usize,
        image_width: ?usize,
        image_height: ?usize,
    };

    const receipt = OCRReceipt{
        .status = if (infer_output != null) "ocr_infer_completed" else "ocr_model_ready",
        .operation = request.operation,
        .model_family = model_family,
        .model_path = model_path,
        .input_path = request.input.asString(),
        .loaded_tensors = if (infer_output) |output| output.loaded_tensors else null,
        .image_width = if (infer_output) |output| output.image_width else null,
        .image_height = if (infer_output) |output| output.image_height else null,
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}

fn buildTaskRequest(handle: *const handle_mod.ModelHandle, request: types.RuntimeRequest) task.TaskRequest {
    return .{
        .spec = .{
            .modality = handle.normalized.descriptor.modality,
            .operation = request.operation,
            .model_family = handle.normalized.descriptor.family,
            .adapter_id = handle.normalized.descriptor.id,
            .execution = request.execution,
        },
        .input = request.input,
        .generation = request.generation,
    };
}

fn resolvePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) return try allocator.dupe(u8, path);
    return try std.fs.cwd().realpathAlloc(allocator, path);
}
