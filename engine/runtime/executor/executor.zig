const std = @import("std");
const task = @import("../../core/task.zig");
const handle_mod = @import("../model/handle.zig");
const ocr_shared = @import("../providers/ocr_shared.zig");
const text_shared = @import("../providers/text_shared.zig");
const types = @import("../types.zig");
const vision_shared = @import("../providers/vision_shared.zig");

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
    var legacy_result = try text_shared.executeLegacySingle(
        allocator,
        handle.normalized.artifacts.model_dir,
        .auto,
        handle.normalized.descriptor.id,
        buildTaskRequest(handle, request),
    );
    return text_shared.adoptRuntimeResult(&legacy_result);
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

    const legacy_results = try text_shared.executeLegacyBatch(
        allocator,
        handle.normalized.artifacts.model_dir,
        .auto,
        handle.normalized.descriptor.id,
        task_requests,
    );
    defer allocator.free(legacy_results);
    return try text_shared.adoptRuntimeBatchResults(allocator, legacy_results);
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
    const summary = try vision_shared.loadSummary(allocator, graph_path);
    defer allocator.free(summary.model_name);

    var detection_output = try vision_shared.maybeRunDetect(
        allocator,
        graph_path,
        weights_path,
        request.operation,
        request.execution,
        request.input.asString(),
    );
    defer if (detection_output) |*output| output.deinit(allocator);

    const output = try vision_shared.buildOutputJson(allocator, .{
        .operation = request.operation,
        .model_name = summary.model_name,
        .model_family = handle.normalized.descriptor.family,
        .input_path = request.input.asString(),
        .execution_nodes = summary.execution_nodes,
        .tensor_count = summary.tensor_count,
        .class_count = summary.class_count,
    }, detection_output);
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
    const infer_output = try ocr_shared.maybeRunInfer(
        allocator,
        model_path,
        request.operation,
        request.execution,
        request.input.asString(),
    );
    const output = try ocr_shared.buildOutputJson(allocator, .{
        .operation = request.operation,
        .model_family = handle.normalized.descriptor.family,
        .model_path = model_path,
        .input_path = request.input.asString(),
    }, infer_output);
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
