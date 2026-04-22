const std = @import("std");
const backend = @import("../../../../artifacts/backend/backend.zig");
const task = @import("../../../../core/task.zig");
const qwen_native = @import("qwen_native.zig");
const types = @import("../../../types.zig");

pub const NativeQwenRuntime = struct {
    pub fn executeQwenSingle(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: backend.WeightScheme,
        request: task.TaskRequest,
    ) ![]u8 {
        return try qwen_native.executeQwenSingle(allocator, model_dir, preferred_weights, request);
    }

    pub const NativeBatchOutput = qwen_native.NativeBatchOutput;

    pub fn executeQwenBatch(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: backend.WeightScheme,
        requests: []const task.TaskRequest,
    ) !NativeBatchOutput {
        return try qwen_native.executeQwenBatch(allocator, model_dir, preferred_weights, requests);
    }
};

pub fn canUseNativeQwenSingle(is_qwen3: bool, request: task.TaskRequest) bool {
    if (!is_qwen3) return false;
    if (!request.generation.native_execution) return false;
    if (request.spec.execution != .sync) return false;
    if (!std.mem.eql(u8, request.spec.operation, "generate") and !std.mem.eql(u8, request.spec.operation, "chat")) return false;
    return switch (request.input) {
        .none, .text => true,
        else => false,
    };
}

pub fn canUseNativeQwenBatch(is_qwen3: bool, requests: []const task.TaskRequest) bool {
    if (!is_qwen3) return false;
    if (requests.len <= 1) return false;

    for (requests, 0..) |request, index| {
        if (!request.generation.native_execution) return false;
        if (request.spec.execution != .sync) return false;
        if (!std.mem.eql(u8, request.spec.operation, "generate") and !std.mem.eql(u8, request.spec.operation, "chat")) return false;
        switch (request.input) {
            .text => {},
            else => return false,
        }

        if (index != 0 and request.generation.max_tokens != requests[0].generation.max_tokens) return false;
    }

    return true;
}

pub fn executeSingle(
    allocator: std.mem.Allocator,
    runtime_impl: type,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
    submission: types.Submission,
    request: task.TaskRequest,
) !types.ExecutionResult {
    const use_native = canUseNativeQwenSingle(true, request);
    const output = if (use_native)
        types.OutputPayload{ .text = try runtime_impl.executeQwenSingle(
            allocator,
            model_dir,
            preferred_weights,
            request,
        ) }
    else
        .none;

    return .{
        .submission = submission,
        .origin = if (use_native) .native_single else .shared_adapter,
        .note = if (use_native) .text_native_qwen_single else .text_request_ready,
        .output = output,
    };
}

pub fn executeBatch(
    allocator: std.mem.Allocator,
    runtime_impl: type,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
    adapter_id: []const u8,
    requests: []const task.TaskRequest,
) ![]types.ExecutionResult {
    const use_native = canUseNativeQwenBatch(true, requests);
    var native_output: ?runtime_impl.NativeBatchOutput = null;
    defer if (native_output) |*output| output.deinit(allocator);

    if (use_native) {
        native_output = try runtime_impl.executeQwenBatch(
            allocator,
            model_dir,
            preferred_weights,
            requests,
        );
    }

    const results = try allocator.alloc(types.ExecutionResult, requests.len);
    errdefer allocator.free(results);

    for (requests, results, 0..) |request, *result, index| {
        result.* = .{
            .submission = .{
                .adapter_id = adapter_id,
                .accepted = true,
                .execution = request.spec.execution,
            },
            .origin = if (use_native) .native_batch else .shared_adapter,
            .note = if (use_native) .text_native_qwen_batch else .text_request_ready,
            .output = if (use_native)
                .{ .text = try allocator.dupe(u8, native_output.?.texts[index]) }
            else
                .none,
        };
    }

    return results;
}
