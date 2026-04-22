const std = @import("std");
const builtin = @import("builtin");
const backend = @import("../../artifacts/backend/backend.zig");
const task = @import("../../core/task.zig");
const text_runtime = @import("../text/text.zig");
const types = @import("../types.zig");

pub const NativeRuntimeBinding = if (builtin.is_test) struct {
    pub fn executeQwenSingle(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: backend.WeightScheme,
        request: task.TaskRequest,
    ) ![]u8 {
        _ = model_dir;
        _ = preferred_weights;
        _ = request;
        return try allocator.dupe(u8, "test-native-single");
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
        preferred_weights: backend.WeightScheme,
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
            text.* = try allocator.dupe(u8, "test-native-batch");
            initialized += 1;
        }
        return .{
            .texts = texts,
            .total_decoded_tokens = requests.len,
            .finished_requests = requests.len,
        };
    }
} else text_runtime.native_dispatch.NativeQwenRuntime;

pub fn buildSubmission(adapter_id: []const u8, execution: task.ExecutionMode) types.Submission {
    return .{
        .adapter_id = adapter_id,
        .accepted = true,
        .execution = execution,
    };
}

pub fn buildReadyResult(submission: types.Submission) types.ExecutionResult {
    return .{
        .submission = submission,
        .origin = .shared_adapter,
        .note = .text_request_ready,
    };
}

pub fn buildReadyBatchResults(
    allocator: std.mem.Allocator,
    adapter_id: []const u8,
    requests: []const task.TaskRequest,
) ![]types.ExecutionResult {
    const results = try allocator.alloc(types.ExecutionResult, requests.len);
    errdefer allocator.free(results);

    for (requests, results) |request, *result| {
        result.* = buildReadyResult(buildSubmission(adapter_id, request.spec.execution));
    }
    return results;
}

pub fn executeQwenSingle(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
    adapter_id: []const u8,
    request: task.TaskRequest,
) !types.RuntimeResult {
    var execution_result = try text_runtime.native_dispatch.executeSingle(
        allocator,
        NativeRuntimeBinding,
        model_dir,
        preferred_weights,
        buildSubmission(adapter_id, request.spec.execution),
        request,
    );
    return runtimeResultFromExecution(&execution_result);
}

pub fn executeQwenBatch(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
    adapter_id: []const u8,
    requests: []const task.TaskRequest,
) !types.RuntimeBatchResults {
    const execution_results = try text_runtime.native_dispatch.executeBatch(
        allocator,
        NativeRuntimeBinding,
        model_dir,
        preferred_weights,
        adapter_id,
        requests,
    );
    return try runtimeBatchResultsFromExecution(allocator, execution_results);
}

pub fn runtimeResultFromExecution(execution_result: *types.ExecutionResult) types.RuntimeResult {
    const output = execution_result.output;
    execution_result.output = .none;
    return .{
        .origin = execution_result.origin,
        .note = execution_result.note,
        .output = output,
    };
}

pub fn runtimeBatchResultsFromExecution(
    allocator: std.mem.Allocator,
    execution_results: []types.ExecutionResult,
) !types.RuntimeBatchResults {
    defer allocator.free(execution_results);
    const results = try allocator.alloc(types.RuntimeResult, execution_results.len);
    errdefer allocator.free(results);

    for (execution_results, results) |*execution_result, *result| {
        result.* = runtimeResultFromExecution(execution_result);
    }

    return .{
        .allocator = allocator,
        .items = results,
    };
}
