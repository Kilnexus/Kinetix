const std = @import("std");
const builtin = @import("builtin");
const adapter_mod = @import("../../adapter/adapter.zig");
const task = @import("../../core/task.zig");
const handle_mod = @import("../model/handle.zig");
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
            else => error.RuntimeExecutionNotImplemented,
        };
    }

    pub fn executeBatch(self: Executor, handle: *const handle_mod.ModelHandle, plan: *const types.ExecutionPlan) !types.RuntimeBatchResults {
        if (plan.request_count == 0 or plan.requests.len == 0) return error.InvalidExecutionPlan;

        return switch (handle.normalized.provider_key) {
            .qwen3_text => try executeQwen3Batch(self.allocator, handle, plan.requests),
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
