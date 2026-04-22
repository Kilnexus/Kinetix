const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../../../backend/backend.zig");
const backend_scheme = @import("../../../text/backend_scheme.zig");
const task = @import("../../../../core/task.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const qwen_native = @import("../../../text/families/qwen3/qwen_native.zig");
const text_shared = @import("../../../shared/text/runtime.zig");
const text_runtime = @import("../../../text/generator_runtime.zig");
const types = @import("../../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .qwen3_text,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
    .execute_stream_fn = executeStream,
    .execute_batch_fn = executeBatch,
};

const State = struct {
    base: backend_mod.OpenState,
    runtime: ?text_runtime.GeneratorRuntime = null,

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        if (self.runtime) |*runtime| runtime.deinit();
        allocator.free(self.base.model_dir);
        allocator.destroy(self);
    }
};

fn open(
    allocator: std.mem.Allocator,
    model: *const normalized.NormalizedModel,
) !?*anyopaque {
    const state = try allocator.create(State);
    errdefer allocator.destroy(state);
    state.* = .{
        .base = .{
            .provider_key = .qwen3_text,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .runtime = null,
    };
    errdefer allocator.free(state.base.model_dir);

    if (!builtin.is_test) {
        state.runtime = try text_runtime.GeneratorRuntime.init(
            allocator,
            model.artifacts.model_dir,
            backend_scheme.Scheme.auto,
            0,
        );
    }
    return state;
}

fn deinit(allocator: std.mem.Allocator, raw_state: ?*anyopaque) void {
    const raw = raw_state orelse return;
    const state: *State = @ptrCast(@alignCast(raw));
    state.destroy(allocator);
}

fn execute(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    const state = stateFromHandle(handle);
    if (state) |qwen_state| {
        if (qwen_state.runtime) |*runtime| {
            var execution_result = types.ExecutionResult{
                .submission = buildSubmission(handle, request),
                .origin = .native_single,
                .note = .text_native_qwen_single,
                .output = .{ .text = try qwen_native.executeQwenSingleWithRuntime(
                    allocator,
                    runtime,
                    buildTaskRequest(handle, request),
                ) },
            };
            return text_shared.runtimeResultFromExecution(&execution_result);
        }
    }

    return try text_shared.executeQwenSingle(
        allocator,
        handle.normalized.artifacts.model_dir,
        .auto,
        handle.normalized.descriptor.id,
        buildTaskRequest(handle, request),
    );
}

fn executeStream(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    return try execute(allocator, handle, request);
}

fn executeBatch(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    requests: []const types.RuntimeRequest,
) !types.RuntimeBatchResults {
    const task_requests = try allocator.alloc(task.TaskRequest, requests.len);
    defer allocator.free(task_requests);

    for (requests, task_requests) |request, *slot| {
        slot.* = buildTaskRequest(handle, request);
    }

    const state = stateFromHandle(handle);
    if (state) |qwen_state| {
        if (qwen_state.runtime) |*runtime| {
            const native_output = try qwen_native.executeQwenBatchWithRuntime(allocator, runtime, task_requests);
            defer {
                var output = native_output;
                output.deinit(allocator);
            }

            const execution_results = try allocator.alloc(types.ExecutionResult, task_requests.len);
            errdefer allocator.free(execution_results);

            for (task_requests, execution_results, 0..) |request, *result, index| {
                result.* = .{
                    .submission = .{
                        .adapter_id = handle.normalized.descriptor.id,
                        .accepted = true,
                        .execution = request.spec.execution,
                    },
                    .origin = .native_batch,
                    .note = .text_native_qwen_batch,
                    .output = .{ .text = try allocator.dupe(u8, native_output.texts[index]) },
                };
            }
            return try text_shared.runtimeBatchResultsFromExecution(allocator, execution_results);
        }
    }

    return try text_shared.executeQwenBatch(
        allocator,
        handle.normalized.artifacts.model_dir,
        .auto,
        handle.normalized.descriptor.id,
        task_requests,
    );
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

fn buildSubmission(handle: *const handle_mod.ModelHandle, request: types.RuntimeRequest) types.Submission {
    return .{
        .adapter_id = handle.normalized.descriptor.id,
        .accepted = true,
        .execution = request.execution,
    };
}

fn stateFromHandle(handle: *const handle_mod.ModelHandle) ?*State {
    const raw = handle.provider_state orelse return null;
    return @ptrCast(@alignCast(raw));
}
